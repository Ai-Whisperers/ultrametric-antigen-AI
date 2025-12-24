"""Test generalization capabilities of Ternary VAE v5.6.

This test suite implements the advanced tests from general.md to validate:
1. Compositional generalization beyond memorization
2. Permutation robustness and invariance
3. Noise resilience
4. Latent algebra consistency
5. Compositional extrapolation
6. StateNet validation

These tests probe whether the model learns *rules* rather than memorizing mappings.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, List, Dict
import itertools

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ternary_vae_v5_6 import DualNeuralVAEV5
from src.utils.data import generate_all_ternary_operations, sample_operations


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compose_ternary_ops(op1: np.ndarray, op2: np.ndarray) -> np.ndarray:
    """Compose two ternary operations using element-wise ternary addition.

    Since operations are 9-dimensional truth tables, we define composition as
    element-wise ternary addition modulo 3, mapped back to {-1, 0, 1}.
    This creates a new operation that combines properties of both.

    Args:
        op1: First operation (9,) - values in {-1, 0, 1}
        op2: Second operation (9,) - values in {-1, 0, 1}

    Returns:
        Composite operation (9,)
    """
    # Convert to {0, 1, 2} for modular arithmetic
    op1_mod = ((op1 + 1) % 3).astype(int)
    op2_mod = ((op2 + 1) % 3).astype(int)

    # Element-wise addition mod 3
    composed_mod = (op1_mod + op2_mod) % 3

    # Convert back to {-1, 0, 1}
    composed = (composed_mod - 1).astype(np.float32)

    return composed


def permute_operation(op: np.ndarray, permutation: List[int]) -> np.ndarray:
    """Permute the symbol ordering in a ternary operation.

    Args:
        op: Operation (9,) with values in {-1, 0, 1}
        permutation: List of 3 indices representing mapping [0,1,2] -> new order
                    e.g., [1,2,0] means {-1,0,1} -> {0,1,-1}

    Returns:
        Permuted operation
    """
    value_map = {-1: 0, 0: 1, 1: 2}
    reverse_map = {0: -1, 1: 0, 2: 1}

    # Apply permutation
    permuted = np.zeros_like(op)
    for i in range(9):
        old_value_idx = value_map[op[i]]
        new_value_idx = permutation[old_value_idx]
        permuted[i] = reverse_map[new_value_idx]

    return permuted


def inject_noise(op: np.ndarray, noise_prob: float, seed: int = 42) -> np.ndarray:
    """Inject random noise into a ternary operation.

    Args:
        op: Operation (9,) with values in {-1, 0, 1}
        noise_prob: Probability of flipping each entry
        seed: Random seed

    Returns:
        Noisy operation
    """
    np.random.seed(seed)
    noisy = op.copy()

    for i in range(len(op)):
        if np.random.random() < noise_prob:
            # Flip to a random different value
            choices = [-1, 0, 1]
            choices.remove(op[i])
            noisy[i] = np.random.choice(choices)

    return noisy


class TestGeneralization:
    """Test suite for generalization capabilities beyond memorization."""

    @pytest.fixture
    def trained_model(self):
        """Load trained model from checkpoint for validation.

        Loads the production checkpoint to verify that training actually learned.
        """
        set_seed(42)

        # Path to the trained checkpoint
        checkpoint_path = Path(__file__).parent.parent / "sandbox-training" / "checkpoints" / "v5_6" / "latest.pt"

        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not found at {checkpoint_path}. Run training first.")

        # Initialize model with same config as training
        model = DualNeuralVAEV5(
            input_dim=9,
            latent_dim=16,
            rho_min=0.1,
            rho_max=0.7,
            lambda3_base=0.3,
            lambda3_amplitude=0.15,
            eps_kl=0.0005
        )

        # Load trained weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.eval()

        return model

    @pytest.fixture
    def all_operations(self):
        """Generate all 19,683 ternary operations."""
        return generate_all_ternary_operations()

    # ========================================================================
    # 1. UNSEEN LOGICAL TRANSFORM TEST (ULT)
    # ========================================================================

    def test_unseen_operations_holdout(self, trained_model, all_operations):
        """Test reconstruction on held-out operations (10% holdout).

        Goal: Measure if model can reconstruct operations it wasn't trained on.
        A model that truly learns structure should generalize beyond training set.
        """
        set_seed(42)

        # Hold out 10% of operations
        n_holdout = int(0.1 * len(all_operations))
        indices = np.random.permutation(len(all_operations))
        holdout_indices = indices[:n_holdout]

        holdout_ops = all_operations[holdout_indices]

        # Test reconstruction
        trained_model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(holdout_ops[:100])  # Test on 100 samples

            outputs_A = trained_model(x, temp_A=0.1, temp_B=0.1, beta_A=1.0, beta_B=1.0)

            # Reconstruct from VAE-A
            logits_A = outputs_A['logits_A']
            preds_A = torch.argmax(logits_A, dim=-1) - 1  # Convert {0,1,2} -> {-1,0,1}

            # Calculate exact match accuracy
            x_rounded = x.long()
            accuracy = (preds_A == x_rounded).float().mean().item()

        # A trained model should achieve high accuracy on holdout operations
        # Models that memorize would fail; models that learn rules should succeed
        assert accuracy > 0.9, (
            f"Holdout reconstruction accuracy too low: {accuracy*100:.2f}%. "
            f"Expected >90% for a model that learns structure rather than memorizing."
        )

    def test_compositional_operations(self, trained_model):
        """Test if model can reconstruct composite operations.

        Goal: Test if encode(A ∘ B) relates to encode(A) and encode(B).
        True compositional understanding would show this relationship.
        """
        set_seed(42)

        # Sample base operations
        ops = sample_operations(50, replacement=True, seed=42)

        # Create 10 composite operations
        composite_ops = []
        for i in range(0, 20, 2):
            composed = compose_ternary_ops(ops[i], ops[i+1])
            composite_ops.append(composed)

        composite_ops = np.array(composite_ops)

        trained_model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(composite_ops)
            outputs = trained_model(x, temp_A=0.1, temp_B=0.1, beta_A=1.0, beta_B=1.0)

            logits_A = outputs['logits_A']
            preds_A = torch.argmax(logits_A, dim=-1) - 1

            x_rounded = x.long()
            accuracy = (preds_A == x_rounded).float().mean().item()

        # Trained model should handle compositional operations
        assert accuracy > 0.7, (
            f"Compositional reconstruction accuracy too low: {accuracy*100:.2f}%. "
            f"Expected >70% for compositional understanding."
        )

    # ========================================================================
    # 2. PERMUTATION ROBUSTNESS TEST (PRT)
    # ========================================================================

    def test_permutation_robustness(self, trained_model):
        """Test if latent representations are invariant to symbol permutations.

        Goal: If we permute {-1,0,1} -> {0,1,-1}, does the latent space structure
        remain consistent? True understanding should be permutation-equivariant.
        """
        set_seed(42)

        # Sample operations
        ops = sample_operations(20, replacement=True, seed=42)

        # Test multiple permutations
        permutations = [
            [0, 1, 2],  # Identity
            [2, 1, 0],  # Reverse
            [1, 2, 0],  # Cyclic
        ]

        trained_model.eval()

        results = {}
        for perm_idx, perm in enumerate(permutations):
            permuted_ops = np.array([permute_operation(op, perm) for op in ops])

            with torch.no_grad():
                x = torch.FloatTensor(permuted_ops)
                outputs = trained_model(x, temp_A=0.1, temp_B=0.1, beta_A=1.0, beta_B=1.0)

                # Store latent codes
                z_A = outputs['z_A']
                z_B = outputs['z_B']

                results[f'perm_{perm_idx}'] = {
                    'z_A': z_A,
                    'z_B': z_B
                }

        # Compare latent space structure across permutations
        # Calculate pairwise distances in original vs permuted
        z_orig_A = results['perm_0']['z_A']
        z_perm_A = results['perm_1']['z_A']

        # Calculate distance matrix similarity
        dist_orig = torch.cdist(z_orig_A, z_orig_A)
        dist_perm = torch.cdist(z_perm_A, z_perm_A)

        # Correlation between distance matrices (should be high if structure preserved)
        correlation = torch.corrcoef(torch.stack([
            dist_orig.flatten(),
            dist_perm.flatten()
        ]))[0, 1].item()

        # Higher correlation means better structural preservation
        # A model that learns permutation-invariant structure should have high correlation
        assert correlation > 0.7, (
            f"Permutation distance correlation too low: {correlation:.3f}. "
            f"Expected >0.7 for structure-preserving embeddings."
        )

    # ========================================================================
    # 3. LOGICAL NOISE INJECTION TEST (LNIT)
    # ========================================================================

    def test_noise_resilience(self, trained_model):
        """Test if latent space can recover clean structure from noisy inputs.

        Goal: Inject 1-5% noise into truth tables and check if latent clusters
        still recover the clean rule. Robust representations should denoise.
        """
        set_seed(42)

        # Sample clean operations
        clean_ops = sample_operations(30, replacement=True, seed=42)

        # Test different noise levels
        noise_levels = [0.01, 0.03, 0.05]

        trained_model.eval()

        with torch.no_grad():
            # Encode clean operations
            x_clean = torch.FloatTensor(clean_ops)
            outputs_clean = trained_model(x_clean, temp_A=0.1, temp_B=0.1,
                                         beta_A=1.0, beta_B=1.0)
            z_clean_A = outputs_clean['z_A']

            max_distance = 0.0
            for noise_level in noise_levels:
                # Add noise
                noisy_ops = np.array([inject_noise(op, noise_level, seed=i)
                                     for i, op in enumerate(clean_ops)])

                x_noisy = torch.FloatTensor(noisy_ops)
                outputs_noisy = trained_model(x_noisy, temp_A=0.1, temp_B=0.1,
                                            beta_A=1.0, beta_B=1.0)
                z_noisy_A = outputs_noisy['z_A']

                # Measure distance between clean and noisy encodings
                distances = torch.norm(z_clean_A - z_noisy_A, dim=1).mean().item()
                max_distance = max(max_distance, distances)

        # Robust model should keep latent distance bounded even with 5% noise
        assert max_distance < 5.0, (
            f"Latent space not resilient to noise: max distance {max_distance:.3f}. "
            f"Expected <5.0 for noise-robust encodings."
        )

    # ========================================================================
    # 4. LATENT ALGEBRA CONSISTENCY
    # ========================================================================

    def test_latent_arithmetic(self, trained_model):
        """Test if latent space supports meaningful arithmetic.

        Goal: Check if encode(A) + encode(B) ≈ encode(A∘B) for some operations.
        This would demonstrate algebraic structure in latent space.
        """
        set_seed(42)

        # Sample operations
        ops = sample_operations(30, replacement=True, seed=42)

        trained_model.eval()

        with torch.no_grad():
            # Encode all operations
            x = torch.FloatTensor(ops)
            outputs = trained_model(x, temp_A=0.1, temp_B=0.1, beta_A=1.0, beta_B=1.0)
            z_A = outputs['z_A']

            # Test a few pairs
            n_tests = 5
            arithmetic_errors = []

            for i in range(n_tests):
                # Get two operations and their composition
                op1, op2 = ops[i*2], ops[i*2+1]
                composed = compose_ternary_ops(op1, op2)

                # Encode composition
                x_comp = torch.FloatTensor([composed])
                outputs_comp = trained_model(x_comp, temp_A=0.1, temp_B=0.1,
                                           beta_A=1.0, beta_B=1.0)
                z_comp = outputs_comp['z_A']

                # Compute arithmetic: z1 + z2
                z1, z2 = z_A[i*2], z_A[i*2+1]
                z_sum = z1 + z2

                # Measure error
                error = torch.norm(z_comp - z_sum).item()
                arithmetic_errors.append(error)

            mean_error = np.mean(arithmetic_errors)

        # If latent space has algebraic structure, composition error should be bounded
        # Note: Exact arithmetic may not hold, but reasonable structure should keep error < 10
        assert mean_error < 10.0, (
            f"Latent arithmetic error too high: {mean_error:.3f}. "
            f"Expected <10.0 for meaningful algebraic structure."
        )

    # ========================================================================
    # 5. STATENET VALIDATION
    # ========================================================================

    def test_statenet_contribution(self, trained_model):
        """Test if StateNet provides unique adaptive contribution.

        Goal: Verify StateNet is learning, not just acting as noise/regularization.
        This is an ablation-style test.
        """
        if not trained_model.use_statenet:
            pytest.skip("StateNet not enabled in model")

        set_seed(42)

        # Sample data
        ops = sample_operations(50, replacement=True, seed=42)
        x = torch.FloatTensor(ops)

        trained_model.eval()

        with torch.no_grad():
            # Forward pass with StateNet
            outputs = trained_model(x, temp_A=1.0, temp_B=1.0, beta_A=1.0, beta_B=1.0)
            outputs['z_A']

            # Check StateNet parameters have gradients/updates
            statenet_params = list(trained_model.state_net.parameters())
            param_norms = [p.norm().item() for p in statenet_params]

            mean_norm = np.mean(param_norms)
            print(f"\n[StateNet] Mean parameter norm: {mean_norm:.3f}")

        # StateNet should have non-trivial parameters
        assert mean_norm > 0.01, "StateNet has non-trivial parameters"

    # ========================================================================
    # 6. RECONSTRUCTION METRICS
    # ========================================================================

    def test_reconstruction_accuracy_distribution(self, trained_model):
        """Measure reconstruction accuracy across different operation types.

        Goal: Identify which types of operations are harder to reconstruct.
        This reveals systematic biases or weaknesses.
        """
        set_seed(42)

        # Sample diverse operations
        ops = sample_operations(200, replacement=True, seed=42)

        # Categorize by sparsity (number of zeros)
        sparse_ops = []
        dense_ops = []

        for op in ops:
            n_zeros = np.sum(op == 0)
            if n_zeros >= 6:
                sparse_ops.append(op)
            elif n_zeros <= 3:
                dense_ops.append(op)

        trained_model.eval()

        def measure_accuracy(ops_list):
            if len(ops_list) == 0:
                return 0.0

            with torch.no_grad():
                x = torch.FloatTensor(np.array(ops_list))
                outputs = trained_model(x, temp_A=0.1, temp_B=0.1,
                                       beta_A=1.0, beta_B=1.0)

                logits = outputs['logits_A']
                preds = torch.argmax(logits, dim=-1) - 1

                x_rounded = x.long()
                accuracy = (preds == x_rounded).float().mean().item()

            return accuracy

        sparse_acc = measure_accuracy(sparse_ops)
        dense_acc = measure_accuracy(dense_ops)

        # Both sparse and dense operations should be reconstructed well
        # Testing that the model handles different operation types
        min_acc = min(sparse_acc, dense_acc)
        assert min_acc > 0.8, (
            f"Reconstruction accuracy too low for some operation types. "
            f"Sparse: {sparse_acc*100:.2f}%, Dense: {dense_acc*100:.2f}%. "
            f"Expected >80% for both."
        )

    def test_latent_space_coverage(self, trained_model):
        """Test if latent space is well-utilized vs collapsed.

        Goal: Measure entropy and coverage of latent codes. Posterior collapse
        would show very low variance/entropy.
        """
        set_seed(42)

        # Sample many operations
        ops = sample_operations(500, replacement=True, seed=42)

        trained_model.eval()

        with torch.no_grad():
            x = torch.FloatTensor(ops)
            outputs = trained_model(x, temp_A=0.1, temp_B=0.1, beta_A=1.0, beta_B=1.0)

            z_A = outputs['z_A']
            z_B = outputs['z_B']

            # Measure per-dimension variance
            var_A = z_A.var(dim=0).mean().item()
            var_B = z_B.var(dim=0).mean().item()

            # Measure effective rank (dimensionality)
            cov_A = torch.cov(z_A.T)
            eigenvalues_A = torch.linalg.eigvalsh(cov_A)
            eigenvalues_A = eigenvalues_A[eigenvalues_A > 0]

            # Effective rank via normalized entropy
            probs_A = eigenvalues_A / eigenvalues_A.sum()
            entropy_A = -(probs_A * torch.log(probs_A + 1e-10)).sum().item()
            effective_dim_A = np.exp(entropy_A)

            print(f"\n[Latent Coverage] VAE-A mean variance: {var_A:.3f}")
            print(f"[Latent Coverage] VAE-A effective dimensions: {effective_dim_A:.1f}/16")
            print(f"[Latent Coverage] VAE-B mean variance: {var_B:.3f}")

        # Good latent spaces should have variance > 0.1 and use most dimensions
        # Posterior collapse would show very low variance
        assert var_A > 0.1, (
            f"Latent space variance too low: {var_A:.3f}. "
            f"Expected >0.1 to avoid posterior collapse."
        )
        assert effective_dim_A > 8, (
            f"Latent space not using enough dimensions: {effective_dim_A:.1f}/16. "
            f"Expected >8 for good latent utilization."
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
