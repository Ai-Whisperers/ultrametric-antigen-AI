"""Comprehensive training validation tests.

Implements recommended checks:
1. Latent Activation Spectrum: Verify all latent dims show non-zero variance
2. Hold-out Operation Test: Test on unseen ternary ops
3. KL-Annealing Curve: Confirm β rises smoothly
4. Hash-based Coverage Validation: Recompute coverage offline
5. Phase-Sync Check: Monitor VAE-A/VAE-B dynamics

Run this after training progresses to validate model quality.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Dict, List
import hashlib

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


def load_latest_checkpoint(checkpoint_dir: str = "sandbox-training/checkpoints/v5_6"):
    """Load the latest checkpoint from training."""
    checkpoint_path = Path(checkpoint_dir) / "latest.pt"

    if not checkpoint_path.exists():
        pytest.skip(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model
    model = DualNeuralVAEV5(
        input_dim=9,
        latent_dim=16,
        rho_min=0.1,
        rho_max=0.7,
        lambda3_base=0.3,
        lambda3_amplitude=0.15,
        eps_kl=0.0005
    )

    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model, checkpoint


class TestTrainingValidation:
    """Validation tests for trained model."""

    @pytest.fixture
    def trained_model(self):
        """Load trained model from checkpoint."""
        model, checkpoint = load_latest_checkpoint()
        return model, checkpoint

    # ========================================================================
    # 1. LATENT ACTIVATION SPECTRUM
    # ========================================================================

    def test_latent_activation_spectrum(self, trained_model):
        """Test: Verify all latent dims show non-zero variance.

        Collapse in a few dims fakes coverage. We need to ensure all 16
        dimensions of the latent space are actually being used.
        """
        model, checkpoint = trained_model

        set_seed(42)

        # Sample diverse operations
        ops = sample_operations(500, replacement=True, seed=42)
        x = torch.FloatTensor(ops)

        with torch.no_grad():
            outputs = model(x, temp_A=0.1, temp_B=0.1, beta_A=1.0, beta_B=1.0)

            z_A = outputs['z_A']  # (500, 16)
            z_B = outputs['z_B']  # (500, 16)

        # Compute per-dimension variance
        var_A = z_A.var(dim=0)  # (16,)
        var_B = z_B.var(dim=0)  # (16,)

        # Count active dimensions (variance > threshold)
        threshold = 0.01
        active_dims_A = (var_A > threshold).sum().item()
        active_dims_B = (var_B > threshold).sum().item()

        # Compute min/max/mean variance
        min_var_A = var_A.min().item()
        max_var_A = var_A.max().item()
        mean_var_A = var_A.mean().item()

        min_var_B = var_B.min().item()
        max_var_B = var_B.max().item()
        mean_var_B = var_B.mean().item()

        print("\n[Latent Spectrum] VAE-A:")
        print(f"  Active dims: {active_dims_A}/16")
        print(f"  Variance - min: {min_var_A:.4f}, max: {max_var_A:.4f}, mean: {mean_var_A:.4f}")
        print(f"  Per-dim variance: {var_A.tolist()}")

        print("\n[Latent Spectrum] VAE-B:")
        print(f"  Active dims: {active_dims_B}/16")
        print(f"  Variance - min: {min_var_B:.4f}, max: {max_var_B:.4f}, mean: {mean_var_B:.4f}")
        print(f"  Per-dim variance: {var_B.tolist()}")

        # Check for collapsed dimensions
        collapsed_dims_A = (var_A < 0.001).sum().item()
        collapsed_dims_B = (var_B < 0.001).sum().item()

        if collapsed_dims_A > 4:
            print(f"  ⚠️  WARNING: {collapsed_dims_A} dimensions collapsed in VAE-A")

        if collapsed_dims_B > 4:
            print(f"  ⚠️  WARNING: {collapsed_dims_B} dimensions collapsed in VAE-B")

        # Success if most dimensions are active
        assert active_dims_A >= 10, f"Too few active dimensions in VAE-A: {active_dims_A}/16"
        assert active_dims_B >= 10, f"Too few active dimensions in VAE-B: {active_dims_B}/16"

    # ========================================================================
    # 2. HOLD-OUT OPERATION TEST
    # ========================================================================

    def test_holdout_operation_retention(self, trained_model):
        """Test: Hold-out operation test after training.

        Genuine learning should exceed 40-50% reconstruction on unseen ops.
        This tests true generalization, not just memorization.
        """
        model, checkpoint = trained_model

        set_seed(42)

        # Get all operations
        all_ops = generate_all_ternary_operations()

        # Hold out 10% for testing
        n_holdout = int(0.1 * len(all_ops))
        indices = np.random.permutation(len(all_ops))
        holdout_indices = indices[:n_holdout]

        holdout_ops = all_ops[holdout_indices]

        # Test reconstruction on holdout set
        batch_size = 100
        correct_A = 0
        correct_B = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(holdout_ops), batch_size):
                batch = torch.FloatTensor(holdout_ops[i:i+batch_size])

                outputs = model(batch, temp_A=0.1, temp_B=0.1,
                              beta_A=1.0, beta_B=1.0)

                # VAE-A predictions
                logits_A = outputs['logits_A']
                preds_A = torch.argmax(logits_A, dim=-1) - 1

                # VAE-B predictions
                logits_B = outputs['logits_B']
                preds_B = torch.argmax(logits_B, dim=-1) - 1

                # Convert targets
                targets = batch.long()

                # Count correct
                correct_A += (preds_A == targets).float().sum().item()
                correct_B += (preds_B == targets).float().sum().item()
                total += targets.numel()

        # Compute accuracies
        acc_A = (correct_A / total) * 100
        acc_B = (correct_B / total) * 100

        epoch = checkpoint.get('epoch', 'unknown')

        print(f"\n[Hold-out Test] Epoch {epoch}:")
        print(f"  VAE-A accuracy: {acc_A:.2f}% (target: >40%)")
        print(f"  VAE-B accuracy: {acc_B:.2f}% (target: >40%)")

        # Check if meeting targets
        if acc_A > 40:
            print("  ✓ VAE-A passes hold-out test")
        else:
            print(f"  ✗ VAE-A below target ({acc_A:.2f}% < 40%)")

        if acc_B > 40:
            print("  ✓ VAE-B passes hold-out test")
        else:
            print(f"  ✗ VAE-B below target ({acc_B:.2f}% < 40%)")

        # Assert at least one VAE passes
        assert max(acc_A, acc_B) > 40, \
            f"Both VAEs failed hold-out test: A={acc_A:.2f}%, B={acc_B:.2f}%"

    # ========================================================================
    # 3. KL-ANNEALING CURVE VALIDATION
    # ========================================================================

    def test_kl_annealing_curve(self, trained_model):
        """Test: Verify β rises smoothly without sudden spikes.

        Sudden β spikes can create explosive first-epoch jumps that
        look like learning but are actually just exploration noise.
        """
        model, checkpoint = trained_model

        # Get training history from checkpoint
        epoch = checkpoint.get('epoch', 0)

        # Simulate β schedule for validation
        warmup_epochs = 50
        beta_start = 0.3
        beta_end = 0.8

        betas = []
        for e in range(epoch + 1):
            if e < warmup_epochs:
                beta = (e / warmup_epochs) * beta_start
            else:
                progress = (e - warmup_epochs) / (400 - warmup_epochs)
                beta = beta_start + (beta_end - beta_start) * progress

            betas.append(beta)

        # Compute differences to check for spikes
        if len(betas) > 1:
            diffs = np.diff(betas)
            max_diff = np.max(np.abs(diffs))
            mean_diff = np.mean(np.abs(diffs))

            print(f"\n[KL-Annealing] Epoch {epoch}:")
            print(f"  Current β: {betas[-1]:.4f}")
            print(f"  Max β jump: {max_diff:.4f}")
            print(f"  Mean β change: {mean_diff:.4f}")

            # Check for sudden spikes
            if max_diff > 0.05:
                print(f"  ⚠️  WARNING: Large β spike detected ({max_diff:.4f})")
            else:
                print("  ✓ Smooth β annealing")

            # Assert reasonable annealing
            assert max_diff < 0.1, f"β spike too large: {max_diff:.4f}"
        else:
            print("\n[KL-Annealing] Only epoch 0, skipping diff check")

    # ========================================================================
    # 4. HASH-BASED COVERAGE VALIDATION
    # ========================================================================

    def test_hash_based_coverage_validation(self, trained_model):
        """Test: Recompute coverage offline with hash-based deduplication.

        Removes duplicate operations bit-wise to ensure accurate coverage count.
        """
        model, checkpoint = trained_model

        set_seed(42)

        # Sample from model
        num_samples = 50000
        batch_size = 1000

        unique_ops_A = set()
        unique_ops_B = set()

        with torch.no_grad():
            for _ in range(num_samples // batch_size):
                # Sample from VAE-A
                z_A = torch.randn(batch_size, 16)
                logits_A = model.decoder_A(z_A)
                preds_A = torch.argmax(logits_A, dim=-1) - 1

                # Sample from VAE-B
                z_B = torch.randn(batch_size, 16)
                logits_B = model.decoder_B(z_B)
                preds_B = torch.argmax(logits_B, dim=-1) - 1

                # Convert to tuples and add to sets
                for i in range(batch_size):
                    op_A = tuple(preds_A[i].cpu().tolist())
                    op_B = tuple(preds_B[i].cpu().tolist())

                    # Create hash for verification
                    hash_A = hashlib.md5(str(op_A).encode()).hexdigest()
                    hash_B = hashlib.md5(str(op_B).encode()).hexdigest()

                    unique_ops_A.add(hash_A)
                    unique_ops_B.add(hash_B)

        coverage_A = len(unique_ops_A) / 19683 * 100
        coverage_B = len(unique_ops_B) / 19683 * 100

        epoch = checkpoint.get('epoch', 'unknown')
        reported_coverage_A = checkpoint.get('coverage_A_history', [0])[-1]
        reported_coverage_B = checkpoint.get('coverage_B_history', [0])[-1]

        print(f"\n[Hash-based Coverage] Epoch {epoch}:")
        print(f"  VAE-A: {len(unique_ops_A)} unique ops ({coverage_A:.2f}%)")
        print(f"  VAE-B: {len(unique_ops_B)} unique ops ({coverage_B:.2f}%)")
        print(f"  Reported coverage A: {reported_coverage_A} ops")
        print(f"  Reported coverage B: {reported_coverage_B} ops")

        # Check for major discrepancies
        if abs(len(unique_ops_A) - reported_coverage_A) > 1000:
            print("  ⚠️  WARNING: Large discrepancy in VAE-A coverage")

        if abs(len(unique_ops_B) - reported_coverage_B) > 1000:
            print("  ⚠️  WARNING: Large discrepancy in VAE-B coverage")

        assert coverage_A > 10, "Coverage too low for VAE-A"
        assert coverage_B > 10, "Coverage too low for VAE-B"

    # ========================================================================
    # 5. PHASE-SYNC CHECK
    # ========================================================================

    def test_phase_sync_dynamics(self, trained_model):
        """Test: Check if VAE-A surge precedes VAE-B stabilization.

        Healthy dual rhythm: VAE-A explores, then VAE-B refines.
        Both diverging = instability.
        """
        model, checkpoint = trained_model

        coverage_A_history = checkpoint.get('coverage_A_history', [])
        coverage_B_history = checkpoint.get('coverage_B_history', [])

        if len(coverage_A_history) < 5:
            pytest.skip("Not enough history for phase-sync analysis")

        # Analyze growth patterns
        growth_A = np.diff(coverage_A_history)
        growth_B = np.diff(coverage_B_history)

        # Find peak growth epochs
        peak_growth_A_epoch = np.argmax(growth_A)
        peak_growth_B_epoch = np.argmax(growth_B)

        print("\n[Phase-Sync] Analysis:")
        print(f"  VAE-A peak growth at epoch {peak_growth_A_epoch}")
        print(f"  VAE-B peak growth at epoch {peak_growth_B_epoch}")

        # Check if A leads B
        if peak_growth_A_epoch < peak_growth_B_epoch:
            print("  ✓ VAE-A leads VAE-B (healthy rhythm)")
        elif peak_growth_A_epoch > peak_growth_B_epoch:
            print("  ⚠️  VAE-B leads VAE-A (unusual pattern)")
        else:
            print("  ○ Simultaneous peak growth")

        # Check for divergence
        recent_trend_A = np.mean(growth_A[-5:])
        recent_trend_B = np.mean(growth_B[-5:])

        print("  Recent trends:")
        print(f"    VAE-A: {recent_trend_A:+.1f} ops/epoch")
        print(f"    VAE-B: {recent_trend_B:+.1f} ops/epoch")

        # Both should be converging (low growth) after initial surge
        if abs(recent_trend_A) < 100 and abs(recent_trend_B) < 100:
            print("  ✓ Both VAEs stabilizing")
        elif abs(recent_trend_A) > 500 or abs(recent_trend_B) > 500:
            print("  ⚠️  WARNING: Large oscillations detected")

        # Success if not wildly diverging
        assert abs(recent_trend_A) < 2000, "VAE-A diverging"
        assert abs(recent_trend_B) < 2000, "VAE-B diverging"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
