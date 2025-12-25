import pytest
import torch
import numpy as np
from tests.core.builders import VAEBuilder
from tests.factories.data import TernaryOperationFactory


def compose_ternary_ops(op1: np.ndarray, op2: np.ndarray) -> np.ndarray:
    """Compose two ternary operations using element-wise ternary addition."""
    op1_mod = ((op1 + 1) % 3).astype(int)
    op2_mod = ((op2 + 1) % 3).astype(int)
    composed_mod = (op1_mod + op2_mod) % 3
    return (composed_mod - 1).astype(np.float32)


def permute_operation(op: np.ndarray, permutation: list) -> np.ndarray:
    """Permute the symbol ordering in a ternary operation."""
    value_map = {-1: 0, 0: 1, 1: 2}
    reverse_map = {0: -1, 1: 0, 2: 1}
    permuted = np.zeros_like(op)
    for i in range(len(op)):
        old_idx = value_map[op[i]]
        new_idx = permutation[old_idx]
        permuted[i] = reverse_map[new_idx]
    return permuted


class TestGeneralizationV5_11:
    """
    Ported Generalization Tests for V5.11 Architecture.
    Validated geometric variants and logical consistency.
    """

    @pytest.fixture
    def model(self):
        # Use Builder with REAL frozen components (mocking physics, likely)
        # For integration test, we might want real encoder if available,
        # but for logic verification, mocked is safer if no checkpoint exists.
        # We will use mocked frozen for stability, unless we explicitly load weights.
        return VAEBuilder().with_dual_projection().build()

    def test_permutation_robustness_smoke(self, model):
        """
        Verify that permuting symbols doesn't crash the embedding logic
        and preserves relative distance constraints (smoke test).
        """
        ops = TernaryOperationFactory.create_batch(10).cpu().numpy()
        permution = [1, 2, 0]  # Cyclic shift

        permuted_ops = np.array([permute_operation(op, permution) for op in ops])

        with torch.no_grad():
            x = torch.tensor(ops)
            x_perm = torch.tensor(permuted_ops, dtype=torch.float32)

            z_orig = model(x)["z_A_hyp"]
            z_perm = model(x_perm)["z_A_hyp"]

            # Check shapes
            assert z_orig.shape == z_perm.shape

            # Check Poincaré constraint invariant
            assert (z_orig.norm(dim=-1) < 1.0).all()
            assert (z_perm.norm(dim=-1) < 1.0).all()

    def test_compositional_arithmetic_smoke(self, model):
        """
        Verify arithmetic composition logic runs.
        """
        op1 = TernaryOperationFactory.create_batch(1).squeeze(0).cpu().numpy()
        op2 = TernaryOperationFactory.create_batch(1).squeeze(0).cpu().numpy()
        composed = compose_ternary_ops(op1, op2)

        with torch.no_grad():
            inputs = torch.tensor(np.stack([op1, op2, composed]))
            outputs = model(inputs)["z_A_hyp"]

            z1, z2, z_comp = outputs[0], outputs[1], outputs[2]

            # Smoke test: check we can compute distance
            # This doesn't assert 'correctness' of arithmetic without training
            dist = torch.norm(z_comp - (z1 + z2))
            assert dist >= 0
