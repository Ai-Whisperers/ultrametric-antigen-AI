import torch


class HyperbolicMatcher:
    """Custom assertions for hyperbolic geometry."""

    @staticmethod
    def assert_on_poincare_disk(tensor, max_norm=1.0, tolerance=1e-5):
        """Asserts all points in tensor have norm < max_norm."""
        norms = torch.norm(tensor, dim=-1)
        if not (norms < max_norm + tolerance).all():
            max_val = norms.max().item()
            raise AssertionError(f"Points outside Poincaré disk! Max norm: {max_val} >= {max_norm}")

    @staticmethod
    def assert_distance_invariant(dist_aa, atol=1e-6):
        """Asserts distance(x, x) is approximately 0."""
        if not torch.allclose(dist_aa, torch.zeros_like(dist_aa), atol=atol):
            raise AssertionError(f"Identity distance violation: max err {dist_aa.max().item()}")


def expect_poincare(tensor):
    return HyperbolicMatcher.assert_on_poincare_disk(tensor)
