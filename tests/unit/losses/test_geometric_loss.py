import pytest
import torch

from src.losses.geometric_loss import GeometricAlignmentLoss


@pytest.fixture
def geometric_loss():
    return GeometricAlignmentLoss(symmetry_group="tetrahedral", scale=1.0)


def test_initialization():
    loss_fn = GeometricAlignmentLoss(symmetry_group="octahedral")
    assert loss_fn.target_vertices.shape[0] == 6
    assert loss_fn.target_vertices.shape[1] == 3


def test_perfect_alignment_tetrahedral():
    """Test that points perfectly aligned with tetrahedral vertices yield ~0 loss."""
    loss_fn = GeometricAlignmentLoss(symmetry_group="tetrahedral")

    # Vertices of tetrahedron
    targets = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
    )
    # Normalize to match loss internal normalization
    targets = torch.nn.functional.normalize(targets, dim=1)

    # Create a batch that duplicates these targets
    # Batch size 4
    loss, metrics = loss_fn(targets)

    # Alignment loss should be near zero (numerical precision)
    assert metrics["geo_alignment_loss"] < 1e-5
    assert metrics["geo_coverage_loss"] < 1e-5


def test_high_loss_for_noise():
    """Test that random points yield significantly higher loss."""
    loss_fn = GeometricAlignmentLoss(symmetry_group="icosahedral")

    torch.manual_seed(42)
    # Random batch of 12 points
    z = torch.randn(12, 3)
    loss, metrics = loss_fn(z)

    assert metrics["geo_alignment_loss"] > 0.01


def test_forward_shape_handling():
    """Ensure it handles dim > 3 by slicing."""
    loss_fn = GeometricAlignmentLoss()
    z = torch.randn(10, 16)  # 16 dimensions
    loss, metrics = loss_fn(z)

    assert isinstance(loss, torch.Tensor)
    assert "mean_rmsd" in metrics
