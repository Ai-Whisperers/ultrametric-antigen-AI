import pytest
import torch
import torch.testing as testing
from src.geometry.poincare import poincare_distance, project_to_poincare, mobius_add


def test_poincare_distance_identity(device):
    """Verify distance(x, x) approx 0."""
    x = torch.zeros(1, 2, device=device)
    dist = poincare_distance(x, x)
    testing.assert_close(dist, torch.zeros_like(dist), atol=1e-6, rtol=0)


def test_poincare_distance_shape(device):
    """Verify output shape of distance function."""
    x = torch.randn(10, 5, device=device)
    y = torch.randn(10, 5, device=device)
    # Ensure they are on manifold first
    x = project_to_poincare(x)
    y = project_to_poincare(y)

    dist = poincare_distance(x, y, keepdim=True)
    assert dist.shape == (10, 1)

    dist_flat = poincare_distance(x, y, keepdim=False)
    assert dist_flat.shape == (10,)


def test_projection_limits(device):
    """Verify projection constrains norm to < 1.0 (with safe margin)."""
    # Create vectors with large norms
    x = torch.randn(10, 3, device=device) * 10
    x_proj = project_to_poincare(x, max_norm=0.95)

    norms = x_proj.norm(p=2, dim=-1)
    assert (norms <= 0.95 + 1e-5).all(), "Projected points exceed max_norm"


def test_mobius_addition_zero(device):
    """Verify Mobius addition with zero vector: x (+) 0 = x."""
    x = torch.tensor([[0.5, 0.5]], device=device)
    x = project_to_poincare(x)
    zero = torch.zeros_like(x)

    res = mobius_add(x, zero)
    testing.assert_close(res, x, atol=1e-6, rtol=1e-5)


def test_poincare_triangle_inequality(device):
    """Verify dist(a,c) <= dist(a,b) + dist(b,c)."""
    a = project_to_poincare(torch.randn(5, 3, device=device))
    b = project_to_poincare(torch.randn(5, 3, device=device))
    c = project_to_poincare(torch.randn(5, 3, device=device))

    d_ab = poincare_distance(a, b, keepdim=False)
    d_bc = poincare_distance(b, c, keepdim=False)
    d_ac = poincare_distance(a, c, keepdim=False)

    # Allow slight numerical error margin
    assert (d_ac <= d_ab + d_bc + 1e-5).all(), "Triangle inequality violated"
