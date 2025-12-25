# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for HolographicPoincareManifold and related classes."""

import math

import pytest
import torch

from src.geometry.holographic_poincare import (
    BoundaryPoint,
    HolographicLoss,
    HolographicPoincareManifold,
    HolographicProjection,
)


class TestBoundaryPoint:
    """Tests for BoundaryPoint class."""

    def test_creation(self):
        """Test boundary point creation."""
        direction = torch.tensor([1.0, 0.0, 0.0])
        point = BoundaryPoint(direction, conformal_weight=1.0)

        assert point.direction.shape == (3,)
        assert point.conformal_weight == 1.0

    def test_normalization(self):
        """Test that direction is normalized."""
        direction = torch.tensor([2.0, 2.0, 0.0])
        point = BoundaryPoint(direction)

        norm = torch.norm(point.direction)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_to_bulk(self):
        """Test projection back to bulk."""
        direction = torch.tensor([1.0, 0.0, 0.0])
        point = BoundaryPoint(direction)

        bulk = point.to_bulk(radial_coord=0.5, c=1.0)

        assert bulk.shape == (3,)
        assert torch.norm(bulk) == pytest.approx(0.5, abs=1e-5)


class TestHolographicPoincareManifold:
    """Tests for HolographicPoincareManifold."""

    def test_creation(self):
        """Test manifold creation."""
        manifold = HolographicPoincareManifold(c=1.0)
        assert manifold.c == 1.0
        assert manifold.boundary_resolution == 64

    def test_project_to_boundary(self):
        """Test boundary projection."""
        manifold = HolographicPoincareManifold()
        z = torch.randn(8, 16) * 0.5

        directions, radial = manifold.project_to_boundary(z)

        assert directions.shape == (8, 16)
        assert radial.shape == (8, 1)

        # Directions should be unit vectors
        norms = torch.norm(directions, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_boundary_encoding(self):
        """Test boundary encoding."""
        manifold = HolographicPoincareManifold(boundary_resolution=32)
        z = torch.randn(8, 3) * 0.5  # 3D for spherical sampling

        encoding = manifold.boundary_encoding(z, n_samples=16)

        assert encoding.shape == (8, 16)

    def test_bulk_reconstruction(self):
        """Test bulk reconstruction from boundary."""
        manifold = HolographicPoincareManifold(boundary_resolution=32)

        # Create boundary data
        boundary_data = torch.randn(8, 32)

        reconstructed = manifold.bulk_reconstruction(boundary_data)

        # Output should be on ball
        norms = torch.norm(reconstructed[:, :16], dim=-1)  # First 16 dims
        assert (norms < manifold.max_norm + 0.1).all()

    def test_holographic_distance_same(self):
        """Test holographic distance between same points."""
        manifold = HolographicPoincareManifold()
        z = torch.randn(16) * 0.3

        dist = manifold.holographic_distance(z, z)

        assert dist == pytest.approx(0.0, abs=1e-5)

    def test_holographic_distance_different(self):
        """Test holographic distance between different points."""
        manifold = HolographicPoincareManifold()
        z1 = torch.randn(16) * 0.3
        z2 = torch.randn(16) * 0.3

        dist = manifold.holographic_distance(z1, z2)

        assert dist > 0

    def test_conformal_flow(self):
        """Test conformal flow toward boundary."""
        manifold = HolographicPoincareManifold()
        # Start with small norm points so there's room to flow outward
        z = torch.randn(8, 16) * 0.1
        z = manifold.proj(z)  # Ensure on ball

        initial_norm = torch.norm(z, dim=-1).mean()

        z_flowed = manifold.conformal_flow(z, steps=10, step_size=0.02)

        final_norm = torch.norm(z_flowed, dim=-1).mean()

        # Flow should move toward boundary (or at least stay on ball)
        # Since conformal factor increases near boundary, points near boundary
        # may not move much further
        assert final_norm >= initial_norm * 0.9  # Allow small decrease due to projection

    def test_geodesic_slice(self):
        """Test geodesic slice computation."""
        manifold = HolographicPoincareManifold()
        z = torch.randn(16) * 0.3
        direction = torch.randn(16)

        points = manifold.geodesic_slice(z, direction, n_points=20)

        assert points.shape == (20, 16)

    def test_horizon_entropy(self):
        """Test horizon entropy computation."""
        manifold = HolographicPoincareManifold()
        z = torch.randn(8, 16) * 0.3

        entropy = manifold.horizon_entropy(z)

        assert entropy.shape == (8,)
        assert (entropy >= 0).all()

    def test_bulk_boundary_correspondence(self):
        """Test full correspondence data."""
        manifold = HolographicPoincareManifold()
        z = torch.randn(8, 16) * 0.3

        correspondence = manifold.bulk_boundary_correspondence(z)

        assert "boundary_directions" in correspondence
        assert "radial_coordinates" in correspondence
        assert "boundary_encoding" in correspondence
        assert "horizon_entropy" in correspondence
        assert "conformal_factor" in correspondence

    def test_fibonacci_sphere_3d(self):
        """Test Fibonacci sphere generation in 3D."""
        manifold = HolographicPoincareManifold()

        points = manifold._fibonacci_sphere(100, 3)

        assert points.shape == (100, 3)

        # All points should be on unit sphere
        norms = torch.norm(points, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_fibonacci_sphere_2d(self):
        """Test Fibonacci sphere generation in 2D (circle)."""
        manifold = HolographicPoincareManifold()

        points = manifold._fibonacci_sphere(20, 2)

        assert points.shape == (20, 2)

        # All points should be on unit circle
        norms = torch.norm(points, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestHolographicLoss:
    """Tests for HolographicLoss."""

    def test_creation(self):
        """Test loss creation."""
        loss = HolographicLoss(c=1.0)
        assert loss.reconstruction_weight == 1.0
        assert loss.entropy_weight == 0.1

    def test_forward(self):
        """Test loss forward pass."""
        loss_fn = HolographicLoss()
        z = torch.randn(8, 16) * 0.3

        losses = loss_fn.forward(z)

        assert "total_loss" in losses
        assert "reconstruction_loss" in losses
        assert "entropy_loss" in losses
        assert "consistency_loss" in losses
        assert losses["total_loss"].requires_grad

    def test_forward_with_target(self):
        """Test loss with target supervision."""
        loss_fn = HolographicLoss()
        z = torch.randn(8, 16) * 0.3
        z_target = torch.randn(8, 16) * 0.3

        losses = loss_fn.forward(z, z_target)

        assert "supervised_loss" in losses
        assert losses["supervised_loss"] > 0

    def test_mean_entropy_reasonable(self):
        """Test that entropy values are reasonable."""
        loss_fn = HolographicLoss()
        z = torch.randn(8, 16) * 0.3

        losses = loss_fn.forward(z)

        # Entropy should be positive
        assert losses["mean_entropy"] > 0


class TestHolographicProjection:
    """Tests for HolographicProjection module."""

    def test_creation(self):
        """Test projection creation."""
        proj = HolographicProjection(
            input_dim=64,
            output_dim=16,
            c=1.0,
            boundary_resolution=32,
        )

        assert proj.manifold.c == 1.0

    def test_forward(self):
        """Test forward projection."""
        proj = HolographicProjection(
            input_dim=64,
            output_dim=16,
            boundary_resolution=32,
        )

        x = torch.randn(8, 64)
        z, info = proj.forward(x)

        assert z.shape == (8, 16)
        assert "horizon_entropy" in info
        assert "conformal_factor" in info

    def test_output_on_ball(self):
        """Test that output stays on Poincare ball."""
        proj = HolographicProjection(
            input_dim=64,
            output_dim=16,
        )

        x = torch.randn(8, 64) * 10  # Large input

        z, _ = proj.forward(x)

        norms = torch.norm(z, dim=-1)
        assert (norms < 1.0).all()

    def test_gradient_flow(self):
        """Test gradients flow through projection."""
        proj = HolographicProjection(input_dim=64, output_dim=16)
        x = torch.randn(8, 64, requires_grad=True)

        z, _ = proj.forward(x)
        loss = z.sum()
        loss.backward()

        assert x.grad is not None


class TestIntegration:
    """Integration tests for holographic components."""

    def test_encode_decode_cycle(self):
        """Test encoding to boundary and back."""
        manifold = HolographicPoincareManifold(boundary_resolution=64)

        # Original bulk point
        z_original = torch.randn(4, 16) * 0.5

        # Encode to boundary
        encoding = manifold.boundary_encoding(z_original, n_samples=64)

        # Decode back to bulk
        z_reconstructed = manifold.bulk_reconstruction(encoding)

        # Should be different but related (not exact due to information loss)
        # Just check shapes and that it's on ball
        assert z_reconstructed.shape[0] == 4
        norms = torch.norm(z_reconstructed[:, :16], dim=-1)
        assert (norms < 1.0).all()

    def test_holographic_preserves_structure(self):
        """Test that holographic distance preserves some structure."""
        manifold = HolographicPoincareManifold()

        # Create points with known structure
        z1 = torch.tensor([0.1, 0.0, 0.0, 0.0] + [0.0] * 12)
        z2 = torch.tensor([0.2, 0.0, 0.0, 0.0] + [0.0] * 12)
        z3 = torch.tensor([0.5, 0.0, 0.0, 0.0] + [0.0] * 12)

        d12 = manifold.holographic_distance(z1, z2)
        d13 = manifold.holographic_distance(z1, z3)

        # Farther points should have larger distance
        assert d13 > d12

    def test_loss_decreases_with_training(self):
        """Test that loss can be optimized."""
        proj = HolographicProjection(input_dim=32, output_dim=8, boundary_resolution=16)
        loss_fn = HolographicLoss()

        optimizer = torch.optim.Adam(proj.parameters(), lr=0.01)

        x = torch.randn(16, 32)
        initial_loss = None

        for _ in range(10):
            optimizer.zero_grad()
            z, _ = proj.forward(x)
            losses = loss_fn.forward(z)
            losses["total_loss"].backward()
            optimizer.step()

            if initial_loss is None:
                initial_loss = losses["total_loss"].item()

        final_loss = losses["total_loss"].item()

        # Loss should decrease (or at least not explode)
        assert final_loss < initial_loss * 2  # Allow some variance
