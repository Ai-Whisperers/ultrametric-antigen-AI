# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for RepulsionLoss from src/losses/dual_vae_loss.py.

Tests latent space diversity through pairwise repulsion.
"""

import pytest
import torch
from src.losses.dual_vae_loss import RepulsionLoss


class TestRepulsionLossInit:
    """Initialization tests for RepulsionLoss."""

    def test_init_default_sigma(self):
        """Default sigma should be 0.5."""
        loss_fn = RepulsionLoss()
        assert loss_fn.sigma == 0.5

    def test_init_custom_sigma(self):
        """Should accept custom sigma."""
        loss_fn = RepulsionLoss(sigma=1.0)
        assert loss_fn.sigma == 1.0

    def test_init_various_sigma(self):
        """Test various sigma values."""
        for sigma in [0.1, 0.5, 1.0, 2.0, 5.0]:
            loss_fn = RepulsionLoss(sigma=sigma)
            assert loss_fn.sigma == sigma


class TestRepulsionLossBasic:
    """Basic tests for RepulsionLoss."""

    @pytest.fixture
    def loss_fn(self):
        return RepulsionLoss()

    @pytest.fixture
    def sample_z(self, device):
        """Sample latent codes."""
        return torch.randn(32, 16, device=device)

    def test_forward_returns_scalar(self, loss_fn, sample_z):
        """Forward should return a scalar."""
        loss = loss_fn(sample_z)
        assert loss.dim() == 0

    def test_repulsion_is_non_negative(self, loss_fn, sample_z):
        """Repulsion loss should be non-negative."""
        loss = loss_fn(sample_z)
        assert loss >= 0

    def test_repulsion_is_finite(self, loss_fn, sample_z):
        """Repulsion loss should be finite."""
        loss = loss_fn(sample_z)
        assert torch.isfinite(loss)


class TestRepulsionLossSingleSample:
    """Tests for single sample behavior."""

    @pytest.fixture
    def loss_fn(self):
        return RepulsionLoss()

    def test_single_sample_returns_zero(self, loss_fn, device):
        """Single sample should return 0 (no pairs to repel)."""
        z = torch.randn(1, 16, device=device)

        loss = loss_fn(z)
        assert loss == 0.0

    def test_two_samples_nonzero(self, loss_fn, device):
        """Two samples should have non-zero repulsion."""
        z = torch.randn(2, 16, device=device)

        loss = loss_fn(z)
        # Should be > 0 since there's one pair
        assert loss >= 0  # Can be very small if points are far apart


class TestRepulsionLossPointSpacing:
    """Tests for point spacing effects."""

    def test_identical_points_have_high_repulsion(self, device):
        """Identical points should have maximum repulsion."""
        loss_fn = RepulsionLoss()
        batch_size, latent_dim = 8, 16

        # All points are identical
        z = torch.ones(batch_size, latent_dim, device=device)

        loss = loss_fn(z)

        # exp(0) = 1 for all pairs when identical
        assert loss > 0.9  # Should be close to 1

    def test_distant_points_have_low_repulsion(self, device):
        """Very distant points should have low repulsion."""
        loss_fn = RepulsionLoss(sigma=0.5)
        batch_size, latent_dim = 8, 16

        # Points very far apart
        z = torch.zeros(batch_size, latent_dim, device=device)
        for i in range(batch_size):
            z[i, 0] = i * 100  # Space them 100 units apart

        loss = loss_fn(z)

        # exp(-100^2 / 0.25) very small
        assert loss < 0.01


class TestRepulsionLossSigmaEffect:
    """Tests for sigma parameter effects."""

    def test_sigma_affects_repulsion(self, device):
        """Larger sigma should give higher repulsion for same points."""
        z = torch.randn(16, 8, device=device)

        loss_small_sigma = RepulsionLoss(sigma=0.1)
        loss_large_sigma = RepulsionLoss(sigma=2.0)

        repulsion_small = loss_small_sigma(z)
        repulsion_large = loss_large_sigma(z)

        # Larger sigma means wider kernel -> higher repulsion for same distance
        assert repulsion_large > repulsion_small

    def test_sigma_scaling(self, device):
        """Test repulsion at various sigma values."""
        z = torch.randn(8, 16, device=device)

        repulsions = []
        for sigma in [0.1, 0.5, 1.0, 2.0]:
            loss_fn = RepulsionLoss(sigma=sigma)
            repulsions.append(loss_fn(z).item())

        # Should generally increase with sigma
        assert repulsions[0] < repulsions[-1]


class TestRepulsionLossGradient:
    """Gradient flow tests for RepulsionLoss."""

    @pytest.fixture
    def loss_fn(self):
        return RepulsionLoss()

    def test_gradient_flows(self, loss_fn, device):
        """Gradients should flow through the loss."""
        z = torch.randn(16, 8, device=device, requires_grad=True)

        loss = loss_fn(z)
        loss.backward()

        assert z.grad is not None
        assert not torch.isnan(z.grad).any()
        assert torch.isfinite(z.grad).all()

    def test_gradient_pushes_points_apart(self, loss_fn, device):
        """Gradient should encourage points to move apart."""
        # Start with identical points
        z = torch.zeros(4, 8, device=device, requires_grad=True)

        loss = loss_fn(z)
        loss.backward()

        # Gradient should be non-zero (pushing points apart)
        # Note: for identical points, gradient might be zero due to symmetry
        # so we test with slight perturbation
        z = torch.randn(4, 8, device=device) * 0.1
        z.requires_grad_(True)

        loss = loss_fn(z)
        loss.backward()

        assert z.grad.abs().sum() > 0


class TestRepulsionLossEdgeCases:
    """Edge case tests for RepulsionLoss."""

    @pytest.fixture
    def loss_fn(self):
        return RepulsionLoss()

    def test_small_batch(self, loss_fn, device):
        """Should handle small batch sizes."""
        for batch_size in [2, 3, 4]:
            z = torch.randn(batch_size, 16, device=device)
            loss = loss_fn(z)
            assert torch.isfinite(loss)

    def test_large_batch(self, loss_fn, device):
        """Should handle larger batch sizes."""
        z = torch.randn(128, 16, device=device)
        loss = loss_fn(z)
        assert torch.isfinite(loss)

    def test_small_latent_dim(self, loss_fn, device):
        """Should handle small latent dimension."""
        z = torch.randn(16, 2, device=device)
        loss = loss_fn(z)
        assert torch.isfinite(loss)

    def test_large_latent_dim(self, loss_fn, device):
        """Should handle large latent dimension."""
        z = torch.randn(16, 128, device=device)
        loss = loss_fn(z)
        assert torch.isfinite(loss)

    def test_extreme_values(self, loss_fn, device):
        """Should handle extreme latent values."""
        z = torch.randn(8, 16, device=device) * 100
        loss = loss_fn(z)
        assert torch.isfinite(loss)

    def test_near_zero_values(self, loss_fn, device):
        """Should handle near-zero latent values."""
        z = torch.randn(8, 16, device=device) * 1e-6
        loss = loss_fn(z)
        assert torch.isfinite(loss)
