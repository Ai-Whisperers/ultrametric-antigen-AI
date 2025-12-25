# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for KLDivergenceLoss from src/losses/dual_vae_loss.py.

Tests KL divergence with optional free bits for VAE training.
"""

import pytest
import torch
from src.losses.dual_vae_loss import KLDivergenceLoss


class TestKLDivergenceLossInit:
    """Initialization tests for KLDivergenceLoss."""

    def test_init_default_free_bits(self):
        """Default free bits should be 0."""
        loss_fn = KLDivergenceLoss()
        assert loss_fn.free_bits == 0.0

    def test_init_custom_free_bits(self):
        """Should accept custom free bits."""
        loss_fn = KLDivergenceLoss(free_bits=2.0)
        assert loss_fn.free_bits == 2.0

    def test_init_various_free_bits(self):
        """Test various free_bits values."""
        for fb in [0.0, 0.5, 1.0, 2.0, 5.0]:
            loss_fn = KLDivergenceLoss(free_bits=fb)
            assert loss_fn.free_bits == fb


class TestKLDivergenceLossBasic:
    """Basic tests for KLDivergenceLoss."""

    @pytest.fixture
    def loss_fn(self):
        return KLDivergenceLoss()

    @pytest.fixture
    def sample_mu(self, device):
        """Sample mean of variational posterior."""
        return torch.randn(32, 16, device=device)

    @pytest.fixture
    def sample_logvar(self, device):
        """Sample log variance of variational posterior."""
        return torch.randn(32, 16, device=device)

    def test_forward_returns_scalar(self, loss_fn, sample_mu, sample_logvar):
        """Forward should return a scalar loss."""
        loss = loss_fn(sample_mu, sample_logvar)

        assert loss.dim() == 0
        assert loss.dtype == torch.float32

    def test_kl_is_non_negative(self, loss_fn, sample_mu, sample_logvar):
        """KL divergence should be non-negative."""
        loss = loss_fn(sample_mu, sample_logvar)
        assert loss >= 0


class TestKLDivergenceLossStandardNormal:
    """Tests for KL divergence from standard normal."""

    @pytest.fixture
    def loss_fn(self):
        return KLDivergenceLoss()

    def test_kl_zero_for_standard_normal(self, loss_fn, device):
        """KL(N(0,1) || N(0,1)) should be 0."""
        batch_size, latent_dim = 32, 16
        mu = torch.zeros(batch_size, latent_dim, device=device)
        logvar = torch.zeros(batch_size, latent_dim, device=device)

        loss = loss_fn(mu, logvar)
        assert loss < 0.01  # Should be very close to 0

    def test_kl_increases_with_shifted_mean(self, loss_fn, device):
        """KL should increase when mean deviates from 0."""
        batch_size, latent_dim = 32, 16
        mu_zero = torch.zeros(batch_size, latent_dim, device=device)
        mu_shifted = torch.ones(batch_size, latent_dim, device=device) * 2
        logvar = torch.zeros(batch_size, latent_dim, device=device)

        kl_zero = loss_fn(mu_zero, logvar)
        kl_shifted = loss_fn(mu_shifted, logvar)

        assert kl_shifted > kl_zero

    def test_kl_increases_with_larger_variance(self, loss_fn, device):
        """KL should increase when variance deviates from 1."""
        batch_size, latent_dim = 32, 16
        mu = torch.zeros(batch_size, latent_dim, device=device)
        logvar_one = torch.zeros(batch_size, latent_dim, device=device)  # var = 1
        logvar_large = torch.ones(batch_size, latent_dim, device=device) * 2  # var = e^2

        kl_one = loss_fn(mu, logvar_one)
        kl_large = loss_fn(mu, logvar_large)

        assert kl_large > kl_one


class TestKLDivergenceLossFreeBits:
    """Tests for free bits mechanism."""

    @pytest.fixture
    def sample_mu(self, device):
        return torch.randn(32, 16, device=device)

    @pytest.fixture
    def sample_logvar(self, device):
        return torch.randn(32, 16, device=device)

    def test_free_bits_changes_kl(self, sample_mu, sample_logvar):
        """Free bits should change effective KL."""
        kl_no_free_bits = KLDivergenceLoss(free_bits=0.0)
        kl_with_free_bits = KLDivergenceLoss(free_bits=1.0)

        loss_no_fb = kl_no_free_bits(sample_mu, sample_logvar)
        loss_with_fb = kl_with_free_bits(sample_mu, sample_logvar)

        # With free bits, KL is modified
        assert loss_no_fb != loss_with_fb

    def test_increasing_free_bits(self, sample_mu, sample_logvar):
        """Test effect of increasing free bits values."""
        losses = []
        for fb in [0.0, 0.5, 1.0, 2.0]:
            loss_fn = KLDivergenceLoss(free_bits=fb)
            losses.append(loss_fn(sample_mu, sample_logvar).item())

        # Verify that the losses are different
        assert len(set(losses)) > 1


class TestKLDivergenceLossGradient:
    """Gradient flow tests for KLDivergenceLoss."""

    @pytest.fixture
    def loss_fn(self):
        return KLDivergenceLoss()

    def test_gradient_flows_to_mu(self, loss_fn, device):
        """Gradients should flow to mu."""
        mu = torch.randn(32, 16, device=device, requires_grad=True)
        logvar = torch.randn(32, 16, device=device)

        loss = loss_fn(mu, logvar)
        loss.backward()

        assert mu.grad is not None
        assert not torch.isnan(mu.grad).any()

    def test_gradient_flows_to_logvar(self, loss_fn, device):
        """Gradients should flow to logvar."""
        mu = torch.randn(32, 16, device=device)
        logvar = torch.randn(32, 16, device=device, requires_grad=True)

        loss = loss_fn(mu, logvar)
        loss.backward()

        assert logvar.grad is not None
        assert not torch.isnan(logvar.grad).any()

    def test_gradient_flows_to_both(self, loss_fn, device):
        """Gradients should flow through both mu and logvar."""
        mu = torch.randn(32, 16, device=device, requires_grad=True)
        logvar = torch.randn(32, 16, device=device, requires_grad=True)

        loss = loss_fn(mu, logvar)
        loss.backward()

        assert mu.grad is not None
        assert logvar.grad is not None
        assert torch.isfinite(mu.grad).all()
        assert torch.isfinite(logvar.grad).all()


class TestKLDivergenceLossEdgeCases:
    """Edge case tests for KLDivergenceLoss."""

    @pytest.fixture
    def loss_fn(self):
        return KLDivergenceLoss()

    def test_single_sample(self, loss_fn, device):
        """Should handle batch size of 1."""
        mu = torch.randn(1, 16, device=device)
        logvar = torch.randn(1, 16, device=device)

        loss = loss_fn(mu, logvar)
        assert torch.isfinite(loss)

    def test_small_latent_dim(self, loss_fn, device):
        """Should handle small latent dimension."""
        mu = torch.randn(8, 2, device=device)
        logvar = torch.randn(8, 2, device=device)

        loss = loss_fn(mu, logvar)
        assert torch.isfinite(loss)

    def test_large_latent_dim(self, loss_fn, device):
        """Should handle large latent dimension."""
        mu = torch.randn(8, 128, device=device)
        logvar = torch.randn(8, 128, device=device)

        loss = loss_fn(mu, logvar)
        assert torch.isfinite(loss)

    def test_very_negative_logvar(self, loss_fn, device):
        """Should handle very negative logvar (small variance)."""
        mu = torch.zeros(8, 16, device=device)
        logvar = torch.full((8, 16), -10.0, device=device)

        loss = loss_fn(mu, logvar)
        assert torch.isfinite(loss)

    def test_very_positive_logvar(self, loss_fn, device):
        """Should handle very positive logvar (large variance)."""
        mu = torch.zeros(8, 16, device=device)
        logvar = torch.full((8, 16), 5.0, device=device)

        loss = loss_fn(mu, logvar)
        assert torch.isfinite(loss)
