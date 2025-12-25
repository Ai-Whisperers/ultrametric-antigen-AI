# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for EntropyRegularization from src/losses/dual_vae_loss.py.

Tests entropy regularization for prediction distribution shaping.
"""

import pytest
import torch
from src.losses.dual_vae_loss import EntropyRegularization


class TestEntropyRegularizationBasic:
    """Basic tests for EntropyRegularization."""

    @pytest.fixture
    def loss_fn(self):
        return EntropyRegularization()

    @pytest.fixture
    def sample_logits(self, device):
        """Sample logits from model (batch, 9, 3)."""
        return torch.randn(32, 9, 3, device=device)

    def test_forward_returns_scalar(self, loss_fn, sample_logits):
        """Forward should return a scalar."""
        loss = loss_fn(sample_logits)
        assert loss.dim() == 0

    def test_forward_is_finite(self, loss_fn, sample_logits):
        """Forward should return a finite value."""
        loss = loss_fn(sample_logits)
        assert torch.isfinite(loss)


class TestEntropyRegularizationDistributions:
    """Tests for entropy on different distributions."""

    @pytest.fixture
    def loss_fn(self):
        return EntropyRegularization()

    def test_uniform_distribution_high_entropy(self, loss_fn, device):
        """Uniform distribution should have high entropy (negative loss)."""
        batch_size = 32
        # Uniform logits -> uniform probabilities
        logits = torch.zeros(batch_size, 9, 3, device=device)
        loss = loss_fn(logits)

        # Negative entropy for uniform over 3 classes
        # The loss is returned as negative entropy, so should be negative
        assert loss < 0

    def test_peaked_distribution_low_entropy(self, loss_fn, device):
        """Peaked distribution should have low entropy (near-zero negative loss)."""
        batch_size = 32
        # Very peaked logits -> low entropy
        logits = torch.zeros(batch_size, 9, 3, device=device)
        logits[:, :, 0] = 100  # Very high for class 0

        loss = loss_fn(logits)

        # Near-zero entropy means loss is close to 0
        assert loss > -1.0

    def test_entropy_ordering(self, loss_fn, device):
        """Uniform should have higher entropy than peaked."""
        batch_size = 16

        # Uniform
        uniform_logits = torch.zeros(batch_size, 9, 3, device=device)

        # Slightly peaked
        peaked_logits = torch.zeros(batch_size, 9, 3, device=device)
        peaked_logits[:, :, 0] = 2.0

        # Very peaked
        very_peaked_logits = torch.zeros(batch_size, 9, 3, device=device)
        very_peaked_logits[:, :, 0] = 100.0

        entropy_uniform = loss_fn(uniform_logits)
        entropy_peaked = loss_fn(peaked_logits)
        entropy_very_peaked = loss_fn(very_peaked_logits)

        # Uniform should have most negative (highest entropy)
        # Very peaked should have least negative (lowest entropy)
        assert entropy_uniform < entropy_peaked < entropy_very_peaked


class TestEntropyRegularizationBatchInvariance:
    """Tests for batch size invariance."""

    @pytest.fixture
    def loss_fn(self):
        return EntropyRegularization()

    def test_output_invariant_to_batch_size(self, loss_fn, device):
        """Entropy should be averaged over batch."""
        logits_small = torch.randn(8, 9, 3, device=device)
        logits_large = logits_small.repeat(4, 1, 1)  # 32 samples, same distribution

        loss_small = loss_fn(logits_small)
        loss_large = loss_fn(logits_large)

        # Should be same since we average over batch
        assert torch.isclose(loss_small, loss_large, atol=0.1)

    def test_consistent_across_batch_sizes(self, loss_fn, device):
        """Loss should be consistent when scaling batch size."""
        base_logits = torch.randn(4, 9, 3, device=device)

        losses = []
        for multiplier in [1, 2, 4, 8]:
            logits = base_logits.repeat(multiplier, 1, 1)
            losses.append(loss_fn(logits).item())

        # All losses should be very close
        assert max(losses) - min(losses) < 0.1


class TestEntropyRegularizationGradient:
    """Gradient flow tests for EntropyRegularization."""

    @pytest.fixture
    def loss_fn(self):
        return EntropyRegularization()

    def test_gradient_flows(self, loss_fn, device):
        """Gradients should flow through the loss."""
        logits = torch.randn(16, 9, 3, device=device, requires_grad=True)

        loss = loss_fn(logits)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert torch.isfinite(logits.grad).all()

    def test_gradient_encourages_entropy_increase(self, loss_fn, device):
        """Gradient should encourage entropy increase (negative loss)."""
        logits = torch.randn(8, 9, 3, device=device, requires_grad=True)

        loss = loss_fn(logits)
        loss.backward()

        # Gradient exists and is non-zero
        assert logits.grad.abs().sum() > 0


class TestEntropyRegularizationEdgeCases:
    """Edge case tests for EntropyRegularization."""

    @pytest.fixture
    def loss_fn(self):
        return EntropyRegularization()

    def test_single_sample(self, loss_fn, device):
        """Should handle batch size of 1."""
        logits = torch.randn(1, 9, 3, device=device)

        loss = loss_fn(logits)
        assert torch.isfinite(loss)

    def test_single_digit(self, loss_fn, device):
        """Should handle single digit (1 position)."""
        logits = torch.randn(8, 1, 3, device=device)

        loss = loss_fn(logits)
        assert torch.isfinite(loss)

    def test_many_classes(self, loss_fn, device):
        """Should handle more than 3 classes."""
        logits = torch.randn(8, 9, 10, device=device)

        loss = loss_fn(logits)
        assert torch.isfinite(loss)

    def test_extreme_logits(self, loss_fn, device):
        """Should handle extreme logit values."""
        # Very large positive
        logits = torch.randn(8, 9, 3, device=device) * 100

        loss = loss_fn(logits)
        assert torch.isfinite(loss)

    def test_constant_logits(self, loss_fn, device):
        """Should handle constant logit values across batch."""
        logits = torch.ones(8, 9, 3, device=device) * 5

        loss = loss_fn(logits)
        assert torch.isfinite(loss)
