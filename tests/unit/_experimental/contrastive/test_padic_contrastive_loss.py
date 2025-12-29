# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for PAdicContrastiveLoss class."""

from __future__ import annotations

import pytest
import torch

from src.contrastive import PAdicContrastiveLoss


class TestPAdicContrastiveLossInit:
    """Tests for PAdicContrastiveLoss initialization."""

    def test_default_init(self):
        """Test default initialization."""
        loss = PAdicContrastiveLoss()
        assert loss.temperature == 0.07
        assert loss.valuation_threshold == 2
        assert loss.prime == 3

    def test_custom_temperature(self):
        """Test custom temperature."""
        loss = PAdicContrastiveLoss(temperature=0.1)
        assert loss.temperature == 0.1

    def test_custom_threshold(self):
        """Test custom valuation threshold."""
        loss = PAdicContrastiveLoss(valuation_threshold=3)
        assert loss.valuation_threshold == 3

    def test_custom_prime(self):
        """Test custom prime."""
        loss = PAdicContrastiveLoss(prime=5)
        assert loss.prime == 5


class TestPAdicValuation:
    """Tests for valuation computation."""

    def test_valuation_zero_diff(self, contrastive_loss, device):
        """Test valuation of zero is max."""
        diff = torch.tensor([0], device=device)
        val = contrastive_loss._compute_valuation(diff)
        assert val.item() == contrastive_loss.max_valuation

    def test_valuation_power_of_p(self, contrastive_loss, device):
        """Test valuation of powers of prime."""
        # p = 3
        diff = torch.tensor([3, 9, 27], device=device)
        val = contrastive_loss._compute_valuation(diff)
        assert val[0].item() == 1
        assert val[1].item() == 2
        assert val[2].item() == 3

    def test_valuation_not_divisible(self, contrastive_loss, device):
        """Test valuation of non-divisible numbers."""
        diff = torch.tensor([1, 2, 5, 7], device=device)
        val = contrastive_loss._compute_valuation(diff)
        assert (val == 0).all()

    def test_valuation_mixed(self, contrastive_loss, device):
        """Test valuation of mixed numbers."""
        # 6 = 2 * 3 -> v_3(6) = 1
        # 18 = 2 * 9 -> v_3(18) = 2
        diff = torch.tensor([6, 18], device=device)
        val = contrastive_loss._compute_valuation(diff)
        assert val[0].item() == 1
        assert val[1].item() == 2


class TestPositiveMask:
    """Tests for positive mask generation."""

    def test_mask_shape(self, contrastive_loss, padic_indices):
        """Test mask has correct shape."""
        mask = contrastive_loss._get_positive_mask(padic_indices)
        batch_size = len(padic_indices)
        assert mask.shape == (batch_size, batch_size)

    def test_mask_excludes_self(self, contrastive_loss, padic_indices):
        """Test diagonal is False (no self-pairs)."""
        mask = contrastive_loss._get_positive_mask(padic_indices)
        assert not mask.diag().any()

    def test_mask_symmetric(self, contrastive_loss, padic_indices):
        """Test mask is symmetric."""
        mask = contrastive_loss._get_positive_mask(padic_indices)
        assert (mask == mask.T).all()

    def test_mask_detects_positives(self, device):
        """Test mask correctly identifies p-adically close pairs."""
        loss = PAdicContrastiveLoss(valuation_threshold=1)
        # 0, 3, 6, 9 are all divisible by 3
        indices = torch.tensor([0, 3, 6, 9], device=device)
        mask = loss._get_positive_mask(indices)

        # All pairs should be positive (differences divisible by 3)
        # Except diagonal
        expected = torch.ones(4, 4, dtype=torch.bool, device=device)
        expected.fill_diagonal_(False)
        assert (mask == expected).all()


class TestContrastiveLossForward:
    """Tests for forward pass."""

    def test_loss_nonnegative(self, contrastive_loss, batch_embeddings, padic_indices):
        """Test loss is non-negative."""
        loss = contrastive_loss(batch_embeddings, padic_indices)
        assert loss >= 0

    def test_loss_finite(self, contrastive_loss, batch_embeddings, padic_indices):
        """Test loss is finite."""
        loss = contrastive_loss(batch_embeddings, padic_indices)
        assert torch.isfinite(loss)

    def test_loss_small_batch(self, contrastive_loss, device):
        """Test loss with batch size 1 returns zero."""
        embeddings = torch.randn(1, 64, device=device)
        indices = torch.tensor([0], device=device)
        loss = contrastive_loss(embeddings, indices)
        assert loss.item() == 0.0

    def test_loss_no_positives(self, device):
        """Test loss when no positive pairs exist."""
        loss_fn = PAdicContrastiveLoss(valuation_threshold=5)
        embeddings = torch.randn(4, 64, device=device)
        # Indices 1, 2, 4, 5 have no pairs with valuation >= 5
        indices = torch.tensor([1, 2, 4, 5], device=device)
        loss = loss_fn(embeddings, indices)
        assert loss.item() == 0.0

    def test_loss_gradient_flows(self, contrastive_loss, device):
        """Test gradients flow through loss."""
        embeddings = torch.randn(8, 64, device=device, requires_grad=True)
        indices = torch.arange(8, device=device) * 3  # All divisible by 3
        loss = contrastive_loss(embeddings, indices)
        loss.backward()
        assert embeddings.grad is not None
        assert torch.isfinite(embeddings.grad).all()


class TestContrastiveLossPrimes:
    """Tests for different primes."""

    @pytest.mark.parametrize("prime", [2, 3, 5, 7])
    def test_different_primes(self, prime, device):
        """Test loss works with different primes."""
        loss_fn = PAdicContrastiveLoss(prime=prime, valuation_threshold=1)
        embeddings = torch.randn(8, 64, device=device)
        indices = torch.arange(8, device=device) * prime
        loss = loss_fn(embeddings, indices)
        assert torch.isfinite(loss)
