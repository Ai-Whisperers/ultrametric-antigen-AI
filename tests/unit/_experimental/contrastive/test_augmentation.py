# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for ContrastiveDataAugmentation class."""

from __future__ import annotations

import pytest
import torch

from src.contrastive import ContrastiveDataAugmentation


class TestAugmentationInit:
    """Tests for ContrastiveDataAugmentation initialization."""

    def test_default_init(self):
        """Test default initialization."""
        aug = ContrastiveDataAugmentation()
        assert aug.noise_scale == 0.1
        assert aug.mask_prob == 0.15

    def test_custom_params(self):
        """Test custom parameters."""
        aug = ContrastiveDataAugmentation(noise_scale=0.2, mask_prob=0.3)
        assert aug.noise_scale == 0.2
        assert aug.mask_prob == 0.3


class TestNoiseAugmentation:
    """Tests for noise augmentation."""

    def test_add_noise_shape(self, augmentation, device):
        """Test noise preserves shape."""
        x = torch.randn(8, 64, device=device)
        y = augmentation.add_noise(x)
        assert y.shape == x.shape

    def test_add_noise_modifies(self, augmentation, device):
        """Test noise modifies input."""
        x = torch.randn(8, 64, device=device)
        y = augmentation.add_noise(x)
        assert not torch.allclose(x, y)

    def test_noise_scale(self, device):
        """Test noise magnitude scales with parameter."""
        aug_small = ContrastiveDataAugmentation(noise_scale=0.01)
        aug_large = ContrastiveDataAugmentation(noise_scale=1.0)

        x = torch.zeros(100, 64, device=device)
        y_small = aug_small.add_noise(x)
        y_large = aug_large.add_noise(x)

        # Larger scale should have larger magnitude
        assert y_small.abs().mean() < y_large.abs().mean()


class TestMaskAugmentation:
    """Tests for mask augmentation."""

    def test_random_mask_shape(self, augmentation, device):
        """Test mask preserves shape."""
        x = torch.randn(8, 64, device=device)
        y = augmentation.random_mask(x)
        assert y.shape == x.shape

    def test_random_mask_zeros(self, device):
        """Test mask creates zeros."""
        aug = ContrastiveDataAugmentation(mask_prob=0.5)
        x = torch.ones(1000, 64, device=device)
        y = aug.random_mask(x)
        # With 50% mask prob, should have many zeros
        zero_ratio = (y == 0).float().mean()
        assert 0.4 < zero_ratio < 0.6


class TestDropoutAugmentation:
    """Tests for dropout augmentation."""

    def test_dropout_shape(self, augmentation, device):
        """Test dropout preserves shape."""
        x = torch.randn(8, 64, device=device)
        y = augmentation.dropout(x, p=0.1)
        assert y.shape == x.shape

    def test_dropout_training_mode(self, augmentation, device):
        """Test dropout in training mode."""
        augmentation.training = True
        x = torch.ones(1000, 64, device=device)
        y = augmentation.dropout(x, p=0.5)
        # Should have some zeros
        assert (y == 0).any()

    def test_dropout_eval_mode(self, device):
        """Test dropout in eval mode."""
        aug = ContrastiveDataAugmentation()
        aug.training = False
        x = torch.ones(8, 64, device=device)
        y = aug.dropout(x, p=0.5)
        # Should be unchanged
        assert torch.allclose(x, y)


class TestCallableInterface:
    """Tests for __call__ interface."""

    def test_call_noise(self, augmentation, device):
        """Test call with noise augmentation."""
        x = torch.randn(8, 64, device=device)
        y = augmentation(x, augmentation="noise")
        assert not torch.allclose(x, y)

    def test_call_mask(self, augmentation, device):
        """Test call with mask augmentation."""
        x = torch.ones(8, 64, device=device)
        y = augmentation(x, augmentation="mask")
        # Should have some zeros
        assert (y == 0).any()

    def test_call_dropout(self, augmentation, device):
        """Test call with dropout augmentation."""
        x = torch.ones(8, 64, device=device)
        y = augmentation(x, augmentation="dropout")
        assert y.shape == x.shape

    def test_call_unknown(self, augmentation, device):
        """Test call with unknown augmentation returns input."""
        x = torch.randn(8, 64, device=device)
        y = augmentation(x, augmentation="unknown")
        assert torch.allclose(x, y)
