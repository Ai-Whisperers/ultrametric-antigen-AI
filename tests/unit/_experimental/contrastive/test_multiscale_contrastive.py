# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for MultiScaleContrastive class."""

from __future__ import annotations

import pytest
import torch

from src.contrastive import MultiScaleContrastive


class TestMultiScaleInit:
    """Tests for MultiScaleContrastive initialization."""

    def test_default_init(self):
        """Test default initialization."""
        loss = MultiScaleContrastive()
        assert loss.n_levels == 3
        assert loss.base_temperature == 0.07
        assert len(loss.losses) == 3

    def test_custom_levels(self):
        """Test custom number of levels."""
        loss = MultiScaleContrastive(n_levels=5)
        assert loss.n_levels == 5
        assert len(loss.losses) == 5

    def test_custom_weights(self):
        """Test custom level weights."""
        weights = [0.5, 0.3, 0.2]
        loss = MultiScaleContrastive(n_levels=3, level_weights=weights)
        assert torch.allclose(loss.level_weights, torch.tensor(weights))

    def test_temperature_scaling(self):
        """Test temperature scales across levels."""
        loss = MultiScaleContrastive(
            n_levels=3, base_temperature=0.1, temperature_scale=2.0
        )
        assert loss.losses[0].temperature == 0.1
        assert loss.losses[1].temperature == 0.2
        assert loss.losses[2].temperature == 0.4


class TestMultiScaleForward:
    """Tests for forward pass."""

    def test_returns_tuple(self, multiscale_loss, batch_embeddings, padic_indices):
        """Test forward returns tuple."""
        total_loss, level_losses = multiscale_loss(batch_embeddings, padic_indices)
        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(level_losses, dict)

    def test_level_losses_count(self, multiscale_loss, batch_embeddings, padic_indices):
        """Test correct number of level losses."""
        _, level_losses = multiscale_loss(batch_embeddings, padic_indices)
        assert len(level_losses) == multiscale_loss.n_levels

    def test_level_loss_keys(self, multiscale_loss, batch_embeddings, padic_indices):
        """Test level loss dictionary keys."""
        _, level_losses = multiscale_loss(batch_embeddings, padic_indices)
        for i in range(multiscale_loss.n_levels):
            assert f"level_{i}" in level_losses

    def test_total_loss_finite(self, multiscale_loss, batch_embeddings, padic_indices):
        """Test total loss is finite."""
        total_loss, _ = multiscale_loss(batch_embeddings, padic_indices)
        assert torch.isfinite(total_loss)

    def test_total_loss_nonnegative(self, multiscale_loss, batch_embeddings, padic_indices):
        """Test total loss is non-negative."""
        total_loss, _ = multiscale_loss(batch_embeddings, padic_indices)
        assert total_loss >= 0

    def test_gradient_flows(self, multiscale_loss, device):
        """Test gradients flow through all levels."""
        embeddings = torch.randn(8, 64, device=device, requires_grad=True)
        indices = torch.arange(8, device=device) * 3
        total_loss, _ = multiscale_loss(embeddings, indices)
        total_loss.backward()
        assert embeddings.grad is not None


class TestMultiScaleHierarchy:
    """Tests for hierarchical behavior."""

    def test_different_thresholds(self, device):
        """Test each level uses different threshold."""
        loss = MultiScaleContrastive(n_levels=3)
        assert loss.losses[0].valuation_threshold == 1
        assert loss.losses[1].valuation_threshold == 2
        assert loss.losses[2].valuation_threshold == 3

    def test_hierarchical_indices(self, hierarchical_indices, device):
        """Test with hierarchical index structure."""
        loss = MultiScaleContrastive(n_levels=3)
        embeddings = torch.randn(len(hierarchical_indices), 64, device=device)
        total_loss, level_losses = loss(embeddings, hierarchical_indices)

        # All levels should compute something
        assert torch.isfinite(total_loss)
