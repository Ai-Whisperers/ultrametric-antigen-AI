# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Fixtures for contrastive learning tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class SimpleEncoder(nn.Module):
    """Simple encoder for testing."""

    def __init__(self, input_dim: int = 32, output_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")


@pytest.fixture
def simple_encoder(device):
    """Create a simple encoder."""
    return SimpleEncoder(input_dim=32, output_dim=64).to(device)


@pytest.fixture
def batch_embeddings(device):
    """Create batch of embeddings."""
    return torch.randn(16, 64, device=device)


@pytest.fixture
def padic_indices(device):
    """Create p-adic indices with known structure.

    Creates indices that form hierarchy groups:
    - Indices 0, 3, 6, 9, 12, 15 are divisible by 3 (valuation >= 1)
    - Indices 0, 9 are divisible by 9 (valuation >= 2)
    """
    return torch.arange(16, device=device)


@pytest.fixture
def hierarchical_indices(device):
    """Create indices with clear hierarchical structure.

    Groups:
    - 0, 27: valuation >= 3 (divisible by 27)
    - 0, 9, 18, 27: valuation >= 2 (divisible by 9)
    - 0, 3, 6, 9, ..., 27: valuation >= 1 (divisible by 3)
    """
    return torch.tensor([0, 3, 6, 9, 18, 27, 1, 2], device=device)


@pytest.fixture
def contrastive_loss():
    """Create p-adic contrastive loss."""
    from src.contrastive import PAdicContrastiveLoss
    return PAdicContrastiveLoss(temperature=0.1, valuation_threshold=1)


@pytest.fixture
def multiscale_loss():
    """Create multi-scale contrastive loss."""
    from src.contrastive import MultiScaleContrastive
    return MultiScaleContrastive(n_levels=3, base_temperature=0.1)


@pytest.fixture
def simclr_encoder(simple_encoder, device):
    """Create SimCLR encoder."""
    from src.contrastive import SimCLREncoder
    return SimCLREncoder(
        base_encoder=simple_encoder,
        representation_dim=64,
        projection_dim=32,
        hidden_dim=48,
    ).to(device)


@pytest.fixture
def moco_encoder(simple_encoder, device):
    """Create MoCo encoder with small queue for testing."""
    from src.contrastive import MomentumContrastEncoder
    return MomentumContrastEncoder(
        encoder=simple_encoder,
        dim=64,
        queue_size=64,
        momentum=0.99,
        temperature=0.1,
    ).to(device)


@pytest.fixture
def positive_sampler():
    """Create p-adic positive sampler."""
    from src.contrastive import PAdicPositiveSampler
    return PAdicPositiveSampler(min_valuation=1, max_valuation=9, prime=3)


@pytest.fixture
def augmentation():
    """Create data augmentation."""
    from src.contrastive import ContrastiveDataAugmentation
    return ContrastiveDataAugmentation(noise_scale=0.1, mask_prob=0.15)
