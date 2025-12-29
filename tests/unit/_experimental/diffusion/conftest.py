# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Fixtures for diffusion model tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")


@pytest.fixture
def codon_sequence(device):
    """Create sample codon sequence."""
    # Random codon indices (0-63)
    return torch.randint(0, 64, (4, 50), device=device)


@pytest.fixture
def timesteps(device):
    """Create sample timesteps."""
    return torch.randint(0, 1000, (4,), device=device)


@pytest.fixture
def backbone_coords(device):
    """Create sample backbone coordinates."""
    # (batch, n_residues, 3)
    return torch.randn(2, 30, 3, device=device)


@pytest.fixture
def continuous_data(device):
    """Create continuous data for noise scheduling."""
    return torch.randn(4, 32, device=device)
