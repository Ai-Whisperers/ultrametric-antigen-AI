# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Fixtures for physics module tests."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")


@pytest.fixture
def small_spin_glass(device):
    """Create small spin glass for testing."""
    from src.physics import SpinGlassLandscape
    return SpinGlassLandscape(n_sites=4, n_states=2, coupling_scale=0.5)


@pytest.fixture
def sample_configuration(device):
    """Create sample spin configuration."""
    return torch.randint(0, 2, (4,), device=device)


@pytest.fixture
def distance_matrix(device):
    """Create sample distance matrix."""
    n = 5
    D = torch.randn(n, n, device=device).abs()
    D = (D + D.T) / 2  # Symmetrize
    D.fill_diagonal_(0)  # Zero diagonal
    return D


@pytest.fixture
def spin_samples(device):
    """Create spin samples for overlap analysis."""
    return torch.randint(0, 2, (20, 10), device=device)
