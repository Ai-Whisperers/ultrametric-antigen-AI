# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Fixtures for tropical geometry tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn


@pytest.fixture
def simple_relu_network():
    """Create a simple 2-layer ReLU network."""
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    )
    return model


@pytest.fixture
def deep_relu_network():
    """Create a deeper ReLU network."""
    model = nn.Sequential(
        nn.Linear(2, 3),
        nn.ReLU(),
        nn.Linear(3, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
    )
    return model


@pytest.fixture
def sample_points():
    """Create sample points for tropical convex hull."""
    return np.array([
        [0.0, 0.0],
        [1.0, 2.0],
        [2.0, 1.0],
        [1.5, 1.5],
    ])


@pytest.fixture
def sample_tree_edges():
    """Create sample edge lengths for a tree."""
    return np.array([0.5, 0.3, 0.7, 0.2])
