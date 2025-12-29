# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Fixtures for category theory tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.categorical import TensorType, CategoricalLayer


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")


@pytest.fixture
def small_type():
    """Create small tensor type."""
    return TensorType((4,))


@pytest.fixture
def medium_type():
    """Create medium tensor type."""
    return TensorType((8,))


@pytest.fixture
def large_type():
    """Create large tensor type."""
    return TensorType((16,))


@pytest.fixture
def simple_layer(small_type, medium_type):
    """Create simple categorical layer."""
    return CategoricalLayer(
        input_type=small_type,
        output_type=medium_type,
        name="simple"
    )


@pytest.fixture
def sample_batch(small_type, device):
    """Create sample batch tensor."""
    return torch.randn(5, small_type.shape[0], device=device)
