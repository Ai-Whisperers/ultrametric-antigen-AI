# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit test fixtures.

Fixtures specific to unit tests. These are lighter weight and focus on
testing individual components in isolation.
"""

from unittest.mock import MagicMock

import pytest
import torch

# =============================================================================
# Lightweight Mock Fixtures for Unit Tests
# =============================================================================


@pytest.fixture
def mock_encoder():
    """Returns a mock encoder module."""
    encoder = MagicMock()
    encoder.return_value = (
        torch.randn(4, 16),  # mu
        torch.randn(4, 16),  # logvar
    )
    return encoder


@pytest.fixture
def mock_decoder():
    """Returns a mock decoder module."""
    decoder = MagicMock()
    decoder.return_value = torch.randn(4, 9, 3)  # logits
    return decoder


@pytest.fixture
def mock_projection():
    """Returns a mock projection module."""
    projection = MagicMock()
    # Returns point inside Poincaré disk
    projection.return_value = torch.randn(4, 16) * 0.5
    return projection


@pytest.fixture
def isolated_test(monkeypatch):
    """Fixture that ensures test isolation by preventing external calls."""
    # Prevent any accidental file writes
    monkeypatch.setattr("builtins.open", MagicMock(side_effect=PermissionError))
    yield


# =============================================================================
# Deterministic Fixtures for Reproducible Tests
# =============================================================================


@pytest.fixture
def deterministic_ternary_ops(device):
    """Returns deterministic ternary operations for reproducible tests."""
    torch.manual_seed(42)
    ops = torch.randint(-1, 2, (32, 9), device=device).float()
    return ops


@pytest.fixture
def known_ternary_ops(device):
    """Returns known specific ternary operations for exact testing."""
    return torch.tensor(
        [
            [-1, -1, -1, -1, -1, -1, -1, -1, -1],  # Index 0
            [0, -1, -1, -1, -1, -1, -1, -1, -1],  # Index 1
            [1, -1, -1, -1, -1, -1, -1, -1, -1],  # Index 2
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Index 9841 (middle)
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # Index 19682 (max)
        ],
        device=device,
        dtype=torch.float32,
    )


@pytest.fixture
def known_indices():
    """Returns the indices corresponding to known_ternary_ops."""
    return torch.tensor([0, 1, 2, 9841, 19682])
