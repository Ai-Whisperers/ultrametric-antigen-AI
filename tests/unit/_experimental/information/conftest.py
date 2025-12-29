# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Fixtures for information geometry tests."""

from __future__ import annotations

from typing import Iterator, Tuple

import pytest
import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
    """Simple classifier for testing."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, num_classes: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class SimpleMLP(nn.Module):
    """Simple MLP for testing K-FAC."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 16, num_classes: int = 3):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        return self.layer2(x)


def create_data_loader(
    n_samples: int = 50,
    input_dim: int = 10,
    num_classes: int = 5,
    batch_size: int = 10,
    device: str = "cpu",
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Create a simple data loader for testing."""
    n_batches = n_samples // batch_size

    for _ in range(n_batches):
        inputs = torch.randn(batch_size, input_dim, device=device)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)
        yield inputs, targets


@pytest.fixture
def device():
    """Get device for testing. Force CPU for unit tests to avoid CUDA state issues."""
    return torch.device("cpu")


@pytest.fixture
def simple_classifier(device):
    """Create a simple classifier model."""
    model = SimpleClassifier(input_dim=10, hidden_dim=20, num_classes=5)
    return model.to(device)


@pytest.fixture
def simple_mlp(device):
    """Create a simple MLP model."""
    model = SimpleMLP(input_dim=10, hidden_dim=16, num_classes=3)
    return model.to(device)


@pytest.fixture
def data_loader(device):
    """Create a data loader fixture."""
    return list(create_data_loader(n_samples=50, device=device))


@pytest.fixture
def small_data_loader(device):
    """Create a small data loader for quick tests.

    Uses num_classes=3 to match SimpleMLP.
    """
    return list(create_data_loader(n_samples=20, batch_size=5, num_classes=3, device=device))


@pytest.fixture
def fisher_matrix():
    """Create a sample Fisher matrix."""
    # Create positive definite matrix
    A = torch.randn(10, 10)
    F = A @ A.T + 0.1 * torch.eye(10)
    return F
