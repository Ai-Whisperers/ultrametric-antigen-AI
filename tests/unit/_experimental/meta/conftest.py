# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Fixtures for meta-learning tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.meta import Task


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")


@pytest.fixture
def simple_model():
    """Create a simple model for meta-learning."""
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )


@pytest.fixture
def sample_task(device):
    """Create a sample task."""
    return Task(
        support_x=torch.randn(5, 8, device=device),
        support_y=torch.randint(0, 3, (5,), device=device),
        query_x=torch.randn(10, 8, device=device),
        query_y=torch.randint(0, 3, (10,), device=device),
        task_id=0,
    )


@pytest.fixture
def task_batch(device):
    """Create a batch of tasks."""
    tasks = []
    for i in range(4):
        task = Task(
            support_x=torch.randn(5, 8, device=device),
            support_y=torch.randint(0, 3, (5,), device=device),
            query_x=torch.randn(10, 8, device=device),
            query_y=torch.randint(0, 3, (10,), device=device),
            task_id=i,
        )
        tasks.append(task)
    return tasks
