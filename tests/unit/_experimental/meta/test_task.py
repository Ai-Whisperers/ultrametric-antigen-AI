# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for Task dataclass."""

from __future__ import annotations

import pytest
import torch

from src.meta import Task


class TestTaskCreation:
    """Tests for Task creation."""

    def test_basic_creation(self, device):
        """Test basic task creation."""
        task = Task(
            support_x=torch.randn(5, 8, device=device),
            support_y=torch.randint(0, 3, (5,), device=device),
            query_x=torch.randn(10, 8, device=device),
            query_y=torch.randint(0, 3, (10,), device=device),
        )
        assert task.n_support == 5
        assert task.n_query == 10

    def test_with_task_id(self, device):
        """Test task with task_id."""
        task = Task(
            support_x=torch.randn(5, 8, device=device),
            support_y=torch.randint(0, 3, (5,), device=device),
            query_x=torch.randn(10, 8, device=device),
            query_y=torch.randint(0, 3, (10,), device=device),
            task_id=42,
        )
        assert task.task_id == 42

    def test_with_metadata(self, device):
        """Test task with metadata."""
        task = Task(
            support_x=torch.randn(5, 8, device=device),
            support_y=torch.randint(0, 3, (5,), device=device),
            query_x=torch.randn(10, 8, device=device),
            query_y=torch.randint(0, 3, (10,), device=device),
            metadata={"disease": "HIV", "level": 3},
        )
        assert task.metadata["disease"] == "HIV"
        assert task.metadata["level"] == 3


class TestTaskProperties:
    """Tests for Task properties."""

    def test_n_support(self, sample_task):
        """Test n_support property."""
        assert sample_task.n_support == 5

    def test_n_query(self, sample_task):
        """Test n_query property."""
        assert sample_task.n_query == 10


class TestTaskDevice:
    """Tests for Task device handling."""

    def test_to_device(self, device):
        """Test moving task to device."""
        task = Task(
            support_x=torch.randn(5, 8),
            support_y=torch.randint(0, 3, (5,)),
            query_x=torch.randn(10, 8),
            query_y=torch.randint(0, 3, (10,)),
        )
        task_on_device = task.to(device)

        assert task_on_device.support_x.device == device
        assert task_on_device.support_y.device == device
        assert task_on_device.query_x.device == device
        assert task_on_device.query_y.device == device

    def test_to_preserves_metadata(self, device):
        """Test that to() preserves metadata."""
        task = Task(
            support_x=torch.randn(5, 8),
            support_y=torch.randint(0, 3, (5,)),
            query_x=torch.randn(10, 8),
            query_y=torch.randint(0, 3, (10,)),
            task_id=7,
            metadata={"key": "value"},
        )
        task_on_device = task.to(device)

        assert task_on_device.task_id == 7
        assert task_on_device.metadata["key"] == "value"
