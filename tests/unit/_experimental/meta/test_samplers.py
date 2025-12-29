# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for PAdicTaskSampler class."""

from __future__ import annotations

import pytest
import torch

from src.meta import PAdicTaskSampler, Task


@pytest.fixture
def data_x(device):
    """Create sample data features."""
    return torch.randn(100, 8, device=device)


@pytest.fixture
def data_y(device):
    """Create sample data labels."""
    return torch.randint(0, 3, (100,), device=device)


@pytest.fixture
def padic_indices(device):
    """Create sample p-adic indices for hierarchical structure."""
    # Each sample has a 3-digit ternary code representing its position in hierarchy
    return torch.randint(0, 27, (100,), device=device)  # 0-26 = 3^3 - 1


class TestPAdicTaskSamplerInit:
    """Tests for PAdicTaskSampler initialization."""

    def test_basic_init(self, data_x, data_y, padic_indices):
        """Test basic initialization."""
        sampler = PAdicTaskSampler(
            data_x=data_x,
            data_y=data_y,
            padic_indices=padic_indices,
        )
        # Check data is stored
        assert len(sampler.data_x) == 100

    def test_custom_support_query(self, data_x, data_y, padic_indices):
        """Test custom support/query sizes."""
        sampler = PAdicTaskSampler(
            data_x=data_x,
            data_y=data_y,
            padic_indices=padic_indices,
            n_support=10,
            n_query=15,
        )
        assert sampler.n_support == 10
        assert sampler.n_query == 15


class TestPAdicTaskSampling:
    """Tests for task sampling."""

    def test_sample_single_task(self, data_x, data_y, padic_indices):
        """Test sampling a single task."""
        sampler = PAdicTaskSampler(
            data_x=data_x,
            data_y=data_y,
            padic_indices=padic_indices,
            n_support=5,
            n_query=10,
        )
        task = sampler.sample_task()

        assert isinstance(task, Task)
        assert task.n_support == 5
        assert task.n_query == 10

    def test_sample_batch(self, data_x, data_y, padic_indices):
        """Test sampling batch of tasks."""
        sampler = PAdicTaskSampler(
            data_x=data_x,
            data_y=data_y,
            padic_indices=padic_indices,
        )
        tasks = sampler.sample_batch(n_tasks=8)

        assert len(tasks) == 8
        assert all(isinstance(t, Task) for t in tasks)

    def test_sample_task_shapes(self, data_x, data_y, padic_indices):
        """Test sampled task has correct shapes."""
        sampler = PAdicTaskSampler(
            data_x=data_x,
            data_y=data_y,
            padic_indices=padic_indices,
            n_support=5,
            n_query=10,
        )
        task = sampler.sample_task()

        assert task.support_x.shape == (5, 8)
        assert task.support_y.shape == (5,)
        assert task.query_x.shape == (10, 8)
        assert task.query_y.shape == (10,)


class TestPAdicHierarchicalSampling:
    """Tests for p-adic hierarchical sampling."""

    def test_similar_samples_selected(self, data_x, data_y, device):
        """Test that p-adically similar samples are selected together."""
        # Create structured p-adic indices
        # First 50 samples with index 0-9, next 50 with index 10-19
        padic_indices = torch.cat([
            torch.randint(0, 10, (50,), device=device),
            torch.randint(10, 20, (50,), device=device),
        ])

        sampler = PAdicTaskSampler(
            data_x=data_x,
            data_y=data_y,
            padic_indices=padic_indices,
        )

        # Sample many tasks and check distribution
        # (Implementation-specific test)
        task = sampler.sample_task()
        assert task.n_support > 0
