# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for Reptile and FewShotAdapter classes."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.meta import Reptile, FewShotAdapter, Task


class TestReptileInit:
    """Tests for Reptile initialization."""

    def test_basic_init(self, simple_model):
        """Test basic initialization."""
        reptile = Reptile(simple_model, inner_lr=0.01, n_inner_steps=5)
        assert reptile.inner_lr == 0.01
        assert reptile.n_inner_steps == 5

    def test_meta_step_size(self, simple_model):
        """Test meta step size parameter."""
        reptile = Reptile(simple_model, meta_step_size=0.5)
        assert reptile.meta_step_size == 0.5


class TestReptileTrainOnTask:
    """Tests for Reptile train_on_task method."""

    def test_train_on_task_returns_dict(self, simple_model, sample_task, device):
        """Test train_on_task returns parameter differences."""
        reptile = Reptile(simple_model, inner_lr=0.01, n_inner_steps=1)
        diffs = reptile.train_on_task(sample_task)
        assert isinstance(diffs, dict)
        assert len(diffs) > 0

    def test_train_on_task_preserves_params(self, simple_model, sample_task, device):
        """Test train_on_task preserves original parameters."""
        reptile = Reptile(simple_model, inner_lr=0.1, n_inner_steps=3)

        # Get original parameters
        orig_params = {name: p.clone() for name, p in simple_model.named_parameters()}

        # Train on task
        _ = reptile.train_on_task(sample_task)

        # Check parameters are restored
        for name, param in simple_model.named_parameters():
            assert torch.allclose(orig_params[name], param)


class TestReptileMetaStep:
    """Tests for Reptile meta_step."""

    def test_meta_step_returns_metrics(self, simple_model, task_batch, device):
        """Test meta_step returns metrics."""
        reptile = Reptile(simple_model, inner_lr=0.01, n_inner_steps=2)
        metrics = reptile.meta_step(task_batch)
        assert isinstance(metrics, dict)

    def test_meta_step_updates_params(self, simple_model, task_batch, device):
        """Test meta_step updates model parameters."""
        reptile = Reptile(simple_model, inner_lr=0.01, n_inner_steps=2, meta_step_size=0.5)

        # Get original parameters
        orig_params = {name: p.clone() for name, p in simple_model.named_parameters()}

        # Meta step
        _ = reptile.meta_step(task_batch)

        # Check parameters changed
        params_changed = False
        for name, param in simple_model.named_parameters():
            if not torch.allclose(orig_params[name], param, atol=1e-6):
                params_changed = True
                break
        assert params_changed


class TestFewShotAdapterInit:
    """Tests for FewShotAdapter initialization."""

    def test_basic_init(self, simple_model):
        """Test basic initialization."""
        # FewShotAdapter takes encoder, not base_model
        adapter = FewShotAdapter(
            encoder=simple_model,
            prototype_dim=16,
            n_adapt_steps=3,
        )
        assert adapter.prototype_dim == 16
        assert adapter.n_adapt_steps == 3

    def test_adapt_lr(self, simple_model):
        """Test adaptation learning rate."""
        adapter = FewShotAdapter(
            encoder=simple_model,
            prototype_dim=16,
            adapt_lr=0.2,
        )
        assert adapter.adapt_lr == 0.2


class TestFewShotAdapterPredict:
    """Tests for FewShotAdapter prediction."""

    def test_adapt_and_predict(self, simple_model, device):
        """Test adapt_and_predict method."""
        adapter = FewShotAdapter(
            encoder=simple_model,
            prototype_dim=3,  # Match model output
            n_adapt_steps=1,
        )

        # Create task with labels in valid range for prototype_dim
        task = Task(
            support_x=torch.randn(5, 8, device=device),
            support_y=torch.zeros(5, device=device).long(),  # All same class to avoid issues
            query_x=torch.randn(10, 8, device=device),
            query_y=torch.zeros(10, device=device).long(),
        )

        predictions = adapter.adapt_and_predict(task)
        assert predictions.shape[0] == task.n_query

    def test_forward_shape(self, simple_model, device):
        """Test forward returns correct shape."""
        adapter = FewShotAdapter(
            encoder=simple_model,
            prototype_dim=3,
            n_adapt_steps=1,
        )

        # Create task
        task = Task(
            support_x=torch.randn(5, 8, device=device),
            support_y=torch.randint(0, 3, (5,), device=device),
            query_x=torch.randn(10, 8, device=device),
            query_y=torch.randint(0, 3, (10,), device=device),
        )

        predictions = adapter.adapt_and_predict(task)
        assert predictions.shape == (10,) or predictions.shape == (10, 3)
