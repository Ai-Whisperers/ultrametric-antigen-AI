# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for NaturalGradientOptimizer class."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.information import NaturalGradientOptimizer


class TestNaturalGradientInit:
    """Tests for NaturalGradientOptimizer initialization."""

    def test_default_init(self, simple_classifier):
        """Test default initialization."""
        optimizer = NaturalGradientOptimizer(simple_classifier.parameters())
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert optimizer.param_groups[0]["damping"] == 1e-4

    def test_custom_lr(self, simple_classifier):
        """Test custom learning rate."""
        optimizer = NaturalGradientOptimizer(simple_classifier.parameters(), lr=0.1)
        assert optimizer.param_groups[0]["lr"] == 0.1

    def test_custom_damping(self, simple_classifier):
        """Test custom damping."""
        optimizer = NaturalGradientOptimizer(
            simple_classifier.parameters(), damping=1e-3
        )
        assert optimizer.param_groups[0]["damping"] == 1e-3

    def test_custom_ema_decay(self, simple_classifier):
        """Test custom EMA decay."""
        optimizer = NaturalGradientOptimizer(
            simple_classifier.parameters(), cov_ema_decay=0.9
        )
        assert optimizer.param_groups[0]["cov_ema_decay"] == 0.9


class TestNaturalGradientStep:
    """Tests for optimization step."""

    def test_step_updates_params(self, simple_classifier, small_data_loader, device):
        """Test step updates parameters."""
        model = simple_classifier.to(device)
        optimizer = NaturalGradientOptimizer(model.parameters(), lr=0.01)

        # Get initial params
        initial_params = {n: p.clone() for n, p in model.named_parameters()}

        # Do one step
        inputs, targets = small_data_loader[0]
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        # Check params changed
        for name, param in model.named_parameters():
            assert not torch.allclose(param, initial_params[name]), f"{name} not updated"

    def test_step_with_closure(self, simple_classifier, small_data_loader, device):
        """Test step with closure."""
        model = simple_classifier.to(device)
        optimizer = NaturalGradientOptimizer(model.parameters(), lr=0.01)
        inputs, targets = small_data_loader[0]

        def closure():
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        assert loss is not None
        assert loss.item() > 0

    def test_state_initialized(self, simple_classifier, small_data_loader, device):
        """Test optimizer state is initialized after step."""
        model = simple_classifier.to(device)
        optimizer = NaturalGradientOptimizer(model.parameters())

        inputs, targets = small_data_loader[0]
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        # Check state for each param
        for p in model.parameters():
            assert p in optimizer.state
            assert "step" in optimizer.state[p]
            assert "cov" in optimizer.state[p]

    def test_step_count_increments(self, simple_classifier, small_data_loader, device):
        """Test step count increments."""
        model = simple_classifier.to(device)
        optimizer = NaturalGradientOptimizer(model.parameters())

        for i in range(3):
            inputs, targets = small_data_loader[i % len(small_data_loader)]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

        # Check step count
        for p in model.parameters():
            assert optimizer.state[p]["step"] == 3


class TestNaturalGradientCovarianceUpdate:
    """Tests for covariance update mechanism."""

    def test_cov_updated(self, simple_classifier, small_data_loader, device):
        """Test covariance is updated."""
        model = simple_classifier.to(device)
        optimizer = NaturalGradientOptimizer(model.parameters())

        inputs, targets = small_data_loader[0]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        # Covariance should not be all ones anymore
        for p in model.parameters():
            cov = optimizer.state[p]["cov"]
            assert not torch.allclose(cov, torch.ones_like(cov))

    def test_cov_positive(self, simple_classifier, small_data_loader, device):
        """Test covariance stays positive."""
        model = simple_classifier.to(device)
        optimizer = NaturalGradientOptimizer(model.parameters())

        for i in range(5):
            inputs, targets = small_data_loader[i % len(small_data_loader)]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

        for p in model.parameters():
            assert (optimizer.state[p]["cov"] > 0).all()


class TestNaturalGradientWeightDecay:
    """Tests for weight decay."""

    def test_weight_decay_shrinks_params(self, simple_classifier, device):
        """Test weight decay shrinks parameters.

        With lr > 0 and weight_decay > 0, parameters should shrink.
        """
        model = simple_classifier.to(device)

        # Non-zero lr with weight decay
        optimizer = NaturalGradientOptimizer(
            model.parameters(), lr=0.1, weight_decay=0.5
        )

        # Get initial norm
        initial_norm = sum(p.norm() ** 2 for p in model.parameters()).sqrt().item()

        # Create near-zero gradients to isolate weight decay effect
        for p in model.parameters():
            p.grad = torch.zeros_like(p)

        # Multiple steps to accumulate weight decay effect
        for _ in range(5):
            optimizer.step()

        # Norm should decrease
        final_norm = sum(p.norm() ** 2 for p in model.parameters()).sqrt().item()
        assert final_norm < initial_norm


class TestNaturalGradientSparseGradient:
    """Tests for sparse gradient handling."""

    def test_sparse_gradient_raises(self, simple_classifier, device):
        """Test sparse gradients raise error."""
        model = simple_classifier.to(device)
        optimizer = NaturalGradientOptimizer(model.parameters())

        # Create sparse gradient
        for p in model.parameters():
            p.grad = p.clone().to_sparse()
            break

        with pytest.raises(RuntimeError, match="sparse"):
            optimizer.step()
