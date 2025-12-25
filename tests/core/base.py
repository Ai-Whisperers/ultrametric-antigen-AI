# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Base test classes providing common test patterns.

This module provides abstract base classes and mixins for testing:
- LossTestCase: Standard tests for loss functions
- ModelTestCase: Standard tests for neural network modules
- GeometryTestCase: Standard tests for geometric operations
- TrainingTestCase: Standard tests for training components

Usage:
    class TestMyLoss(LossTestCase):
        loss_class = MyLoss
        loss_kwargs = {'margin': 0.1}

        def test_custom_behavior(self):
            ...
"""

import pytest
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional, List
from unittest.mock import MagicMock

from .assertions import (
    TensorAssertions,
    GeometryAssertions,
    ModelAssertions,
    LossAssertions,
    TernaryAssertions,
)


class BaseTestCase(ABC):
    """Abstract base class for all test cases.

    Provides common utilities and setup patterns.
    """

    @pytest.fixture(autouse=True)
    def setup_base(self, device):
        """Common setup for all tests."""
        self.device = device

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to test device."""
        return tensor.to(self.device)

    def _create_random_tensor(self, *shape, requires_grad: bool = False) -> torch.Tensor:
        """Create random tensor on test device."""
        return torch.randn(*shape, device=self.device, requires_grad=requires_grad)

    def _create_ternary_batch(self, batch_size: int = 32) -> torch.Tensor:
        """Create random ternary operations batch."""
        return torch.randint(-1, 2, (batch_size, 9), device=self.device).float()


class LossTestCase(BaseTestCase):
    """Base class for testing loss functions.

    Subclasses should define:
        loss_class: The loss class to test
        loss_kwargs: Default kwargs for loss initialization

    Provides standard tests for:
        - Initialization
        - Forward pass shape
        - Gradient flow
        - Numerical stability
        - Batch size invariance
    """

    loss_class: Type[nn.Module] = None
    loss_kwargs: Dict[str, Any] = {}

    @pytest.fixture
    def loss_fn(self):
        """Create loss function instance."""
        if self.loss_class is None:
            pytest.skip("loss_class not defined")
        return self.loss_class(**self.loss_kwargs).to(self.device)

    def test_initialization(self, loss_fn):
        """Test loss function initializes correctly."""
        assert isinstance(loss_fn, nn.Module)

    def test_forward_returns_tensor(self, loss_fn):
        """Test forward pass returns tensor."""
        inputs = self._get_test_inputs()
        result = loss_fn(*inputs)

        if isinstance(result, tuple):
            loss = result[0]
        else:
            loss = result

        assert isinstance(loss, torch.Tensor)

    def test_forward_returns_scalar(self, loss_fn):
        """Test forward pass returns scalar loss."""
        inputs = self._get_test_inputs()
        result = loss_fn(*inputs)

        if isinstance(result, tuple):
            loss = result[0]
        else:
            loss = result

        assert loss.dim() == 0 or loss.numel() == 1

    def test_loss_is_finite(self, loss_fn):
        """Test loss value is finite."""
        inputs = self._get_test_inputs()
        result = loss_fn(*inputs)

        if isinstance(result, tuple):
            loss = result[0]
        else:
            loss = result

        TensorAssertions.assert_finite(loss, "Loss should be finite")

    def test_loss_is_non_negative(self, loss_fn):
        """Test loss value is non-negative (if applicable)."""
        inputs = self._get_test_inputs()
        result = loss_fn(*inputs)

        if isinstance(result, tuple):
            loss = result[0]
        else:
            loss = result

        # Most losses should be non-negative
        # Override in subclass if loss can be negative
        TensorAssertions.assert_non_negative(loss, "Loss should be non-negative")

    def test_gradient_flow(self, loss_fn):
        """Test gradients flow through loss."""
        inputs = self._get_test_inputs(requires_grad=True)
        result = loss_fn(*inputs)

        if isinstance(result, tuple):
            loss = result[0]
        else:
            loss = result

        loss.backward()

        # Check at least one input has gradients
        has_grad = any(
            inp.grad is not None and inp.grad.abs().sum() > 0
            for inp in inputs if isinstance(inp, torch.Tensor) and inp.requires_grad
        )
        assert has_grad, "No gradients found in any input"

    def test_batch_size_flexibility(self, loss_fn):
        """Test loss works with different batch sizes."""
        for batch_size in [1, 4, 32]:
            inputs = self._get_test_inputs(batch_size=batch_size)
            result = loss_fn(*inputs)

            if isinstance(result, tuple):
                loss = result[0]
            else:
                loss = result

            TensorAssertions.assert_finite(loss, f"Loss should be finite for batch_size={batch_size}")

    @abstractmethod
    def _get_test_inputs(self, batch_size: int = 32, requires_grad: bool = False):
        """Return test inputs for the loss function.

        Must be implemented by subclasses.
        """
        pass


class ModelTestCase(BaseTestCase):
    """Base class for testing neural network modules.

    Subclasses should define:
        model_class: The model class to test
        model_kwargs: Default kwargs for model initialization
        input_shape: Expected input shape (without batch dimension)
        expected_output_keys: Keys expected in model output dict

    Provides standard tests for:
        - Initialization
        - Forward pass
        - Output shapes
        - Gradient flow
        - Parameter counting
    """

    model_class: Type[nn.Module] = None
    model_kwargs: Dict[str, Any] = {}
    input_shape: tuple = (9,)  # Default for ternary operations
    expected_output_keys: List[str] = []

    @pytest.fixture
    def model(self):
        """Create model instance."""
        if self.model_class is None:
            pytest.skip("model_class not defined")
        return self.model_class(**self.model_kwargs).to(self.device)

    def test_initialization(self, model):
        """Test model initializes correctly."""
        assert isinstance(model, nn.Module)

    def test_forward_pass(self, model):
        """Test forward pass completes without error."""
        batch_size = 4
        x = self._create_test_input(batch_size)
        output = model(x)
        assert output is not None

    def test_output_keys(self, model):
        """Test forward pass returns expected keys."""
        if not self.expected_output_keys:
            pytest.skip("expected_output_keys not defined")

        batch_size = 4
        x = self._create_test_input(batch_size)
        output = model(x)

        if isinstance(output, dict):
            ModelAssertions.assert_output_keys(output, self.expected_output_keys)

    def test_gradient_flow(self, model):
        """Test gradients reach trainable parameters."""
        batch_size = 4
        x = self._create_test_input(batch_size)

        output = model(x)

        # Get a loss from output
        if isinstance(output, dict):
            # Use first tensor value
            for v in output.values():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    loss = v.mean()
                    break
            else:
                loss = list(output.values())[0]
                if isinstance(loss, torch.Tensor):
                    loss = loss.mean()
                else:
                    pytest.skip("No tensor output found")
        else:
            loss = output.mean() if isinstance(output, torch.Tensor) else output[0].mean()

        loss.backward()

        # Check at least some parameters have gradients
        has_grad = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break

        assert has_grad, "No gradients found in model parameters"

    def test_parameter_count(self, model):
        """Test model has trainable parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total_params > 0, "Model has no parameters"
        assert trainable_params > 0, "Model has no trainable parameters"

    def test_eval_mode(self, model):
        """Test model works in eval mode."""
        model.eval()
        batch_size = 4
        x = self._create_test_input(batch_size)

        with torch.no_grad():
            output = model(x)

        assert output is not None

    def test_train_mode(self, model):
        """Test model works in train mode."""
        model.train()
        batch_size = 4
        x = self._create_test_input(batch_size)
        output = model(x)
        assert output is not None

    def _create_test_input(self, batch_size: int) -> torch.Tensor:
        """Create test input for model."""
        # For ternary operations
        if self.input_shape == (9,):
            return self._create_ternary_batch(batch_size)
        else:
            return self._create_random_tensor(batch_size, *self.input_shape)


class GeometryTestCase(BaseTestCase):
    """Base class for testing geometric operations.

    Provides standard tests for:
        - Distance properties (symmetry, identity, triangle inequality)
        - Poincaré disk constraints
        - Numerical stability
    """

    @pytest.fixture
    def manifold(self, poincare):
        """Get Poincaré manifold."""
        return poincare

    def _create_poincare_points(self, n_points: int, dim: int = 16) -> torch.Tensor:
        """Create random points on Poincaré disk."""
        points = torch.randn(n_points, dim, device=self.device)
        norms = torch.norm(points, dim=-1, keepdim=True)
        # Scale to be inside disk with max radius 0.9
        points = points / (norms + 1e-8) * 0.9 * torch.rand(n_points, 1, device=self.device)
        return points

    def test_distance_symmetry(self, manifold):
        """Test distance is symmetric."""
        a = self._create_poincare_points(10)
        b = self._create_poincare_points(10)

        dist_ab = manifold.dist(a, b)
        dist_ba = manifold.dist(b, a)

        GeometryAssertions.assert_distance_symmetry(dist_ab, dist_ba)

    def test_distance_identity(self, manifold):
        """Test distance to self is zero."""
        a = self._create_poincare_points(10)
        dist_aa = manifold.dist(a, a)

        GeometryAssertions.assert_distance_identity(dist_aa)

    def test_distance_non_negative(self, manifold):
        """Test distance is non-negative."""
        a = self._create_poincare_points(10)
        b = self._create_poincare_points(10)

        dist = manifold.dist(a, b)
        TensorAssertions.assert_non_negative(dist)

    def test_triangle_inequality(self, manifold):
        """Test triangle inequality holds."""
        a = self._create_poincare_points(10)
        b = self._create_poincare_points(10)
        c = self._create_poincare_points(10)

        dist_ab = manifold.dist(a, b)
        dist_bc = manifold.dist(b, c)
        dist_ac = manifold.dist(a, c)

        GeometryAssertions.assert_triangle_inequality(dist_ab, dist_bc, dist_ac)


class ComponentTestCase(BaseTestCase):
    """Base class for testing LossComponent implementations.

    Subclasses should define:
        component_class: The component class to test
        component_kwargs: Default kwargs for initialization

    Provides tests for LossComponent interface compliance.
    """

    component_class: Type = None
    component_kwargs: Dict[str, Any] = {}

    @pytest.fixture
    def component(self):
        """Create component instance."""
        if self.component_class is None:
            pytest.skip("component_class not defined")
        return self.component_class(**self.component_kwargs)

    @pytest.fixture
    def mock_outputs(self, device) -> Dict[str, torch.Tensor]:
        """Create mock model outputs."""
        batch_size = 32
        latent_dim = 16

        return {
            'logits_A': torch.randn(batch_size, 9, 3, device=device),
            'logits_B': torch.randn(batch_size, 9, 3, device=device),
            'mu_A': torch.randn(batch_size, latent_dim, device=device),
            'mu_B': torch.randn(batch_size, latent_dim, device=device),
            'logvar_A': torch.randn(batch_size, latent_dim, device=device),
            'logvar_B': torch.randn(batch_size, latent_dim, device=device),
            'z_A': torch.randn(batch_size, latent_dim, device=device) * 0.5,
            'z_B': torch.randn(batch_size, latent_dim, device=device) * 0.5,
            'H_A': torch.tensor(2.0, device=device),
            'H_B': torch.tensor(2.1, device=device),
            'beta_A': 0.1,
            'beta_B': 0.1,
        }

    def test_initialization(self, component):
        """Test component initializes correctly."""
        assert hasattr(component, 'weight')
        assert hasattr(component, 'name')

    def test_has_forward(self, component):
        """Test component has forward method."""
        assert hasattr(component, 'forward')
        assert callable(component.forward)

    def test_forward_returns_loss_result(self, component, mock_outputs, device):
        """Test forward returns LossResult."""
        from src.losses.base import LossResult

        targets = torch.randint(-1, 2, (32, 9), device=device).float()
        result = component.forward(mock_outputs, targets)

        assert isinstance(result, LossResult)
        assert hasattr(result, 'loss')
        assert hasattr(result, 'metrics')
        assert hasattr(result, 'weight')


class SchedulerTestCase(BaseTestCase):
    """Base class for testing learning rate and parameter schedulers.

    Subclasses should define:
        scheduler_class: The scheduler class to test
        scheduler_kwargs: Default kwargs for initialization
    """

    scheduler_class: Type = None
    scheduler_kwargs: Dict[str, Any] = {}

    @pytest.fixture
    def scheduler(self):
        """Create scheduler instance."""
        if self.scheduler_class is None:
            pytest.skip("scheduler_class not defined")
        return self.scheduler_class(**self.scheduler_kwargs)

    def test_initialization(self, scheduler):
        """Test scheduler initializes correctly."""
        assert scheduler is not None

    def test_returns_value(self, scheduler):
        """Test scheduler returns a value."""
        value = scheduler(0)  # Epoch 0
        assert value is not None

    def test_value_changes(self, scheduler):
        """Test scheduler value changes over epochs."""
        value_0 = scheduler(0)
        value_50 = scheduler(50)
        value_100 = scheduler(100)

        # At least one pair should be different
        values = [value_0, value_50, value_100]
        assert len(set(values)) > 1 or self._is_constant_schedule()

    def _is_constant_schedule(self) -> bool:
        """Override to return True if schedule is constant."""
        return False
