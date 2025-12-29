# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for ParametricLens and LinearLens classes."""

from __future__ import annotations

import pytest
import torch

from src.categorical import LinearLens, ProductCategory, MonoidalCategory, TensorType, CategoricalLayer


class TestLinearLens:
    """Tests for LinearLens class."""

    def test_linear_lens_creation(self):
        """Test linear lens creation."""
        lens = LinearLens(in_features=4, out_features=8)
        assert lens.input_type.shape == (4,)
        assert lens.output_type.shape == (8,)

    def test_linear_lens_forward(self):
        """Test linear lens forward pass."""
        lens = LinearLens(in_features=4, out_features=8)
        x = torch.randn(5, 4)
        y = lens(x)
        assert y.shape == (5, 8)

    def test_linear_lens_params(self):
        """Test linear lens has parameters."""
        lens = LinearLens(in_features=4, out_features=8)
        # Should have weight and bias combined
        assert lens.params.shape == (8, 5)  # out x (in + 1)

    def test_linear_lens_trainable(self):
        """Test linear lens parameters are trainable."""
        lens = LinearLens(in_features=4, out_features=8)
        x = torch.randn(5, 4)
        y = lens(x)
        loss = y.sum()
        loss.backward()

        assert lens.params.grad is not None


class TestProductCategory:
    """Tests for ProductCategory class."""

    def test_product_type(self):
        """Test product type creation."""
        t1 = TensorType((4,))
        t2 = TensorType((8,))
        product = ProductCategory.product_type(t1, t2)
        assert product.shape == (12,)

    def test_product_type_2d(self):
        """Test product type with 2D shapes."""
        t1 = TensorType((4, 4))
        t2 = TensorType((4, 8))
        product = ProductCategory.product_type(t1, t2)
        assert product.shape == (4, 12)


class TestMonoidalCategory:
    """Tests for MonoidalCategory class."""

    def test_tensor_product(self):
        """Test tensor product of layers."""
        t1 = TensorType((4,))
        t2 = TensorType((8,))
        t3 = TensorType((2,))
        t4 = TensorType((3,))

        layer1 = CategoricalLayer(t1, t2, name="l1")
        layer2 = CategoricalLayer(t3, t4, name="l2")

        parallel = MonoidalCategory.tensor_product(layer1, layer2)

        # Input: (4 + 2) = 6
        # Output: (8 + 3) = 11
        x = torch.randn(5, 6)
        y = parallel(x)
        assert y.shape == (5, 11)

    def test_unit_object(self):
        """Test unit object."""
        unit = MonoidalCategory.unit_object()
        assert unit.shape == ()
