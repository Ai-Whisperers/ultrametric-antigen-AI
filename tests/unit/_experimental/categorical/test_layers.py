# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for CategoricalLayer and Morphism classes."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.categorical import TensorType, Morphism, CategoricalLayer


class TestMorphism:
    """Tests for Morphism class."""

    def test_morphism_creation(self):
        """Test morphism creation."""
        source = TensorType((4,))
        target = TensorType((8,))
        m = Morphism(source=source, target=target, name="test")
        assert m.source == source
        assert m.target == target
        assert m.name == "test"

    def test_morphism_call(self):
        """Test morphism call."""
        source = TensorType((4,))
        target = TensorType((4,))
        m = Morphism(
            source=source,
            target=target,
            name="double",
            transform=lambda x: x * 2,
        )
        x = torch.randn(5, 4)
        result = m(x)
        assert torch.allclose(result, x * 2)

    def test_morphism_no_transform_raises(self):
        """Test morphism without transform raises on call."""
        m = Morphism(source=TensorType((4,)), target=TensorType((8,)))
        with pytest.raises(ValueError):
            m(torch.randn(5, 4))

    def test_morphism_compose(self):
        """Test morphism composition."""
        t1 = TensorType((4,))
        t2 = TensorType((8,))
        t3 = TensorType((16,))

        m1 = Morphism(t1, t2, "m1", lambda x: torch.cat([x, x], dim=-1))
        m2 = Morphism(t2, t3, "m2", lambda x: torch.cat([x, x], dim=-1))

        composed = m1.compose(m2)
        assert composed.source == t1
        assert composed.target == t3

        x = torch.randn(5, 4)
        result = composed(x)
        assert result.shape == (5, 16)

    def test_morphism_compose_type_mismatch(self):
        """Test composition fails on type mismatch."""
        t1 = TensorType((4,))
        t2 = TensorType((8,))
        t3 = TensorType((16,))

        m1 = Morphism(t1, t2, "m1")
        m2 = Morphism(t3, t3, "m2")  # Source doesn't match m1's target

        with pytest.raises(TypeError):
            m1.compose(m2)


class TestCategoricalLayer:
    """Tests for CategoricalLayer class."""

    def test_layer_creation(self, small_type, medium_type):
        """Test layer creation."""
        layer = CategoricalLayer(
            input_type=small_type,
            output_type=medium_type,
            name="test"
        )
        assert layer.input_type == small_type
        assert layer.output_type == medium_type

    def test_layer_forward(self, small_type, medium_type, device):
        """Test layer forward pass."""
        layer = CategoricalLayer(small_type, medium_type)
        x = torch.randn(5, small_type.shape[0], device=device)
        y = layer(x)
        assert y.shape == (5, medium_type.shape[0])

    def test_layer_type_check_input(self, small_type, medium_type, device):
        """Test layer type checks input."""
        layer = CategoricalLayer(small_type, medium_type)
        # Wrong input size
        x = torch.randn(5, 16, device=device)
        with pytest.raises(TypeError):
            layer(x)

    def test_layer_custom_module(self, small_type, medium_type, device):
        """Test layer with custom module."""
        custom = nn.Sequential(
            nn.Linear(small_type.shape[0], medium_type.shape[0]),
            nn.ReLU()
        )
        layer = CategoricalLayer(small_type, medium_type, layer=custom)
        x = torch.randn(5, small_type.shape[0], device=device)
        y = layer(x)
        assert y.shape == (5, medium_type.shape[0])

    def test_layer_compose(self, small_type, medium_type, large_type, device):
        """Test layer composition."""
        layer1 = CategoricalLayer(small_type, medium_type, name="l1")
        layer2 = CategoricalLayer(medium_type, large_type, name="l2")

        composed = layer1.compose(layer2)
        assert composed.input_type == small_type
        assert composed.output_type == large_type

        x = torch.randn(5, small_type.shape[0], device=device)
        y = composed(x)
        assert y.shape == (5, large_type.shape[0])

    def test_layer_rshift_operator(self, small_type, medium_type, large_type, device):
        """Test >> operator for composition."""
        layer1 = CategoricalLayer(small_type, medium_type)
        layer2 = CategoricalLayer(medium_type, large_type)

        composed = layer1 >> layer2
        x = torch.randn(5, small_type.shape[0], device=device)
        y = composed(x)
        assert y.shape == (5, large_type.shape[0])

    def test_layer_compose_type_mismatch(self, small_type, medium_type, large_type):
        """Test composition fails on type mismatch."""
        layer1 = CategoricalLayer(small_type, medium_type)
        layer2 = CategoricalLayer(large_type, small_type)  # Mismatched

        with pytest.raises(TypeError):
            layer1.compose(layer2)

    def test_layer_as_morphism(self, small_type, medium_type, device):
        """Test converting layer to morphism."""
        layer = CategoricalLayer(small_type, medium_type, name="test")
        morph = layer.as_morphism()

        assert morph.source == small_type
        assert morph.target == medium_type

        x = torch.randn(5, small_type.shape[0], device=device)
        y = morph(x)
        assert y.shape == (5, medium_type.shape[0])
