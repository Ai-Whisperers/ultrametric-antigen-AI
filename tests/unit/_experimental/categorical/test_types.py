# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for TensorType class."""

from __future__ import annotations

import pytest
import torch

from src.categorical import TensorType


class TestTensorTypeInit:
    """Tests for TensorType initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        t = TensorType((4, 8))
        assert t.shape == (4, 8)
        assert t.dtype == torch.float32

    def test_custom_dtype(self):
        """Test custom dtype."""
        t = TensorType((4,), dtype=torch.float64)
        assert t.dtype == torch.float64

    def test_custom_device(self):
        """Test custom device."""
        t = TensorType((4,), device="cuda")
        assert t.device == "cuda"


class TestTensorTypeCompatibility:
    """Tests for type compatibility checking."""

    def test_compatible_tensor(self):
        """Test compatible tensor."""
        t = TensorType((4,))
        tensor = torch.randn(5, 4)  # Batch of 5, feature size 4
        assert t.is_compatible(tensor)

    def test_incompatible_shape(self):
        """Test incompatible shape."""
        t = TensorType((4,))
        tensor = torch.randn(5, 8)  # Wrong feature size
        assert not t.is_compatible(tensor)

    def test_incompatible_dtype(self):
        """Test incompatible dtype."""
        t = TensorType((4,), dtype=torch.float32)
        tensor = torch.randn(5, 4).double()
        assert not t.is_compatible(tensor)

    def test_compatible_higher_dims(self):
        """Test compatibility with higher dimensions."""
        t = TensorType((4, 8))
        tensor = torch.randn(5, 4, 8)
        assert t.is_compatible(tensor)

    def test_too_few_dims(self):
        """Test tensor with too few dimensions."""
        t = TensorType((4, 8))
        tensor = torch.randn(4)  # Missing batch and one dim
        assert not t.is_compatible(tensor)


class TestTensorTypeFromTensor:
    """Tests for creating TensorType from tensor."""

    def test_from_tensor(self):
        """Test from_tensor method."""
        tensor = torch.randn(5, 4, 8)
        t = TensorType.from_tensor(tensor)
        # Should exclude batch dimension
        assert t.shape == (4, 8)
        assert t.dtype == tensor.dtype

    def test_from_tensor_1d(self):
        """Test from_tensor with 1D feature."""
        tensor = torch.randn(10, 16)
        t = TensorType.from_tensor(tensor)
        assert t.shape == (16,)


class TestTensorTypeEquality:
    """Tests for type equality."""

    def test_equal_types(self):
        """Test equal types."""
        t1 = TensorType((4,))
        t2 = TensorType((4,))
        assert t1 == t2

    def test_unequal_shapes(self):
        """Test unequal shapes."""
        t1 = TensorType((4,))
        t2 = TensorType((8,))
        assert t1 != t2

    def test_unequal_dtypes(self):
        """Test unequal dtypes."""
        t1 = TensorType((4,), dtype=torch.float32)
        t2 = TensorType((4,), dtype=torch.float64)
        assert t1 != t2
