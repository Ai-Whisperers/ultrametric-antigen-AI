# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for TropicalSemiring class."""

from __future__ import annotations

import pytest
import torch

from src.tropical import TropicalSemiring


class TestTropicalSemiringOperations:
    """Tests for tropical semiring operations."""

    def test_tropical_add_max(self):
        """Test tropical addition is max."""
        assert TropicalSemiring.add(3.0, 5.0) == 5.0
        assert TropicalSemiring.add(5.0, 3.0) == 5.0
        assert TropicalSemiring.add(-1.0, 2.0) == 2.0

    def test_tropical_add_with_neg_inf(self):
        """Test tropical addition with negative infinity (identity)."""
        assert TropicalSemiring.add(5.0, TropicalSemiring.NEG_INF) == 5.0
        assert TropicalSemiring.add(TropicalSemiring.NEG_INF, 3.0) == 3.0

    def test_tropical_multiply_sum(self):
        """Test tropical multiplication is sum."""
        assert TropicalSemiring.multiply(3.0, 5.0) == 8.0
        assert TropicalSemiring.multiply(-1.0, 2.0) == 1.0
        assert TropicalSemiring.multiply(0.0, 5.0) == 5.0

    def test_tropical_multiply_with_neg_inf(self):
        """Test tropical multiplication with negative infinity (zero element)."""
        assert TropicalSemiring.multiply(5.0, TropicalSemiring.NEG_INF) == TropicalSemiring.NEG_INF
        assert TropicalSemiring.multiply(TropicalSemiring.NEG_INF, 3.0) == TropicalSemiring.NEG_INF

    def test_tropical_power(self):
        """Test tropical power is scalar multiplication."""
        assert TropicalSemiring.power(3.0, 2) == 6.0
        assert TropicalSemiring.power(5.0, 0) == 0.0  # Multiplicative identity
        assert TropicalSemiring.power(2.0, 3) == 6.0

    def test_tropical_power_neg_inf(self):
        """Test tropical power with negative infinity."""
        assert TropicalSemiring.power(TropicalSemiring.NEG_INF, 5) == TropicalSemiring.NEG_INF


class TestTropicalSemiringTensors:
    """Tests for tropical operations on tensors."""

    def test_add_tensor(self):
        """Test tropical addition on tensors."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([3.0, 1.0, 2.0])
        result = TropicalSemiring.add_tensor(a, b)
        expected = torch.tensor([3.0, 2.0, 3.0])
        assert torch.allclose(result, expected)

    def test_multiply_tensor(self):
        """Test tropical multiplication on tensors."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([3.0, 1.0, 2.0])
        result = TropicalSemiring.multiply_tensor(a, b)
        expected = torch.tensor([4.0, 3.0, 5.0])
        assert torch.allclose(result, expected)


class TestSemiringProperties:
    """Tests for semiring algebraic properties."""

    def test_associativity_add(self):
        """Test associativity of tropical addition."""
        a, b, c = 2.0, 5.0, 3.0
        assert TropicalSemiring.add(TropicalSemiring.add(a, b), c) == TropicalSemiring.add(a, TropicalSemiring.add(b, c))

    def test_associativity_multiply(self):
        """Test associativity of tropical multiplication."""
        a, b, c = 2.0, 5.0, 3.0
        result1 = TropicalSemiring.multiply(TropicalSemiring.multiply(a, b), c)
        result2 = TropicalSemiring.multiply(a, TropicalSemiring.multiply(b, c))
        assert result1 == pytest.approx(result2)

    def test_commutativity_add(self):
        """Test commutativity of tropical addition."""
        a, b = 3.0, 7.0
        assert TropicalSemiring.add(a, b) == TropicalSemiring.add(b, a)

    def test_commutativity_multiply(self):
        """Test commutativity of tropical multiplication."""
        a, b = 3.0, 7.0
        assert TropicalSemiring.multiply(a, b) == TropicalSemiring.multiply(b, a)

    def test_distributivity(self):
        """Test distributivity: a*(b+c) = (a*b)+(a*c)."""
        a, b, c = 2.0, 3.0, 5.0
        left = TropicalSemiring.multiply(a, TropicalSemiring.add(b, c))
        right = TropicalSemiring.add(
            TropicalSemiring.multiply(a, b),
            TropicalSemiring.multiply(a, c)
        )
        assert left == pytest.approx(right)
