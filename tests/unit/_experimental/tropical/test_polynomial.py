# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for TropicalMonomial and TropicalPolynomial classes."""

from __future__ import annotations

import numpy as np
import pytest

from src.tropical import TropicalMonomial, TropicalPolynomial, TropicalSemiring


class TestTropicalMonomial:
    """Tests for TropicalMonomial class."""

    def test_monomial_creation(self):
        """Test monomial creation."""
        m = TropicalMonomial(coefficient=1.0, exponents=(2, 3))
        assert m.coefficient == 1.0
        assert m.exponents == (2, 3)

    def test_monomial_evaluate(self):
        """Test monomial evaluation at a point."""
        # c + a1*x1 + a2*x2 = 1 + 2*3 + 3*4 = 1 + 6 + 12 = 19
        m = TropicalMonomial(coefficient=1.0, exponents=(2, 3))
        x = np.array([3.0, 4.0])
        result = m.evaluate(x)
        assert result == pytest.approx(19.0)

    def test_monomial_evaluate_zero_exponent(self):
        """Test monomial with zero exponent."""
        m = TropicalMonomial(coefficient=5.0, exponents=(0, 1))
        x = np.array([10.0, 2.0])
        result = m.evaluate(x)
        # 5 + 0*10 + 1*2 = 7
        assert result == pytest.approx(7.0)

    def test_monomial_wrong_dimension(self):
        """Test monomial with wrong input dimension raises."""
        m = TropicalMonomial(coefficient=1.0, exponents=(1, 2))
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            m.evaluate(x)


class TestTropicalPolynomial:
    """Tests for TropicalPolynomial class."""

    def test_polynomial_creation_empty(self):
        """Test empty polynomial creation."""
        p = TropicalPolynomial()
        assert p.n_monomials == 0

    def test_polynomial_add_monomial(self):
        """Test adding monomials to polynomial."""
        p = TropicalPolynomial()
        p.add_monomial(TropicalMonomial(1.0, (1, 0)))
        p.add_monomial(TropicalMonomial(2.0, (0, 1)))
        assert p.n_monomials == 2

    def test_polynomial_evaluate(self):
        """Test polynomial evaluation (tropical max)."""
        p = TropicalPolynomial()
        # max(1 + x1, 2 + x2)
        p.add_monomial(TropicalMonomial(1.0, (1, 0)))
        p.add_monomial(TropicalMonomial(2.0, (0, 1)))

        x1 = np.array([5.0, 0.0])  # max(1+5, 2+0) = max(6, 2) = 6
        assert p.evaluate(x1) == pytest.approx(6.0)

        x2 = np.array([0.0, 5.0])  # max(1+0, 2+5) = max(1, 7) = 7
        assert p.evaluate(x2) == pytest.approx(7.0)

    def test_polynomial_evaluate_empty(self):
        """Test empty polynomial evaluates to -inf."""
        p = TropicalPolynomial()
        result = p.evaluate(np.array([1.0, 2.0]))
        assert result == TropicalSemiring.NEG_INF

    def test_polynomial_active_monomial(self):
        """Test finding active (maximizing) monomial."""
        p = TropicalPolynomial()
        p.add_monomial(TropicalMonomial(0.0, (1, 0)))  # x1
        p.add_monomial(TropicalMonomial(0.0, (0, 1)))  # x2

        x = np.array([5.0, 3.0])  # x1 > x2, so monomial 0 is active
        assert p.active_monomial(x) == 0

        x = np.array([2.0, 8.0])  # x2 > x1, so monomial 1 is active
        assert p.active_monomial(x) == 1

    def test_tropical_add(self):
        """Test tropical addition of polynomials."""
        p1 = TropicalPolynomial([TropicalMonomial(1.0, (1,))])
        p2 = TropicalPolynomial([TropicalMonomial(2.0, (1,))])
        result = p1.tropical_add(p2)
        assert result.n_monomials == 2

    def test_tropical_multiply(self):
        """Test tropical multiplication of polynomials."""
        p1 = TropicalPolynomial([TropicalMonomial(1.0, (1, 0))])
        p2 = TropicalPolynomial([TropicalMonomial(2.0, (0, 1))])
        result = p1.tropical_multiply(p2)

        # Result: (1+2, (1+0, 0+1)) = (3, (1, 1))
        assert result.n_monomials == 1
        assert result.monomials[0].coefficient == pytest.approx(3.0)
        assert result.monomials[0].exponents == (1, 1)

    def test_evaluate_batch(self):
        """Test batch evaluation."""
        p = TropicalPolynomial()
        p.add_monomial(TropicalMonomial(0.0, (1, 0)))

        X = np.array([
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ])
        results = p.evaluate_batch(X)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(results, expected)
