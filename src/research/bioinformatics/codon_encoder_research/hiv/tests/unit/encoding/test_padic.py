"""
Unit tests for p-adic arithmetic.
"""
import pytest
from src.encoding.padic.number import PadicNumber
from src.encoding.padic.arithmetic import padic_add, padic_multiply, padic_subtract
from src.encoding.padic.distance import padic_distance, padic_norm, is_ultrametric


class TestPadicNumber:
    """Tests for PadicNumber class."""

    def test_creation_from_integer(self):
        """Test p-adic number creation from integer."""
        p = PadicNumber.from_integer(11, prime=3)
        assert p.prime == 3
        # 11 = 1*9 + 0*3 + 2 = 102 in base 3
        assert p.digits == [2, 0, 1]

    def test_creation_from_string(self):
        """Test creation from string."""
        p = PadicNumber.from_string("102", prime=3)
        assert p.to_integer() == 11

    def test_valuation_nonzero(self):
        """Test valuation of nonzero number."""
        # 9 = 0*1 + 0*3 + 1*9 = 100 in base 3
        p = PadicNumber.from_integer(9, prime=3)
        assert p.valuation() == 2

    def test_valuation_zero(self):
        """Test valuation of zero."""
        p = PadicNumber(digits=[0], prime=3)
        assert p.valuation() == p.precision

    def test_norm(self):
        """Test p-adic norm."""
        # |9|_3 = 3^(-2) = 1/9
        p = PadicNumber.from_integer(9, prime=3)
        assert p.norm() == pytest.approx(1 / 9)

    def test_to_integer_roundtrip(self):
        """Test integer conversion roundtrip."""
        for n in [0, 1, 5, 27, 100]:
            p = PadicNumber.from_integer(n, prime=3)
            assert p.to_integer() == n


class TestPadicArithmetic:
    """Tests for p-adic arithmetic operations."""

    def test_addition(self):
        """Test p-adic addition."""
        a = PadicNumber.from_integer(5, prime=3)
        b = PadicNumber.from_integer(7, prime=3)
        result = padic_add(a, b)
        assert result.to_integer() == 12

    def test_subtraction(self):
        """Test p-adic subtraction."""
        a = PadicNumber.from_integer(10, prime=3)
        b = PadicNumber.from_integer(3, prime=3)
        result = padic_subtract(a, b)
        assert result.to_integer() == 7

    def test_multiplication(self):
        """Test p-adic multiplication."""
        a = PadicNumber.from_integer(4, prime=3)
        b = PadicNumber.from_integer(5, prime=3)
        result = padic_multiply(a, b)
        assert result.to_integer() == 20

    def test_prime_mismatch_error(self):
        """Test error on prime mismatch."""
        a = PadicNumber.from_integer(5, prime=3)
        b = PadicNumber.from_integer(5, prime=5)
        with pytest.raises(ValueError):
            padic_add(a, b)


class TestPadicDistance:
    """Tests for p-adic distance."""

    def test_distance_to_self(self):
        """Distance to self is 0."""
        p = PadicNumber.from_integer(11, prime=3)
        assert padic_distance(p, p) == 0

    def test_distance_symmetry(self):
        """Distance is symmetric."""
        a = PadicNumber.from_integer(5, prime=3)
        b = PadicNumber.from_integer(14, prime=3)
        assert padic_distance(a, b) == padic_distance(b, a)

    def test_ultrametric_inequality(self):
        """Test ultrametric inequality: d(a,c) <= max(d(a,b), d(b,c))."""
        a = PadicNumber.from_integer(1, prime=3)
        b = PadicNumber.from_integer(4, prime=3)
        c = PadicNumber.from_integer(7, prime=3)
        assert is_ultrametric(a, b, c)

    def test_distance_divisibility(self):
        """Closer when difference divisible by higher power of p."""
        # 3 and 12 differ by 9 = 3^2, so distance = 3^(-2) = 1/9
        a = PadicNumber.from_integer(3, prime=3)
        b = PadicNumber.from_integer(12, prime=3)
        assert padic_distance(a, b) == pytest.approx(1 / 9)

    def test_norm_equals_self_distance(self):
        """Norm equals distance from 0."""
        p = PadicNumber.from_integer(9, prime=3)
        zero = PadicNumber(digits=[0], prime=3)
        assert padic_norm(p) == padic_distance(zero, p)
