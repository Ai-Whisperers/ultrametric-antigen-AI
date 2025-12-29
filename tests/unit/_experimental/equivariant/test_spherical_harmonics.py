# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for spherical harmonics utilities."""

from __future__ import annotations

import math

import pytest
import torch

from src.equivariant import (
    ClebschGordanCoefficients,
    SphericalHarmonics,
    associated_legendre,
    spherical_harmonics_manual,
)


class TestAssociatedLegendre:
    """Tests for associated Legendre polynomials."""

    def test_p00(self, device):
        """Test P_0^0(x) = 1."""
        x = torch.linspace(-1, 1, 10, device=device)
        result = associated_legendre(0, 0, x)
        assert torch.allclose(result, torch.ones_like(x))

    def test_p10(self, device):
        """Test P_1^0(x) = x."""
        x = torch.linspace(-0.99, 0.99, 10, device=device)
        result = associated_legendre(1, 0, x)
        assert torch.allclose(result, x, atol=1e-5)

    def test_p20(self, device):
        """Test P_2^0(x) = (3x^2 - 1)/2."""
        x = torch.linspace(-0.99, 0.99, 10, device=device)
        result = associated_legendre(2, 0, x)
        expected = 0.5 * (3 * x**2 - 1)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_m_greater_than_l(self, device):
        """Test that P_l^m = 0 when |m| > l."""
        x = torch.linspace(-1, 1, 10, device=device)
        result = associated_legendre(2, 5, x)
        assert torch.allclose(result, torch.zeros_like(x))


class TestSphericalHarmonicsManual:
    """Tests for manual spherical harmonics computation."""

    def test_y00_constant(self, device):
        """Test Y_0^0 is constant 1/(2*sqrt(pi))."""
        theta = torch.rand(10, device=device) * math.pi
        phi = torch.rand(10, device=device) * 2 * math.pi

        result = spherical_harmonics_manual(0, 0, theta, phi)
        expected = 1.0 / (2 * math.sqrt(math.pi))

        assert torch.allclose(result, torch.full_like(result, expected), atol=1e-5)

    def test_orthogonality(self, device):
        """Test orthogonality of spherical harmonics (approximate)."""
        # Sample points on sphere
        n_points = 1000
        theta = torch.rand(n_points, device=device) * math.pi
        phi = torch.rand(n_points, device=device) * 2 * math.pi

        # Y_0^0 and Y_1^0 should be approximately orthogonal
        y00 = spherical_harmonics_manual(0, 0, theta, phi)
        y10 = spherical_harmonics_manual(1, 0, theta, phi)

        # Numerical integration with sin(theta) weight
        weights = torch.sin(theta)
        integral = (y00 * y10 * weights).mean()

        # Should be close to zero (not exact due to finite sampling)
        assert abs(integral.item()) < 0.1


class TestSphericalHarmonicsModule:
    """Tests for SphericalHarmonics module."""

    def test_init_default(self):
        """Test default initialization."""
        sh = SphericalHarmonics()
        assert sh.lmax == 2
        assert sh.n_harmonics == 9  # (2+1)^2 = 9

    def test_init_custom_lmax(self):
        """Test custom lmax initialization."""
        sh = SphericalHarmonics(lmax=3)
        assert sh.lmax == 3
        assert sh.n_harmonics == 16  # (3+1)^2 = 16

    def test_output_dim(self):
        """Test output dimension property."""
        sh = SphericalHarmonics(lmax=4)
        assert sh.output_dim == 25  # (4+1)^2 = 25

    def test_forward_shape(self, random_vectors):
        """Test forward pass output shape."""
        sh = SphericalHarmonics(lmax=2, use_e3nn=False)
        result = sh(random_vectors)
        assert result.shape == (10, 9)

    def test_forward_batched(self, device):
        """Test forward with batched input."""
        sh = SphericalHarmonics(lmax=2, use_e3nn=False)
        vectors = torch.randn(5, 10, 3, device=device)
        result = sh(vectors)
        assert result.shape == (5, 10, 9)

    def test_forward_unit_vectors(self, device):
        """Test forward with unit vectors."""
        sh = SphericalHarmonics(lmax=2, use_e3nn=False)
        # Unit vectors along axes
        vectors = torch.eye(3, device=device)
        result = sh(vectors)
        assert result.shape == (3, 9)
        assert torch.all(torch.isfinite(result))


class TestClebschGordanCoefficients:
    """Tests for Clebsch-Gordan coefficients."""

    def test_init(self):
        """Test initialization."""
        cg = ClebschGordanCoefficients(lmax=2)
        assert cg.lmax == 2

    def test_selection_rule_m(self):
        """Test m1 + m2 = m3 selection rule."""
        cg = ClebschGordanCoefficients()

        # Valid: m1 + m2 = m3
        val1 = cg(1, 0, 1, 0, 0, 0)

        # Invalid: m1 + m2 != m3
        val2 = cg(1, 0, 1, 0, 0, 1)

        assert val2 == 0.0

    def test_selection_rule_l(self):
        """Test triangular inequality for l values."""
        cg = ClebschGordanCoefficients()

        # |l1 - l2| <= l3 <= l1 + l2
        # Invalid: l3 > l1 + l2
        val = cg(1, 0, 1, 0, 5, 0)
        assert val == 0.0

    def test_known_value(self):
        """Test known Clebsch-Gordan coefficient."""
        cg = ClebschGordanCoefficients()

        # <1,0;1,0|0,0> should be non-zero
        val = cg(1, 0, 1, 0, 0, 0)
        # This is -1/sqrt(3) from tables
        assert abs(abs(val) - 1.0 / math.sqrt(3)) < 0.1

    def test_caching(self):
        """Test that coefficients are cached."""
        cg = ClebschGordanCoefficients()

        # Compute same coefficient twice
        val1 = cg(1, 0, 1, 0, 2, 0)
        val2 = cg(1, 0, 1, 0, 2, 0)

        assert val1 == val2
        assert (1, 0, 1, 0, 2, 0) in cg._cache
