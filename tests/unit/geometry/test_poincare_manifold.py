# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for manifold creation and caching in src/geometry/poincare.py.

Tests the Poincare ball manifold factory and caching mechanism.
"""

import pytest
from src.geometry.poincare import get_manifold


class TestGetManifold:
    """Test manifold creation and caching."""

    def test_default_curvature(self):
        """Default curvature should be 1.0."""
        manifold = get_manifold()
        assert float(manifold.c) == pytest.approx(1.0)

    def test_custom_curvature(self):
        """Custom curvature should be respected."""
        manifold = get_manifold(c=0.5)
        assert float(manifold.c) == pytest.approx(0.5)

    def test_caching(self):
        """Same curvature should return cached manifold."""
        m1 = get_manifold(c=1.0)
        m2 = get_manifold(c=1.0)
        assert m1 is m2

    def test_different_curvatures_cached_separately(self):
        """Different curvatures should create different manifolds."""
        m1 = get_manifold(c=1.0)
        m2 = get_manifold(c=2.0)
        assert m1 is not m2
        assert m1.c != m2.c

    def test_various_curvatures(self):
        """Test various curvature values."""
        for c in [0.1, 0.5, 1.0, 2.0, 5.0]:
            manifold = get_manifold(c=c)
            assert float(manifold.c) == pytest.approx(c)

    def test_small_curvature(self):
        """Test very small curvature (nearly Euclidean)."""
        manifold = get_manifold(c=0.01)
        assert float(manifold.c) == pytest.approx(0.01, rel=1e-5)

    def test_large_curvature(self):
        """Test large curvature (highly hyperbolic)."""
        manifold = get_manifold(c=10.0)
        assert float(manifold.c) == pytest.approx(10.0)
