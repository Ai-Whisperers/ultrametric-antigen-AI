# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for TropicalConvexHull class."""

from __future__ import annotations

import numpy as np
import pytest

from src.tropical import TropicalConvexHull


class TestTropicalConvexHullInit:
    """Tests for convex hull initialization."""

    def test_init(self, sample_points):
        """Test initialization with points."""
        hull = TropicalConvexHull(sample_points)
        assert hull.n_points == 4
        assert hull.dim == 2

    def test_init_single_point(self):
        """Test initialization with single point."""
        points = np.array([[1.0, 2.0]])
        hull = TropicalConvexHull(points)
        assert hull.n_points == 1
        assert hull.dim == 2


class TestContains:
    """Tests for containment checking."""

    def test_contains_vertex(self, sample_points):
        """Test that vertices are contained."""
        hull = TropicalConvexHull(sample_points)
        for point in sample_points:
            assert hull.contains(point)

    def test_contains_interior(self, sample_points):
        """Test interior point is contained."""
        hull = TropicalConvexHull(sample_points)
        # Point in the middle should be contained
        interior = np.array([1.0, 1.0])
        assert hull.contains(interior)

    def test_not_contains_exterior(self, sample_points):
        """Test exterior point is not contained."""
        hull = TropicalConvexHull(sample_points)
        # Point clearly outside
        exterior = np.array([10.0, 10.0])
        assert not hull.contains(exterior)

    def test_not_contains_below_min(self, sample_points):
        """Test point below minimum is not contained."""
        hull = TropicalConvexHull(sample_points)
        below = np.array([-5.0, 0.0])
        assert not hull.contains(below)


class TestExtremePoints:
    """Tests for extreme point finding."""

    def test_extreme_points_exist(self, sample_points):
        """Test that extreme points are found."""
        hull = TropicalConvexHull(sample_points)
        extreme = hull.extreme_points()
        assert len(extreme) >= 1

    def test_extreme_points_subset(self, sample_points):
        """Test extreme points are subset of original."""
        hull = TropicalConvexHull(sample_points)
        extreme = hull.extreme_points()

        # Each extreme point should be in original points
        for ext in extreme:
            found = False
            for orig in sample_points:
                if np.allclose(ext, orig):
                    found = True
                    break
            assert found

    def test_single_point_extreme(self):
        """Test single point is its own extreme."""
        points = np.array([[1.0, 2.0]])
        hull = TropicalConvexHull(points)
        extreme = hull.extreme_points()
        assert len(extreme) == 1
        np.testing.assert_allclose(extreme[0], points[0])


class TestEdgeCases:
    """Tests for edge cases."""

    def test_collinear_points(self):
        """Test with collinear points."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ])
        hull = TropicalConvexHull(points)
        # Should still work
        assert hull.n_points == 3
        extreme = hull.extreme_points()
        assert len(extreme) >= 2  # At least endpoints

    def test_2d_points(self):
        """Test with 2D points."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        hull = TropicalConvexHull(points)
        assert hull.dim == 2

        # Test contains for corner
        assert hull.contains(np.array([0.5, 0.5]))

    def test_higher_dimension(self):
        """Test with higher dimensional points."""
        points = np.random.rand(10, 5)
        hull = TropicalConvexHull(points)
        assert hull.dim == 5

        # Test contains for point within bounds
        query = np.mean(points, axis=0)
        assert hull.contains(query)
