# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for TropicalPhylogeneticTree and TropicalPhylogeneticDistance classes."""

from __future__ import annotations

import numpy as np
import pytest

from src.tropical import TropicalPhylogeneticTree, TropicalPhylogeneticDistance


class TestTropicalPhylogeneticTree:
    """Tests for TropicalPhylogeneticTree class."""

    def test_tree_creation(self, sample_tree_edges):
        """Test tree creation."""
        tree = TropicalPhylogeneticTree(
            n_taxa=3,
            edge_lengths=sample_tree_edges,
        )
        assert tree.n_taxa == 3
        assert len(tree.edge_lengths) == 4

    def test_to_tropical_coordinates(self, sample_tree_edges):
        """Test conversion to tropical coordinates."""
        tree = TropicalPhylogeneticTree(
            n_taxa=3,
            edge_lengths=sample_tree_edges,
        )
        coords = tree.to_tropical_coordinates()
        # For 3 taxa, should have 3 choose 2 = 3 coordinates
        assert len(coords) == 3

    def test_tropical_coordinates_shape(self):
        """Test tropical coordinates shape for different sizes."""
        for n in [3, 4, 5]:
            tree = TropicalPhylogeneticTree(
                n_taxa=n,
                edge_lengths=np.ones(2 * n - 2),
            )
            coords = tree.to_tropical_coordinates()
            expected_pairs = n * (n - 1) // 2
            assert len(coords) == expected_pairs


class TestTropicalPhylogeneticDistance:
    """Tests for TropicalPhylogeneticDistance class."""

    def test_distance_init(self):
        """Test distance calculator initialization."""
        calc = TropicalPhylogeneticDistance(n_taxa=4)
        assert calc.n_taxa == 4
        assert calc.n_pairs == 6  # 4 choose 2

    def test_distance_same_tree(self):
        """Test distance between identical trees is zero."""
        tree1 = TropicalPhylogeneticTree(
            n_taxa=3,
            edge_lengths=np.array([1.0, 2.0, 1.5, 0.5]),
        )
        tree2 = TropicalPhylogeneticTree(
            n_taxa=3,
            edge_lengths=np.array([1.0, 2.0, 1.5, 0.5]),
        )
        calc = TropicalPhylogeneticDistance(n_taxa=3)
        d = calc.distance(tree1, tree2)
        assert d == pytest.approx(0.0)

    def test_distance_symmetric(self):
        """Test distance is symmetric."""
        tree1 = TropicalPhylogeneticTree(
            n_taxa=3,
            edge_lengths=np.array([1.0, 2.0, 1.5, 0.5]),
        )
        tree2 = TropicalPhylogeneticTree(
            n_taxa=3,
            edge_lengths=np.array([2.0, 1.0, 0.5, 1.5]),
        )
        calc = TropicalPhylogeneticDistance(n_taxa=3)
        d12 = calc.distance(tree1, tree2)
        d21 = calc.distance(tree2, tree1)
        assert d12 == pytest.approx(d21)

    def test_distance_non_negative(self):
        """Test distance is non-negative."""
        calc = TropicalPhylogeneticDistance(n_taxa=3)
        for _ in range(10):
            tree1 = TropicalPhylogeneticTree(
                n_taxa=3,
                edge_lengths=np.random.rand(4),
            )
            tree2 = TropicalPhylogeneticTree(
                n_taxa=3,
                edge_lengths=np.random.rand(4),
            )
            d = calc.distance(tree1, tree2)
            assert d >= 0


class TestDistanceMatrix:
    """Tests for distance matrix computation."""

    def test_distance_matrix_shape(self):
        """Test distance matrix shape."""
        calc = TropicalPhylogeneticDistance(n_taxa=3)
        trees = [
            TropicalPhylogeneticTree(n_taxa=3, edge_lengths=np.random.rand(4))
            for _ in range(5)
        ]
        D = calc.distance_matrix(trees)
        assert D.shape == (5, 5)

    def test_distance_matrix_symmetric(self):
        """Test distance matrix is symmetric."""
        calc = TropicalPhylogeneticDistance(n_taxa=3)
        trees = [
            TropicalPhylogeneticTree(n_taxa=3, edge_lengths=np.random.rand(4))
            for _ in range(4)
        ]
        D = calc.distance_matrix(trees)
        np.testing.assert_allclose(D, D.T)

    def test_distance_matrix_zero_diagonal(self):
        """Test distance matrix has zero diagonal."""
        calc = TropicalPhylogeneticDistance(n_taxa=3)
        trees = [
            TropicalPhylogeneticTree(n_taxa=3, edge_lengths=np.random.rand(4))
            for _ in range(4)
        ]
        D = calc.distance_matrix(trees)
        np.testing.assert_allclose(np.diag(D), np.zeros(4))


class TestFrechetMean:
    """Tests for Frechet mean computation."""

    def test_frechet_mean_shape(self):
        """Test Frechet mean has correct shape."""
        calc = TropicalPhylogeneticDistance(n_taxa=3)
        trees = [
            TropicalPhylogeneticTree(n_taxa=3, edge_lengths=np.random.rand(4))
            for _ in range(5)
        ]
        mean = calc.frechet_mean(trees)
        assert len(mean) == 3  # n_pairs for n_taxa=3


class TestGeodesic:
    """Tests for geodesic computation."""

    def test_geodesic_endpoints(self):
        """Test geodesic starts and ends at correct points."""
        tree1 = TropicalPhylogeneticTree(
            n_taxa=3,
            edge_lengths=np.array([1.0, 1.0, 1.0, 1.0]),
        )
        tree2 = TropicalPhylogeneticTree(
            n_taxa=3,
            edge_lengths=np.array([2.0, 2.0, 2.0, 2.0]),
        )
        calc = TropicalPhylogeneticDistance(n_taxa=3)
        path = calc.geodesic(tree1, tree2, n_points=10)

        assert len(path) == 10
        np.testing.assert_allclose(path[0], tree1.to_tropical_coordinates())
        np.testing.assert_allclose(path[-1], tree2.to_tropical_coordinates())
