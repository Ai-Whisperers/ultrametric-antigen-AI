# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for PersistenceDiagram class.

Tests cover:
- Creation and validation
- Properties (persistence, midlife)
- Filtering
- Tensor conversion
- Wasserstein distance
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.topology import PersistenceDiagram


class TestPersistenceDiagramCreation:
    """Tests for PersistenceDiagram creation."""

    def test_basic_creation(self):
        """Test basic diagram creation."""
        diag = PersistenceDiagram(
            dimension=0,
            birth=np.array([0.0, 0.1]),
            death=np.array([0.5, 0.8]),
        )
        assert diag.dimension == 0
        assert len(diag) == 2

    def test_creation_from_lists(self):
        """Test creation from lists."""
        diag = PersistenceDiagram(
            dimension=1,
            birth=[0.0, 0.1, 0.2],
            death=[0.5, 0.8, 1.0],
        )
        assert len(diag) == 3
        assert isinstance(diag.birth, np.ndarray)

    def test_empty_diagram(self):
        """Test empty diagram creation."""
        diag = PersistenceDiagram.empty(dimension=1)
        assert len(diag) == 0
        assert diag.dimension == 1

    def test_mismatched_arrays_error(self):
        """Test error on mismatched array lengths."""
        with pytest.raises(ValueError, match="same length"):
            PersistenceDiagram(
                dimension=0,
                birth=np.array([0.0, 0.1]),
                death=np.array([0.5]),
            )


class TestPersistenceDiagramProperties:
    """Tests for PersistenceDiagram properties."""

    def test_persistence(self, simple_diagram):
        """Test persistence computation."""
        pers = simple_diagram.persistence
        expected = np.array([0.5, 0.8, 1.2])
        np.testing.assert_allclose(pers, expected)

    def test_midlife(self, simple_diagram):
        """Test midlife computation."""
        mid = simple_diagram.midlife
        expected = np.array([0.25, 0.4, 0.6])
        np.testing.assert_allclose(mid, expected)

    def test_empty_persistence(self, empty_diagram):
        """Test persistence of empty diagram."""
        pers = empty_diagram.persistence
        assert len(pers) == 0


class TestPersistenceDiagramFiltering:
    """Tests for diagram filtering."""

    def test_filter_by_persistence(self, simple_diagram):
        """Test filtering by persistence threshold."""
        filtered = simple_diagram.filter_by_persistence(threshold=0.6)
        assert len(filtered) == 2  # 0.8 and 1.2 survive

    def test_filter_all(self, simple_diagram):
        """Test filtering that removes all."""
        filtered = simple_diagram.filter_by_persistence(threshold=2.0)
        assert len(filtered) == 0

    def test_filter_none(self, simple_diagram):
        """Test filtering that keeps all."""
        filtered = simple_diagram.filter_by_persistence(threshold=0.0)
        assert len(filtered) == 3


class TestPersistenceDiagramConversion:
    """Tests for diagram conversion."""

    def test_to_tensor(self, simple_diagram):
        """Test conversion to tensor."""
        tensor = simple_diagram.to_tensor()
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 2)
        assert tensor.dtype == torch.float32

    def test_to_tensor_empty(self, empty_diagram):
        """Test tensor conversion of empty diagram."""
        tensor = empty_diagram.to_tensor()
        assert tensor.shape == (0, 2)


class TestWassersteinDistance:
    """Tests for Wasserstein distance."""

    def test_distance_to_self(self, simple_diagram):
        """Test distance to self is zero."""
        dist = simple_diagram.wasserstein_distance(simple_diagram)
        assert dist == pytest.approx(0.0)

    def test_distance_symmetric(self):
        """Test distance is symmetric."""
        d1 = PersistenceDiagram(
            dimension=0,
            birth=np.array([0.0, 0.1]),
            death=np.array([0.5, 0.8]),
        )
        d2 = PersistenceDiagram(
            dimension=0,
            birth=np.array([0.1, 0.2]),
            death=np.array([0.6, 0.9]),
        )
        dist_12 = d1.wasserstein_distance(d2)
        dist_21 = d2.wasserstein_distance(d1)
        assert dist_12 == pytest.approx(dist_21, abs=1e-5)

    def test_distance_empty_diagrams(self, empty_diagram):
        """Test distance between empty diagrams."""
        dist = empty_diagram.wasserstein_distance(empty_diagram)
        assert dist == 0.0

    def test_distance_different_sizes(self, simple_diagram, h1_diagram):
        """Test distance between diagrams of different sizes."""
        dist = simple_diagram.wasserstein_distance(h1_diagram)
        assert dist > 0
        assert np.isfinite(dist)
