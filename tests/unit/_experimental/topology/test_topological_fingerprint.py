# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for TopologicalFingerprint class.

Tests cover:
- Creation and access
- Properties
- Betti numbers
"""

from __future__ import annotations

import numpy as np
import pytest

from src.topology import TopologicalFingerprint, PersistenceDiagram


class TestTopologicalFingerprintCreation:
    """Tests for TopologicalFingerprint creation."""

    def test_empty_creation(self):
        """Test empty fingerprint creation."""
        fp = TopologicalFingerprint()
        assert len(fp) == 0
        assert fp.total_features == 0

    def test_with_diagrams(self, simple_diagram, h1_diagram):
        """Test creation with diagrams."""
        fp = TopologicalFingerprint(diagrams={0: simple_diagram, 1: h1_diagram})
        assert len(fp) == 2
        assert 0 in fp.diagrams
        assert 1 in fp.diagrams

    def test_with_metadata(self):
        """Test creation with metadata."""
        fp = TopologicalFingerprint(metadata={"key": "value"})
        assert fp.metadata["key"] == "value"


class TestTopologicalFingerprintAccess:
    """Tests for diagram access."""

    def test_getitem_existing(self, multi_dim_fingerprint):
        """Test accessing existing dimension."""
        h0 = multi_dim_fingerprint[0]
        assert isinstance(h0, PersistenceDiagram)
        assert h0.dimension == 0

    def test_getitem_missing(self, multi_dim_fingerprint):
        """Test accessing missing dimension returns empty."""
        h2 = multi_dim_fingerprint[2]
        assert isinstance(h2, PersistenceDiagram)
        assert len(h2) == 0


class TestTopologicalFingerprintProperties:
    """Tests for fingerprint properties."""

    def test_max_dimension(self, multi_dim_fingerprint):
        """Test max_dimension property."""
        assert multi_dim_fingerprint.max_dimension == 1

    def test_max_dimension_empty(self):
        """Test max_dimension for empty fingerprint."""
        fp = TopologicalFingerprint()
        assert fp.max_dimension == -1

    def test_total_features(self, multi_dim_fingerprint):
        """Test total_features property."""
        # simple_diagram has 3, h1_diagram has 2
        assert multi_dim_fingerprint.total_features == 5

    def test_total_features_empty(self):
        """Test total_features for empty fingerprint."""
        fp = TopologicalFingerprint()
        assert fp.total_features == 0


class TestBettiNumbers:
    """Tests for Betti number computation."""

    def test_betti_at_threshold(self):
        """Test Betti numbers at specific threshold."""
        # Create diagram where features are alive at different thresholds
        diag = PersistenceDiagram(
            dimension=0,
            birth=np.array([0.0, 0.0, 0.0]),
            death=np.array([0.5, 1.0, 1.5]),
        )
        fp = TopologicalFingerprint(diagrams={0: diag})

        # At threshold 0.3: all 3 features alive
        betti_03 = fp.betti_numbers(threshold=0.3)
        assert betti_03[0] == 3

        # At threshold 0.7: 2 features alive (death > 0.7)
        betti_07 = fp.betti_numbers(threshold=0.7)
        assert betti_07[0] == 2

        # At threshold 1.2: 1 feature alive
        betti_12 = fp.betti_numbers(threshold=1.2)
        assert betti_12[0] == 1

    def test_betti_empty(self):
        """Test Betti numbers for empty fingerprint."""
        fp = TopologicalFingerprint()
        betti = fp.betti_numbers(threshold=0.5)
        assert betti == {}

    def test_betti_multiple_dimensions(self, multi_dim_fingerprint):
        """Test Betti numbers for multiple dimensions."""
        betti = multi_dim_fingerprint.betti_numbers(threshold=0.0)
        assert 0 in betti
        assert 1 in betti
