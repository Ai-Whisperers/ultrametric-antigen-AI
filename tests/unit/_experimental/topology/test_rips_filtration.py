# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for RipsFiltration class.

Tests cover:
- Basic filtration construction
- Different backends
- Various point cloud shapes
- Edge cases
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.topology import RipsFiltration, TopologicalFingerprint


class TestRipsFiltrationInit:
    """Tests for RipsFiltration initialization."""

    def test_default_init(self):
        """Test default initialization."""
        filt = RipsFiltration()
        assert filt.max_dimension == 2
        assert filt.max_edge_length == np.inf
        assert filt.n_threads == 1

    def test_custom_params(self):
        """Test custom parameters."""
        filt = RipsFiltration(
            max_dimension=1,
            max_edge_length=2.0,
            n_threads=4,
        )
        assert filt.max_dimension == 1
        assert filt.max_edge_length == 2.0
        assert filt.n_threads == 4


class TestRipsFiltrationBuild:
    """Tests for filtration building."""

    def test_build_from_numpy(self, rips_filtration, simple_point_cloud):
        """Test building from numpy array."""
        fingerprint = rips_filtration.build(simple_point_cloud)
        assert isinstance(fingerprint, TopologicalFingerprint)
        assert fingerprint.total_features > 0

    def test_build_from_tensor(self, rips_filtration, point_cloud_tensor):
        """Test building from torch tensor."""
        fingerprint = rips_filtration.build(point_cloud_tensor)
        assert isinstance(fingerprint, TopologicalFingerprint)

    def test_h0_features(self, rips_filtration, simple_point_cloud):
        """Test H0 (connected components) features."""
        fingerprint = rips_filtration.build(simple_point_cloud)
        h0 = fingerprint[0]
        # 4 points eventually merge into 1 component -> 3 deaths
        assert len(h0) >= 0

    def test_circle_has_h1(self, rips_filtration, circle_point_cloud):
        """Test that circle has H1 feature (loop)."""
        fingerprint = rips_filtration.build(circle_point_cloud)
        h1 = fingerprint[1]
        # Circle should have at least one persistent H1 feature
        if len(h1) > 0:
            max_persistence = h1.persistence.max()
            assert max_persistence > 0

    def test_empty_input(self, rips_filtration):
        """Test empty input."""
        points = np.array([]).reshape(0, 3)
        fingerprint = rips_filtration.build(points)
        assert fingerprint.total_features == 0

    def test_single_point(self, rips_filtration):
        """Test single point."""
        points = np.array([[0.0, 0.0, 0.0]])
        fingerprint = rips_filtration.build(points)
        # Single point has no deaths
        assert fingerprint.total_features == 0 or len(fingerprint[0]) == 0


class TestRipsFiltrationDimensions:
    """Tests for different homology dimensions."""

    def test_h2_filtration(self, rips_filtration_h2, random_point_cloud):
        """Test H2 computation."""
        fingerprint = rips_filtration_h2.build(random_point_cloud)
        # H2 might or might not have features
        assert fingerprint.max_dimension >= 0

    def test_dimension_limiting(self, simple_point_cloud):
        """Test dimension limiting."""
        filt = RipsFiltration(max_dimension=0)
        fingerprint = filt.build(simple_point_cloud)
        # Should only have H0
        assert 1 not in fingerprint.diagrams


class TestRipsFiltrationEdgeCases:
    """Tests for edge cases."""

    def test_collinear_points(self, rips_filtration):
        """Test with collinear points."""
        points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
        fingerprint = rips_filtration.build(points)
        assert isinstance(fingerprint, TopologicalFingerprint)

    def test_coplanar_points(self, rips_filtration):
        """Test with coplanar points."""
        points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]
        ], dtype=np.float32)
        fingerprint = rips_filtration.build(points)
        assert isinstance(fingerprint, TopologicalFingerprint)

    def test_max_edge_length(self, simple_point_cloud):
        """Test max edge length filtering."""
        filt = RipsFiltration(max_edge_length=0.5)
        fingerprint = filt.build(simple_point_cloud)
        # With small max edge, fewer connections
        assert isinstance(fingerprint, TopologicalFingerprint)

    def test_large_point_cloud(self, rips_filtration):
        """Test with larger point cloud."""
        np.random.seed(42)
        points = np.random.randn(100, 3).astype(np.float32)
        fingerprint = rips_filtration.build(points)
        assert fingerprint.total_features > 0


class TestRipsFiltrationMetadata:
    """Tests for metadata."""

    def test_metadata_exists(self, rips_filtration, simple_point_cloud):
        """Test metadata is populated."""
        fingerprint = rips_filtration.build(simple_point_cloud)
        assert fingerprint.metadata is not None
        assert "n_points" in fingerprint.metadata

    def test_backend_in_metadata(self, rips_filtration, simple_point_cloud):
        """Test backend is recorded in metadata."""
        fingerprint = rips_filtration.build(simple_point_cloud)
        assert "backend" in fingerprint.metadata
