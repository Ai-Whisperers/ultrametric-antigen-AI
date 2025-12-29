# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for PersistenceVectorizer class.

Tests cover:
- Statistics vectorization
- Landscape vectorization
- Image vectorization
- Output dimensions
"""

from __future__ import annotations

import numpy as np
import pytest

from src.topology import PersistenceVectorizer, TopologicalFingerprint, PersistenceDiagram


class TestPersistenceVectorizerInit:
    """Tests for PersistenceVectorizer initialization."""

    def test_default_init(self):
        """Test default initialization."""
        vec = PersistenceVectorizer()
        assert vec.method == "statistics"
        assert vec.resolution == 50
        assert vec.dimensions == [0, 1]

    def test_custom_method(self):
        """Test custom method."""
        vec = PersistenceVectorizer(method="landscape")
        assert vec.method == "landscape"

    def test_custom_resolution(self):
        """Test custom resolution."""
        vec = PersistenceVectorizer(resolution=100)
        assert vec.resolution == 100


class TestStatisticsVectorization:
    """Tests for statistics vectorization."""

    def test_output_shape(self, statistics_vectorizer, multi_dim_fingerprint):
        """Test output shape."""
        vector = statistics_vectorizer.transform(multi_dim_fingerprint)
        # 10 stats per dimension, 2 dimensions
        assert vector.shape == (20,)

    def test_output_dtype(self, statistics_vectorizer, multi_dim_fingerprint):
        """Test output dtype."""
        vector = statistics_vectorizer.transform(multi_dim_fingerprint)
        assert vector.dtype == np.float32

    def test_empty_fingerprint(self, statistics_vectorizer):
        """Test with empty fingerprint."""
        empty = TopologicalFingerprint()
        vector = statistics_vectorizer.transform(empty)
        assert vector.shape == (20,)
        assert (vector == 0).all()

    def test_statistics_content(self, statistics_vectorizer, multi_dim_fingerprint):
        """Test statistics content is reasonable."""
        vector = statistics_vectorizer.transform(multi_dim_fingerprint)
        # First value should be count of H0 features
        assert vector[0] == 3  # simple_diagram has 3 features
        # Mean persistence should be positive
        assert vector[1] > 0


class TestLandscapeVectorization:
    """Tests for landscape vectorization."""

    def test_output_shape(self, landscape_vectorizer, multi_dim_fingerprint):
        """Test output shape."""
        vector = landscape_vectorizer.transform(multi_dim_fingerprint)
        # resolution per dimension
        assert vector.shape == (40,)  # 20 * 2

    def test_empty_fingerprint(self, landscape_vectorizer):
        """Test with empty fingerprint."""
        empty = TopologicalFingerprint()
        vector = landscape_vectorizer.transform(empty)
        assert (vector == 0).all()

    def test_landscape_non_negative(self, landscape_vectorizer, multi_dim_fingerprint):
        """Test landscape is non-negative."""
        vector = landscape_vectorizer.transform(multi_dim_fingerprint)
        assert (vector >= 0).all()


class TestImageVectorization:
    """Tests for image vectorization."""

    def test_output_shape(self, image_vectorizer, multi_dim_fingerprint):
        """Test output shape."""
        vector = image_vectorizer.transform(multi_dim_fingerprint)
        # resolution^2 per dimension
        assert vector.shape == (200,)  # 10*10*2

    def test_empty_fingerprint(self, image_vectorizer):
        """Test with empty fingerprint."""
        empty = TopologicalFingerprint()
        vector = image_vectorizer.transform(empty)
        assert (vector == 0).all()

    def test_image_non_negative(self, image_vectorizer, multi_dim_fingerprint):
        """Test image is non-negative."""
        vector = image_vectorizer.transform(multi_dim_fingerprint)
        assert (vector >= 0).all()


class TestVectorizerOutputDim:
    """Tests for output dimension property."""

    def test_statistics_output_dim(self):
        """Test statistics output dimension."""
        vec = PersistenceVectorizer(method="statistics", dimensions=[0, 1])
        assert vec.output_dim == 20

    def test_landscape_output_dim(self):
        """Test landscape output dimension."""
        vec = PersistenceVectorizer(method="landscape", resolution=30, dimensions=[0])
        assert vec.output_dim == 30

    def test_image_output_dim(self):
        """Test image output dimension."""
        vec = PersistenceVectorizer(method="image", resolution=15, dimensions=[0, 1])
        assert vec.output_dim == 15 * 15 * 2

    def test_invalid_method_error(self):
        """Test error on invalid method."""
        vec = PersistenceVectorizer(method="invalid")
        with pytest.raises(ValueError, match="Unknown method"):
            vec.output_dim


class TestVectorizerDimensions:
    """Tests for dimension handling."""

    def test_single_dimension(self, multi_dim_fingerprint):
        """Test with single dimension."""
        vec = PersistenceVectorizer(method="statistics", dimensions=[0])
        vector = vec.transform(multi_dim_fingerprint)
        assert vector.shape == (10,)

    def test_three_dimensions(self):
        """Test with three dimensions."""
        vec = PersistenceVectorizer(method="statistics", dimensions=[0, 1, 2])
        empty = TopologicalFingerprint()
        vector = vec.transform(empty)
        assert vector.shape == (30,)

    def test_missing_dimension(self, multi_dim_fingerprint):
        """Test with dimension not in fingerprint."""
        vec = PersistenceVectorizer(method="statistics", dimensions=[0, 1, 2])
        vector = vec.transform(multi_dim_fingerprint)
        # H2 portion should be zeros
        assert vector.shape == (30,)
