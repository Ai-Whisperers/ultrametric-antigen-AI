# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for UltrametricTreeExtractor class."""

from __future__ import annotations

import pytest
import torch

from src.physics import UltrametricTreeExtractor


class TestUltrametricInit:
    """Tests for UltrametricTreeExtractor initialization."""

    def test_default_init(self):
        """Test default initialization."""
        extractor = UltrametricTreeExtractor()
        assert extractor.linkage == "average"
        assert extractor.prime == 3

    def test_custom_linkage(self):
        """Test custom linkage."""
        extractor = UltrametricTreeExtractor(linkage="complete")
        assert extractor.linkage == "complete"


class TestUltrametricCheck:
    """Tests for ultrametricity checking."""

    def test_check_identity_matrix(self, device):
        """Test identity-like matrix (zeros on diagonal)."""
        D = torch.zeros(4, 4, device=device)
        extractor = UltrametricTreeExtractor()
        is_ultra, violation = extractor.check_ultrametricity(D)
        # Zero matrix should be ultrametric
        assert is_ultra

    def test_check_ultrametric_matrix(self, device):
        """Test known ultrametric matrix."""
        # Construct ultrametric: equal distances within clusters
        D = torch.tensor([
            [0, 1, 2, 2],
            [1, 0, 2, 2],
            [2, 2, 0, 1],
            [2, 2, 1, 0],
        ], dtype=torch.float32, device=device)
        extractor = UltrametricTreeExtractor()
        is_ultra, violation = extractor.check_ultrametricity(D)
        assert is_ultra
        assert violation <= 1e-6


class TestMakeUltrametric:
    """Tests for making distance matrix ultrametric."""

    def test_make_ultrametric_shape(self, distance_matrix):
        """Test output shape matches input."""
        extractor = UltrametricTreeExtractor()
        ultra_D = extractor.make_ultrametric(distance_matrix)
        assert ultra_D.shape == distance_matrix.shape

    def test_make_ultrametric_symmetric(self, distance_matrix):
        """Test result is symmetric."""
        extractor = UltrametricTreeExtractor()
        ultra_D = extractor.make_ultrametric(distance_matrix)
        assert torch.allclose(ultra_D, ultra_D.T)

    def test_make_ultrametric_zero_diagonal(self, distance_matrix):
        """Test diagonal is zero."""
        extractor = UltrametricTreeExtractor()
        ultra_D = extractor.make_ultrametric(distance_matrix)
        assert (ultra_D.diag() == 0).all()

    def test_result_is_ultrametric(self, distance_matrix):
        """Test result satisfies ultrametric inequality."""
        extractor = UltrametricTreeExtractor()
        ultra_D = extractor.make_ultrametric(distance_matrix)
        is_ultra, violation = extractor.check_ultrametricity(ultra_D)
        assert is_ultra, f"Violation: {violation}"


class TestExtractTree:
    """Tests for tree extraction."""

    def test_extract_tree_returns_dict(self, distance_matrix):
        """Test extract_tree returns dict."""
        extractor = UltrametricTreeExtractor()
        result = extractor.extract_tree(distance_matrix)
        assert isinstance(result, dict)
        assert "ultrametric_distance" in result
        assert "heights" in result

    def test_extract_tree_keys(self, distance_matrix):
        """Test all expected keys present."""
        extractor = UltrametricTreeExtractor()
        result = extractor.extract_tree(distance_matrix)
        expected_keys = [
            "ultrametric_distance",
            "merge_order",
            "heights",
            "padic_valuations",
            "was_ultrametric",
            "max_violation",
        ]
        for key in expected_keys:
            assert key in result


class TestLinkageTypes:
    """Tests for different linkage types."""

    @pytest.mark.parametrize("linkage", ["average", "complete", "single"])
    def test_linkage_types(self, linkage, distance_matrix):
        """Test different linkage types work."""
        extractor = UltrametricTreeExtractor(linkage=linkage)
        ultra_D = extractor.make_ultrametric(distance_matrix)
        is_ultra, _ = extractor.check_ultrametricity(ultra_D)
        assert is_ultra

    def test_invalid_linkage(self, distance_matrix):
        """Test invalid linkage raises."""
        extractor = UltrametricTreeExtractor(linkage="invalid")
        with pytest.raises(ValueError):
            extractor.make_ultrametric(distance_matrix)
