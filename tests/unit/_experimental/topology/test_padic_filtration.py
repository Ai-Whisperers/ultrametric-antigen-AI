# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for PAdicFiltration class.

Tests cover:
- P-adic valuation
- P-adic distance
- Filtration construction
- Different primes
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.topology import PAdicFiltration, TopologicalFingerprint


class TestPAdicFiltrationInit:
    """Tests for PAdicFiltration initialization."""

    def test_default_init(self):
        """Test default initialization."""
        filt = PAdicFiltration()
        assert filt.prime == 3
        assert filt.max_dimension == 1
        assert filt.max_valuation == 9

    def test_custom_prime(self):
        """Test custom prime."""
        filt = PAdicFiltration(prime=5)
        assert filt.prime == 5

    def test_custom_params(self):
        """Test custom parameters."""
        filt = PAdicFiltration(prime=2, max_dimension=2, max_valuation=10)
        assert filt.prime == 2
        assert filt.max_dimension == 2
        assert filt.max_valuation == 10


class TestPAdicValuation:
    """Tests for p-adic valuation computation."""

    def test_valuation_zero(self, padic_filtration):
        """Test valuation of zero."""
        v = padic_filtration._compute_valuation(0)
        assert v == padic_filtration.max_valuation

    def test_valuation_power_of_p(self, padic_filtration):
        """Test valuation of powers of p."""
        # p = 3
        assert padic_filtration._compute_valuation(3) == 1
        assert padic_filtration._compute_valuation(9) == 2
        assert padic_filtration._compute_valuation(27) == 3

    def test_valuation_non_divisible(self, padic_filtration):
        """Test valuation of numbers not divisible by p."""
        assert padic_filtration._compute_valuation(1) == 0
        assert padic_filtration._compute_valuation(2) == 0
        assert padic_filtration._compute_valuation(5) == 0

    def test_valuation_mixed(self, padic_filtration):
        """Test valuation of mixed numbers."""
        # 6 = 2 * 3, so v_3(6) = 1
        assert padic_filtration._compute_valuation(6) == 1
        # 18 = 2 * 9, so v_3(18) = 2
        assert padic_filtration._compute_valuation(18) == 2


class TestPAdicDistance:
    """Tests for p-adic distance computation."""

    def test_distance_to_self(self, padic_filtration):
        """Test distance to self is zero."""
        d = padic_filtration._padic_distance(5, 5)
        assert d == 0.0

    def test_distance_symmetric(self, padic_filtration):
        """Test distance is symmetric."""
        d12 = padic_filtration._padic_distance(3, 9)
        d21 = padic_filtration._padic_distance(9, 3)
        assert d12 == d21

    def test_distance_hierarchy(self, padic_filtration):
        """Test hierarchical distance property."""
        # Numbers differing by 3 are closer than those differing by 1
        # d_3(0, 3) < d_3(0, 1) because |0-3|=3 has v_3=1, |0-1|=1 has v_3=0
        d_3 = padic_filtration._padic_distance(0, 3)  # v_3(3) = 1, d = 3^(-1)
        d_1 = padic_filtration._padic_distance(0, 1)  # v_3(1) = 0, d = 3^0 = 1
        assert d_3 < d_1


class TestPAdicFiltrationBuild:
    """Tests for filtration building."""

    def test_build_from_numpy(self, padic_filtration, padic_indices):
        """Test building from numpy array."""
        fingerprint = padic_filtration.build(padic_indices)
        assert isinstance(fingerprint, TopologicalFingerprint)

    def test_build_from_tensor(self, padic_filtration, padic_indices_tensor):
        """Test building from torch tensor."""
        fingerprint = padic_filtration.build(padic_indices_tensor)
        assert isinstance(fingerprint, TopologicalFingerprint)

    def test_build_from_list(self, padic_filtration):
        """Test building from list."""
        indices = [0, 1, 3, 9]
        fingerprint = padic_filtration.build(indices)
        assert isinstance(fingerprint, TopologicalFingerprint)

    def test_empty_input(self, padic_filtration):
        """Test empty input."""
        fingerprint = padic_filtration.build([])
        assert fingerprint.total_features == 0


class TestPAdicFiltrationPrimes:
    """Tests for different primes."""

    @pytest.mark.parametrize("prime", [2, 3, 5, 7])
    def test_different_primes(self, prime):
        """Test with different primes."""
        filt = PAdicFiltration(prime=prime)
        indices = np.array([0, 1, prime, prime**2])
        fingerprint = filt.build(indices)
        assert isinstance(fingerprint, TopologicalFingerprint)

    def test_prime_2(self):
        """Test binary (prime=2) filtration."""
        filt = PAdicFiltration(prime=2)
        # Powers of 2 have increasing valuation
        assert filt._compute_valuation(2) == 1
        assert filt._compute_valuation(4) == 2
        assert filt._compute_valuation(8) == 3


class TestPAdicFiltrationMetadata:
    """Tests for metadata."""

    def test_metadata_exists(self, padic_filtration, padic_indices):
        """Test metadata is populated."""
        fingerprint = padic_filtration.build(padic_indices)
        assert fingerprint.metadata is not None
        assert "prime" in fingerprint.metadata

    def test_prime_in_metadata(self, padic_indices):
        """Test prime is recorded correctly."""
        filt = PAdicFiltration(prime=5)
        fingerprint = filt.build(padic_indices)
        assert fingerprint.metadata["prime"] == 5
