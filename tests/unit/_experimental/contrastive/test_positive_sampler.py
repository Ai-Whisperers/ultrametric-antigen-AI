# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for PAdicPositiveSampler class."""

from __future__ import annotations

import pytest
import torch

from src.contrastive import PAdicPositiveSampler


class TestPositiveSamplerInit:
    """Tests for PAdicPositiveSampler initialization."""

    def test_default_init(self):
        """Test default initialization."""
        sampler = PAdicPositiveSampler()
        assert sampler.min_valuation == 2
        assert sampler.max_valuation == 9
        assert sampler.prime == 3

    def test_custom_params(self):
        """Test custom parameters."""
        sampler = PAdicPositiveSampler(min_valuation=1, max_valuation=6, prime=5)
        assert sampler.min_valuation == 1
        assert sampler.max_valuation == 6
        assert sampler.prime == 5


class TestPositiveCandidates:
    """Tests for positive candidate selection."""

    def test_candidates_shape(self, positive_sampler, padic_indices):
        """Test candidates mask shape."""
        candidates = positive_sampler.get_positive_candidates(0, padic_indices)
        assert candidates.shape == padic_indices.shape

    def test_excludes_anchor(self, positive_sampler, padic_indices):
        """Test anchor is not in candidates."""
        candidates = positive_sampler.get_positive_candidates(0, padic_indices)
        assert not candidates[0].item()

    def test_finds_positives(self, device):
        """Test finds p-adically close indices."""
        sampler = PAdicPositiveSampler(min_valuation=1, prime=3)
        # 0, 3, 6, 9 are all divisible by 3
        indices = torch.tensor([0, 3, 6, 9, 1, 2], device=device)

        candidates = sampler.get_positive_candidates(0, indices)

        # Indices 1, 2, 3 (values 3, 6, 9) should be candidates
        assert candidates[1].item()  # 3
        assert candidates[2].item()  # 6
        assert candidates[3].item()  # 9
        # Indices 4, 5 (values 1, 2) should not be candidates
        assert not candidates[4].item()
        assert not candidates[5].item()

    def test_higher_threshold(self, device):
        """Test with higher valuation threshold."""
        sampler = PAdicPositiveSampler(min_valuation=2, prime=3)
        indices = torch.tensor([0, 3, 9, 27, 1], device=device)

        candidates = sampler.get_positive_candidates(0, indices)

        # Only 9 and 27 are divisible by 9 (valuation >= 2)
        assert not candidates[1].item()  # 3 has valuation 1
        assert candidates[2].item()  # 9 has valuation 2
        assert candidates[3].item()  # 27 has valuation 3


class TestSamplePositive:
    """Tests for sampling single positive."""

    def test_returns_valid_index(self, positive_sampler, device):
        """Test returns valid index when positives exist."""
        indices = torch.tensor([0, 3, 6, 9], device=device)
        pos = positive_sampler.sample_positive(0, indices)
        assert pos in [1, 2, 3]

    def test_returns_none_when_no_positives(self, device):
        """Test returns None when no positives exist."""
        sampler = PAdicPositiveSampler(min_valuation=5, prime=3)
        indices = torch.tensor([1, 2, 4, 5], device=device)
        pos = sampler.sample_positive(0, indices)
        assert pos is None

    def test_never_returns_anchor(self, positive_sampler, device):
        """Test never returns anchor index."""
        indices = torch.tensor([0, 3, 6, 9], device=device)
        for _ in range(20):
            pos = positive_sampler.sample_positive(0, indices)
            assert pos != 0


class TestDifferentPrimes:
    """Tests for different primes."""

    @pytest.mark.parametrize("prime", [2, 3, 5, 7])
    def test_prime_specific_candidates(self, prime, device):
        """Test candidates are prime-specific."""
        sampler = PAdicPositiveSampler(min_valuation=1, prime=prime)
        indices = torch.tensor([0, prime, prime * 2, prime * 3, 1], device=device)

        candidates = sampler.get_positive_candidates(0, indices)

        # Multiples of prime should be candidates
        assert candidates[1].item()
        assert candidates[2].item()
        assert candidates[3].item()
        # 1 should not be candidate
        assert not candidates[4].item()
