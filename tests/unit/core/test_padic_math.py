# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for src/core/padic_math.py - P-adic mathematics module.

This module provides comprehensive tests for:
- padic_valuation: Computing p-adic valuations
- padic_norm: Computing p-adic norms
- padic_distance: Computing p-adic distances
- padic_digits: Computing p-adic digit expansions
- padic_shift: P-adic shift operations
- Vectorized operations (GPU-optimized tensor functions)
- Goldilocks zone scoring (autoimmune risk)
- Hierarchical embeddings
"""

import pytest
import torch
import numpy as np

from src.core.padic_math import (
    DEFAULT_P,
    PADIC_INFINITY,
    PADIC_INFINITY_INT,
    padic_valuation,
    padic_norm,
    padic_distance,
    padic_digits,
    padic_shift,
    PAdicShiftResult,
    padic_valuation_vectorized,
    padic_distance_vectorized,
    padic_distance_matrix,
    padic_distance_batch,
    compute_goldilocks_score,
    is_in_goldilocks_zone,
    compute_goldilocks_tensor,
    compute_hierarchical_embedding,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Returns 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_default_p_is_3(self):
        """Default prime should be 3 for ternary."""
        assert DEFAULT_P == 3

    def test_padic_infinity_int(self):
        """Infinity integer representation should be 100."""
        assert PADIC_INFINITY_INT == 100

    def test_padic_infinity_float(self):
        """Infinity float representation should be inf."""
        assert PADIC_INFINITY == float("inf")


# =============================================================================
# Padic Valuation Tests
# =============================================================================


class TestPadicValuation:
    """Tests for padic_valuation function."""

    def test_valuation_of_zero_returns_infinity(self):
        """v_p(0) should return infinity (100)."""
        result = padic_valuation(0, 3)
        assert result == PADIC_INFINITY_INT

    def test_valuation_of_one_is_zero(self):
        """v_p(1) = 0 for any p (1 is not divisible by any prime)."""
        assert padic_valuation(1, 3) == 0
        assert padic_valuation(1, 5) == 0
        assert padic_valuation(1, 7) == 0

    def test_valuation_powers_of_p(self):
        """v_p(p^k) = k."""
        # 3-adic valuations
        assert padic_valuation(3, 3) == 1
        assert padic_valuation(9, 3) == 2
        assert padic_valuation(27, 3) == 3
        assert padic_valuation(81, 3) == 4
        assert padic_valuation(243, 3) == 5

    def test_valuation_multiples_of_p(self):
        """v_3(3k) = 1 + v_3(k) when k is not divisible by 3."""
        assert padic_valuation(6, 3) == 1  # 6 = 2 * 3
        assert padic_valuation(15, 3) == 1  # 15 = 5 * 3
        assert padic_valuation(18, 3) == 2  # 18 = 2 * 9

    def test_valuation_not_divisible_by_p(self):
        """v_p(n) = 0 when n is not divisible by p."""
        assert padic_valuation(5, 3) == 0
        assert padic_valuation(7, 3) == 0
        assert padic_valuation(11, 3) == 0

    def test_valuation_negative_numbers(self):
        """Valuation should work for negative numbers (absolute value)."""
        assert padic_valuation(-9, 3) == 2
        assert padic_valuation(-27, 3) == 3

    @pytest.mark.parametrize(
        "n,p,expected",
        [
            (1, 3, 0),
            (3, 3, 1),
            (9, 3, 2),
            (27, 3, 3),
            (2, 3, 0),
            (6, 3, 1),
            (18, 3, 2),
            (0, 3, 100),
            (5, 5, 1),
            (25, 5, 2),
            (2, 2, 1),
            (8, 2, 3),
        ],
    )
    def test_valuation_parametrized(self, n: int, p: int, expected: int):
        """Parametrized tests for various valuations."""
        assert padic_valuation(n, p) == expected


# =============================================================================
# Padic Norm Tests
# =============================================================================


class TestPadicNorm:
    """Tests for padic_norm function."""

    def test_norm_of_zero_is_zero(self):
        """|0|_p = 0."""
        assert padic_norm(0, 3) == 0.0

    def test_norm_of_one_is_one(self):
        """|1|_p = 1."""
        assert padic_norm(1, 3) == pytest.approx(1.0)

    def test_norm_of_p_is_inverse_p(self):
        """|p|_p = 1/p."""
        assert padic_norm(3, 3) == pytest.approx(1 / 3)
        assert padic_norm(5, 5) == pytest.approx(1 / 5)

    def test_norm_of_p_squared(self):
        """|p^2|_p = 1/p^2."""
        assert padic_norm(9, 3) == pytest.approx(1 / 9)
        assert padic_norm(25, 5) == pytest.approx(1 / 25)

    def test_norm_not_divisible_by_p(self):
        """|n|_p = 1 when n is not divisible by p."""
        assert padic_norm(2, 3) == pytest.approx(1.0)
        assert padic_norm(5, 3) == pytest.approx(1.0)
        assert padic_norm(7, 3) == pytest.approx(1.0)

    def test_norm_multiplicative(self):
        """|ab|_p = |a|_p * |b|_p."""
        # |6|_3 = |2|_3 * |3|_3 = 1 * (1/3) = 1/3
        assert padic_norm(6, 3) == pytest.approx(padic_norm(2, 3) * padic_norm(3, 3))


# =============================================================================
# Padic Distance Tests
# =============================================================================


class TestPadicDistance:
    """Tests for padic_distance function."""

    def test_distance_to_self_is_zero(self):
        """d_p(a, a) = 0."""
        assert padic_distance(5, 5, 3) == 0.0
        assert padic_distance(0, 0, 3) == 0.0
        assert padic_distance(27, 27, 3) == 0.0

    def test_distance_symmetry(self):
        """d_p(a, b) = d_p(b, a)."""
        assert padic_distance(5, 14, 3) == padic_distance(14, 5, 3)
        assert padic_distance(0, 27, 3) == padic_distance(27, 0, 3)

    def test_distance_known_values(self):
        """Test known distance values."""
        # d_3(0, 9) = |9|_3 = 1/9
        assert padic_distance(0, 9, 3) == pytest.approx(1 / 9)
        # d_3(1, 4) = |3|_3 = 1/3
        assert padic_distance(1, 4, 3) == pytest.approx(1 / 3)
        # d_3(0, 27) = |27|_3 = 1/27
        assert padic_distance(0, 27, 3) == pytest.approx(1 / 27)

    def test_ultrametric_inequality(self):
        """d(a,c) <= max(d(a,b), d(b,c)) - ultrametric property."""
        a, b, c = 0, 9, 27
        d_ac = padic_distance(a, c, 3)
        d_ab = padic_distance(a, b, 3)
        d_bc = padic_distance(b, c, 3)
        assert d_ac <= max(d_ab, d_bc) + 1e-10  # Small epsilon for floating point


# =============================================================================
# Padic Digits Tests
# =============================================================================


class TestPadicDigits:
    """Tests for padic_digits function."""

    def test_digits_of_zero(self):
        """0 should have all zero digits."""
        assert padic_digits(0, 3, 4) == [0, 0, 0, 0]

    def test_digits_of_small_numbers(self):
        """Test digit expansion of small numbers."""
        # 10 = 1 + 0*3 + 1*9 = [1, 0, 1, 0]
        assert padic_digits(10, 3, 4) == [1, 0, 1, 0]
        # 5 = 2 + 1*3 = [2, 1, 0, 0]
        assert padic_digits(5, 3, 4) == [2, 1, 0, 0]

    def test_digits_binary(self):
        """Test binary (2-adic) expansion."""
        # 5 = 1 + 0*2 + 1*4 = [1, 0, 1, 0]
        assert padic_digits(5, 2, 4) == [1, 0, 1, 0]
        # 7 = 1 + 1*2 + 1*4 = [1, 1, 1, 0]
        assert padic_digits(7, 2, 4) == [1, 1, 1, 0]

    def test_digits_reconstruction(self):
        """Digits should reconstruct the original number."""
        n = 42
        digits = padic_digits(n, 3, 6)
        reconstructed = sum(d * (3**i) for i, d in enumerate(digits))
        assert reconstructed == n

    def test_digits_negative_uses_absolute_value(self):
        """Negative numbers should use absolute value."""
        assert padic_digits(-10, 3, 4) == padic_digits(10, 3, 4)


# =============================================================================
# Padic Shift Tests
# =============================================================================


class TestPadicShift:
    """Tests for padic_shift function."""

    def test_shift_result_type(self):
        """Should return PAdicShiftResult."""
        result = padic_shift(27, 1, 3)
        assert isinstance(result, PAdicShiftResult)

    def test_right_shift_divides_by_p(self):
        """Right shift (positive) divides by p."""
        result = padic_shift(27, 1, 3)
        assert result.shift_value == 9.0  # 27 / 3 = 9

        result = padic_shift(27, 2, 3)
        assert result.shift_value == 3.0  # 27 / 9 = 3

    def test_left_shift_multiplies_by_p(self):
        """Left shift (negative) multiplies by p."""
        result = padic_shift(3, -1, 3)
        assert result.shift_value == 9.0  # 3 * 3 = 9

        result = padic_shift(3, -2, 3)
        assert result.shift_value == 27.0  # 3 * 9 = 27

    def test_shift_valuation_correct(self):
        """Shifted value should have correct valuation."""
        result = padic_shift(27, 0, 3)
        assert result.valuation == 3  # v_3(27) = 3

    def test_shift_canonical_form(self):
        """Canonical form should be formatted correctly."""
        result = padic_shift(9, 0, 3)
        assert result.canonical_form == "9_(3)"


# =============================================================================
# Vectorized Valuation Tests
# =============================================================================


class TestPadicValuationVectorized:
    """Tests for padic_valuation_vectorized function."""

    def test_returns_tensor(self, device):
        """Should return a tensor."""
        diff = torch.tensor([1, 3, 9, 27], device=device)
        result = padic_valuation_vectorized(diff, 3)
        assert isinstance(result, torch.Tensor)

    def test_known_valuations(self, device):
        """Test known valuation values."""
        diff = torch.tensor([1, 3, 9, 27, 81], device=device)
        result = padic_valuation_vectorized(diff, 3)
        expected = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], device=device)
        torch.testing.assert_close(result, expected)

    def test_zero_has_max_valuation(self, device):
        """Zero should have the maximum valuation checked."""
        diff = torch.tensor([0, 0, 0], device=device)
        result = padic_valuation_vectorized(diff, 3, max_depth=10)
        # Zero stays at 0 because (0 % p^k == 0) but (0 > 0) is False
        expected = torch.tensor([0.0, 0.0, 0.0], device=device)
        torch.testing.assert_close(result, expected)

    def test_batch_processing(self, device):
        """Should work with 2D tensors."""
        diff = torch.tensor([[1, 3], [9, 27]], device=device)
        result = padic_valuation_vectorized(diff, 3)
        expected = torch.tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
        torch.testing.assert_close(result, expected)


# =============================================================================
# Vectorized Distance Tests
# =============================================================================


class TestPadicDistanceVectorized:
    """Tests for padic_distance_vectorized function."""

    def test_equal_elements_zero_distance(self, device):
        """Equal elements should have zero distance."""
        a = torch.tensor([1, 2, 3], device=device)
        b = torch.tensor([1, 2, 3], device=device)
        result = padic_distance_vectorized(a, b, 3)
        expected = torch.zeros(3, device=device)
        torch.testing.assert_close(result, expected)

    def test_known_distances(self, device):
        """Test known distance values."""
        a = torch.tensor([0, 0, 0], device=device)
        b = torch.tensor([3, 9, 27], device=device)
        result = padic_distance_vectorized(a, b, 3)
        expected = torch.tensor([1 / 3, 1 / 9, 1 / 27], device=device)
        torch.testing.assert_close(result, expected)

    def test_symmetry(self, device):
        """d(a, b) = d(b, a)."""
        a = torch.tensor([0, 5, 10], device=device)
        b = torch.tensor([9, 14, 19], device=device)
        d_ab = padic_distance_vectorized(a, b, 3)
        d_ba = padic_distance_vectorized(b, a, 3)
        torch.testing.assert_close(d_ab, d_ba)


# =============================================================================
# Distance Matrix Tests
# =============================================================================


class TestPadicDistanceMatrix:
    """Tests for padic_distance_matrix function."""

    def test_matrix_shape(self, device):
        """Result should be (n, n)."""
        indices = torch.tensor([0, 3, 9, 27], device=device)
        result = padic_distance_matrix(indices, 3)
        assert result.shape == (4, 4)

    def test_diagonal_is_zero(self, device):
        """Diagonal should be all zeros (d(a,a) = 0)."""
        indices = torch.tensor([0, 3, 9, 27], device=device)
        result = padic_distance_matrix(indices, 3)
        diagonal = torch.diagonal(result)
        expected = torch.zeros(4, device=device)
        torch.testing.assert_close(diagonal, expected)

    def test_symmetry(self, device):
        """Matrix should be symmetric."""
        indices = torch.tensor([0, 3, 9, 27], device=device)
        result = padic_distance_matrix(indices, 3)
        torch.testing.assert_close(result, result.T)

    def test_known_values(self, device):
        """Test known distance values."""
        indices = torch.tensor([0, 9], device=device)
        result = padic_distance_matrix(indices, 3)
        # d(0, 9) = |9|_3 = 1/9
        assert result[0, 1].item() == pytest.approx(1 / 9)
        assert result[1, 0].item() == pytest.approx(1 / 9)


# =============================================================================
# Distance Batch Tests
# =============================================================================


class TestPadicDistanceBatch:
    """Tests for padic_distance_batch function."""

    def test_batch_shape(self, device):
        """Result should be (batch, seq, seq)."""
        indices = torch.tensor([[0, 3, 9], [1, 4, 27]], device=device)
        precomputed = padic_distance_matrix(torch.arange(64, device=device), 3)
        result = padic_distance_batch(indices, precomputed)
        assert result.shape == (2, 3, 3)

    def test_uses_precomputed_matrix(self, device):
        """Should use the precomputed matrix values."""
        indices = torch.tensor([[0, 9]], device=device)
        precomputed = padic_distance_matrix(torch.arange(64, device=device), 3)
        result = padic_distance_batch(indices, precomputed)
        # d(0, 9) = 1/9
        assert result[0, 0, 1].item() == pytest.approx(1 / 9)

    def test_diagonal_is_zero(self, device):
        """Diagonal of each batch element should be zero."""
        indices = torch.tensor([[0, 3, 9], [1, 4, 27]], device=device)
        precomputed = padic_distance_matrix(torch.arange(64, device=device), 3)
        result = padic_distance_batch(indices, precomputed)
        for i in range(2):
            diagonal = torch.diagonal(result[i])
            expected = torch.zeros(3, device=device)
            torch.testing.assert_close(diagonal, expected)


# =============================================================================
# Goldilocks Zone Tests
# =============================================================================


class TestGoldilocksScore:
    """Tests for compute_goldilocks_score function."""

    def test_center_has_max_score(self):
        """Distance at center should have score of 1."""
        score = compute_goldilocks_score(0.5, center=0.5, width=0.15)
        assert score == pytest.approx(1.0)

    def test_far_from_center_low_score(self):
        """Distance far from center should have low score."""
        score_low = compute_goldilocks_score(0.0, center=0.5, width=0.15)
        score_high = compute_goldilocks_score(1.0, center=0.5, width=0.15)
        assert score_low < 0.1
        assert score_high < 0.1

    def test_score_between_zero_and_one(self):
        """Score should always be in [0, 1]."""
        for distance in [0.0, 0.25, 0.5, 0.75, 1.0]:
            score = compute_goldilocks_score(distance, center=0.5, width=0.15)
            assert 0.0 <= score <= 1.0

    def test_gaussian_shape(self):
        """Score should decrease with distance from center."""
        center = 0.5
        scores = [compute_goldilocks_score(d, center=center, width=0.15) for d in [0.5, 0.4, 0.3, 0.2]]
        # Each score should be <= previous
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1]

    def test_width_affects_spread(self):
        """Larger width should give higher scores away from center."""
        narrow = compute_goldilocks_score(0.7, center=0.5, width=0.1)
        wide = compute_goldilocks_score(0.7, center=0.5, width=0.3)
        assert wide > narrow


class TestIsInGoldilocksZone:
    """Tests for is_in_goldilocks_zone function."""

    def test_center_is_in_zone(self):
        """Distance at center should be in zone."""
        assert is_in_goldilocks_zone(0.5, center=0.5, width=0.15) is True

    def test_far_from_center_not_in_zone(self):
        """Distance far from center should not be in zone."""
        assert is_in_goldilocks_zone(0.0, center=0.5, width=0.15) is False
        assert is_in_goldilocks_zone(1.0, center=0.5, width=0.15) is False

    def test_threshold_affects_boundary(self):
        """Higher threshold should be stricter."""
        distance = 0.6
        low_threshold = is_in_goldilocks_zone(distance, center=0.5, width=0.15, threshold=0.3)
        high_threshold = is_in_goldilocks_zone(distance, center=0.5, width=0.15, threshold=0.9)
        assert low_threshold is True
        assert high_threshold is False


class TestGoldilocksScoreTensor:
    """Tests for compute_goldilocks_tensor function."""

    def test_returns_tensor(self, device):
        """Should return a tensor."""
        distances = torch.tensor([0.3, 0.5, 0.7], device=device)
        result = compute_goldilocks_tensor(distances, center=0.5, width=0.15)
        assert isinstance(result, torch.Tensor)

    def test_center_has_max_score(self, device):
        """Center distance should have score of 1."""
        distances = torch.tensor([0.5], device=device)
        result = compute_goldilocks_tensor(distances, center=0.5, width=0.15)
        assert result[0].item() == pytest.approx(1.0)

    def test_batch_processing(self, device):
        """Should work with batched tensors."""
        distances = torch.tensor([[0.3, 0.5], [0.7, 0.5]], device=device)
        result = compute_goldilocks_tensor(distances, center=0.5, width=0.15)
        assert result.shape == (2, 2)
        # Centers should be 1.0
        assert result[0, 1].item() == pytest.approx(1.0)
        assert result[1, 1].item() == pytest.approx(1.0)


# =============================================================================
# Hierarchical Embedding Tests
# =============================================================================


class TestHierarchicalEmbedding:
    """Tests for compute_hierarchical_embedding function."""

    def test_output_shape(self, device):
        """Output should have n_digits as last dimension."""
        indices = torch.tensor([0, 10, 27], device=device)
        result = compute_hierarchical_embedding(indices, n_digits=9, p=3)
        assert result.shape == (3, 9)

    def test_zero_embedding(self, device):
        """Zero should have all zero digits."""
        indices = torch.tensor([0], device=device)
        result = compute_hierarchical_embedding(indices, n_digits=4, p=3)
        expected = torch.zeros(1, 4, device=device)
        torch.testing.assert_close(result, expected)

    def test_digit_values(self, device):
        """Digits should match p-adic expansion."""
        indices = torch.tensor([10], device=device)  # 10 = 1 + 0*3 + 1*9
        result = compute_hierarchical_embedding(indices, n_digits=4, p=3)
        expected = torch.tensor([[1.0, 0.0, 1.0, 0.0]], device=device)
        torch.testing.assert_close(result, expected)

    def test_reconstruction(self, device):
        """Embedding should reconstruct original index."""
        indices = torch.tensor([42], device=device)
        embedding = compute_hierarchical_embedding(indices, n_digits=6, p=3)
        # Reconstruct: sum of d * p^i
        powers = torch.tensor([3.0**i for i in range(6)], device=device)
        reconstructed = (embedding * powers).sum(dim=-1)
        torch.testing.assert_close(reconstructed, indices.float())

    def test_batch_processing(self, device):
        """Should work with batched input."""
        indices = torch.tensor([[0, 10], [27, 42]], device=device)
        result = compute_hierarchical_embedding(indices, n_digits=4, p=3)
        assert result.shape == (2, 2, 4)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPadicMathIntegration:
    """Integration tests for p-adic math module."""

    def test_scalar_and_vectorized_consistency(self, device):
        """Scalar and vectorized functions should give same results."""
        pairs = [(0, 9), (1, 4), (0, 27)]

        for a, b in pairs:
            scalar_dist = padic_distance(a, b, 3)
            tensor_a = torch.tensor([a], device=device)
            tensor_b = torch.tensor([b], device=device)
            vectorized_dist = padic_distance_vectorized(tensor_a, tensor_b, 3)[0].item()
            assert scalar_dist == pytest.approx(vectorized_dist)

    def test_distance_matrix_consistency(self, device):
        """Distance matrix should be consistent with pairwise computation."""
        indices = [0, 3, 9, 27]
        tensor_indices = torch.tensor(indices, device=device)

        matrix = padic_distance_matrix(tensor_indices, 3)

        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices):
                expected = padic_distance(idx_i, idx_j, 3)
                actual = matrix[i, j].item()
                assert actual == pytest.approx(expected)

    def test_goldilocks_zone_biological_relevance(self):
        """Goldilocks zone should identify mid-range distances."""
        # Low distance: self-like
        low_score = compute_goldilocks_score(0.1, center=0.5, width=0.15)
        # High distance: clearly foreign
        high_score = compute_goldilocks_score(0.9, center=0.5, width=0.15)
        # Middle distance: potentially cross-reactive
        mid_score = compute_goldilocks_score(0.5, center=0.5, width=0.15)

        assert mid_score > low_score
        assert mid_score > high_score

    def test_hierarchical_embedding_preserves_structure(self, device):
        """Close indices should have similar embeddings."""
        # 0 and 3 differ in first digit only
        indices = torch.tensor([0, 3, 9, 27], device=device)
        embeddings = compute_hierarchical_embedding(indices, n_digits=4, p=3)

        # Compute pairwise L2 distances in embedding space
        diff_01 = torch.norm(embeddings[0] - embeddings[1]).item()
        diff_02 = torch.norm(embeddings[0] - embeddings[2]).item()
        diff_03 = torch.norm(embeddings[0] - embeddings[3]).item()

        # 0 and 3 should be closest (differ in 1 digit)
        # All should differ from 0 (which is all zeros)
        assert diff_01 > 0  # They differ

    def test_full_pipeline(self, device):
        """Test complete pipeline from indices to distances to Goldilocks."""
        # Create indices
        indices = torch.arange(10, device=device)

        # Compute distance matrix
        dist_matrix = padic_distance_matrix(indices, 3)

        # Compute Goldilocks scores
        goldilocks = compute_goldilocks_tensor(dist_matrix, center=0.5, width=0.15)

        # All scores should be valid
        assert torch.all(goldilocks >= 0)
        assert torch.all(goldilocks <= 1)

        # Diagonal should have high scores (d=0, mapped to ~0 after normalization)
        # Actually diagonal is 0, which is far from center 0.5
        diagonal_scores = torch.diagonal(goldilocks)
        assert torch.all(diagonal_scores < 0.5)  # 0 is far from 0.5


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_large_numbers(self):
        """Should handle large numbers."""
        large = 3**20  # Very large power of 3
        assert padic_valuation(large, 3) == 20

    def test_prime_as_base(self):
        """Should work with different primes."""
        assert padic_valuation(25, 5) == 2
        assert padic_valuation(49, 7) == 2
        assert padic_valuation(121, 11) == 2

    def test_empty_tensor(self, device):
        """Should handle empty tensors gracefully."""
        indices = torch.tensor([], dtype=torch.long, device=device)
        result = padic_distance_matrix(indices, 3)
        assert result.shape == (0, 0)

    def test_single_element(self, device):
        """Should handle single element tensors."""
        indices = torch.tensor([42], device=device)
        result = padic_distance_matrix(indices, 3)
        assert result.shape == (1, 1)
        assert result[0, 0] == 0.0  # d(a, a) = 0

    def test_negative_shift_amount(self):
        """Should handle negative shift amounts."""
        result = padic_shift(3, -3, 3)
        assert result.shift_value == 81.0  # 3 * 27 = 81

    def test_zero_shift(self):
        """Zero shift should return original value."""
        result = padic_shift(42, 0, 3)
        assert result.shift_value == 42.0
