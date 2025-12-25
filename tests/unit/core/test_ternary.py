"""Tests for src/core/ternary.py - Ternary algebra operations.

This module tests the TernarySpace singleton which is the single source of truth
for all 3-adic operations in the codebase.
"""

import pytest
import torch

from src.core.ternary import (TERNARY, distance, from_ternary, to_ternary,
                              valuation)


class TestTernarySpaceConstants:
    """Test class constants are correct."""

    def test_n_digits(self):
        """Verify 9 trits per operation."""
        assert TERNARY.N_DIGITS == 9

    def test_n_operations(self):
        """Verify 3^9 = 19,683 total operations."""
        assert TERNARY.N_OPERATIONS == 19683
        assert TERNARY.N_OPERATIONS == 3**9

    def test_max_valuation(self):
        """Verify max valuation is 9 (for n=0)."""
        assert TERNARY.MAX_VALUATION == 9

    def test_ternary_values(self):
        """Verify valid trit values are -1, 0, 1."""
        assert TERNARY.TERNARY_VALUES == (-1, 0, 1)


class TestValuation:
    """Test 3-adic valuation computation."""

    def test_valuation_of_zero(self):
        """v_3(0) should be MAX_VALUATION (infinity in theory)."""
        indices = torch.tensor([0])
        v = TERNARY.valuation(indices)
        assert v.item() == TERNARY.MAX_VALUATION

    def test_valuation_of_one(self):
        """v_3(1) should be 0 (1 is not divisible by 3)."""
        indices = torch.tensor([1])
        v = TERNARY.valuation(indices)
        assert v.item() == 0

    def test_valuation_of_powers_of_three(self):
        """v_3(3^k) should be k."""
        for k in range(6):
            n = 3**k
            if n < TERNARY.N_OPERATIONS:
                v = TERNARY.valuation(torch.tensor([n]))
                assert v.item() == k, f"v_3({n}) should be {k}, got {v.item()}"

    def test_valuation_non_multiples_of_three(self):
        """v_3(n) = 0 when n is not divisible by 3."""
        non_multiples = [1, 2, 4, 5, 7, 8, 10, 11]
        indices = torch.tensor(non_multiples)
        v = TERNARY.valuation(indices)
        assert (v == 0).all(), "Non-multiples of 3 should have valuation 0"

    def test_valuation_batch(self):
        """Test valuation on batch of indices."""
        indices = torch.tensor([0, 1, 3, 9, 27, 81])
        expected = torch.tensor([9, 0, 1, 2, 3, 4])  # 0 gets MAX_VALUATION
        v = TERNARY.valuation(indices)
        assert torch.equal(v, expected)

    def test_valuation_shape_preserved(self):
        """Valuation should preserve input shape."""
        indices = torch.randint(0, 100, (4, 5, 6))
        v = TERNARY.valuation(indices)
        assert v.shape == indices.shape

    def test_valuation_clamping(self):
        """Indices out of range should be clamped."""
        indices = torch.tensor([-1, 0, TERNARY.N_OPERATIONS, TERNARY.N_OPERATIONS + 100])
        v = TERNARY.valuation(indices)
        # Should not raise, values should be within valid range
        assert v.shape == indices.shape


class TestDistance:
    """Test 3-adic distance computation."""

    def test_distance_identity(self):
        """d(i, i) = 0 for all i."""
        indices = torch.arange(100)
        d = TERNARY.distance(indices, indices)
        assert torch.allclose(d, torch.zeros_like(d))

    def test_distance_symmetry(self):
        """d(i, j) = d(j, i)."""
        i = torch.tensor([0, 1, 5, 10, 100])
        j = torch.tensor([1, 5, 10, 100, 1000])
        d_ij = TERNARY.distance(i, j)
        d_ji = TERNARY.distance(j, i)
        assert torch.allclose(d_ij, d_ji)

    def test_distance_adjacent(self):
        """d(0, 1) = 3^(-0) = 1 since v_3(|0-1|) = v_3(1) = 0."""
        d = TERNARY.distance(torch.tensor([0]), torch.tensor([1]))
        assert torch.isclose(d, torch.tensor([1.0]))

    def test_distance_difference_of_three(self):
        """d(0, 3) = 3^(-1) = 1/3 since v_3(|0-3|) = v_3(3) = 1."""
        d = TERNARY.distance(torch.tensor([0]), torch.tensor([3]))
        expected = torch.tensor([1.0 / 3.0])
        assert torch.isclose(d, expected)

    def test_distance_difference_of_nine(self):
        """d(0, 9) = 3^(-2) = 1/9 since v_3(|0-9|) = v_3(9) = 2."""
        d = TERNARY.distance(torch.tensor([0]), torch.tensor([9]))
        expected = torch.tensor([1.0 / 9.0])
        assert torch.isclose(d, expected)

    def test_distance_range(self):
        """Distance should be in [0, 1]."""
        i = torch.randint(0, TERNARY.N_OPERATIONS, (100,))
        j = torch.randint(0, TERNARY.N_OPERATIONS, (100,))
        d = TERNARY.distance(i, j)
        assert (d >= 0).all() and (d <= 1).all()

    def test_ultrametric_inequality(self):
        """3-adic distance satisfies ultrametric: d(x,z) <= max(d(x,y), d(y,z))."""
        # Sample random triplets
        torch.manual_seed(42)
        n = 50
        x = torch.randint(0, TERNARY.N_OPERATIONS, (n,))
        y = torch.randint(0, TERNARY.N_OPERATIONS, (n,))
        z = torch.randint(0, TERNARY.N_OPERATIONS, (n,))

        d_xy = TERNARY.distance(x, y)
        d_yz = TERNARY.distance(y, z)
        d_xz = TERNARY.distance(x, z)

        max_d = torch.max(d_xy, d_yz)
        assert (d_xz <= max_d + 1e-6).all(), "Ultrametric inequality violated"


class TestTernaryConversion:
    """Test ternary <-> index conversion."""

    def test_to_ternary_index_zero(self):
        """Index 0 should map to all -1s (since 0 in base-3 is 000...0 -> -1,-1,...,-1)."""
        ternary = TERNARY.to_ternary(torch.tensor([0]))
        assert ternary.shape == (1, 9)
        assert (ternary == -1).all()

    def test_to_ternary_index_one(self):
        """Index 1 should map to [0, -1, -1, ..., -1] (1 in base-3)."""
        ternary = TERNARY.to_ternary(torch.tensor([1]))
        # First digit is (1 % 3) - 1 = 0
        assert ternary[0, 0] == 0
        assert (ternary[0, 1:] == -1).all()

    def test_roundtrip_conversion(self):
        """from_ternary(to_ternary(i)) == i for all indices."""
        indices = torch.arange(100)
        ternary = TERNARY.to_ternary(indices)
        recovered = TERNARY.from_ternary(ternary)
        assert torch.equal(recovered, indices)

    def test_roundtrip_all_indices(self):
        """Roundtrip works for all 19,683 indices."""
        all_indices = torch.arange(TERNARY.N_OPERATIONS)
        ternary = TERNARY.to_ternary(all_indices)
        recovered = TERNARY.from_ternary(ternary)
        assert torch.equal(recovered, all_indices)

    def test_ternary_values_valid(self):
        """All ternary representations should have values in {-1, 0, 1}."""
        all_ternary = TERNARY.all_ternary()
        assert ((all_ternary == -1) | (all_ternary == 0) | (all_ternary == 1)).all()


class TestConvenienceMethods:
    """Test convenience methods."""

    def test_is_valid_index(self):
        """Test valid index checking."""
        valid = torch.tensor([0, 1, 100, TERNARY.N_OPERATIONS - 1])
        invalid = torch.tensor([-1, TERNARY.N_OPERATIONS, 99999])

        assert TERNARY.is_valid_index(valid).all()
        assert not TERNARY.is_valid_index(invalid).any()

    def test_is_valid_ternary(self):
        """Test valid ternary checking."""
        valid = torch.tensor([[0, 1, -1, 0, 1, -1, 0, 1, -1]]).float()
        invalid = torch.tensor([[0, 1, 2, 0, 1, -1, 0, 1, -1]]).float()  # 2 is invalid

        assert TERNARY.is_valid_ternary(valid).all()
        assert not TERNARY.is_valid_ternary(invalid).all()

    def test_sample_indices(self):
        """Test random index sampling."""
        n = 100
        samples = TERNARY.sample_indices(n)
        assert samples.shape == (n,)
        assert TERNARY.is_valid_index(samples).all()

    def test_all_indices(self):
        """Test getting all indices."""
        all_idx = TERNARY.all_indices()
        assert all_idx.shape == (TERNARY.N_OPERATIONS,)
        assert torch.equal(all_idx, torch.arange(TERNARY.N_OPERATIONS))

    def test_all_ternary_shape(self):
        """Test getting all ternary representations."""
        all_t = TERNARY.all_ternary()
        assert all_t.shape == (TERNARY.N_OPERATIONS, TERNARY.N_DIGITS)


class TestBatchOperations:
    """Test batch operations for GPU efficiency."""

    def test_prefix_level_zero(self):
        """Level 0 prefix should map all to root (0)."""
        indices = torch.arange(100)
        prefixes = TERNARY.prefix(indices, 0)
        assert (prefixes == 0).all()

    def test_prefix_level_max(self):
        """Level 9 prefix should be identity."""
        indices = torch.arange(100)
        prefixes = TERNARY.prefix(indices, TERNARY.N_DIGITS)
        assert torch.equal(prefixes, indices)

    def test_prefix_grouping(self):
        """Indices with same prefix at level k share first k digits."""
        # At level 1, indices 0-2 should have same prefix
        indices = torch.tensor([0, 1, 2, 3, 4, 5])
        prefixes = TERNARY.prefix(indices, 1)
        # 0,1,2 differ only in last 8 digits when level=1
        # Actually, prefix at level k = index // 3^(9-k)
        # level=1: divisor = 3^8 = 6561
        # All indices 0-5 should map to prefix 0
        assert (prefixes == 0).all()

    def test_level_mask(self):
        """Test level mask computation."""
        # Indices with valuation = level should match
        indices = torch.tensor([0, 1, 3, 9, 27, 81])
        valuations = torch.tensor([9, 0, 1, 2, 3, 4])

        for level in range(5):
            mask = TERNARY.level_mask(indices, level)
            expected = valuations == level
            assert torch.equal(mask, expected)


class TestAnalysisMethods:
    """Test analysis methods."""

    def test_valuation_histogram(self):
        """Test histogram computation."""
        indices = torch.tensor([0, 1, 3, 9, 27, 81, 2, 4, 5])
        hist = TERNARY.valuation_histogram(indices)

        assert hist[0] == 4  # 1, 2, 4, 5 have valuation 0
        assert hist[1] == 1  # 3 has valuation 1
        assert hist[2] == 1  # 9 has valuation 2
        assert hist[3] == 1  # 27 has valuation 3
        assert hist[4] == 1  # 81 has valuation 4
        assert hist[9] == 1  # 0 has valuation 9

    def test_expected_valuation(self):
        """Expected valuation should be positive and less than MAX_VALUATION."""
        ev = TERNARY.expected_valuation()
        assert 0 < ev < TERNARY.MAX_VALUATION


class TestModuleFunctions:
    """Test module-level convenience functions."""

    def test_valuation_function(self):
        """Module function should delegate to TERNARY."""
        indices = torch.tensor([0, 1, 3, 9])
        assert torch.equal(valuation(indices), TERNARY.valuation(indices))

    def test_distance_function(self):
        """Module function should delegate to TERNARY."""
        i = torch.tensor([0, 1, 2])
        j = torch.tensor([1, 2, 3])
        assert torch.allclose(distance(i, j), TERNARY.distance(i, j))

    def test_to_ternary_function(self):
        """Module function should delegate to TERNARY."""
        indices = torch.tensor([0, 1, 2, 3])
        assert torch.equal(to_ternary(indices), TERNARY.to_ternary(indices))

    def test_from_ternary_function(self):
        """Module function should delegate to TERNARY."""
        ternary = torch.zeros(5, 9)
        assert torch.equal(from_ternary(ternary), TERNARY.from_ternary(ternary))


class TestDeviceCaching:
    """Test device caching behavior."""

    def test_cpu_operations(self):
        """Operations should work on CPU."""
        indices = torch.tensor([0, 1, 2, 3], device="cpu")
        v = TERNARY.valuation(indices)
        assert v.device == indices.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_operations(self):
        """Operations should work on CUDA."""
        indices = torch.tensor([0, 1, 2, 3], device="cuda")
        v = TERNARY.valuation(indices)
        assert v.device == indices.device

    def test_cache_reuse(self):
        """Same device should reuse cached LUT."""
        indices1 = torch.tensor([0, 1, 2])
        indices2 = torch.tensor([3, 4, 5])

        # First call populates cache
        TERNARY.valuation(indices1)

        # Second call should reuse
        TERNARY.valuation(indices2)

        # Cache should have entry for CPU
        assert "valuation_cpu" in TERNARY._device_cache


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_tensor(self):
        """Operations should handle empty tensors."""
        empty = torch.tensor([], dtype=torch.long)
        v = TERNARY.valuation(empty)
        assert v.shape == (0,)

    def test_single_element(self):
        """Operations should handle single elements."""
        single = torch.tensor([42])
        v = TERNARY.valuation(single)
        assert v.shape == (1,)

    def test_large_batch(self):
        """Operations should handle large batches efficiently."""
        large = torch.randint(0, TERNARY.N_OPERATIONS, (10000,))
        v = TERNARY.valuation(large)
        assert v.shape == large.shape

    def test_multidimensional_input(self):
        """Operations should preserve arbitrary dimensions."""
        md = torch.randint(0, TERNARY.N_OPERATIONS, (2, 3, 4, 5))
        v = TERNARY.valuation(md)
        assert v.shape == md.shape

        t = TERNARY.to_ternary(md.flatten())
        assert t.shape == (2 * 3 * 4 * 5, 9)
