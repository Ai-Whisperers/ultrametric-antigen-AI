# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for citrullination boundary optimizer."""

import pytest
import torch

from src.analysis.codon_optimization import (
    CitrullinationBoundaryOptimizer,
    CodonChoice,
    CodonContextOptimizer,
    OptimizationResult,
    PAdicBoundaryAnalyzer,
    compute_padic_distance,
)
from src.biology.codons import codon_to_index


class TestCodonToIndex:
    """Tests for codon_to_index function."""

    def test_basic_codons(self):
        """Test index computation for basic codons."""
        # UUU should be 0 (0*16 + 0*4 + 0)
        assert codon_to_index("UUU") == 0

        # GGG should be 63 (3*16 + 3*4 + 3)
        assert codon_to_index("GGG") == 63

        # CGU (arginine) should have specific index
        idx_cgu = codon_to_index("CGU")
        assert 0 <= idx_cgu <= 63

    def test_dna_to_rna_conversion(self):
        """Test that T is converted to U."""
        assert codon_to_index("TTT") == codon_to_index("UUU")


class TestPAdicDistance:
    """Tests for compute_padic_distance function."""

    def test_same_index(self):
        """Test distance between same indices is 0."""
        assert compute_padic_distance(0, 0) == 0.0
        assert compute_padic_distance(27, 27) == 0.0

    def test_different_indices(self):
        """Test distance between different indices."""
        # Distance should be positive for different indices
        dist = compute_padic_distance(0, 1)
        assert dist > 0.0

    def test_p_divisibility(self):
        """Test p-adic distance reflects divisibility by p."""
        # Difference of 3 (divisible by 3 once)
        dist_3 = compute_padic_distance(0, 3, p=3)

        # Difference of 9 (divisible by 3 twice)
        dist_9 = compute_padic_distance(0, 9, p=3)

        # Difference of 1 (not divisible by 3)
        dist_1 = compute_padic_distance(0, 1, p=3)

        # 9 should have smaller distance than 3 (higher valuation)
        assert dist_9 < dist_3
        # 1 should have larger distance than 3
        assert dist_1 > dist_3


class TestCodonChoice:
    """Tests for CodonChoice dataclass."""

    def test_creation(self):
        """Test dataclass creation."""
        choice = CodonChoice(
            codon="CGU",
            amino_acid="R",
            padic_index=24,
            boundary_distance=0.15,
            usage_frequency=0.08,
            tRNA_abundance=0.5,
        )

        assert choice.codon == "CGU"
        assert choice.amino_acid == "R"
        assert choice.padic_index == 24


class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""

    def test_creation(self):
        """Test result creation."""
        result = OptimizationResult(
            original_sequence="ATGCGT",
            optimized_sequence="ATGAGA",
            original_codons=["AUG", "CGU"],
            optimized_codons=["AUG", "AGA"],
            changes_made=[(1, "CGU", "AGA")],
            original_padic_distance=0.3,
            optimized_padic_distance=0.5,
            improvement_score=0.2,
            immunogenicity_reduction=0.15,
        )

        assert result.original_sequence == "ATGCGT"
        assert result.optimized_sequence == "ATGAGA"
        assert result.improvement_score == 0.2
        assert result.immunogenicity_reduction == 0.15


class TestPAdicBoundaryAnalyzer:
    """Tests for PAdicBoundaryAnalyzer."""

    def test_creation(self):
        """Test analyzer creation."""
        analyzer = PAdicBoundaryAnalyzer(p=3, zone_min=0.1, zone_max=0.5)
        assert analyzer.p == 3
        assert analyzer.zone_min == 0.1
        assert analyzer.zone_max == 0.5

    def test_get_safest_arginine_codon(self):
        """Test getting safest arginine codon."""
        analyzer = PAdicBoundaryAnalyzer(p=3)

        safest = analyzer.get_safest_arginine_codon()

        # Should be one of the arginine codons
        arginine_codons = {"CGU", "CGC", "CGA", "CGG", "AGA", "AGG"}
        assert safest in arginine_codons

    def test_rank_arginine_codons(self):
        """Test ranking arginine codons."""
        analyzer = PAdicBoundaryAnalyzer(p=3)

        ranked = analyzer.rank_arginine_codons()

        # Should have 6 arginine codons
        assert len(ranked) == 6

        # Should be sorted by boundary distance (descending)
        distances = [d for _, d in ranked]
        assert distances == sorted(distances, reverse=True)

    def test_is_in_danger_zone(self):
        """Test danger zone checking."""
        analyzer = PAdicBoundaryAnalyzer(p=3, zone_min=0.1, zone_max=0.5)

        # Check various indices
        result = analyzer.is_in_danger_zone(0)
        assert isinstance(result, bool)


class TestCodonContextOptimizer:
    """Tests for CodonContextOptimizer."""

    def test_creation(self):
        """Test optimizer creation."""
        optimizer = CodonContextOptimizer(context_size=5, embedding_dim=32)
        assert optimizer.context_size == 5

    def test_encode_codon_sequence(self):
        """Test codon sequence encoding."""
        optimizer = CodonContextOptimizer()

        codons = ["AUG", "CGU", "ACG"]
        encoded = optimizer.encode_codon_sequence(codons)

        assert len(encoded) == 3
        assert all(0 <= idx <= 63 for idx in encoded)

    def test_forward(self):
        """Test forward pass."""
        optimizer = CodonContextOptimizer(context_size=2)

        # Context of 5 codons (2 on each side + center)
        contexts = [
            ["AUG", "GCU", "CGU", "ACG", "UGG"],
            ["GGG", "AAA", "AGA", "CCC", "UUU"],
        ]

        result = optimizer(contexts)

        assert "safety_score" in result
        assert "suggestion_probabilities" in result
        assert len(result["safety_score"]) == 2


class TestCitrullinationBoundaryOptimizer:
    """Tests for CitrullinationBoundaryOptimizer."""

    def test_creation(self):
        """Test optimizer creation."""
        optimizer = CitrullinationBoundaryOptimizer(
            p=3, zone_min=0.1, zone_max=0.5
        )
        assert optimizer.p == 3

    def test_dna_to_codons(self):
        """Test DNA to codon conversion."""
        optimizer = CitrullinationBoundaryOptimizer()

        dna = "ATGCGTACG"  # 3 codons
        codons = optimizer.dna_to_codons(dna)

        assert len(codons) == 3
        assert codons[0] == "AUG"  # T -> U
        assert codons[1] == "CGU"
        assert codons[2] == "ACG"

    def test_codons_to_dna(self):
        """Test codon to DNA conversion."""
        optimizer = CitrullinationBoundaryOptimizer()

        codons = ["AUG", "CGU", "ACG"]
        dna = optimizer.codons_to_dna(codons)

        assert dna == "ATGCGTACG"

    def test_find_arginine_positions(self):
        """Test finding arginine positions."""
        optimizer = CitrullinationBoundaryOptimizer()

        # ATG CGT ACG AGA (Met Arg Thr Arg)
        codons = ["AUG", "CGU", "ACG", "AGA"]
        positions = optimizer.find_arginine_positions(codons)

        # Positions 1 and 3 have arginine codons
        assert 1 in positions
        assert 3 in positions
        assert 0 not in positions
        assert 2 not in positions

    def test_optimize_arginine_codons(self):
        """Test arginine codon optimization."""
        optimizer = CitrullinationBoundaryOptimizer()

        codons = ["CGU", "AGA"]
        optimized = optimizer.optimize_arginine_codons(codons)

        assert len(optimized) == 2
        # All optimized codons should still encode arginine
        arginine_codons = {"CGU", "CGC", "CGA", "CGG", "AGA", "AGG"}
        for codon in optimized:
            assert codon in arginine_codons


class TestIntegration:
    """Integration tests for optimization pipeline."""

    def test_roundtrip_conversion(self):
        """Test DNA -> codons -> DNA conversion."""
        optimizer = CitrullinationBoundaryOptimizer()

        original = "ATGCGTACGAGA"
        codons = optimizer.dna_to_codons(original)
        reconstructed = optimizer.codons_to_dna(codons)

        assert reconstructed == original

    def test_arginine_codon_preservation(self):
        """Test that optimization preserves arginine codons (as arginine)."""
        optimizer = CitrullinationBoundaryOptimizer()

        # Sequence with arginine codons
        codons = ["AUG", "CGU", "ACG", "AGA"]
        arg_positions = optimizer.find_arginine_positions(codons)

        # Optimize arginine codons
        arg_codons = [codons[i] for i in arg_positions]
        optimized_args = optimizer.optimize_arginine_codons(arg_codons)

        # All should still be arginine codons
        arginine_codons = {"CGU", "CGC", "CGA", "CGG", "AGA", "AGG"}
        for codon in optimized_args:
            assert codon in arginine_codons
