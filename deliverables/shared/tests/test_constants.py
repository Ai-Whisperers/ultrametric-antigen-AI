# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for shared constants module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add deliverables to path
deliverables_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(deliverables_dir))

from shared.constants import (
    AMINO_ACIDS,
    CHARGES,
    CODON_TABLE,
    FLEXIBILITY,
    HELIX_PROPENSITY,
    HYDROPHOBICITY,
    MOLECULAR_WEIGHTS,
    NUCLEOTIDES,
    SHEET_PROPENSITY,
    VOLUMES,
    WHO_CRITICAL_PATHOGENS,
)


class TestAminoAcids:
    """Tests for amino acid constants."""

    def test_amino_acids_count(self):
        """Should have exactly 20 standard amino acids."""
        assert len(AMINO_ACIDS) == 20

    def test_amino_acids_unique(self):
        """All amino acids should be unique."""
        assert len(set(AMINO_ACIDS)) == 20

    def test_amino_acids_uppercase(self):
        """All amino acids should be uppercase."""
        assert AMINO_ACIDS == AMINO_ACIDS.upper()

    def test_all_standard_aa_present(self):
        """All 20 standard amino acids should be present."""
        standard = set("ACDEFGHIKLMNPQRSTVWY")
        assert set(AMINO_ACIDS) == standard


class TestHydrophobicity:
    """Tests for hydrophobicity scale."""

    def test_all_aa_have_hydrophobicity(self):
        """All 20 amino acids should have hydrophobicity values."""
        for aa in AMINO_ACIDS:
            assert aa in HYDROPHOBICITY

    def test_isoleucine_most_hydrophobic(self):
        """Isoleucine should be highly hydrophobic."""
        assert HYDROPHOBICITY["I"] > 3.0  # Known value ~4.5

    def test_arginine_most_hydrophilic(self):
        """Arginine should be most hydrophilic."""
        assert HYDROPHOBICITY["R"] < -4.0  # Known value ~-4.5

    def test_hydrophobicity_range(self):
        """Hydrophobicity should be in reasonable range."""
        for aa, value in HYDROPHOBICITY.items():
            assert -5.0 <= value <= 5.0


class TestCharges:
    """Tests for amino acid charges."""

    def test_all_aa_have_charges(self):
        """All 20 amino acids should have charge values."""
        for aa in AMINO_ACIDS:
            assert aa in CHARGES

    def test_basic_residues_positive(self):
        """K and R should have positive charge."""
        assert CHARGES["K"] > 0
        assert CHARGES["R"] > 0

    def test_acidic_residues_negative(self):
        """D and E should have negative charge."""
        assert CHARGES["D"] < 0
        assert CHARGES["E"] < 0

    def test_neutral_residues(self):
        """Non-polar residues should be neutral."""
        neutral = "ACFGILMPTVW"
        for aa in neutral:
            assert CHARGES[aa] == 0


class TestVolumes:
    """Tests for amino acid volumes."""

    def test_all_aa_have_volumes(self):
        """All 20 amino acids should have volume values."""
        for aa in AMINO_ACIDS:
            assert aa in VOLUMES

    def test_glycine_smallest(self):
        """Glycine should be the smallest residue."""
        assert VOLUMES["G"] == min(VOLUMES.values())

    def test_tryptophan_largest(self):
        """Tryptophan should be the largest residue."""
        assert VOLUMES["W"] == max(VOLUMES.values())

    def test_volume_range(self):
        """Volumes should be in reasonable range."""
        for aa, value in VOLUMES.items():
            assert 50 < value < 250  # Angstrom^3


class TestMolecularWeights:
    """Tests for amino acid molecular weights."""

    def test_all_aa_have_weights(self):
        """All 20 amino acids should have molecular weight values."""
        for aa in AMINO_ACIDS:
            assert aa in MOLECULAR_WEIGHTS

    def test_glycine_lightest(self):
        """Glycine should be the lightest residue."""
        assert MOLECULAR_WEIGHTS["G"] == min(MOLECULAR_WEIGHTS.values())

    def test_tryptophan_heaviest(self):
        """Tryptophan should be the heaviest residue."""
        assert MOLECULAR_WEIGHTS["W"] == max(MOLECULAR_WEIGHTS.values())


class TestFlexibility:
    """Tests for amino acid flexibility."""

    def test_all_aa_have_flexibility(self):
        """All 20 amino acids should have flexibility values."""
        for aa in AMINO_ACIDS:
            assert aa in FLEXIBILITY

    def test_glycine_flexible(self):
        """Glycine should be one of the most flexible."""
        assert FLEXIBILITY["G"] > 0.5

    def test_flexibility_range(self):
        """Flexibility should be between 0 and 1."""
        for aa, value in FLEXIBILITY.items():
            assert 0 < value < 1


class TestSecondaryStructurePropensity:
    """Tests for secondary structure propensities."""

    def test_helix_propensity_complete(self):
        """All 20 amino acids should have helix propensity."""
        for aa in AMINO_ACIDS:
            assert aa in HELIX_PROPENSITY

    def test_sheet_propensity_complete(self):
        """All 20 amino acids should have sheet propensity."""
        for aa in AMINO_ACIDS:
            assert aa in SHEET_PROPENSITY

    def test_alanine_helix_former(self):
        """Alanine should be a strong helix former."""
        assert HELIX_PROPENSITY["A"] > 1.0

    def test_proline_helix_breaker(self):
        """Proline should be a helix breaker."""
        assert HELIX_PROPENSITY["P"] < 1.0

    def test_valine_sheet_former(self):
        """Valine should be a strong sheet former."""
        assert SHEET_PROPENSITY["V"] > 1.0


class TestCodonTable:
    """Tests for genetic code codon table."""

    def test_codon_count(self):
        """Should have 64 codons."""
        assert len(CODON_TABLE) == 64

    def test_start_codon(self):
        """ATG should code for Methionine (start)."""
        assert CODON_TABLE["ATG"] == "M"

    def test_stop_codons(self):
        """Should have 3 stop codons."""
        stop_codons = [k for k, v in CODON_TABLE.items() if v == "*"]
        assert len(stop_codons) == 3
        assert "TAA" in stop_codons
        assert "TAG" in stop_codons
        assert "TGA" in stop_codons

    def test_amino_acid_coverage(self):
        """All amino acids should be encoded by at least one codon."""
        encoded_aa = set(v for v in CODON_TABLE.values() if v != "*")
        for aa in AMINO_ACIDS:
            if aa != "X":  # X is not a standard AA
                assert aa in encoded_aa


class TestWHOPathogens:
    """Tests for WHO priority pathogen list."""

    def test_critical_pathogens_not_empty(self):
        """Critical pathogen list should not be empty."""
        assert len(WHO_CRITICAL_PATHOGENS) >= 3

    def test_acinetobacter_is_critical(self):
        """A. baumannii should be in critical list."""
        assert any("baumannii" in p for p in WHO_CRITICAL_PATHOGENS)

    def test_pseudomonas_is_critical(self):
        """P. aeruginosa should be in critical list."""
        assert any("aeruginosa" in p for p in WHO_CRITICAL_PATHOGENS)


class TestNucleotides:
    """Tests for nucleotide constants."""

    def test_nucleotide_count(self):
        """Should have exactly 4 nucleotides."""
        assert len(NUCLEOTIDES) == 4

    def test_nucleotides_content(self):
        """Should contain A, C, G, T."""
        assert set(NUCLEOTIDES) == {"A", "C", "G", "T"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
