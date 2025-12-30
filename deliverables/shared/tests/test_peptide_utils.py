# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for shared peptide utilities.

Tests cover:
    - compute_peptide_properties
    - compute_amino_acid_composition
    - compute_ml_features
    - decode_latent_to_sequence
    - compute_physicochemical_descriptors
    - validate_sequence
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add deliverables to path
deliverables_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(deliverables_dir))

from shared.peptide_utils import (
    AA_PROPERTIES,
    compute_amino_acid_composition,
    compute_ml_features,
    compute_peptide_properties,
    compute_physicochemical_descriptors,
    decode_latent_to_sequence,
    decode_latent_with_vae,
    validate_sequence,
)


class TestComputePeptideProperties:
    """Tests for compute_peptide_properties function."""

    def test_empty_sequence(self):
        """Empty sequence should return zero values."""
        result = compute_peptide_properties("")
        assert result["net_charge"] == 0
        assert result["hydrophobicity"] == 0
        assert result["length"] == 0
        assert result["hydrophobic_ratio"] == 0
        assert result["cationic_ratio"] == 0

    def test_known_positive_peptide(self):
        """Test with a known cationic peptide."""
        # KKK should have high positive charge
        result = compute_peptide_properties("KKK")
        assert result["net_charge"] == 3.0  # Three lysines
        assert result["length"] == 3
        assert result["cationic_ratio"] == 1.0

    def test_known_negative_peptide(self):
        """Test with a known anionic peptide."""
        # DDD should have negative charge
        result = compute_peptide_properties("DDD")
        assert result["net_charge"] == -3.0  # Three aspartates
        assert result["length"] == 3
        assert result["cationic_ratio"] == 0

    def test_hydrophobic_peptide(self):
        """Test with a hydrophobic peptide."""
        # LLLL should be hydrophobic
        result = compute_peptide_properties("LLLL")
        assert result["hydrophobicity"] > 0  # Positive hydrophobicity
        assert result["hydrophobic_ratio"] == 1.0

    def test_mixed_peptide(self):
        """Test with a mixed peptide."""
        result = compute_peptide_properties("KKLLDD")
        assert result["length"] == 6
        assert result["net_charge"] == 0  # +2K -2D = 0
        assert result["hydrophobic_ratio"] == 2 / 6  # 2 L's
        assert result["cationic_ratio"] == 2 / 6  # 2 K's

    def test_real_amp_sequence(self):
        """Test with a real antimicrobial peptide (Magainin 2)."""
        magainin = "GIGKFLHSAKKFGKAFVGEIMNS"
        result = compute_peptide_properties(magainin)
        assert result["length"] == 23
        assert result["net_charge"] > 0  # AMPs are typically cationic
        assert result["cationic_ratio"] > 0


class TestComputeAminoAcidComposition:
    """Tests for compute_amino_acid_composition function."""

    def test_empty_sequence(self):
        """Empty sequence should return zero array."""
        result = compute_amino_acid_composition("")
        assert len(result) == 20
        assert np.all(result == 0)

    def test_single_aa_composition(self):
        """Single AA should give 100% for that residue."""
        result = compute_amino_acid_composition("AAAA")
        assert result.sum() == pytest.approx(1.0, rel=1e-6)
        # A should be the first in the standard order
        assert result[0] == 1.0 or any(result == 1.0)

    def test_composition_sums_to_one(self):
        """Composition should sum to 1.0 for any sequence."""
        result = compute_amino_acid_composition("ACDEFGHIKLMNPQRSTVWY")
        assert result.sum() == pytest.approx(1.0, rel=1e-6)

    def test_composition_values_reasonable(self):
        """Composition values should be between 0 and 1."""
        result = compute_amino_acid_composition("KKLLDD")
        assert np.all(result >= 0)
        assert np.all(result <= 1)


class TestComputeMLFeatures:
    """Tests for compute_ml_features function."""

    def test_feature_vector_length(self):
        """Feature vector should be 25-dimensional."""
        result = compute_ml_features("KKLLDD")
        assert len(result) == 25  # 5 props + 20 AA composition

    def test_features_are_numeric(self):
        """All features should be numeric."""
        result = compute_ml_features("GIGKFLHSAKKFGKAFVGEIMNS")
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_empty_sequence_features(self):
        """Empty sequence should return valid features."""
        result = compute_ml_features("")
        assert len(result) == 25
        assert not np.any(np.isnan(result))


class TestDecodeLatentToSequence:
    """Tests for decode_latent_to_sequence function."""

    def test_returns_string(self):
        """Should return a string sequence."""
        z = np.random.randn(16)
        result = decode_latent_to_sequence(z, length=20, use_vae=False)
        assert isinstance(result, str)
        assert len(result) == 20

    def test_valid_amino_acids(self):
        """Generated sequence should contain only valid AAs."""
        z = np.random.randn(16)
        result = decode_latent_to_sequence(z, length=30, use_vae=False)
        valid_aa = set(AA_PROPERTIES.keys())
        for aa in result:
            assert aa in valid_aa

    def test_deterministic_with_seed(self):
        """Same seed should produce same sequence."""
        z = np.random.randn(16)
        seq1 = decode_latent_to_sequence(z, length=20, seed=42, use_vae=False)
        seq2 = decode_latent_to_sequence(z, length=20, seed=42, use_vae=False)
        assert seq1 == seq2

    def test_different_latent_vectors(self):
        """Different latent vectors should produce different sequences."""
        z1 = np.array([1.0, -1.0] + [0.0] * 14)
        z2 = np.array([-1.0, 1.0] + [0.0] * 14)
        seq1 = decode_latent_to_sequence(z1, length=50, use_vae=False)
        seq2 = decode_latent_to_sequence(z2, length=50, use_vae=False)
        # With sufficiently different latent vectors, sequences should differ
        assert seq1 != seq2


class TestDecodeLatentWithVae:
    """Tests for decode_latent_with_vae function."""

    def test_returns_tuple(self):
        """Should return tuple of (sequence, is_real)."""
        z = np.random.randn(16)
        result = decode_latent_with_vae(z)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], bool)


class TestComputePhysicochemicalDescriptors:
    """Tests for compute_physicochemical_descriptors function."""

    def test_contains_basic_properties(self):
        """Should include basic properties."""
        result = compute_physicochemical_descriptors("KKLLDD")
        assert "net_charge" in result
        assert "hydrophobicity" in result
        assert "length" in result

    def test_contains_extended_properties(self):
        """Should include extended physicochemical properties."""
        result = compute_physicochemical_descriptors("KKLLDD")
        assert "aromaticity" in result
        assert "aliphatic_index" in result
        assert "tiny_ratio" in result
        assert "small_ratio" in result
        assert "large_ratio" in result
        assert "polar_ratio" in result

    def test_aromatic_sequence(self):
        """Aromatic sequence should have high aromaticity."""
        result = compute_physicochemical_descriptors("FFWWYY")
        assert result["aromaticity"] == 1.0

    def test_aliphatic_sequence(self):
        """Aliphatic sequence should have high aliphatic index."""
        result = compute_physicochemical_descriptors("AVILLL")
        assert result["aliphatic_index"] > 100  # High aliphatic index


class TestValidateSequence:
    """Tests for validate_sequence function."""

    def test_valid_sequence(self):
        """Valid sequence should pass."""
        is_valid, msg = validate_sequence("ACDEFGHIKLMNPQRSTVWY")
        assert is_valid
        assert msg == ""

    def test_empty_sequence(self):
        """Empty sequence should fail."""
        is_valid, msg = validate_sequence("")
        assert not is_valid
        assert "Empty" in msg

    def test_invalid_characters(self):
        """Sequence with invalid characters should fail."""
        is_valid, msg = validate_sequence("ACDE123")
        assert not is_valid
        assert "Invalid" in msg

    def test_lowercase_not_accepted(self):
        """Lowercase letters should be rejected."""
        is_valid, msg = validate_sequence("acde")
        assert not is_valid

    def test_ambiguous_allowed(self):
        """Ambiguous codes should be allowed when flag is set."""
        is_valid, msg = validate_sequence("ACDEX", allow_ambiguous=True)
        assert is_valid

    def test_ambiguous_rejected_by_default(self):
        """Ambiguous codes should be rejected by default."""
        is_valid, msg = validate_sequence("ACDEX")
        assert not is_valid


class TestAAProperties:
    """Tests for AA_PROPERTIES dictionary."""

    def test_all_standard_aa_present(self):
        """All 20 standard amino acids should be present."""
        standard_aa = "ACDEFGHIKLMNPQRSTVWY"
        for aa in standard_aa:
            assert aa in AA_PROPERTIES

    def test_properties_have_required_keys(self):
        """Each AA should have charge, hydrophobicity, and volume."""
        for aa, props in AA_PROPERTIES.items():
            assert "charge" in props
            assert "hydrophobicity" in props
            assert "volume" in props

    def test_known_charged_residues(self):
        """K, R, H should be positive; D, E should be negative."""
        assert AA_PROPERTIES["K"]["charge"] > 0
        assert AA_PROPERTIES["R"]["charge"] > 0
        assert AA_PROPERTIES["D"]["charge"] < 0
        assert AA_PROPERTIES["E"]["charge"] < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
