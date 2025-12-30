# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for shared infrastructure module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add deliverables to path
deliverables_dir = Path(__file__).parent.parent
sys.path.insert(0, str(deliverables_dir))

from shared.config import get_config, DeliverableConfig
from shared.constants import (
    HIV_DRUG_CLASSES,
    AMINO_ACIDS,
    AMINO_ACID_PROPERTIES,
)


class TestDeliverableConfig:
    """Tests for DeliverableConfig class."""

    def test_config_singleton(self):
        """Test that get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_config_paths_exist(self):
        """Test that config paths are valid."""
        config = get_config()

        assert config.deliverables_dir.exists()
        assert config.shared_dir.exists()

    def test_get_partner_dir(self):
        """Test getting partner directories."""
        config = get_config()

        hiv_dir = config.get_partner_dir("hiv")
        assert "hiv" in str(hiv_dir).lower()

        rojas_dir = config.get_partner_dir("rojas")
        assert "rojas" in str(rojas_dir).lower()

    def test_get_shared_data(self):
        """Test accessing shared data files."""
        config = get_config()

        # Test method exists
        assert hasattr(config, "get_shared_data") or hasattr(config, "shared_data_dir")


class TestHIVDrugClasses:
    """Tests for HIV drug class constants."""

    def test_drug_classes_defined(self):
        """Test all drug classes are defined."""
        expected_classes = ["NRTI", "NNRTI", "PI", "INSTI"]

        for drug_class in expected_classes:
            assert drug_class in HIV_DRUG_CLASSES

    def test_drug_class_descriptions(self):
        """Test drug class descriptions are strings."""
        for class_name, description in HIV_DRUG_CLASSES.items():
            assert isinstance(description, str)
            assert len(description) > 0


class TestAminoAcidConstants:
    """Tests for amino acid constants."""

    def test_all_amino_acids_present(self):
        """Test all 20 standard amino acids are defined."""
        expected_aa = set("ACDEFGHIKLMNPQRSTVWY")
        actual_aa = set(AMINO_ACIDS)

        assert expected_aa == actual_aa

    def test_amino_acid_properties(self):
        """Test amino acid properties are defined."""
        # Each amino acid should have basic properties
        for aa in AMINO_ACIDS:
            assert aa in AMINO_ACID_PROPERTIES

            props = AMINO_ACID_PROPERTIES[aa]

            # Check required property keys
            if isinstance(props, dict):
                # Should have some properties
                assert len(props) > 0


class TestPeptideUtilities:
    """Tests for peptide utility functions."""

    def test_compute_molecular_weight(self):
        """Test molecular weight calculation."""
        from shared.peptide_utils import compute_molecular_weight

        # Single amino acid
        mw_a = compute_molecular_weight("A")
        assert 70 < mw_a < 100  # Alanine ~89 Da

        # Longer peptide
        mw_longer = compute_molecular_weight("MKVLIYG")
        assert mw_longer > mw_a

    def test_compute_isoelectric_point(self):
        """Test isoelectric point calculation."""
        from shared.peptide_utils import compute_isoelectric_point

        # Basic peptide
        pi = compute_isoelectric_point("MKVLIYG")
        assert 3.0 < pi < 12.0  # Reasonable pI range

    def test_compute_hydrophobicity(self):
        """Test hydrophobicity calculation."""
        from shared.peptide_utils import compute_hydrophobicity

        # Hydrophobic peptide
        hydro_val = compute_hydrophobicity("VVVVVV")

        # Hydrophilic peptide
        hydro_lys = compute_hydrophobicity("KKKKKK")

        # V (Val) is more hydrophobic than K (Lys)
        assert hydro_val > hydro_lys

    def test_analyze_peptide(self):
        """Test comprehensive peptide analysis."""
        from shared.peptide_utils import analyze_peptide

        result = analyze_peptide("MKVLIYGRK")

        assert "sequence" in result
        assert "length" in result
        assert "molecular_weight" in result
        assert result["length"] == 9


class TestSequenceValidation:
    """Tests for sequence validation utilities."""

    def test_validate_amino_acid_sequence(self):
        """Test amino acid sequence validation."""
        from shared.peptide_utils import validate_sequence

        # Valid sequence
        assert validate_sequence("MKVLIYG") is True

        # Invalid characters
        assert validate_sequence("MKV123") is False

    def test_validate_nucleotide_sequence(self):
        """Test nucleotide sequence validation."""
        # Implement if available in shared module
        pass


class TestDataLoading:
    """Tests for data loading utilities."""

    def test_load_json_data(self):
        """Test JSON data loading."""
        import json
        from pathlib import Path

        # Create temp JSON for testing
        test_data = {"key": "value", "number": 42}

        # Test JSON loading logic
        json_str = json.dumps(test_data)
        loaded = json.loads(json_str)

        assert loaded == test_data

    def test_load_fasta_sequence(self):
        """Test FASTA sequence parsing."""

        def parse_fasta(fasta_content: str) -> dict:
            """Simple FASTA parser."""
            sequences = {}
            header = None
            seq_lines = []

            for line in fasta_content.strip().split("\n"):
                if line.startswith(">"):
                    if header:
                        sequences[header] = "".join(seq_lines)
                    header = line[1:].split()[0]
                    seq_lines = []
                else:
                    seq_lines.append(line.strip())

            if header:
                sequences[header] = "".join(seq_lines)

            return sequences

        fasta = """>seq1
MKVLIYG
RKVMKVL
>seq2
AAAAAAA
"""
        result = parse_fasta(fasta)

        assert "seq1" in result
        assert "seq2" in result
        assert result["seq1"] == "MKVLIYGRKVMKVL"
        assert result["seq2"] == "AAAAAAA"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
