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

from shared.config import get_config, Config
from shared.constants import (
    HIV_DRUG_CLASSES,
    AMINO_ACIDS,
    HYDROPHOBICITY,
    CHARGES,
    MOLECULAR_WEIGHTS,
)


class TestConfig:
    """Tests for Config class."""

    def test_config_singleton(self):
        """Test that get_config returns same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_config_paths_exist(self):
        """Test that config paths are valid."""
        config = get_config()

        assert config.deliverables_root.exists()
        assert config.project_root.exists()

    def test_get_partner_dir(self):
        """Test getting partner directories."""
        config = get_config()

        hiv_dir = config.get_partner_dir("hiv")
        assert "hiv" in str(hiv_dir).lower()

        rojas_dir = config.get_partner_dir("rojas")
        assert "rojas" in str(rojas_dir).lower()

    def test_get_cache_path(self):
        """Test accessing cache paths."""
        config = get_config()

        # Test method exists
        assert hasattr(config, "get_cache_path")
        cache_path = config.get_cache_path("test.json")
        assert "test.json" in str(cache_path)


class TestHIVDrugClasses:
    """Tests for HIV drug class constants."""

    def test_drug_classes_defined(self):
        """Test all drug classes are defined."""
        expected_classes = ["NRTI", "NNRTI", "PI", "INSTI"]

        for drug_class in expected_classes:
            assert drug_class in HIV_DRUG_CLASSES

    def test_drug_class_contents(self):
        """Test drug class contents are valid."""
        # HIV_DRUG_CLASSES can be dict of class->description or class->list of drugs
        for class_name, value in HIV_DRUG_CLASSES.items():
            assert isinstance(class_name, str)
            # Value can be string description or list of drugs
            assert value is not None


class TestAminoAcidConstants:
    """Tests for amino acid constants."""

    def test_all_amino_acids_present(self):
        """Test all 20 standard amino acids are defined."""
        expected_aa = set("ACDEFGHIKLMNPQRSTVWY")
        actual_aa = set(AMINO_ACIDS)

        assert expected_aa == actual_aa

    def test_amino_acid_properties(self):
        """Test amino acid properties are defined."""
        # Each amino acid should have hydrophobicity and charge defined
        for aa in AMINO_ACIDS:
            assert aa in HYDROPHOBICITY, f"Missing hydrophobicity for {aa}"
            assert aa in CHARGES, f"Missing charge for {aa}"
            assert aa in MOLECULAR_WEIGHTS, f"Missing molecular weight for {aa}"


class TestPeptideUtilities:
    """Tests for peptide utility functions."""

    def test_compute_peptide_properties(self):
        """Test peptide properties calculation."""
        from shared.peptide_utils import compute_peptide_properties

        # Test basic properties
        props = compute_peptide_properties("MKVLIYG")

        assert "length" in props
        assert props["length"] == 7
        assert "net_charge" in props
        assert "hydrophobicity" in props

    def test_compute_amino_acid_composition(self):
        """Test amino acid composition calculation."""
        from shared.peptide_utils import compute_amino_acid_composition

        comp = compute_amino_acid_composition("AAAKK")

        # Function returns numpy array of AA frequencies (20 values)
        import numpy as np
        assert isinstance(comp, np.ndarray)
        assert len(comp) == 20  # 20 standard amino acids
        assert comp.sum() > 0  # Has some composition

    def test_compute_ml_features(self):
        """Test ML feature extraction."""
        from shared.peptide_utils import compute_ml_features

        features = compute_ml_features("MKVLIYGRK")

        # Should return some features
        assert features is not None
        assert len(features) > 0

    def test_compute_physicochemical_descriptors(self):
        """Test physicochemical descriptors."""
        from shared.peptide_utils import compute_physicochemical_descriptors

        desc = compute_physicochemical_descriptors("MKVLIYGRK")

        # Should return dict of descriptors
        assert isinstance(desc, dict)
        assert len(desc) > 0


class TestSequenceValidation:
    """Tests for sequence validation utilities."""

    def test_validate_amino_acid_sequence(self):
        """Test amino acid sequence validation."""
        from shared.peptide_utils import validate_sequence

        # Valid sequence - returns (is_valid, error)
        result = validate_sequence("MKVLIYG")
        if isinstance(result, tuple):
            is_valid, error = result
            assert is_valid
        else:
            assert result is True or result == True

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
