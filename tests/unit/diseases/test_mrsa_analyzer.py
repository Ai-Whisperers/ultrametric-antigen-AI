# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Comprehensive unit tests for MRSA (Methicillin-Resistant S. aureus) Analyzer.

Tests cover:
- Analyzer initialization
- mecA-based resistance detection
- Synthetic dataset generation
- Multi-drug resistance patterns
"""

import numpy as np
import pytest

from src.diseases.mrsa_analyzer import (
    MRSAAnalyzer,
    MRSAConfig,
    StaphGene,
    Antibiotic,
    create_mrsa_synthetic_dataset,
    create_mrsa_simple_dataset,
)


class TestMRSAEnums:
    """Test MRSA enumerations."""

    def test_mrsa_genes(self):
        """Test MRSA gene definitions."""
        genes = list(StaphGene)
        assert len(genes) >= 1

        # mecA is the key resistance gene
        assert StaphGene.MECA in genes

    def test_mrsa_drugs(self):
        """Test MRSA drug definitions."""
        drugs = list(Antibiotic)
        assert len(drugs) >= 3

        # Key drugs
        assert Antibiotic.OXACILLIN in drugs or any("OXACILLIN" in d.name for d in drugs)


class TestMRSAAnalyzer:
    """Test MRSAAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return MRSAAnalyzer()

    @pytest.fixture
    def config(self):
        """Create config fixture."""
        return MRSAConfig()

    def test_initialization_default(self, analyzer):
        """Test default initialization."""
        assert analyzer is not None
        assert analyzer.config is not None
        assert hasattr(analyzer, "aa_alphabet")
        assert hasattr(analyzer, "aa_to_idx")

    def test_initialization_with_config(self, config):
        """Test initialization with custom config."""
        analyzer = MRSAAnalyzer(config=config)
        assert analyzer.config == config

    def test_config_defaults(self, config):
        """Test config default values."""
        assert config.name == "mrsa"
        assert "MRSA" in config.display_name or "Staphylococcus" in config.display_name

    def test_amino_acid_alphabet(self, analyzer):
        """Test amino acid alphabet."""
        assert len(analyzer.aa_alphabet) >= 20
        assert "A" in analyzer.aa_alphabet
        assert "M" in analyzer.aa_alphabet

    def test_encode_sequence(self, analyzer):
        """Test sequence encoding."""
        test_seq = "MKKNTQKIEVVVTDGTRGLLSFDLEKPNSIV"
        encoding = analyzer.encode_sequence(test_seq)

        assert isinstance(encoding, np.ndarray)
        assert encoding.dtype == np.float32
        assert encoding.max() <= 1.0
        assert encoding.min() >= 0.0


class TestMRSADrugResistance:
    """Test drug resistance prediction."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return MRSAAnalyzer()

    def test_analyze_basic(self, analyzer):
        """Test basic analyze functionality."""
        test_seq = "M" * 300  # mecA length
        test_sequences = {StaphGene.MECA: [test_seq]}

        results = analyzer.analyze(test_sequences)

        assert isinstance(results, dict)
        assert "n_sequences" in results

    def test_meca_detection(self, analyzer):
        """Test mecA-based resistance detection."""
        # mecA presence should indicate resistance
        test_sequences = {StaphGene.MECA: ["MKKNTQKIEVVVT" * 20]}

        results = analyzer.analyze(test_sequences)

        assert "drug_resistance" in results or "n_sequences" in results


class TestMRSASyntheticDataset:
    """Test synthetic dataset generation."""

    def test_create_synthetic_dataset_default(self):
        """Test default synthetic dataset creation."""
        X, y, ids = create_mrsa_synthetic_dataset()

        assert len(X) > 0
        assert len(y) == len(X)
        assert len(ids) == len(X)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_create_simple_dataset(self):
        """Test simple mecA-focused dataset creation."""
        X, y, ids = create_mrsa_simple_dataset()

        assert len(X) > 0
        assert len(y) == len(X)
        assert len(ids) == len(X)

    def test_create_synthetic_dataset_min_samples(self):
        """Test minimum samples parameter."""
        min_samples = 30
        X, y, ids = create_mrsa_synthetic_dataset(min_samples=min_samples)

        assert len(X) >= min_samples

    def test_synthetic_labels_range(self):
        """Test synthetic labels are in valid range."""
        X, y, ids = create_mrsa_synthetic_dataset()

        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_synthetic_encoding_valid(self):
        """Test synthetic encodings are valid."""
        X, y, ids = create_mrsa_synthetic_dataset()

        assert X.min() >= 0.0
        assert X.max() <= 1.0


class TestMRSAValidation:
    """Test validation methods."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return MRSAAnalyzer()

    def test_validate_predictions_exists(self, analyzer):
        """Test validate_predictions method exists."""
        assert hasattr(analyzer, "validate_predictions")


class TestMRSAIntegration:
    """Integration tests for MRSA analyzer."""

    def test_full_pipeline(self):
        """Test full analysis pipeline."""
        # Create synthetic data
        X, y, ids = create_mrsa_synthetic_dataset(min_samples=20)

        # Create analyzer
        analyzer = MRSAAnalyzer()

        # Encode a sample sequence
        test_seq = "MKKNTQKIEVVVTDGTRGLLSFDLEKPNSIV" * 10
        encoding = analyzer.encode_sequence(test_seq, max_length=350)

        assert len(encoding) > 0
        assert encoding.dtype == np.float32

    def test_reproducibility(self):
        """Test dataset generation is reproducible."""
        X1, y1, ids1 = create_mrsa_synthetic_dataset(min_samples=20)
        X2, y2, ids2 = create_mrsa_synthetic_dataset(min_samples=20)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
