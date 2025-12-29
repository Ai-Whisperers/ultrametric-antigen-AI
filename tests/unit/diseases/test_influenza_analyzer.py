# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Comprehensive unit tests for Influenza Analyzer.

Tests cover:
- Analyzer initialization
- NAI (neuraminidase inhibitor) resistance detection
- Antigenic analysis
- Synthetic dataset generation
- Multi-subtype support
"""

import numpy as np
import pytest

from src.diseases.influenza_analyzer import (
    InfluenzaAnalyzer,
    InfluenzaConfig,
    InfluenzaGene,
    InfluenzaDrug,
    InfluenzaSubtype,
    create_influenza_synthetic_dataset,
    NA_MUTATIONS,
    PA_MUTATIONS,
)


class TestInfluenzaEnums:
    """Test Influenza enumerations."""

    def test_influenza_subtypes(self):
        """Test Influenza subtype definitions."""
        subtypes = list(InfluenzaSubtype)
        assert len(subtypes) >= 5

        # Main subtypes
        assert InfluenzaSubtype.H3N2 in subtypes
        assert InfluenzaSubtype.H1N1_SEASONAL in subtypes or InfluenzaSubtype.H1N1_PANDEMIC in subtypes

    def test_influenza_genes(self):
        """Test Influenza gene segment definitions."""
        genes = list(InfluenzaGene)
        assert len(genes) >= 8

        # Key surface glycoproteins
        assert InfluenzaGene.HA in genes  # Hemagglutinin
        assert InfluenzaGene.NA in genes  # Neuraminidase

        # Polymerase components
        assert InfluenzaGene.PA in genes
        assert InfluenzaGene.PB1 in genes
        assert InfluenzaGene.PB2 in genes

    def test_influenza_drugs(self):
        """Test Influenza drug definitions."""
        drugs = list(InfluenzaDrug)
        assert len(drugs) >= 4

        # NAIs
        assert InfluenzaDrug.OSELTAMIVIR in drugs
        assert InfluenzaDrug.ZANAMIVIR in drugs

        # PA inhibitor
        assert InfluenzaDrug.BALOXAVIR in drugs


class TestInfluenzaMutationDatabase:
    """Test Influenza mutation database."""

    def test_na_mutations_structure(self):
        """Test NA mutation database structure."""
        assert len(NA_MUTATIONS) > 0

        for pos, info in NA_MUTATIONS.items():
            assert isinstance(pos, int)
            assert pos > 0
            ref_aa = list(info.keys())[0]
            assert "mutations" in info[ref_aa]
            assert "effect" in info[ref_aa]
            assert "drugs" in info[ref_aa]

    def test_key_nai_resistance_positions(self):
        """Test key NAI resistance mutation positions."""
        # H275Y (N2 numbering) - key oseltamivir resistance
        key_positions = [119, 275, 292, 294]  # Common resistance positions

        positions_found = [p for p in key_positions if p in NA_MUTATIONS]
        assert len(positions_found) >= 1, "Should have at least one key NAI resistance position"

    def test_pa_mutations_for_baloxavir(self):
        """Test PA mutations for baloxavir resistance."""
        if PA_MUTATIONS:
            assert len(PA_MUTATIONS) > 0

            # I38T is key baloxavir resistance mutation
            if 38 in PA_MUTATIONS:
                info = PA_MUTATIONS[38]
                ref_aa = list(info.keys())[0]
                assert "T" in info[ref_aa]["mutations"] or "M" in info[ref_aa]["mutations"]


class TestInfluenzaAnalyzer:
    """Test InfluenzaAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return InfluenzaAnalyzer()

    @pytest.fixture
    def config(self):
        """Create config fixture."""
        return InfluenzaConfig()

    def test_initialization_default(self, analyzer):
        """Test default initialization."""
        assert analyzer is not None
        assert analyzer.config is not None
        assert hasattr(analyzer, "aa_alphabet")
        assert hasattr(analyzer, "aa_to_idx")

    def test_initialization_with_config(self, config):
        """Test initialization with custom config."""
        analyzer = InfluenzaAnalyzer(config=config)
        assert analyzer.config == config

    def test_config_defaults(self, config):
        """Test config default values."""
        assert config.name == "influenza"
        assert "Influenza" in config.display_name

    def test_amino_acid_alphabet(self, analyzer):
        """Test amino acid alphabet."""
        assert len(analyzer.aa_alphabet) >= 20
        assert "A" in analyzer.aa_alphabet
        assert "M" in analyzer.aa_alphabet

    def test_encode_sequence(self, analyzer):
        """Test sequence encoding."""
        test_seq = "MNPNQKIITIGSVSLTISTIC"
        encoding = analyzer.encode_sequence(test_seq)

        assert isinstance(encoding, np.ndarray)
        assert encoding.dtype == np.float32
        assert encoding.max() <= 1.0
        assert encoding.min() >= 0.0


class TestInfluenzaDrugResistance:
    """Test drug resistance prediction."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return InfluenzaAnalyzer()

    def test_analyze_basic(self, analyzer):
        """Test basic analyze functionality."""
        test_seq = "M" * 469  # NA length
        test_sequences = {InfluenzaGene.NA: [test_seq]}

        results = analyzer.analyze(test_sequences)

        assert isinstance(results, dict)
        assert "n_sequences" in results

    def test_analyze_with_subtype(self, analyzer):
        """Test analyze with subtype parameter."""
        test_seq = "M" * 469
        test_sequences = {InfluenzaGene.NA: [test_seq]}

        results = analyzer.analyze(test_sequences, subtype=InfluenzaSubtype.H3N2)

        assert "subtype" in results or "n_sequences" in results

    def test_drug_resistance_scores(self, analyzer):
        """Test drug resistance scores."""
        test_seq = "M" * 469
        test_sequences = {InfluenzaGene.NA: [test_seq]}

        results = analyzer.analyze(test_sequences)

        if "drug_resistance" in results:
            for drug, data in results["drug_resistance"].items():
                for score in data.get("scores", []):
                    assert 0.0 <= score <= 1.0


class TestInfluenzaSyntheticDataset:
    """Test synthetic dataset generation."""

    def test_create_synthetic_dataset_default(self):
        """Test default synthetic dataset creation."""
        X, y, ids = create_influenza_synthetic_dataset()

        assert len(X) > 0
        assert len(y) == len(X)
        assert len(ids) == len(X)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_create_synthetic_dataset_min_samples(self):
        """Test minimum samples parameter."""
        min_samples = 30
        X, y, ids = create_influenza_synthetic_dataset(min_samples=min_samples)

        assert len(X) >= min_samples

    def test_create_synthetic_dataset_specific_drug(self):
        """Test dataset for specific drug."""
        X, y, ids = create_influenza_synthetic_dataset(drug=InfluenzaDrug.OSELTAMIVIR)

        assert len(X) > 0
        assert len(y) == len(X)

    def test_synthetic_labels_range(self):
        """Test synthetic labels are in valid range."""
        X, y, ids = create_influenza_synthetic_dataset()

        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_synthetic_encoding_valid(self):
        """Test synthetic encodings are valid."""
        X, y, ids = create_influenza_synthetic_dataset()

        assert X.min() >= 0.0
        assert X.max() <= 1.0


class TestInfluenzaValidation:
    """Test validation methods."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return InfluenzaAnalyzer()

    def test_validate_predictions_exists(self, analyzer):
        """Test validate_predictions method exists."""
        assert hasattr(analyzer, "validate_predictions")


class TestInfluenzaIntegration:
    """Integration tests for Influenza analyzer."""

    def test_full_pipeline(self):
        """Test full analysis pipeline."""
        # Create synthetic data
        X, y, ids = create_influenza_synthetic_dataset(min_samples=20)

        # Create analyzer
        analyzer = InfluenzaAnalyzer()

        # Encode a sample sequence
        test_seq = "MNPNQKIITIGSVSLTISTIC" * 20
        encoding = analyzer.encode_sequence(test_seq, max_length=469)

        assert len(encoding) > 0
        assert encoding.dtype == np.float32

    def test_reproducibility(self):
        """Test dataset generation is reproducible."""
        X1, y1, ids1 = create_influenza_synthetic_dataset(min_samples=20)
        X2, y2, ids2 = create_influenza_synthetic_dataset(min_samples=20)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
