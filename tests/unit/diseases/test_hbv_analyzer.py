# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Comprehensive unit tests for HBV (Hepatitis B Virus) Analyzer.

Tests cover:
- Analyzer initialization
- RT domain mutation detection
- Synthetic dataset generation
- Cross-resistance analysis
- S gene impact analysis (vaccine escape)
"""

import numpy as np
import pytest

from src.diseases.hbv_analyzer import (
    HBVAnalyzer,
    HBVConfig,
    HBVGene,
    HBVDrug,
    HBVGenotype,
    create_hbv_synthetic_dataset,
    RT_MUTATIONS,
    CROSS_RESISTANCE,
)


class TestHBVEnums:
    """Test HBV enumerations."""

    def test_hbv_genes(self):
        """Test HBV gene definitions."""
        genes = list(HBVGene)
        assert len(genes) >= 4

        assert HBVGene.P in genes  # Polymerase (RT domain)
        assert HBVGene.S in genes  # Surface antigen
        assert HBVGene.C in genes  # Core
        assert HBVGene.X in genes  # X protein

    def test_hbv_drugs(self):
        """Test HBV drug definitions."""
        drugs = list(HBVDrug)
        assert len(drugs) >= 6

        # First-line drugs
        assert HBVDrug.ENTECAVIR in drugs
        assert HBVDrug.TENOFOVIR_DF in drugs or HBVDrug.TENOFOVIR_AF in drugs

        # Older drugs
        assert HBVDrug.LAMIVUDINE in drugs
        assert HBVDrug.ADEFOVIR in drugs

    def test_hbv_genotypes(self):
        """Test HBV genotype definitions."""
        genotypes = list(HBVGenotype)
        assert len(genotypes) >= 8

        # Main genotypes A-H
        assert HBVGenotype.A in genotypes
        assert HBVGenotype.B in genotypes
        assert HBVGenotype.C in genotypes
        assert HBVGenotype.D in genotypes


class TestHBVMutationDatabase:
    """Test the HBV mutation database."""

    def test_rt_mutations_structure(self):
        """Test RT mutation database structure."""
        assert len(RT_MUTATIONS) > 0

        for pos, info in RT_MUTATIONS.items():
            assert isinstance(pos, int)
            assert pos > 0
            ref_aa = list(info.keys())[0]
            assert "mutations" in info[ref_aa]
            assert "effect" in info[ref_aa]
            assert "drugs" in info[ref_aa]

    def test_key_resistance_positions(self):
        """Test key resistance mutation positions are included."""
        key_positions = [180, 204, 184, 250]  # LAM/ETV resistance

        for pos in key_positions:
            assert pos in RT_MUTATIONS, f"Missing key position {pos}"

    def test_rt_m204vi_mutation(self):
        """Test M204V/I (YMDD motif) mutation."""
        assert 204 in RT_MUTATIONS
        info = RT_MUTATIONS[204]
        ref_aa = list(info.keys())[0]
        assert ref_aa == "M"
        assert "V" in info[ref_aa]["mutations"] or "I" in info[ref_aa]["mutations"]
        assert info[ref_aa]["effect"] == "high"

    def test_rt_l180m_mutation(self):
        """Test L180M mutation (LAM resistance)."""
        assert 180 in RT_MUTATIONS
        info = RT_MUTATIONS[180]
        ref_aa = list(info.keys())[0]
        assert ref_aa == "L"
        assert "M" in info[ref_aa]["mutations"]

    def test_cross_resistance_patterns(self):
        """Test cross-resistance pattern definitions."""
        assert len(CROSS_RESISTANCE) > 0

        # LAM and LdT should be cross-resistant
        if "lamivudine" in CROSS_RESISTANCE:
            assert "telbivudine" in CROSS_RESISTANCE["lamivudine"]

        # TDF and TAF should be cross-resistant
        if "tenofovir_df" in CROSS_RESISTANCE:
            assert "tenofovir_af" in CROSS_RESISTANCE["tenofovir_df"]


class TestHBVAnalyzer:
    """Test HBVAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return HBVAnalyzer()

    @pytest.fixture
    def config(self):
        """Create config fixture."""
        return HBVConfig()

    def test_initialization_default(self, analyzer):
        """Test default initialization."""
        assert analyzer is not None
        assert analyzer.config is not None
        assert hasattr(analyzer, "aa_alphabet")
        assert hasattr(analyzer, "aa_to_idx")

    def test_initialization_with_config(self, config):
        """Test initialization with custom config."""
        analyzer = HBVAnalyzer(config=config)
        assert analyzer.config == config

    def test_config_defaults(self, config):
        """Test config default values."""
        assert config.name == "hbv"
        assert "Hepatitis B" in config.display_name

    def test_amino_acid_alphabet(self, analyzer):
        """Test amino acid alphabet."""
        assert len(analyzer.aa_alphabet) >= 20
        assert "A" in analyzer.aa_alphabet
        assert "M" in analyzer.aa_alphabet

    def test_encode_sequence(self, analyzer):
        """Test sequence encoding."""
        test_seq = "MPLSYQHFRKLLLLDDEAGPLEEELPRLADEGLNRRVAEDLNLG"
        encoding = analyzer.encode_sequence(test_seq)

        assert isinstance(encoding, np.ndarray)
        assert encoding.dtype == np.float32
        assert encoding.max() <= 1.0
        assert encoding.min() >= 0.0

    def test_encode_sequence_max_length(self, analyzer):
        """Test encoding with max_length parameter."""
        test_seq = "MK" * 200  # 400 AA
        max_length = 350
        encoding = analyzer.encode_sequence(test_seq, max_length=max_length)

        n_aa = len(analyzer.aa_alphabet)
        expected_size = max_length * n_aa
        assert len(encoding) == expected_size


class TestHBVDrugResistance:
    """Test drug resistance prediction."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return HBVAnalyzer()

    def test_analyze_basic(self, analyzer):
        """Test basic analyze functionality."""
        test_seq = "M" * 350  # RT domain length
        test_sequences = {HBVGene.P: [test_seq]}

        results = analyzer.analyze(test_sequences)

        assert isinstance(results, dict)
        assert "n_sequences" in results
        assert "drug_resistance" in results

    def test_analyze_all_drugs(self, analyzer):
        """Test analyze returns results for all drugs."""
        test_seq = "M" * 350
        test_sequences = {HBVGene.P: [test_seq]}

        results = analyzer.analyze(test_sequences)

        assert "drug_resistance" in results
        # Should have resistance data for multiple drugs
        assert len(results["drug_resistance"]) >= 1

    def test_resistance_score_range(self, analyzer):
        """Test resistance scores are in valid range."""
        test_seq = "M" * 350
        test_sequences = {HBVGene.P: [test_seq]}

        results = analyzer.analyze(test_sequences)

        for drug, data in results["drug_resistance"].items():
            for score in data["scores"]:
                assert 0.0 <= score <= 1.0, f"Score {score} out of range for {drug}"

    def test_analyze_with_genotype(self, analyzer):
        """Test analyze with genotype parameter."""
        test_seq = "M" * 350
        test_sequences = {HBVGene.P: [test_seq]}

        results = analyzer.analyze(test_sequences, genotype=HBVGenotype.D)

        assert results["genotype"] == "D"


class TestHBVCrossResistance:
    """Test cross-resistance analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return HBVAnalyzer()

    def test_cross_resistance_analysis(self, analyzer):
        """Test cross-resistance is analyzed."""
        test_seq = "M" * 350
        test_sequences = {HBVGene.P: [test_seq]}

        results = analyzer.analyze(test_sequences)

        assert "cross_resistance" in results


class TestHBVSGeneImpact:
    """Test S gene impact analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return HBVAnalyzer()

    def test_s_gene_impact_analysis(self, analyzer):
        """Test S gene impact is analyzed."""
        test_seq = "M" * 350
        test_sequences = {HBVGene.P: [test_seq]}

        results = analyzer.analyze(test_sequences)

        assert "s_gene_impact" in results


class TestHBVSyntheticDataset:
    """Test synthetic dataset generation."""

    def test_create_synthetic_dataset_default(self):
        """Test default synthetic dataset creation."""
        X, y, ids = create_hbv_synthetic_dataset()

        assert len(X) > 0
        assert len(y) == len(X)
        assert len(ids) == len(X)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_create_synthetic_dataset_min_samples(self):
        """Test minimum samples parameter."""
        min_samples = 30
        X, y, ids = create_hbv_synthetic_dataset(min_samples=min_samples)

        assert len(X) >= min_samples

    def test_create_synthetic_dataset_specific_drug(self):
        """Test dataset for specific drug."""
        X, y, ids = create_hbv_synthetic_dataset(drug=HBVDrug.ENTECAVIR)

        assert len(X) > 0
        assert len(y) == len(X)

    def test_synthetic_labels_range(self):
        """Test synthetic labels are in valid range."""
        X, y, ids = create_hbv_synthetic_dataset()

        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_synthetic_encoding_valid(self):
        """Test synthetic encodings are valid."""
        X, y, ids = create_hbv_synthetic_dataset()

        # Check encoding values
        assert X.min() >= 0.0
        assert X.max() <= 1.0


class TestHBVValidation:
    """Test validation methods."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return HBVAnalyzer()

    def test_validate_predictions_exists(self, analyzer):
        """Test validate_predictions method exists."""
        assert hasattr(analyzer, "validate_predictions")

    def test_validate_predictions_returns_metrics(self, analyzer):
        """Test validation returns metrics."""
        predictions = {
            "drug_resistance": {
                "entecavir": {"scores": [0.1, 0.5, 0.9]},
            }
        }
        ground_truth = {
            "entecavir": [0.0, 0.5, 1.0],
        }

        metrics = analyzer.validate_predictions(predictions, ground_truth)
        assert isinstance(metrics, dict)


class TestHBVIntegration:
    """Integration tests for HBV analyzer."""

    def test_full_pipeline(self):
        """Test full analysis pipeline."""
        # Create synthetic data
        X, y, ids = create_hbv_synthetic_dataset(min_samples=20)

        # Create analyzer
        analyzer = HBVAnalyzer()

        # Encode a sample sequence
        test_seq = "MPLSYQHFRKLLLLDDEAGPLEEELPRL" * 10
        encoding = analyzer.encode_sequence(test_seq, max_length=350)

        assert len(encoding) > 0
        assert encoding.dtype == np.float32

    def test_reproducibility(self):
        """Test dataset generation is reproducible."""
        X1, y1, ids1 = create_hbv_synthetic_dataset(min_samples=20)
        X2, y2, ids2 = create_hbv_synthetic_dataset(min_samples=20)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
