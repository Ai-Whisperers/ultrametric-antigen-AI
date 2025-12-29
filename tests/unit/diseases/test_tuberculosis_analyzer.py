# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Comprehensive unit tests for Tuberculosis (M. tuberculosis) Analyzer.

Tests cover:
- Analyzer initialization and configuration
- Drug resistance mutation detection
- Synthetic dataset generation
- Multi-drug resistance patterns
- Cross-resistance analysis
"""

import numpy as np
import pytest

from src.diseases.tuberculosis_analyzer import (
    TuberculosisAnalyzer,
    TuberculosisConfig,
    TBGene,
    TBDrug,
    TBDrugCategory,
    create_tb_synthetic_dataset,
    GENE_MUTATIONS,
    DRUG_TO_GENE,
)


class TestTBEnums:
    """Test TB enumerations."""

    def test_tb_genes(self):
        """Test all TB genes are defined."""
        genes = list(TBGene)
        assert len(genes) >= 6
        assert TBGene.RPOB in genes  # RIF target
        assert TBGene.KATG in genes  # INH target
        assert TBGene.INHA in genes  # INH/ETH target
        assert TBGene.GYRA in genes  # FQ target

    def test_tb_drugs(self):
        """Test all TB drugs are defined."""
        drugs = list(TBDrug)
        assert len(drugs) >= 8

        # First-line drugs
        assert TBDrug.ISONIAZID in drugs
        assert TBDrug.RIFAMPICIN in drugs
        assert TBDrug.ETHAMBUTOL in drugs
        assert TBDrug.PYRAZINAMIDE in drugs

        # Second-line drugs
        assert TBDrug.LEVOFLOXACIN in drugs or TBDrug.MOXIFLOXACIN in drugs

    def test_drug_categories(self):
        """Test drug category definitions."""
        categories = list(TBDrugCategory)
        assert TBDrugCategory.FIRST_LINE in categories
        assert TBDrugCategory.SECOND_LINE_INJECTABLE in categories


class TestTBMutationDatabase:
    """Test the TB mutation database."""

    def test_gene_mutations_structure(self):
        """Test mutation database structure."""
        assert len(GENE_MUTATIONS) > 0

        for gene, positions in GENE_MUTATIONS.items():
            assert isinstance(gene, TBGene)
            assert isinstance(positions, dict)
            assert len(positions) > 0

    def test_rpob_mutations(self):
        """Test rpoB mutations for rifampicin."""
        assert TBGene.RPOB in GENE_MUTATIONS
        rpob_muts = GENE_MUTATIONS[TBGene.RPOB]

        # Key RRDR mutations
        key_positions = [531, 526, 516]
        for pos in key_positions:
            if pos in rpob_muts:
                info = rpob_muts[pos]
                ref_aa = list(info.keys())[0]
                assert "mutations" in info[ref_aa]
                assert "effect" in info[ref_aa]
                assert "drugs" in info[ref_aa]

    def test_katg_mutations(self):
        """Test katG mutations for isoniazid."""
        assert TBGene.KATG in GENE_MUTATIONS
        katg_muts = GENE_MUTATIONS[TBGene.KATG]

        # S315T is the most common INH resistance mutation
        if 315 in katg_muts:
            info = katg_muts[315]
            ref_aa = list(info.keys())[0]
            assert "T" in info[ref_aa]["mutations"] or "N" in info[ref_aa]["mutations"]

    def test_drug_to_gene_mapping(self):
        """Test drug to gene mapping."""
        assert len(DRUG_TO_GENE) > 0

        # RIF targets rpoB
        if TBDrug.RIFAMPICIN in DRUG_TO_GENE:
            assert TBGene.RPOB in DRUG_TO_GENE[TBDrug.RIFAMPICIN]

        # INH targets katG and inhA
        if TBDrug.ISONIAZID in DRUG_TO_GENE:
            genes = DRUG_TO_GENE[TBDrug.ISONIAZID]
            assert TBGene.KATG in genes or TBGene.INHA in genes


class TestTuberculosisAnalyzer:
    """Test TuberculosisAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return TuberculosisAnalyzer()

    @pytest.fixture
    def config(self):
        """Create config fixture."""
        return TuberculosisConfig()

    def test_initialization_default(self, analyzer):
        """Test default initialization."""
        assert analyzer is not None
        assert analyzer.config is not None
        assert hasattr(analyzer, "aa_alphabet")
        assert hasattr(analyzer, "aa_to_idx")

    def test_initialization_with_config(self, config):
        """Test initialization with custom config."""
        analyzer = TuberculosisAnalyzer(config=config)
        assert analyzer.config == config

    def test_config_defaults(self, config):
        """Test config default values."""
        assert config.name == "tuberculosis" or config.name == "tb"
        assert "M. tuberculosis" in config.display_name or "Tuberculosis" in config.display_name

    def test_amino_acid_alphabet(self, analyzer):
        """Test amino acid alphabet."""
        assert len(analyzer.aa_alphabet) >= 20
        assert "A" in analyzer.aa_alphabet
        assert "M" in analyzer.aa_alphabet
        assert "-" in analyzer.aa_alphabet or "X" in analyzer.aa_alphabet

    def test_encode_sequence(self, analyzer):
        """Test sequence encoding."""
        test_seq = "MKLVFLVLLFLGALGLCLA"
        encoding = analyzer.encode_gene_sequence(test_seq)

        assert isinstance(encoding, np.ndarray)
        assert encoding.dtype == np.float32
        assert encoding.max() <= 1.0
        assert encoding.min() >= 0.0

    def test_encode_sequence_with_unknown(self, analyzer):
        """Test encoding with unknown amino acids."""
        test_seq = "MKXYZ"  # Contains unknown AAs
        encoding = analyzer.encode_gene_sequence(test_seq)

        assert isinstance(encoding, np.ndarray)
        # Should handle unknowns gracefully

    def test_encode_sequence_padding(self, analyzer):
        """Test sequence padding."""
        short_seq = "MK"
        max_length = 100
        encoding = analyzer.encode_gene_sequence(short_seq, max_length=max_length)

        n_aa = len(analyzer.aa_alphabet)
        expected_size = max_length * n_aa
        assert len(encoding) == expected_size


class TestTBDrugResistance:
    """Test drug resistance prediction."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return TuberculosisAnalyzer()

    def test_predict_drug_resistance(self, analyzer):
        """Test resistance prediction method exists."""
        # Create test sequences
        test_sequences = {
            TBGene.RPOB: ["MKLVFLVLLFLGALGLCLA" * 5],
        }

        results = analyzer.analyze(test_sequences)

        assert isinstance(results, dict)
        assert "drug_resistance" in results or "n_sequences" in results

    def test_analyze_returns_expected_keys(self, analyzer):
        """Test analyze returns expected result keys."""
        test_sequences = {
            TBGene.KATG: ["MKTEFPSASLYQNIDVLYQ" * 10],
        }

        results = analyzer.analyze(test_sequences)

        expected_keys = ["n_isolates", "genes_analyzed"]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

    def test_resistance_score_range(self, analyzer):
        """Test resistance scores are in valid range."""
        test_sequences = {
            TBGene.RPOB: ["MKLVFLVLLFLGALGLCLA" * 10],
        }

        results = analyzer.analyze(test_sequences)

        if "drug_resistance" in results:
            for drug, data in results["drug_resistance"].items():
                if "scores" in data:
                    for score in data["scores"]:
                        assert 0.0 <= score <= 1.0, f"Score {score} out of range for {drug}"


class TestTBSyntheticDataset:
    """Test synthetic dataset generation."""

    def test_create_synthetic_dataset_default(self):
        """Test default synthetic dataset creation."""
        X, y, ids = create_tb_synthetic_dataset()

        assert len(X) > 0
        assert len(y) == len(X)
        assert len(ids) == len(X)

        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_create_synthetic_dataset_min_samples(self):
        """Test minimum samples parameter."""
        min_samples = 30
        X, y, ids = create_tb_synthetic_dataset(min_samples=min_samples)

        assert len(X) >= min_samples

    def test_create_synthetic_dataset_specific_drug(self):
        """Test dataset for specific drug."""
        X, y, ids = create_tb_synthetic_dataset(drug=TBDrug.RIFAMPICIN)

        assert len(X) > 0
        assert len(y) == len(X)

    def test_synthetic_labels_range(self):
        """Test synthetic labels are in valid range."""
        X, y, ids = create_tb_synthetic_dataset()

        assert y.min() >= 0.0
        assert y.max() <= 1.0

    def test_synthetic_encoding_valid(self):
        """Test synthetic encodings are valid."""
        X, y, ids = create_tb_synthetic_dataset()

        # Check encoding values
        assert X.min() >= 0.0
        assert X.max() <= 1.0

        # Check for reasonable sparsity (one-hot encoding)
        row_sums = np.sum(X, axis=1)
        assert np.all(row_sums > 0)


class TestTBMDRPatterns:
    """Test multi-drug resistance pattern detection."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return TuberculosisAnalyzer()

    def test_mdr_detection(self, analyzer):
        """Test MDR-TB detection (INH + RIF resistant)."""
        # This would require sequences with both katG and rpoB mutations
        # For now, test that analyzer can process multiple genes
        test_sequences = {
            TBGene.RPOB: ["MKLVFLVLLFLGALGLCLA" * 10],
            TBGene.KATG: ["MKTEFPSASLYQNIDVLYQ" * 10],
        }

        results = analyzer.analyze(test_sequences)
        assert "genes_analyzed" in results
        assert len(results["genes_analyzed"]) >= 1


class TestTBValidation:
    """Test validation methods."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return TuberculosisAnalyzer()

    def test_validate_predictions_method_exists(self, analyzer):
        """Test validate_predictions method exists."""
        assert hasattr(analyzer, "validate_predictions")

    def test_validate_predictions_returns_metrics(self, analyzer):
        """Test validation returns metrics."""
        predictions = {
            "drug_resistance": {
                "rifampicin": {"scores": [0.1, 0.5, 0.9]},
            }
        }
        ground_truth = {
            "rifampicin": [0.0, 0.5, 1.0],
        }

        try:
            metrics = analyzer.validate_predictions(predictions, ground_truth)
            assert isinstance(metrics, dict)
        except NotImplementedError:
            pytest.skip("validate_predictions not implemented")


class TestTBIntegration:
    """Integration tests for TB analyzer."""

    def test_full_pipeline(self):
        """Test full analysis pipeline."""
        # Create synthetic data
        X, y, ids = create_tb_synthetic_dataset(min_samples=20)

        # Create analyzer
        analyzer = TuberculosisAnalyzer()

        # Encode a sample sequence
        test_seq = "MKLVFLVLLFLGALGLCLA" * 10
        encoding = analyzer.encode_gene_sequence(test_seq)

        assert len(encoding) > 0
        assert encoding.dtype == np.float32

    def test_reproducibility(self):
        """Test dataset generation is reproducible."""
        X1, y1, ids1 = create_tb_synthetic_dataset(min_samples=20)
        X2, y2, ids2 = create_tb_synthetic_dataset(min_samples=20)

        # Should be reproducible with same seed
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
