# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for N. gonorrhoeae Analyzer.

Tests cover:
- Drug resistance prediction
- Efflux pump analysis
- MDR/XDR classification
- Treatment option assessment
"""

from __future__ import annotations

import pytest

from src.diseases.gonorrhoeae_analyzer import (
    GonorrhoeaeAnalyzer,
    GonorrhoeaeConfig,
    GCDrug,
    GCGene,
    GCSequenceType,
)


class TestGCSequenceType:
    """Tests for GCSequenceType enum."""

    def test_sequence_type_values(self):
        """Test sequence type enum values."""
        assert GCSequenceType.ST1901.value == "ST1901"
        assert GCSequenceType.ST7363.value == "ST7363"
        assert GCSequenceType.ST1407.value == "ST1407"


class TestGCGene:
    """Tests for GCGene enum."""

    def test_resistance_genes(self):
        """Test resistance gene values."""
        assert GCGene.PENA.value == "penA"
        assert GCGene.GYRA.value == "gyrA"
        assert GCGene.MTRR.value == "mtrR"

    def test_ribosomal_genes(self):
        """Test ribosomal gene values."""
        assert GCGene.RRL_23S.value == "23S_rRNA"
        assert GCGene.RPSJ.value == "rpsJ"


class TestGCDrug:
    """Tests for GCDrug enum."""

    def test_first_line_drugs(self):
        """Test first-line drug values."""
        assert GCDrug.CEFTRIAXONE.value == "ceftriaxone"
        assert GCDrug.AZITHROMYCIN.value == "azithromycin"

    def test_alternative_drugs(self):
        """Test alternative drug values."""
        assert GCDrug.GENTAMICIN.value == "gentamicin"
        assert GCDrug.SPECTINOMYCIN.value == "spectinomycin"


class TestGonorrhoeaeConfig:
    """Tests for GonorrhoeaeConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = GonorrhoeaeConfig()
        assert config.name == "gonorrhoeae"
        assert config.display_name == "Neisseria gonorrhoeae (Gonorrhea)"
        assert "pubmlst" in config.data_sources

    def test_disease_type(self):
        """Test disease type is bacterial."""
        config = GonorrhoeaeConfig()
        from src.diseases.base import DiseaseType
        assert config.disease_type == DiseaseType.BACTERIAL

    def test_gc_specific_settings(self):
        """Test N. gonorrhoeae-specific configuration options."""
        config = GonorrhoeaeConfig()
        assert config.predict_mdr is True
        assert config.predict_xdr is True
        assert config.assess_treatment_options is True


class TestGonorrhoeaeAnalyzer:
    """Tests for GonorrhoeaeAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return GonorrhoeaeAnalyzer()

    @pytest.fixture
    def sample_pena_sequence(self):
        """Create sample PenA sequence."""
        return "MKKLLFAAVVLALLAGCSNAADDKKTSAVPVKVKAGAK" * 15

    @pytest.fixture
    def sample_gyra_sequence(self):
        """Create sample GyrA sequence."""
        return "MSDLAREITPVNIEEELKSSYLDYAMSVIVGRALPDVRDG" * 10

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.config.name == "gonorrhoeae"
        assert len(analyzer.aa_alphabet) > 0

    def test_analyze_empty_sequences(self, analyzer):
        """Test analysis with empty sequences."""
        results = analyzer.analyze({})
        assert results["n_sequences"] == 0
        assert results["genes_analyzed"] == []

    def test_analyze_single_gene(self, analyzer, sample_pena_sequence):
        """Test analysis with single gene."""
        sequences = {GCGene.PENA: [sample_pena_sequence]}
        results = analyzer.analyze(sequences)

        assert results["n_sequences"] == 1
        assert GCGene.PENA.value in results["genes_analyzed"]

    def test_analyze_with_sequence_type(self, analyzer, sample_pena_sequence):
        """Test analysis with specified sequence type."""
        sequences = {GCGene.PENA: [sample_pena_sequence]}
        results = analyzer.analyze(sequences, sequence_type=GCSequenceType.ST1901)

        assert results["sequence_type"] == "ST1901"


class TestDrugResistance:
    """Tests for drug resistance prediction."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return GonorrhoeaeAnalyzer()

    @pytest.fixture
    def pena_sequence(self):
        """Create PenA sequence."""
        return "A" * 600

    @pytest.fixture
    def gyra_sequence(self):
        """Create GyrA sequence."""
        return "S" * 300

    def test_predict_ceftriaxone_resistance(self, analyzer, pena_sequence):
        """Test ceftriaxone resistance prediction."""
        results = analyzer.predict_drug_resistance(
            [pena_sequence],
            GCDrug.CEFTRIAXONE,
            GCGene.PENA,
        )

        assert "scores" in results
        assert "classifications" in results
        assert "mutations" in results
        assert "mic_predictions" in results
        assert len(results["scores"]) == 1

    def test_predict_azithromycin_resistance(self, analyzer):
        """Test azithromycin resistance prediction."""
        # 23S rRNA sequence
        rrl_sequence = "A" * 3000
        results = analyzer.predict_drug_resistance(
            [rrl_sequence],
            GCDrug.AZITHROMYCIN,
            GCGene.RRL_23S,
        )

        assert "scores" in results
        assert len(results["scores"]) == 1

    def test_predict_ciprofloxacin_resistance(self, analyzer, gyra_sequence):
        """Test ciprofloxacin resistance prediction."""
        results = analyzer.predict_drug_resistance(
            [gyra_sequence],
            GCDrug.CIPROFLOXACIN,
            GCGene.GYRA,
        )

        assert "scores" in results
        assert len(results["scores"]) == 1

    def test_resistance_classification(self, analyzer, pena_sequence):
        """Test resistance classification levels."""
        results = analyzer.predict_drug_resistance(
            [pena_sequence],
            GCDrug.CEFTRIAXONE,
            GCGene.PENA,
        )

        classification = results["classifications"][0]
        assert classification in ["susceptible", "reduced_susceptibility", "resistant"]

    def test_mic_prediction(self, analyzer, pena_sequence):
        """Test MIC prediction."""
        results = analyzer.predict_drug_resistance(
            [pena_sequence],
            GCDrug.CEFTRIAXONE,
            GCGene.PENA,
        )

        mic = results["mic_predictions"][0]
        assert mic in ["<=0.016", "0.06-0.125", "0.125-0.25", ">=0.25"]


class TestEffluxAnalysis:
    """Tests for efflux pump analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return GonorrhoeaeAnalyzer()

    def test_efflux_analysis(self, analyzer):
        """Test MtrR efflux pump analysis."""
        mtrr_sequence = "MAKKPLVIGDLFTPNARITLRGLDVKRLKQPFRKG" * 5
        results = analyzer._analyze_efflux([mtrr_sequence])

        assert "overexpression_risk" in results
        assert "mutations_detected" in results
        assert len(results["overexpression_risk"]) == 1

    def test_efflux_risk_levels(self, analyzer):
        """Test efflux risk level values."""
        mtrr_sequence = "A" * 100
        results = analyzer._analyze_efflux([mtrr_sequence])

        risk = results["overexpression_risk"][0]
        assert risk in ["low", "moderate", "high"]


class TestMDRClassification:
    """Tests for MDR/XDR classification."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return GonorrhoeaeAnalyzer()

    def test_susceptible_classification(self, analyzer):
        """Test susceptible classification."""
        resistance_data = {}
        results = analyzer._classify_mdr(resistance_data)

        assert results["classification"] == "susceptible"

    def test_mdr_classification(self, analyzer):
        """Test MDR classification."""
        resistance_data = {
            "ceftriaxone": {"classifications": ["susceptible"]},
            "azithromycin": {"classifications": ["resistant"]},
            "ciprofloxacin": {"classifications": ["resistant"]},
            "penicillin": {"classifications": ["resistant"]},
        }
        results = analyzer._classify_mdr(resistance_data)

        assert results["classification"] == "MDR"
        assert len(results["resistance_classes"]) >= 3

    def test_xdr_classification(self, analyzer):
        """Test XDR classification."""
        resistance_data = {
            "ceftriaxone": {"classifications": ["resistant"]},
            "azithromycin": {"classifications": ["resistant"]},
        }
        results = analyzer._classify_mdr(resistance_data)

        assert results["classification"] == "XDR"


class TestTreatmentOptions:
    """Tests for treatment option assessment."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return GonorrhoeaeAnalyzer()

    def test_standard_treatment(self, analyzer):
        """Test standard treatment for susceptible strain."""
        resistance_data = {
            "ceftriaxone": {"classifications": ["susceptible"]},
            "azithromycin": {"classifications": ["susceptible"]},
        }
        options = analyzer._assess_treatment_options(resistance_data)

        assert len(options["recommended"]) > 0
        assert "Ceftriaxone" in options["recommended"][0]

    def test_azithromycin_resistant_treatment(self, analyzer):
        """Test treatment for azithromycin-resistant strain."""
        resistance_data = {
            "ceftriaxone": {"classifications": ["susceptible"]},
            "azithromycin": {"classifications": ["resistant"]},
        }
        options = analyzer._assess_treatment_options(resistance_data)

        assert len(options["notes"]) > 0
        assert any("Azithromycin" in c for c in options["contraindicated"])

    def test_xdr_treatment(self, analyzer):
        """Test treatment options for XDR strain."""
        resistance_data = {
            "ceftriaxone": {"classifications": ["resistant"]},
            "azithromycin": {"classifications": ["resistant"]},
        }
        options = analyzer._assess_treatment_options(resistance_data)

        assert len(options["alternative"]) > 0
        assert "Ceftriaxone" in options["contraindicated"]


class TestIntegration:
    """Integration tests for N. gonorrhoeae Analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return GonorrhoeaeAnalyzer()

    def test_full_analysis_workflow(self, analyzer):
        """Test complete analysis workflow."""
        sequences = {
            GCGene.PENA: ["MKKLLFAAVVLALLAGCSNAADDKKTSAVPVKVKAGAK" * 15],
            GCGene.GYRA: ["MSDLAREITPVNIEEELKSSYLDYAMSVIVGRALPDVRDG" * 10],
            GCGene.MTRR: ["MAKKPLVIGDLFTPNARITLRGLDVKRLKQPFRKG" * 5],
        }

        results = analyzer.analyze(sequences, sequence_type=GCSequenceType.ST1901)

        assert results["n_sequences"] == 1
        assert "drug_resistance" in results
        assert "efflux_status" in results
        assert "mdr_classification" in results
        assert "treatment_options" in results

    def test_batch_analysis(self, analyzer):
        """Test analysis of multiple sequences."""
        n_seqs = 3
        sequences = {
            GCGene.PENA: ["MKKLLFAAVVLALLAGCSNAADDKKTSAVPVKVKAGAK" * 15] * n_seqs,
        }

        results = analyzer.analyze(sequences)

        assert results["n_sequences"] == n_seqs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
