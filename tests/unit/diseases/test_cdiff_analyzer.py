# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for C. difficile Analyzer.

Tests cover:
- Ribotype classification
- Drug resistance prediction
- Toxin analysis
- Hypervirulence prediction
- Recurrence risk
"""

from __future__ import annotations

import pytest

from src.diseases.cdiff_analyzer import (
    CDiffAnalyzer,
    CDiffConfig,
    CDiffDrug,
    CDiffGene,
    CDiffRibotype,
)


class TestCDiffRibotype:
    """Tests for CDiffRibotype enum."""

    def test_ribotype_values(self):
        """Test ribotype enum values."""
        assert CDiffRibotype.RT027.value == "027"
        assert CDiffRibotype.RT078.value == "078"
        assert CDiffRibotype.RT001.value == "001"
        assert CDiffRibotype.RT017.value == "017"


class TestCDiffGene:
    """Tests for CDiffGene enum."""

    def test_toxin_genes(self):
        """Test toxin gene values."""
        assert CDiffGene.TCDA.value == "tcdA"
        assert CDiffGene.TCDB.value == "tcdB"
        assert CDiffGene.CDTA.value == "cdtA"
        assert CDiffGene.CDTB.value == "cdtB"

    def test_resistance_genes(self):
        """Test resistance gene values."""
        assert CDiffGene.VAN.value == "vanG"
        assert CDiffGene.NIM.value == "nim"
        assert CDiffGene.GYRA.value == "gyrA"


class TestCDiffDrug:
    """Tests for CDiffDrug enum."""

    def test_first_line_drugs(self):
        """Test first-line treatment drug values."""
        assert CDiffDrug.VANCOMYCIN.value == "vancomycin"
        assert CDiffDrug.FIDAXOMICIN.value == "fidaxomicin"

    def test_alternative_drugs(self):
        """Test alternative drug values."""
        assert CDiffDrug.METRONIDAZOLE.value == "metronidazole"
        assert CDiffDrug.RIFAXIMIN.value == "rifaximin"


class TestCDiffConfig:
    """Tests for CDiffConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = CDiffConfig()
        assert config.name == "cdiff"
        assert config.display_name == "Clostridioides difficile Infection"
        assert "pubmlst" in config.data_sources

    def test_disease_type(self):
        """Test disease type is bacterial."""
        config = CDiffConfig()
        from src.diseases.base import DiseaseType
        assert config.disease_type == DiseaseType.BACTERIAL

    def test_cdiff_specific_settings(self):
        """Test C. diff-specific configuration options."""
        config = CDiffConfig()
        assert config.predict_toxin_expression is True
        assert config.predict_hypervirulence is True
        assert config.classify_ribotype is True


class TestCDiffAnalyzer:
    """Tests for CDiffAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return CDiffAnalyzer()

    @pytest.fixture
    def sample_tcda_sequence(self):
        """Create sample TcdA sequence."""
        # TcdA is ~2710 amino acids
        return "MKISNVKKTYDIKEEEIIKNLNKVNNILKKINEYLANIPKDLKKELE" * 58

    @pytest.fixture
    def sample_tcdb_sequence(self):
        """Create sample TcdB sequence."""
        # TcdB is ~2366 amino acids
        return "MSLENIFKDIPKVNDLLKNIKKNQKNLIKNINEYLANIPKDLKKELE" * 50

    @pytest.fixture
    def sample_rpob_sequence(self):
        """Create sample RpoB sequence."""
        return "MKLIIFLPDGSRFTYVDGPRK" * 30

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.config.name == "cdiff"
        assert len(analyzer.aa_alphabet) > 0

    def test_analyze_empty_sequences(self, analyzer):
        """Test analysis with empty sequences."""
        results = analyzer.analyze({})
        assert results["n_sequences"] == 0
        assert results["genes_analyzed"] == []

    def test_analyze_single_gene(self, analyzer, sample_rpob_sequence):
        """Test analysis with single gene."""
        sequences = {CDiffGene.RPOB: [sample_rpob_sequence]}
        results = analyzer.analyze(sequences)

        assert results["n_sequences"] == 1
        assert CDiffGene.RPOB.value in results["genes_analyzed"]

    def test_analyze_with_ribotype(self, analyzer, sample_tcda_sequence):
        """Test analysis with specified ribotype."""
        sequences = {CDiffGene.TCDA: [sample_tcda_sequence]}
        results = analyzer.analyze(sequences, ribotype=CDiffRibotype.RT027)

        assert results["ribotype"] == "027"

    def test_analyze_with_prior_cdi(self, analyzer, sample_tcda_sequence):
        """Test analysis with prior CDI history."""
        sequences = {CDiffGene.TCDA: [sample_tcda_sequence]}
        results = analyzer.analyze(sequences, prior_cdi=True)

        assert results["prior_cdi"] is True


class TestDrugResistance:
    """Tests for drug resistance prediction."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return CDiffAnalyzer()

    @pytest.fixture
    def van_sequence(self):
        """Create vanG sequence."""
        return "A" * 200

    @pytest.fixture
    def rpob_sequence(self):
        """Create RpoB sequence."""
        return "D" * 600

    def test_predict_vancomycin_resistance(self, analyzer, van_sequence):
        """Test vancomycin resistance prediction."""
        results = analyzer.predict_drug_resistance(
            [van_sequence],
            CDiffDrug.VANCOMYCIN,
            CDiffGene.VAN,
        )

        assert "scores" in results
        assert "classifications" in results
        assert "mutations" in results
        assert len(results["scores"]) == 1

    def test_predict_fidaxomicin_resistance(self, analyzer, rpob_sequence):
        """Test fidaxomicin resistance prediction."""
        results = analyzer.predict_drug_resistance(
            [rpob_sequence],
            CDiffDrug.FIDAXOMICIN,
            CDiffGene.RPOB,
        )

        assert "scores" in results
        assert len(results["scores"]) == 1

    def test_resistance_classification(self, analyzer, van_sequence):
        """Test resistance classification levels."""
        results = analyzer.predict_drug_resistance(
            [van_sequence],
            CDiffDrug.VANCOMYCIN,
            CDiffGene.VAN,
        )

        classification = results["classifications"][0]
        assert classification in ["susceptible", "reduced_susceptibility", "resistant"]

    def test_multiple_sequences(self, analyzer):
        """Test prediction on multiple sequences."""
        sequences = ["A" * 200, "V" * 200, "T" * 200]
        results = analyzer.predict_drug_resistance(
            sequences,
            CDiffDrug.VANCOMYCIN,
            CDiffGene.VAN,
        )

        assert len(results["scores"]) == 3
        assert len(results["classifications"]) == 3


class TestToxinAnalysis:
    """Tests for toxin gene analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return CDiffAnalyzer()

    @pytest.fixture
    def tcda_sequence(self):
        """Create TcdA sequence."""
        return "MKISNVKKTYDIKEEEIIKNLNKVNNILKKINEYLANIPKDLKKELE" * 58

    @pytest.fixture
    def tcdb_sequence(self):
        """Create TcdB sequence."""
        return "MSLENIFKDIPKVNDLLKNIKKNQKNLIKNINEYLANIPKDLKKELE" * 50

    def test_tcda_analysis(self, analyzer, tcda_sequence):
        """Test TcdA toxin analysis."""
        results = analyzer._analyze_toxin([tcda_sequence], "tcdA")

        assert results["present"] is True
        assert "expression_levels" in results
        assert "domain_integrity" in results
        assert len(results["expression_levels"]) == 1

    def test_tcdb_analysis(self, analyzer, tcdb_sequence):
        """Test TcdB toxin analysis."""
        results = analyzer._analyze_toxin([tcdb_sequence], "tcdB")

        assert results["present"] is True
        assert "expression_levels" in results
        assert len(results["expression_levels"]) == 1

    def test_expression_levels(self, analyzer, tcda_sequence):
        """Test expression level values."""
        results = analyzer._analyze_toxin([tcda_sequence], "tcdA")

        level = results["expression_levels"][0]
        assert level in ["low", "moderate", "high"]

    def test_variant_types(self, analyzer, tcda_sequence):
        """Test variant type classification."""
        results = analyzer._analyze_toxin([tcda_sequence], "tcdA")

        variant = results["variant_type"][0]
        assert variant in ["wild_type", "minor_variant", "major_variant"]


class TestHypervirulence:
    """Tests for hypervirulence prediction."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return CDiffAnalyzer()

    def test_hypervirulence_with_rt027(self, analyzer):
        """Test hypervirulence with RT027 ribotype."""
        sequences = {CDiffGene.TCDA: ["A" * 1000]}
        results = analyzer._predict_hypervirulence(sequences, CDiffRibotype.RT027)

        assert "markers" in results
        assert results["score"] >= 0.5
        assert any("027" in marker for marker in results["markers"])

    def test_hypervirulence_with_binary_toxin(self, analyzer):
        """Test hypervirulence with binary toxin."""
        sequences = {
            CDiffGene.TCDA: ["A" * 1000],
            CDiffGene.CDTA: ["B" * 500],
        }
        results = analyzer._predict_hypervirulence(sequences)

        assert any("Binary toxin" in marker for marker in results["markers"])

    def test_hypervirulence_classification(self, analyzer):
        """Test hypervirulence classification levels."""
        sequences = {CDiffGene.TCDA: ["A" * 1000]}
        results = analyzer._predict_hypervirulence(sequences)

        assert results["classification"] in [
            "hypervirulent",
            "potentially_hypervirulent",
            "non-hypervirulent",
        ]


class TestRibotypeClassification:
    """Tests for ribotype classification."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return CDiffAnalyzer()

    def test_classify_ribotype(self, analyzer):
        """Test ribotype classification."""
        sequences = {CDiffGene.TCDA: ["A" * 1000]}
        results = analyzer._classify_ribotype(sequences)

        assert "predicted_ribotype" in results
        assert "confidence" in results
        assert "markers_detected" in results

    def test_rt027_classification(self, analyzer):
        """Test RT027 classification with markers."""
        # Simulate RT027 markers
        gyra_seq = list("A" * 100)
        gyra_seq[82] = "I"  # Thr83Ile
        sequences = {
            CDiffGene.CDTA: ["B" * 500],
            CDiffGene.TCDC: ["C" * 150],  # Truncated
            CDiffGene.GYRA: ["".join(gyra_seq)],
        }
        results = analyzer._classify_ribotype(sequences)

        assert results["predicted_ribotype"] == "027"


class TestRecurrenceRisk:
    """Tests for recurrence risk prediction."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return CDiffAnalyzer()

    def test_recurrence_with_prior_cdi(self, analyzer):
        """Test recurrence risk with prior CDI."""
        sequences = {CDiffGene.TCDA: ["A" * 1000]}
        results = analyzer._predict_recurrence_risk(sequences, prior_cdi=True, resistance_data={})

        assert any("Prior CDI" in factor for factor in results["risk_factors"])
        assert results["risk_score"] >= 0.4

    def test_recurrence_with_resistance(self, analyzer):
        """Test recurrence risk with drug resistance."""
        sequences = {CDiffGene.TCDA: ["A" * 1000]}
        resistance_data = {
            "vancomycin": {"classifications": ["resistant"]}
        }
        results = analyzer._predict_recurrence_risk(sequences, prior_cdi=False, resistance_data=resistance_data)

        assert any("resistance" in factor for factor in results["risk_factors"])

    def test_recurrence_risk_levels(self, analyzer):
        """Test recurrence risk level classification."""
        sequences = {CDiffGene.TCDA: ["A" * 1000]}
        results = analyzer._predict_recurrence_risk(sequences, prior_cdi=False, resistance_data={})

        assert results["risk_level"] in ["low", "moderate", "high"]


class TestTreatmentRecommendations:
    """Tests for treatment recommendations."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return CDiffAnalyzer()

    def test_recommendations_susceptible(self, analyzer):
        """Test recommendations for susceptible strain."""
        analysis_results = {
            "drug_resistance": {},
            "hypervirulence": {"classification": "non-hypervirulent"},
            "recurrence_risk": {"risk_level": "low"},
        }
        recommendations = analyzer.get_treatment_recommendations(analysis_results)

        assert "first_line" in recommendations
        assert len(recommendations["first_line"]) > 0

    def test_recommendations_hypervirulent(self, analyzer):
        """Test recommendations for hypervirulent strain."""
        analysis_results = {
            "drug_resistance": {},
            "hypervirulence": {"classification": "hypervirulent"},
            "recurrence_risk": {"risk_level": "moderate"},
        }
        recommendations = analyzer.get_treatment_recommendations(analysis_results)

        assert "fidaxomicin" in recommendations["first_line"][0].lower() or \
               any("fidaxomicin" in r.lower() for r in recommendations["first_line"])

    def test_recommendations_resistant(self, analyzer):
        """Test recommendations for resistant strain."""
        analysis_results = {
            "drug_resistance": {
                "vancomycin": {"classifications": ["resistant"]}
            },
            "hypervirulence": {"classification": "non-hypervirulent"},
            "recurrence_risk": {"risk_level": "low"},
        }
        recommendations = analyzer.get_treatment_recommendations(analysis_results)

        assert len(recommendations["cautions"]) > 0


class TestIntegration:
    """Integration tests for C. difficile Analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return CDiffAnalyzer()

    def test_full_analysis_workflow(self, analyzer):
        """Test complete analysis workflow."""
        sequences = {
            CDiffGene.TCDA: ["MKISNVKKTYDIKEEEIIKNLNKVNNILKKIN" * 85],
            CDiffGene.TCDB: ["MSLENIFKDIPKVNDLLKNIKKNQKNLIKNI" * 76],
            CDiffGene.RPOB: ["MKLIIFLPDGSRFTY" * 40],
        }

        results = analyzer.analyze(sequences, ribotype=CDiffRibotype.RT027)

        assert results["n_sequences"] == 1
        assert "toxin_analysis" in results
        assert "hypervirulence" in results
        assert "recurrence_risk" in results

    def test_batch_analysis(self, analyzer):
        """Test analysis of multiple sequences."""
        n_seqs = 3
        sequences = {
            CDiffGene.TCDA: ["MKISNVKKTYDIKEEEIIKNLNKVNNILKKIN" * 85] * n_seqs,
        }

        results = analyzer.analyze(sequences)

        assert results["n_sequences"] == n_seqs

    def test_full_recommendation_workflow(self, analyzer):
        """Test full workflow with recommendations."""
        sequences = {
            CDiffGene.TCDA: ["MKISNVKKTYDIKEEEIIKNLNKVNNILKKIN" * 85],
            CDiffGene.RPOB: ["MKLIIFLPDGSRFTY" * 40],
        }

        results = analyzer.analyze(sequences)
        recommendations = analyzer.get_treatment_recommendations(results)

        assert "first_line" in recommendations
        assert "alternatives" in recommendations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
