# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for Zika Virus Analyzer.

Tests cover:
- Lineage classification
- Drug resistance prediction
- Congenital Zika Syndrome risk
- Guillain-Barré Syndrome markers
- Neurovirulence prediction
"""

from __future__ import annotations

import pytest

from src.diseases.zika_analyzer import (
    ZikaAnalyzer,
    ZikaConfig,
    ZikaDrug,
    ZikaGene,
    ZikaLineage,
)


class TestZikaLineage:
    """Tests for ZikaLineage enum."""

    def test_lineage_values(self):
        """Test lineage enum values."""
        assert ZikaLineage.AFRICAN.value == "African"
        assert ZikaLineage.ASIAN.value == "Asian"
        assert ZikaLineage.AMERICAN.value == "American"


class TestZikaGene:
    """Tests for ZikaGene enum."""

    def test_structural_genes(self):
        """Test structural gene values."""
        assert ZikaGene.C.value == "C"
        assert ZikaGene.prM.value == "prM"
        assert ZikaGene.E.value == "E"

    def test_nonstructural_genes(self):
        """Test non-structural gene values."""
        assert ZikaGene.NS1.value == "NS1"
        assert ZikaGene.NS3.value == "NS3"
        assert ZikaGene.NS5.value == "NS5"


class TestZikaDrug:
    """Tests for ZikaDrug enum."""

    def test_protease_inhibitors(self):
        """Test NS3 protease inhibitor values."""
        assert ZikaDrug.TEMOPORFIN.value == "temoporfin"
        assert ZikaDrug.NICLOSAMIDE.value == "niclosamide"

    def test_polymerase_inhibitors(self):
        """Test NS5 polymerase inhibitor values."""
        assert ZikaDrug.SOFOSBUVIR.value == "sofosbuvir"
        assert ZikaDrug.FAVIPIRAVIR.value == "favipiravir"
        assert ZikaDrug.RIBAVIRIN.value == "ribavirin"


class TestZikaConfig:
    """Tests for ZikaConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = ZikaConfig()
        assert config.name == "zika"
        assert config.display_name == "Zika Fever"
        assert "genbank" in config.data_sources

    def test_disease_type(self):
        """Test disease type is viral."""
        config = ZikaConfig()
        from src.diseases.base import DiseaseType
        assert config.disease_type == DiseaseType.VIRAL

    def test_zika_specific_settings(self):
        """Test Zika-specific configuration options."""
        config = ZikaConfig()
        assert config.predict_czs_risk is True
        assert config.predict_gbs_risk is True
        assert config.predict_neurovirulence is True
        assert config.classify_lineage is True


class TestZikaAnalyzer:
    """Tests for ZikaAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return ZikaAnalyzer()

    @pytest.fixture
    def sample_e_sequence(self):
        """Create sample E protein sequence."""
        # ~500 amino acids for envelope protein
        return "IRCIGVSNRDFVEGMSGGTWVDVVLEHGGCVTVMAQDKPTVDIELVTTTVSNMAEVRSYC" * 8

    @pytest.fixture
    def sample_ns3_sequence(self):
        """Create sample NS3 sequence."""
        # ~620 amino acids for NS3
        return "MGKRSAGSIMWLASLAVVIACAGAMKLSNFQGKVMMTVNATDVTDVITIPTAAGKNLCIV" * 10

    @pytest.fixture
    def sample_ns5_sequence(self):
        """Create sample NS5 sequence."""
        # ~900 amino acids for NS5
        return "GTGNIGETLGEKWKSRLNALGKSEFQIYKKSGIQEVDRTLAKEGIKRGETDHHAVSRGSA" * 15

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.config.name == "zika"
        assert len(analyzer.aa_alphabet) > 0

    def test_analyze_empty_sequences(self, analyzer):
        """Test analysis with empty sequences."""
        results = analyzer.analyze({})
        assert results["n_sequences"] == 0
        assert results["genes_analyzed"] == []

    def test_analyze_single_gene(self, analyzer, sample_ns3_sequence):
        """Test analysis with single gene."""
        sequences = {ZikaGene.NS3: [sample_ns3_sequence]}
        results = analyzer.analyze(sequences)

        assert results["n_sequences"] == 1
        assert ZikaGene.NS3.value in results["genes_analyzed"]
        assert "drug_resistance" in results

    def test_analyze_multiple_genes(
        self, analyzer, sample_e_sequence, sample_ns3_sequence, sample_ns5_sequence
    ):
        """Test analysis with multiple genes."""
        sequences = {
            ZikaGene.E: [sample_e_sequence],
            ZikaGene.NS3: [sample_ns3_sequence],
            ZikaGene.NS5: [sample_ns5_sequence],
        }
        results = analyzer.analyze(sequences, lineage=ZikaLineage.ASIAN)

        assert results["n_sequences"] == 1
        assert len(results["genes_analyzed"]) == 3
        assert results["lineage"] == "Asian"

    def test_analyze_with_pregnancy_context(self, analyzer, sample_e_sequence):
        """Test analysis with pregnancy context."""
        sequences = {ZikaGene.E: [sample_e_sequence]}
        results = analyzer.analyze(sequences, pregnancy_context=True)

        assert results["pregnancy_context"] is True


class TestDrugResistance:
    """Tests for drug resistance prediction."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return ZikaAnalyzer()

    @pytest.fixture
    def ns3_sequence(self):
        """Create NS3 sequence with potential resistance mutations."""
        seq = list("A" * 350)  # Base sequence
        return "".join(seq)

    @pytest.fixture
    def ns5_sequence(self):
        """Create NS5 sequence."""
        return "G" * 600  # Base sequence for NS5

    def test_predict_ns3_resistance(self, analyzer, ns3_sequence):
        """Test NS3 protease inhibitor resistance prediction."""
        results = analyzer.predict_drug_resistance(
            [ns3_sequence],
            ZikaDrug.TEMOPORFIN,
            ZikaGene.NS3,
        )

        assert "scores" in results
        assert "classifications" in results
        assert "mutations" in results
        assert len(results["scores"]) == 1

    def test_predict_ns5_resistance(self, analyzer, ns5_sequence):
        """Test NS5 polymerase inhibitor resistance prediction."""
        results = analyzer.predict_drug_resistance(
            [ns5_sequence],
            ZikaDrug.SOFOSBUVIR,
            ZikaGene.NS5,
        )

        assert "scores" in results
        assert len(results["scores"]) == 1

    def test_resistance_classification(self, analyzer, ns3_sequence):
        """Test resistance classification levels."""
        results = analyzer.predict_drug_resistance(
            [ns3_sequence],
            ZikaDrug.TEMOPORFIN,
            ZikaGene.NS3,
        )

        classification = results["classifications"][0]
        assert classification in ["susceptible", "reduced_susceptibility", "resistant"]

    def test_multiple_sequences(self, analyzer):
        """Test prediction on multiple sequences."""
        sequences = ["A" * 350, "G" * 350, "V" * 350]
        results = analyzer.predict_drug_resistance(
            sequences,
            ZikaDrug.TEMOPORFIN,
            ZikaGene.NS3,
        )

        assert len(results["scores"]) == 3
        assert len(results["classifications"]) == 3


class TestLineageClassification:
    """Tests for lineage classification."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return ZikaAnalyzer()

    def test_classify_lineage(self, analyzer):
        """Test lineage classification from E sequence."""
        e_sequence = "V" * 500  # Generic sequence
        results = analyzer._classify_lineage([e_sequence])

        assert "predictions" in results
        assert "confidences" in results
        assert len(results["predictions"]) == 1

    def test_classify_multiple_sequences(self, analyzer):
        """Test classification of multiple sequences."""
        sequences = ["V" * 500, "I" * 500, "A" * 500]
        results = analyzer._classify_lineage(sequences)

        assert len(results["predictions"]) == 3
        assert "predicted_lineage" in results


class TestNeurovirulence:
    """Tests for neurovirulence analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return ZikaAnalyzer()

    @pytest.fixture
    def e_sequence(self):
        """Create E protein sequence."""
        return "IRCIGVSNRDFVEGMSGGTWVDVVLEHGGCVTVMAQDKPTVDIELVTTT" * 10

    def test_neurovirulence_analysis(self, analyzer, e_sequence):
        """Test neurovirulence analysis."""
        results = analyzer._analyze_neurovirulence([e_sequence], ZikaGene.E)

        assert "risk_scores" in results
        assert "risk_levels" in results
        assert "markers_detected" in results
        assert len(results["risk_scores"]) == 1

    def test_neurovirulence_risk_levels(self, analyzer, e_sequence):
        """Test neurovirulence risk levels."""
        results = analyzer._analyze_neurovirulence([e_sequence], ZikaGene.E)

        risk_level = results["risk_levels"][0]
        assert risk_level in ["low", "moderate", "high"]


class TestCZSRisk:
    """Tests for Congenital Zika Syndrome risk analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return ZikaAnalyzer()

    @pytest.fixture
    def e_sequence(self):
        """Create E protein sequence."""
        return "IRCIGVSNRDFVEGMSGGTWVDVVLEHGGCVTVMAQDKPTVDIELVTTT" * 10

    def test_czs_risk_analysis(self, analyzer, e_sequence):
        """Test CZS risk analysis."""
        results = analyzer._analyze_czs_risk([e_sequence], ZikaGene.E)

        assert "risk_scores" in results
        assert "risk_levels" in results
        assert "markers_detected" in results
        assert len(results["risk_scores"]) == 1

    def test_czs_risk_levels(self, analyzer, e_sequence):
        """Test CZS risk levels."""
        results = analyzer._analyze_czs_risk([e_sequence], ZikaGene.E)

        risk_level = results["risk_levels"][0]
        assert risk_level in ["low", "moderate", "high"]

    def test_overall_czs_risk(self, analyzer):
        """Test overall CZS risk calculation."""
        czs_results = {
            "E": {"risk_scores": [0.3, 0.4]},
            "NS4B": {"risk_scores": [0.5, 0.6]},
        }
        overall = analyzer._calculate_overall_czs_risk(czs_results)
        assert overall in ["low", "moderate", "high", "very_high", "unknown"]


class TestGBSRisk:
    """Tests for Guillain-Barré Syndrome risk analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return ZikaAnalyzer()

    @pytest.fixture
    def e_sequence(self):
        """Create E protein sequence."""
        return "IRCIGVSNRDFVEGMSGGTWVDVVLEHGGCVTVMAQDKPTVDIELVTTT" * 10

    def test_gbs_risk_analysis(self, analyzer, e_sequence):
        """Test GBS risk analysis."""
        results = analyzer._analyze_gbs_risk([e_sequence], ZikaGene.E)

        assert "risk_scores" in results
        assert "risk_levels" in results
        assert "markers_detected" in results
        assert len(results["risk_scores"]) == 1

    def test_gbs_risk_levels(self, analyzer, e_sequence):
        """Test GBS risk levels."""
        results = analyzer._analyze_gbs_risk([e_sequence], ZikaGene.E)

        risk_level = results["risk_levels"][0]
        assert risk_level in ["low", "moderate", "high"]


class TestPregnancyRecommendations:
    """Tests for pregnancy-specific recommendations."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return ZikaAnalyzer()

    def test_low_risk_recommendations(self, analyzer):
        """Test recommendations for low-risk analysis."""
        analysis_results = {
            "czs_risk": {"overall_risk": "low"},
            "neurovirulence": {},
            "drug_resistance": {},
        }
        recommendations = analyzer.get_pregnancy_recommendations(analysis_results)

        assert recommendations["monitoring_level"] == "standard"

    def test_high_risk_recommendations(self, analyzer):
        """Test recommendations for high-risk analysis."""
        analysis_results = {
            "czs_risk": {"overall_risk": "high"},
            "neurovirulence": {
                "E": {"risk_levels": ["high"]}
            },
            "drug_resistance": {
                "sofosbuvir": {"classifications": ["susceptible"]}
            },
        }
        recommendations = analyzer.get_pregnancy_recommendations(analysis_results)

        assert recommendations["monitoring_level"] == "intensive"
        assert len(recommendations["warnings"]) > 0
        assert len(recommendations["actions"]) > 0


class TestIntegration:
    """Integration tests for Zika Analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return ZikaAnalyzer()

    def test_full_analysis_workflow(self, analyzer):
        """Test complete analysis workflow."""
        # Create sample sequences for all genes
        sequences = {
            ZikaGene.E: ["IRCIGVSNRDFVEGMSGGTWVDVVLEHGGCVT" * 15],
            ZikaGene.NS3: ["MGKRSAGSIMWLASLAVVIACAGAMKLSNFQGK" * 19],
            ZikaGene.NS5: ["GTGNIGETLGEKWKSRLNALGKSEFQIYKKSG" * 28],
        }

        results = analyzer.analyze(sequences, lineage=ZikaLineage.ASIAN)

        assert results["n_sequences"] == 1
        assert len(results["genes_analyzed"]) == 3
        assert "drug_resistance" in results
        assert "lineage_classification" in results
        assert "neurovirulence" in results
        assert "czs_risk" in results

    def test_batch_analysis(self, analyzer):
        """Test analysis of multiple sequences."""
        n_seqs = 5
        sequences = {
            ZikaGene.NS3: ["MGKRSAGSIMWLASLAVVIACAGA" * 15] * n_seqs,
        }

        results = analyzer.analyze(sequences, lineage=ZikaLineage.AMERICAN)

        assert results["n_sequences"] == n_seqs

    def test_pregnancy_workflow(self, analyzer):
        """Test pregnancy-specific workflow."""
        sequences = {
            ZikaGene.E: ["IRCIGVSNRDFVEGMSGGTWVDVVLEHGGCVT" * 15],
        }

        # Analyze with pregnancy context
        results = analyzer.analyze(sequences, pregnancy_context=True)
        recommendations = analyzer.get_pregnancy_recommendations(results)

        assert "monitoring_level" in recommendations
        assert "actions" in recommendations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
