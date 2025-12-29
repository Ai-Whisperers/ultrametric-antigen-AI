# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for Dengue Virus Analyzer.

Tests cover:
- Serotype classification
- Drug resistance prediction
- ADE risk analysis
- NS1 antigenicity
- Cross-protection matrix
"""

from __future__ import annotations

import pytest

from src.diseases.dengue_analyzer import (
    DengueAnalyzer,
    DengueConfig,
    DengueDrug,
    DengueGene,
    DengueSerotype,
)


class TestDengueSerotype:
    """Tests for DengueSerotype enum."""

    def test_serotype_values(self):
        """Test serotype enum values."""
        assert DengueSerotype.DENV1.value == "DENV-1"
        assert DengueSerotype.DENV2.value == "DENV-2"
        assert DengueSerotype.DENV3.value == "DENV-3"
        assert DengueSerotype.DENV4.value == "DENV-4"


class TestDengueGene:
    """Tests for DengueGene enum."""

    def test_structural_genes(self):
        """Test structural gene values."""
        assert DengueGene.C.value == "C"
        assert DengueGene.prM.value == "prM"
        assert DengueGene.E.value == "E"

    def test_nonstructural_genes(self):
        """Test non-structural gene values."""
        assert DengueGene.NS1.value == "NS1"
        assert DengueGene.NS3.value == "NS3"
        assert DengueGene.NS5.value == "NS5"


class TestDengueDrug:
    """Tests for DengueDrug enum."""

    def test_protease_inhibitors(self):
        """Test NS3 protease inhibitor values."""
        assert DengueDrug.ASUNAPREVIR.value == "asunaprevir"
        assert DengueDrug.BORTEZOMIB.value == "bortezomib"

    def test_polymerase_inhibitors(self):
        """Test NS5 polymerase inhibitor values."""
        assert DengueDrug.SOFOSBUVIR.value == "sofosbuvir"
        assert DengueDrug.BALAPIRAVIR.value == "balapiravir"
        assert DengueDrug.RIBAVIRIN.value == "ribavirin"


class TestDengueConfig:
    """Tests for DengueConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = DengueConfig()
        assert config.name == "dengue"
        assert config.display_name == "Dengue Fever"
        assert "vipr" in config.data_sources

    def test_disease_type(self):
        """Test disease type is viral."""
        config = DengueConfig()
        from src.diseases.base import DiseaseType
        assert config.disease_type == DiseaseType.VIRAL


class TestDengueAnalyzer:
    """Tests for DengueAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return DengueAnalyzer()

    @pytest.fixture
    def sample_e_sequence(self):
        """Create sample E protein sequence."""
        # ~400 amino acids for envelope protein
        return "MRCIGISNRDFVEGVSGGSWVDIVLEHGSCVTTMAKNKPTLDFELIKTEAKQPATLRKYC" * 7

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

    @pytest.fixture
    def sample_ns1_sequence(self):
        """Create sample NS1 sequence."""
        # ~352 amino acids for NS1
        return "DSGCVVSWKNKELKCGSGIFITDNVHTWTEQYKFQPESPARLASAILNAHKDGVCGIRS" * 6

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.config.name == "dengue"
        assert len(analyzer.aa_alphabet) > 0

    def test_analyze_empty_sequences(self, analyzer):
        """Test analysis with empty sequences."""
        results = analyzer.analyze({})
        assert results["n_sequences"] == 0
        assert results["genes_analyzed"] == []

    def test_analyze_single_gene(self, analyzer, sample_ns3_sequence):
        """Test analysis with single gene."""
        sequences = {DengueGene.NS3: [sample_ns3_sequence]}
        results = analyzer.analyze(sequences)

        assert results["n_sequences"] == 1
        assert DengueGene.NS3.value in results["genes_analyzed"]
        assert "drug_resistance" in results

    def test_analyze_multiple_genes(
        self, analyzer, sample_e_sequence, sample_ns3_sequence, sample_ns5_sequence
    ):
        """Test analysis with multiple genes."""
        sequences = {
            DengueGene.E: [sample_e_sequence],
            DengueGene.NS3: [sample_ns3_sequence],
            DengueGene.NS5: [sample_ns5_sequence],
        }
        results = analyzer.analyze(sequences, serotype=DengueSerotype.DENV2)

        assert results["n_sequences"] == 1
        assert len(results["genes_analyzed"]) == 3
        assert results["serotype"] == "DENV-2"

    def test_analyze_with_serotype(self, analyzer, sample_ns3_sequence):
        """Test analysis with specified serotype."""
        sequences = {DengueGene.NS3: [sample_ns3_sequence]}
        results = analyzer.analyze(sequences, serotype=DengueSerotype.DENV1)

        assert results["serotype"] == "DENV-1"


class TestDrugResistance:
    """Tests for drug resistance prediction."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return DengueAnalyzer()

    @pytest.fixture
    def ns3_sequence(self):
        """Create NS3 sequence with potential resistance mutations."""
        seq = list("A" * 350)  # Base sequence
        return "".join(seq)

    @pytest.fixture
    def ns5_sequence(self):
        """Create NS5 sequence."""
        return "G" * 450  # Base sequence for NS5

    def test_predict_ns3_resistance(self, analyzer, ns3_sequence):
        """Test NS3 protease inhibitor resistance prediction."""
        results = analyzer.predict_drug_resistance(
            [ns3_sequence],
            DengueDrug.ASUNAPREVIR,
            DengueGene.NS3,
        )

        assert "scores" in results
        assert "classifications" in results
        assert "mutations" in results
        assert len(results["scores"]) == 1

    def test_predict_ns5_resistance(self, analyzer, ns5_sequence):
        """Test NS5 polymerase inhibitor resistance prediction."""
        results = analyzer.predict_drug_resistance(
            [ns5_sequence],
            DengueDrug.SOFOSBUVIR,
            DengueGene.NS5,
        )

        assert "scores" in results
        assert len(results["scores"]) == 1

    def test_resistance_classification(self, analyzer, ns3_sequence):
        """Test resistance classification levels."""
        results = analyzer.predict_drug_resistance(
            [ns3_sequence],
            DengueDrug.ASUNAPREVIR,
            DengueGene.NS3,
        )

        classification = results["classifications"][0]
        assert classification in ["susceptible", "reduced_susceptibility", "resistant"]

    def test_multiple_sequences(self, analyzer):
        """Test prediction on multiple sequences."""
        sequences = ["A" * 350, "G" * 350, "V" * 350]
        results = analyzer.predict_drug_resistance(
            sequences,
            DengueDrug.ASUNAPREVIR,
            DengueGene.NS3,
        )

        assert len(results["scores"]) == 3
        assert len(results["classifications"]) == 3


class TestADERisk:
    """Tests for ADE risk analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return DengueAnalyzer()

    @pytest.fixture
    def e_sequence(self):
        """Create E protein sequence."""
        return "MRCIGISNRDFVEGVSGGSWVDIVLEHGSCVTTMAKNKPTLDFELIKTEAKQPATLRKYCIEAKL" * 6

    def test_ade_risk_analysis(self, analyzer, e_sequence):
        """Test ADE risk analysis."""
        results = analyzer._analyze_ade_risk([e_sequence], DengueSerotype.DENV2)

        assert "risk_scores" in results
        assert "risk_levels" in results
        assert "fusion_loop_conservation" in results
        assert len(results["risk_scores"]) == 1

    def test_ade_risk_levels(self, analyzer, e_sequence):
        """Test ADE risk levels."""
        results = analyzer._analyze_ade_risk([e_sequence])

        risk_level = results["risk_levels"][0]
        assert risk_level in ["low", "moderate", "high"]

    def test_cross_reactive_epitopes(self, analyzer, e_sequence):
        """Test cross-reactive epitope analysis."""
        results = analyzer._analyze_ade_risk([e_sequence])

        epitopes = results["cross_reactive_epitopes"][0]
        assert len(epitopes) > 0
        assert "epitope" in epitopes[0]
        assert "variation_score" in epitopes[0]


class TestSerotypeClassification:
    """Tests for serotype classification."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return DengueAnalyzer()

    def test_classify_serotype(self, analyzer):
        """Test serotype classification from E sequence."""
        e_sequence = "V" * 400  # Generic sequence
        results = analyzer._classify_serotype([e_sequence])

        assert "predictions" in results
        assert "confidences" in results
        assert len(results["predictions"]) == 1

    def test_classify_multiple_sequences(self, analyzer):
        """Test classification of multiple sequences."""
        sequences = ["V" * 400, "I" * 400, "A" * 400]
        results = analyzer._classify_serotype(sequences)

        assert len(results["predictions"]) == 3
        assert "predicted_serotype" in results


class TestNS1Antigenicity:
    """Tests for NS1 antigenicity analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return DengueAnalyzer()

    @pytest.fixture
    def ns1_sequence(self):
        """Create NS1 sequence."""
        return "DSGCVVSWKNKELKCGSGIFITDNVHTWTEQYKFQPESPARLASAILNAHKDGVCGIRS" * 6

    def test_ns1_antigenicity_analysis(self, analyzer, ns1_sequence):
        """Test NS1 antigenicity analysis."""
        results = analyzer._analyze_ns1_antigenicity([ns1_sequence])

        assert "antigenicity_scores" in results
        assert "diagnostic_impact" in results
        assert len(results["antigenicity_scores"]) == 1

    def test_diagnostic_impact_levels(self, analyzer, ns1_sequence):
        """Test diagnostic impact levels."""
        results = analyzer._analyze_ns1_antigenicity([ns1_sequence])

        impact = results["diagnostic_impact"][0]
        assert impact in ["minimal", "low", "moderate", "high"]

    def test_site_variations(self, analyzer, ns1_sequence):
        """Test antigenic site variation analysis."""
        results = analyzer._analyze_ns1_antigenicity([ns1_sequence])

        site_data = results["site_variations"][0]
        assert "site_A" in site_data or len(site_data) > 0


class TestCrossProtection:
    """Tests for cross-protection matrix."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return DengueAnalyzer()

    def test_cross_protection_matrix(self, analyzer):
        """Test cross-protection matrix retrieval."""
        matrix = analyzer.get_cross_protection_matrix(DengueSerotype.DENV1)

        assert "DENV-1" in matrix
        assert "DENV-2" in matrix
        assert matrix["DENV-1"] == 1.0  # Same serotype = full protection

    def test_cross_protection_values(self, analyzer):
        """Test cross-protection values are in valid range."""
        for serotype in DengueSerotype:
            matrix = analyzer.get_cross_protection_matrix(serotype)
            for value in matrix.values():
                assert 0 <= value <= 1


class TestSevereDengueRisk:
    """Tests for severe dengue risk prediction."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return DengueAnalyzer()

    @pytest.fixture
    def e_sequence(self):
        """Create E protein sequence."""
        return "MRCIGISNRDFVEGVSGGSWVDIVLEHGSCVTTMAKNKPTLDFELIK" * 9

    def test_primary_infection(self, analyzer, e_sequence):
        """Test risk for primary infection."""
        results = analyzer.predict_severe_dengue_risk(
            e_sequence,
            prior_serotypes=[],
            current_serotype=DengueSerotype.DENV2,
        )

        assert results["secondary_infection"] is False
        assert results["overall_risk"] == "low"

    def test_secondary_infection(self, analyzer, e_sequence):
        """Test risk for secondary infection."""
        results = analyzer.predict_severe_dengue_risk(
            e_sequence,
            prior_serotypes=[DengueSerotype.DENV1],
            current_serotype=DengueSerotype.DENV2,
        )

        assert results["secondary_infection"] is True
        assert "risk_score" in results
        assert results["overall_risk"] != "low" or results["risk_score"] < 0.3

    def test_multiple_prior_infections(self, analyzer, e_sequence):
        """Test risk with multiple prior infections."""
        results = analyzer.predict_severe_dengue_risk(
            e_sequence,
            prior_serotypes=[DengueSerotype.DENV1, DengueSerotype.DENV3],
            current_serotype=DengueSerotype.DENV2,
        )

        assert len(results["prior_serotypes"]) == 2

    def test_risk_levels(self, analyzer, e_sequence):
        """Test risk level classification."""
        results = analyzer.predict_severe_dengue_risk(
            e_sequence,
            prior_serotypes=[DengueSerotype.DENV1],
            current_serotype=DengueSerotype.DENV2,
        )

        assert results["overall_risk"] in ["low", "moderate", "high", "very_high"]


class TestIntegration:
    """Integration tests for Dengue Analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return DengueAnalyzer()

    def test_full_analysis_workflow(self, analyzer):
        """Test complete analysis workflow."""
        # Create sample sequences for all genes
        sequences = {
            DengueGene.E: ["MRCIGISNRDFVEGVSGGSWVDIVLEHGSCVTT" * 12],
            DengueGene.NS1: ["DSGCVVSWKNKELKCGSGIFITDNVHTWTEQYK" * 11],
            DengueGene.NS3: ["MGKRSAGSIMWLASLAVVIACAGAMKLSNFQGK" * 19],
            DengueGene.NS5: ["GTGNIGETLGEKWKSRLNALGKSEFQIYKKSG" * 28],
        }

        results = analyzer.analyze(sequences, serotype=DengueSerotype.DENV2)

        assert results["n_sequences"] == 1
        assert len(results["genes_analyzed"]) == 4
        assert "drug_resistance" in results
        assert "ade_risk" in results
        assert "ns1_antigenicity" in results

    def test_batch_analysis(self, analyzer):
        """Test analysis of multiple sequences."""
        n_seqs = 5
        sequences = {
            DengueGene.NS3: ["MGKRSAGSIMWLASLAVVIACAGA" * 15] * n_seqs,
        }

        results = analyzer.analyze(sequences, serotype=DengueSerotype.DENV1)

        assert results["n_sequences"] == n_seqs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
