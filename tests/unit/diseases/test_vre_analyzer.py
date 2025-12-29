# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for VRE Analyzer.

Tests cover:
- Van genotype classification
- Drug resistance prediction
- HLAR detection
- Virulence factor analysis
- Treatment option assessment
"""

from __future__ import annotations

import pytest

from src.diseases.vre_analyzer import (
    VREAnalyzer,
    VREConfig,
    VREDrug,
    VREGene,
    EnterococcusSpecies,
)


class TestEnterococcusSpecies:
    """Tests for EnterococcusSpecies enum."""

    def test_species_values(self):
        """Test species enum values."""
        assert EnterococcusSpecies.E_FAECIUM.value == "E. faecium"
        assert EnterococcusSpecies.E_FAECALIS.value == "E. faecalis"
        assert EnterococcusSpecies.E_GALLINARUM.value == "E. gallinarum"


class TestVREGene:
    """Tests for VREGene enum."""

    def test_van_genes(self):
        """Test van gene values."""
        assert VREGene.VANA.value == "vanA"
        assert VREGene.VANB.value == "vanB"
        assert VREGene.VANC.value == "vanC"

    def test_resistance_genes(self):
        """Test resistance gene values."""
        assert VREGene.LIAFSR.value == "liaFSR"
        assert VREGene.OPTRA.value == "optrA"


class TestVREDrug:
    """Tests for VREDrug enum."""

    def test_glycopeptide_drugs(self):
        """Test glycopeptide drug values."""
        assert VREDrug.VANCOMYCIN.value == "vancomycin"
        assert VREDrug.TEICOPLANIN.value == "teicoplanin"

    def test_alternative_drugs(self):
        """Test alternative drug values."""
        assert VREDrug.DAPTOMYCIN.value == "daptomycin"
        assert VREDrug.LINEZOLID.value == "linezolid"


class TestVREConfig:
    """Tests for VREConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = VREConfig()
        assert config.name == "vre"
        assert config.display_name == "Vancomycin-Resistant Enterococcus"
        assert "cdc_ar" in config.data_sources

    def test_disease_type(self):
        """Test disease type is bacterial."""
        config = VREConfig()
        from src.diseases.base import DiseaseType
        assert config.disease_type == DiseaseType.BACTERIAL

    def test_vre_specific_settings(self):
        """Test VRE-specific configuration options."""
        config = VREConfig()
        assert config.classify_species is True
        assert config.detect_virulence is True
        assert config.check_hlar is True


class TestVREAnalyzer:
    """Tests for VREAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return VREAnalyzer()

    @pytest.fixture
    def sample_vana_sequence(self):
        """Create sample VanA sequence."""
        return "MKKLGAVLGLVFFSATGAVAAPQTQNIQNEQINQAQDLYAHEGFLPK" * 8

    @pytest.fixture
    def sample_liasr_sequence(self):
        """Create sample LiaFSR sequence."""
        return "MWKDEEGQRPPEDVLQRLGQQGFADTSIQEIQRMLRWHDVTVDPAPG" * 5

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.config.name == "vre"
        assert len(analyzer.aa_alphabet) > 0

    def test_analyze_empty_sequences(self, analyzer):
        """Test analysis with empty sequences."""
        results = analyzer.analyze({})
        assert results["n_sequences"] == 0
        assert results["genes_analyzed"] == []

    def test_analyze_with_vana(self, analyzer, sample_vana_sequence):
        """Test analysis with VanA gene."""
        sequences = {VREGene.VANA: [sample_vana_sequence]}
        results = analyzer.analyze(sequences)

        assert results["n_sequences"] == 1
        assert results["van_genotype"]["genotype"] == "vanA"

    def test_analyze_with_species(self, analyzer, sample_vana_sequence):
        """Test analysis with specified species."""
        sequences = {VREGene.VANA: [sample_vana_sequence]}
        results = analyzer.analyze(sequences, species=EnterococcusSpecies.E_FAECIUM)

        assert results["species"] == "E. faecium"


class TestVanGenotype:
    """Tests for van genotype determination."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return VREAnalyzer()

    def test_vana_genotype(self, analyzer):
        """Test VanA genotype detection."""
        sequences = {VREGene.VANA: ["A" * 200]}
        results = analyzer._determine_van_genotype(sequences)

        assert results["genotype"] == "vanA"
        assert results["phenotype"] == "high_level_resistance"
        assert results["teicoplanin_status"] == "resistant"

    def test_vanb_genotype(self, analyzer):
        """Test VanB genotype detection."""
        sequences = {VREGene.VANB: ["B" * 200]}
        results = analyzer._determine_van_genotype(sequences)

        assert results["genotype"] == "vanB"
        assert results["teicoplanin_status"] == "susceptible"

    def test_vanc_genotype(self, analyzer):
        """Test VanC genotype detection."""
        sequences = {VREGene.VANC: ["C" * 200]}
        results = analyzer._determine_van_genotype(sequences)

        assert results["genotype"] == "vanC"
        assert results["phenotype"] == "low_level_resistance"

    def test_susceptible(self, analyzer):
        """Test susceptible (no van gene)."""
        sequences = {VREGene.ESP: ["E" * 100]}  # Not a van gene
        results = analyzer._determine_van_genotype(sequences)

        assert results["genotype"] == "susceptible"


class TestDrugResistance:
    """Tests for drug resistance prediction."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return VREAnalyzer()

    def test_predict_vancomycin_resistance(self, analyzer):
        """Test vancomycin resistance prediction."""
        vana_sequence = "A" * 200
        results = analyzer.predict_drug_resistance(
            [vana_sequence],
            VREDrug.VANCOMYCIN,
            VREGene.VANA,
        )

        assert "scores" in results
        assert "classifications" in results
        assert len(results["scores"]) == 1

    def test_predict_daptomycin_resistance(self, analyzer):
        """Test daptomycin resistance prediction."""
        liasr_sequence = "W" * 100
        results = analyzer.predict_drug_resistance(
            [liasr_sequence],
            VREDrug.DAPTOMYCIN,
            VREGene.LIAFSR,
        )

        assert "scores" in results
        assert len(results["scores"]) == 1

    def test_resistance_classification(self, analyzer):
        """Test resistance classification levels."""
        vana_sequence = "A" * 200
        results = analyzer.predict_drug_resistance(
            [vana_sequence],
            VREDrug.VANCOMYCIN,
            VREGene.VANA,
        )

        classification = results["classifications"][0]
        assert classification in ["susceptible", "reduced_susceptibility", "resistant"]


class TestHLAR:
    """Tests for high-level aminoglycoside resistance detection."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return VREAnalyzer()

    def test_hlar_detection(self, analyzer):
        """Test HLAR gene detection."""
        sequences = {VREGene.AAC6_APH2: ["A" * 500]}
        results = analyzer._detect_hlar(sequences)

        assert results["gentamicin_hlar"] is True
        assert "aac(6')-Ie-aph(2'')-Ia" in results["genes_detected"]
        assert results["synergy_expected"] is False

    def test_no_hlar(self, analyzer):
        """Test no HLAR genes."""
        sequences = {VREGene.VANA: ["A" * 200]}
        results = analyzer._detect_hlar(sequences)

        assert results["gentamicin_hlar"] is False
        assert results["synergy_expected"] is True


class TestVirulence:
    """Tests for virulence factor detection."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return VREAnalyzer()

    def test_virulence_detection(self, analyzer):
        """Test virulence factor detection."""
        sequences = {
            VREGene.ESP: ["E" * 300],
            VREGene.GEL: ["G" * 200],
        }
        results = analyzer._detect_virulence(sequences)

        assert len(results["factors_detected"]) >= 2
        assert results["virulence_score"] > 0

    def test_biofilm_potential(self, analyzer):
        """Test biofilm potential classification."""
        sequences = {
            VREGene.ESP: ["E" * 300],
            VREGene.GEL: ["G" * 200],
            VREGene.AS: ["A" * 200],
        }
        results = analyzer._detect_virulence(sequences)

        assert results["biofilm_potential"] in ["low", "moderate", "high"]


class TestTreatmentOptions:
    """Tests for treatment option assessment."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return VREAnalyzer()

    def test_susceptible_treatment(self, analyzer):
        """Test treatment for susceptible strain."""
        van_genotype = {"genotype": "susceptible"}
        resistance_data = {}
        options = analyzer._assess_treatment_options(van_genotype, resistance_data)

        assert "Vancomycin" in options["recommended"]

    def test_vana_treatment(self, analyzer):
        """Test treatment for VanA strain."""
        van_genotype = {"genotype": "vanA"}
        resistance_data = {}
        options = analyzer._assess_treatment_options(van_genotype, resistance_data)

        assert "Vancomycin" in options["contraindicated"]
        assert "Teicoplanin" in options["contraindicated"]

    def test_e_faecium_treatment(self, analyzer):
        """Test E. faecium-specific treatment options."""
        van_genotype = {"genotype": "vanA"}
        resistance_data = {}
        options = analyzer._assess_treatment_options(
            van_genotype, resistance_data, EnterococcusSpecies.E_FAECIUM
        )

        assert any("Quinupristin" in opt for opt in options["alternative"])


class TestIntegration:
    """Integration tests for VRE Analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return VREAnalyzer()

    def test_full_analysis_workflow(self, analyzer):
        """Test complete analysis workflow."""
        sequences = {
            VREGene.VANA: ["MKKLGAVLGLVFFSATGAVAAPQTQNIQNEQINQAQDLYAHEGFLPK" * 8],
            VREGene.VANH: ["MTIKVGITNPAAASSDFFTVSGKKVEAEGGFHK" * 6],
            VREGene.ESP: ["MKKTLLTVVLSALFGAVSVAQETTVKETVKPKVGETEKTVT" * 10],
        }

        results = analyzer.analyze(sequences, species=EnterococcusSpecies.E_FAECIUM)

        assert results["n_sequences"] == 1
        assert results["van_genotype"]["genotype"] == "vanA"
        assert "hlar_status" in results
        assert "virulence_factors" in results
        assert "treatment_options" in results

    def test_batch_analysis(self, analyzer):
        """Test analysis of multiple sequences."""
        n_seqs = 3
        sequences = {
            VREGene.VANA: ["MKKLGAVLGLVFFSATGAVAAPQTQNIQNEQINQAQDLYAHEGFLPK" * 8] * n_seqs,
        }

        results = analyzer.analyze(sequences)

        assert results["n_sequences"] == n_seqs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
