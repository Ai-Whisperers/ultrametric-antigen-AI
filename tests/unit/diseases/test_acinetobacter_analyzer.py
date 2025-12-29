# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for A. baumannii Analyzer.

Tests cover:
- Carbapenemase detection
- Drug resistance prediction
- Efflux pump analysis
- Colistin resistance detection
- Resistance profile classification
- Treatment option assessment
"""

from __future__ import annotations

import pytest

from src.diseases.acinetobacter_analyzer import (
    AcinetobacterAnalyzer,
    AcinetobacterConfig,
    ABDrug,
    ABGene,
    ABClonalComplex,
)


class TestABClonalComplex:
    """Tests for ABClonalComplex enum."""

    def test_clonal_complex_values(self):
        """Test clonal complex enum values."""
        assert ABClonalComplex.IC1.value == "IC1"
        assert ABClonalComplex.IC2.value == "IC2"
        assert ABClonalComplex.IC3.value == "IC3"


class TestABGene:
    """Tests for ABGene enum."""

    def test_carbapenemase_genes(self):
        """Test carbapenemase gene values."""
        assert ABGene.OXA23.value == "blaOXA-23"
        assert ABGene.OXA51.value == "blaOXA-51-like"
        assert ABGene.NDM.value == "blaNDM"

    def test_efflux_genes(self):
        """Test efflux pump gene values."""
        assert ABGene.ADEA.value == "adeA"
        assert ABGene.ADEB.value == "adeB"


class TestABDrug:
    """Tests for ABDrug enum."""

    def test_carbapenem_drugs(self):
        """Test carbapenem drug values."""
        assert ABDrug.MEROPENEM.value == "meropenem"
        assert ABDrug.IMIPENEM.value == "imipenem"

    def test_alternative_drugs(self):
        """Test alternative drug values."""
        assert ABDrug.COLISTIN.value == "colistin"
        assert ABDrug.TIGECYCLINE.value == "tigecycline"


class TestAcinetobacterConfig:
    """Tests for AcinetobacterConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AcinetobacterConfig()
        assert config.name == "acinetobacter"
        assert config.display_name == "Acinetobacter baumannii"
        assert "cdc_ar" in config.data_sources

    def test_disease_type(self):
        """Test disease type is bacterial."""
        config = AcinetobacterConfig()
        from src.diseases.base import DiseaseType
        assert config.disease_type == DiseaseType.BACTERIAL

    def test_ab_specific_settings(self):
        """Test A. baumannii-specific configuration options."""
        config = AcinetobacterConfig()
        assert config.predict_carbapenem_resistance is True
        assert config.detect_mbl is True
        assert config.detect_colistin_resistance is True


class TestAcinetobacterAnalyzer:
    """Tests for AcinetobacterAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return AcinetobacterAnalyzer()

    @pytest.fixture
    def sample_oxa23_sequence(self):
        """Create sample OXA-23 sequence."""
        return "MRVLILAALLSSSFAAQATTTSSKLDSKSLLNQVIKTNKASYKFSLKK" * 6

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.config.name == "acinetobacter"
        assert len(analyzer.aa_alphabet) > 0

    def test_analyze_empty_sequences(self, analyzer):
        """Test analysis with empty sequences."""
        results = analyzer.analyze({})
        assert results["n_sequences"] == 0
        assert results["genes_analyzed"] == []

    def test_analyze_with_oxa51(self, analyzer):
        """Test species confirmation with OXA-51."""
        sequences = {ABGene.OXA51: ["A" * 200]}
        results = analyzer.analyze(sequences)

        assert results["species_confirmed"] is True

    def test_analyze_with_clonal_complex(self, analyzer, sample_oxa23_sequence):
        """Test analysis with specified clonal complex."""
        sequences = {ABGene.OXA23: [sample_oxa23_sequence]}
        results = analyzer.analyze(sequences, clonal_complex=ABClonalComplex.IC2)

        assert results["clonal_complex"] == "IC2"


class TestCarbapenemaseDetection:
    """Tests for carbapenemase detection."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return AcinetobacterAnalyzer()

    def test_oxa_detection(self, analyzer):
        """Test OXA carbapenemase detection."""
        sequences = {ABGene.OXA23: ["A" * 300]}
        results = analyzer._detect_carbapenemases(sequences)

        assert "blaOXA-23" in results["detected"]
        assert "blaOXA-23" in results["oxa_type"]

    def test_mbl_detection(self, analyzer):
        """Test MBL detection."""
        sequences = {ABGene.NDM: ["N" * 300]}
        results = analyzer._detect_carbapenemases(sequences)

        assert "blaNDM" in results["detected"]
        assert "blaNDM" in results["mbl_type"]
        assert results["highest_resistance_level"] == "high_level"

    def test_no_carbapenemase(self, analyzer):
        """Test no carbapenemase detected."""
        sequences = {ABGene.OXA51: ["A" * 200]}  # Intrinsic, not acquired
        results = analyzer._detect_carbapenemases(sequences)

        assert len(results["detected"]) == 0
        assert results["highest_resistance_level"] == "susceptible"


class TestDrugResistance:
    """Tests for drug resistance prediction."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return AcinetobacterAnalyzer()

    def test_predict_colistin_resistance(self, analyzer):
        """Test colistin resistance prediction."""
        pmrab_sequence = "S" * 300
        results = analyzer.predict_drug_resistance(
            [pmrab_sequence],
            ABDrug.COLISTIN,
            ABGene.PMRAB,
        )

        assert "scores" in results
        assert "classifications" in results
        assert len(results["scores"]) == 1

    def test_arma_pan_resistance(self, analyzer):
        """Test armA pan-aminoglycoside resistance."""
        arma_sequence = "M" * 200
        results = analyzer.predict_drug_resistance(
            [arma_sequence],
            ABDrug.AMIKACIN,
            ABGene.ARMA,
        )

        assert results["classifications"][0] == "resistant"
        assert results["scores"][0] >= 0.9


class TestEffluxAnalysis:
    """Tests for efflux pump analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return AcinetobacterAnalyzer()

    def test_ade_abc_detection(self, analyzer):
        """Test AdeABC efflux pump detection."""
        sequences = {
            ABGene.ADEA: ["A" * 200],
            ABGene.ADEB: ["B" * 200],
            ABGene.ADEC: ["C" * 200],
        }
        results = analyzer._analyze_efflux(sequences)

        assert results["ade_abc_present"] is True

    def test_ade_ijk_detection(self, analyzer):
        """Test AdeIJK efflux pump detection."""
        sequences = {
            ABGene.ADEI: ["I" * 200],
            ABGene.ADEJ: ["J" * 200],
        }
        results = analyzer._analyze_efflux(sequences)

        assert results["ade_ijk_present"] is True

    def test_efflux_impact(self, analyzer):
        """Test efflux impact classification."""
        sequences = {
            ABGene.ADEA: ["A" * 200],
            ABGene.ADEB: ["B" * 200],
            ABGene.ADEI: ["I" * 200],
            ABGene.ADEJ: ["J" * 200],
        }
        results = analyzer._analyze_efflux(sequences)

        assert results["impact"] == "high"


class TestColistinResistance:
    """Tests for colistin resistance detection."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return AcinetobacterAnalyzer()

    def test_colistin_susceptible(self, analyzer):
        """Test colistin susceptible detection."""
        sequences = {ABGene.OXA23: ["A" * 200]}  # No colistin genes
        results = analyzer._detect_colistin_resistance(sequences)

        assert results["resistant"] is False

    def test_lpx_loss(self, analyzer):
        """Test lpx gene loss detection."""
        sequences = {}  # No lpx genes
        results = analyzer._detect_colistin_resistance(sequences)

        # lpx absence noted
        assert any("lpx" in m for m in results["mechanism"])


class TestResistanceProfile:
    """Tests for resistance profile classification."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return AcinetobacterAnalyzer()

    def test_susceptible_profile(self, analyzer):
        """Test susceptible profile."""
        carbapenemases = {"detected": []}
        resistance_data = {}
        colistin_data = {"resistant": False}

        results = analyzer._classify_resistance_profile(
            carbapenemases, resistance_data, colistin_data
        )

        assert results["profile"] == "susceptible"

    def test_crab_profile(self, analyzer):
        """Test CRAB (Carbapenem-Resistant) profile."""
        carbapenemases = {"detected": ["blaOXA-23"]}
        resistance_data = {}
        colistin_data = {"resistant": False}

        results = analyzer._classify_resistance_profile(
            carbapenemases, resistance_data, colistin_data
        )

        assert results["carbapenem_status"] == "resistant"

    def test_pdr_profile(self, analyzer):
        """Test PDR (Pan-Drug-Resistant) profile."""
        carbapenemases = {"detected": ["blaNDM"]}
        resistance_data = {}
        colistin_data = {"resistant": True}

        results = analyzer._classify_resistance_profile(
            carbapenemases, resistance_data, colistin_data
        )

        assert results["profile"] == "PDR"
        assert results["pdr"] is True


class TestTreatmentOptions:
    """Tests for treatment option assessment."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return AcinetobacterAnalyzer()

    def test_susceptible_treatment(self, analyzer):
        """Test treatment for susceptible strain."""
        analysis_results = {
            "resistance_profile": {
                "carbapenem_status": "susceptible",
                "colistin_status": "susceptible",
            }
        }
        options = analyzer._assess_treatment_options(analysis_results)

        assert any("Meropenem" in opt for opt in options["recommended"])

    def test_crab_treatment(self, analyzer):
        """Test treatment for CRAB strain."""
        analysis_results = {
            "resistance_profile": {
                "carbapenem_status": "resistant",
                "colistin_status": "susceptible",
            }
        }
        options = analyzer._assess_treatment_options(analysis_results)

        assert any("Colistin" in opt for opt in options["recommended"])
        assert "Carbapenems (monotherapy)" in options["contraindicated"]

    def test_pdr_treatment(self, analyzer):
        """Test treatment for PDR strain."""
        analysis_results = {
            "resistance_profile": {
                "carbapenem_status": "resistant",
                "colistin_status": "resistant",
            }
        }
        options = analyzer._assess_treatment_options(analysis_results)

        assert any("Cefiderocol" in opt for opt in options["alternative"])
        assert len(options["notes"]) > 0


class TestIntegration:
    """Integration tests for A. baumannii Analyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer fixture."""
        return AcinetobacterAnalyzer()

    def test_full_analysis_workflow(self, analyzer):
        """Test complete analysis workflow."""
        sequences = {
            ABGene.OXA51: ["MRVLILAALLSSSFAAQATTTSSKLDSKSLLNQVIKTNK" * 7],
            ABGene.OXA23: ["MRVLILAALLSSSFAAQATTTSSKLDSKSLLNQVIKTNK" * 7],
            ABGene.ADEA: ["MKLSAFVLFTALLFSSVSAQADNQTRSNQTNTTNTGNNN" * 5],
            ABGene.ADEB: ["MKKISLILASALLFSAPAFAAQSETSTSTVTAPVTSEPA" * 7],
        }

        results = analyzer.analyze(sequences, clonal_complex=ABClonalComplex.IC2)

        assert results["n_sequences"] == 1
        assert results["species_confirmed"] is True
        assert "carbapenemases" in results
        assert "blaOXA-23" in results["carbapenemases"]["detected"]
        assert "efflux_status" in results
        assert "resistance_profile" in results
        assert "treatment_options" in results

    def test_batch_analysis(self, analyzer):
        """Test analysis of multiple sequences."""
        n_seqs = 3
        sequences = {
            ABGene.OXA23: ["MRVLILAALLSSSFAAQATTTSSKLDSKSLLNQVIKTNK" * 7] * n_seqs,
        }

        results = analyzer.analyze(sequences)

        assert results["n_sequences"] == n_seqs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
