# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for quantum biology analyzer."""

import pytest

from src.quantum.biology import (
    QUANTUM_ENZYMES,
    TUNNELING_RESIDUES,
    CatalyticSiteAnalysis,
    QuantumBiologyAnalyzer,
    QuantumEnzyme,
    QuantumMechanism,
)


class TestQuantumEnzymes:
    """Tests for quantum enzyme database."""

    def test_quantum_enzymes_exist(self):
        """Test that enzyme database is populated."""
        assert len(QUANTUM_ENZYMES) > 0

    def test_soybean_lipoxygenase(self):
        """Test soybean lipoxygenase data (highest known KIE)."""
        slo = QUANTUM_ENZYMES["soybean_lipoxygenase"]
        assert slo.name == "Soybean Lipoxygenase-1"
        assert slo.mechanism == QuantumMechanism.HYDROGEN_TUNNELING
        assert slo.isotope_effect > 50  # Very high KIE

    def test_fmo_complex(self):
        """Test FMO complex (quantum coherence)."""
        fmo = QUANTUM_ENZYMES["fmo_complex"]
        assert fmo.mechanism == QuantumMechanism.COHERENT_ENERGY_TRANSFER

    def test_all_enzymes_have_required_fields(self):
        """Test that all enzymes have required information."""
        for name, enzyme in QUANTUM_ENZYMES.items():
            assert enzyme.name, f"{name} missing name"
            assert isinstance(enzyme.mechanism, QuantumMechanism)
            assert len(enzyme.active_site_residues) > 0
            assert enzyme.tunneling_distance > 0


class TestTunnelingResidues:
    """Tests for tunneling residue scores."""

    def test_histidine_high_score(self):
        """Test that histidine has high tunneling score."""
        assert TUNNELING_RESIDUES["H"] >= 0.9

    def test_cysteine_high_score(self):
        """Test that cysteine has high tunneling score."""
        assert TUNNELING_RESIDUES["C"] >= 0.7

    def test_all_scores_valid_range(self):
        """Test that all scores are in valid range."""
        for aa, score in TUNNELING_RESIDUES.items():
            assert 0 <= score <= 1, f"Score for {aa} out of range"


class TestQuantumBiologyAnalyzer:
    """Tests for QuantumBiologyAnalyzer."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = QuantumBiologyAnalyzer()
        assert analyzer.p == 3

    def test_initialization_custom_p(self):
        """Test analyzer with custom prime."""
        analyzer = QuantumBiologyAnalyzer(p=5)
        assert analyzer.p == 5

    def test_analyze_catalytic_site_basic(self):
        """Test basic catalytic site analysis."""
        analyzer = QuantumBiologyAnalyzer()
        sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH"

        # Analyze positions 10, 20, 30
        result = analyzer.analyze_catalytic_site(sequence, [10, 20, 30])

        assert isinstance(result, CatalyticSiteAnalysis)
        assert len(result.site_sequence) == 3
        assert len(result.residue_positions) == 3

    def test_analyze_catalytic_site_metrics(self):
        """Test that analysis produces valid metrics."""
        analyzer = QuantumBiologyAnalyzer()
        sequence = "ACHDEFGHICKLMCH"  # Contains tunneling residues H, C, D, E

        result = analyzer.analyze_catalytic_site(sequence, [0, 2, 4, 5, 9])

        assert 0 <= result.padic_clustering_score <= 1
        assert 0 <= result.tunneling_propensity <= 1
        assert result.predicted_kie >= 1.0

    def test_tunneling_propensity_high_for_critical_residues(self):
        """Test that tunneling propensity is high for known tunneling residues."""
        analyzer = QuantumBiologyAnalyzer()

        # Sequence with many tunneling residues (H, C, D, E)
        high_tunnel_seq = "HCDEYHCDE"
        # Sequence with few tunneling residues
        low_tunnel_seq = "AVILLMAIV"

        high_result = analyzer.analyze_catalytic_site(high_tunnel_seq, list(range(9)))
        low_result = analyzer.analyze_catalytic_site(low_tunnel_seq, list(range(9)))

        assert high_result.tunneling_propensity > low_result.tunneling_propensity

    def test_mechanism_likelihood(self):
        """Test mechanism likelihood assessment."""
        analyzer = QuantumBiologyAnalyzer()

        # Sequence with proton tunneling residues
        sequence = "HDEKHDEK"
        result = analyzer.analyze_catalytic_site(sequence, list(range(8)))

        assert QuantumMechanism.PROTON_TUNNELING in result.mechanism_likelihood
        assert result.mechanism_likelihood[QuantumMechanism.PROTON_TUNNELING] > 0

    def test_critical_residues_identified(self):
        """Test that critical residues are identified."""
        analyzer = QuantumBiologyAnalyzer()

        sequence = "HCDEAAILV"
        result = analyzer.analyze_catalytic_site(sequence, list(range(9)))

        # H, C, D, E should be identified as critical
        assert "H" in result.critical_residues or "C" in result.critical_residues

    def test_predict_tunneling_probability(self):
        """Test tunneling probability prediction."""
        analyzer = QuantumBiologyAnalyzer()

        sequence = "HCDEYHCDE"
        site_analysis = analyzer.analyze_catalytic_site(sequence, list(range(9)))

        prob = analyzer.predict_tunneling_probability(site_analysis)

        assert 0 <= prob <= 1

    def test_tunneling_probability_temperature_effect(self):
        """Test that temperature affects tunneling probability."""
        analyzer = QuantumBiologyAnalyzer()

        sequence = "HCDEYHCDE"
        site_analysis = analyzer.analyze_catalytic_site(sequence, list(range(9)))

        prob_low_temp = analyzer.predict_tunneling_probability(site_analysis, temperature_kelvin=250)
        prob_high_temp = analyzer.predict_tunneling_probability(site_analysis, temperature_kelvin=350)

        # Higher temperature should increase probability (in this simplified model)
        assert prob_high_temp > prob_low_temp

    def test_compare_to_known_enzyme(self):
        """Test comparison to known enzyme."""
        analyzer = QuantumBiologyAnalyzer()

        # Create sequence similar to soybean lipoxygenase active site
        sequence = "HHNNI"  # H499, H504, H690, N694, I553
        site_analysis = analyzer.analyze_catalytic_site(sequence, list(range(5)))

        name, score = analyzer.compare_to_known_enzyme(site_analysis)

        assert name in QUANTUM_ENZYMES or name == "unknown"
        assert 0 <= score <= 1

    def test_empty_site_analysis(self):
        """Test analysis of empty site."""
        analyzer = QuantumBiologyAnalyzer()

        result = analyzer.analyze_catalytic_site("ACDEF", [])

        assert result.site_sequence == ""
        assert result.tunneling_propensity == 0.0

    def test_out_of_bounds_positions(self):
        """Test that out-of-bounds positions are handled."""
        analyzer = QuantumBiologyAnalyzer()

        result = analyzer.analyze_catalytic_site("ACDEF", [0, 2, 100, 200])

        # Should only include valid positions
        assert len(result.site_sequence) == 2


class TestPadicClustering:
    """Tests for p-adic clustering calculations."""

    def test_padic_valuation(self):
        """Test p-adic valuation calculation."""
        analyzer = QuantumBiologyAnalyzer(p=3)

        assert analyzer._compute_padic_valuation(9) == 2
        assert analyzer._compute_padic_valuation(27) == 3
        assert analyzer._compute_padic_valuation(0) >= 100

    def test_padic_distance(self):
        """Test p-adic distance calculation."""
        analyzer = QuantumBiologyAnalyzer(p=3)

        # Distance between numbers differing by 9 (3^2)
        dist = analyzer._compute_padic_distance(10, 19)
        assert dist == pytest.approx(1 / 9)  # 1/3^2

    def test_clustering_score_adjacent(self):
        """Test clustering score for adjacent positions."""
        analyzer = QuantumBiologyAnalyzer()

        # Adjacent positions should have high clustering
        adjacent_positions = [1, 2, 3, 4, 5]
        seq = "A" * 10
        result = analyzer.analyze_catalytic_site(seq, adjacent_positions)

        # Not necessarily high - depends on p-adic distances
        assert 0 <= result.padic_clustering_score <= 1

    def test_clustering_score_spread(self):
        """Test clustering score for spread positions."""
        analyzer = QuantumBiologyAnalyzer()

        # Spread positions
        spread_positions = [3, 9, 27, 81]  # Powers of 3
        seq = "A" * 100
        result = analyzer.analyze_catalytic_site(seq, spread_positions)

        assert 0 <= result.padic_clustering_score <= 1


class TestQuantumMechanism:
    """Tests for QuantumMechanism enum."""

    def test_mechanism_values(self):
        """Test mechanism enum values."""
        assert QuantumMechanism.PROTON_TUNNELING.value == "proton_tunneling"
        assert QuantumMechanism.ELECTRON_TRANSFER.value == "electron_transfer"

    def test_all_mechanisms_in_enzymes(self):
        """Test that all enzyme mechanisms are valid enum values."""
        for enzyme in QUANTUM_ENZYMES.values():
            assert isinstance(enzyme.mechanism, QuantumMechanism)
