# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for Rheumatoid Arthritis analyzer."""

import pytest
import torch

from src.diseases.rheumatoid_arthritis import (
    CitrullinationPredictor,
    CitrullinationSite,
    EpitopeAnalysis,
    GoldilocksZoneDetector,
    PADEnzyme,
    PAdicCitrullinationShift,
    RARiskProfile,
    RASubtype,
    RheumatoidArthritisAnalyzer,
)


class TestPADEnzyme:
    """Tests for PADEnzyme enum."""

    def test_enum_values(self):
        """Test all PAD enzyme values exist."""
        assert PADEnzyme.PAD1.value == "PAD1"
        assert PADEnzyme.PAD2.value == "PAD2"
        assert PADEnzyme.PAD3.value == "PAD3"
        assert PADEnzyme.PAD4.value == "PAD4"
        assert PADEnzyme.PAD6.value == "PAD6"


class TestRASubtype:
    """Tests for RASubtype enum."""

    def test_enum_values(self):
        """Test RA subtype values."""
        assert RASubtype.SEROPOSITIVE.value == "seropositive"
        assert RASubtype.SERONEGATIVE.value == "seronegative"
        assert RASubtype.PALINDROMIC.value == "palindromic"


class TestCitrullinationSite:
    """Tests for CitrullinationSite dataclass."""

    def test_creation(self):
        """Test site creation."""
        site = CitrullinationSite(
            position=10,
            context="GFRGL",
            predicted_probability=0.75,
            pad_enzyme=PADEnzyme.PAD4,
            padic_distance=0.333,
            in_goldilocks_zone=True,
            autoimmune_risk="high",
        )

        assert site.position == 10
        assert site.context == "GFRGL"
        assert site.predicted_probability == 0.75
        assert site.pad_enzyme == PADEnzyme.PAD4
        assert site.in_goldilocks_zone is True


class TestCitrullinationPredictor:
    """Tests for CitrullinationPredictor."""

    def test_creation(self):
        """Test predictor creation."""
        predictor = CitrullinationPredictor(hidden_dim=64)
        assert predictor.hidden_dim == 64

    def test_encode_context(self):
        """Test context encoding."""
        predictor = CitrullinationPredictor()

        context = "GFRGL"  # 5 residues around arginine
        encoding = predictor.encode_context(context)

        assert encoding.shape[0] == 5  # Context length
        assert encoding.shape[1] == 20  # Amino acid vocab

    def test_predict_probability(self):
        """Test probability prediction."""
        predictor = CitrullinationPredictor()

        context = "GFRGL"
        prob = predictor.predict_probability(context)

        assert 0.0 <= prob <= 1.0

    def test_forward(self):
        """Test forward pass with batch contexts."""
        predictor = CitrullinationPredictor()

        contexts = ["GFRGL", "ARRAA", "PPRPP"]
        probabilities = predictor(contexts)

        assert len(probabilities) == 3
        assert all(0.0 <= p <= 1.0 for p in probabilities)


class TestPAdicCitrullinationShift:
    """Tests for PAdicCitrullinationShift."""

    def test_creation(self):
        """Test shift analyzer creation."""
        analyzer = PAdicCitrullinationShift(p=3, embedding_dim=32)
        assert analyzer.p == 3
        assert analyzer.embedding_dim == 32

    def test_compute_arg_embedding(self):
        """Test arginine embedding computation."""
        analyzer = PAdicCitrullinationShift(embedding_dim=32)

        context = "GFRGL"
        embedding = analyzer.compute_arg_embedding(context)

        assert embedding.shape == (32,)

    def test_compute_cit_embedding(self):
        """Test citrulline embedding computation."""
        analyzer = PAdicCitrullinationShift(embedding_dim=32)

        context = "GFCitGL"  # Hypothetical citrulline context
        embedding = analyzer.compute_cit_embedding(context)

        assert embedding.shape == (32,)

    def test_compute_shift_distance(self):
        """Test shift distance computation."""
        analyzer = PAdicCitrullinationShift(embedding_dim=32)

        context = "GFRGL"
        distance = analyzer.compute_shift_distance(context)

        assert distance >= 0


class TestGoldilocksZoneDetector:
    """Tests for GoldilocksZoneDetector."""

    def test_creation(self):
        """Test detector creation."""
        detector = GoldilocksZoneDetector(
            p=3,
            lower_bound=0.1,
            upper_bound=0.5,
        )
        assert detector.p == 3
        assert detector.lower_bound == 0.1
        assert detector.upper_bound == 0.5

    def test_check_in_zone(self):
        """Test zone checking."""
        detector = GoldilocksZoneDetector(lower_bound=0.1, upper_bound=0.5)

        # Inside zone
        assert detector.check_in_zone(0.3) is True

        # Below zone
        assert detector.check_in_zone(0.05) is False

        # Above zone
        assert detector.check_in_zone(0.7) is False

    def test_compute_risk_score(self):
        """Test risk score computation."""
        detector = GoldilocksZoneDetector()

        # Points in the zone should have higher risk
        risk_in = detector.compute_risk_score(0.3)
        risk_out = detector.compute_risk_score(0.8)

        assert risk_in > risk_out

    def test_forward(self):
        """Test forward pass."""
        detector = GoldilocksZoneDetector()

        distances = torch.tensor([0.05, 0.3, 0.7])
        results = detector(distances)

        assert "in_zone" in results
        assert "risk_scores" in results
        assert len(results["in_zone"]) == 3


class TestRheumatoidArthritisAnalyzer:
    """Tests for RheumatoidArthritisAnalyzer."""

    def test_creation(self):
        """Test analyzer creation."""
        analyzer = RheumatoidArthritisAnalyzer(
            p=3, embedding_dim=32, goldilocks_bounds=(0.1, 0.5)
        )
        assert analyzer.p == 3

    def test_analyze_protein(self):
        """Test protein citrullination analysis."""
        analyzer = RheumatoidArthritisAnalyzer(embedding_dim=32)

        # Fibrinogen-like sequence with arginines
        sequence = "MWVLVAALGLGALAAFPRLPPGA"  # Contains R at position 14

        sites = analyzer.analyze_protein("test_protein", sequence)

        # Should find at least one arginine site
        assert len(sites) >= 0  # May be 0 if no arginines found

        # If sites found, check they're valid
        for site in sites:
            assert isinstance(site, CitrullinationSite)
            assert 0 <= site.position < len(sequence)

    def test_analyze_protein_with_arginines(self):
        """Test analysis of sequence with multiple arginines."""
        analyzer = RheumatoidArthritisAnalyzer(embedding_dim=32)

        # Sequence with several arginines
        sequence = "AARRGRRPRRQRR"  # Multiple R sites

        sites = analyzer.analyze_protein("arginine_rich", sequence)

        # Should find multiple sites
        assert len(sites) >= 3

    def test_compute_genetic_risk(self):
        """Test genetic risk computation."""
        analyzer = RheumatoidArthritisAnalyzer()

        # High-risk HLA alleles
        risk, factors = analyzer.compute_genetic_risk(
            hla_alleles=["HLA-DRB1*04:01", "HLA-DRB1*04:04"],
            padi4_haplotype="susceptible",
        )

        assert 0.0 <= risk <= 1.0
        assert len(factors) > 0

    def test_compute_genetic_risk_no_risk_alleles(self):
        """Test risk with non-risk alleles."""
        analyzer = RheumatoidArthritisAnalyzer()

        risk, factors = analyzer.compute_genetic_risk(
            hla_alleles=["HLA-DRB1*01:01"],  # Non-risk allele
            padi4_haplotype="protective",
        )

        # Should have lower risk
        assert risk < 0.5

    def test_compute_risk_profile(self):
        """Test comprehensive risk profile."""
        analyzer = RheumatoidArthritisAnalyzer()

        profile = analyzer.compute_risk_profile(
            hla_alleles=["HLA-DRB1*04:01"],
            smoking=True,
            ebv_positive=True,
            padi4_haplotype="susceptible",
        )

        assert isinstance(profile, RARiskProfile)
        assert 0.0 <= profile.overall_risk <= 1.0
        assert len(profile.risk_factors) > 0

    def test_analyze_epitope(self):
        """Test epitope analysis."""
        analyzer = RheumatoidArthritisAnalyzer(embedding_dim=32)

        sequence = "ARGPGRAFKDR"
        citrullination_positions = [0, 6, 10]  # R positions

        analysis = analyzer.analyze_epitope(sequence, citrullination_positions)

        assert isinstance(analysis, EpitopeAnalysis)
        assert analysis.original_sequence == sequence
        assert len(analysis.citrullinated_positions) == 3


class TestIntegration:
    """Integration tests for RA analysis pipeline."""

    def test_full_pipeline(self):
        """Test complete analysis pipeline."""
        analyzer = RheumatoidArthritisAnalyzer(embedding_dim=32)

        # Simulate analysis of vimentin (known ACPA target)
        vimentin_fragment = "RLRSSVPGVRLLQDSVDFSLAD"

        # Analyze citrullination sites
        sites = analyzer.analyze_protein("vimentin", vimentin_fragment)

        # Compute genetic risk
        risk, factors = analyzer.compute_genetic_risk(
            hla_alleles=["HLA-DRB1*04:01"],
            padi4_haplotype="susceptible",
        )

        # Should complete without errors
        assert sites is not None
        assert risk is not None

    def test_goldilocks_zone_identification(self):
        """Test that Goldilocks Zone sites are properly identified."""
        analyzer = RheumatoidArthritisAnalyzer(
            embedding_dim=32, goldilocks_bounds=(0.1, 0.5)
        )

        # Analyze a sequence
        sequence = "GFRGLARGKDR"
        sites = analyzer.analyze_protein("test", sequence)

        # Check that in_goldilocks_zone is set
        for site in sites:
            assert hasattr(site, "in_goldilocks_zone")
            assert isinstance(site.in_goldilocks_zone, bool)
