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
        assert RASubtype.ACPA_POSITIVE.value == "acpa_positive"
        assert RASubtype.ACPA_NEGATIVE.value == "acpa_negative"
        assert RASubtype.RF_POSITIVE.value == "rf_positive"
        assert RASubtype.SERONEGATIVE.value == "seronegative"


class TestCitrullinationSite:
    """Tests for CitrullinationSite dataclass."""

    def test_creation(self):
        """Test site creation."""
        site = CitrullinationSite(
            protein_name="vimentin",
            position=10,
            sequence_context="GFRGLARGKDRRFRG",
            padic_distance_to_self=0.333,
            immunogenicity_score=0.75,
            in_goldilocks_zone=True,
            known_acpa_target=True,
        )

        assert site.protein_name == "vimentin"
        assert site.position == 10
        assert site.sequence_context == "GFRGLARGKDRRFRG"
        assert site.immunogenicity_score == 0.75
        assert site.in_goldilocks_zone is True
        assert site.known_acpa_target is True


class TestCitrullinationPredictor:
    """Tests for CitrullinationPredictor."""

    def test_creation(self):
        """Test predictor creation."""
        predictor = CitrullinationPredictor(context_size=15, hidden_dim=64)
        assert predictor.context_size == 15

    def test_encode_sequence(self):
        """Test sequence encoding."""
        predictor = CitrullinationPredictor()

        sequence = "GFRGLARGKDRRFRG"  # 15 residues
        encoding = predictor.encode_sequence(sequence)

        assert encoding.shape[0] == 15

    def test_forward(self):
        """Test forward pass."""
        predictor = CitrullinationPredictor()

        # Context sequences should be 15-mers centered on arginine
        contexts = ["GFRGLARGKDRRFRG", "ARRAARRGAARRAAR"]
        result = predictor(contexts)

        assert "propensity" in result
        assert len(result["propensity"]) == 2


class TestPAdicCitrullinationShift:
    """Tests for PAdicCitrullinationShift."""

    def test_creation(self):
        """Test shift analyzer creation."""
        analyzer = PAdicCitrullinationShift(p=3, embedding_dim=64)
        assert analyzer.p == 3
        assert analyzer.embedding_dim == 64


class TestGoldilocksZoneDetector:
    """Tests for GoldilocksZoneDetector."""

    def test_creation(self):
        """Test detector creation."""
        detector = GoldilocksZoneDetector(
            p=3,
            zone_min=0.15,
            zone_max=0.30,
        )
        assert detector.p == 3
        assert detector.zone_min == 0.15
        assert detector.zone_max == 0.30

    def test_is_in_zone(self):
        """Test zone checking."""
        detector = GoldilocksZoneDetector(zone_min=0.1, zone_max=0.5)

        # Inside zone
        inside = torch.tensor([0.3])
        assert detector.is_in_zone(inside).item() is True

        # Below zone
        below = torch.tensor([0.05])
        assert detector.is_in_zone(below).item() is False

        # Above zone
        above = torch.tensor([0.7])
        assert detector.is_in_zone(above).item() is False

    def test_zone_risk_score(self):
        """Test risk score computation."""
        detector = GoldilocksZoneDetector()

        # Points in the zone should have higher risk
        inside = torch.tensor([0.2])
        outside = torch.tensor([0.8])

        risk_in = detector.zone_risk_score(inside)
        risk_out = detector.zone_risk_score(outside)

        # Inside zone should have higher risk
        assert risk_in > risk_out


class TestRheumatoidArthritisAnalyzer:
    """Tests for RheumatoidArthritisAnalyzer."""

    def test_creation(self):
        """Test analyzer creation."""
        analyzer = RheumatoidArthritisAnalyzer(p=3)
        assert analyzer.p == 3

    def test_find_arginine_positions(self):
        """Test finding arginine positions in sequence."""
        analyzer = RheumatoidArthritisAnalyzer()

        # Sequence with arginines at positions 3, 5, 7
        sequence = "AAARARAAR"

        positions = analyzer.find_arginine_positions(sequence)

        assert 3 in positions
        assert 5 in positions
        assert 7 in positions

    def test_find_arginine_positions_no_arginines(self):
        """Test finding arginines when none present."""
        analyzer = RheumatoidArthritisAnalyzer()

        sequence = "AAAAAAAAA"
        positions = analyzer.find_arginine_positions(sequence)

        assert len(positions) == 0

    def test_citrullinate_sequence(self):
        """Test sequence citrullination."""
        analyzer = RheumatoidArthritisAnalyzer()

        sequence = "AAARAGAA"  # R at position 3
        citrullinated = analyzer.citrullinate_sequence(sequence, position=3)

        # Citrulline represented as X
        assert "X" in citrullinated or "Cit" in citrullinated.upper()


class TestRARiskProfile:
    """Tests for RARiskProfile dataclass."""

    def test_creation(self):
        """Test risk profile creation."""
        profile = RARiskProfile(
            hla_alleles=["HLA-DRB1*04:01"],
            shared_epitope_positive=True,
            padi4_haplotype="susceptible",
            smoking_history=True,
            ebv_positive=False,
            genetic_risk_score=0.7,
            environmental_risk_score=0.4,
            overall_risk=0.6,
            risk_category="high",
        )

        assert profile.genetic_risk_score == 0.7
        assert profile.shared_epitope_positive is True
        assert profile.overall_risk == 0.6
        assert profile.risk_category == "high"


class TestEpitopeAnalysis:
    """Tests for EpitopeAnalysis dataclass."""

    def test_creation(self):
        """Test epitope analysis creation."""
        native_emb = torch.randn(32)
        cit_emb = torch.randn(32)

        analysis = EpitopeAnalysis(
            sequence="ARGPGRAFKDR",
            citrullination_positions=[0, 5, 10],
            native_padic_embedding=native_emb,
            citrullinated_padic_embedding=cit_emb,
            padic_shift=0.25,
            hla_binding_affinity=0.7,
            tcr_cross_reactivity=0.3,
            predicted_pathogenicity=0.5,
        )

        assert analysis.sequence == "ARGPGRAFKDR"
        assert len(analysis.citrullination_positions) == 3
        assert analysis.padic_shift == 0.25
