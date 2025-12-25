# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for Multiple Sclerosis Analyzer module."""

import pytest
import torch

from src.diseases.multiple_sclerosis import (
    HLABindingPredictor,
    MolecularMimicryDetector,
    MSRiskProfile,
    MSSubtype,
    MultipleSclerosisAnalyzer,
    MyelinTarget,
    MOLECULAR_MIMICRY_PAIRS,
    MS_RISK_HLA_ALLELES,
)


class TestMSSubtype:
    """Tests for MSSubtype enum."""

    def test_subtypes(self):
        """Test MS subtypes."""
        assert MSSubtype.RRMS.value == "relapsing_remitting"
        assert MSSubtype.SPMS.value == "secondary_progressive"
        assert MSSubtype.PPMS.value == "primary_progressive"
        assert MSSubtype.CIS.value == "clinically_isolated_syndrome"


class TestMyelinTarget:
    """Tests for MyelinTarget enum."""

    def test_targets(self):
        """Test myelin targets."""
        assert MyelinTarget.MBP.value == "myelin_basic_protein"
        assert MyelinTarget.MOG.value == "myelin_oligodendrocyte_glycoprotein"
        assert MyelinTarget.GLIALCAM.value == "glial_cell_adhesion_molecule"


class TestMolecularMimicryData:
    """Tests for molecular mimicry data."""

    def test_ebna1_glialcam_pair(self):
        """Test EBNA1-GlialCAM mimicry pair."""
        pair = MOLECULAR_MIMICRY_PAIRS["EBNA1_386_405"]
        assert pair["self_target"] == MyelinTarget.GLIALCAM
        assert pair["hla_restriction"] == "DRB1*15:01"
        assert pair["evidence_level"] == "high"

    def test_all_pairs_have_sequences(self):
        """Test that all pairs have required fields."""
        for name, pair in MOLECULAR_MIMICRY_PAIRS.items():
            assert "viral_sequence" in pair
            assert "self_sequence" in pair
            assert "self_target" in pair
            assert len(pair["viral_sequence"]) > 0
            assert len(pair["self_sequence"]) > 0


class TestHLARiskData:
    """Tests for HLA risk data."""

    def test_drb1_15_01_high_risk(self):
        """Test DRB1*15:01 is high risk."""
        assert MS_RISK_HLA_ALLELES["DRB1*15:01"] > 2.0

    def test_protective_alleles(self):
        """Test protective alleles have OR < 1."""
        assert MS_RISK_HLA_ALLELES["DRB1*14:01"] < 1.0
        assert MS_RISK_HLA_ALLELES["A*02:01"] < 1.0


class TestMolecularMimicryDetector:
    """Tests for MolecularMimicryDetector."""

    def test_creation(self):
        """Test detector creation."""
        detector = MolecularMimicryDetector()
        assert detector.similarity_threshold == 0.7

    def test_sequence_to_indices(self):
        """Test sequence to index conversion."""
        detector = MolecularMimicryDetector()
        indices = detector.sequence_to_indices("ACDEFG")
        assert indices.shape == (6,)
        assert indices[0] == 0  # A

    def test_padic_distance(self):
        """Test p-adic distance computation."""
        detector = MolecularMimicryDetector()

        # Identical sequences should have distance 0
        dist = detector.compute_padic_distance("ACDEF", "ACDEF")
        assert dist == 0.0

        # Different sequences should have positive distance
        dist = detector.compute_padic_distance("ACDEF", "XXXXX")
        assert dist > 0.0

    def test_structural_similarity(self):
        """Test structural similarity computation."""
        detector = MolecularMimicryDetector()

        # Identical sequences
        sim = detector.compute_structural_similarity("ACDEF", "ACDEF")
        assert sim > 0.9

        # Similar amino acids
        sim = detector.compute_structural_similarity("AAAA", "AVAV")  # Similar hydrophobicity
        assert sim > 0.5

    def test_forward_mimicry_detection(self):
        """Test mimicry detection forward pass."""
        detector = MolecularMimicryDetector()

        result = detector("PRHRDTLMLFSS", "NRHSRNMHQALS")

        assert "mimicry_score" in result
        assert "is_mimicry" in result
        assert "padic_distance" in result
        assert "structural_similarity" in result
        assert 0 <= result["mimicry_score"] <= 1

    def test_known_mimicry_pair(self):
        """Test known mimicry pair detection."""
        detector = MolecularMimicryDetector(similarity_threshold=0.3)

        # EBNA1/GlialCAM pair
        viral = MOLECULAR_MIMICRY_PAIRS["EBNA1_386_405"]["viral_sequence"]
        self_seq = MOLECULAR_MIMICRY_PAIRS["EBNA1_386_405"]["self_sequence"]

        result = detector(viral, self_seq)

        # Should have some similarity
        assert result["mimicry_score"] > 0.2


class TestHLABindingPredictor:
    """Tests for HLABindingPredictor."""

    def test_creation(self):
        """Test predictor creation."""
        predictor = HLABindingPredictor()
        assert predictor.embedding_dim == 64

    def test_encode_peptide(self):
        """Test peptide encoding."""
        predictor = HLABindingPredictor()
        emb = predictor.encode_peptide("ACDEFGHIKLMNP")
        assert emb.shape == (64,)

    def test_forward(self):
        """Test binding prediction."""
        predictor = HLABindingPredictor()

        result = predictor("ACDEFGHIKLMNP", hla_idx=0)

        assert "binding_score" in result
        assert "ic50_nm" in result
        assert "is_strong_binder" in result
        assert 0 <= result["binding_score"] <= 1


class TestMultipleSclerosisAnalyzer:
    """Tests for MultipleSclerosisAnalyzer."""

    def test_creation(self):
        """Test analyzer creation."""
        analyzer = MultipleSclerosisAnalyzer()
        assert analyzer.p == 3

    def test_padic_valuation(self):
        """Test p-adic valuation computation."""
        analyzer = MultipleSclerosisAnalyzer(p=3)

        assert analyzer.compute_padic_valuation(0) == float("inf")
        assert analyzer.compute_padic_valuation(1) == 0
        assert analyzer.compute_padic_valuation(3) == 1
        assert analyzer.compute_padic_valuation(9) == 2
        assert analyzer.compute_padic_valuation(27) == 3

    def test_analyze_known_mimicry(self):
        """Test analysis of known mimicry pairs."""
        analyzer = MultipleSclerosisAnalyzer()
        pairs = analyzer.analyze_known_mimicry()

        assert len(pairs) == len(MOLECULAR_MIMICRY_PAIRS)
        for pair in pairs:
            assert hasattr(pair, "similarity_score")
            assert hasattr(pair, "padic_distance")

    def test_compute_genetic_risk(self):
        """Test genetic risk computation."""
        analyzer = MultipleSclerosisAnalyzer()

        # High risk alleles
        risk, factors, protective = analyzer.compute_genetic_risk(["DRB1*15:01", "DRB1*03:01"])
        assert risk > 1.0
        assert len(factors) > 0

        # Protective alleles
        risk, factors, protective = analyzer.compute_genetic_risk(["DRB1*14:01", "A*02:01"])
        assert risk < 1.0
        assert len(protective) > 0

    def test_ebv_mimicry_risk(self):
        """Test EBV mimicry risk assessment."""
        analyzer = MultipleSclerosisAnalyzer()
        risk = analyzer.assess_ebv_mimicry_risk()

        assert 0 <= risk <= 1

    def test_compute_risk_profile(self):
        """Test comprehensive risk profile."""
        analyzer = MultipleSclerosisAnalyzer()

        profile = analyzer.compute_risk_profile(
            hla_alleles=["DRB1*15:01"],
            ebv_positive=True,
            vitamin_d_level=15,
            smoking=True,
            family_history=True,
        )

        assert isinstance(profile, MSRiskProfile)
        assert 0 <= profile.overall_risk <= 1
        assert len(profile.risk_factors) > 0

    def test_risk_profile_protective(self):
        """Test risk profile with protective factors."""
        analyzer = MultipleSclerosisAnalyzer()

        profile = analyzer.compute_risk_profile(
            hla_alleles=["DRB1*14:01", "A*02:01"],
            ebv_positive=False,
            vitamin_d_level=50,
            smoking=False,
            family_history=False,
        )

        assert len(profile.protective_factors) > 0
        assert profile.overall_risk < 0.5

    def test_predict_demyelination_rrms(self):
        """Test demyelination prediction for RRMS."""
        analyzer = MultipleSclerosisAnalyzer()

        prediction = analyzer.predict_demyelination_pattern(
            lesion_locations=["periventricular", "optic_nerve"],
            symptom_onset_age=30,
            subtype=MSSubtype.RRMS,
        )

        assert len(prediction.affected_regions) > 0
        assert prediction.progression_rate > 0
        assert prediction.lesion_pattern in ["typical_ms", "spinal_predominant", "atypical"]

    def test_predict_demyelination_ppms(self):
        """Test demyelination prediction for PPMS."""
        analyzer = MultipleSclerosisAnalyzer()

        prediction = analyzer.predict_demyelination_pattern(
            lesion_locations=["spinal_cord"],
            symptom_onset_age=55,
            subtype=MSSubtype.PPMS,
        )

        # PPMS should have higher progression rate
        assert prediction.progression_rate > 0.5
        # Spinal cord should be affected
        assert "spinal_cord" in prediction.affected_regions

    def test_scan_for_novel_mimicry(self):
        """Test scanning for novel mimicry."""
        analyzer = MultipleSclerosisAnalyzer()

        viral_proteome = {
            "test_protein": "ACDEFGHIKLMNPQRSTVWY" * 5,
        }

        candidates = analyzer.scan_for_novel_mimicry(
            viral_proteome=viral_proteome,
            threshold=0.3,  # Lower threshold for testing
        )

        # Should find some candidates (or empty list)
        assert isinstance(candidates, list)
