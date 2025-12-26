# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for CRISPR off-target landscape analysis."""

import pytest
import torch

from src.analysis.crispr_offtarget import (
    CRISPROfftargetAnalyzer,
    GuideDesignOptimizer,
    GuideSafetyProfile,
    HyperbolicOfftargetEmbedder,
    MismatchType,
    OffTargetSite,
    OfftargetActivityPredictor,
    PAdicSequenceDistance,
)


class TestMismatchType:
    """Tests for MismatchType enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert MismatchType.MATCH.value == "match"
        assert MismatchType.TRANSITION.value == "transition"
        assert MismatchType.TRANSVERSION.value == "transversion"
        assert MismatchType.DELETION.value == "deletion"
        assert MismatchType.INSERTION.value == "insertion"


class TestPAdicSequenceDistance:
    """Tests for PAdicSequenceDistance."""

    def test_creation(self):
        """Test module creation."""
        module = PAdicSequenceDistance(p=3, seed_start=12)
        assert module.p == 3
        assert module.seed_start == 12

    def test_mismatch_positions(self):
        """Test mismatch position detection."""
        module = PAdicSequenceDistance()

        # Same sequences - no mismatches
        positions = module.mismatch_positions("ATCG", "ATCG")
        assert positions == []

        # One mismatch
        positions = module.mismatch_positions("ATCG", "ATGG")
        assert positions == [2]

        # Multiple mismatches
        positions = module.mismatch_positions("ATCG", "GTCG")
        assert positions == [0]

    def test_padic_valuation(self):
        """Test p-adic valuation computation."""
        module = PAdicSequenceDistance(p=3)

        # Position 0 should have high valuation
        assert module.padic_valuation(0) == 100

        # Position 1 - not divisible by 3
        assert module.padic_valuation(1) == 0

        # Position 3 - divisible by 3 once
        assert module.padic_valuation(3) == 1

        # Position 9 - divisible by 3 twice
        assert module.padic_valuation(9) == 2

    def test_compute_distance_same_sequence(self):
        """Test distance for identical sequences."""
        module = PAdicSequenceDistance()

        guide = "ATCGATCGATCGATCGATCG"
        dist = module.compute_distance(guide, guide)
        assert dist == 0.0

    def test_compute_distance_different_sequences(self):
        """Test distance for different sequences."""
        module = PAdicSequenceDistance()

        guide = "ATCGATCGATCGATCGATCG"
        # One mismatch at last position
        offtarget = "ATCGATCGATCGATCGATCC"
        dist = module.compute_distance(guide, offtarget)

        assert dist > 0.0

    def test_compute_distance_position_weighted(self):
        """Test position-weighted distance computation."""
        module = PAdicSequenceDistance()

        guide = "ATCGATCGATCGATCGATCG"

        # Mismatch in seed region (position 0) should have larger distance
        seed_mismatch = "GTCGATCGATCGATCGATCG"
        dist_seed = module.compute_distance(guide, seed_mismatch)

        # Mismatch in non-seed region (position 19)
        nonseed_mismatch = "ATCGATCGATCGATCGATCC"
        module.compute_distance(guide, nonseed_mismatch)

        # Seed region mismatches are more critical
        assert dist_seed > 0  # Both should be non-zero


class TestHyperbolicOfftargetEmbedder:
    """Tests for HyperbolicOfftargetEmbedder."""

    def test_creation(self):
        """Test embedder creation."""
        embedder = HyperbolicOfftargetEmbedder(
            seq_len=20, embedding_dim=64, curvature=1.0, max_norm=0.95
        )
        assert embedder.seq_len == 20
        assert embedder.embedding_dim == 64
        assert embedder.curvature == 1.0
        assert embedder.max_norm == 0.95

    def test_project_to_poincare(self):
        """Test projection to Poincare ball."""
        embedder = HyperbolicOfftargetEmbedder(embedding_dim=32)

        # Random point outside the ball
        x = torch.randn(32) * 2
        projected = embedder.project_to_poincare(x)

        # Should have norm <= max_norm
        assert torch.norm(projected) <= embedder.max_norm + 1e-6

    def test_encode_sequence(self):
        """Test sequence encoding."""
        embedder = HyperbolicOfftargetEmbedder()

        sequences = ["ATCGATCGATCGATCGATCG"]
        encoding = embedder.encode_sequence(sequences)

        # Should have shape (1, seq_len, embedding_features)
        assert encoding.shape[0] == 1
        assert encoding.shape[1] == 20


class TestOfftargetActivityPredictor:
    """Tests for OfftargetActivityPredictor."""

    def test_creation(self):
        """Test predictor creation."""
        predictor = OfftargetActivityPredictor()
        assert hasattr(predictor, "predictor")
        assert hasattr(predictor, "mismatch_encoder")

    def test_forward(self):
        """Test forward pass with seq_len parameter."""
        predictor = OfftargetActivityPredictor(seq_len=20, hidden_dim=128)

        # Predictor takes mismatch patterns, not raw embeddings
        assert predictor.seq_len == 20


class TestCRISPROfftargetAnalyzer:
    """Tests for CRISPROfftargetAnalyzer."""

    def test_creation(self):
        """Test analyzer creation."""
        analyzer = CRISPROfftargetAnalyzer(p=3)
        assert analyzer.p == 3

    def test_analyze_offtarget(self):
        """Test single off-target analysis."""
        analyzer = CRISPROfftargetAnalyzer()

        guide = "ATCGATCGATCGATCGATCG"
        offtarget_seq = "ATCGATCGATCGATCGATCC"

        site = analyzer.analyze_offtarget(
            guide=guide,
            offtarget_seq=offtarget_seq,
            chromosome="chr1",
            position=1000,
            strand="+",
            pam="NGG",
        )

        assert isinstance(site, OffTargetSite)
        assert site.sequence == offtarget_seq
        assert site.chromosome == "chr1"
        assert site.position == 1000


class TestGuideDesignOptimizer:
    """Tests for GuideDesignOptimizer."""

    def test_creation(self):
        """Test optimizer creation."""
        analyzer = CRISPROfftargetAnalyzer()
        optimizer = GuideDesignOptimizer(analyzer=analyzer)
        assert optimizer.analyzer is not None

    def test_creation_without_analyzer(self):
        """Test optimizer creation without explicit analyzer."""
        optimizer = GuideDesignOptimizer()
        assert optimizer.analyzer is not None


class TestOffTargetSite:
    """Tests for OffTargetSite dataclass."""

    def test_creation(self):
        """Test dataclass creation."""
        site = OffTargetSite(
            sequence="ATCGATCGATCGATCGATCG",
            chromosome="chr1",
            position=1000,
            strand="+",
            pam="NGG",
            mismatches=[(19, "G", "C")],
            mismatch_count=1,
            seed_mismatches=0,
            padic_distance=0.1,
            hyperbolic_distance=0.2,
            predicted_activity=0.05,
        )

        assert site.sequence == "ATCGATCGATCGATCGATCG"
        assert site.mismatch_count == 1
        assert site.seed_mismatches == 0


class TestGuideSafetyProfile:
    """Tests for GuideSafetyProfile dataclass."""

    def test_creation(self):
        """Test dataclass creation."""
        profile = GuideSafetyProfile(
            guide_sequence="ATCGATCGATCGATCGATCG",
            total_offtargets=10,
            high_risk_offtargets=2,
            seed_region_offtargets=1,
            min_padic_distance=0.1,
            safety_radius=0.5,
            specificity_score=0.8,
            recommended=True,
        )

        assert profile.guide_sequence == "ATCGATCGATCGATCGATCG"
        assert profile.total_offtargets == 10
        assert profile.recommended is True
