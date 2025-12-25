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
        module = PAdicSequenceDistance(p=3, guide_length=20)
        assert module.p == 3
        assert module.guide_length == 20

    def test_classify_mismatch(self):
        """Test mismatch classification."""
        module = PAdicSequenceDistance()

        # Matches
        assert module.classify_mismatch("A", "A") == MismatchType.MATCH
        assert module.classify_mismatch("G", "G") == MismatchType.MATCH

        # Transitions (purine-purine or pyrimidine-pyrimidine)
        assert module.classify_mismatch("A", "G") == MismatchType.TRANSITION
        assert module.classify_mismatch("C", "T") == MismatchType.TRANSITION

        # Transversions (purine-pyrimidine)
        assert module.classify_mismatch("A", "C") == MismatchType.TRANSVERSION
        assert module.classify_mismatch("G", "T") == MismatchType.TRANSVERSION

    def test_compute_position_weights(self):
        """Test position weight computation."""
        module = PAdicSequenceDistance()

        weights = module.compute_position_weights()
        assert len(weights) == 20

        # Seed region (positions 0-11) should have higher weights
        seed_mean = sum(weights[:12]) / 12
        nonseed_mean = sum(weights[12:]) / 8
        assert seed_mean > nonseed_mean

    def test_compute_distance(self):
        """Test distance computation between sequences."""
        module = PAdicSequenceDistance()

        guide = "ATCGATCGATCGATCGATCG"
        # Same sequence - distance should be 0 or very small
        dist_same = module.compute_distance(guide, guide)
        assert dist_same < 0.001

        # Different sequence - distance should be larger
        offtarget = "ATCGATCGATCGATCGATCC"  # One mismatch at last position
        dist_diff = module.compute_distance(guide, offtarget)
        assert dist_diff > dist_same

    def test_compute_distance_batch(self):
        """Test batch distance computation."""
        module = PAdicSequenceDistance()

        guide = "ATCGATCGATCGATCGATCG"
        offtargets = [
            "ATCGATCGATCGATCGATCG",  # Perfect match
            "ATCGATCGATCGATCGATCC",  # 1 mismatch non-seed
            "GTCGATCGATCGATCGATCG",  # 1 mismatch seed
        ]
        distances = module.compute_distance_batch(guide, offtargets)

        assert len(distances) == 3
        assert distances[0] < distances[1]  # Perfect match < 1 mismatch
        assert distances[1] < distances[2]  # Non-seed mismatch < seed mismatch


class TestHyperbolicOfftargetEmbedder:
    """Tests for HyperbolicOfftargetEmbedder."""

    def test_creation(self):
        """Test embedder creation."""
        embedder = HyperbolicOfftargetEmbedder(
            embedding_dim=64, hidden_dim=128, curvature=1.0
        )
        assert embedder.embedding_dim == 64
        assert embedder.hidden_dim == 128
        assert embedder.curvature == 1.0

    def test_encode_sequence(self):
        """Test sequence encoding."""
        embedder = HyperbolicOfftargetEmbedder(embedding_dim=32)

        seq = "ATCGATCGATCGATCGATCG"
        encoding = embedder.encode_sequence(seq)

        assert encoding.shape == (20, 5)  # sequence_length x vocab_size

    def test_embed_sequence(self):
        """Test hyperbolic embedding."""
        embedder = HyperbolicOfftargetEmbedder(embedding_dim=32)

        seq = "ATCGATCGATCGATCGATCG"
        embedding = embedder.embed_sequence(seq)

        assert embedding.shape == (32,)
        # Check embedding is in Poincare ball (norm < 1)
        assert torch.norm(embedding) < 1.0

    def test_poincare_distance(self):
        """Test Poincare distance computation."""
        embedder = HyperbolicOfftargetEmbedder(embedding_dim=32)

        x = torch.zeros(32)
        y = torch.zeros(32)
        y[0] = 0.5  # Point at distance 0.5 from origin

        dist = embedder.poincare_distance(x, y)
        assert dist > 0

        # Distance to self should be 0
        dist_self = embedder.poincare_distance(x, x)
        assert dist_self < 1e-6

    def test_forward(self):
        """Test forward pass."""
        embedder = HyperbolicOfftargetEmbedder(embedding_dim=32)

        guide = "ATCGATCGATCGATCGATCG"
        offtargets = [
            "ATCGATCGATCGATCGATCG",
            "ATCGATCGATCGATCGATCC",
        ]

        result = embedder(guide, offtargets)

        assert "guide_embedding" in result
        assert "offtarget_embeddings" in result
        assert "hyperbolic_distances" in result
        assert len(result["offtarget_embeddings"]) == 2


class TestOfftargetActivityPredictor:
    """Tests for OfftargetActivityPredictor."""

    def test_creation(self):
        """Test predictor creation."""
        predictor = OfftargetActivityPredictor(embedding_dim=64, hidden_dim=128)
        assert predictor.embedding_dim == 64

    def test_predict_activity(self):
        """Test activity prediction."""
        predictor = OfftargetActivityPredictor(embedding_dim=32)

        # Create mock embeddings
        guide_emb = torch.randn(32) * 0.5  # Scale to fit in Poincare ball
        guide_emb = guide_emb / (torch.norm(guide_emb) + 1.0)

        offtarget_emb = torch.randn(32) * 0.5
        offtarget_emb = offtarget_emb / (torch.norm(offtarget_emb) + 1.0)

        activity = predictor.predict_activity(guide_emb, offtarget_emb)

        assert 0.0 <= activity <= 1.0

    def test_forward(self):
        """Test forward pass."""
        predictor = OfftargetActivityPredictor(embedding_dim=32)

        # Create mock inputs
        guide_emb = torch.randn(32) * 0.3
        offtarget_embs = [torch.randn(32) * 0.3 for _ in range(3)]

        activities = predictor(guide_emb, offtarget_embs)

        assert len(activities) == 3
        assert all(0.0 <= a <= 1.0 for a in activities)


class TestCRISPROfftargetAnalyzer:
    """Tests for CRISPROfftargetAnalyzer."""

    def test_creation(self):
        """Test analyzer creation."""
        analyzer = CRISPROfftargetAnalyzer(p=3, embedding_dim=32)
        assert analyzer.p == 3
        assert analyzer.embedding_dim == 32

    def test_analyze_offtarget(self):
        """Test single off-target analysis."""
        analyzer = CRISPROfftargetAnalyzer(embedding_dim=32)

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
        assert site.mismatch_count == 1
        assert site.padic_distance > 0
        assert 0.0 <= site.predicted_activity <= 1.0

    def test_analyze_guide(self):
        """Test guide safety analysis."""
        analyzer = CRISPROfftargetAnalyzer(embedding_dim=32)

        guide = "ATCGATCGATCGATCGATCG"
        offtargets = [
            "ATCGATCGATCGATCGATCC",  # 1 mismatch
            "ATCGATCGATCGATCGATTG",  # 2 mismatches
            "ATCGATCGATCGATCGATCG",  # Perfect match (on-target)
        ]

        profile = analyzer.analyze_guide(guide, offtargets)

        assert isinstance(profile, GuideSafetyProfile)
        assert profile.guide_sequence == guide
        assert profile.total_offtargets == 3
        assert 0.0 <= profile.specificity_score <= 1.0

    def test_compute_landscape(self):
        """Test landscape computation."""
        analyzer = CRISPROfftargetAnalyzer(embedding_dim=32)

        guide = "ATCGATCGATCGATCGATCG"
        offtargets = [
            "ATCGATCGATCGATCGATCC",
            "ATCGATCGATCGATCGATTG",
        ]

        landscape = analyzer.compute_landscape(guide, offtargets)

        assert "guide_embedding" in landscape
        assert "offtarget_embeddings" in landscape
        assert "distances" in landscape
        assert "activities" in landscape


class TestGuideDesignOptimizer:
    """Tests for GuideDesignOptimizer."""

    def test_creation(self):
        """Test optimizer creation."""
        optimizer = GuideDesignOptimizer(
            embedding_dim=32, population_size=10, n_generations=5
        )
        assert optimizer.population_size == 10
        assert optimizer.n_generations == 5

    def test_evaluate_guide(self):
        """Test guide evaluation."""
        optimizer = GuideDesignOptimizer(embedding_dim=32)

        guide = "ATCGATCGATCGATCGATCG"
        offtargets = [
            "ATCGATCGATCGATCGATCC",
            "ATCGATCGATCGATCGATTG",
        ]

        score = optimizer.evaluate_guide(guide, offtargets)

        assert isinstance(score, float)
        assert score >= 0

    def test_mutate_guide(self):
        """Test guide mutation."""
        optimizer = GuideDesignOptimizer()

        guide = "ATCGATCGATCGATCGATCG"
        mutated = optimizer.mutate_guide(guide, mutation_rate=0.2)

        assert len(mutated) == len(guide)
        assert all(n in "ACGT" for n in mutated)

    def test_optimize(self):
        """Test guide optimization."""
        optimizer = GuideDesignOptimizer(
            embedding_dim=32, population_size=5, n_generations=2
        )

        target = "ATCGATCGATCGATCGATCG"
        offtargets = [
            "ATCGATCGATCGATCGATCC",
            "ATCGATCGATCGATCGATTG",
        ]

        best_guide, score = optimizer.optimize(
            target_region=target, known_offtargets=offtargets
        )

        assert len(best_guide) == 20
        assert all(n in "ACGT" for n in best_guide)
        assert isinstance(score, float)
