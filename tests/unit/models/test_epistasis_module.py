# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Comprehensive unit tests for Epistasis Module.

Tests cover:
- PairwiseInteractionModule
- HigherOrderInteractionModule
- EpistasisModule
- EpistasisPredictor
- EpistasisResult dataclass
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.epistasis_module import (
    EpistasisModule,
    EpistasisPredictor,
    EpistasisResult,
    HigherOrderInteractionModule,
    PairwiseInteractionModule,
)


class TestEpistasisResult:
    """Test EpistasisResult dataclass."""

    def test_result_creation(self):
        """Test result creation."""
        result = EpistasisResult(
            interaction_score=torch.tensor([0.5, 0.6]),
            antagonistic=torch.tensor([0.3, 0.4]),
            synergistic=torch.tensor([0.7, 0.6]),
        )

        assert result.interaction_score.shape == (2,)
        assert result.pairwise_scores is None

    def test_result_with_details(self):
        """Test result with all details."""
        result = EpistasisResult(
            interaction_score=torch.tensor([0.5]),
            pairwise_scores=torch.randn(1, 3, 3),
            higher_order_score=torch.tensor([0.2]),
            sign_epistasis=torch.tensor([True]),
            antagonistic=torch.tensor([0.3]),
            synergistic=torch.tensor([0.7]),
        )

        assert result.pairwise_scores is not None
        assert result.higher_order_score is not None


class TestPairwiseInteractionModule:
    """Test PairwiseInteractionModule."""

    @pytest.fixture
    def module(self):
        """Create module fixture."""
        return PairwiseInteractionModule(
            n_positions=300,
            embed_dim=64,
            n_amino_acids=21,
        )

    def test_initialization(self, module):
        """Test module initialization."""
        assert module.n_positions == 300
        assert module.embed_dim == 64
        assert module.position_embedding is not None

    def test_forward_shape(self, module):
        """Test forward pass shape."""
        positions = torch.randint(0, 300, (4, 5))  # 4 samples, 5 mutations each
        amino_acids = torch.randint(0, 21, (4, 5))

        scores = module(positions, amino_acids)

        assert scores.shape == (4, 5, 5)

    def test_forward_symmetry(self, module):
        """Test pairwise scores are symmetric."""
        positions = torch.randint(0, 300, (2, 3))
        scores = module(positions)

        # Should be approximately symmetric
        assert torch.allclose(scores, scores.transpose(1, 2), atol=1e-5)

    def test_diagonal_zero(self, module):
        """Test diagonal is zero (no self-interaction)."""
        positions = torch.randint(0, 300, (2, 4))
        scores = module(positions)

        diagonal = torch.diagonal(scores, dim1=1, dim2=2)
        assert torch.allclose(diagonal, torch.zeros_like(diagonal))

    def test_set_known_pairs(self, module):
        """Test setting known epistatic pairs."""
        pairs = [(10, 20), (50, 100), (184, 215)]
        module.set_known_epistatic_pairs(pairs)

        mask = module.known_pairs_mask
        assert mask[10, 20] == 1.0
        assert mask[20, 10] == 1.0
        assert mask[50, 100] == 1.0

    def test_without_aa_embedding(self):
        """Test without amino acid embedding."""
        module = PairwiseInteractionModule(
            n_positions=100,
            embed_dim=32,
            use_aa_embedding=False,
        )

        positions = torch.randint(0, 100, (2, 3))
        scores = module(positions)

        assert scores.shape == (2, 3, 3)


class TestHigherOrderInteractionModule:
    """Test HigherOrderInteractionModule."""

    @pytest.fixture
    def module(self):
        """Create module fixture."""
        return HigherOrderInteractionModule(
            embed_dim=64,
            n_heads=4,
            n_layers=2,
            max_mutations=20,
        )

    def test_initialization(self, module):
        """Test module initialization."""
        assert module.embed_dim == 64
        assert module.max_mutations == 20

    def test_forward_shape(self, module):
        """Test forward pass shape."""
        embeddings = torch.randn(4, 5, 64)  # 4 samples, 5 mutations
        score = module(embeddings)

        assert score.shape == (4,)

    def test_forward_with_mask(self, module):
        """Test forward with mask."""
        embeddings = torch.randn(4, 8, 64)
        mask = torch.ones(4, 8)
        mask[:, 5:] = 0  # Mask last 3 positions

        score = module(embeddings, mask)

        assert score.shape == (4,)

    def test_order_embedding(self, module):
        """Test order-specific embedding."""
        # Different number of mutations should give different results
        emb3 = torch.randn(2, 3, 64)
        emb5 = torch.randn(2, 5, 64)

        score3 = module(emb3)
        score5 = module(emb5)

        # Scores should be different (different order)
        assert not torch.allclose(score3, score5)


class TestEpistasisModule:
    """Test main EpistasisModule."""

    @pytest.fixture
    def module(self):
        """Create module fixture."""
        return EpistasisModule(
            n_positions=300,
            embed_dim=64,
            n_amino_acids=21,
        )

    def test_initialization(self, module):
        """Test module initialization."""
        assert module.n_positions == 300
        assert module.embed_dim == 64
        assert module.pairwise is not None
        assert module.higher_order is not None

    def test_forward_basic(self, module):
        """Test basic forward pass."""
        positions = torch.randint(0, 300, (4, 5))
        amino_acids = torch.randint(0, 21, (4, 5))

        result = module(positions, amino_acids)

        assert isinstance(result, EpistasisResult)
        assert result.interaction_score.shape == (4,)
        assert result.antagonistic.shape == (4,)
        assert result.synergistic.shape == (4,)

    def test_forward_with_details(self, module):
        """Test forward with return_details=True."""
        positions = torch.randint(0, 300, (2, 4))

        result = module(positions, return_details=True)

        assert result.pairwise_scores is not None
        assert result.pairwise_scores.shape == (2, 4, 4)
        assert result.higher_order_score is not None

    def test_antagonistic_synergistic_sum(self, module):
        """Test antagonistic + synergistic probabilities sum to 1."""
        positions = torch.randint(0, 300, (4, 5))

        result = module(positions)

        sums = result.antagonistic + result.synergistic
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_single_mutation(self, module):
        """Test with single mutation (no epistasis possible)."""
        positions = torch.randint(0, 300, (4, 1))

        result = module(positions)

        # Should still work
        assert result.interaction_score.shape == (4,)

    def test_two_mutations(self, module):
        """Test with two mutations (only pairwise)."""
        positions = torch.randint(0, 300, (4, 2))

        result = module(positions, return_details=True)

        # Higher order should be zero (need 3+ mutations)
        assert torch.allclose(result.higher_order_score, torch.zeros(4))

    def test_get_epistasis_matrix(self, module):
        """Test getting epistasis matrix."""
        matrix = module.get_epistasis_matrix()

        assert matrix.shape == (300, 300)
        # Should be symmetric
        assert torch.allclose(matrix, matrix.t())

    def test_get_top_epistatic_pairs(self, module):
        """Test getting top epistatic pairs."""
        pairs = module.get_top_epistatic_pairs(k=10, min_distance=5)

        assert len(pairs) <= 10
        for pos1, pos2, score in pairs:
            assert abs(pos1 - pos2) >= 5
            assert pos1 < pos2

    def test_without_higher_order(self):
        """Test module without higher-order interactions."""
        module = EpistasisModule(
            n_positions=100,
            embed_dim=32,
            use_higher_order=False,
        )

        positions = torch.randint(0, 100, (2, 5))
        result = module(positions, return_details=True)

        # Higher order should be zero
        assert torch.allclose(result.higher_order_score, torch.zeros(2))

    def test_temperature_effect(self):
        """Test temperature parameter."""
        module1 = EpistasisModule(n_positions=100, temperature=1.0)
        module2 = EpistasisModule(n_positions=100, temperature=0.1)

        positions = torch.randint(0, 100, (2, 3))

        # Copy weights for fair comparison
        module2.load_state_dict(module1.state_dict())

        result1 = module1(positions)
        result2 = module2(positions)

        # Lower temperature should give larger scores
        # (Before softmax normalization)
        assert not torch.allclose(result1.interaction_score, result2.interaction_score)


class TestEpistasisPredictor:
    """Test EpistasisPredictor."""

    @pytest.fixture
    def predictor(self):
        """Create predictor fixture."""
        return EpistasisPredictor(
            n_positions=300,
            n_amino_acids=21,
            embed_dim=64,
            n_outputs=5,  # 5 drugs
        )

    def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.epistasis is not None
        assert predictor.individual_effects.shape == (300, 21, 5)

    def test_forward(self, predictor):
        """Test forward pass."""
        positions = torch.randint(0, 300, (4, 3))
        amino_acids = torch.randint(0, 21, (4, 3))

        outputs = predictor(positions, amino_acids)

        assert "predictions" in outputs
        assert "epistasis_score" in outputs
        assert "individual_effects" in outputs

        assert outputs["predictions"].shape == (4, 5)
        assert outputs["epistasis_score"].shape == (4,)

    def test_individual_effects(self, predictor):
        """Test individual mutation effects."""
        positions = torch.randint(0, 300, (2, 4))
        amino_acids = torch.randint(0, 21, (2, 4))

        outputs = predictor(positions, amino_acids)

        # Individual effects should sum over mutations
        assert outputs["individual_effects"].shape == (2, 5)

    def test_pairwise_scores_available(self, predictor):
        """Test pairwise scores in output."""
        positions = torch.randint(0, 300, (2, 3))
        amino_acids = torch.randint(0, 21, (2, 3))

        outputs = predictor(positions, amino_acids)

        assert "pairwise_scores" in outputs
        assert outputs["pairwise_scores"].shape == (2, 3, 3)

    def test_sign_epistasis_detection(self, predictor):
        """Test sign epistasis detection."""
        positions = torch.randint(0, 300, (4, 5))
        amino_acids = torch.randint(0, 21, (4, 5))

        outputs = predictor(positions, amino_acids)

        assert "antagonistic" in outputs
        assert "synergistic" in outputs

        # Probabilities should sum to 1
        sums = outputs["antagonistic"] + outputs["synergistic"]
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestEpistasisGradients:
    """Test gradient flow through epistasis module."""

    @pytest.fixture
    def module(self):
        """Create module fixture."""
        return EpistasisModule(n_positions=100, embed_dim=32)

    def test_gradient_flow(self, module):
        """Test gradients flow through module."""
        positions = torch.randint(0, 100, (4, 3))
        amino_acids = torch.randint(0, 21, (4, 3))

        result = module(positions, amino_acids)
        loss = result.interaction_score.sum()

        loss.backward()

        # Check gradients exist for parameters in the compute path
        # (sign_detector not used in interaction_score.sum())
        has_gradient = False
        for name, param in module.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_gradient = True
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
        assert has_gradient, "At least one parameter should have a gradient"

    def test_predictor_gradients(self):
        """Test gradients through full predictor."""
        predictor = EpistasisPredictor(n_positions=50, n_outputs=3)

        positions = torch.randint(0, 50, (2, 2))
        amino_acids = torch.randint(0, 21, (2, 2))

        outputs = predictor(positions, amino_acids)
        loss = outputs["predictions"].sum()

        loss.backward()

        # Individual effects should have gradients
        assert predictor.individual_effects.grad is not None


class TestEpistasisEdgeCases:
    """Test edge cases."""

    def test_empty_positions(self):
        """Test with empty position tensor."""
        module = EpistasisModule(n_positions=100)

        # Empty positions should be handled gracefully
        positions = torch.randint(0, 100, (4, 0))

        # Should return zero interaction (no mutations to interact)
        result = module(positions)
        assert result.interaction_score.shape[0] == 4

    def test_large_positions(self):
        """Test with positions outside range."""
        module = EpistasisModule(n_positions=100)

        positions = torch.tensor([[150, 200]])  # Out of range

        # Should clamp
        result = module(positions)
        assert result.interaction_score is not None

    def test_batch_size_one(self):
        """Test with batch size 1."""
        module = EpistasisModule(n_positions=100)

        positions = torch.randint(0, 100, (1, 3))
        result = module(positions)

        assert result.interaction_score.shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
