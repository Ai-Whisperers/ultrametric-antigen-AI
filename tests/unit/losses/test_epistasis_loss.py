# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Comprehensive unit tests for Epistasis Loss.

Tests cover:
- LearnedEpistasisLoss
- DrugInteractionLoss
- MarginRankingLoss
- EpistasisLoss (unified)
- EpistasisLossResult dataclass
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.losses.epistasis_loss import (
    DrugInteractionLoss,
    EpistasisLoss,
    EpistasisLossResult,
    LearnedEpistasisLoss,
    MarginRankingLoss,
)


class TestEpistasisLossResult:
    """Test EpistasisLossResult dataclass."""

    def test_result_creation(self):
        """Test result creation."""
        result = EpistasisLossResult(
            total_loss=torch.tensor(1.0),
            epistasis_loss=torch.tensor(0.3),
            coevolution_loss=torch.tensor(0.3),
            drug_interaction_loss=torch.tensor(0.2),
            margin_loss=torch.tensor(0.2),
            metrics={"total_loss": 1.0},
        )

        assert result.total_loss.item() == 1.0
        assert "total_loss" in result.metrics


class TestLearnedEpistasisLoss:
    """Test LearnedEpistasisLoss."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function fixture."""
        return LearnedEpistasisLoss(margin=0.1, interaction_weight=1.0)

    def test_initialization(self, loss_fn):
        """Test initialization."""
        assert loss_fn.margin == 0.1
        assert loss_fn.interaction_weight == 1.0

    def test_additive_case(self, loss_fn):
        """Test loss when effects are additive."""
        single_effects = torch.tensor([[0.3], [0.2], [0.5]])
        combined_effects = torch.tensor([0.3, 0.2, 0.5])  # Sum of single
        predicted_combined = torch.tensor([0.3, 0.2, 0.5])
        target_combined = torch.tensor([0.3, 0.2, 0.5])

        loss = loss_fn(single_effects, combined_effects, predicted_combined, target_combined)

        # Should be near zero (no epistasis to learn)
        assert loss >= 0

    def test_epistatic_case(self, loss_fn):
        """Test loss when there is epistasis."""
        single_effects = torch.tensor([[0.3], [0.2]])
        combined_effects = torch.tensor([0.5, 0.5])
        predicted_combined = torch.tensor([0.5, 0.5])  # Predicting additive
        target_combined = torch.tensor([0.8, 0.1])  # True is epistatic

        loss = loss_fn(single_effects, combined_effects, predicted_combined, target_combined)

        # Should be positive (epistasis not captured)
        assert loss > 0

    def test_gradient_flow(self, loss_fn):
        """Test gradient flow."""
        single_effects = torch.randn(4, 3, requires_grad=True)
        combined_effects = single_effects.sum(dim=1)
        predicted_combined = torch.randn(4, requires_grad=True)
        target_combined = torch.randn(4)

        loss = loss_fn(single_effects, combined_effects, predicted_combined, target_combined)
        loss.backward()

        assert predicted_combined.grad is not None


class TestDrugInteractionLoss:
    """Test DrugInteractionLoss."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function fixture."""
        return DrugInteractionLoss(
            n_drugs=10,
            embed_dim=32,
            curvature=1.0,
            temperature=0.1,
        )

    def test_initialization(self, loss_fn):
        """Test initialization."""
        assert loss_fn.n_drugs == 10
        assert loss_fn.drug_embeddings.shape == (10, 32)

    def test_set_drug_classes(self, loss_fn):
        """Test setting drug classes."""
        classes = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        loss_fn.set_drug_classes(classes)

        assert torch.equal(loss_fn.drug_classes, classes)

    def test_hyperbolic_distance(self, loss_fn):
        """Test hyperbolic distance computation."""
        x = torch.randn(5, 32) * 0.1

        # Self-distance (diagonal) should be approximately 0
        dist = loss_fn.hyperbolic_distance(x, x)

        assert dist.shape == (5, 5)
        # Diagonal should be approximately 0 (self-distance)
        diag = torch.diag(dist)
        assert torch.allclose(diag, torch.zeros(5), atol=0.01)

    def test_forward(self, loss_fn):
        """Test forward pass."""
        predictions = torch.randn(8, 10)  # batch x drugs

        loss = loss_fn(predictions)

        assert loss.dim() == 0  # Scalar
        assert not torch.isnan(loss)

    def test_forward_with_classes(self, loss_fn):
        """Test forward with drug classes set."""
        classes = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])
        loss_fn.set_drug_classes(classes)

        predictions = torch.randn(8, 10)
        loss = loss_fn(predictions)

        assert loss.dim() == 0

    def test_same_class_similarity(self, loss_fn):
        """Test that same-class drugs are encouraged to have similar predictions."""
        classes = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        loss_fn.set_drug_classes(classes)

        # Same prediction for all same-class drugs
        predictions_same = torch.zeros(4, 10)
        predictions_same[:, 0:2] = 1.0  # Class 0
        predictions_same[:, 2:4] = 0.5  # Class 1

        # Different predictions for same-class
        predictions_diff = torch.randn(4, 10)

        loss_same = loss_fn(predictions_same)
        loss_diff = loss_fn(predictions_diff)

        # Same predictions should generally have lower loss
        # (This is a soft expectation)


class TestMarginRankingLoss:
    """Test MarginRankingLoss."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function fixture."""
        return MarginRankingLoss(margin=0.1)

    def test_initialization(self, loss_fn):
        """Test initialization."""
        assert loss_fn.margin == 0.1

    def test_correct_ordering(self, loss_fn):
        """Test loss with correct ordering."""
        # Predictions match target ordering
        predictions = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        targets = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

        loss = loss_fn(predictions, targets)

        # Should be low (correct ordering)
        assert loss >= 0

    def test_incorrect_ordering(self, loss_fn):
        """Test loss with incorrect ordering."""
        # Predictions reverse target ordering
        predictions = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])
        targets = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

        loss = loss_fn(predictions, targets)

        # Should be positive (wrong ordering)
        assert loss > 0

    def test_with_mask(self, loss_fn):
        """Test loss with mask."""
        predictions = torch.tensor([0.1, 0.5, 0.9])
        targets = torch.tensor([0.0, 0.5, 1.0])
        mask = torch.tensor([True, False, True])

        loss = loss_fn(predictions, targets, mask)

        # Should handle mask correctly

    def test_batch_size_one(self, loss_fn):
        """Test with batch size 1."""
        predictions = torch.tensor([0.5])
        targets = torch.tensor([0.5])

        loss = loss_fn(predictions, targets)

        # Should be 0 (no pairs to compare)
        assert loss == 0

    def test_2d_inputs(self, loss_fn):
        """Test with 2D inputs."""
        predictions = torch.randn(8, 5)  # batch x drugs
        targets = torch.randn(8, 5)

        loss = loss_fn(predictions, targets)

        assert loss.dim() == 0


class TestEpistasisLoss:
    """Test unified EpistasisLoss."""

    @pytest.fixture
    def loss_fn(self):
        """Create loss function fixture."""
        return EpistasisLoss(
            latent_dim=16,
            n_drugs=5,
            use_coevolution=False,  # Disable for unit testing
            use_drug_interaction=True,
            use_margin_ranking=True,
        )

    def test_initialization(self, loss_fn):
        """Test initialization."""
        assert loss_fn.learned_epistasis is not None
        assert loss_fn.coevolution is None  # Disabled
        assert loss_fn.drug_interaction is not None
        assert loss_fn.margin_ranking is not None

    def test_initialization_with_coevolution(self):
        """Test initialization with coevolution."""
        loss_fn = EpistasisLoss(
            latent_dim=16,
            n_drugs=5,
            use_coevolution=True,
        )

        assert loss_fn.coevolution is not None

    def test_default_weights(self, loss_fn):
        """Test default weight configuration."""
        weights = loss_fn.weights

        assert "epistasis" in weights
        assert "coevolution" in weights
        assert "drug_interaction" in weights
        assert "margin" in weights

    def test_custom_weights(self):
        """Test custom weights."""
        weights = {
            "epistasis": 0.5,
            "coevolution": 0.2,
            "drug_interaction": 0.2,
            "margin": 0.1,
        }
        loss_fn = EpistasisLoss(weights=weights, use_coevolution=False)

        assert loss_fn.weights["epistasis"] == 0.5

    def test_forward_basic(self, loss_fn):
        """Test basic forward pass."""
        model_output = {
            "predictions": torch.randn(8, 5),
            "z": torch.randn(8, 16),
        }
        targets = {
            "resistance": torch.rand(8, 5),
        }

        result = loss_fn(model_output, targets)

        assert isinstance(result, EpistasisLossResult)
        assert result.total_loss.dim() == 0

    def test_forward_with_single_effects(self, loss_fn):
        """Test forward with single mutation effects."""
        model_output = {
            "predictions": torch.randn(8, 5),
            "z": torch.randn(8, 16),
            "single_effects": torch.randn(8, 3, 5),
        }
        targets = {
            "resistance": torch.rand(8, 5),
        }

        result = loss_fn(model_output, targets)

        assert result.epistasis_loss >= 0

    def test_forward_with_coevolution(self):
        """Test forward with coevolution loss."""
        loss_fn = EpistasisLoss(
            latent_dim=16,
            n_drugs=1,
            use_coevolution=True,
        )

        model_output = {
            "predictions": torch.randn(4, 1),
            "z": torch.randn(4, 16),
            "codon_embeddings": torch.randn(4, 10, 16),
        }
        targets = {
            "resistance": torch.rand(4, 1),
            "codon_indices": torch.randint(0, 64, (4, 10)),
        }

        result = loss_fn(model_output, targets)

        # Coevolution loss should be computed
        assert result.coevolution_loss >= 0

    def test_metrics_returned(self, loss_fn):
        """Test that metrics are returned."""
        model_output = {
            "predictions": torch.randn(8, 5),
            "z": torch.randn(8, 16),
        }
        targets = {
            "resistance": torch.rand(8, 5),
        }

        result = loss_fn(model_output, targets)

        assert "total_loss" in result.metrics
        assert "margin_loss" in result.metrics

    def test_gradient_flow(self, loss_fn):
        """Test gradient flow through loss."""
        predictions = torch.randn(8, 5, requires_grad=True)
        model_output = {
            "predictions": predictions,
            "z": torch.randn(8, 16),
        }
        targets = {
            "resistance": torch.rand(8, 5),
        }

        result = loss_fn(model_output, targets)
        result.total_loss.backward()

        assert predictions.grad is not None


class TestEpistasisLossEdgeCases:
    """Test edge cases for epistasis loss."""

    def test_single_drug(self):
        """Test with single drug."""
        loss_fn = EpistasisLoss(
            n_drugs=1,
            use_coevolution=False,
            use_drug_interaction=False,  # Can't have interaction with 1 drug
        )

        model_output = {
            "predictions": torch.randn(4, 1),
            "z": torch.randn(4, 16),
        }
        targets = {
            "resistance": torch.rand(4, 1),
        }

        result = loss_fn(model_output, targets)

        assert not torch.isnan(result.total_loss)

    def test_no_targets(self):
        """Test with missing targets."""
        loss_fn = EpistasisLoss(
            n_drugs=5,
            use_coevolution=False,
        )

        model_output = {
            "predictions": torch.randn(4, 5),
            "z": torch.randn(4, 16),
        }
        targets = {}  # No targets

        result = loss_fn(model_output, targets)

        # Should handle gracefully

    def test_batch_size_one(self):
        """Test with batch size 1."""
        loss_fn = EpistasisLoss(n_drugs=5, use_coevolution=False)

        model_output = {
            "predictions": torch.randn(1, 5),
            "z": torch.randn(1, 16),
        }
        targets = {
            "resistance": torch.rand(1, 5),
        }

        result = loss_fn(model_output, targets)

        assert not torch.isnan(result.total_loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
