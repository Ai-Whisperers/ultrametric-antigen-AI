# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for multi-objective optimization functions."""

import torch
import pytest

from src.objectives import (
    Objective,
    ObjectiveRegistry,
    ObjectiveResult,
    BindingObjective,
    SolubilityObjective,
    StabilityObjective,
    ManufacturabilityObjective,
    ProductionCostObjective,
)


class TestObjectiveResult:
    """Tests for ObjectiveResult dataclass."""

    def test_creation(self):
        """Test creating an ObjectiveResult."""
        score = torch.tensor([0.5, 0.3])
        result = ObjectiveResult(score=score, name="test")

        assert result.name == "test"
        assert torch.allclose(result.score, score)
        assert result.metadata == {}

    def test_with_metadata(self):
        """Test ObjectiveResult with metadata."""
        result = ObjectiveResult(
            score=torch.tensor(0.5),
            name="test",
            metadata={"key": "value"},
        )

        assert result.metadata["key"] == "value"

    def test_repr(self):
        """Test string representation."""
        result = ObjectiveResult(score=torch.tensor(0.5), name="test")
        assert "test" in repr(result)
        assert "0.5" in repr(result)


class TestObjectiveRegistry:
    """Tests for ObjectiveRegistry."""

    def test_empty_registry(self):
        """Test empty registry."""
        registry = ObjectiveRegistry()
        assert registry.num_objectives == 0
        assert registry.objective_names == []

    def test_register_objective(self):
        """Test registering an objective."""
        registry = ObjectiveRegistry()
        objective = SolubilityObjective()

        registry.register("sol", objective, weight=1.5)

        assert registry.num_objectives == 1
        assert "sol" in registry.objective_names
        assert registry._weights["sol"] == 1.5

    def test_unregister_objective(self):
        """Test unregistering an objective."""
        registry = ObjectiveRegistry()
        registry.register("sol", SolubilityObjective())
        registry.register("stab", StabilityObjective())

        registry.unregister("sol")

        assert registry.num_objectives == 1
        assert "sol" not in registry.objective_names

    def test_evaluate_all(self):
        """Test evaluating all objectives."""
        registry = ObjectiveRegistry()
        registry.register("sol", SolubilityObjective())
        registry.register("stab", StabilityObjective())

        latent = torch.randn(8, 16)
        results = registry.evaluate_all(latent)

        assert len(results) == 2
        assert "sol" in results
        assert "stab" in results
        assert results["sol"].score.shape == (8,)

    def test_get_score_matrix(self):
        """Test getting score matrix for Pareto optimization."""
        registry = ObjectiveRegistry()
        registry.register("sol", SolubilityObjective())
        registry.register("stab", StabilityObjective())
        registry.register("cost", ProductionCostObjective())

        latent = torch.randn(10, 32)
        scores = registry.get_score_matrix(latent)

        assert scores.shape == (10, 3)

    def test_weighted_sum(self):
        """Test computing weighted sum of objectives."""
        registry = ObjectiveRegistry()
        registry.register("sol", SolubilityObjective(), weight=2.0)
        registry.register("stab", StabilityObjective(), weight=1.0)

        latent = torch.randn(5, 16)
        weighted = registry.weighted_sum(latent)

        assert weighted.shape == (5,)

    def test_method_chaining(self):
        """Test method chaining for registration."""
        registry = (
            ObjectiveRegistry()
            .register("sol", SolubilityObjective())
            .register("stab", StabilityObjective())
            .register("cost", ProductionCostObjective())
        )

        assert registry.num_objectives == 3


class TestBindingObjective:
    """Tests for BindingObjective."""

    def test_initialization(self):
        """Test objective initialization."""
        objective = BindingObjective()
        assert objective.name == "binding"
        assert objective.weight == 1.0

    def test_evaluate_no_target(self):
        """Test evaluation without target profile."""
        objective = BindingObjective()
        latent = torch.randn(4, 16)

        result = objective.evaluate(latent)

        assert result.name == "binding"
        assert result.score.shape == (4,)

    def test_evaluate_with_target(self):
        """Test evaluation with target profile."""
        target = torch.randn(16)
        objective = BindingObjective(target_profile=target)
        latent = torch.randn(4, 16)

        result = objective.evaluate(latent)

        assert result.score.shape == (4,)
        assert "mean_similarity" in result.metadata

    def test_similar_vectors_bind_stronger(self):
        """Test that similar vectors have stronger binding (lower score)."""
        target = torch.randn(16)
        objective = BindingObjective(target_profile=target)

        # Similar latent
        similar = target.unsqueeze(0) + 0.1 * torch.randn(1, 16)
        # Dissimilar latent
        dissimilar = -target.unsqueeze(0) + 0.1 * torch.randn(1, 16)

        similar_result = objective.evaluate(similar)
        dissimilar_result = objective.evaluate(dissimilar)

        # Similar should have lower (better) score
        assert similar_result.score.item() < dissimilar_result.score.item()


class TestSolubilityObjective:
    """Tests for SolubilityObjective."""

    def test_initialization(self):
        """Test objective initialization."""
        objective = SolubilityObjective(target_hydrophobicity=-0.5)
        assert objective.name == "solubility"
        assert objective.target_hydrophobicity == -0.5

    def test_evaluate(self):
        """Test evaluation."""
        objective = SolubilityObjective()
        latent = torch.randn(8, 32)

        result = objective.evaluate(latent)

        assert result.score.shape == (8,)
        assert "mean_hydrophobicity" in result.metadata
        assert "mean_aggregation" in result.metadata

    def test_extreme_values_penalized(self):
        """Test that extreme values are penalized."""
        objective = SolubilityObjective()

        # Normal latent
        normal = torch.randn(4, 16) * 0.5
        # Extreme latent (high aggregation propensity)
        extreme = torch.randn(4, 16) * 5.0

        normal_result = objective.evaluate(normal)
        extreme_result = objective.evaluate(extreme)

        # Extreme should have higher (worse) score
        assert normal_result.score.mean() < extreme_result.score.mean()


class TestStabilityObjective:
    """Tests for StabilityObjective."""

    def test_initialization(self):
        """Test objective initialization."""
        objective = StabilityObjective(target_compactness=0.7)
        assert objective.name == "stability"
        assert objective.target_compactness == 0.7

    def test_evaluate(self):
        """Test evaluation."""
        objective = StabilityObjective()
        latent = torch.randn(6, 24)

        result = objective.evaluate(latent)

        assert result.score.shape == (6,)
        assert "mean_compactness" in result.metadata
        assert "mean_entropy" in result.metadata


class TestManufacturabilityObjective:
    """Tests for ManufacturabilityObjective."""

    def test_initialization(self):
        """Test objective initialization."""
        objective = ManufacturabilityObjective(expression_system="yeast")
        assert objective.name == "manufacturability"
        assert objective.expression_system == "yeast"

    def test_evaluate(self):
        """Test evaluation."""
        objective = ManufacturabilityObjective()
        latent = torch.randn(5, 20)

        result = objective.evaluate(latent)

        assert result.score.shape == (5,)
        assert result.metadata["expression_system"] == "ecoli"


class TestProductionCostObjective:
    """Tests for ProductionCostObjective."""

    def test_initialization(self):
        """Test objective initialization."""
        objective = ProductionCostObjective(base_cost_per_residue=2.0)
        assert objective.name == "production_cost"
        assert objective.base_cost_per_residue == 2.0

    def test_evaluate(self):
        """Test evaluation."""
        objective = ProductionCostObjective()
        latent = torch.randn(4, 16)

        result = objective.evaluate(latent)

        assert result.score.shape == (4,)
        assert "estimated_length" in result.metadata
        assert "rare_fraction" in result.metadata


class TestIntegration:
    """Integration tests for multi-objective system."""

    def test_full_pipeline(self):
        """Test full multi-objective evaluation pipeline."""
        # Create registry with all objectives
        registry = (
            ObjectiveRegistry()
            .register("binding", BindingObjective(), weight=2.0)
            .register("solubility", SolubilityObjective(), weight=1.5)
            .register("stability", StabilityObjective(), weight=1.0)
            .register("manufacturability", ManufacturabilityObjective(), weight=1.0)
            .register("cost", ProductionCostObjective(), weight=0.5)
        )

        # Generate batch of candidates
        latent = torch.randn(32, 64)

        # Get all scores
        results = registry.evaluate_all(latent)
        assert len(results) == 5

        # Get score matrix for Pareto
        score_matrix = registry.get_score_matrix(latent)
        assert score_matrix.shape == (32, 5)

        # Get weighted sum
        weighted = registry.weighted_sum(latent)
        assert weighted.shape == (32,)

    def test_pareto_integration(self):
        """Test integration with ParetoFrontOptimizer."""
        from src.training.optimizers.multi_objective import ParetoFrontOptimizer

        registry = (
            ObjectiveRegistry()
            .register("sol", SolubilityObjective())
            .register("stab", StabilityObjective())
        )

        # Generate candidates
        candidates = torch.randn(50, 32)
        scores = registry.get_score_matrix(candidates)

        # Find Pareto front
        optimizer = ParetoFrontOptimizer()
        front_candidates, front_scores = optimizer.identify_pareto_front(candidates, scores)

        # Should have fewer candidates on front than total
        assert front_candidates.shape[0] <= candidates.shape[0]
        assert front_scores.shape[0] == front_candidates.shape[0]
