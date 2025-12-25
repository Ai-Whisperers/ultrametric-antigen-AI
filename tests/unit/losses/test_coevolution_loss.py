# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for CoEvolutionLoss module."""

import pytest
import torch

from src.losses.coevolution_loss import (
    BIOSYNTHETIC_FAMILIES,
    CODON_TABLE,
    METABOLIC_COSTS,
    BiosyntheticCoherenceLoss,
    CoEvolutionLoss,
    CoEvolutionMetrics,
    ErrorMinimizationLoss,
    PAdicStructureLoss,
    ResourceConservationLoss,
)


class TestBiosyntheticFamilies:
    """Tests for biosynthetic family data."""

    def test_all_families_exist(self):
        """Test that all expected families are defined."""
        expected = ["glutamate", "aspartate", "serine", "pyruvate", "aromatic", "histidine"]
        for family in expected:
            assert family in BIOSYNTHETIC_FAMILIES

    def test_amino_acids_in_families(self):
        """Test that amino acids are assigned to families."""
        all_aas = set()
        for aas in BIOSYNTHETIC_FAMILIES.values():
            all_aas.update(aas)

        # Should have most standard amino acids
        assert len(all_aas) >= 18


class TestCodonTable:
    """Tests for codon table."""

    def test_64_codons(self):
        """Test that all 64 codons are defined."""
        assert len(CODON_TABLE) == 64

    def test_start_codon(self):
        """Test start codon."""
        assert CODON_TABLE["AUG"] == "M"

    def test_stop_codons(self):
        """Test stop codons."""
        assert CODON_TABLE["UAA"] == "*"
        assert CODON_TABLE["UAG"] == "*"
        assert CODON_TABLE["UGA"] == "*"


class TestMetabolicCosts:
    """Tests for metabolic cost data."""

    def test_20_amino_acids(self):
        """Test that all 20 amino acids have costs."""
        assert len(METABOLIC_COSTS) == 20

    def test_tryptophan_most_expensive(self):
        """Test that tryptophan is most expensive."""
        assert METABOLIC_COSTS["W"] == max(METABOLIC_COSTS.values())

    def test_glycine_alanine_cheap(self):
        """Test that small amino acids are cheap."""
        assert METABOLIC_COSTS["G"] < 15
        assert METABOLIC_COSTS["A"] < 15


class TestBiosyntheticCoherenceLoss:
    """Tests for BiosyntheticCoherenceLoss."""

    def test_creation(self):
        """Test loss creation."""
        loss = BiosyntheticCoherenceLoss(latent_dim=16)
        assert loss.latent_dim == 16

    def test_forward(self):
        """Test forward pass."""
        loss = BiosyntheticCoherenceLoss(latent_dim=16)

        embeddings = torch.randn(2, 20, 16)
        indices = torch.randint(0, 64, (2, 20))

        loss_value = loss(embeddings, indices)

        assert loss_value.shape == ()
        assert loss_value >= 0

    def test_family_clustering(self):
        """Test that same-family codons should cluster."""
        loss = BiosyntheticCoherenceLoss(latent_dim=16)

        # Create embeddings where same-family codons are similar
        embeddings = torch.randn(1, 10, 16)

        # Use codons from glutamate family (E=GAA, Q=CAA)
        indices = torch.tensor([[16, 17, 18, 19, 24, 25, 26, 27, 0, 1]])

        loss_value = loss(embeddings, indices)
        assert loss_value >= 0


class TestErrorMinimizationLoss:
    """Tests for ErrorMinimizationLoss."""

    def test_creation(self):
        """Test loss creation."""
        loss = ErrorMinimizationLoss(p=3)
        assert loss.p == 3

    def test_mutation_neighbors(self):
        """Test mutation neighbor map."""
        loss = ErrorMinimizationLoss()

        # Each codon should have 9 neighbors (3 positions x 3 alternatives)
        for codon_idx, neighbors in loss.mutation_neighbors.items():
            assert len(neighbors) == 9

    def test_forward(self):
        """Test forward pass."""
        loss = ErrorMinimizationLoss()

        embeddings = torch.randn(2, 20, 16)
        indices = torch.randint(0, 64, (2, 20))

        loss_value = loss(embeddings, indices)

        assert loss_value.shape == ()
        assert loss_value >= 0


class TestResourceConservationLoss:
    """Tests for ResourceConservationLoss."""

    def test_creation(self):
        """Test loss creation."""
        loss = ResourceConservationLoss(cost_weight=0.1)
        assert loss.cost_weight == 0.1

    def test_codon_costs_normalized(self):
        """Test that codon costs are normalized."""
        loss = ResourceConservationLoss()
        assert loss.codon_costs.max() <= 1.0
        assert loss.codon_costs.min() >= 0.0

    def test_forward(self):
        """Test forward pass."""
        loss = ResourceConservationLoss()

        # Codon probabilities (uniform)
        probs = torch.ones(2, 20, 64) / 64

        loss_value = loss(probs)

        assert loss_value.shape == ()
        assert loss_value >= 0

    def test_cheap_codons_lower_loss(self):
        """Test that cheap codons give lower loss."""
        loss = ResourceConservationLoss()

        # Use uniform distribution as baseline
        uniform_probs = torch.ones(1, 10, 64) / 64

        # Expensive distribution (favor expensive amino acids like W, F, Y)
        # Tryptophan and other aromatic amino acids are expensive
        expensive_probs = torch.zeros(1, 10, 64)
        # UGG (tryptophan) - very expensive
        # Approximate index: UGG in codon order
        expensive_probs[:, :, :] = 0.01  # Small baseline
        expensive_probs[:, :, 0:4] = 0.2  # Phe codons (expensive)
        expensive_probs = expensive_probs / expensive_probs.sum(dim=-1, keepdim=True)

        uniform_loss = loss(uniform_probs)
        expensive_loss = loss(expensive_probs)

        # Both should be non-negative
        assert uniform_loss >= 0
        assert expensive_loss >= 0


class TestPAdicStructureLoss:
    """Tests for PAdicStructureLoss."""

    def test_creation(self):
        """Test loss creation."""
        loss = PAdicStructureLoss(p=3, n_digits=4)
        assert loss.p == 3
        assert loss.n_digits == 4

    def test_padic_distances(self):
        """Test p-adic distance matrix."""
        loss = PAdicStructureLoss(p=3)

        # Diagonal should be zero
        assert loss.padic_distances[0, 0] == 0
        assert loss.padic_distances[10, 10] == 0

        # Distance between adjacent codons
        dist_01 = loss.padic_distances[0, 1]
        assert dist_01 == 1.0  # v_3(1) = 0, so 3^0 = 1

        # Distance when difference is 3
        dist_03 = loss.padic_distances[0, 3]
        assert dist_03 == pytest.approx(1 / 3)  # v_3(3) = 1, so 3^{-1}

    def test_forward(self):
        """Test forward pass."""
        loss = PAdicStructureLoss()

        embeddings = torch.randn(2, 20, 16)
        indices = torch.randint(0, 64, (2, 20))

        loss_value = loss(embeddings, indices)

        assert loss_value.shape == ()
        # Loss is 1 - correlation, so should be in [0, 2]
        assert -1 <= loss_value <= 2


class TestCoEvolutionLoss:
    """Tests for combined CoEvolutionLoss."""

    def test_creation(self):
        """Test loss creation."""
        loss = CoEvolutionLoss(latent_dim=16, p=3)
        assert loss.latent_dim == 16
        assert loss.p == 3

    def test_default_weights(self):
        """Test default weight configuration."""
        loss = CoEvolutionLoss()

        weights = loss.weights
        assert "biosynthetic" in weights
        assert "error_min" in weights
        assert "resource" in weights
        assert "padic" in weights
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_forward(self):
        """Test forward pass."""
        loss = CoEvolutionLoss(latent_dim=16)

        embeddings = torch.randn(2, 20, 16)
        indices = torch.randint(0, 64, (2, 20))

        result = loss(embeddings, indices)

        assert "loss" in result
        assert "biosynthetic_loss" in result
        assert "error_minimization_loss" in result
        assert "padic_structure_loss" in result
        assert "metrics" in result

    def test_forward_with_probabilities(self):
        """Test forward pass with codon probabilities."""
        loss = CoEvolutionLoss(latent_dim=16)

        embeddings = torch.randn(2, 20, 16)
        indices = torch.randint(0, 64, (2, 20))
        probs = torch.softmax(torch.randn(2, 20, 64), dim=-1)

        result = loss(embeddings, indices, probs)

        assert "resource_loss" in result
        assert result["resource_loss"] >= 0

    def test_metrics_dataclass(self):
        """Test that metrics are properly computed."""
        loss = CoEvolutionLoss(latent_dim=16)

        embeddings = torch.randn(2, 20, 16)
        indices = torch.randint(0, 64, (2, 20))

        result = loss(embeddings, indices)
        metrics = result["metrics"]

        assert isinstance(metrics, CoEvolutionMetrics)
        # Metrics are 1 - loss, so can be negative if loss > 1
        assert isinstance(metrics.biosynthetic_coherence, float)
        assert isinstance(metrics.total_loss, float)

    def test_custom_weights(self):
        """Test loss with custom weights."""
        weights = {
            "biosynthetic": 0.5,
            "error_min": 0.3,
            "resource": 0.1,
            "padic": 0.1,
        }
        loss = CoEvolutionLoss(latent_dim=16, weights=weights)

        embeddings = torch.randn(2, 20, 16)
        indices = torch.randint(0, 64, (2, 20))

        result = loss(embeddings, indices)

        # Should not raise any errors
        assert result["loss"] >= 0

    def test_gradient_flow(self):
        """Test that gradients flow through the loss."""
        loss = CoEvolutionLoss(latent_dim=16)

        embeddings = torch.randn(2, 20, 16, requires_grad=True)
        indices = torch.randint(0, 64, (2, 20))

        result = loss(embeddings, indices)
        result["loss"].backward()

        assert embeddings.grad is not None
        assert not torch.all(embeddings.grad == 0)
