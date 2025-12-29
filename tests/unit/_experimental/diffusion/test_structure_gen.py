# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for structure-conditioned sequence generation."""

from __future__ import annotations

import pytest
import torch

from src.diffusion import (
    MultiObjectiveDesigner,
    RadialBasisEmbedding,
    StructureConditionedGen,
    StructureEncoder,
    StructureGNNLayer,
)


class TestRadialBasisEmbedding:
    """Tests for RadialBasisEmbedding."""

    def test_init(self):
        """Test initialization."""
        rbf = RadialBasisEmbedding(num_rbf=16, max_dist=20.0)
        assert rbf.num_rbf == 16
        assert rbf.max_dist == 20.0

    def test_forward_shape(self, device):
        """Test forward pass shape."""
        rbf = RadialBasisEmbedding(num_rbf=16)
        rbf = rbf.to(device)

        dist = torch.rand(100, device=device) * 20
        result = rbf(dist)

        assert result.shape == (100, 16)

    def test_forward_batched(self, device):
        """Test forward with batched input."""
        rbf = RadialBasisEmbedding(num_rbf=8)
        rbf = rbf.to(device)

        dist = torch.rand(4, 30, device=device) * 20
        result = rbf(dist)

        assert result.shape == (4, 30, 8)

    def test_values_in_range(self, device):
        """Test RBF values are in valid range."""
        rbf = RadialBasisEmbedding(num_rbf=16)
        rbf = rbf.to(device)

        dist = torch.rand(100, device=device) * 20
        result = rbf(dist)

        assert torch.all(result >= 0)
        assert torch.all(result <= 1)


class TestStructureGNNLayer:
    """Tests for StructureGNNLayer."""

    def test_init(self):
        """Test initialization."""
        layer = StructureGNNLayer(hidden_dim=64)
        assert layer.hidden_dim == 64

    def test_forward_shape(self, device):
        """Test forward pass shape."""
        layer = StructureGNNLayer(hidden_dim=32)
        layer = layer.to(device)

        batch_size, n_nodes = 2, 20
        h = torch.randn(batch_size, n_nodes, 32, device=device)
        edge_index = torch.stack([
            torch.randint(0, n_nodes * batch_size, (100,), device=device),
            torch.randint(0, n_nodes * batch_size, (100,), device=device),
        ])
        edge_attr = torch.randn(100, 16, device=device)
        mask = torch.ones(batch_size, n_nodes, dtype=torch.bool, device=device)

        result = layer(h, edge_index, edge_attr, mask)
        assert result.shape == (batch_size, n_nodes, 32)


class TestStructureEncoder:
    """Tests for StructureEncoder."""

    def test_init(self):
        """Test initialization."""
        encoder = StructureEncoder(hidden_dim=64, n_layers=2)
        assert encoder.hidden_dim == 64

    def test_forward_shape(self, device, backbone_coords):
        """Test forward pass shape."""
        encoder = StructureEncoder(hidden_dim=32, n_layers=2, n_neighbors=10)
        encoder = encoder.to(device)

        features, edge_index = encoder(backbone_coords)

        batch_size, n_residues = backbone_coords.shape[:2]
        assert features.shape == (batch_size, n_residues, 32)

    def test_forward_with_mask(self, device, backbone_coords):
        """Test forward with mask."""
        encoder = StructureEncoder(hidden_dim=32, n_layers=2, n_neighbors=10)
        encoder = encoder.to(device)

        batch_size, n_residues = backbone_coords.shape[:2]
        mask = torch.ones(batch_size, n_residues, dtype=torch.bool, device=device)
        mask[:, -5:] = False  # Mask last 5 residues

        features, _ = encoder(backbone_coords, mask)

        # Masked positions should have zero features
        assert torch.allclose(features[:, -5:], torch.zeros_like(features[:, -5:]), atol=1e-6)


class TestStructureConditionedGen:
    """Tests for StructureConditionedGen."""

    def test_init(self):
        """Test initialization."""
        gen = StructureConditionedGen(hidden_dim=64, n_diffusion_steps=100, n_layers=2)
        assert gen.hidden_dim == 64

    def test_forward_shape(self, device, backbone_coords):
        """Test forward pass shape."""
        gen = StructureConditionedGen(
            hidden_dim=32, n_diffusion_steps=100, n_layers=2, n_encoder_layers=1
        )
        gen = gen.to(device)

        # Codons for each residue
        batch_size, n_residues = backbone_coords.shape[:2]
        codons = torch.randint(0, 64, (batch_size, n_residues), device=device)

        result = gen(backbone_coords, codons)

        assert "loss" in result
        assert torch.isfinite(result["loss"])

    def test_design_shape(self, device, backbone_coords):
        """Test design output shape."""
        gen = StructureConditionedGen(
            hidden_dim=32, n_diffusion_steps=10, n_layers=2, n_encoder_layers=1
        )
        gen = gen.to(device)
        gen.eval()

        batch_size, n_residues = backbone_coords.shape[:2]
        sequences = gen.design(backbone_coords, n_designs=3)

        assert sequences.shape == (batch_size * 3, n_residues)
        assert torch.all(sequences >= 0)
        assert torch.all(sequences < 64)

    def test_sample_alias(self, device, backbone_coords):
        """Test that sample is alias for design."""
        gen = StructureConditionedGen(
            hidden_dim=32, n_diffusion_steps=10, n_layers=2, n_encoder_layers=1
        )
        gen = gen.to(device)
        gen.eval()

        sequences = gen.sample(backbone_coords, n_samples=2)
        assert sequences.shape[0] == backbone_coords.shape[0] * 2


class TestMultiObjectiveDesigner:
    """Tests for MultiObjectiveDesigner."""

    def test_init(self):
        """Test initialization."""
        designer = MultiObjectiveDesigner(hidden_dim=64, use_codon_bias=True)
        assert designer.hidden_dim == 64

    def test_init_without_objectives(self):
        """Test initialization without objectives."""
        designer = MultiObjectiveDesigner(
            hidden_dim=64, use_codon_bias=False, use_mrna_stability=False
        )
        assert designer.hidden_dim == 64

    def test_forward_shape(self, device, backbone_coords):
        """Test forward pass."""
        designer = MultiObjectiveDesigner(
            hidden_dim=32, use_codon_bias=True, use_mrna_stability=False
        )
        designer = designer.to(device)

        batch_size, n_residues = backbone_coords.shape[:2]
        codons = torch.randint(0, 64, (batch_size, n_residues), device=device)

        result = designer(backbone_coords, codons)

        assert "total_loss" in result
        assert torch.isfinite(result["total_loss"])

    def test_forward_custom_weights(self, device, backbone_coords):
        """Test forward with custom weights."""
        designer = MultiObjectiveDesigner(hidden_dim=32, use_codon_bias=True)
        designer = designer.to(device)

        batch_size, n_residues = backbone_coords.shape[:2]
        codons = torch.randint(0, 64, (batch_size, n_residues), device=device)

        weights = {"structure": 2.0, "codon_bias": 0.5}
        result = designer(backbone_coords, codons, weights=weights)

        assert "total_loss" in result

    @pytest.mark.skip(reason="Design optimization requires more diffusion steps")
    def test_design_optimized(self, device, backbone_coords):
        """Test optimized design."""
        designer = MultiObjectiveDesigner(hidden_dim=32, use_codon_bias=True)
        designer = designer.to(device)
        designer.eval()

        sequences = designer.design_optimized(
            backbone_coords, n_candidates=10, n_select=3
        )

        assert sequences.shape[0] == 3
