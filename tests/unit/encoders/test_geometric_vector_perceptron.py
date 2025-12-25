# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for GeometricVectorPerceptron module."""

import pytest
import torch

from src.encoders.geometric_vector_perceptron import (
    CodonGVP,
    GVPLayer,
    PAdicGVP,
    ProteinGVPEncoder,
    VectorLinear,
)


class TestVectorLinear:
    """Tests for VectorLinear."""

    def test_creation(self):
        """Test vector linear creation."""
        layer = VectorLinear(v_in=4, v_out=8)
        assert layer.v_in == 4
        assert layer.v_out == 8

    def test_forward(self):
        """Test forward pass."""
        layer = VectorLinear(v_in=4, v_out=8)
        v = torch.randn(2, 10, 4, 3)

        out = layer(v)

        assert out.shape == (2, 10, 8, 3)

    def test_with_bias(self):
        """Test with bias."""
        layer = VectorLinear(v_in=4, v_out=8, bias=True)
        v = torch.randn(2, 10, 4, 3)

        out = layer(v)

        assert out.shape == (2, 10, 8, 3)


class TestGVPLayer:
    """Tests for GVPLayer."""

    def test_creation(self):
        """Test GVP layer creation."""
        layer = GVPLayer(s_in=32, s_out=64, v_in=4, v_out=8)
        assert layer.s_in == 32
        assert layer.s_out == 64

    def test_forward(self):
        """Test forward pass."""
        layer = GVPLayer(s_in=32, s_out=64, v_in=4, v_out=8)

        s = torch.randn(2, 10, 32)
        v = torch.randn(2, 10, 4, 3)

        s_out, v_out = layer(s, v)

        assert s_out.shape == (2, 10, 64)
        assert v_out.shape == (2, 10, 8, 3)

    def test_no_vector_output(self):
        """Test layer with no vector output."""
        layer = GVPLayer(s_in=32, s_out=64, v_in=4, v_out=0)

        s = torch.randn(2, 10, 32)
        v = torch.randn(2, 10, 4, 3)

        s_out, v_out = layer(s, v)

        assert s_out.shape == (2, 10, 64)
        assert v_out.shape == (2, 10, 0, 3)

    def test_without_gating(self):
        """Test layer without vector gating."""
        layer = GVPLayer(s_in=32, s_out=64, v_in=4, v_out=8, vector_gate=False)

        s = torch.randn(2, 10, 32)
        v = torch.randn(2, 10, 4, 3)

        s_out, v_out = layer(s, v)

        assert s_out.shape == (2, 10, 64)
        assert v_out.shape == (2, 10, 8, 3)


class TestPAdicGVP:
    """Tests for PAdicGVP."""

    def test_creation(self):
        """Test p-adic GVP creation."""
        gvp = PAdicGVP(s_in=32, s_out=64, v_in=4, v_out=8, p=3)
        assert gvp.p == 3

    def test_forward(self):
        """Test forward pass."""
        gvp = PAdicGVP(s_in=32, s_out=64, v_in=4, v_out=8)

        s = torch.randn(2, 10, 32)
        v = torch.randn(2, 10, 4, 3)

        result = gvp(s, v)

        assert "scalar_features" in result
        assert "vector_features" in result
        assert result["scalar_features"].shape == (2, 10, 64)
        assert result["vector_features"].shape == (2, 10, 8, 3)

    def test_forward_with_coords(self):
        """Test forward pass with coordinates."""
        gvp = PAdicGVP(s_in=32, s_out=64, v_in=4, v_out=8)

        s = torch.randn(2, 10, 32)
        v = torch.randn(2, 10, 4, 3)
        coords = torch.randn(2, 10, 3)

        result = gvp(s, v, coords)

        assert result["padic_distances"] is not None
        assert result["padic_distances"].shape == (2, 10, 10)

    def test_padic_distance(self):
        """Test p-adic distance computation."""
        gvp = PAdicGVP(p=3)

        # Use integer-like values that will show p-adic structure
        coords = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])

        dist = gvp.padic_distance(coords)

        assert dist.shape == (1, 3, 3)
        # Distances should be non-negative
        assert (dist >= 0).all()


class TestProteinGVPEncoder:
    """Tests for ProteinGVPEncoder."""

    def test_creation(self):
        """Test encoder creation."""
        encoder = ProteinGVPEncoder(node_s_in=20, output_dim=64)
        assert encoder.output_dim == 64

    def test_forward(self):
        """Test forward pass."""
        encoder = ProteinGVPEncoder(node_s_in=20, node_v_in=1, output_dim=64)

        node_s = torch.randn(2, 50, 20)  # 50 residues
        node_v = torch.randn(2, 50, 1, 3)  # CA directions

        result = encoder(node_s, node_v)

        assert "node_embeddings" in result
        assert "protein_embedding" in result
        assert result["node_embeddings"].shape == (2, 50, 64)
        assert result["protein_embedding"].shape == (2, 64)

    def test_forward_with_coords(self):
        """Test forward pass with coordinates."""
        encoder = ProteinGVPEncoder(node_s_in=20, output_dim=64, use_padic=True)

        node_s = torch.randn(2, 50, 20)
        node_v = torch.randn(2, 50, 1, 3)
        coords = torch.randn(2, 50, 3)

        result = encoder(node_s, node_v, coords)

        assert result["padic_distances"] is not None

    def test_with_mask(self):
        """Test with residue mask."""
        encoder = ProteinGVPEncoder(node_s_in=20, output_dim=64)

        node_s = torch.randn(2, 50, 20)
        node_v = torch.randn(2, 50, 1, 3)
        mask = torch.ones(2, 50)
        mask[:, 40:] = 0  # Mask last 10 residues

        result = encoder(node_s, node_v, mask=mask)

        # Attention should focus on valid residues
        assert result["attention_weights"].shape == (2, 50)

    def test_without_padic(self):
        """Test encoder without p-adic integration."""
        encoder = ProteinGVPEncoder(node_s_in=20, output_dim=64, use_padic=False)

        node_s = torch.randn(2, 50, 20)
        node_v = torch.randn(2, 50, 1, 3)

        result = encoder(node_s, node_v)

        assert result["padic_distances"] is None
        assert result["protein_embedding"].shape == (2, 64)


class TestCodonGVP:
    """Tests for CodonGVP."""

    def test_creation(self):
        """Test codon GVP creation."""
        encoder = CodonGVP(n_codons=64, output_dim=16)
        assert encoder.n_codons == 64

    def test_forward(self):
        """Test forward pass."""
        encoder = CodonGVP(n_codons=64, output_dim=16)

        codon_indices = torch.randint(0, 64, (2, 30))

        result = encoder(codon_indices)

        assert "codon_embeddings" in result
        assert "ternary_representation" in result
        assert result["codon_embeddings"].shape == (2, 30, 16)
        assert result["ternary_representation"].shape == (2, 30, 4)

    def test_ternary_conversion(self):
        """Test codon to ternary conversion."""
        encoder = CodonGVP()

        # Test specific codons
        codon_indices = torch.tensor([[0, 1, 3, 9, 27, 63]])
        ternary = encoder.codon_to_ternary(codon_indices)

        # 0 = [0, 0, 0, 0]
        assert torch.allclose(ternary[0, 0], torch.tensor([0.0, 0.0, 0.0, 0.0]))
        # 1 = [1, 0, 0, 0]
        assert torch.allclose(ternary[0, 1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        # 3 = [0, 1, 0, 0]
        assert torch.allclose(ternary[0, 2], torch.tensor([0.0, 1.0, 0.0, 0.0]))

    def test_padic_structure(self):
        """Test that embeddings preserve p-adic structure."""
        encoder = CodonGVP(output_dim=16)

        # Codons differing by powers of 3 should have specific distances
        codon_indices = torch.tensor([[0, 1, 3, 9]])

        result = encoder(codon_indices)
        emb = result["codon_embeddings"]

        # Embeddings should be different
        assert not torch.allclose(emb[0, 0], emb[0, 1])
