# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for codon symmetry layers."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.equivariant import (
    AMINO_ACIDS,
    GENETIC_CODE,
    CodonAttention,
    CodonEmbedding,
    CodonSymmetryLayer,
    CodonTransformer,
    SynonymousPooling,
    WobbleAwareConv,
    codon_to_index,
    get_synonymous_groups,
    get_wobble_equivalences,
    index_to_codon,
)


class TestCodonUtilities:
    """Tests for codon utility functions."""

    def test_codon_to_index(self):
        """Test codon to index conversion."""
        # TTT should be 0 (T=0, T=0, T=0 -> 0*16 + 0*4 + 0 = 0)
        assert codon_to_index("TTT") == 0
        # GGG should be 63 (G=3, G=3, G=3 -> 3*16 + 3*4 + 3 = 63)
        assert codon_to_index("GGG") == 63
        # ATG (start codon): A=2, T=0, G=3 -> 2*16 + 0*4 + 3 = 35
        assert codon_to_index("ATG") == 35

    def test_index_to_codon(self):
        """Test index to codon conversion."""
        assert index_to_codon(0) == "TTT"
        assert index_to_codon(63) == "GGG"
        assert index_to_codon(35) == "ATG"

    def test_roundtrip(self):
        """Test codon-index roundtrip."""
        for codon in GENETIC_CODE.keys():
            idx = codon_to_index(codon)
            recovered = index_to_codon(idx)
            assert recovered == codon

    def test_genetic_code_complete(self):
        """Test genetic code has all 64 codons."""
        assert len(GENETIC_CODE) == 64

    def test_amino_acids_complete(self):
        """Test all amino acids are present."""
        assert len(AMINO_ACIDS) == 21  # 20 amino acids + stop

    def test_get_synonymous_groups(self):
        """Test synonymous groups."""
        groups = get_synonymous_groups()

        # Should have 21 groups (20 amino acids + stop)
        assert len(groups) == 21

        # Leucine should have 6 codons
        assert len(groups["L"]) == 6

        # Methionine should have 1 codon (ATG)
        assert len(groups["M"]) == 1

        # Tryptophan should have 1 codon (TGG)
        assert len(groups["W"]) == 1

    def test_get_wobble_equivalences(self):
        """Test wobble equivalences."""
        pairs = get_wobble_equivalences()

        # Should have pairs
        assert len(pairs) > 0

        # Each pair should have same amino acid and differ at position 3
        for idx1, idx2 in pairs:
            codon1 = index_to_codon(idx1)
            codon2 = index_to_codon(idx2)

            # Same first two positions
            assert codon1[:2] == codon2[:2]
            # Same amino acid
            assert GENETIC_CODE[codon1] == GENETIC_CODE[codon2]


class TestCodonEmbedding:
    """Tests for CodonEmbedding."""

    def test_init_default(self):
        """Test default initialization."""
        embed = CodonEmbedding(embedding_dim=64)
        assert embed.embedding_dim == 64
        assert embed.share_synonymous is True

    def test_init_no_sharing(self):
        """Test initialization without synonymous sharing."""
        embed = CodonEmbedding(embedding_dim=64, share_synonymous=False)
        assert embed.share_synonymous is False

    def test_forward_shape(self, device, codon_sequence):
        """Test forward pass shape."""
        embed = CodonEmbedding(embedding_dim=64)
        embed = embed.to(device)

        result = embed(codon_sequence)
        assert result.shape == (4, 50, 64)

    def test_synonymous_codons_similar(self, device):
        """Test that synonymous codons have similar embeddings."""
        embed = CodonEmbedding(embedding_dim=64, share_synonymous=True, learn_deviation=False)
        embed = embed.to(device)

        # Get embeddings for synonymous codons (Leucine: TTG and CTG)
        ttg_idx = codon_to_index("TTG")
        ctg_idx = codon_to_index("CTG")

        codons = torch.tensor([[ttg_idx, ctg_idx]], device=device)
        embeddings = embed(codons)

        # Without deviation learning, should be exactly the same
        assert torch.allclose(embeddings[0, 0], embeddings[0, 1])

    def test_different_aa_different_embedding(self, device):
        """Test that different amino acids have different embeddings."""
        embed = CodonEmbedding(embedding_dim=64, share_synonymous=True)
        embed = embed.to(device)

        # Met (ATG) and Trp (TGG) are different amino acids
        atg_idx = codon_to_index("ATG")
        tgg_idx = codon_to_index("TGG")

        codons = torch.tensor([[atg_idx, tgg_idx]], device=device)
        embeddings = embed(codons)

        # Should be different
        assert not torch.allclose(embeddings[0, 0], embeddings[0, 1])


class TestSynonymousPooling:
    """Tests for SynonymousPooling."""

    def test_init(self):
        """Test initialization."""
        pool = SynonymousPooling(pool_type="mean")
        assert pool.pool_type == "mean"
        assert pool.mapping.shape == (64, 21)

    def test_forward_shape(self, device):
        """Test forward pass shape."""
        pool = SynonymousPooling(pool_type="mean")
        pool = pool.to(device)

        features = torch.randn(4, 64, 32, device=device)  # (batch, 64 codons, features)
        result = pool(features)

        assert result.shape == (4, 21, 32)  # (batch, 21 amino acids, features)

    def test_pooling_aggregates_correctly(self, device):
        """Test that pooling aggregates synonymous codons."""
        pool = SynonymousPooling(pool_type="mean")
        pool = pool.to(device)

        # Create features where synonymous codons have same features
        features = torch.zeros(1, 64, 16, device=device)

        # Set Leucine codons (L) to have value 1
        leu_indices = get_synonymous_groups()["L"]
        for idx in leu_indices:
            features[0, idx, :] = 1.0

        result = pool(features)
        leu_aa_idx = AMINO_ACIDS.index("L")

        # Leucine features should be 1
        assert torch.allclose(result[0, leu_aa_idx], torch.ones(16, device=device))


class TestWobbleAwareConv:
    """Tests for WobbleAwareConv."""

    def test_init(self):
        """Test initialization."""
        conv = WobbleAwareConv(in_channels=16, out_channels=32)
        assert conv.in_channels == 16
        assert conv.out_channels == 32

    def test_forward_shape(self, device):
        """Test forward pass shape."""
        conv = WobbleAwareConv(in_channels=16, out_channels=32)
        conv = conv.to(device)

        # (batch, seq_len, 3 positions, channels)
        x = torch.randn(4, 50, 3, 16, device=device)
        result = conv(x)

        assert result.shape == (4, 50, 32)


class TestCodonSymmetryLayer:
    """Tests for CodonSymmetryLayer."""

    def test_init_default(self):
        """Test default initialization."""
        layer = CodonSymmetryLayer(hidden_dim=64)
        assert layer.hidden_dim == 64
        assert layer.respect_wobble is True
        assert layer.respect_synonymy is True

    def test_forward_shape(self, device, codon_sequence):
        """Test forward pass shape."""
        layer = CodonSymmetryLayer(hidden_dim=64)
        layer = layer.to(device)

        result = layer(codon_sequence)
        assert result.shape == (4, 50, 64)

    def test_get_amino_acid_features(self, device):
        """Test amino acid feature extraction."""
        layer = CodonSymmetryLayer(hidden_dim=32)
        layer = layer.to(device)

        # Create features for each codon
        codon_features = torch.randn(2, 64, 32, device=device)
        aa_features = layer.get_amino_acid_features(codon_features)

        assert aa_features.shape == (2, 21, 32)

    def test_wobble_similarity_loss(self, device):
        """Test wobble similarity loss computation."""
        layer = CodonSymmetryLayer(hidden_dim=32)
        layer = layer.to(device)

        codon_features = torch.randn(2, 64, 32, device=device)
        loss = layer.wobble_similarity(codon_features)

        # Loss should be a scalar
        assert loss.shape == ()
        # Loss should be finite
        assert torch.isfinite(loss)


class TestCodonAttention:
    """Tests for CodonAttention."""

    def test_init(self):
        """Test initialization."""
        attn = CodonAttention(hidden_dim=64, n_heads=4)
        assert attn.hidden_dim == 64
        assert attn.n_heads == 4

    def test_forward_without_codons(self, device):
        """Test forward without codon indices."""
        attn = CodonAttention(hidden_dim=64, n_heads=4)
        attn = attn.to(device)

        x = torch.randn(4, 50, 64, device=device)
        result = attn(x)

        assert result.shape == (4, 50, 64)

    def test_forward_with_codons(self, device, codon_sequence):
        """Test forward with codon indices."""
        attn = CodonAttention(hidden_dim=64, n_heads=4)
        attn = attn.to(device)

        x = torch.randn(4, 50, 64, device=device)
        result = attn(x, codon_sequence)

        assert result.shape == (4, 50, 64)

    def test_synonymy_bias_effect(self, device):
        """Test that synonymy bias affects attention."""
        attn_with = CodonAttention(hidden_dim=32, n_heads=2, synonymy_bias=2.0)
        attn_without = CodonAttention(hidden_dim=32, n_heads=2, synonymy_bias=0.0)

        attn_with = attn_with.to(device)
        attn_without = attn_without.to(device)

        # Use same weights
        attn_without.load_state_dict(attn_with.state_dict())

        x = torch.randn(2, 10, 32, device=device)
        codons = torch.randint(0, 64, (2, 10), device=device)

        out_with = attn_with(x, codons)
        out_without = attn_without(x, codons)

        # Results should be different due to synonymy bias
        assert not torch.allclose(out_with, out_without)


class TestCodonTransformer:
    """Tests for CodonTransformer."""

    def test_init(self):
        """Test initialization."""
        model = CodonTransformer(
            vocab_size=64,
            hidden_dim=128,
            n_layers=4,
        )
        assert len(model.layers) == 4

    def test_forward_shape(self, device, codon_sequence):
        """Test forward pass shape."""
        model = CodonTransformer(
            vocab_size=64,
            hidden_dim=64,
            n_layers=2,
        )
        model = model.to(device)

        result = model(codon_sequence)
        assert result.shape == (4, 50, 64)

    def test_forward_hidden(self, device, codon_sequence):
        """Test forward with hidden state return."""
        model = CodonTransformer(
            vocab_size=64,
            hidden_dim=64,
            n_layers=2,
        )
        model = model.to(device)

        hidden = model(codon_sequence, return_hidden=True)
        assert hidden.shape == (4, 50, 64)

    def test_gradient_flow(self, device, codon_sequence):
        """Test gradient flows through model."""
        model = CodonTransformer(
            vocab_size=64,
            hidden_dim=32,
            n_layers=2,
        )
        model = model.to(device)

        codons = codon_sequence.clone()
        output = model(codons)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_without_synonymy(self, device, codon_sequence):
        """Test transformer without synonymy awareness."""
        model = CodonTransformer(
            vocab_size=64,
            hidden_dim=64,
            n_layers=2,
            respect_synonymy=False,
        )
        model = model.to(device)

        result = model(codon_sequence)
        assert result.shape == (4, 50, 64)
