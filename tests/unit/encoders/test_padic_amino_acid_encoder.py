# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for p-adic amino acid encoders.

Tests cover:
- 5-adic Amino Acid Encoder
- 7-adic Secondary Structure Encoder
- Multi-Prime Amino Acid Encoder
- Mutation Type Embedding
- Distance computations
"""

import numpy as np
import pytest
import torch

from src.encoders.padic_amino_acid_encoder import (
    AA_TO_GROUP,
    AA_TO_INDEX,
    AminoAcidGroup,
    FiveAdicAminoAcidEncoder,
    MultiPrimeAminoAcidEncoder,
    MutationType,
    MutationTypeEmbedding,
    SevenAdicSecondaryStructureEncoder,
    compute_5adic_distance,
    compute_5adic_distance_matrix,
)


class TestAminoAcidGroup:
    """Tests for amino acid group classifications."""

    def test_group_values(self):
        """Test group enum values."""
        assert AminoAcidGroup.HYDROPHOBIC == 0
        assert AminoAcidGroup.POLAR == 1
        assert AminoAcidGroup.POSITIVE == 2
        assert AminoAcidGroup.NEGATIVE == 3
        assert AminoAcidGroup.SPECIAL == 4

    def test_hydrophobic_assignment(self):
        """Test hydrophobic amino acids are correctly grouped."""
        hydrophobic = ["A", "V", "L", "I", "M", "F", "W", "Y"]
        for aa in hydrophobic:
            assert AA_TO_GROUP[aa] == AminoAcidGroup.HYDROPHOBIC

    def test_polar_assignment(self):
        """Test polar amino acids are correctly grouped."""
        polar = ["S", "T", "N", "Q"]
        for aa in polar:
            assert AA_TO_GROUP[aa] == AminoAcidGroup.POLAR

    def test_positive_assignment(self):
        """Test positively charged amino acids."""
        positive = ["K", "R", "H"]
        for aa in positive:
            assert AA_TO_GROUP[aa] == AminoAcidGroup.POSITIVE

    def test_negative_assignment(self):
        """Test negatively charged amino acids."""
        negative = ["D", "E"]
        for aa in negative:
            assert AA_TO_GROUP[aa] == AminoAcidGroup.NEGATIVE

    def test_special_assignment(self):
        """Test special amino acids."""
        special = ["G", "P", "C", "*"]
        for aa in special:
            assert AA_TO_GROUP[aa] == AminoAcidGroup.SPECIAL


class TestFiveAdicDistance:
    """Tests for 5-adic distance computation."""

    def test_same_aa_zero_distance(self):
        """Test same amino acid has zero distance."""
        for aa in AA_TO_INDEX.keys():
            dist = compute_5adic_distance(aa, aa)
            assert dist == 0.0

    def test_same_group_small_distance(self):
        """Test same group has small distance."""
        # Hydrophobic pairs
        dist_lv = compute_5adic_distance("L", "V")
        assert 0 < dist_lv < 0.2

        # Polar pairs
        dist_st = compute_5adic_distance("S", "T")
        assert 0 < dist_st < 0.2

    def test_different_group_large_distance(self):
        """Test different groups have distance 1."""
        # Hydrophobic vs polar
        dist = compute_5adic_distance("L", "S")
        assert dist == 1.0

        # Positive vs negative
        dist = compute_5adic_distance("K", "D")
        assert dist == 1.0

    def test_symmetric(self):
        """Test distance is symmetric."""
        for aa1 in ["A", "K", "S", "D", "G"]:
            for aa2 in ["V", "R", "T", "E", "P"]:
                assert compute_5adic_distance(aa1, aa2) == compute_5adic_distance(aa2, aa1)


class TestFiveAdicDistanceMatrix:
    """Tests for 5-adic distance matrix."""

    def test_matrix_shape(self):
        """Test matrix has correct shape."""
        matrix = compute_5adic_distance_matrix()
        assert matrix.shape == (22, 22)

    def test_diagonal_zero(self):
        """Test diagonal is zero."""
        matrix = compute_5adic_distance_matrix()
        assert np.allclose(np.diag(matrix), 0.0)

    def test_symmetric(self):
        """Test matrix is symmetric."""
        matrix = compute_5adic_distance_matrix()
        assert np.allclose(matrix, matrix.T)

    def test_values_in_range(self):
        """Test all values are in [0, 1]."""
        matrix = compute_5adic_distance_matrix()
        assert np.all(matrix >= 0)
        assert np.all(matrix <= 1)


class TestFiveAdicAminoAcidEncoder:
    """Tests for FiveAdicAminoAcidEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create encoder fixture."""
        return FiveAdicAminoAcidEncoder(embedding_dim=64, use_mds_init=False)

    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.embedding_dim == 64
        assert encoder.n_amino_acids == 22

    def test_forward_single(self, encoder):
        """Test forward pass with single index."""
        idx = torch.tensor([0])  # Alanine
        embed = encoder(idx)

        assert embed.shape == (1, 64)

    def test_forward_batch(self, encoder):
        """Test forward pass with batch."""
        indices = torch.randint(0, 22, (32,))
        embed = encoder(indices)

        assert embed.shape == (32, 64)

    def test_forward_sequence(self, encoder):
        """Test forward pass with sequence."""
        indices = torch.randint(0, 22, (8, 100))  # Batch of 8 sequences
        embed = encoder(indices)

        assert embed.shape == (8, 100, 64)

    def test_return_components(self, encoder):
        """Test returning intermediate components."""
        indices = torch.tensor([0, 1, 2])
        embed, components = encoder(indices, return_components=True)

        assert "group_embed" in components
        assert "aa_embed" in components
        assert "groups" in components

    def test_group_embedding(self, encoder):
        """Test group embeddings are learned."""
        # Same group should have similar group embeddings
        ala_idx = torch.tensor([AA_TO_INDEX["A"]])  # Hydrophobic
        val_idx = torch.tensor([AA_TO_INDEX["V"]])  # Hydrophobic

        _, ala_comp = encoder(ala_idx, return_components=True)
        _, val_comp = encoder(val_idx, return_components=True)

        assert torch.allclose(ala_comp["groups"], val_comp["groups"])

    def test_get_distance_matrix(self, encoder):
        """Test getting embedding distance matrix."""
        dist_matrix = encoder.get_distance_matrix()

        assert dist_matrix.shape == (22, 22)
        assert torch.allclose(torch.diag(dist_matrix), torch.zeros(22), atol=1e-5)

    def test_get_5adic_distance_matrix(self, encoder):
        """Test getting theoretical distance matrix."""
        matrix = encoder.get_5adic_distance_matrix()

        assert matrix.shape == (22, 22)
        assert isinstance(matrix, np.ndarray)

    def test_with_mds_init(self):
        """Test encoder with MDS initialization."""
        try:
            encoder = FiveAdicAminoAcidEncoder(embedding_dim=64, use_mds_init=True)
            idx = torch.tensor([0, 1, 2])
            embed = encoder(idx)
            assert embed.shape == (3, 64)
        except ImportError:
            pytest.skip("sklearn not available for MDS")


class TestSevenAdicSecondaryStructureEncoder:
    """Tests for SevenAdicSecondaryStructureEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create encoder fixture."""
        return SevenAdicSecondaryStructureEncoder(embedding_dim=32)

    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.embedding_dim == 32

    def test_forward(self, encoder):
        """Test forward pass."""
        # 0=H, 1=G, 2=I, 3=E, 4=B, 5=T, 6=C
        indices = torch.tensor([0, 3, 6])  # H, E, C
        embed = encoder(indices)

        assert embed.shape == (3, 32)

    def test_forward_batch(self, encoder):
        """Test batch forward pass."""
        indices = torch.randint(0, 7, (16, 50))
        embed = encoder(indices)

        assert embed.shape == (16, 50, 32)

    def test_helix_similarity(self, encoder):
        """Test helices are similar in embedding space."""
        helix_types = torch.tensor([0, 1, 2])  # H, G, I
        helix_embeds = encoder(helix_types)

        # Compute pairwise distances
        dists = torch.cdist(helix_embeds.unsqueeze(0), helix_embeds.unsqueeze(0)).squeeze(0)

        # All should be relatively small
        assert dists.max() < 5.0  # Arbitrary threshold


class TestMultiPrimeAminoAcidEncoder:
    """Tests for MultiPrimeAminoAcidEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create encoder fixture."""
        return MultiPrimeAminoAcidEncoder(embedding_dim=128, use_attention_fusion=False)

    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.embedding_dim == 128

    def test_forward_single(self, encoder):
        """Test single amino acid encoding."""
        idx = torch.tensor([5])
        embed = encoder(idx)

        assert embed.shape == (1, 128)

    def test_forward_batch(self, encoder):
        """Test batch encoding."""
        indices = torch.randint(0, 22, (32,))
        embed = encoder(indices)

        assert embed.shape == (32, 128)

    def test_forward_sequence(self, encoder):
        """Test sequence encoding."""
        indices = torch.randint(0, 22, (8, 50))
        embed = encoder(indices)

        assert embed.shape == (8, 50, 128)

    def test_with_attention_fusion(self):
        """Test with attention-based fusion."""
        encoder = MultiPrimeAminoAcidEncoder(embedding_dim=128, use_attention_fusion=True)
        indices = torch.randint(0, 22, (4, 10))
        embed = encoder(indices)

        assert embed.shape == (4, 10, 128)


class TestMutationType:
    """Tests for MutationType enum."""

    def test_mutation_types(self):
        """Test mutation type values."""
        assert MutationType.SYNONYMOUS == 0
        assert MutationType.CONSERVATIVE == 1
        assert MutationType.MODERATE == 2
        assert MutationType.RADICAL == 3
        assert MutationType.NONSENSE == 4
        assert MutationType.FRAMESHIFT == 5


class TestMutationTypeEmbedding:
    """Tests for MutationTypeEmbedding."""

    @pytest.fixture
    def encoder(self):
        """Create encoder fixture."""
        return MutationTypeEmbedding(embedding_dim=32)

    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.embedding_dim == 32

    def test_classify_synonymous(self, encoder):
        """Test synonymous mutation classification."""
        mut_type = encoder.classify_mutation("A", "A")
        assert mut_type == MutationType.SYNONYMOUS

    def test_classify_conservative(self, encoder):
        """Test conservative mutation classification."""
        # L to V (both hydrophobic)
        mut_type = encoder.classify_mutation("L", "V")
        assert mut_type == MutationType.CONSERVATIVE

    def test_classify_nonsense(self, encoder):
        """Test nonsense mutation classification."""
        mut_type = encoder.classify_mutation("A", "*")
        assert mut_type == MutationType.NONSENSE

    def test_classify_radical(self, encoder):
        """Test radical mutation classification."""
        # K (positive) to D (negative) - radical change
        mut_type = encoder.classify_mutation("K", "D")
        assert mut_type == MutationType.RADICAL

    def test_forward(self, encoder):
        """Test forward pass."""
        mut_types = torch.tensor([0, 1, 2, 3])
        embed = encoder(mut_types)

        assert embed.shape == (4, 32)

    def test_encode_mutation(self, encoder):
        """Test encoding single mutation."""
        embed = encoder.encode_mutation("A", "V", position=50, seq_length=100)

        assert embed.shape == (1, 32)

    def test_encode_mutation_without_position(self, encoder):
        """Test encoding without position info."""
        encoder_no_pos = MutationTypeEmbedding(embedding_dim=32, include_position=False)
        embed = encoder_no_pos.encode_mutation("A", "V")

        assert embed.shape == (1, 32)


class TestIntegration:
    """Integration tests for amino acid encoders."""

    def test_encoder_gradient_flow(self):
        """Test that gradients flow through encoder."""
        encoder = FiveAdicAminoAcidEncoder(embedding_dim=64, use_mds_init=False)

        indices = torch.randint(0, 22, (8,))
        embed = encoder(indices)
        loss = embed.sum()
        loss.backward()

        # Check gradients exist
        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_multi_prime_gradient_flow(self):
        """Test gradients flow through multi-prime encoder."""
        encoder = MultiPrimeAminoAcidEncoder(embedding_dim=64, use_attention_fusion=False)

        indices = torch.randint(0, 22, (8,))
        embed = encoder(indices)
        loss = embed.sum()
        loss.backward()

        for name, param in encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_mutation_embedding_gradient_flow(self):
        """Test gradients flow through mutation embedding."""
        encoder = MutationTypeEmbedding(embedding_dim=32)

        mut_types = torch.randint(0, 6, (8,))
        embed = encoder(mut_types)
        loss = embed.sum()
        loss.backward()

        # Check that type embedding layer gets gradients (main component)
        assert encoder.type_embedding.weight.grad is not None, "No gradient for type_embedding"
        # Fusion layer should also get gradients
        assert encoder.fusion.weight.grad is not None, "No gradient for fusion"

    def test_combined_encoding(self):
        """Test combining AA encoding with mutation embedding."""
        aa_encoder = FiveAdicAminoAcidEncoder(embedding_dim=64, use_mds_init=False)
        mut_encoder = MutationTypeEmbedding(embedding_dim=32)

        # Encode amino acids
        aa_indices = torch.randint(0, 22, (8,))
        aa_embed = aa_encoder(aa_indices)

        # Encode mutations
        mut_types = torch.randint(0, 6, (8,))
        mut_embed = mut_encoder(mut_types)

        # Combine
        combined = torch.cat([aa_embed, mut_embed], dim=-1)

        assert combined.shape == (8, 96)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
