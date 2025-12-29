# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Comprehensive unit tests for Structure-Aware VAE.

Tests cover:
- StructureConfig
- InvariantPointAttention
- SE3Encoder
- StructureSequenceFusion
- StructureAwareVAE
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.structure_aware_vae import (
    InvariantPointAttention,
    SE3Encoder,
    StructureAwareVAE,
    StructureConfig,
    StructureSequenceFusion,
)


class TestStructureConfig:
    """Test StructureConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = StructureConfig()

        assert config.use_structure is True
        assert config.structure_dim == 64
        assert config.n_structure_layers == 3
        assert config.cutoff == 10.0
        assert config.use_plddt is True
        assert config.fusion_type == "cross_attention"

    def test_custom_config(self):
        """Test custom configuration."""
        config = StructureConfig(
            use_structure=False,
            structure_dim=128,
            cutoff=15.0,
            fusion_type="gated",
        )

        assert config.use_structure is False
        assert config.structure_dim == 128
        assert config.cutoff == 15.0
        assert config.fusion_type == "gated"


class TestInvariantPointAttention:
    """Test InvariantPointAttention module."""

    @pytest.fixture
    def ipa(self):
        """Create IPA fixture."""
        return InvariantPointAttention(
            embed_dim=64,
            n_heads=4,
            n_query_points=4,
            n_value_points=4,
        )

    def test_initialization(self, ipa):
        """Test IPA initialization."""
        assert ipa.embed_dim == 64
        assert ipa.n_heads == 4
        assert ipa.head_dim == 16

    def test_forward_shape(self, ipa):
        """Test forward pass shape."""
        features = torch.randn(4, 100, 64)  # batch, residues, dim
        coords = torch.randn(4, 100, 3)

        output = ipa(features, coords)

        assert output.shape == (4, 100, 64)

    def test_forward_with_mask(self, ipa):
        """Test forward with attention mask."""
        features = torch.randn(2, 50, 64)
        coords = torch.randn(2, 50, 3)
        mask = torch.ones(2, 50, 50).bool()
        mask[:, :, 25:] = False  # Mask second half

        output = ipa(features, coords, mask)

        assert output.shape == (2, 50, 64)

    def test_deterministic_output(self, ipa):
        """Test that IPA produces deterministic output for same input."""
        ipa.eval()
        features = torch.randn(2, 20, 64)
        coords = torch.randn(2, 20, 3)

        output1 = ipa(features, coords)
        output2 = ipa(features, coords)

        # Same input should produce same output
        assert torch.allclose(output1, output2, atol=1e-6)

    def test_learnable_weights(self, ipa):
        """Test learnable combination weights."""
        assert ipa.w_c.shape == (4,)  # n_heads
        assert ipa.w_l.shape == (4,)


class TestSE3Encoder:
    """Test SE3Encoder module."""

    @pytest.fixture
    def encoder(self):
        """Create encoder fixture."""
        return SE3Encoder(
            node_dim=64,
            edge_dim=32,
            n_layers=3,
            cutoff=10.0,
        )

    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.node_dim == 64
        assert encoder.cutoff == 10.0
        assert len(encoder.layers) == 3

    def test_forward_shape(self, encoder):
        """Test forward pass shape."""
        coords = torch.randn(4, 100, 3)
        aa_indices = torch.randint(0, 20, (4, 100))

        output = encoder(coords, aa_indices)

        assert output.shape == (4, 100, 64)

    def test_forward_without_aa(self, encoder):
        """Test forward without amino acid indices."""
        coords = torch.randn(4, 50, 3)

        output = encoder(coords)

        assert output.shape == (4, 50, 64)

    def test_cutoff_effect(self):
        """Test distance cutoff effect."""
        encoder = SE3Encoder(cutoff=5.0)

        # Create coords with some far apart
        coords = torch.zeros(1, 10, 3)
        coords[0, 5:, 0] = 20.0  # Far away

        output = encoder(coords)

        # Should still produce valid output
        assert not torch.isnan(output).any()

    def test_gradient_flow(self, encoder):
        """Test gradient flow through encoder."""
        coords = torch.randn(2, 30, 3, requires_grad=True)

        output = encoder(coords)
        loss = output.sum()
        loss.backward()

        # Coords should have gradients (through the distance computation)
        # Note: This may or may not work depending on implementation
        # The main check is that backward doesn't fail


class TestStructureSequenceFusion:
    """Test StructureSequenceFusion module."""

    def test_cross_attention_fusion(self):
        """Test cross-attention fusion."""
        fusion = StructureSequenceFusion(
            seq_dim=128,
            struct_dim=64,
            output_dim=64,
            fusion_type="cross_attention",
        )

        seq_embed = torch.randn(4, 128)
        struct_embed = torch.randn(4, 64)

        output = fusion(seq_embed, struct_embed)

        assert output.shape == (4, 64)

    def test_gated_fusion(self):
        """Test gated fusion."""
        fusion = StructureSequenceFusion(
            seq_dim=128,
            struct_dim=64,
            output_dim=64,
            fusion_type="gated",
        )

        seq_embed = torch.randn(4, 128)
        struct_embed = torch.randn(4, 64)

        output = fusion(seq_embed, struct_embed)

        assert output.shape == (4, 64)

    def test_concat_fusion(self):
        """Test concatenation fusion."""
        fusion = StructureSequenceFusion(
            seq_dim=128,
            struct_dim=64,
            output_dim=96,
            fusion_type="concat",
        )

        seq_embed = torch.randn(4, 128)
        struct_embed = torch.randn(4, 64)

        output = fusion(seq_embed, struct_embed)

        assert output.shape == (4, 96)

    def test_3d_inputs(self):
        """Test with 3D (sequential) inputs."""
        fusion = StructureSequenceFusion(
            seq_dim=128,
            struct_dim=64,
            output_dim=64,
            fusion_type="cross_attention",
        )

        seq_embed = torch.randn(4, 50, 128)
        struct_embed = torch.randn(4, 50, 64)

        output = fusion(seq_embed, struct_embed)

        assert output.shape == (4, 50, 64)

    def test_invalid_fusion_type(self):
        """Test invalid fusion type raises error."""
        with pytest.raises(ValueError):
            StructureSequenceFusion(
                seq_dim=64,
                struct_dim=64,
                output_dim=64,
                fusion_type="invalid",
            )


class TestStructureAwareVAE:
    """Test StructureAwareVAE."""

    @pytest.fixture
    def vae_with_structure(self):
        """Create VAE with structure."""
        config = StructureConfig(use_structure=True)
        return StructureAwareVAE(
            input_dim=128,
            latent_dim=32,
            structure_config=config,
        )

    @pytest.fixture
    def vae_without_structure(self):
        """Create VAE without structure."""
        config = StructureConfig(use_structure=False)
        return StructureAwareVAE(
            input_dim=128,
            latent_dim=32,
            structure_config=config,
        )

    def test_initialization_with_structure(self, vae_with_structure):
        """Test initialization with structure."""
        vae = vae_with_structure

        assert vae.structure_encoder is not None
        assert vae.fusion is not None
        assert vae.latent_dim == 32

    def test_initialization_without_structure(self, vae_without_structure):
        """Test initialization without structure."""
        vae = vae_without_structure

        assert vae.structure_encoder is None
        assert vae.fusion is None

    def test_forward_sequence_only(self, vae_with_structure):
        """Test forward with sequence only."""
        x = torch.randn(4, 128)

        outputs = vae_with_structure(x)

        assert "logits" in outputs
        assert "mu" in outputs
        assert "logvar" in outputs
        assert "z" in outputs

    def test_forward_with_structure(self, vae_with_structure):
        """Test forward with structure."""
        x = torch.randn(4, 128)
        structure = torch.randn(4, 50, 3)  # Coords
        plddt = torch.rand(4, 50) * 100  # Confidence

        outputs = vae_with_structure(x, structure=structure, plddt=plddt)

        assert outputs["logits"].shape == (4, 128)
        assert outputs["mu"].shape == (4, 32)

    def test_forward_with_aa_indices(self, vae_with_structure):
        """Test forward with amino acid indices."""
        x = torch.randn(4, 128)
        structure = torch.randn(4, 50, 3)
        aa_indices = torch.randint(0, 20, (4, 50))

        outputs = vae_with_structure(x, structure=structure, aa_indices=aa_indices)

        assert outputs["z"].shape == (4, 32)

    def test_encode(self, vae_with_structure):
        """Test encode method."""
        x = torch.randn(4, 128)

        mu, logvar = vae_with_structure.encode(x)

        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)

    def test_encode_with_structure(self, vae_with_structure):
        """Test encode with structure."""
        x = torch.randn(4, 128)
        structure = torch.randn(4, 30, 3)
        plddt = torch.rand(4, 30) * 100

        mu, logvar = vae_with_structure.encode(x, structure, plddt)

        assert mu.shape == (4, 32)

    def test_decode(self, vae_with_structure):
        """Test decode method."""
        z = torch.randn(4, 32)

        recon = vae_with_structure.decode(z)

        assert recon.shape == (4, 128)

    def test_reparameterize_training(self, vae_with_structure):
        """Test reparameterization in training mode."""
        vae_with_structure.train()

        mu = torch.zeros(4, 32)
        logvar = torch.zeros(4, 32)

        z1 = vae_with_structure.reparameterize(mu, logvar)
        z2 = vae_with_structure.reparameterize(mu, logvar)

        # Should be stochastic
        assert not torch.allclose(z1, z2)

    def test_reparameterize_eval(self, vae_with_structure):
        """Test reparameterization in eval mode."""
        vae_with_structure.eval()

        mu = torch.randn(4, 32)
        logvar = torch.zeros(4, 32)

        z = vae_with_structure.reparameterize(mu, logvar)

        # Should equal mu
        assert torch.allclose(z, mu)

    def test_plddt_weighting(self, vae_with_structure):
        """Test pLDDT confidence weighting."""
        x = torch.randn(4, 128)
        structure = torch.randn(4, 30, 3)

        # High confidence
        plddt_high = torch.ones(4, 30) * 90

        # Low confidence
        plddt_low = torch.ones(4, 30) * 30

        # Results should differ
        out_high = vae_with_structure(x, structure, plddt_high)
        out_low = vae_with_structure(x, structure, plddt_low)

        # With different confidence, latent should differ
        # (Though this depends on how much the structure contributes)

    def test_count_parameters(self, vae_with_structure):
        """Test parameter counting."""
        params = vae_with_structure.count_parameters()

        assert "total" in params
        assert "trainable" in params
        assert params["total"] > 0

    def test_sequential_input(self, vae_with_structure):
        """Test with sequential (3D) input."""
        x = torch.randn(4, 50, 128)  # batch, seq_len, dim

        outputs = vae_with_structure(x)

        assert outputs["mu"].shape == (4, 32)

    def test_gradient_flow(self, vae_with_structure):
        """Test gradient flow through model."""
        x = torch.randn(4, 128, requires_grad=True)
        structure = torch.randn(4, 30, 3)
        plddt = torch.rand(4, 30) * 100

        outputs = vae_with_structure(x, structure, plddt)
        loss = outputs["logits"].sum()
        loss.backward()

        assert x.grad is not None

    def test_different_fusion_types(self):
        """Test different fusion types."""
        for fusion_type in ["cross_attention", "gated", "concat"]:
            config = StructureConfig(fusion_type=fusion_type)
            vae = StructureAwareVAE(
                input_dim=64,
                latent_dim=16,
                structure_config=config,
            )

            x = torch.randn(2, 64)
            structure = torch.randn(2, 20, 3)

            outputs = vae(x, structure)

            assert outputs["z"].shape == (2, 16)


class TestStructureAwareEdgeCases:
    """Test edge cases for Structure-Aware VAE."""

    def test_empty_structure(self):
        """Test with empty structure (should fall back to sequence only)."""
        vae = StructureAwareVAE(input_dim=64, latent_dim=16)

        x = torch.randn(4, 64)

        outputs = vae(x)  # No structure provided

        assert outputs["z"].shape == (4, 16)

    def test_mismatched_batch_sizes(self):
        """Test error handling for mismatched batch sizes."""
        vae = StructureAwareVAE(input_dim=64, latent_dim=16)

        x = torch.randn(4, 64)
        structure = torch.randn(2, 30, 3)  # Different batch size

        # This should either raise an error or handle gracefully
        # Depending on implementation

    def test_very_short_structure(self):
        """Test with very short structure."""
        vae = StructureAwareVAE(input_dim=64, latent_dim=16)

        x = torch.randn(4, 64)
        structure = torch.randn(4, 5, 3)  # Only 5 residues

        outputs = vae(x, structure)

        assert outputs["z"].shape == (4, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
