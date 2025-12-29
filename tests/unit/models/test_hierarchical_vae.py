# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for Hierarchical VAE.

Tests cover:
- Model initialization
- Forward pass through all hierarchy levels
- Bottom-up encoding
- Top-down decoding with learned priors
- Hierarchical KL divergence computation
- Sampling from hierarchical prior
- Loss computation with per-level breakdown
- Parameter counting
"""

from __future__ import annotations

import pytest
import torch

from src.models.hierarchical_vae import (
    HierarchicalVAE,
    HierarchicalVAEConfig,
    LadderEncoderBlock,
    LadderDecoderBlock,
    TopDownPrior,
)


class TestHierarchicalVAEConfig:
    """Tests for HierarchicalVAEConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HierarchicalVAEConfig()

        assert config.n_levels == 3
        assert len(config.latent_dims) == 3
        assert len(config.hidden_dims_per_level) == 3
        assert config.free_bits == 0.25

    def test_custom_config(self):
        """Test custom configuration."""
        config = HierarchicalVAEConfig(
            input_dim=99,
            n_levels=4,
            latent_dims=[4, 8, 16, 32],
            hidden_dims_per_level=[128, 64, 32, 16],
        )

        assert config.n_levels == 4
        assert config.latent_dims == [4, 8, 16, 32]

    def test_config_inherits_from_vae_config(self):
        """Test that config inherits from VAEConfig."""
        config = HierarchicalVAEConfig(dropout=0.2)
        assert config.dropout == 0.2
        assert hasattr(config, "latent_dim")


class TestLadderEncoderBlock:
    """Tests for LadderEncoderBlock."""

    @pytest.fixture
    def block(self):
        """Create encoder block fixture."""
        return LadderEncoderBlock(
            input_dim=64,
            hidden_dim=128,
            latent_dim=16,
        )

    def test_forward_shape(self, block):
        """Test output shapes."""
        x = torch.randn(8, 64)
        mu, logvar, next_features = block(x)

        assert mu.shape == (8, 16)
        assert logvar.shape == (8, 16)
        assert next_features.shape == (8, 64)  # 128 // 2

    def test_output_types(self, block):
        """Test output types are tensors."""
        x = torch.randn(4, 64)
        mu, logvar, next_features = block(x)

        assert isinstance(mu, torch.Tensor)
        assert isinstance(logvar, torch.Tensor)
        assert isinstance(next_features, torch.Tensor)


class TestLadderDecoderBlock:
    """Tests for LadderDecoderBlock."""

    @pytest.fixture
    def block(self):
        """Create decoder block fixture."""
        return LadderDecoderBlock(
            latent_dim=16,
            hidden_dim=64,
            output_dim=128,
        )

    def test_forward_shape(self, block):
        """Test output shapes."""
        z = torch.randn(8, 16)
        top_down = torch.randn(8, 64)
        output = block(z, top_down)

        assert output.shape == (8, 128)

    def test_with_residual(self):
        """Test decoder with residual connection."""
        block = LadderDecoderBlock(
            latent_dim=16,
            hidden_dim=64,
            output_dim=64,  # Same as hidden for residual
            use_residual=True,
        )
        z = torch.randn(4, 16)
        top_down = torch.randn(4, 64)
        output = block(z, top_down)

        assert output.shape == (4, 64)


class TestTopDownPrior:
    """Tests for TopDownPrior."""

    @pytest.fixture
    def prior(self):
        """Create prior fixture."""
        return TopDownPrior(input_dim=64, latent_dim=16)

    def test_forward_shape(self, prior):
        """Test output shapes."""
        features = torch.randn(8, 64)
        mu, logvar = prior(features)

        assert mu.shape == (8, 16)
        assert logvar.shape == (8, 16)


class TestHierarchicalVAE:
    """Tests for HierarchicalVAE class."""

    @pytest.fixture
    def config(self):
        """Create config fixture."""
        return HierarchicalVAEConfig(
            input_dim=99,
            n_levels=3,
            latent_dims=[8, 16, 32],
            hidden_dims_per_level=[256, 128, 64],
            output_classes=3,
        )

    @pytest.fixture
    def model(self, config):
        """Create model fixture."""
        return HierarchicalVAE(config)

    def test_initialization(self, model, config):
        """Test model initialization."""
        assert model.n_levels == 3
        assert len(model.encoder_blocks) == 3
        assert len(model.decoder_blocks) == 3
        assert len(model.top_down_priors) == 2  # n_levels - 1

    def test_forward_pass(self, model):
        """Test forward pass produces correct outputs."""
        x = torch.randn(8, 99)
        outputs = model(x)

        assert "logits" in outputs
        assert "mu" in outputs
        assert "logvar" in outputs
        assert "z" in outputs
        assert "all_mus" in outputs
        assert "all_logvars" in outputs
        assert "all_zs" in outputs
        assert "prior_mus" in outputs
        assert "prior_logvars" in outputs

    def test_output_shapes(self, model):
        """Test output tensor shapes."""
        x = torch.randn(8, 99)
        outputs = model(x)

        # Reconstruction
        assert outputs["logits"].shape == (8, 99, 3)

        # Top level latent
        assert outputs["mu"].shape == (8, 32)
        assert outputs["logvar"].shape == (8, 32)
        assert outputs["z"].shape == (8, 32)

        # All levels
        assert len(outputs["all_mus"]) == 3
        assert len(outputs["all_zs"]) == 3

    def test_encode(self, model):
        """Test encode returns top-level parameters."""
        x = torch.randn(8, 99)
        mu, logvar = model.encode(x)

        assert mu.shape == (8, 32)
        assert logvar.shape == (8, 32)

    def test_encode_all(self, model):
        """Test encode_all returns all levels."""
        x = torch.randn(8, 99)
        all_mus, all_logvars, all_features = model.encode_all(x)

        assert len(all_mus) == 3
        assert all_mus[0].shape == (8, 8)   # Level 0
        assert all_mus[1].shape == (8, 16)  # Level 1
        assert all_mus[2].shape == (8, 32)  # Level 2 (top)

    def test_decode(self, model):
        """Test decode from top-level latent."""
        z = torch.randn(8, 32)
        logits = model.decode(z)

        assert logits.shape == (8, 99, 3)

    def test_decode_all(self, model):
        """Test decode from all levels."""
        zs = [
            torch.randn(8, 8),   # Level 0
            torch.randn(8, 16),  # Level 1
            torch.randn(8, 32),  # Level 2
        ]
        logits = model.decode_all(zs)

        assert logits.shape == (8, 99, 3)


class TestHierarchicalKL:
    """Tests for hierarchical KL divergence."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        config = HierarchicalVAEConfig(
            input_dim=99,
            n_levels=3,
            latent_dims=[8, 16, 32],
            hidden_dims_per_level=[256, 128, 64],
        )
        return HierarchicalVAE(config)

    def test_kl_divergence_shape(self, model):
        """Test KL divergence returns scalar and list."""
        x = torch.randn(8, 99)
        outputs = model(x)

        total_kl, kl_per_level = model.kl_divergence_hierarchical(
            outputs["all_mus"],
            outputs["all_logvars"],
            outputs["prior_mus"],
            outputs["prior_logvars"],
        )

        assert total_kl.dim() == 0  # Scalar
        assert len(kl_per_level) == 3

    def test_kl_non_negative(self, model):
        """Test KL is non-negative at all levels."""
        x = torch.randn(8, 99)
        outputs = model(x)

        total_kl, kl_per_level = model.kl_divergence_hierarchical(
            outputs["all_mus"],
            outputs["all_logvars"],
            outputs["prior_mus"],
            outputs["prior_logvars"],
        )

        assert total_kl >= 0
        for kl in kl_per_level:
            assert kl >= 0

    def test_free_bits(self, model):
        """Test free bits prevent collapse."""
        x = torch.randn(8, 99)
        outputs = model(x)

        # With free bits = 0.25, minimum KL per dim
        _, kl_per_level = model.kl_divergence_hierarchical(
            outputs["all_mus"],
            outputs["all_logvars"],
            outputs["prior_mus"],
            outputs["prior_logvars"],
            free_bits=0.25,
        )

        # Each level should have some minimum KL
        for kl in kl_per_level:
            assert kl > 0


class TestHierarchicalLoss:
    """Tests for hierarchical loss computation."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        config = HierarchicalVAEConfig(
            input_dim=99,
            n_levels=3,
            latent_dims=[8, 16, 32],
            hidden_dims_per_level=[256, 128, 64],
        )
        return HierarchicalVAE(config)

    def test_compute_loss(self, model):
        """Test loss computation."""
        x = torch.randn(8, 99)
        losses = model.compute_loss(x)

        assert "total" in losses
        assert "recon" in losses
        assert "kl" in losses
        assert "kl_level_0" in losses
        assert "kl_level_1" in losses
        assert "kl_level_2" in losses

    def test_loss_values(self, model):
        """Test loss values are valid."""
        x = torch.randn(8, 99)
        losses = model.compute_loss(x)

        assert not torch.isnan(losses["total"])
        assert not torch.isinf(losses["total"])
        assert losses["recon"] >= 0
        assert losses["kl"] >= 0

    def test_loss_with_beta(self, model):
        """Test loss with custom beta."""
        x = torch.randn(8, 99)

        losses_b1 = model.compute_loss(x, beta=1.0)
        losses_b0 = model.compute_loss(x, beta=0.0)

        # With beta=0, total should equal recon
        assert torch.allclose(losses_b0["total"], losses_b0["recon"], atol=1e-5)


class TestHierarchicalSampling:
    """Tests for hierarchical sampling."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        config = HierarchicalVAEConfig(
            input_dim=99,
            n_levels=3,
            latent_dims=[8, 16, 32],
            hidden_dims_per_level=[256, 128, 64],
            output_classes=3,
        )
        return HierarchicalVAE(config)

    def test_sample_shape(self, model):
        """Test sample produces correct shape."""
        samples = model.sample(n_samples=16)
        assert samples.shape == (16, 99, 3)

    def test_sample_with_temperature(self, model):
        """Test sampling with temperature."""
        samples_t1 = model.sample(n_samples=8, temperature=1.0)
        samples_t01 = model.sample(n_samples=8, temperature=0.1)

        # Both should have same shape
        assert samples_t1.shape == samples_t01.shape

    def test_sample_from_prior(self, model):
        """Test sampling from learned prior."""
        features = torch.randn(8, 128)
        z, prior_mu, prior_logvar = model.sample_from_prior(features, level=0)

        assert z.shape == (8, 8)  # Level 0 latent dim
        assert prior_mu.shape == (8, 8)


class TestHierarchicalReconstruction:
    """Tests for hierarchical reconstruction."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        config = HierarchicalVAEConfig(
            input_dim=99,
            n_levels=3,
            latent_dims=[8, 16, 32],
            hidden_dims_per_level=[256, 128, 64],
            output_classes=3,
        )
        return HierarchicalVAE(config)

    def test_reconstruct(self, model):
        """Test reconstruction."""
        x = torch.randn(8, 99)
        recon = model.reconstruct(x)

        assert recon.shape == (8, 99)
        # Values should be in {-1, 0, 1}
        assert torch.all((recon >= -1) & (recon <= 1))


class TestLevelEmbeddings:
    """Tests for level-specific embeddings."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        config = HierarchicalVAEConfig(
            input_dim=99,
            n_levels=3,
            latent_dims=[8, 16, 32],
            hidden_dims_per_level=[256, 128, 64],
        )
        return HierarchicalVAE(config)

    def test_get_level_embeddings_single(self, model):
        """Test getting embeddings from single level."""
        x = torch.randn(8, 99)

        level_0_emb = model.get_level_embeddings(x, level=0)
        level_2_emb = model.get_level_embeddings(x, level=2)

        assert level_0_emb.shape == (8, 8)
        assert level_2_emb.shape == (8, 32)

    def test_get_level_embeddings_all(self, model):
        """Test getting embeddings from all levels."""
        x = torch.randn(8, 99)
        all_embeddings = model.get_level_embeddings(x, level=None)

        assert len(all_embeddings) == 3
        assert all_embeddings[0].shape == (8, 8)
        assert all_embeddings[1].shape == (8, 16)
        assert all_embeddings[2].shape == (8, 32)


class TestParameterCounting:
    """Tests for parameter counting."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        config = HierarchicalVAEConfig(
            input_dim=99,
            n_levels=3,
            latent_dims=[8, 16, 32],
            hidden_dims_per_level=[256, 128, 64],
        )
        return HierarchicalVAE(config)

    def test_count_parameters(self, model):
        """Test parameter counting."""
        counts = model.count_parameters()

        assert "total" in counts
        assert "trainable" in counts
        assert counts["total"] > 0
        assert counts["trainable"] == counts["total"]  # All trainable by default

    def test_count_parameters_per_level(self, model):
        """Test per-level parameter counting."""
        counts = model.count_parameters()

        assert "encoder_level_0" in counts
        assert "encoder_level_1" in counts
        assert "encoder_level_2" in counts
        assert "decoder_level_0" in counts


class TestGradientFlow:
    """Tests for gradient flow through hierarchy."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        config = HierarchicalVAEConfig(
            input_dim=99,
            n_levels=3,
            latent_dims=[8, 16, 32],
            hidden_dims_per_level=[256, 128, 64],
        )
        return HierarchicalVAE(config)

    def test_gradient_flow(self, model):
        """Test gradients flow through all levels."""
        x = torch.randn(8, 99)
        losses = model.compute_loss(x)
        losses["total"].backward()

        # Check gradients exist for encoder blocks (except to_next on top level)
        for i, block in enumerate(model.encoder_blocks):
            for name, param in block.named_parameters():
                # Top level's to_next output isn't used, so no gradient expected
                if i == len(model.encoder_blocks) - 1 and "to_next" in name:
                    continue
                assert param.grad is not None, f"No gradient for encoder block {i} {name}"

    def test_no_nan_gradients(self, model):
        """Test no NaN gradients."""
        x = torch.randn(8, 99)
        losses = model.compute_loss(x)
        losses["total"].backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestTrainingMode:
    """Tests for training vs eval mode."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        config = HierarchicalVAEConfig(
            input_dim=99,
            n_levels=3,
            latent_dims=[8, 16, 32],
            hidden_dims_per_level=[256, 128, 64],
        )
        return HierarchicalVAE(config)

    def test_train_mode_stochastic(self, model):
        """Test training mode is stochastic."""
        model.train()
        x = torch.randn(8, 99)

        outputs1 = model(x)
        outputs2 = model(x)

        # Z should be different (stochastic)
        assert not torch.allclose(outputs1["z"], outputs2["z"])

    def test_eval_mode_deterministic(self, model):
        """Test eval mode is deterministic."""
        model.eval()
        x = torch.randn(8, 99)

        with torch.no_grad():
            outputs1 = model(x)
            outputs2 = model(x)

        # Z should be same (deterministic = mu)
        assert torch.allclose(outputs1["z"], outputs2["z"])


class TestIntegration:
    """Integration tests for Hierarchical VAE."""

    def test_training_step(self):
        """Test a single training step."""
        config = HierarchicalVAEConfig(
            input_dim=99,
            n_levels=3,
            latent_dims=[8, 16, 32],
            hidden_dims_per_level=[256, 128, 64],
        )
        model = HierarchicalVAE(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training step
        x = torch.randn(16, 99)
        losses = model.compute_loss(x)
        losses["total"].backward()
        optimizer.step()
        optimizer.zero_grad()

        # Should complete without error

    def test_batch_sizes(self):
        """Test various batch sizes."""
        config = HierarchicalVAEConfig(
            input_dim=99,
            n_levels=3,
            latent_dims=[8, 16, 32],
            hidden_dims_per_level=[256, 128, 64],
        )
        model = HierarchicalVAE(config)

        for batch_size in [1, 4, 16, 64]:
            x = torch.randn(batch_size, 99)
            outputs = model(x)
            assert outputs["logits"].shape[0] == batch_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
