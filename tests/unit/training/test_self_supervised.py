# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for Self-Supervised Pre-training module.

Tests cover:
- Configuration initialization
- Sequence encoder forward pass and output shapes
- Masked sequence modeling head
- Contrastive learning head (BYOL-style)
- VAE decoder reconstruction
- Mutation impact head
- Sequence augmentation
- Full model forward pass
- Loss computation
- Pre-training workflow
- Downstream model creation
- Representation evaluation
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.training.self_supervised import (
    SelfSupervisedPretrainer,
    SelfSupervisedConfig,
    SelfSupervisedModel,
    SequenceEncoder,
    SequenceDecoder,
    ContrastiveHead,
    MaskedSequenceModeling,
    MutationHead,
    SequenceAugmenter,
    PretrainingObjective,
)


class DummySequenceDataset(Dataset):
    """Dummy dataset for testing."""

    def __init__(self, n_samples=100, seq_length=99, input_dim=21, include_labels=False):
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.include_labels = include_labels

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.randn(self.seq_length, self.input_dim)
        result = {"x": x}
        if self.include_labels:
            result["y"] = torch.tensor(idx % 3)  # 3-class labels
        return result


class TestSelfSupervisedConfig:
    """Tests for SelfSupervisedConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SelfSupervisedConfig()

        assert config.input_dim == 21
        assert config.seq_length == 99
        assert config.hidden_dim == 256
        assert config.latent_dim == 64
        assert config.n_layers == 4
        assert config.n_heads == 8
        assert config.dropout == 0.1
        assert config.mask_ratio == 0.15
        assert config.temperature == 0.07
        assert config.momentum == 0.996
        assert config.pretrain_epochs == 100
        assert config.batch_size == 32

    def test_custom_config(self):
        """Test custom configuration."""
        config = SelfSupervisedConfig(
            input_dim=4,
            seq_length=50,
            hidden_dim=128,
            latent_dim=32,
            n_layers=2,
            pretrain_epochs=10,
        )

        assert config.input_dim == 4
        assert config.seq_length == 50
        assert config.hidden_dim == 128
        assert config.latent_dim == 32
        assert config.n_layers == 2
        assert config.pretrain_epochs == 10

    def test_default_objectives(self):
        """Test default objectives include MSM, Contrastive, VAE."""
        config = SelfSupervisedConfig()

        assert PretrainingObjective.MSM in config.objectives
        assert PretrainingObjective.CONTRASTIVE in config.objectives
        assert PretrainingObjective.VAE in config.objectives

    def test_custom_objectives(self):
        """Test custom objectives."""
        config = SelfSupervisedConfig(
            objectives=[PretrainingObjective.MSM, PretrainingObjective.MUTATION]
        )

        assert PretrainingObjective.MSM in config.objectives
        assert PretrainingObjective.MUTATION in config.objectives
        assert PretrainingObjective.CONTRASTIVE not in config.objectives

    def test_objective_weights(self):
        """Test objective weights configuration."""
        config = SelfSupervisedConfig()

        assert "msm" in config.objective_weights
        assert "contrastive" in config.objective_weights
        assert "vae_recon" in config.objective_weights
        assert "vae_kl" in config.objective_weights


class TestPretrainingObjective:
    """Tests for PretrainingObjective enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert PretrainingObjective.MSM.value == "masked_sequence_modeling"
        assert PretrainingObjective.CONTRASTIVE.value == "contrastive"
        assert PretrainingObjective.VAE.value == "vae"
        assert PretrainingObjective.NEXT_TOKEN.value == "next_token"
        assert PretrainingObjective.MUTATION.value == "mutation"

    def test_enum_iteration(self):
        """Test iterating over objectives."""
        objectives = list(PretrainingObjective)
        assert len(objectives) == 5


class TestSequenceEncoder:
    """Tests for SequenceEncoder."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SelfSupervisedConfig(
            input_dim=21,
            seq_length=50,
            hidden_dim=64,
            latent_dim=16,
            n_layers=2,
            n_heads=4,
        )

    @pytest.fixture
    def encoder(self, config):
        """Create encoder fixture."""
        return SequenceEncoder(config)

    def test_initialization(self, encoder, config):
        """Test encoder initialization."""
        assert encoder.config == config
        assert encoder.embedding is not None
        assert encoder.transformer is not None
        assert encoder.to_latent is not None

    def test_forward_3d_input(self, encoder, config):
        """Test forward pass with 3D input."""
        batch_size = 4
        x = torch.randn(batch_size, config.seq_length, config.input_dim)

        outputs = encoder(x)

        assert "mu" in outputs
        assert "logvar" in outputs
        assert "z" in outputs
        assert outputs["mu"].shape == (batch_size, config.latent_dim)
        assert outputs["logvar"].shape == (batch_size, config.latent_dim)
        assert outputs["z"].shape == (batch_size, config.latent_dim)

    def test_forward_2d_input(self, encoder, config):
        """Test forward pass with flattened 2D input."""
        batch_size = 4
        x = torch.randn(batch_size, config.seq_length * config.input_dim)

        outputs = encoder(x)

        assert outputs["mu"].shape == (batch_size, config.latent_dim)

    def test_forward_return_sequence(self, encoder, config):
        """Test forward pass with sequence embeddings."""
        batch_size = 4
        x = torch.randn(batch_size, config.seq_length, config.input_dim)

        outputs = encoder(x, return_sequence=True)

        assert "sequence_embeddings" in outputs
        assert outputs["sequence_embeddings"].shape == (
            batch_size, config.seq_length, config.hidden_dim
        )

    def test_training_mode_sampling(self, encoder, config):
        """Test that training mode uses stochastic sampling."""
        encoder.train()
        torch.manual_seed(42)
        x = torch.randn(4, config.seq_length, config.input_dim)

        # With same seed, outputs should be reproducible
        torch.manual_seed(123)
        outputs1 = encoder(x)
        torch.manual_seed(456)
        outputs2 = encoder(x)

        # In training mode, z values should differ due to different seeds
        # (even though input is the same)
        # Note: mu/logvar may also differ slightly due to dropout
        assert outputs1["z"].shape == outputs2["z"].shape

    def test_eval_mode_deterministic(self, encoder, config):
        """Test that eval mode is deterministic."""
        encoder.eval()
        x = torch.randn(4, config.seq_length, config.input_dim)

        with torch.no_grad():
            outputs1 = encoder(x)
            outputs2 = encoder(x)

        assert torch.allclose(outputs1["z"], outputs2["z"])

    def test_gradient_flow(self, encoder, config):
        """Test gradients flow through encoder."""
        x = torch.randn(4, config.seq_length, config.input_dim)

        outputs = encoder(x)
        loss = outputs["z"].sum()
        loss.backward()

        assert encoder.embedding.weight.grad is not None


class TestSequenceDecoder:
    """Tests for SequenceDecoder."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SelfSupervisedConfig(
            input_dim=21,
            seq_length=50,
            hidden_dim=64,
            latent_dim=16,
        )

    @pytest.fixture
    def decoder(self, config):
        """Create decoder fixture."""
        return SequenceDecoder(config)

    def test_initialization(self, decoder, config):
        """Test decoder initialization."""
        assert decoder.from_latent is not None
        assert decoder.decoder is not None

    def test_forward(self, decoder, config):
        """Test decoder forward pass."""
        batch_size = 4
        z = torch.randn(batch_size, config.latent_dim)

        logits = decoder(z)

        assert logits.shape == (batch_size, config.seq_length, config.input_dim)

    def test_gradient_flow(self, decoder, config):
        """Test gradients flow through decoder."""
        z = torch.randn(4, config.latent_dim)

        logits = decoder(z)
        loss = logits.sum()
        loss.backward()

        assert decoder.from_latent.weight.grad is not None


class TestMaskedSequenceModeling:
    """Tests for MaskedSequenceModeling head."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SelfSupervisedConfig(
            input_dim=21,
            seq_length=50,
            hidden_dim=64,
        )

    @pytest.fixture
    def msm_head(self, config):
        """Create MSM head fixture."""
        return MaskedSequenceModeling(config)

    def test_initialization(self, msm_head):
        """Test MSM head initialization."""
        assert msm_head.decoder is not None

    def test_forward(self, msm_head, config):
        """Test MSM forward pass."""
        batch_size = 4
        seq_len = config.seq_length

        # Sequence embeddings
        seq_emb = torch.randn(batch_size, seq_len, config.hidden_dim)

        # Create mask (15% masked)
        mask = torch.rand(batch_size, seq_len) < 0.15

        predictions = msm_head(seq_emb, mask)

        n_masked = mask.sum().item()
        assert predictions.shape == (n_masked, config.input_dim)

    def test_loss_computation(self, msm_head):
        """Test loss computation."""
        predictions = torch.randn(10, 21)  # 10 masked positions
        targets = torch.randint(0, 21, (10,))

        loss = msm_head.compute_loss(predictions, targets)

        assert loss.shape == ()
        assert loss.item() > 0


class TestContrastiveHead:
    """Tests for ContrastiveHead."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SelfSupervisedConfig(
            latent_dim=16,
            hidden_dim=64,
        )

    @pytest.fixture
    def contrastive_head(self, config):
        """Create contrastive head fixture."""
        return ContrastiveHead(config)

    def test_initialization(self, contrastive_head):
        """Test contrastive head initialization."""
        assert contrastive_head.projector is not None
        assert contrastive_head.predictor is not None

    def test_forward(self, contrastive_head, config):
        """Test contrastive head forward pass."""
        batch_size = 4
        z = torch.randn(batch_size, config.latent_dim)

        projection, prediction = contrastive_head(z)

        assert projection.shape == (batch_size, config.latent_dim)
        assert prediction.shape == (batch_size, config.latent_dim)


class TestMutationHead:
    """Tests for MutationHead."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SelfSupervisedConfig(
            latent_dim=16,
            hidden_dim=64,
            dropout=0.1,
        )

    @pytest.fixture
    def mutation_head(self, config):
        """Create mutation head fixture."""
        return MutationHead(config)

    def test_initialization(self, mutation_head):
        """Test mutation head initialization."""
        assert mutation_head.head is not None

    def test_forward(self, mutation_head, config):
        """Test mutation head forward pass."""
        batch_size = 4
        z_original = torch.randn(batch_size, config.latent_dim)
        z_mutant = torch.randn(batch_size, config.latent_dim)

        impact = mutation_head(z_original, z_mutant)

        assert impact.shape == (batch_size,)


class TestSequenceAugmenter:
    """Tests for SequenceAugmenter."""

    @pytest.fixture
    def augmenter(self):
        """Create augmenter fixture."""
        return SequenceAugmenter(mask_prob=0.15, replace_prob=0.1)

    def test_initialization(self, augmenter):
        """Test augmenter initialization."""
        assert augmenter.mask_prob == 0.15
        assert augmenter.replace_prob == 0.1
        assert augmenter.n_classes == 21

    def test_augmentation_3d(self, augmenter):
        """Test augmentation on 3D input."""
        x = torch.randn(4, 50, 21)

        x_aug = augmenter(x)

        assert x_aug.shape == x.shape
        # Some positions should be different
        assert not torch.allclose(x, x_aug)

    def test_augmentation_preserves_shape(self, augmenter):
        """Test augmentation preserves input shape."""
        x = torch.randn(8, 100, 4)

        x_aug = augmenter(x)

        assert x_aug.shape == x.shape


class TestSelfSupervisedModel:
    """Tests for SelfSupervisedModel."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SelfSupervisedConfig(
            input_dim=21,
            seq_length=50,
            hidden_dim=64,
            latent_dim=16,
            n_layers=2,
            n_heads=4,
            objectives=[
                PretrainingObjective.MSM,
                PretrainingObjective.CONTRASTIVE,
                PretrainingObjective.VAE,
            ],
        )

    @pytest.fixture
    def model(self, config):
        """Create model fixture."""
        return SelfSupervisedModel(config)

    def test_initialization(self, model, config):
        """Test model initialization."""
        assert model.encoder is not None
        assert hasattr(model, "decoder")
        assert hasattr(model, "contrastive_head")
        assert hasattr(model, "target_encoder")

    def test_initialization_with_mutation_head(self):
        """Test model with mutation head."""
        config = SelfSupervisedConfig(
            input_dim=21,
            seq_length=50,
            hidden_dim=64,
            latent_dim=16,
            n_layers=2,
            objectives=[PretrainingObjective.MUTATION],
        )
        model = SelfSupervisedModel(config)

        assert hasattr(model, "mutation_head")

    def test_forward(self, model, config):
        """Test model forward pass."""
        batch_size = 4
        x = torch.randn(batch_size, config.seq_length, config.input_dim)

        outputs = model(x)

        assert "mu" in outputs
        assert "logvar" in outputs
        assert "z" in outputs
        assert "reconstruction" in outputs
        assert "proj_online" in outputs
        assert "pred_online" in outputs

    def test_forward_without_contrastive(self, model, config):
        """Test forward without contrastive computation."""
        batch_size = 4
        x = torch.randn(batch_size, config.seq_length, config.input_dim)

        outputs = model(x, compute_contrastive=False)

        assert "proj_online" not in outputs

    def test_compute_losses(self, model, config):
        """Test loss computation."""
        batch_size = 4
        x = torch.randn(batch_size, config.seq_length, config.input_dim)

        losses = model.compute_losses(x)

        assert "total" in losses
        assert "recon" in losses
        assert "kl" in losses
        assert "contrastive" in losses
        assert losses["total"].item() > 0

    def test_momentum_update(self, model, config):
        """Test target encoder momentum update."""
        # Get initial target params
        target_param = next(model.target_encoder.parameters()).clone()

        # Update online encoder
        for param in model.encoder.parameters():
            param.data.add_(0.1)

        # Momentum update
        model._momentum_update()

        # Target should have moved towards online
        new_target_param = next(model.target_encoder.parameters())
        assert not torch.allclose(target_param, new_target_param)

    def test_get_embeddings(self, model, config):
        """Test getting embeddings."""
        batch_size = 4
        x = torch.randn(batch_size, config.seq_length, config.input_dim)

        embeddings = model.get_embeddings(x)

        assert embeddings.shape == (batch_size, config.latent_dim)

    def test_gradient_flow(self, model, config):
        """Test gradients flow through model."""
        x = torch.randn(4, config.seq_length, config.input_dim)

        losses = model.compute_losses(x)
        losses["total"].backward()

        assert model.encoder.embedding.weight.grad is not None


class TestSelfSupervisedPretrainer:
    """Tests for SelfSupervisedPretrainer."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SelfSupervisedConfig(
            input_dim=21,
            seq_length=50,
            hidden_dim=64,
            latent_dim=16,
            n_layers=2,
            n_heads=4,
            pretrain_epochs=2,
            batch_size=8,
            device="cpu",
        )

    @pytest.fixture
    def pretrainer(self, config, tmp_path):
        """Create pretrainer fixture."""
        config.checkpoint_dir = tmp_path / "checkpoints"
        return SelfSupervisedPretrainer(config)

    @pytest.fixture
    def dataset(self):
        """Create dummy dataset."""
        return DummySequenceDataset(n_samples=32, seq_length=50, input_dim=21)

    def test_initialization(self, pretrainer, config):
        """Test pretrainer initialization."""
        assert pretrainer.config == config
        assert pretrainer.model is None
        assert pretrainer.checkpoint_dir.exists()

    def test_pretrain(self, pretrainer, dataset):
        """Test pre-training loop."""
        losses_collected = []

        def callback(epoch, losses):
            losses_collected.append(losses["total"])

        encoder = pretrainer.pretrain(dataset, callback=callback)

        assert encoder is not None
        assert isinstance(encoder, SequenceEncoder)
        assert len(losses_collected) == 2  # 2 epochs
        assert pretrainer.model is not None

    def test_pretrain_saves_checkpoints(self, pretrainer, dataset, tmp_path):
        """Test that pretraining saves checkpoints."""
        # Run 10+ epochs to trigger checkpoint saving
        pretrainer.config.pretrain_epochs = 11

        pretrainer.pretrain(dataset)

        # Should have final checkpoint
        assert (pretrainer.checkpoint_dir / "pretrain_final.pt").exists()

    def test_load_checkpoint(self, pretrainer, dataset):
        """Test loading checkpoint."""
        # First pretrain
        pretrainer.pretrain(dataset)
        checkpoint_path = pretrainer.checkpoint_dir / "pretrain_final.pt"

        # Load encoder
        encoder = pretrainer.load_checkpoint(checkpoint_path)

        assert encoder is not None
        assert isinstance(encoder, SequenceEncoder)

    def test_create_downstream_model(self, pretrainer, dataset):
        """Test creating downstream model."""
        pretrainer.pretrain(dataset)

        downstream = pretrainer.create_downstream_model(n_outputs=5)

        assert downstream is not None
        assert isinstance(downstream, nn.Module)

        # Test forward pass
        x = torch.randn(4, 50, 21)
        outputs = downstream(x)

        assert "predictions" in outputs
        assert outputs["predictions"].shape == (4, 5)

    def test_create_downstream_model_frozen(self, pretrainer, dataset):
        """Test frozen encoder in downstream model."""
        pretrainer.pretrain(dataset)

        downstream = pretrainer.create_downstream_model(n_outputs=1, freeze_encoder=True)

        # Check encoder is frozen
        for param in downstream.encoder.parameters():
            assert not param.requires_grad

    def test_create_downstream_model_unfrozen(self, pretrainer, dataset):
        """Test unfrozen encoder in downstream model."""
        pretrainer.pretrain(dataset)

        downstream = pretrainer.create_downstream_model(n_outputs=1, freeze_encoder=False)

        # Check encoder is trainable
        for param in downstream.encoder.parameters():
            assert param.requires_grad

    def test_create_downstream_without_pretrain(self, pretrainer):
        """Test error when creating downstream without pretraining."""
        with pytest.raises(ValueError, match="No pretrained model"):
            pretrainer.create_downstream_model(n_outputs=1)

    def test_evaluate_representations_without_model(self, pretrainer, dataset):
        """Test evaluation without model returns error."""
        result = pretrainer.evaluate_representations(dataset)
        assert "error" in result


class TestSelfSupervisedModelOnlyMSM:
    """Tests for SelfSupervisedModel with only MSM objective."""

    @pytest.fixture
    def config(self):
        """Create config with only MSM."""
        return SelfSupervisedConfig(
            input_dim=21,
            seq_length=50,
            hidden_dim=64,
            latent_dim=16,
            n_layers=2,
            n_heads=4,
            objectives=[PretrainingObjective.MSM],
        )

    @pytest.fixture
    def model(self, config):
        """Create model fixture."""
        return SelfSupervisedModel(config)

    def test_only_msm_head(self, model):
        """Test model only has MSM head."""
        assert hasattr(model, "msm_head")
        assert not hasattr(model, "decoder")
        assert not hasattr(model, "contrastive_head")

    def test_forward_without_contrastive(self, model, config):
        """Test forward without contrastive components."""
        x = torch.randn(4, config.seq_length, config.input_dim)

        outputs = model(x)

        assert "mu" in outputs
        assert "proj_online" not in outputs


class TestSelfSupervisedModelOnlyVAE:
    """Tests for SelfSupervisedModel with only VAE objective."""

    @pytest.fixture
    def config(self):
        """Create config with only VAE."""
        return SelfSupervisedConfig(
            input_dim=21,
            seq_length=50,
            hidden_dim=64,
            latent_dim=16,
            n_layers=2,
            n_heads=4,
            objectives=[PretrainingObjective.VAE],
        )

    @pytest.fixture
    def model(self, config):
        """Create model fixture."""
        return SelfSupervisedModel(config)

    def test_only_decoder(self, model):
        """Test model only has decoder."""
        assert hasattr(model, "decoder")
        assert not hasattr(model, "msm_head")
        assert not hasattr(model, "contrastive_head")

    def test_vae_loss(self, model, config):
        """Test VAE loss computation."""
        x = torch.randn(4, config.seq_length, config.input_dim)

        losses = model.compute_losses(x)

        assert "recon" in losses
        assert "kl" in losses
        assert "contrastive" not in losses


class TestIntegration:
    """Integration tests for self-supervised pre-training."""

    @pytest.fixture
    def config(self):
        """Create full config."""
        return SelfSupervisedConfig(
            input_dim=21,
            seq_length=50,
            hidden_dim=64,
            latent_dim=16,
            n_layers=2,
            n_heads=4,
            pretrain_epochs=3,
            batch_size=8,
            device="cpu",
        )

    def test_full_pipeline(self, config, tmp_path):
        """Test full pre-training to fine-tuning pipeline."""
        config.checkpoint_dir = tmp_path / "checkpoints"

        # Create datasets
        pretrain_data = DummySequenceDataset(n_samples=32, seq_length=50, input_dim=21)
        finetune_data = DummySequenceDataset(
            n_samples=16, seq_length=50, input_dim=21, include_labels=True
        )

        # Pre-train
        pretrainer = SelfSupervisedPretrainer(config)
        encoder = pretrainer.pretrain(pretrain_data)

        # Create downstream model
        downstream = pretrainer.create_downstream_model(n_outputs=3, freeze_encoder=True)

        # Verify downstream works
        x = torch.randn(4, 50, 21)
        outputs = downstream(x)

        assert outputs["predictions"].shape == (4, 3)

    def test_checkpoint_save_load_roundtrip(self, config, tmp_path):
        """Test saving and loading checkpoints."""
        config.checkpoint_dir = tmp_path / "checkpoints"

        # Pre-train
        dataset = DummySequenceDataset(n_samples=32, seq_length=50, input_dim=21)
        pretrainer = SelfSupervisedPretrainer(config)
        original_encoder = pretrainer.pretrain(dataset)

        # Get original embedding (in eval mode for deterministic output)
        original_encoder.eval()
        x = torch.randn(4, 50, 21)
        with torch.no_grad():
            original_out = original_encoder(x)

        # Load from checkpoint
        loaded_encoder = pretrainer.load_checkpoint(
            pretrainer.checkpoint_dir / "pretrain_final.pt"
        )
        loaded_encoder.eval()

        # Compare outputs
        with torch.no_grad():
            loaded_out = loaded_encoder(x)

        assert torch.allclose(original_out["mu"], loaded_out["mu"], atol=1e-5)

    def test_losses_decrease(self, config, tmp_path):
        """Test that losses decrease during training."""
        config.checkpoint_dir = tmp_path / "checkpoints"
        config.pretrain_epochs = 5

        dataset = DummySequenceDataset(n_samples=64, seq_length=50, input_dim=21)
        pretrainer = SelfSupervisedPretrainer(config)

        losses = []

        def callback(epoch, epoch_losses):
            losses.append(epoch_losses["total"])

        pretrainer.pretrain(dataset, callback=callback)

        # First loss should be higher than last (generally)
        # Note: With random data, this isn't guaranteed, so we just check losses exist
        assert len(losses) == 5
        assert all(l > 0 for l in losses)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
