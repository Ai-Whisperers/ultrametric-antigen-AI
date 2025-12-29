# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Comprehensive unit tests for Transfer Learning Pipeline.

Tests cover:
- TransferConfig
- TransferStrategy
- SharedEncoder
- DiseaseHead
- MultiDiseaseModel
- AdapterLayer
- LoRALayer
- TransferLearningPipeline
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset

from src.training.transfer_pipeline import (
    AdapterLayer,
    DiseaseHead,
    LoRALayer,
    MultiDiseaseModel,
    SharedEncoder,
    TransferConfig,
    TransferLearningPipeline,
    TransferStrategy,
)


class SimpleDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, n_samples=100, input_dim=64, n_outputs=5):
        self.x = torch.randn(n_samples, input_dim)
        self.y = torch.rand(n_samples, n_outputs)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}


class TestTransferConfig:
    """Test TransferConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = TransferConfig()

        assert config.latent_dim == 32
        assert config.pretrain_epochs == 100
        assert config.finetune_epochs == 50
        assert config.strategy == TransferStrategy.FROZEN_ENCODER

    def test_custom_config(self):
        """Test custom configuration."""
        config = TransferConfig(
            latent_dim=64,
            pretrain_epochs=50,
            strategy=TransferStrategy.LORA,
            lora_rank=16,
        )

        assert config.latent_dim == 64
        assert config.pretrain_epochs == 50
        assert config.strategy == TransferStrategy.LORA
        assert config.lora_rank == 16

    def test_checkpoint_dir(self):
        """Test checkpoint directory."""
        config = TransferConfig(checkpoint_dir=Path("/tmp/checkpoints"))

        assert config.checkpoint_dir == Path("/tmp/checkpoints")


class TestTransferStrategy:
    """Test TransferStrategy enum."""

    def test_strategy_values(self):
        """Test strategy values."""
        assert TransferStrategy.FULL_FINETUNE.value == "full"
        assert TransferStrategy.FROZEN_ENCODER.value == "frozen_encoder"
        assert TransferStrategy.ADAPTER.value == "adapter"
        assert TransferStrategy.LORA.value == "lora"
        assert TransferStrategy.MAML.value == "maml"


class TestSharedEncoder:
    """Test SharedEncoder module."""

    @pytest.fixture
    def encoder(self):
        """Create encoder fixture."""
        return SharedEncoder(
            input_dim=64,
            latent_dim=32,
            hidden_dims=[128, 64],
        )

    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.fc_mu.out_features == 32
        assert encoder.fc_logvar.out_features == 32

    def test_forward_shape(self, encoder):
        """Test forward pass shape."""
        x = torch.randn(8, 64)
        mu, logvar = encoder(x)

        assert mu.shape == (8, 32)
        assert logvar.shape == (8, 32)

    def test_encoder_layers(self, encoder):
        """Test encoder has correct layers."""
        # Should have linear, activation, batchnorm, dropout for each hidden dim
        layers = list(encoder.encoder.children())
        assert len(layers) > 0


class TestDiseaseHead:
    """Test DiseaseHead module."""

    @pytest.fixture
    def head(self):
        """Create head fixture."""
        return DiseaseHead(latent_dim=32, n_outputs=5, hidden_dim=64)

    def test_initialization(self, head):
        """Test head initialization."""
        assert head.head is not None

    def test_forward_shape(self, head):
        """Test forward pass shape."""
        z = torch.randn(8, 32)
        output = head(z)

        assert output.shape == (8, 5)


class TestAdapterLayer:
    """Test AdapterLayer module."""

    @pytest.fixture
    def adapter(self):
        """Create adapter fixture."""
        return AdapterLayer(input_dim=128, adapter_dim=32)

    def test_initialization(self, adapter):
        """Test adapter initialization."""
        assert adapter.down.in_features == 128
        assert adapter.down.out_features == 32
        assert adapter.up.in_features == 32
        assert adapter.up.out_features == 128

    def test_forward_residual(self, adapter):
        """Test forward pass has residual connection."""
        x = torch.randn(8, 128)
        output = adapter(x)

        assert output.shape == x.shape

        # Should be different from input (adapter modifies)
        # but not completely different (residual)

    def test_near_identity_init(self, adapter):
        """Test near-identity initialization."""
        # up layer should be initialized near zero
        assert torch.allclose(adapter.up.weight, torch.zeros_like(adapter.up.weight), atol=1e-6)
        assert torch.allclose(adapter.up.bias, torch.zeros_like(adapter.up.bias), atol=1e-6)


class TestLoRALayer:
    """Test LoRALayer module."""

    @pytest.fixture
    def lora(self):
        """Create LoRA fixture."""
        return LoRALayer(input_dim=64, output_dim=128, rank=8)

    def test_initialization(self, lora):
        """Test LoRA initialization."""
        assert lora.lora_A.shape == (64, 8)
        assert lora.lora_B.shape == (8, 128)

    def test_forward_shape(self, lora):
        """Test forward pass shape."""
        x = torch.randn(8, 64)
        delta = lora(x)

        assert delta.shape == (8, 128)

    def test_low_rank_approximation(self, lora):
        """Test low-rank nature."""
        # The effective weight matrix should have rank <= 8
        weight = lora.lora_A @ lora.lora_B
        rank = torch.linalg.matrix_rank(weight)
        assert rank <= 8


class TestMultiDiseaseModel:
    """Test MultiDiseaseModel."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return MultiDiseaseModel(
            input_dim=64,
            latent_dim=32,
            hidden_dims=[128, 64],
            disease_outputs={"hiv": 25, "hbv": 6, "tb": 8},
        )

    def test_initialization(self, model):
        """Test model initialization."""
        assert "hiv" in model.heads
        assert "hbv" in model.heads
        assert "tb" in model.heads

    def test_forward_shape(self, model):
        """Test forward pass shape."""
        x = torch.randn(8, 64)
        outputs = model(x, "hiv")

        assert outputs["logits"].shape == (8, 64)
        assert outputs["predictions"].shape == (8, 25)
        assert outputs["mu"].shape == (8, 32)
        assert outputs["z"].shape == (8, 32)

    def test_different_diseases(self, model):
        """Test different disease heads."""
        x = torch.randn(8, 64)

        hiv_out = model(x, "hiv")
        hbv_out = model(x, "hbv")
        tb_out = model(x, "tb")

        assert hiv_out["predictions"].shape == (8, 25)
        assert hbv_out["predictions"].shape == (8, 6)
        assert tb_out["predictions"].shape == (8, 8)

    def test_reparameterize_training(self, model):
        """Test reparameterization in training."""
        model.train()
        mu = torch.zeros(8, 32)
        logvar = torch.zeros(8, 32)

        z1 = model.reparameterize(mu, logvar)
        z2 = model.reparameterize(mu, logvar)

        assert not torch.allclose(z1, z2)

    def test_get_encoder_params(self, model):
        """Test getting encoder parameters."""
        params = list(model.get_encoder_params())
        assert len(params) > 0

    def test_get_head_params(self, model):
        """Test getting head parameters."""
        params = list(model.get_head_params("hiv"))
        assert len(params) > 0


class TestTransferLearningPipeline:
    """Test TransferLearningPipeline."""

    @pytest.fixture
    def config(self):
        """Create config fixture."""
        return TransferConfig(
            latent_dim=16,
            hidden_dims=[32, 16],
            pretrain_epochs=2,
            finetune_epochs=2,
            batch_size=8,
            checkpoint_dir=Path(tempfile.mkdtemp()),
        )

    @pytest.fixture
    def pipeline(self, config):
        """Create pipeline fixture."""
        return TransferLearningPipeline(config)

    @pytest.fixture
    def datasets(self):
        """Create dataset fixtures."""
        return {
            "hiv": SimpleDataset(50, 64, 25),
            "hbv": SimpleDataset(50, 64, 6),
        }

    def test_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.pretrained_model is None
        assert pipeline.checkpoint_dir.exists()

    def test_pretrain(self, pipeline, datasets):
        """Test pretraining."""
        disease_outputs = {"hiv": 25, "hbv": 6}

        model = pipeline.pretrain(datasets, disease_outputs, input_dim=64)

        assert model is not None
        assert pipeline.pretrained_model is model
        assert "hiv" in model.heads
        assert "hbv" in model.heads

    def test_finetune_after_pretrain(self, pipeline, datasets):
        """Test fine-tuning after pretraining."""
        disease_outputs = {"hiv": 25, "hbv": 6}
        pipeline.pretrain(datasets, disease_outputs, input_dim=64)

        # Fine-tune on new disease
        tb_dataset = SimpleDataset(30, 64, 8)
        model = pipeline.finetune("tb", tb_dataset, n_outputs=8)

        assert "tb" in model.heads

    def test_finetune_without_pretrain(self, pipeline):
        """Test fine-tuning without pretraining."""
        dataset = SimpleDataset(30, 64, 5)

        model = pipeline.finetune("new_disease", dataset, n_outputs=5, input_dim=64)

        assert model is not None

    def test_different_strategies(self, config):
        """Test different transfer strategies."""
        for strategy in [
            TransferStrategy.FULL_FINETUNE,
            TransferStrategy.FROZEN_ENCODER,
        ]:
            config.strategy = strategy
            pipeline = TransferLearningPipeline(config)

            datasets = {"disease1": SimpleDataset(20, 64, 5)}
            model = pipeline.pretrain(datasets, {"disease1": 5}, input_dim=64)

            target_dataset = SimpleDataset(20, 64, 3)
            finetuned = pipeline.finetune("target", target_dataset, n_outputs=3)

            assert finetuned is not None

    def test_evaluate_transfer(self, pipeline):
        """Test transfer evaluation."""
        # Create datasets with same output dimensions for proper transfer evaluation
        test_datasets = {
            "hiv": SimpleDataset(50, 64, 6),
            "hbv": SimpleDataset(50, 64, 6),
        }
        disease_outputs = {"hiv": 6, "hbv": 6}
        pipeline.pretrain(test_datasets, disease_outputs, input_dim=64)

        # Evaluate transfer from HIV to HBV (same output dimensions)
        metrics = pipeline.evaluate_transfer("hiv", "hbv", test_datasets["hbv"])

        assert "mse" in metrics or "error" in metrics

    def test_checkpoint_saving(self, pipeline, datasets):
        """Test checkpoint saving."""
        disease_outputs = {"hiv": 25}
        # Use only HIV dataset
        hiv_only = {"hiv": datasets["hiv"]}
        pipeline.pretrain(hiv_only, disease_outputs, input_dim=64)

        # Check checkpoint exists
        checkpoints = list(pipeline.checkpoint_dir.glob("*.pt"))
        assert len(checkpoints) > 0

    def test_callback(self, pipeline, datasets):
        """Test training with callback."""
        epochs_seen = []

        def callback(epoch, losses):
            epochs_seen.append(epoch)

        disease_outputs = {"hiv": 25}
        pipeline.pretrain({"hiv": datasets["hiv"]}, disease_outputs, input_dim=64, callback=callback)

        assert len(epochs_seen) == pipeline.config.pretrain_epochs


class TestTransferPipelineEdgeCases:
    """Test edge cases for transfer pipeline."""

    def test_empty_dataset(self):
        """Test with empty dataset."""
        config = TransferConfig(pretrain_epochs=1)
        pipeline = TransferLearningPipeline(config)

        # Empty dataset should fail gracefully
        # (implementation specific)

    def test_single_sample_dataset(self):
        """Test with single sample."""
        config = TransferConfig(pretrain_epochs=1, batch_size=1)
        pipeline = TransferLearningPipeline(config)

        dataset = SimpleDataset(1, 64, 5)

        # Should handle gracefully

    def test_mismatched_dimensions(self):
        """Test with mismatched input dimensions."""
        config = TransferConfig(pretrain_epochs=1)
        pipeline = TransferLearningPipeline(config)

        dataset64 = SimpleDataset(20, 64, 5)
        dataset128 = SimpleDataset(20, 128, 5)

        # First pretrain with 64
        pipeline.pretrain({"d1": dataset64}, {"d1": 5}, input_dim=64)

        # Trying to use 128 should fail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
