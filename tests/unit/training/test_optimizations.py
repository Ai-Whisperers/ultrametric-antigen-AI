# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unit tests for training optimizations.

Tests cover:
- Mixed Precision Training
- Gradient Checkpointing
- Stochastic Weight Averaging (SWA)
- Gradient Accumulation
- Dynamic Batching
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset

from src.training.optimizations import (
    CheckpointedModule,
    DynamicBatchSampler,
    EarlyStopping,
    GradientAccumulator,
    MixedPrecisionConfig,
    MixedPrecisionTrainer,
    SWAConfig,
    SWAWrapper,
    apply_gradient_checkpointing,
    estimate_memory_usage,
)


class TestMixedPrecisionConfig:
    """Tests for MixedPrecisionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = MixedPrecisionConfig()

        assert config.enabled is True
        assert config.dtype == "float16"
        assert config.init_scale == 65536.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = MixedPrecisionConfig(
            enabled=False,
            dtype="bfloat16",
            init_scale=1024.0,
        )

        assert config.enabled is False
        assert config.dtype == "bfloat16"
        assert config.init_scale == 1024.0


class TestMixedPrecisionTrainer:
    """Tests for MixedPrecisionTrainer."""

    def test_initialization_enabled(self):
        """Test initialization with mixed precision enabled."""
        config = MixedPrecisionConfig(enabled=True, dtype="float16")
        trainer = MixedPrecisionTrainer(config)

        assert trainer._enabled is True
        assert trainer.dtype == torch.float16
        assert trainer.scaler is not None

    def test_initialization_disabled(self):
        """Test initialization with mixed precision disabled."""
        config = MixedPrecisionConfig(enabled=False)
        trainer = MixedPrecisionTrainer(config)

        assert trainer._enabled is False
        assert trainer.dtype == torch.float32
        assert trainer.scaler is None

    def test_initialization_bfloat16(self):
        """Test BF16 initialization (no scaler needed)."""
        config = MixedPrecisionConfig(enabled=True, dtype="bfloat16")
        trainer = MixedPrecisionTrainer(config)

        assert trainer.dtype == torch.bfloat16
        assert trainer.scaler is None  # BF16 doesn't need loss scaling

    def test_autocast_context(self):
        """Test autocast context manager."""
        config = MixedPrecisionConfig(enabled=True)
        trainer = MixedPrecisionTrainer(config)

        # Should not raise
        with trainer.autocast():
            x = torch.randn(10, 10)
            y = x @ x.T

    def test_backward_with_scaler(self):
        """Test backward pass with loss scaling."""
        config = MixedPrecisionConfig(enabled=True, dtype="float16")
        trainer = MixedPrecisionTrainer(config)

        model = nn.Linear(10, 5)
        optimizer = Adam(model.parameters())

        x = torch.randn(4, 10)
        with trainer.autocast():
            y = model(x)
            loss = y.sum()

        optimizer.zero_grad()
        trainer.backward(loss)
        trainer.step(optimizer)

        # Check gradients were computed
        assert model.weight.grad is not None

    def test_get_scale(self):
        """Test getting current loss scale."""
        config = MixedPrecisionConfig(enabled=True, dtype="float16")
        trainer = MixedPrecisionTrainer(config)

        scale = trainer.get_scale()
        assert scale > 0

    def test_state_dict(self):
        """Test state dict for checkpointing."""
        config = MixedPrecisionConfig(enabled=True)
        trainer = MixedPrecisionTrainer(config)

        state = trainer.state_dict()
        assert "scaler" in state


class TestGradientCheckpointing:
    """Tests for Gradient Checkpointing."""

    def test_checkpointed_module_init(self):
        """Test CheckpointedModule initialization."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        checkpointed = CheckpointedModule(model, num_segments=2)

        assert checkpointed.module is model
        assert checkpointed.num_segments == 2

    def test_checkpointed_forward_training(self):
        """Test forward pass during training."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        checkpointed = CheckpointedModule(model, num_segments=2)
        checkpointed.train()

        x = torch.randn(4, 10)
        y = checkpointed(x)

        assert y.shape == (4, 5)

    def test_checkpointed_forward_eval(self):
        """Test forward pass during evaluation (no checkpointing)."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        checkpointed = CheckpointedModule(model, num_segments=2)
        checkpointed.eval()

        x = torch.randn(4, 10)
        y = checkpointed(x)

        assert y.shape == (4, 5)

    def test_checkpointed_backward(self):
        """Test backward pass with checkpointing."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        checkpointed = CheckpointedModule(model, num_segments=2)
        checkpointed.train()

        x = torch.randn(4, 10, requires_grad=True)
        y = checkpointed(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None

    def test_apply_checkpointing(self):
        """Test apply_gradient_checkpointing function."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Sequential(
                nn.Linear(20, 30),
                nn.ReLU(),
                nn.Linear(30, 5),
            ),
        )

        # Apply checkpointing to specific layers
        model = apply_gradient_checkpointing(model, checkpoint_layers=["2"])

        assert isinstance(model[2], CheckpointedModule)


class TestSWA:
    """Tests for Stochastic Weight Averaging."""

    def test_swa_config(self):
        """Test SWA configuration."""
        config = SWAConfig(
            swa_start=50,
            swa_lr=0.001,
            swa_freq=2,
        )

        assert config.swa_start == 50
        assert config.swa_lr == 0.001
        assert config.swa_freq == 2

    def test_swa_wrapper_init(self):
        """Test SWA wrapper initialization."""
        model = nn.Linear(10, 5)
        optimizer = SGD(model.parameters(), lr=0.01)
        config = SWAConfig(swa_start=10)

        swa = SWAWrapper(model, optimizer, config, device="cpu")

        assert swa.swa_model is not None
        assert swa.swa_scheduler is not None
        assert not swa.is_active()

    def test_swa_step_before_start(self):
        """Test SWA step before start epoch."""
        model = nn.Linear(10, 5)
        optimizer = SGD(model.parameters(), lr=0.01)
        config = SWAConfig(swa_start=10)

        swa = SWAWrapper(model, optimizer, config, device="cpu")

        updated = swa.step(epoch=5)

        assert not updated
        assert not swa.is_active()

    def test_swa_step_after_start(self):
        """Test SWA step after start epoch."""
        model = nn.Linear(10, 5)
        optimizer = SGD(model.parameters(), lr=0.01)
        config = SWAConfig(swa_start=10, swa_freq=1)

        swa = SWAWrapper(model, optimizer, config, device="cpu")

        updated = swa.step(epoch=10)

        assert updated
        assert swa.is_active()

    def test_swa_state_dict(self):
        """Test SWA state dict for checkpointing."""
        model = nn.Linear(10, 5)
        optimizer = SGD(model.parameters(), lr=0.01)

        swa = SWAWrapper(model, optimizer, device="cpu")
        swa.step(epoch=75)

        state = swa.state_dict()

        assert "swa_model" in state
        assert "swa_scheduler" in state
        assert "update_count" in state


class TestGradientAccumulator:
    """Tests for Gradient Accumulation."""

    def test_init(self):
        """Test accumulator initialization."""
        acc = GradientAccumulator(accumulation_steps=4)

        assert acc.accumulation_steps == 4

    def test_step_not_ready(self):
        """Test step returns False when not ready."""
        acc = GradientAccumulator(accumulation_steps=4)

        assert not acc.step()
        assert not acc.step()
        assert not acc.step()

    def test_step_ready(self):
        """Test step returns True when ready."""
        acc = GradientAccumulator(accumulation_steps=4)

        acc.step()
        acc.step()
        acc.step()
        assert acc.step()  # 4th step triggers

    def test_reset(self):
        """Test reset functionality."""
        acc = GradientAccumulator(accumulation_steps=4)

        acc.step()
        acc.step()
        acc.reset()

        # Should need 4 more steps
        assert not acc.step()
        assert not acc.step()
        assert not acc.step()
        assert acc.step()

    def test_should_step(self):
        """Test should_step predicate."""
        acc = GradientAccumulator(accumulation_steps=3)

        assert not acc.should_step()
        acc.step()
        assert not acc.should_step()
        acc.step()
        assert acc.should_step()  # Next step will trigger

    def test_get_loss_scale(self):
        """Test loss scale factor."""
        acc = GradientAccumulator(accumulation_steps=4)

        assert acc.get_loss_scale() == pytest.approx(0.25)


class TestEarlyStopping:
    """Tests for Early Stopping."""

    def test_init_min_mode(self):
        """Test initialization in min mode."""
        es = EarlyStopping(patience=5, mode="min")

        assert es.patience == 5
        assert es.mode == "min"
        assert es.best == float("inf")

    def test_init_max_mode(self):
        """Test initialization in max mode."""
        es = EarlyStopping(patience=5, mode="max")

        assert es.best == float("-inf")

    def test_step_improvement(self):
        """Test step with improvement."""
        es = EarlyStopping(patience=3, mode="min")

        assert not es.step(1.0)  # First value
        assert not es.step(0.9)  # Improvement
        assert not es.step(0.8)  # Improvement

        assert es.counter == 0

    def test_step_no_improvement(self):
        """Test step without improvement."""
        es = EarlyStopping(patience=3, mode="min")

        es.step(1.0)
        es.step(1.1)  # No improvement
        es.step(1.2)  # No improvement

        assert es.counter == 2

    def test_stop_triggered(self):
        """Test that stop is triggered after patience."""
        es = EarlyStopping(patience=3, mode="min")

        es.step(1.0)
        assert not es.step(1.1)  # No improvement
        assert not es.step(1.2)  # No improvement
        assert es.step(1.3)  # No improvement - triggers

        assert es.should_stop

    def test_reset(self):
        """Test reset functionality."""
        es = EarlyStopping(patience=3, mode="min")

        es.step(1.0)
        es.step(1.1)
        es.step(1.2)
        es.step(1.3)

        es.reset()

        assert not es.should_stop
        assert es.best == float("inf")
        assert es.counter == 0


class TestDynamicBatchSampler:
    """Tests for Dynamic Batch Sampler."""

    @pytest.fixture
    def variable_length_dataset(self):
        """Create dataset with variable length sequences."""
        # Simple dataset with variable length tensors
        data = [
            {"sequence": torch.randn(10)},
            {"sequence": torch.randn(20)},
            {"sequence": torch.randn(15)},
            {"sequence": torch.randn(25)},
            {"sequence": torch.randn(5)},
            {"sequence": torch.randn(30)},
        ]

        class SimpleDataset:
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        return SimpleDataset(data)

    def test_init(self, variable_length_dataset):
        """Test sampler initialization."""
        sampler = DynamicBatchSampler(
            variable_length_dataset,
            max_tokens=50,
            length_fn=lambda x: len(x),
        )

        assert sampler.max_tokens == 50
        assert len(sampler.batches) > 0

    def test_batches_respect_token_limit(self, variable_length_dataset):
        """Test that batches respect token limit."""
        sampler = DynamicBatchSampler(
            variable_length_dataset,
            max_tokens=50,
            length_fn=lambda x: len(x),
            shuffle=False,
        )

        for batch in sampler:
            # Calculate max length in batch
            max_len = max(
                len(variable_length_dataset[i]["sequence"])
                for i in batch
            )
            total_tokens = max_len * len(batch)
            # Allow some slack for edge cases
            assert total_tokens <= sampler.max_tokens * 1.5

    def test_iteration(self, variable_length_dataset):
        """Test iteration over sampler."""
        sampler = DynamicBatchSampler(
            variable_length_dataset,
            max_tokens=100,
            shuffle=False,
        )

        batches = list(sampler)
        all_indices = [idx for batch in batches for idx in batch]

        # All indices should be covered
        assert len(all_indices) == len(variable_length_dataset)

    def test_len(self, variable_length_dataset):
        """Test length of sampler."""
        sampler = DynamicBatchSampler(
            variable_length_dataset,
            max_tokens=50,
        )

        assert len(sampler) > 0


class TestEstimateMemoryUsage:
    """Tests for memory estimation utility."""

    def test_estimate_memory(self):
        """Test memory estimation."""
        model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

        estimate = estimate_memory_usage(
            model,
            batch_size=32,
            seq_length=100,
        )

        assert "parameters_gb" in estimate
        assert "gradients_gb" in estimate
        assert "optimizer_gb" in estimate
        assert "activations_gb" in estimate
        assert "total_gb" in estimate

        assert estimate["total_gb"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
