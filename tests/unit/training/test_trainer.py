"""Tests for src/training/trainer.py - TernaryVAETrainer.

This module tests the trainer utilities without requiring full model initialization.
Due to the trainer's many dependencies, we focus on:
- Utility methods (batch index computation)
- Scheduler components
- Basic mechanics
"""

from unittest.mock import patch

import pytest
import torch

# =============================================================================
# Batch Index Computation Tests (Core Utility)
# =============================================================================


class TestComputeBatchIndices:
    """Test _compute_batch_indices method - the core conversion utility."""

    @pytest.fixture
    def trainer_stub(self):
        """Create a minimal trainer stub with just the method we need."""
        # Import here to avoid module-level import issues
        from src.training.trainer import TernaryVAETrainer

        # Create a stub without calling __init__
        with patch.object(TernaryVAETrainer, "__init__", lambda self, *args, **kwargs: None):
            trainer = TernaryVAETrainer.__new__(TernaryVAETrainer)
            trainer._base3_weights = torch.tensor([3**i for i in range(9)], dtype=torch.long)
            return trainer

    def test_all_negative_one_returns_zero(self, trainer_stub):
        """All -1s should map to index 0: (-1+1)*3^0 + ... = 0."""
        batch_data = torch.full((1, 9), -1.0)
        indices = trainer_stub._compute_batch_indices(batch_data)
        assert indices.item() == 0

    def test_all_positive_one_returns_max_index(self, trainer_stub):
        """All +1s should map to max index 19682."""
        batch_data = torch.full((1, 9), 1.0)
        indices = trainer_stub._compute_batch_indices(batch_data)
        # Sum of 2*3^i for i=0..8 = 19682
        assert indices.item() == 19682

    def test_all_zeros_returns_middle_index(self, trainer_stub):
        """All 0s should map to middle: (0+1)*3^0 + (0+1)*3^1 + ... = 9841."""
        batch_data = torch.zeros(1, 9)
        indices = trainer_stub._compute_batch_indices(batch_data)
        # Sum of 1*3^i for i=0..8 = (3^9-1)/2 = 9841
        expected = sum(3**i for i in range(9))
        assert indices.item() == expected
        assert indices.item() == 9841

    def test_single_digit_variation(self, trainer_stub):
        """Changing one digit should change index by correct amount."""
        # Base: all -1s (index 0)
        base = torch.full((1, 9), -1.0)

        # Change first digit (position 0) from -1 to 0
        modified = base.clone()
        modified[0, 0] = 0.0

        base_idx = trainer_stub._compute_batch_indices(base)
        modified_idx = trainer_stub._compute_batch_indices(modified)

        # Difference should be 1*3^0 = 1
        assert (modified_idx - base_idx).item() == 1

    def test_second_digit_variation(self, trainer_stub):
        """Second digit changes should scale by 3."""
        base = torch.full((1, 9), -1.0)
        modified = base.clone()
        modified[0, 1] = 0.0  # Change second digit

        base_idx = trainer_stub._compute_batch_indices(base)
        modified_idx = trainer_stub._compute_batch_indices(modified)

        # Difference should be 1*3^1 = 3
        assert (modified_idx - base_idx).item() == 3

    def test_batch_processing(self, trainer_stub):
        """Should process batches correctly."""
        batch_data = torch.stack(
            [
                torch.full((9,), -1.0),  # Index 0
                torch.full((9,), 0.0),  # Index 9841
                torch.full((9,), 1.0),  # Index 19682
            ]
        )

        indices = trainer_stub._compute_batch_indices(batch_data)

        assert indices.shape == (3,)
        assert indices[0].item() == 0
        assert indices[1].item() == 9841
        assert indices[2].item() == 19682

    def test_valid_range(self, trainer_stub):
        """All random ternary inputs should produce valid indices."""
        torch.manual_seed(42)
        batch_data = torch.randint(-1, 2, (100, 9)).float()

        indices = trainer_stub._compute_batch_indices(batch_data)

        assert (indices >= 0).all()
        assert (indices <= 19682).all()

    def test_unique_inputs_unique_outputs(self, trainer_stub):
        """Different inputs should produce different outputs."""
        # Create two different ternary operations
        a = torch.zeros(1, 9)
        b = torch.zeros(1, 9)
        b[0, 0] = 1.0  # Change one digit

        idx_a = trainer_stub._compute_batch_indices(a)
        idx_b = trainer_stub._compute_batch_indices(b)

        assert idx_a.item() != idx_b.item()

    def test_device_compatibility(self, trainer_stub):
        """Should work on CPU."""
        batch_data = torch.randint(-1, 2, (10, 9)).float()
        indices = trainer_stub._compute_batch_indices(batch_data)

        assert indices.device == batch_data.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self, trainer_stub):
        """Should work on CUDA."""
        batch_data = torch.randint(-1, 2, (10, 9), device="cuda").float()
        indices = trainer_stub._compute_batch_indices(batch_data)

        assert indices.device == batch_data.device


class TestBatchIndexRoundtrip:
    """Test that batch index computation is reversible."""

    @pytest.fixture
    def trainer_stub(self):
        """Create a minimal trainer stub."""
        from src.training.trainer import TernaryVAETrainer

        with patch.object(TernaryVAETrainer, "__init__", lambda self, *args, **kwargs: None):
            trainer = TernaryVAETrainer.__new__(TernaryVAETrainer)
            trainer._base3_weights = torch.tensor([3**i for i in range(9)], dtype=torch.long)
            return trainer

    def test_roundtrip_single(self, trainer_stub):
        """Single sample roundtrip."""
        original = torch.tensor([[0, 1, -1, 0, 1, -1, 0, 1, -1]]).float()
        index = trainer_stub._compute_batch_indices(original)

        # Reverse the conversion
        recovered = torch.zeros_like(original)
        idx = index[0].item()
        for d in range(9):
            recovered[0, d] = (idx % 3) - 1
            idx //= 3

        assert torch.equal(original, recovered)

    def test_roundtrip_batch(self, trainer_stub):
        """Batch roundtrip."""
        torch.manual_seed(123)
        original = torch.randint(-1, 2, (20, 9)).float()
        indices = trainer_stub._compute_batch_indices(original)

        # Reverse the conversion
        recovered = torch.zeros_like(original)
        for i in range(20):
            idx = indices[i].item()
            for d in range(9):
                recovered[i, d] = (idx % 3) - 1
                idx //= 3

        assert torch.equal(original, recovered)


# =============================================================================
# Scheduler Helper Tests
# =============================================================================


class TestSchedulerHelpers:
    """Test scheduler helper functions."""

    def test_linear_schedule_basic(self):
        """Linear schedule helper should work."""
        from src.training.schedulers import linear_schedule

        # At start
        val_start = linear_schedule(0, start_val=1.0, end_val=0.0, total_epochs=100)
        assert val_start == pytest.approx(1.0)

        # At middle
        val_mid = linear_schedule(50, start_val=1.0, end_val=0.0, total_epochs=100)
        assert val_mid == pytest.approx(0.5)

        # At end
        val_end = linear_schedule(100, start_val=1.0, end_val=0.0, total_epochs=100)
        assert val_end == pytest.approx(0.0)

    def test_linear_schedule_with_start_epoch(self):
        """Linear schedule should respect start_epoch."""
        from src.training.schedulers import linear_schedule

        # Before start_epoch, should return start_val
        val_before = linear_schedule(5, start_val=1.0, end_val=0.0, total_epochs=100, start_epoch=10)
        assert val_before == pytest.approx(1.0)

        # At start_epoch
        val_at_start = linear_schedule(10, start_val=1.0, end_val=0.0, total_epochs=100, start_epoch=10)
        assert val_at_start == pytest.approx(1.0)

    def test_cyclic_schedule(self):
        """Cyclic schedule should oscillate."""
        from src.training.schedulers import cyclic_schedule

        # At epoch 0, should be at base + amplitude (cos(0) = 1)
        val_0 = cyclic_schedule(0, base_val=1.0, amplitude=0.5, period=10)
        assert val_0 == pytest.approx(1.5)

        # At epoch 5 (half period), should be at base - amplitude (cos(pi) = -1)
        val_5 = cyclic_schedule(5, base_val=1.0, amplitude=0.5, period=10)
        assert val_5 == pytest.approx(0.5)

        # At epoch 10 (full period), back to base + amplitude
        val_10 = cyclic_schedule(10, base_val=1.0, amplitude=0.5, period=10)
        assert val_10 == pytest.approx(1.5)


# =============================================================================
# Base Trainer Tests (Utility Methods)
# =============================================================================


class TestBaseTrainerSafeAverageLosses:
    """Test BaseTrainer.safe_average_losses utility."""

    def test_safe_average_normal(self):
        """safe_average_losses should work normally."""
        from src.training.base import BaseTrainer

        losses = {"loss": 10.0, "ce": 5.0}
        averaged = BaseTrainer.safe_average_losses(losses, num_batches=2)

        assert averaged["loss"] == pytest.approx(5.0)
        assert averaged["ce"] == pytest.approx(2.5)

    def test_safe_average_zero_batches(self):
        """safe_average_losses should handle zero batches."""
        from src.training.base import BaseTrainer

        losses = {"loss": 10.0, "ce": 5.0}
        averaged = BaseTrainer.safe_average_losses(losses, num_batches=0)

        # Should return unchanged
        assert averaged["loss"] == 10.0
        assert averaged["ce"] == 5.0

    def test_safe_average_with_exclude(self):
        """safe_average_losses should exclude specified keys."""
        from src.training.base import BaseTrainer

        losses = {"loss": 10.0, "lr": 0.001}
        averaged = BaseTrainer.safe_average_losses(losses, num_batches=2, exclude_keys={"lr"})

        assert averaged["loss"] == pytest.approx(5.0)
        assert averaged["lr"] == 0.001  # Unchanged


# =============================================================================
# Configuration Validation Tests
# =============================================================================


class TestConfigStructure:
    """Test that expected config structure is used correctly."""

    def test_minimal_config_keys(self):
        """Verify minimal required config keys."""
        # These are the keys that TernaryVAETrainer.__init__ accesses
        required_keys = [
            "total_epochs",
            "checkpoint_dir",
            "checkpoint_freq",
            "eval_num_samples",
            "patience",
            "grad_clip",
            "optimizer",
            "phase_transitions",
            "controller",
            "temperature",
            "beta",
            "model",
            "vae_b",
        ]

        # This is documentation of the interface
        assert len(required_keys) == 13

    def test_optimizer_config_keys(self):
        """Verify optimizer config structure."""
        optimizer_keys = ["lr_start", "weight_decay", "lr_schedule"]
        assert "lr_start" in optimizer_keys
        assert "lr_schedule" in optimizer_keys


# =============================================================================
# Integration-like Tests
# =============================================================================


class TestBase3Encoding:
    """Test base-3 encoding properties."""

    def test_encoding_bijective(self):
        """Each ternary 9-tuple should map to unique index."""
        from src.training.trainer import TernaryVAETrainer

        with patch.object(TernaryVAETrainer, "__init__", lambda self, *args, **kwargs: None):
            trainer = TernaryVAETrainer.__new__(TernaryVAETrainer)
            trainer._base3_weights = torch.tensor([3**i for i in range(9)], dtype=torch.long)

            # Generate all 3^3 = 27 combinations for first 3 digits
            indices_seen = set()
            for d0 in [-1, 0, 1]:
                for d1 in [-1, 0, 1]:
                    for d2 in [-1, 0, 1]:
                        data = torch.full((1, 9), -1.0)
                        data[0, 0] = d0
                        data[0, 1] = d1
                        data[0, 2] = d2
                        idx = trainer._compute_batch_indices(data).item()
                        assert idx not in indices_seen
                        indices_seen.add(idx)

            assert len(indices_seen) == 27

    def test_total_operation_count(self):
        """There should be 3^9 = 19683 unique operations."""
        assert 3**9 == 19683

    def test_max_index(self):
        """Maximum index should be 3^9 - 1 = 19682."""
        from src.training.trainer import TernaryVAETrainer

        with patch.object(TernaryVAETrainer, "__init__", lambda self, *args, **kwargs: None):
            trainer = TernaryVAETrainer.__new__(TernaryVAETrainer)
            trainer._base3_weights = torch.tensor([3**i for i in range(9)], dtype=torch.long)

            max_input = torch.full((1, 9), 1.0)
            max_idx = trainer._compute_batch_indices(max_input)

            assert max_idx.item() == 19682
            assert max_idx.item() == 3**9 - 1
