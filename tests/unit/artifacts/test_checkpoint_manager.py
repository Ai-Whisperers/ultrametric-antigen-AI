# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for src/artifacts/checkpoint_manager.py - CheckpointManager and AsyncCheckpointSaver.

This module tests:
- Basic save/load operations
- State dict round-trip verification
- Edge cases (non-existent paths, corrupted checkpoints)
- Metadata handling
- Best checkpoint tracking
- Async checkpoint saving functionality
"""

import pickle
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.artifacts.checkpoint_manager import AsyncCheckpointSaver, CheckpointManager


# =============================================================================
# Fixtures
# =============================================================================


class SimpleModel(nn.Module):
    """Simple model for testing checkpoint save/load."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, output_dim: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def simple_model() -> SimpleModel:
    """Returns a simple model instance for testing."""
    torch.manual_seed(42)
    return SimpleModel()


@pytest.fixture
def simple_optimizer(simple_model: SimpleModel) -> torch.optim.Optimizer:
    """Returns an optimizer for the simple model."""
    return torch.optim.Adam(simple_model.parameters(), lr=0.001)


@pytest.fixture
def checkpoint_manager(tmp_path: Path) -> CheckpointManager:
    """Returns a CheckpointManager with async saving disabled for predictable testing."""
    manager = CheckpointManager(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_freq=5,
        async_save=False,
    )
    yield manager
    manager.shutdown()


@pytest.fixture
def async_checkpoint_manager(tmp_path: Path) -> CheckpointManager:
    """Returns a CheckpointManager with async saving enabled."""
    manager = CheckpointManager(
        checkpoint_dir=tmp_path / "checkpoints",
        checkpoint_freq=5,
        async_save=True,
    )
    yield manager
    manager.shutdown()


@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Returns sample metadata for checkpoint testing."""
    return {
        "loss": 0.5,
        "accuracy": 0.85,
        "lr": 0.001,
        "custom_field": "test_value",
    }


# =============================================================================
# Basic Save/Load Operations Tests
# =============================================================================


class TestBasicSaveLoad:
    """Tests for basic save and load operations."""

    def test_save_checkpoint_creates_latest_file(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Save should create latest.pt file."""
        checkpoint_manager.save_checkpoint(
            epoch=1,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={"loss": 0.5},
        )

        latest_path = checkpoint_manager.checkpoint_dir / "latest.pt"
        assert latest_path.exists()

    def test_save_checkpoint_creates_numbered_file_at_frequency(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Save should create numbered checkpoint at specified frequency."""
        # Checkpoint freq is 5, so epoch 5 should create epoch_5.pt
        checkpoint_manager.save_checkpoint(
            epoch=5,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={"loss": 0.3},
        )

        epoch_path = checkpoint_manager.checkpoint_dir / "epoch_5.pt"
        assert epoch_path.exists()

    def test_save_checkpoint_no_numbered_file_between_frequency(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Save should NOT create numbered checkpoint between frequency intervals."""
        checkpoint_manager.save_checkpoint(
            epoch=3,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={"loss": 0.4},
        )

        epoch_path = checkpoint_manager.checkpoint_dir / "epoch_3.pt"
        assert not epoch_path.exists()

    def test_load_checkpoint_restores_model_state(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Load should restore model state correctly."""
        # Save original state
        original_state = {k: v.clone() for k, v in simple_model.state_dict().items()}

        checkpoint_manager.save_checkpoint(
            epoch=1,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={"loss": 0.5},
        )

        # Modify model weights
        with torch.no_grad():
            for param in simple_model.parameters():
                param.fill_(999.0)

        # Load checkpoint
        checkpoint_manager.load_checkpoint(
            model=simple_model,
            optimizer=simple_optimizer,
            checkpoint_name="latest",
            device="cpu",
        )

        # Verify state restored
        for key, original_value in original_state.items():
            restored_value = simple_model.state_dict()[key]
            assert torch.equal(original_value, restored_value), f"Mismatch in {key}"

    def test_load_checkpoint_restores_optimizer_state(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
    ):
        """Load should restore optimizer state correctly."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)

        # Perform some steps to create optimizer state
        x = torch.randn(4, 10)
        for _ in range(3):
            loss = simple_model(x).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save checkpoint
        checkpoint_manager.save_checkpoint(
            epoch=1,
            model=simple_model,
            optimizer=optimizer,
            metadata={"loss": 0.5},
        )

        # Create new optimizer with different state
        new_optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)

        # Load checkpoint
        checkpoint_manager.load_checkpoint(
            model=simple_model,
            optimizer=new_optimizer,
            checkpoint_name="latest",
            device="cpu",
        )

        # Verify optimizer state has momentum buffers (from the steps we took)
        assert len(new_optimizer.state) > 0

    def test_state_dict_roundtrip(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Complete round-trip: save -> modify -> load should restore original state."""
        # Capture original state
        original_model_state = {k: v.clone() for k, v in simple_model.state_dict().items()}

        # Save
        checkpoint_manager.save_checkpoint(
            epoch=10,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={"loss": 0.1},
        )

        # Completely reinitialize model
        new_model = SimpleModel()

        # Load into new model
        checkpoint = checkpoint_manager.load_checkpoint(
            model=new_model,
            checkpoint_name="latest",
            device="cpu",
        )

        # Verify complete restoration
        for key, original_value in original_model_state.items():
            restored_value = new_model.state_dict()[key]
            assert torch.allclose(original_value, restored_value), f"Mismatch in {key}"

        # Verify epoch preserved
        assert checkpoint["epoch"] == 10


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_load_nonexistent_checkpoint_raises_file_not_found(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
    ):
        """Loading non-existent checkpoint should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_checkpoint(
                model=simple_model,
                checkpoint_name="nonexistent",
                device="cpu",
            )

    def test_load_latest_when_none_exists_raises_file_not_found(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
    ):
        """Loading 'latest' when no checkpoints exist should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_checkpoint(
                model=simple_model,
                checkpoint_name="latest",
                device="cpu",
            )

    def test_load_best_when_none_exists_raises_file_not_found(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
    ):
        """Loading 'best' when no best checkpoint exists should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            checkpoint_manager.load_checkpoint(
                model=simple_model,
                checkpoint_name="best",
                device="cpu",
            )

    def test_corrupted_checkpoint_handling(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
    ):
        """Loading a corrupted checkpoint file should raise an error."""
        # Create a corrupted checkpoint file
        corrupted_path = checkpoint_manager.checkpoint_dir / "corrupted.pt"
        with open(corrupted_path, "wb") as f:
            f.write(b"not a valid pytorch checkpoint")

        with pytest.raises(Exception):  # torch.load raises various exceptions for corrupted files
            checkpoint_manager.load_checkpoint(
                model=simple_model,
                checkpoint_name="corrupted",
                device="cpu",
            )

    def test_checkpoint_with_missing_model_key(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
    ):
        """Loading checkpoint without 'model' key should raise KeyError."""
        # Create invalid checkpoint
        invalid_path = checkpoint_manager.checkpoint_dir / "invalid.pt"
        torch.save({"epoch": 1, "optimizer": {}}, invalid_path)

        with pytest.raises(KeyError):
            checkpoint_manager.load_checkpoint(
                model=simple_model,
                checkpoint_name="invalid",
                device="cpu",
            )

    def test_checkpoint_manager_creates_directory(self, tmp_path: Path):
        """CheckpointManager should create directory if it doesn't exist."""
        new_dir = tmp_path / "new_checkpoint_dir"
        assert not new_dir.exists()

        manager = CheckpointManager(checkpoint_dir=new_dir, async_save=False)
        manager.shutdown()

        assert new_dir.exists()

    def test_load_without_optimizer(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Loading checkpoint without providing optimizer should work."""
        checkpoint_manager.save_checkpoint(
            epoch=1,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={"loss": 0.5},
        )

        # Load without optimizer - should not raise
        checkpoint = checkpoint_manager.load_checkpoint(
            model=simple_model,
            optimizer=None,  # Explicitly None
            checkpoint_name="latest",
            device="cpu",
        )

        assert checkpoint["epoch"] == 1


# =============================================================================
# Metadata Handling Tests
# =============================================================================


class TestMetadataHandling:
    """Tests for metadata save and load functionality."""

    def test_save_with_custom_metadata(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
        sample_metadata: Dict[str, Any],
    ):
        """Checkpoint should include custom metadata."""
        checkpoint_manager.save_checkpoint(
            epoch=1,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata=sample_metadata,
        )

        # Load and verify metadata
        checkpoint = checkpoint_manager.load_checkpoint(
            model=simple_model,
            checkpoint_name="latest",
            device="cpu",
        )

        assert checkpoint["loss"] == sample_metadata["loss"]
        assert checkpoint["accuracy"] == sample_metadata["accuracy"]
        assert checkpoint["lr"] == sample_metadata["lr"]
        assert checkpoint["custom_field"] == sample_metadata["custom_field"]

    def test_load_returns_complete_metadata(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Load should return complete checkpoint dict including epoch and metadata."""
        metadata = {"val_loss": 0.3, "train_loss": 0.2}

        checkpoint_manager.save_checkpoint(
            epoch=5,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata=metadata,
        )

        result = checkpoint_manager.load_checkpoint(
            model=simple_model,
            checkpoint_name="latest",
            device="cpu",
        )

        # Should have all expected keys
        assert "epoch" in result
        assert "model" in result
        assert "optimizer" in result
        assert "val_loss" in result
        assert "train_loss" in result
        assert result["epoch"] == 5

    def test_metadata_with_nested_dict(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Checkpoint should preserve nested dictionary metadata."""
        nested_metadata = {
            "metrics": {
                "train": {"loss": 0.2, "acc": 0.9},
                "val": {"loss": 0.3, "acc": 0.85},
            },
            "hyperparams": {"lr": 0.001, "batch_size": 32},
        }

        checkpoint_manager.save_checkpoint(
            epoch=1,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata=nested_metadata,
        )

        checkpoint = checkpoint_manager.load_checkpoint(
            model=simple_model,
            checkpoint_name="latest",
            device="cpu",
        )

        assert checkpoint["metrics"]["train"]["loss"] == 0.2
        assert checkpoint["hyperparams"]["batch_size"] == 32

    def test_metadata_with_tensor_values(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Checkpoint should preserve tensor values in metadata."""
        tensor_metadata = {
            "curvature": torch.tensor(1.5),
            "temperature": torch.tensor(0.8),
        }

        checkpoint_manager.save_checkpoint(
            epoch=1,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata=tensor_metadata,
        )

        checkpoint = checkpoint_manager.load_checkpoint(
            model=simple_model,
            checkpoint_name="latest",
            device="cpu",
        )

        assert torch.equal(checkpoint["curvature"], torch.tensor(1.5))
        assert torch.equal(checkpoint["temperature"], torch.tensor(0.8))


# =============================================================================
# Best Checkpoint Tracking Tests
# =============================================================================


class TestBestCheckpointTracking:
    """Tests for best checkpoint save functionality."""

    def test_save_best_creates_best_file(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Saving with is_best=True should create best.pt file."""
        checkpoint_manager.save_checkpoint(
            epoch=1,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={"loss": 0.5},
            is_best=True,
        )

        best_path = checkpoint_manager.checkpoint_dir / "best.pt"
        assert best_path.exists()

    def test_save_not_best_does_not_overwrite_best(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Saving with is_best=False should not overwrite existing best.pt."""
        # Save best
        checkpoint_manager.save_checkpoint(
            epoch=1,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={"loss": 0.3},
            is_best=True,
        )

        # Get best file modification time
        best_path = checkpoint_manager.checkpoint_dir / "best.pt"
        original_mtime = best_path.stat().st_mtime

        # Small delay to ensure mtime would be different
        time.sleep(0.01)

        # Save non-best checkpoint
        checkpoint_manager.save_checkpoint(
            epoch=2,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={"loss": 0.4},
            is_best=False,
        )

        # Best file should not have been modified
        new_mtime = best_path.stat().st_mtime
        assert new_mtime == original_mtime

    def test_overwrite_best_when_improvement(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Saving new best should overwrite previous best.pt."""
        # Save initial best
        checkpoint_manager.save_checkpoint(
            epoch=1,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={"loss": 0.5},
            is_best=True,
        )

        # Save new best
        checkpoint_manager.save_checkpoint(
            epoch=5,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={"loss": 0.2},
            is_best=True,
        )

        # Load best and verify it's the newer one
        checkpoint = checkpoint_manager.load_checkpoint(
            model=simple_model,
            checkpoint_name="best",
            device="cpu",
        )

        assert checkpoint["epoch"] == 5
        assert checkpoint["loss"] == 0.2

    def test_best_and_latest_separate(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Best and latest checkpoints should be independent."""
        # Save best at epoch 3
        checkpoint_manager.save_checkpoint(
            epoch=3,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={"loss": 0.2},
            is_best=True,
        )

        # Save non-best at epoch 10
        checkpoint_manager.save_checkpoint(
            epoch=10,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={"loss": 0.4},
            is_best=False,
        )

        # Load both
        best = checkpoint_manager.load_checkpoint(
            model=simple_model,
            checkpoint_name="best",
            device="cpu",
        )

        latest = checkpoint_manager.load_checkpoint(
            model=simple_model,
            checkpoint_name="latest",
            device="cpu",
        )

        # They should be different
        assert best["epoch"] == 3
        assert latest["epoch"] == 10


# =============================================================================
# List Checkpoints Tests
# =============================================================================


class TestListCheckpoints:
    """Tests for list_checkpoints functionality."""

    def test_list_empty_directory(self, checkpoint_manager: CheckpointManager):
        """List should return empty lists for empty directory."""
        result = checkpoint_manager.list_checkpoints()

        assert result["special"] == []
        assert result["epochs"] == []

    def test_list_with_special_checkpoints(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """List should return special checkpoints (latest, best)."""
        checkpoint_manager.save_checkpoint(
            epoch=1,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={},
            is_best=True,
        )

        result = checkpoint_manager.list_checkpoints()

        assert "latest" in result["special"]
        assert "best" in result["special"]

    def test_list_with_numbered_checkpoints(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """List should return numbered checkpoints in sorted order."""
        # Save at frequency intervals (5, 10, 15)
        for epoch in [5, 10, 15]:
            checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=simple_model,
                optimizer=simple_optimizer,
                metadata={},
            )

        result = checkpoint_manager.list_checkpoints()

        assert result["epochs"] == ["epoch_5", "epoch_10", "epoch_15"]


# =============================================================================
# Get Latest Epoch Tests
# =============================================================================


class TestGetLatestEpoch:
    """Tests for get_latest_epoch functionality."""

    def test_get_latest_epoch_no_checkpoint(self, checkpoint_manager: CheckpointManager):
        """Should return None when no checkpoint exists."""
        result = checkpoint_manager.get_latest_epoch()
        assert result is None

    def test_get_latest_epoch_with_checkpoint(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Should return correct epoch number."""
        checkpoint_manager.save_checkpoint(
            epoch=42,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={},
        )

        result = checkpoint_manager.get_latest_epoch()
        assert result == 42


# =============================================================================
# Async Checkpoint Saver Tests
# =============================================================================


class TestAsyncCheckpointSaver:
    """Tests for AsyncCheckpointSaver functionality."""

    def test_async_saver_initialization(self):
        """AsyncCheckpointSaver should initialize correctly."""
        saver = AsyncCheckpointSaver(max_queue_size=3)
        assert saver.pending_saves == 0
        assert saver.saves_completed == 0
        saver.shutdown()

    def test_async_save_completes(self, tmp_path: Path):
        """Async save should complete successfully."""
        saver = AsyncCheckpointSaver()
        checkpoint = {"data": torch.randn(10, 10)}
        path = tmp_path / "test.pt"

        saver.save_async(checkpoint, path)

        # Wait for save to complete
        saver.shutdown(timeout=5.0)

        assert path.exists()
        assert saver.saves_completed >= 1

    def test_async_save_preserves_data(self, tmp_path: Path):
        """Async saved checkpoint should be loadable with correct data."""
        saver = AsyncCheckpointSaver()
        original_tensor = torch.randn(5, 5)
        checkpoint = {"tensor": original_tensor.clone()}
        path = tmp_path / "test.pt"

        saver.save_async(checkpoint, path)
        saver.shutdown(timeout=5.0)

        loaded = torch.load(path, weights_only=False)
        assert torch.equal(loaded["tensor"], original_tensor)

    def test_async_saver_multiple_saves(self, tmp_path: Path):
        """Multiple async saves should all complete."""
        saver = AsyncCheckpointSaver(max_queue_size=5)

        for i in range(3):
            checkpoint = {"epoch": i}
            path = tmp_path / f"checkpoint_{i}.pt"
            saver.save_async(checkpoint, path)

        saver.shutdown(timeout=10.0)

        for i in range(3):
            path = tmp_path / f"checkpoint_{i}.pt"
            assert path.exists()
            loaded = torch.load(path, weights_only=False)
            assert loaded["epoch"] == i

    def test_async_saver_deep_copies_state_dicts(self, tmp_path: Path):
        """Async saver should deep copy nested dicts (state_dicts) to avoid race conditions."""
        saver = AsyncCheckpointSaver()
        # State dicts are nested dicts, which are deep copied
        state_dict = {"weight": torch.ones(5, 5), "bias": torch.zeros(5)}
        checkpoint = {"model": state_dict}
        path = tmp_path / "test.pt"

        saver.save_async(checkpoint, path)

        # Modify original tensor within the state dict
        state_dict["weight"].fill_(999.0)

        saver.shutdown(timeout=5.0)

        # Loaded checkpoint should have original values (1s) because dict values are deep copied
        loaded = torch.load(path, weights_only=False)
        assert torch.all(loaded["model"]["weight"] == 1.0)


class TestAsyncCheckpointManager:
    """Tests for CheckpointManager with async saving enabled."""

    def test_async_manager_saves_checkpoint(
        self,
        async_checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Async checkpoint manager should save checkpoints."""
        async_checkpoint_manager.save_checkpoint(
            epoch=1,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={"loss": 0.5},
        )

        # Shutdown to wait for pending saves
        async_checkpoint_manager.shutdown()

        latest_path = async_checkpoint_manager.checkpoint_dir / "latest.pt"
        assert latest_path.exists()

    def test_async_manager_saves_best(
        self,
        async_checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Async checkpoint manager should save best checkpoint."""
        async_checkpoint_manager.save_checkpoint(
            epoch=1,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={"loss": 0.5},
            is_best=True,
        )

        async_checkpoint_manager.shutdown()

        best_path = async_checkpoint_manager.checkpoint_dir / "best.pt"
        assert best_path.exists()


# =============================================================================
# Device Compatibility Tests
# =============================================================================


class TestDeviceCompatibility:
    """Tests for device compatibility (CPU/CUDA)."""

    def test_load_to_cpu(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Checkpoint should be loadable to CPU."""
        checkpoint_manager.save_checkpoint(
            epoch=1,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={},
        )

        new_model = SimpleModel()
        checkpoint_manager.load_checkpoint(
            model=new_model,
            checkpoint_name="latest",
            device="cpu",
        )

        # Verify all parameters are on CPU
        for param in new_model.parameters():
            assert param.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_load_to_cuda(
        self,
        checkpoint_manager: CheckpointManager,
        simple_model: SimpleModel,
        simple_optimizer: torch.optim.Optimizer,
    ):
        """Checkpoint should be loadable to CUDA."""
        checkpoint_manager.save_checkpoint(
            epoch=1,
            model=simple_model,
            optimizer=simple_optimizer,
            metadata={},
        )

        new_model = SimpleModel().cuda()
        checkpoint_manager.load_checkpoint(
            model=new_model,
            checkpoint_name="latest",
            device="cuda",
        )

        # Verify all parameters are on CUDA
        for param in new_model.parameters():
            assert param.device.type == "cuda"
