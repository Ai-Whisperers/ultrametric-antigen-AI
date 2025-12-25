# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Root conftest.py - Shared fixtures for all tests.

This module provides the foundation fixtures used across all test suites.
Fixtures are organized by category and scope for optimal reuse.

Fixture Scopes:
    - session: Created once per test session (expensive resources)
    - module: Created once per test module
    - class: Created once per test class
    - function: Created fresh for each test (default)

Categories:
    - Device/Hardware: CPU/GPU device selection
    - Geometry: Poincaré manifold and hyperbolic utilities
    - Data: Ternary operations and batch generation
    - Models: Model configurations and instances
    - Training: Training utilities and mocks
"""

from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# =============================================================================
# Device / Hardware Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def device() -> str:
    """Returns 'cuda' if available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def cpu_device() -> str:
    """Always returns 'cpu' for tests that must run on CPU."""
    return "cpu"


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    """Returns whether CUDA is available."""
    return torch.cuda.is_available()


# =============================================================================
# Random Seed Fixtures
# =============================================================================


@pytest.fixture(autouse=False)
def seed_random():
    """Sets deterministic seeds for reproducibility.

    Use this fixture explicitly when test requires reproducibility.
    Not autouse to avoid affecting tests that need randomness.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    yield
    # No cleanup needed


@pytest.fixture
def random_generator():
    """Provides a seeded random generator for tests."""
    return torch.Generator().manual_seed(42)


# =============================================================================
# Geometry Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def poincare():
    """Returns a PoincaréBall manifold instance with default curvature."""
    from src.geometry.poincare import get_manifold

    return get_manifold(c=1.0)


@pytest.fixture(scope="session")
def poincare_c2():
    """Returns a PoincaréBall manifold with curvature c=2.0."""
    from src.geometry.poincare import get_manifold

    return get_manifold(c=2.0)


@pytest.fixture
def hyperbolic_point(device):
    """Returns a single valid point on the Poincaré disk."""
    # Point with norm < 1
    point = torch.tensor([[0.3, 0.4, 0.0] + [0.0] * 13], device=device)
    return point


@pytest.fixture
def hyperbolic_batch(device):
    """Returns a batch of valid points on the Poincaré disk."""
    batch_size = 32
    latent_dim = 16
    # Generate random points and normalize to be inside disk
    points = torch.randn(batch_size, latent_dim, device=device)
    norms = torch.norm(points, dim=-1, keepdim=True)
    # Scale to have max norm of 0.9
    points = points / (norms + 1e-8) * 0.9 * torch.rand(batch_size, 1, device=device)
    return points


# =============================================================================
# Ternary Data Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def ternary_space():
    """Returns the TERNARY singleton for 3-adic operations."""
    from src.core.ternary import TERNARY

    return TERNARY


@pytest.fixture
def ternary_ops(device) -> torch.Tensor:
    """Returns a batch of valid ternary operations for testing.

    Shape: (32, 9)
    Values: {-1, 0, 1}
    """
    return torch.randint(-1, 2, (32, 9), device=device).float()


@pytest.fixture
def ternary_ops_small(device) -> torch.Tensor:
    """Returns a small batch of ternary operations (4 samples)."""
    return torch.randint(-1, 2, (4, 9), device=device).float()


@pytest.fixture
def ternary_ops_large(device) -> torch.Tensor:
    """Returns a large batch of ternary operations (256 samples)."""
    return torch.randint(-1, 2, (256, 9), device=device).float()


@pytest.fixture
def all_ternary_ops(cpu_device) -> torch.Tensor:
    """Returns all 19,683 ternary operations (CPU only due to size)."""
    from src.data.generation import generate_all_ternary_operations

    ops = generate_all_ternary_operations()
    return torch.tensor(ops, dtype=torch.float32, device=cpu_device)


@pytest.fixture
def batch_indices(ternary_ops, ternary_space) -> torch.Tensor:
    """Returns operation indices for a batch of ternary operations."""
    # Convert ternary ops to indices
    indices = ternary_space.ops_to_indices(ternary_ops)
    return indices


@pytest.fixture
def special_ternary_ops(device) -> Dict[str, torch.Tensor]:
    """Returns special ternary operations for edge case testing."""
    return {
        "all_negative": torch.full((1, 9), -1.0, device=device),
        "all_zero": torch.zeros((1, 9), device=device),
        "all_positive": torch.full((1, 9), 1.0, device=device),
        "mixed": torch.tensor([[-1, 0, 1, -1, 0, 1, -1, 0, 1]], device=device).float(),
        "identity_like": torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]], device=device).float(),
    }


# =============================================================================
# Model Configuration Fixtures
# =============================================================================


@pytest.fixture
def minimal_model_config() -> Dict[str, Any]:
    """Returns minimal model config for fast unit tests."""
    return {
        "latent_dim": 8,
        "hidden_dim": 16,
        "max_radius": 0.95,
        "curvature": 1.0,
        "use_controller": False,
        "use_dual_projection": False,
        "n_projection_layers": 1,
        "projection_dropout": 0.0,
        "learnable_curvature": False,
    }


@pytest.fixture
def standard_model_config() -> Dict[str, Any]:
    """Returns standard model config for integration tests."""
    return {
        "latent_dim": 16,
        "hidden_dim": 32,
        "max_radius": 0.95,
        "curvature": 1.0,
        "use_controller": True,
        "use_dual_projection": False,
        "n_projection_layers": 2,
        "projection_dropout": 0.1,
        "learnable_curvature": False,
    }


@pytest.fixture
def full_model_config() -> Dict[str, Any]:
    """Returns full model config with all features enabled."""
    return {
        "latent_dim": 16,
        "hidden_dim": 64,
        "max_radius": 0.95,
        "curvature": 2.0,
        "use_controller": True,
        "use_dual_projection": True,
        "n_projection_layers": 3,
        "projection_dropout": 0.1,
        "learnable_curvature": True,
    }


# =============================================================================
# Loss Configuration Fixtures
# =============================================================================


@pytest.fixture
def basic_loss_config() -> Dict[str, Any]:
    """Returns basic loss configuration without p-adic losses."""
    return {
        "free_bits": 0.0,
        "repulsion_sigma": 0.5,
    }


@pytest.fixture
def padic_loss_config() -> Dict[str, Any]:
    """Returns loss configuration with p-adic losses enabled."""
    return {
        "enable_metric_loss": True,
        "metric_loss_weight": 0.1,
        "metric_loss_scale": 1.0,
        "metric_n_pairs": 100,
        "enable_ranking_loss": True,
        "ranking_loss_weight": 0.5,
        "ranking_margin": 0.1,
        "ranking_n_triplets": 50,
    }


@pytest.fixture
def hyperbolic_loss_config() -> Dict[str, Any]:
    """Returns loss configuration with hyperbolic losses enabled."""
    return {
        "enable_ranking_loss_hyperbolic": True,
        "ranking_hyperbolic": {
            "base_margin": 0.05,
            "margin_scale": 0.15,
            "n_triplets": 50,
            "hard_negative_ratio": 0.5,
            "curvature": 1.0,
            "radial_weight": 0.1,
            "max_norm": 0.95,
            "weight": 0.5,
        },
    }


# =============================================================================
# Training Configuration Fixtures
# =============================================================================


@pytest.fixture
def minimal_training_config(tmp_path) -> Dict[str, Any]:
    """Returns minimal training configuration for fast tests."""
    return {
        "total_epochs": 2,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "checkpoint_freq": 1,
        "eval_num_samples": 100,
        "patience": 5,
        "grad_clip": 1.0,
        "optimizer": {
            "lr_start": 1e-3,
            "weight_decay": 0.0,
            "lr_schedule": "constant",
        },
        "phase_transitions": {},
        "controller": {"enabled": False},
        "temperature": {"initial": 1.0, "final": 0.5},
        "beta": {"initial": 0.01, "final": 1.0},
        "model": {
            "latent_dim": 8,
            "hidden_dim": 16,
        },
        "vae_b": {"entropy_weight": 0.01, "repulsion_weight": 0.01},
    }


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_model():
    """Returns a mock model for testing training utilities."""
    model = MagicMock()
    model.parameters = lambda: iter([torch.nn.Parameter(torch.randn(10, 10))])
    model.train = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_optimizer():
    """Returns a mock optimizer."""
    optimizer = MagicMock()
    optimizer.zero_grad = MagicMock()
    optimizer.step = MagicMock()
    return optimizer


@pytest.fixture
def mock_dataloader(ternary_ops):
    """Returns a mock dataloader that yields ternary operations."""

    class MockDataLoader:
        def __init__(self, data):
            self.data = data

        def __iter__(self):
            # Yield 2 batches
            half = len(self.data) // 2
            yield self.data[:half]
            yield self.data[half:]

        def __len__(self):
            return 2

    return MockDataLoader(ternary_ops)


# =============================================================================
# Model Output Fixtures
# =============================================================================


@pytest.fixture
def vae_outputs(device) -> Dict[str, torch.Tensor]:
    """Returns mock VAE outputs for loss testing."""
    batch_size = 32
    latent_dim = 16

    return {
        "logits_A": torch.randn(batch_size, 9, 3, device=device),
        "logits_B": torch.randn(batch_size, 9, 3, device=device),
        "mu_A": torch.randn(batch_size, latent_dim, device=device),
        "mu_B": torch.randn(batch_size, latent_dim, device=device),
        "logvar_A": torch.randn(batch_size, latent_dim, device=device),
        "logvar_B": torch.randn(batch_size, latent_dim, device=device),
        "z_A": torch.randn(batch_size, latent_dim, device=device) * 0.5,
        "z_B": torch.randn(batch_size, latent_dim, device=device) * 0.5,
        "z_A_hyp": torch.randn(batch_size, latent_dim, device=device) * 0.3,
        "z_B_hyp": torch.randn(batch_size, latent_dim, device=device) * 0.3,
        "H_A": torch.tensor(2.0, device=device),
        "H_B": torch.tensor(2.1, device=device),
        "beta_A": torch.tensor(0.1, device=device),
        "beta_B": torch.tensor(0.1, device=device),
    }


@pytest.fixture
def vae_outputs_small(device) -> Dict[str, torch.Tensor]:
    """Returns small batch VAE outputs (4 samples)."""
    batch_size = 4
    latent_dim = 16

    return {
        "logits_A": torch.randn(batch_size, 9, 3, device=device),
        "logits_B": torch.randn(batch_size, 9, 3, device=device),
        "mu_A": torch.randn(batch_size, latent_dim, device=device),
        "mu_B": torch.randn(batch_size, latent_dim, device=device),
        "logvar_A": torch.randn(batch_size, latent_dim, device=device),
        "logvar_B": torch.randn(batch_size, latent_dim, device=device),
        "z_A": torch.randn(batch_size, latent_dim, device=device) * 0.5,
        "z_B": torch.randn(batch_size, latent_dim, device=device) * 0.5,
        "H_A": torch.tensor(2.0, device=device),
        "H_B": torch.tensor(2.1, device=device),
        "beta_A": torch.tensor(0.1, device=device),
        "beta_B": torch.tensor(0.1, device=device),
    }


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def tolerance() -> Dict[str, float]:
    """Returns standard tolerances for numerical comparisons."""
    return {
        "atol": 1e-6,
        "rtol": 1e-5,
        "grad_atol": 1e-4,
    }


@pytest.fixture
def tensor_factory(device):
    """Factory for creating test tensors on the correct device."""

    class TensorFactory:
        @staticmethod
        def randn(*shape):
            return torch.randn(*shape, device=device)

        @staticmethod
        def zeros(*shape):
            return torch.zeros(*shape, device=device)

        @staticmethod
        def ones(*shape):
            return torch.ones(*shape, device=device)

        @staticmethod
        def randint(low, high, shape):
            return torch.randint(low, high, shape, device=device)

        @staticmethod
        def from_numpy(arr):
            return torch.from_numpy(arr).to(device)

    return TensorFactory()


# =============================================================================
# Cleanup Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Cleans up CUDA memory after each test if CUDA is available."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Markers Registration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "e2e: marks end-to-end tests")
    config.addinivalue_line("markers", "unit: marks unit tests")
