# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Loss-specific test fixtures.

Fixtures specialized for testing loss functions and components.
"""

import pytest
import torch
from typing import Dict


# =============================================================================
# VAE Output Fixtures for Loss Testing
# =============================================================================


@pytest.fixture
def loss_test_outputs(device) -> Dict[str, torch.Tensor]:
    """Standard VAE outputs for testing loss functions."""
    batch_size = 32
    latent_dim = 16

    torch.manual_seed(42)  # Reproducible

    return {
        'logits_A': torch.randn(batch_size, 9, 3, device=device),
        'logits_B': torch.randn(batch_size, 9, 3, device=device),
        'mu_A': torch.randn(batch_size, latent_dim, device=device),
        'mu_B': torch.randn(batch_size, latent_dim, device=device),
        'logvar_A': torch.randn(batch_size, latent_dim, device=device) - 1,  # Reasonable logvar
        'logvar_B': torch.randn(batch_size, latent_dim, device=device) - 1,
        'z_A': torch.randn(batch_size, latent_dim, device=device) * 0.5,
        'z_B': torch.randn(batch_size, latent_dim, device=device) * 0.5,
        'z_A_hyp': torch.randn(batch_size, latent_dim, device=device) * 0.3,
        'z_B_hyp': torch.randn(batch_size, latent_dim, device=device) * 0.3,
        'H_A': torch.tensor(2.0, device=device),
        'H_B': torch.tensor(2.1, device=device),
        'beta_A': torch.tensor(0.1, device=device),
        'beta_B': torch.tensor(0.1, device=device),
    }


@pytest.fixture
def minimal_loss_outputs(device) -> Dict[str, torch.Tensor]:
    """Minimal VAE outputs for quick tests."""
    batch_size = 4
    latent_dim = 8

    return {
        'logits_A': torch.randn(batch_size, 9, 3, device=device),
        'logits_B': torch.randn(batch_size, 9, 3, device=device),
        'mu_A': torch.randn(batch_size, latent_dim, device=device),
        'mu_B': torch.randn(batch_size, latent_dim, device=device),
        'logvar_A': torch.zeros(batch_size, latent_dim, device=device),
        'logvar_B': torch.zeros(batch_size, latent_dim, device=device),
        'z_A': torch.randn(batch_size, latent_dim, device=device) * 0.5,
        'z_B': torch.randn(batch_size, latent_dim, device=device) * 0.5,
        'H_A': torch.tensor(2.0, device=device),
        'H_B': torch.tensor(2.0, device=device),
        'beta_A': torch.tensor(1.0, device=device),
        'beta_B': torch.tensor(1.0, device=device),
    }


# =============================================================================
# Special Case Outputs
# =============================================================================


@pytest.fixture
def perfect_reconstruction_outputs(device) -> Dict[str, torch.Tensor]:
    """Outputs simulating perfect reconstruction."""
    batch_size = 4
    latent_dim = 8

    # Create logits that strongly predict specific classes
    logits = torch.zeros(batch_size, 9, 3, device=device)
    logits[:, :, 1] = 10.0  # Strongly predict class 1 (value 0)

    return {
        'logits_A': logits.clone(),
        'logits_B': logits.clone(),
        'mu_A': torch.zeros(batch_size, latent_dim, device=device),
        'mu_B': torch.zeros(batch_size, latent_dim, device=device),
        'logvar_A': torch.zeros(batch_size, latent_dim, device=device),
        'logvar_B': torch.zeros(batch_size, latent_dim, device=device),
        'z_A': torch.zeros(batch_size, latent_dim, device=device),
        'z_B': torch.zeros(batch_size, latent_dim, device=device),
        'H_A': torch.tensor(2.0, device=device),
        'H_B': torch.tensor(2.0, device=device),
        'beta_A': torch.tensor(1.0, device=device),
        'beta_B': torch.tensor(1.0, device=device),
    }


@pytest.fixture
def high_kl_outputs(device) -> Dict[str, torch.Tensor]:
    """Outputs with high KL divergence."""
    batch_size = 4
    latent_dim = 8

    return {
        'logits_A': torch.randn(batch_size, 9, 3, device=device),
        'logits_B': torch.randn(batch_size, 9, 3, device=device),
        'mu_A': torch.ones(batch_size, latent_dim, device=device) * 5,  # Far from prior
        'mu_B': torch.ones(batch_size, latent_dim, device=device) * 5,
        'logvar_A': torch.ones(batch_size, latent_dim, device=device) * 2,  # High variance
        'logvar_B': torch.ones(batch_size, latent_dim, device=device) * 2,
        'z_A': torch.randn(batch_size, latent_dim, device=device) * 2,
        'z_B': torch.randn(batch_size, latent_dim, device=device) * 2,
        'H_A': torch.tensor(2.0, device=device),
        'H_B': torch.tensor(2.0, device=device),
        'beta_A': torch.tensor(1.0, device=device),
        'beta_B': torch.tensor(1.0, device=device),
    }


# =============================================================================
# Batch Index Fixtures for P-Adic Testing
# =============================================================================


@pytest.fixture
def padic_test_data(device):
    """Test data with batch indices for p-adic loss testing."""
    batch_size = 32
    latent_dim = 16

    # Create ternary operations and their indices
    ops = torch.randint(-1, 2, (batch_size, 9), device=device).float()

    # Compute indices using base-3 encoding
    weights = torch.tensor([3**i for i in range(9)], device=device)
    indices = ((ops + 1) * weights.unsqueeze(0)).sum(dim=1).long()

    # Create corresponding latent codes
    z = torch.randn(batch_size, latent_dim, device=device) * 0.5

    return {
        'ops': ops,
        'indices': indices,
        'z': z,
    }


@pytest.fixture
def hierarchical_indices(device):
    """Indices with known 3-adic hierarchical structure."""
    # Indices with different valuation levels
    return torch.tensor([
        0,      # v_3(0) = inf (or 9)
        1,      # v_3(1) = 0
        3,      # v_3(3) = 1
        9,      # v_3(9) = 2
        27,     # v_3(27) = 3
        81,     # v_3(81) = 4
        243,    # v_3(243) = 5
        729,    # v_3(729) = 6
        2187,   # v_3(2187) = 7
        6561,   # v_3(6561) = 8
    ], device=device)
