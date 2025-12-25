# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

import torch
import torch.nn as nn

from src.factories.model_factory import TernaryModelFactory
from src.models.ternary_vae import TernaryVAEV5_11
from tests.harnesses.model_harness import ModelTestHarness


def test_factory_creation():
    """Verify factory creating model components and full model."""
    config = {
        "latent_dim": 8,
        "hidden_dim": 16,
        "use_controller": True,
        "use_dual_projection": True,
    }

    # 1. Create components
    components = TernaryModelFactory.create_components(config)
    assert "encoder_A" in components
    assert "projection" in components
    assert components["controller"] is not None

    # 2. Create model via factory
    model = TernaryModelFactory.create_model(config)
    assert isinstance(model, TernaryVAEV5_11)
    assert model.latent_dim == 8
    assert model.use_dual_projection is True


def test_model_harness_verification():
    """Verify the model using the standard harness."""
    config = {
        "latent_dim": 16,
        "hidden_dim": 32,
        "use_controller": False,  # Simplify for gradient check
    }

    # Harness for TernaryVAEV5_11
    harness = ModelTestHarness(TernaryVAEV5_11, config)

    # 1. Init check
    model = harness.verify_initialization()
    assert isinstance(model, TernaryVAEV5_11)

    # 2. Forward pass shape check
    # Input: (Batch, 9)
    # Output keys: z_A_hyp, z_B_hyp, logits_A
    input_shape = (4, 9)
    expected_keys = ["z_A_hyp", "z_B_hyp", "logits_A"]
    harness.verify_forward_pass_shapes(input_shape, expected_keys)

    # 3. Gradient flow
    # Gradients should flow to projection (trainable)
    harness.verify_gradient_flow(loss_key=None)  # Uses default logic


def test_dependency_injection():
    """Verify we can inject mocks into the model."""

    # Mock components
    class MockEncoder(nn.Module):
        def forward(self, x):
            return torch.zeros(x.size(0), 16), torch.zeros(x.size(0), 16)

    mock_enc = MockEncoder()

    # Inject via kwargs
    model = TernaryVAEV5_11(latent_dim=16, encoder_A=mock_enc)

    # Verify injection
    assert model.encoder_A is mock_enc
    # Should work in forward pass
    x = torch.zeros(2, 9)
    out = model(x)
    assert out["mu_A"].sum() == 0
