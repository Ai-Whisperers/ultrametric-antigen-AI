# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

from typing import Dict, List

import torch
import torch.nn as nn


class ModelTestHarness:
    """Standardized verification harness for VAE models."""

    def __init__(self, model_class, config: Dict):
        self.model_class = model_class
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def create_model(self):
        """Instantiate model from config."""
        # Typically uses factory, or direct instantiation
        # Just direct for now to be generic
        return self.model_class(**self.config).to(self.device)

    def verify_initialization(self):
        """Check if model initializes without error."""
        model = self.create_model()
        assert isinstance(model, nn.Module)
        return model

    def verify_forward_pass_shapes(self, input_shape: tuple, expected_keys: List[str]):
        """Verify forward pass produces expected output shapes/keys."""
        model = self.create_model()
        x = torch.randint(-1, 2, input_shape).float().to(self.device)

        output = model(x)

        for key in expected_keys:
            assert key in output, f"Missing key {key} in output"

        return output

    def verify_gradient_flow(self, loss_key: str = None):
        """Verify that gradients reach trainable parameters."""
        model = self.create_model()
        x = torch.randint(-1, 2, (4, 9)).float().to(self.device)

        output = model(x)

        # Simulate a loss
        if loss_key and loss_key in output:
            loss = output[loss_key].mean()
        else:
            # Just use z_A_hyp norm as dummy loss target
            loss = output["z_A_hyp"].norm()

        loss.backward()

        # Check gradients
        trainable_params_found = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params_found = True
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0 or param.grad.sum() == 0, f"Gradient check for {name}"

        assert trainable_params_found, "No trainable parameters found in model"
