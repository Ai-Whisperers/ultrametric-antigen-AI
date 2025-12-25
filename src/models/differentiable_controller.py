# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Differentiable Controller for V5.11.

FIXES THE V5.10 STATENET GRADIENT FLOW PROBLEM.

The core problem in v5.10:
```python
optimizer.lr = lr * (1 + delta_lr.item())  # .item() breaks gradient chain!
self.lambda1 = self.lambda1 + delta.item()  # Python float, no gradients
```

The V5.11 solution: Move control signals INTO the forward pass as tensor operations.

Instead of modifying external hyperparameters, controller outputs become
MULTIPLIERS within the loss computation:

```python
# V5.10 (broken):
self.lambda1 = self.lambda1 + delta_lambda1.item()  # No gradient
total_loss = lambda1 * loss_A + ...  # lambda1 is Python float

# V5.11 (fixed):
control = self.controller(batch_features)  # Tensor output
weight_A = F.softplus(control['weight_recon'])  # Tensor, bounded positive
total_loss = weight_A * loss_A + ...  # Gradient flows through weight_A!
```

Single responsibility: Differentiable training dynamics control.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableController(nn.Module):
    """Lightweight controller that learns to modulate training dynamics.

    Key difference from StateNet: ALL outputs participate in tensor
    operations, so gradients flow back and the controller learns.

    Input: Batch statistics (computed as tensors, gradients intact)
    Output: Control signals (all tensors, gradients flow)

    The controller learns: "what weights minimize total_loss over time?"
    Gradients flow: loss → weights → controller → controller params
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        output_bounds: Optional[Dict[str, tuple]] = None,
    ):
        """Initialize DifferentiableController.

        Args:
            input_dim: Dimension of batch statistics input
            hidden_dim: Hidden layer dimension
            output_bounds: Dict of {name: (min, max)} for output clamping
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Small network - we want this to be lightweight
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),  # 6 control signals
        )

        # Output indices and their meanings:
        # 0: rho (cross-injection strength) [0, 0.5]
        # 1: weight_geodesic (geodesic loss importance) [0.1, inf)
        # 2: weight_radial (radial loss importance) [0, inf)
        # 3: beta_A (VAE-A KL weight) [0.5, inf)
        # 4: beta_B (VAE-B KL weight) [0.5, inf)
        # 5: tau (curriculum position) [0, 1]

        self.output_bounds = output_bounds or {
            "rho": (0.0, 0.5),
            "weight_geodesic": (0.1, 10.0),
            "weight_radial": (0.0, 5.0),
            "beta_A": (0.5, 5.0),
            "beta_B": (0.5, 5.0),
            "tau": (0.0, 1.0),
        }

        # Initialize for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize for stable starting values."""
        with torch.no_grad():
            # Initialize final layer to output near-zero (sigmoid/softplus will center)
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, batch_stats: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute control signals from batch statistics.

        CRITICAL: All outputs are tensors. Gradients flow through every output.

        Args:
            batch_stats: Tensor of batch statistics
                [mean_radius_A, mean_radius_B, H_A, H_B, kl_A, kl_B, geo_loss, rad_loss]
                All should be detached scalars converted to tensor, OR
                computed as part of forward pass with gradients.

        Returns:
            Dict of control signals (all tensors, gradients flow)
        """
        # Ensure batch dimension
        if batch_stats.dim() == 1:
            batch_stats = batch_stats.unsqueeze(0)

        raw = self.net(batch_stats)

        # Apply bounded activations (all tensors, all differentiable)
        return {
            "rho": torch.sigmoid(raw[:, 0]) * 0.5,  # [0, 0.5]
            "weight_geodesic": F.softplus(raw[:, 1]) + 0.1,  # [0.1, inf)
            "weight_radial": F.softplus(raw[:, 2]),  # [0, inf)
            "beta_A": F.softplus(raw[:, 3]) + 0.5,  # [0.5, inf)
            "beta_B": F.softplus(raw[:, 4]) + 0.5,  # [0.5, inf)
            "tau": torch.sigmoid(raw[:, 5]),  # [0, 1]
        }


class ThreeBodyController(nn.Module):
    """Three-Body Controller for VAE-A/VAE-B opposition dynamics.

    Extends DifferentiableController with position-aware control.

    In hyperbolic space, the "right" learning dynamics vary with position:
    - Near origin: stable, exploit (low exploration)
    - Near boundary: unstable, explore (high exploration)

    The controller sees position features and outputs local control signals.
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 32):
        """Initialize ThreeBodyController.

        Args:
            latent_dim: Dimension of latent space
            hidden_dim: Hidden dimension for controller
        """
        super().__init__()

        # Position encoder: extracts features from mean embeddings
        self.position_encoder = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 8),  # Compressed position features
        )

        # Statistics encoder: processes batch-level statistics
        self.stats_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 4),  # Compressed stats
        )

        # Controller head: combines position + stats → control signals
        self.controller_head = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize for stability."""
        with torch.no_grad():
            self.controller_head[-1].weight.mul_(0.01)
            self.controller_head[-1].bias.zero_()

    def forward(
        self,
        z_A_mean: torch.Tensor,
        z_B_mean: torch.Tensor,
        batch_stats: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute position-aware control signals.

        Args:
            z_A_mean: Mean of VAE-A embeddings (1, latent_dim)
            z_B_mean: Mean of VAE-B embeddings (1, latent_dim)
            batch_stats: Batch statistics [H_A, H_B, kl_A, kl_B, geo_loss, rad_loss]

        Returns:
            Dict of control signals (all tensors, gradients flow)
        """
        # Ensure proper shapes
        if z_A_mean.dim() == 1:
            z_A_mean = z_A_mean.unsqueeze(0)
        if z_B_mean.dim() == 1:
            z_B_mean = z_B_mean.unsqueeze(0)
        if batch_stats.dim() == 1:
            batch_stats = batch_stats.unsqueeze(0)

        # Encode position information
        position_input = torch.cat([z_A_mean, z_B_mean], dim=-1)
        position_features = self.position_encoder(position_input)

        # Encode batch statistics
        stats_features = self.stats_encoder(batch_stats)

        # Combine and produce control signals
        combined = torch.cat([position_features, stats_features], dim=-1)
        raw = self.controller_head(combined)

        return {
            "rho": torch.sigmoid(raw[:, 0]) * 0.5,
            "weight_geodesic": F.softplus(raw[:, 1]) + 0.1,
            "weight_radial": F.softplus(raw[:, 2]),
            "beta_A": F.softplus(raw[:, 3]) + 0.5,
            "beta_B": F.softplus(raw[:, 4]) + 0.5,
            "tau": torch.sigmoid(raw[:, 5]),
        }


class PositionDependentControl(nn.Module):
    """Position-dependent control signal modulation.

    Given a base control signal and positions in hyperbolic space,
    modulates the control based on distance from origin.

    Near origin: reduce exploration (rho), increase stability
    Near boundary: increase exploration, allow more chaos
    """

    def __init__(self, sensitivity: float = 2.0):
        """Initialize PositionDependentControl.

        Args:
            sensitivity: How sensitive modulation is to radius
        """
        super().__init__()
        self.sensitivity = sensitivity

    def forward(self, base_control: Dict[str, torch.Tensor], z_hyp: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Modulate control based on position.

        Args:
            base_control: Base control signals from controller
            z_hyp: Points in hyperbolic space (batch, latent_dim)

        Returns:
            Modulated control signals
        """
        # Compute mean radius
        radius = torch.norm(z_hyp, dim=-1).mean()

        # Position factor: 0 at origin, 1 at boundary
        # Using tanh for smooth saturation
        position_factor = torch.tanh(radius * self.sensitivity)

        # Modulate controls based on position
        modulated = {}

        # Rho: higher at boundary (more exploration)
        modulated["rho"] = base_control["rho"] * (0.5 + 0.5 * position_factor)

        # Geodesic weight: higher at boundary (more structure enforcement)
        modulated["weight_geodesic"] = base_control["weight_geodesic"] * (0.5 + 0.5 * position_factor)

        # Radial weight: higher near origin (maintain hierarchy)
        modulated["weight_radial"] = base_control["weight_radial"] * (1.5 - 0.5 * position_factor)

        # Beta: higher at boundary (more regularization to control chaos)
        modulated["beta_A"] = base_control["beta_A"] * (0.8 + 0.4 * position_factor)
        modulated["beta_B"] = base_control["beta_B"] * (0.8 + 0.4 * position_factor)

        # Tau: pass through (curriculum is global, not position-dependent)
        modulated["tau"] = base_control["tau"]

        return modulated


__all__ = [
    "DifferentiableController",
    "ThreeBodyController",
    "PositionDependentControl",
]
