# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

import torch
import torch.nn as nn

from src.geometry.poincare import poincare_distance


class DrugInteractionPenalty(nn.Module):
    """Contrastive loss for modeling drug-interaction constraints in hyperbolic space.

    This loss function encourages:
    - Interacting pairs (interaction=1) to be close in hyperbolic space.
    - Non-interacting pairs (interaction=0) to be separated by at least `margin`.

    This essentially embeds the "interaction graph" into the latent space.
    """

    def __init__(self, curvature: float = 1.0, margin: float = 1.0):
        super().__init__()
        self.curvature = curvature
        self.margin = margin

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, interaction: torch.Tensor) -> torch.Tensor:
        """Compute drug interaction penalty.

        Args:
            z1: Latent vectors for first item in pair (Batch, Dim)
            z2: Latent vectors for second item in pair (Batch, Dim)
            interaction: Boolean or float tensor (Batch,), 1 for interaction, 0 for non-interaction.

        Returns:
            Scalar loss
        """
        # Calculate hyperbolic distance
        # Calculate hyperbolic distance
        dists = poincare_distance(z1, z2, c=self.curvature)

        # Contrastive Loss:
        # L = y * d^2 + (1-y) * max(margin - d, 0)^2
        # Note: We use squared distance/margin for smoother gradients, standard in contrastive loss.

        loss_interaction = interaction * (dists**2)
        loss_no_interaction = (1 - interaction) * torch.relu(self.margin - dists).pow(2)

        return torch.mean(loss_interaction + loss_no_interaction)
