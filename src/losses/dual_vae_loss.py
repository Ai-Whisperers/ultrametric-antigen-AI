"""Loss functions for Dual-Neural VAE.

This module contains all loss computation separated from model architecture:
- Reconstruction loss (cross-entropy)
- KL divergence with free bits
- Entropy regularization
- Repulsion loss
- Entropy alignment
- Gradient normalization
- p-Adic metric loss (Phase 1A)
- p-Adic norm loss (Phase 1B)

Single responsibility: Loss computation only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .padic_losses import PAdicMetricLoss, PAdicRankingLoss, PAdicNormLoss


class ReconstructionLoss(nn.Module):
    """Cross-entropy reconstruction loss for ternary operations."""

    def forward(self, logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss.

        Args:
            logits: Model logits (batch_size, 9, 3)
            x: Input data (batch_size, 9) with values in {-1, 0, 1}

        Returns:
            Reconstruction loss (scalar)
        """
        batch_size = x.size(0)
        x_classes = (x + 1).long()  # Convert {-1, 0, 1} to {0, 1, 2}

        loss = F.cross_entropy(
            logits.view(-1, 3),
            x_classes.view(-1),
            reduction='sum'
        ) / batch_size

        return loss


class KLDivergenceLoss(nn.Module):
    """KL divergence with optional free bits.

    Free bits allows the first 'free_bits' nats of KL per dimension to be
    ignored, preventing posterior collapse while still regularizing.
    """

    def __init__(self, free_bits: float = 0.0):
        """Initialize KL divergence loss.

        Args:
            free_bits: Minimum KL per dimension (in nats) before penalty applies
        """
        super().__init__()
        self.free_bits = free_bits

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence KL(q(z|x) || p(z)).

        Args:
            mu: Mean of variational posterior
            logvar: Log variance of variational posterior

        Returns:
            KL divergence (scalar)
        """
        # Compute KL per dimension: -0.5 * (1 + logvar - mu^2 - exp(logvar))
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        if self.free_bits > 0:
            # Apply free bits: only penalize KL above threshold
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)

        # Sum over dimensions, average over batch
        kl = torch.sum(kl_per_dim) / mu.size(0)
        return kl


class EntropyRegularization(nn.Module):
    """Entropy regularization for output distribution."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute negative entropy of output distribution.

        Args:
            logits: Model logits (batch_size, 9, 3)

        Returns:
            Negative entropy (lower means more entropy/diversity)
        """
        probs = F.softmax(logits, dim=-1).mean(dim=0)  # Average over batch
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        return -entropy  # Return negative entropy as loss


class RepulsionLoss(nn.Module):
    """Repulsion loss to encourage diversity in latent space."""

    def __init__(self, sigma: float = 0.5):
        """Initialize repulsion loss.

        Args:
            sigma: Bandwidth for RBF kernel
        """
        super().__init__()
        self.sigma = sigma

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute repulsion loss.

        Args:
            z: Latent codes (batch_size, latent_dim)

        Returns:
            Repulsion loss (scalar)
        """
        if z.size(0) < 2:
            return torch.tensor(0.0, device=z.device)

        # Pairwise distances
        dists = torch.cdist(z, z, p=2)

        # Mask diagonal (self-distances)
        mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)

        # RBF kernel: penalize points that are too close
        repulsion = torch.exp(-dists[mask] ** 2 / (self.sigma ** 2)).mean()

        return repulsion


class DualVAELoss(nn.Module):
    """Complete loss for Dual-Neural VAE system.

    Combines:
    - Reconstruction losses for both VAEs
    - KL divergences with optional free bits
    - Entropy regularization for VAE-B
    - Repulsion loss for VAE-B
    - Entropy alignment between VAEs
    - Gradient normalization
    - p-Adic metric loss (Phase 1A) - aligns latent geometry to 3-adic metric
    - p-Adic norm loss (Phase 1B) - enforces MSB/LSB hierarchy
    """

    def __init__(
        self,
        free_bits: float = 0.0,
        repulsion_sigma: float = 0.5,
        padic_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize Dual VAE loss.

        Args:
            free_bits: Free bits per dimension for KL (0.0 = disabled)
            repulsion_sigma: Bandwidth for repulsion loss
            padic_config: Configuration for p-adic losses (from config['padic_losses'])
        """
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss()
        self.kl_loss = KLDivergenceLoss(free_bits)
        self.entropy_loss = EntropyRegularization()
        self.repulsion_loss = RepulsionLoss(repulsion_sigma)

        # p-Adic losses (Phase 1A/1B from implement.md)
        self.padic_config = padic_config or {}
        self.enable_metric_loss = self.padic_config.get('enable_metric_loss', False)
        self.enable_ranking_loss = self.padic_config.get('enable_ranking_loss', False)
        self.enable_norm_loss = self.padic_config.get('enable_norm_loss', False)

        # Phase 1A: Metric loss (MSE-based, for small scale mismatches)
        if self.enable_metric_loss:
            self.padic_metric_loss = PAdicMetricLoss(
                scale=self.padic_config.get('metric_loss_scale', 1.0),
                n_pairs=self.padic_config.get('metric_n_pairs', 1000)
            )
            self.metric_loss_weight = self.padic_config.get('metric_loss_weight', 0.1)

        # Phase 1A-alt: Ranking loss (triplet-based, for large scale mismatches)
        if self.enable_ranking_loss:
            self.padic_ranking_loss = PAdicRankingLoss(
                margin=self.padic_config.get('ranking_margin', 0.1),
                n_triplets=self.padic_config.get('ranking_n_triplets', 500)
            )
            self.ranking_loss_weight = self.padic_config.get('ranking_loss_weight', 0.5)

        # Phase 1B: Norm loss
        if self.enable_norm_loss:
            self.padic_norm_loss = PAdicNormLoss(latent_dim=16)
            self.norm_loss_weight = self.padic_config.get('norm_loss_weight', 0.05)

    def forward(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        lambda1: float,
        lambda2: float,
        lambda3: float,
        entropy_weight_B: float,
        repulsion_weight_B: float,
        grad_norm_A_ema: torch.Tensor,
        grad_norm_B_ema: torch.Tensor,
        gradient_balance: bool,
        training: bool,
        batch_indices: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Compute complete Dual VAE loss.

        Args:
            x: Input data (batch_size, 9)
            outputs: Model outputs dict
            lambda1: Weight for VAE-A loss
            lambda2: Weight for VAE-B loss
            lambda3: Weight for entropy alignment
            entropy_weight_B: Weight for VAE-B entropy regularization
            repulsion_weight_B: Weight for VAE-B repulsion loss
            grad_norm_A_ema: EMA of VAE-A gradient norm
            grad_norm_B_ema: EMA of VAE-B gradient norm
            gradient_balance: Whether to apply gradient balancing
            training: Whether in training mode
            batch_indices: Operation indices for p-adic losses (batch_size,)

        Returns:
            Dictionary of losses and metrics
        """
        # Reconstruction losses
        ce_A = self.reconstruction_loss(outputs['logits_A'], x)
        ce_B = self.reconstruction_loss(outputs['logits_B'], x)

        # KL divergences
        kl_A = self.kl_loss(outputs['mu_A'], outputs['logvar_A'])
        kl_B = self.kl_loss(outputs['mu_B'], outputs['logvar_B'])

        # VAE-A loss (simple β-VAE)
        loss_A = ce_A + outputs['beta_A'] * kl_A

        # VAE-B losses
        entropy_B = -self.entropy_loss(outputs['logits_B'])  # Negate to get actual entropy
        entropy_loss_B = -entropy_B  # Loss term (negative entropy)
        repulsion_B = self.repulsion_loss(outputs['z_B'])

        loss_B = (
            ce_B +
            outputs['beta_B'] * kl_B +
            entropy_weight_B * entropy_loss_B +
            repulsion_weight_B * repulsion_B
        )

        # Gradient normalization
        if gradient_balance and training:
            with torch.no_grad():
                grad_scale_A = grad_norm_B_ema / (grad_norm_A_ema + 1e-8)
                grad_scale_B = grad_norm_A_ema / (grad_norm_B_ema + 1e-8)
                grad_scale_A = torch.clamp(grad_scale_A, 0.5, 2.0)
                grad_scale_B = torch.clamp(grad_scale_B, 0.5, 2.0)
        else:
            grad_scale_A = 1.0
            grad_scale_B = 1.0

        # Entropy alignment
        entropy_align = torch.abs(outputs['H_A'] - outputs['H_B'])

        # Total loss (base)
        total_loss = (
            lambda1 * loss_A * grad_scale_A +
            lambda2 * loss_B * grad_scale_B +
            lambda3 * entropy_align
        )

        # p-Adic losses (Phase 1A/1B from implement.md)
        padic_metric_A = torch.tensor(0.0, device=x.device)
        padic_metric_B = torch.tensor(0.0, device=x.device)
        padic_ranking_A = torch.tensor(0.0, device=x.device)
        padic_ranking_B = torch.tensor(0.0, device=x.device)
        padic_norm_A = torch.tensor(0.0, device=x.device)
        padic_norm_B = torch.tensor(0.0, device=x.device)

        if batch_indices is not None:
            # Phase 1A: p-Adic Metric Loss (MSE-based)
            if self.enable_metric_loss:
                padic_metric_A = self.padic_metric_loss(outputs['z_A'], batch_indices)
                padic_metric_B = self.padic_metric_loss(outputs['z_B'], batch_indices)
                total_loss = total_loss + self.metric_loss_weight * (padic_metric_A + padic_metric_B)

            # Phase 1A-alt: p-Adic Ranking Loss (triplet-based, better for scale mismatch)
            if self.enable_ranking_loss:
                padic_ranking_A = self.padic_ranking_loss(outputs['z_A'], batch_indices)
                padic_ranking_B = self.padic_ranking_loss(outputs['z_B'], batch_indices)
                total_loss = total_loss + self.ranking_loss_weight * (padic_ranking_A + padic_ranking_B)

            # Phase 1B: p-Adic Norm Loss
            if self.enable_norm_loss:
                padic_norm_A = self.padic_norm_loss(outputs['z_A'], batch_indices)
                padic_norm_B = self.padic_norm_loss(outputs['z_B'], batch_indices)
                total_loss = total_loss + self.norm_loss_weight * (padic_norm_A + padic_norm_B)

        return {
            'loss': total_loss,
            'ce_A': ce_A,
            'ce_B': ce_B,
            'kl_A': kl_A,
            'kl_B': kl_B,
            'loss_A': loss_A,
            'loss_B': loss_B,
            'entropy_B': entropy_B,
            'repulsion_B': repulsion_B,
            'entropy_align': entropy_align,
            'H_A': outputs['H_A'],
            'H_B': outputs['H_B'],
            'grad_scale_A': grad_scale_A.item() if isinstance(grad_scale_A, torch.Tensor) else grad_scale_A,
            'grad_scale_B': grad_scale_B.item() if isinstance(grad_scale_B, torch.Tensor) else grad_scale_B,
            'lambda1': lambda1,
            'lambda2': lambda2,
            'lambda3': lambda3,
            # p-Adic losses (Phase 1A/1B)
            'padic_metric_A': padic_metric_A.item() if torch.is_tensor(padic_metric_A) else padic_metric_A,
            'padic_metric_B': padic_metric_B.item() if torch.is_tensor(padic_metric_B) else padic_metric_B,
            'padic_ranking_A': padic_ranking_A.item() if torch.is_tensor(padic_ranking_A) else padic_ranking_A,
            'padic_ranking_B': padic_ranking_B.item() if torch.is_tensor(padic_ranking_B) else padic_ranking_B,
            'padic_norm_A': padic_norm_A.item() if torch.is_tensor(padic_norm_A) else padic_norm_A,
            'padic_norm_B': padic_norm_B.item() if torch.is_tensor(padic_norm_B) else padic_norm_B
        }
