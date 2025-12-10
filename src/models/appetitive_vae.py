"""Appetitive Dual-VAE: Bio-inspired VAE with emergent drives.

This module wraps the base DualNeuralVAEV5 with appetitive losses that
create emergent drives toward:
1. Curiosity (exploration)
2. Ordering (metric structure)
3. Hierarchy (MSB/LSB)
4. Symbiosis (A-B coupling)
5. Closure (algebraic)

Single responsibility: Appetitive model integration.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .ternary_vae_v5_6 import DualNeuralVAEV5
from ..losses.appetitive_losses import (
    AdaptiveRankingLoss,
    HierarchicalNormLoss,
    CuriosityModule,
    SymbioticBridge,
    AlgebraicClosureLoss,
    ViolationBuffer
)


class AppetitiveDualVAE(nn.Module):
    """Bio-inspired VAE with emergent drives.

    Wraps DualNeuralVAEV5 with five appetite modules that guide training
    through metric-gated phases toward algebraic closure.
    """

    def __init__(
        self,
        base_model: DualNeuralVAEV5,
        config: Dict[str, Any]
    ):
        """Initialize appetitive dual VAE.

        Args:
            base_model: Pre-existing DualNeuralVAEV5 model
            config: Configuration dictionary with appetite parameters
        """
        super().__init__()
        self.base = base_model
        self.latent_dim = config.get('latent_dim', 16)

        # Appetite modules
        self.ranking = AdaptiveRankingLoss(
            base_margin=config.get('ranking_margin', 0.1),
            n_triplets=config.get('ranking_n_triplets', 1000)
        )
        self.hierarchy = HierarchicalNormLoss(
            latent_dim=self.latent_dim,
            n_groups=config.get('hierarchy_n_groups', 4)
        )
        self.curiosity = CuriosityModule(
            latent_dim=self.latent_dim,
            bandwidth=config.get('curiosity_bandwidth', 1.0),
            max_history=config.get('curiosity_max_history', 5000)
        )
        self.symbiosis = SymbioticBridge(
            latent_dim=self.latent_dim,
            hidden_dim=config.get('symbiosis_hidden_dim', 32)
        )
        self.closure = AlgebraicClosureLoss()
        self.violation_buffer = ViolationBuffer(
            capacity=config.get('violation_capacity', 10000)
        )

        # Appetite weights (can be learned or scheduled)
        self.register_buffer('appetite_ranking', torch.tensor(config.get('appetite_ranking', 0.5)))
        self.register_buffer('appetite_hierarchy', torch.tensor(config.get('appetite_hierarchy', 0.1)))
        self.register_buffer('appetite_curiosity', torch.tensor(config.get('appetite_curiosity', 0.1)))
        self.register_buffer('appetite_symbiosis', torch.tensor(config.get('appetite_symbiosis', 0.1)))
        self.register_buffer('appetite_closure', torch.tensor(config.get('appetite_closure', 0.0)))

        # Phase tracking
        self.current_phase = 1
        self.phase_gates = {
            'phase_1a_to_1b': config.get('phase_1a_gate', 0.8),  # r > 0.8
            'phase_1b_to_2a': config.get('phase_1b_gate', 0.9),  # r > 0.9
            'phase_2a_to_2b': config.get('phase_2a_gate', 2.0),  # MI > 2.0
            'phase_2b_to_3': config.get('phase_2b_gate', 0.5),   # addition > 50%
        }

    def update_phase(self, metrics: Dict[str, float]):
        """Update current phase based on metrics.

        Args:
            metrics: Dictionary with 'correlation', 'mi', 'addition_accuracy'
        """
        corr = metrics.get('correlation', 0.0)
        mi = metrics.get('mi', 0.0)
        add_acc = metrics.get('addition_accuracy', 0.0)

        if self.current_phase == 1 and corr > self.phase_gates['phase_1a_to_1b']:
            self.current_phase = 2
            self._set_phase_weights(2)
            print(f"Phase transition: 1A -> 1B (r={corr:.3f})")

        elif self.current_phase == 2 and corr > self.phase_gates['phase_1b_to_2a']:
            self.current_phase = 3
            self._set_phase_weights(3)
            print(f"Phase transition: 1B -> 2A (r={corr:.3f})")

        elif self.current_phase == 3 and mi > self.phase_gates['phase_2a_to_2b']:
            self.current_phase = 4
            self._set_phase_weights(4)
            print(f"Phase transition: 2A -> 2B (MI={mi:.3f})")

        elif self.current_phase == 4 and add_acc > self.phase_gates['phase_2b_to_3']:
            self.current_phase = 5
            self._set_phase_weights(5)
            print(f"Phase transition: 2B -> 3 (add_acc={add_acc:.1%})")

    def _set_phase_weights(self, phase: int):
        """Set appetite weights based on phase.

        Args:
            phase: Current phase number
        """
        if phase == 1:  # 1A: Metric Foundation
            self.appetite_ranking.fill_(0.5)
            self.appetite_hierarchy.fill_(0.1)
            self.appetite_curiosity.fill_(0.0)
            self.appetite_symbiosis.fill_(0.0)
            self.appetite_closure.fill_(0.0)

        elif phase == 2:  # 1B: Structural Consolidation
            self.appetite_ranking.fill_(0.3)
            self.appetite_hierarchy.fill_(0.2)
            self.appetite_curiosity.fill_(0.05)
            self.appetite_symbiosis.fill_(0.0)
            self.appetite_closure.fill_(0.0)

        elif phase == 3:  # 2A: Symbiotic Coupling
            self.appetite_ranking.fill_(0.2)
            self.appetite_hierarchy.fill_(0.1)
            self.appetite_curiosity.fill_(0.1)
            self.appetite_symbiosis.fill_(0.3)
            self.appetite_closure.fill_(0.0)

        elif phase == 4:  # 2B: Algebraic Awakening
            self.appetite_ranking.fill_(0.1)
            self.appetite_hierarchy.fill_(0.05)
            self.appetite_curiosity.fill_(0.05)
            self.appetite_symbiosis.fill_(0.2)
            self.appetite_closure.fill_(0.5)

        elif phase == 5:  # 3: Algebraic Satiation
            self.appetite_ranking.fill_(0.05)
            self.appetite_hierarchy.fill_(0.05)
            self.appetite_curiosity.fill_(0.05)
            self.appetite_symbiosis.fill_(0.1)
            self.appetite_closure.fill_(0.7)

    def forward(
        self,
        x: torch.Tensor,
        indices: Optional[torch.Tensor] = None,
        compute_appetites: bool = True
    ) -> Dict[str, Any]:
        """Forward pass with appetite computation.

        Args:
            x: Input tensor (batch_size, 9)
            indices: Operation indices (batch_size,) - required for appetites
            compute_appetites: Whether to compute appetite losses

        Returns:
            Dictionary with base outputs and appetite losses
        """
        # Base model forward
        outputs = self.base(x)

        if not compute_appetites or indices is None:
            return outputs

        z_A = outputs['z_A']
        z_B = outputs['z_B']

        # Compute appetite losses
        # 1. Ranking (metric structure)
        ranking_loss_A = self.ranking(z_A, indices)
        ranking_loss_B = self.ranking(z_B, indices)
        ranking_loss = (ranking_loss_A + ranking_loss_B) / 2

        # 2. Hierarchy (MSB/LSB)
        hierarchy_loss_A = self.hierarchy(z_A)
        hierarchy_loss_B = self.hierarchy(z_B)
        hierarchy_loss = (hierarchy_loss_A + hierarchy_loss_B) / 2

        # 3. Curiosity (exploration)
        curiosity_loss_A = self.curiosity(z_A, update=self.training)
        curiosity_loss_B = self.curiosity(z_B, update=self.training)
        curiosity_loss = (curiosity_loss_A + curiosity_loss_B) / 2

        # 4. Symbiosis (A-B coupling)
        symbiosis_out = self.symbiosis(z_A, z_B)
        symbiosis_loss = symbiosis_out['mi_loss']

        # 5. Closure (algebraic)
        closure_loss_A = self.closure(z_A, indices)
        closure_loss_B = self.closure(z_B, indices)
        closure_loss = (closure_loss_A + closure_loss_B) / 2

        # Total appetite loss
        appetite_loss = (
            self.appetite_ranking * ranking_loss +
            self.appetite_hierarchy * hierarchy_loss +
            self.appetite_curiosity * curiosity_loss +
            self.appetite_symbiosis * symbiosis_loss +
            self.appetite_closure * closure_loss
        )

        # Update outputs
        outputs.update({
            'appetite_loss': appetite_loss,
            'ranking_loss': ranking_loss,
            'hierarchy_loss': hierarchy_loss,
            'curiosity_loss': curiosity_loss,
            'symbiosis_loss': symbiosis_loss,
            'closure_loss': closure_loss,
            'adaptive_rho': symbiosis_out['adaptive_rho'],
            'estimated_mi': symbiosis_out['estimated_mi'],
            'current_phase': self.current_phase,
            # Individual VAE losses for logging
            'ranking_loss_A': ranking_loss_A,
            'ranking_loss_B': ranking_loss_B,
        })

        return outputs

    def sample(self, n_samples: int, device: str, vae: str = 'A') -> torch.Tensor:
        """Sample from the model.

        Args:
            n_samples: Number of samples
            device: Device to generate on
            vae: Which VAE to sample from ('A' or 'B')

        Returns:
            Generated samples
        """
        return self.base.sample(n_samples, device, vae)

    def get_phase_description(self) -> str:
        """Get human-readable phase description."""
        descriptions = {
            1: "1A: Metric Foundation (ranking + hierarchy)",
            2: "1B: Structural Consolidation (+ proprioception)",
            3: "2A: Symbiotic Coupling (+ MI)",
            4: "2B: Algebraic Awakening (+ closure)",
            5: "3: Algebraic Satiation (closure dominant)"
        }
        return descriptions.get(self.current_phase, f"Unknown phase {self.current_phase}")


def create_appetitive_vae(
    config: Dict[str, Any],
    device: str = 'cuda'
) -> AppetitiveDualVAE:
    """Create a new AppetitiveDualVAE from config.

    Args:
        config: Configuration dictionary
        device: Device to create model on

    Returns:
        Initialized AppetitiveDualVAE
    """
    # Create base model
    base_model = DualNeuralVAEV5(
        input_dim=config.get('input_dim', 9),
        latent_dim=config.get('latent_dim', 16),
        rho_min=config.get('rho_min', 0.1),
        rho_max=config.get('rho_max', 0.7),
        use_statenet=config.get('use_statenet', True),
        statenet_lr_scale=config.get('statenet_lr_scale', 0.05),
        statenet_lambda_scale=config.get('statenet_lambda_scale', 0.01)
    )

    # Wrap with appetitive modules
    appetitive_config = {
        'latent_dim': config.get('latent_dim', 16),
        'ranking_margin': config.get('ranking_margin', 0.1),
        'ranking_n_triplets': config.get('ranking_n_triplets', 1000),
        'hierarchy_n_groups': config.get('hierarchy_n_groups', 4),
        'curiosity_bandwidth': config.get('curiosity_bandwidth', 1.0),
        'curiosity_max_history': config.get('curiosity_max_history', 5000),
        'symbiosis_hidden_dim': config.get('symbiosis_hidden_dim', 32),
        'violation_capacity': config.get('violation_capacity', 10000),
        # Initial appetite weights (Phase 1A)
        'appetite_ranking': config.get('appetite_ranking', 0.5),
        'appetite_hierarchy': config.get('appetite_hierarchy', 0.1),
        'appetite_curiosity': config.get('appetite_curiosity', 0.0),
        'appetite_symbiosis': config.get('appetite_symbiosis', 0.0),
        'appetite_closure': config.get('appetite_closure', 0.0),
        # Phase gates
        'phase_1a_gate': config.get('phase_1a_gate', 0.8),
        'phase_1b_gate': config.get('phase_1b_gate', 0.9),
        'phase_2a_gate': config.get('phase_2a_gate', 2.0),
        'phase_2b_gate': config.get('phase_2b_gate', 0.5),
    }

    model = AppetitiveDualVAE(base_model, appetitive_config)
    return model.to(device)
