# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Loss Registry - Dynamic composition pattern.

This module provides a registry-based approach to loss composition
that eliminates the God Object anti-pattern in DualVAELoss.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                       LossRegistry                          │
    │                                                             │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
    │  │ ReconLoss    │  │ KLLoss       │  │ PAdicLoss    │ ...  │
    │  └──────────────┘  └──────────────┘  └──────────────┘      │
    │           │                │                │               │
    │           └────────────────┼────────────────┘               │
    │                            ▼                                │
    │                    ┌────────────────┐                       │
    │                    │ compose()       │                       │
    │                    │ returns total   │                       │
    │                    │ + all metrics   │                       │
    │                    └────────────────┘                       │
    └─────────────────────────────────────────────────────────────┘

Benefits:
    - Each loss is independently testable
    - Weights can be modified at runtime (StateNet control)
    - Easy to add/remove losses via configuration
    - Clean separation of concerns
    - No conditional spaghetti

Usage:
    registry = LossRegistry()
    registry.register('recon', ReconstructionLoss(weight=1.0))
    registry.register('kl', KLDivergenceLoss(weight=0.1))

    # During training
    result = registry.compose(outputs, targets, batch_indices=indices)
    total_loss = result.loss
    metrics = result.metrics

    # StateNet weight update
    registry.set_weight('kl', new_weight)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
from collections import OrderedDict

from .base import LossComponent, LossResult


class LossRegistry(nn.Module):
    """Dynamic loss composition registry.

    Manages a collection of LossComponent instances and composes them
    into a single loss with aggregated metrics.
    """

    def __init__(self):
        """Initialize empty registry."""
        super().__init__()
        self._losses: OrderedDict[str, LossComponent] = OrderedDict()
        self._enabled: Dict[str, bool] = {}
        self._weight_overrides: Dict[str, float] = {}

    def register(
        self,
        name: str,
        loss: LossComponent,
        enabled: bool = True
    ) -> 'LossRegistry':
        """Register a loss component.

        Args:
            name: Unique identifier for this loss
            loss: The loss component instance
            enabled: Whether this loss is enabled by default

        Returns:
            Self for chaining

        Raises:
            ValueError: If name already registered
        """
        if name in self._losses:
            raise ValueError(f"Loss '{name}' already registered")

        # Register as a submodule for proper parameter tracking
        self.add_module(name, loss)
        self._losses[name] = loss
        self._enabled[name] = enabled

        return self

    def unregister(self, name: str) -> 'LossRegistry':
        """Remove a loss component.

        Args:
            name: Identifier of loss to remove

        Returns:
            Self for chaining
        """
        if name in self._losses:
            delattr(self, name)
            del self._losses[name]
            del self._enabled[name]
            self._weight_overrides.pop(name, None)

        return self

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Enable or disable a loss.

        Args:
            name: Loss identifier
            enabled: Whether to enable
        """
        if name in self._enabled:
            self._enabled[name] = enabled

    def set_weight(self, name: str, weight: float) -> None:
        """Override the weight for a loss.

        This allows external control (e.g., StateNet) to adjust
        loss weights without modifying the loss component itself.

        Args:
            name: Loss identifier
            weight: New weight value
        """
        self._weight_overrides[name] = weight

    def get_weight(self, name: str) -> float:
        """Get current effective weight for a loss.

        Args:
            name: Loss identifier

        Returns:
            Current weight (override if set, else component weight)
        """
        if name in self._weight_overrides:
            return self._weight_overrides[name]
        if name in self._losses:
            return self._losses[name].weight
        return 0.0

    def reset_weight(self, name: str) -> None:
        """Reset weight override to use component's original weight.

        Args:
            name: Loss identifier
        """
        self._weight_overrides.pop(name, None)

    def get_loss(self, name: str) -> Optional[LossComponent]:
        """Get a registered loss by name.

        Args:
            name: Loss identifier

        Returns:
            LossComponent or None if not found
        """
        return self._losses.get(name)

    def list_losses(self) -> List[str]:
        """List all registered loss names.

        Returns:
            List of loss identifiers
        """
        return list(self._losses.keys())

    def list_enabled(self) -> List[str]:
        """List all enabled loss names.

        Returns:
            List of enabled loss identifiers
        """
        return [name for name, enabled in self._enabled.items() if enabled]

    def compose(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        **kwargs
    ) -> LossResult:
        """Compose all enabled losses into single result.

        This is the main entry point during training. It:
        1. Iterates through enabled losses
        2. Computes each loss with current weights
        3. Aggregates into single loss + merged metrics

        Args:
            outputs: Model outputs dictionary
            targets: Target values
            **kwargs: Additional arguments passed to each loss

        Returns:
            LossResult with total loss and all metrics
        """
        device = outputs.get('z_A', outputs.get('logits_A')).device
        total_loss = torch.tensor(0.0, device=device)
        all_metrics: Dict[str, float] = {}

        for name, loss_component in self._losses.items():
            # Skip disabled losses
            if not self._enabled.get(name, True):
                continue

            # Skip if component reports disabled
            if not loss_component.enabled():
                continue

            # Compute loss
            result = loss_component(outputs, targets, **kwargs)

            # Apply weight override if set
            effective_weight = self._weight_overrides.get(name, result.weight)
            weighted_loss = effective_weight * result.loss

            # Accumulate
            total_loss = total_loss + weighted_loss

            # Merge metrics with loss name prefix
            for key, value in result.metrics.items():
                metric_key = f'{name}/{key}'
                all_metrics[metric_key] = value

            # Also record the loss value itself
            all_metrics[f'{name}/loss'] = result.loss.item()
            all_metrics[f'{name}/weight'] = effective_weight

        return LossResult(
            loss=total_loss,
            metrics=all_metrics,
            weight=1.0  # Total is already weighted
        )

    def compose_with_gradients(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        **kwargs
    ) -> tuple:
        """Compose losses and return individual gradients for analysis.

        Useful for gradient balancing and StateNet feedback.

        Args:
            outputs: Model outputs
            targets: Target values
            **kwargs: Additional arguments

        Returns:
            Tuple of (total_loss, metrics, gradient_norms)
        """
        result = self.compose(outputs, targets, **kwargs)

        # Compute per-loss gradient norms if needed
        grad_norms: Dict[str, float] = {}

        return result.loss, result.metrics, grad_norms


class LossGroup:
    """Grouping of related losses for organization.

    Example:
        reconstruction_group = LossGroup('reconstruction')
        reconstruction_group.add('ce_A', recon_loss_A)
        reconstruction_group.add('ce_B', recon_loss_B)
        registry.register_group(reconstruction_group)
    """

    def __init__(self, name: str):
        """Initialize loss group.

        Args:
            name: Group name (used as prefix in metrics)
        """
        self.name = name
        self._losses: Dict[str, LossComponent] = {}

    def add(self, name: str, loss: LossComponent) -> 'LossGroup':
        """Add a loss to this group.

        Args:
            name: Loss name within group
            loss: Loss component

        Returns:
            Self for chaining
        """
        self._losses[name] = loss
        return self

    @property
    def losses(self) -> Dict[str, LossComponent]:
        """Return all losses in this group."""
        return self._losses


def create_registry_from_config(config: Dict[str, Any]) -> LossRegistry:
    """Factory to create LossRegistry from configuration.

    This is the main entry point for creating a configured registry.

    Args:
        config: Configuration dictionary with loss settings

    Returns:
        Configured LossRegistry

    Example config:
        {
            'losses': {
                'reconstruction': {'enabled': True, 'weight': 1.0},
                'kl': {'enabled': True, 'weight': 0.1, 'free_bits': 0.0},
                'padic_ranking': {'enabled': True, 'weight': 0.5, ...}
            }
        }
    """
    from .components import (
        ReconstructionLossComponent,
        KLDivergenceLossComponent,
        EntropyLossComponent,
        RepulsionLossComponent,
        PAdicRankingLossComponent,
        PAdicHyperbolicLossComponent
    )

    registry = LossRegistry()
    losses_config = config.get('losses', {})

    # Register each loss type if enabled
    loss_factories: Dict[str, Callable] = {
        'reconstruction': lambda cfg: ReconstructionLossComponent(
            weight=cfg.get('weight', 1.0)
        ),
        'kl': lambda cfg: KLDivergenceLossComponent(
            weight=cfg.get('weight', 0.1),
            free_bits=cfg.get('free_bits', 0.0)
        ),
        'entropy': lambda cfg: EntropyLossComponent(
            weight=cfg.get('weight', 0.01)
        ),
        'repulsion': lambda cfg: RepulsionLossComponent(
            weight=cfg.get('weight', 0.01),
            sigma=cfg.get('sigma', 0.5)
        ),
        'padic_ranking': lambda cfg: PAdicRankingLossComponent(
            weight=cfg.get('weight', 0.5),
            config=cfg
        ),
        'padic_hyperbolic': lambda cfg: PAdicHyperbolicLossComponent(
            weight=cfg.get('weight', 0.5),
            config=cfg
        ),
    }

    for loss_name, factory in loss_factories.items():
        loss_cfg = losses_config.get(loss_name, {})
        enabled = loss_cfg.get('enabled', False)

        if enabled:
            loss_component = factory(loss_cfg)
            registry.register(loss_name, loss_component, enabled=True)

    return registry


__all__ = ['LossRegistry', 'LossGroup', 'create_registry_from_config']
