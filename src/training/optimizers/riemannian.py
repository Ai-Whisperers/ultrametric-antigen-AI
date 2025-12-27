# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Riemannian Optimizer Wrapper for Mixed Euclidean/Hyperbolic Training.

This module provides a unified optimizer interface for training models with
both Euclidean parameters (standard nn.Parameter) and Riemannian parameters
(geoopt.ManifoldParameter on Poincare ball).

Key Features:
- Automatic separation of Euclidean and Manifold parameters
- Different learning rates for different parameter types
- Gradient clipping for stability at ball boundary
- Warm-up scheduling for hyperbolic parameters

Usage:
    from src.training.optimizers import MixedRiemannianOptimizer

    optimizer = MixedRiemannianOptimizer(
        model.parameters(),
        euclidean_lr=1e-3,
        manifold_lr=1e-2,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional

# geoopt is a required dependency for hyperbolic geometry
import torch
import torch.nn as nn
from geoopt import ManifoldParameter
from geoopt.optim import RiemannianAdam


@dataclass
class OptimizerConfig:
    """Configuration for mixed Riemannian optimizer."""

    euclidean_lr: float = 1e-3
    manifold_lr: float = 1e-2
    euclidean_betas: tuple[float, float] = (0.9, 0.999)
    manifold_betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    manifold_weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = 1.0
    warmup_steps: int = 100
    stabilize_every: int = 100


class MixedRiemannianOptimizer:
    """Optimizer wrapper that handles both Euclidean and Manifold parameters.

    This class creates separate optimizer instances for Euclidean parameters
    (using standard Adam) and Manifold parameters (using RiemannianAdam),
    providing a unified interface for mixed-geometry training.

    Attributes:
        euclidean_optimizer: Standard Adam optimizer for Euclidean params
        manifold_optimizer: RiemannianAdam for Poincare ball params
        config: Optimizer configuration
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        euclidean_lr: float = 1e-3,
        manifold_lr: float = 1e-2,
        euclidean_betas: tuple[float, float] = (0.9, 0.999),
        manifold_betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.0,
        manifold_weight_decay: float = 0.0,
        grad_clip_norm: Optional[float] = 1.0,
        warmup_steps: int = 100,
        stabilize_every: int = 100,
    ):
        """Initialize mixed optimizer.

        Args:
            params: Iterator over model parameters
            euclidean_lr: Learning rate for Euclidean parameters
            manifold_lr: Learning rate for Manifold (Poincare) parameters
            euclidean_betas: Adam betas for Euclidean optimizer
            manifold_betas: Adam betas for Riemannian optimizer
            weight_decay: Weight decay for Euclidean parameters
            manifold_weight_decay: Weight decay for Manifold parameters
            grad_clip_norm: Max gradient norm for clipping (None to disable)
            warmup_steps: Number of warmup steps for manifold lr
            stabilize_every: Stabilize manifold params every N steps
        """
        self.config = OptimizerConfig(
            euclidean_lr=euclidean_lr,
            manifold_lr=manifold_lr,
            euclidean_betas=euclidean_betas,
            manifold_betas=manifold_betas,
            weight_decay=weight_decay,
            manifold_weight_decay=manifold_weight_decay,
            grad_clip_norm=grad_clip_norm,
            warmup_steps=warmup_steps,
            stabilize_every=stabilize_every,
        )

        # Separate parameters by type
        euclidean_params = []
        manifold_params = []

        for param in params:
            if isinstance(param, ManifoldParameter):
                manifold_params.append(param)
            else:
                euclidean_params.append(param)

        # Create optimizers
        self.euclidean_optimizer = None
        self.manifold_optimizer = None

        if euclidean_params:
            self.euclidean_optimizer = torch.optim.Adam(
                euclidean_params,
                lr=euclidean_lr,
                betas=euclidean_betas,
                weight_decay=weight_decay,
            )

        if manifold_params:
            self.manifold_optimizer = RiemannianAdam(
                manifold_params,
                lr=manifold_lr,
                betas=manifold_betas,
                weight_decay=manifold_weight_decay,
            )

        self._step_count = 0
        self._euclidean_params = euclidean_params
        self._manifold_params = manifold_params

    @property
    def param_groups(self) -> list:
        """Get all parameter groups from both optimizers."""
        groups = []
        if self.euclidean_optimizer:
            groups.extend(self.euclidean_optimizer.param_groups)
        if self.manifold_optimizer:
            groups.extend(self.manifold_optimizer.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients for all parameters."""
        if self.euclidean_optimizer:
            self.euclidean_optimizer.zero_grad(set_to_none=set_to_none)
        if self.manifold_optimizer:
            self.manifold_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        """Perform a single optimization step.

        Args:
            closure: Optional closure that reevaluates the model and returns the loss

        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        # Gradient clipping
        if self.config.grad_clip_norm is not None:
            all_params = self._euclidean_params + self._manifold_params
            torch.nn.utils.clip_grad_norm_(all_params, self.config.grad_clip_norm)

        # Apply warmup for manifold optimizer
        if self.manifold_optimizer and self._step_count <= self.config.warmup_steps:
            warmup_factor = self._step_count / self.config.warmup_steps
            for group in self.manifold_optimizer.param_groups:
                group["lr"] = self.config.manifold_lr * warmup_factor

        # Step both optimizers
        if self.euclidean_optimizer:
            self.euclidean_optimizer.step()

        if self.manifold_optimizer:
            self.manifold_optimizer.step()

            # Periodic stabilization (re-project to manifold)
            if self._step_count % self.config.stabilize_every == 0:
                self._stabilize_manifold_params()

        return loss

    def _stabilize_manifold_params(self) -> None:
        """Re-project manifold parameters to ensure they stay on the manifold."""
        for param in self._manifold_params:
            if hasattr(param, "manifold"):
                with torch.no_grad():
                    param.data = param.manifold.projx(param.data)

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the optimizer as a dict."""
        state = {
            "step_count": self._step_count,
            "config": self.config.__dict__,
        }

        if self.euclidean_optimizer:
            state["euclidean"] = self.euclidean_optimizer.state_dict()

        if self.manifold_optimizer:
            state["manifold"] = self.manifold_optimizer.state_dict()

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the optimizer state."""
        self._step_count = state_dict.get("step_count", 0)

        if "euclidean" in state_dict and self.euclidean_optimizer:
            self.euclidean_optimizer.load_state_dict(state_dict["euclidean"])

        if "manifold" in state_dict and self.manifold_optimizer:
            self.manifold_optimizer.load_state_dict(state_dict["manifold"])

    def get_lr(self) -> Dict[str, float]:
        """Get current learning rates."""
        lrs = {}

        if self.euclidean_optimizer:
            lrs["euclidean_lr"] = self.euclidean_optimizer.param_groups[0]["lr"]

        if self.manifold_optimizer:
            lrs["manifold_lr"] = self.manifold_optimizer.param_groups[0]["lr"]

        return lrs

    def set_lr(self, euclidean_lr: Optional[float] = None, manifold_lr: Optional[float] = None) -> None:
        """Set learning rates."""
        if euclidean_lr is not None and self.euclidean_optimizer:
            for group in self.euclidean_optimizer.param_groups:
                group["lr"] = euclidean_lr

        if manifold_lr is not None and self.manifold_optimizer:
            for group in self.manifold_optimizer.param_groups:
                group["lr"] = manifold_lr
            self.config.manifold_lr = manifold_lr


class HyperbolicScheduler:
    """Learning rate scheduler with special handling for hyperbolic parameters.

    Provides cosine annealing with warm restarts, with optional different
    schedules for Euclidean vs Manifold parameters.
    """

    def __init__(
        self,
        optimizer: MixedRiemannianOptimizer,
        T_0: int = 10,
        T_mult: int = 2,
        eta_min_euclidean: float = 1e-6,
        eta_min_manifold: float = 1e-5,
    ):
        """Initialize scheduler.

        Args:
            optimizer: MixedRiemannianOptimizer instance
            T_0: Number of iterations for the first restart
            T_mult: Factor for increasing T_i after each restart
            eta_min_euclidean: Minimum learning rate for Euclidean params
            eta_min_manifold: Minimum learning rate for Manifold params
        """
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min_euclidean = eta_min_euclidean
        self.eta_min_manifold = eta_min_manifold

        self.base_euclidean_lr = optimizer.config.euclidean_lr
        self.base_manifold_lr = optimizer.config.manifold_lr

        self.T_cur = 0
        self.T_i = T_0
        self.cycle = 0

    def step(self) -> None:
        """Update learning rates based on cosine annealing schedule."""
        import math

        # Compute cosine factor
        factor = (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2

        # Update Euclidean lr
        new_euclidean_lr = self.eta_min_euclidean + (self.base_euclidean_lr - self.eta_min_euclidean) * factor

        # Update Manifold lr
        new_manifold_lr = self.eta_min_manifold + (self.base_manifold_lr - self.eta_min_manifold) * factor

        self.optimizer.set_lr(new_euclidean_lr, new_manifold_lr)

        # Update cycle counter
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
            self.cycle += 1

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state."""
        return {
            "T_cur": self.T_cur,
            "T_i": self.T_i,
            "cycle": self.cycle,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state."""
        self.T_cur = state_dict["T_cur"]
        self.T_i = state_dict["T_i"]
        self.cycle = state_dict["cycle"]


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "mixed_riemannian",
    **kwargs: Any,
) -> Any:
    """Factory function to create optimizer for a model.

    Args:
        model: PyTorch model
        optimizer_type: Type of optimizer ('mixed_riemannian', 'adam', 'sgd')
        **kwargs: Optimizer-specific arguments

    Returns:
        Optimizer instance
    """
    if optimizer_type == "mixed_riemannian":
        return MixedRiemannianOptimizer(model.parameters(), **kwargs)

    elif optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)

    elif optimizer_type == "riemannian_adam":
        return RiemannianAdam(model.parameters(), **kwargs)

    elif optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), **kwargs)

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


__all__ = [
    "OptimizerConfig",
    "MixedRiemannianOptimizer",
    "HyperbolicScheduler",
    "create_optimizer",
]
