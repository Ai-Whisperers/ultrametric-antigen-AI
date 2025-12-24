"""Hierarchical Homeostatic Controller for V5.11.7.

Implements complementary learning systems theory:
- Slow components (encoders): consolidate, freeze when objective met
- Fast components (projections, controller): continuously adapt

Each component has its own freeze trigger:
- encoder_A: coverage-gated (freeze on drop, unfreeze on recovery + hierarchy stall)
- encoder_B: hierarchy-gated (freeze when VAE-B hierarchy plateaus)
- controller: gradient-gated (freeze when weights stabilize)
- projections: always trainable (fast adaptation layer)
"""

import torch
from typing import Dict, Optional, Tuple
from collections import deque


class HomeostasisController:
    """Hierarchical homeostatic freeze/unfreeze controller.

    Monitors training metrics and dynamically adjusts component freeze states
    to balance coverage preservation with geometric structure learning.
    """

    def __init__(
        self,
        # Coverage thresholds for encoder_A
        coverage_freeze_threshold: float = 0.995,  # Freeze when drops below
        coverage_unfreeze_threshold: float = 1.0,  # Unfreeze when reaches

        # Hierarchy thresholds for encoder_B
        hierarchy_plateau_threshold: float = 0.001,  # Freeze when change < this
        hierarchy_plateau_patience: int = 5,  # Epochs of plateau before freeze

        # Controller gradient thresholds
        controller_grad_threshold: float = 0.01,  # Freeze when grad norm < this
        controller_grad_patience: int = 3,  # Epochs of low grad before freeze

        # General settings
        window_size: int = 5,  # Moving average window
        hysteresis_epochs: int = 3,  # Minimum epochs between state changes
        warmup_epochs: int = 5,  # Epochs before homeostasis activates
    ):
        self.coverage_freeze_threshold = coverage_freeze_threshold
        self.coverage_unfreeze_threshold = coverage_unfreeze_threshold
        self.hierarchy_plateau_threshold = hierarchy_plateau_threshold
        self.hierarchy_plateau_patience = hierarchy_plateau_patience
        self.controller_grad_threshold = controller_grad_threshold
        self.controller_grad_patience = controller_grad_patience
        self.window_size = window_size
        self.hysteresis_epochs = hysteresis_epochs
        self.warmup_epochs = warmup_epochs

        # Metric history (moving windows)
        self.coverage_history = deque(maxlen=window_size)
        self.hierarchy_A_history = deque(maxlen=window_size)
        self.hierarchy_B_history = deque(maxlen=window_size)
        self.controller_grad_history = deque(maxlen=window_size)

        # Freeze states
        self.encoder_a_frozen = True  # Starts frozen
        self.encoder_b_frozen = False  # Starts trainable (Option C)
        self.controller_frozen = False  # Starts trainable

        # Last state change epoch (for hysteresis)
        self.encoder_a_last_change = -hysteresis_epochs
        self.encoder_b_last_change = -hysteresis_epochs
        self.controller_last_change = -hysteresis_epochs

        # Plateau counters
        self.hierarchy_b_plateau_count = 0
        self.controller_low_grad_count = 0

        # Unfreeze conditions for encoder_A
        self.hierarchy_a_stalled = False
        self.hierarchy_a_stall_count = 0
        self.hierarchy_stall_patience = 5

    def update(
        self,
        epoch: int,
        coverage: float,
        hierarchy_A: float,
        hierarchy_B: float,
        controller_grad_norm: Optional[float] = None,
    ) -> Dict[str, bool]:
        """Update homeostasis state based on current metrics.

        Args:
            epoch: Current training epoch
            coverage: Current coverage (0-1)
            hierarchy_A: VAE-A radial hierarchy correlation (negative is good)
            hierarchy_B: VAE-B radial hierarchy correlation (negative is good)
            controller_grad_norm: Optional gradient norm of controller params

        Returns:
            Dict with freeze states and any triggered events
        """
        # Update histories
        self.coverage_history.append(coverage)
        self.hierarchy_A_history.append(hierarchy_A)
        self.hierarchy_B_history.append(hierarchy_B)
        if controller_grad_norm is not None:
            self.controller_grad_history.append(controller_grad_norm)

        events = []

        # Skip homeostasis during warmup
        if epoch < self.warmup_epochs:
            return {
                'encoder_a_frozen': self.encoder_a_frozen,
                'encoder_b_frozen': self.encoder_b_frozen,
                'controller_frozen': self.controller_frozen,
                'events': ['warmup']
            }

        # === Encoder A: Coverage-gated ===
        if self._can_change_state(epoch, self.encoder_a_last_change):
            encoder_a_decision = self._decide_encoder_a(coverage)
            if encoder_a_decision is not None:
                self.encoder_a_frozen = encoder_a_decision
                self.encoder_a_last_change = epoch
                events.append(f"encoder_A {'frozen' if encoder_a_decision else 'unfrozen'}")

        # === Encoder B: Hierarchy-gated ===
        if self._can_change_state(epoch, self.encoder_b_last_change):
            encoder_b_decision = self._decide_encoder_b()
            if encoder_b_decision is not None:
                self.encoder_b_frozen = encoder_b_decision
                self.encoder_b_last_change = epoch
                events.append(f"encoder_B {'frozen' if encoder_b_decision else 'unfrozen'}")

        # === Controller: Gradient-gated ===
        if controller_grad_norm is not None:
            if self._can_change_state(epoch, self.controller_last_change):
                controller_decision = self._decide_controller()
                if controller_decision is not None:
                    self.controller_frozen = controller_decision
                    self.controller_last_change = epoch
                    events.append(f"controller {'frozen' if controller_decision else 'unfrozen'}")

        return {
            'encoder_a_frozen': self.encoder_a_frozen,
            'encoder_b_frozen': self.encoder_b_frozen,
            'controller_frozen': self.controller_frozen,
            'events': events
        }

    def _can_change_state(self, epoch: int, last_change: int) -> bool:
        """Check if enough epochs have passed since last state change (hysteresis)."""
        return epoch - last_change >= self.hysteresis_epochs

    def _decide_encoder_a(self, coverage: float) -> Optional[bool]:
        """Decide encoder_A freeze state based on coverage.

        Logic:
        - If unfrozen and coverage drops below threshold -> FREEZE
        - If frozen and coverage=100% AND hierarchy stalled -> UNFREEZE

        Returns:
            True to freeze, False to unfreeze, None for no change
        """
        if not self.encoder_a_frozen:
            # Currently unfrozen - check if should freeze
            if coverage < self.coverage_freeze_threshold:
                return True  # Freeze to protect coverage
        else:
            # Currently frozen - check if should unfreeze
            if coverage >= self.coverage_unfreeze_threshold:
                # Coverage is good, check if hierarchy is stalled
                if self._is_hierarchy_a_stalled():
                    return False  # Unfreeze to escape plateau

        return None  # No change

    def _is_hierarchy_a_stalled(self) -> bool:
        """Check if VAE-A hierarchy has plateaued."""
        if len(self.hierarchy_A_history) < self.window_size:
            return False

        recent = list(self.hierarchy_A_history)
        # Hierarchy is negative, so smaller (more negative) is better
        # Stalled = no improvement in recent window
        improvement = abs(recent[-1]) - abs(recent[0])

        if improvement < self.hierarchy_plateau_threshold:
            self.hierarchy_a_stall_count += 1
        else:
            self.hierarchy_a_stall_count = 0

        return self.hierarchy_a_stall_count >= self.hierarchy_stall_patience

    def _decide_encoder_b(self) -> Optional[bool]:
        """Decide encoder_B freeze state based on VAE-B hierarchy.

        Logic:
        - If unfrozen and hierarchy plateaus for patience epochs -> FREEZE
        - If frozen and hierarchy is not at target -> UNFREEZE

        Returns:
            True to freeze, False to unfreeze, None for no change
        """
        if len(self.hierarchy_B_history) < 2:
            return None

        if not self.encoder_b_frozen:
            # Check for plateau
            recent = list(self.hierarchy_B_history)
            improvement = abs(recent[-1]) - abs(recent[0])

            if improvement < self.hierarchy_plateau_threshold:
                self.hierarchy_b_plateau_count += 1
            else:
                self.hierarchy_b_plateau_count = 0

            if self.hierarchy_b_plateau_count >= self.hierarchy_plateau_patience:
                return True  # Freeze - hierarchy plateaued
        else:
            # Currently frozen - unfreeze if hierarchy degraded
            if len(self.hierarchy_B_history) >= 2:
                if abs(self.hierarchy_B_history[-1]) < abs(self.hierarchy_B_history[-2]) - 0.01:
                    self.hierarchy_b_plateau_count = 0
                    return False  # Unfreeze - hierarchy degraded

        return None

    def _decide_controller(self) -> Optional[bool]:
        """Decide controller freeze state based on gradient norm.

        Logic:
        - If unfrozen and grad norm low for patience epochs -> FREEZE
        - If frozen and grad norm spikes -> UNFREEZE

        Returns:
            True to freeze, False to unfreeze, None for no change
        """
        if len(self.controller_grad_history) < 2:
            return None

        current_grad = self.controller_grad_history[-1]

        if not self.controller_frozen:
            if current_grad < self.controller_grad_threshold:
                self.controller_low_grad_count += 1
            else:
                self.controller_low_grad_count = 0

            if self.controller_low_grad_count >= self.controller_grad_patience:
                return True  # Freeze - controller stabilized
        else:
            # Check for gradient spike (need to adapt again)
            avg_grad = sum(self.controller_grad_history) / len(self.controller_grad_history)
            if current_grad > avg_grad * 2:  # Spike = 2x average
                self.controller_low_grad_count = 0
                return False  # Unfreeze

        return None

    def get_state_summary(self) -> str:
        """Get human-readable summary of current freeze states."""
        states = []
        states.append(f"enc_A:{'F' if self.encoder_a_frozen else 'T'}")
        states.append(f"enc_B:{'F' if self.encoder_b_frozen else 'T'}")
        states.append(f"ctrl:{'F' if self.controller_frozen else 'T'}")
        return " ".join(states)

    def reset(self):
        """Reset all state for new training run."""
        self.coverage_history.clear()
        self.hierarchy_A_history.clear()
        self.hierarchy_B_history.clear()
        self.controller_grad_history.clear()

        self.encoder_a_frozen = True
        self.encoder_b_frozen = False
        self.controller_frozen = False

        self.encoder_a_last_change = -self.hysteresis_epochs
        self.encoder_b_last_change = -self.hysteresis_epochs
        self.controller_last_change = -self.hysteresis_epochs

        self.hierarchy_b_plateau_count = 0
        self.controller_low_grad_count = 0
        self.hierarchy_a_stall_count = 0


__all__ = ['HomeostasisController']
