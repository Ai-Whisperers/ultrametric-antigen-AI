"""Ternary VAE v5.11 - Unified Hyperbolic Geometry with Frozen Coverage.

V5.11 ARCHITECTURE:

1. FROZEN v5.5 Encoder: 100% coverage preserved, no gradients
2. Trainable HyperbolicProjection: Learns radial hierarchy
3. DifferentiableController: Full gradient flow (no .item() calls)
4. Unified PAdicGeodesicLoss: Hierarchy + correlation via geometry

Key insight: v5.5 achieved 100% coverage but inverted radial hierarchy.
V5.11 freezes the coverage and learns only the geometric projection.

Architecture:
```
Input x ──► [FROZEN v5.5 Encoder] ──► z_euclidean (16D)
                 (no gradients)

         ──► [HyperbolicProjection] ──► z_hyp (Poincaré ball)
                 (trainable)

         ──► [DifferentiableController] ──► control signals
                 (trainable, full gradient flow)

         ──► [Unified PAdicGeodesicLoss]
                 (single loss = hierarchy + correlation)
```

Single responsibility: V5.11 model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Tuple

from .hyperbolic_projection import HyperbolicProjection, DualHyperbolicProjection
from .differentiable_controller import DifferentiableController, ThreeBodyController


class FrozenEncoder(nn.Module):
    """Frozen encoder from v5.5 checkpoint.

    This encoder has achieved 100% coverage and NEVER trains.
    We only use it to produce Euclidean latent codes for projection.
    """

    def __init__(self, input_dim: int = 9, latent_dim: int = 16):
        super().__init__()

        # Architecture must match v5.5 exactly
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # FREEZE all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - deterministic, no gradients."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @classmethod
    def from_v5_5_checkpoint(
        cls,
        checkpoint_path: Path,
        encoder_prefix: str = 'encoder_A',
        device: str = 'cpu'
    ) -> 'FrozenEncoder':
        """Load frozen encoder from v5.5 checkpoint.

        Args:
            checkpoint_path: Path to v5.5 checkpoint
            encoder_prefix: Which encoder to load ('encoder_A' or 'encoder_B')
            device: Device to load to

        Returns:
            FrozenEncoder with loaded weights
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint['model']

        # Create encoder
        encoder = cls()

        # Extract and load weights
        encoder_state = {}
        for key, value in model_state.items():
            if key.startswith(f'{encoder_prefix}.'):
                new_key = key[len(encoder_prefix) + 1:]  # Remove prefix
                encoder_state[new_key] = value

        encoder.load_state_dict(encoder_state)
        encoder.to(device)

        # Ensure frozen
        for param in encoder.parameters():
            param.requires_grad = False

        return encoder


class FrozenDecoder(nn.Module):
    """Frozen decoder from v5.5 checkpoint.

    Used for reconstruction verification (not training).
    """

    def __init__(self, latent_dim: int = 16, output_dim: int = 9):
        super().__init__()
        self.output_dim = output_dim

        # Architecture must match v5.5 exactly (decoder_A style)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim * 3)
        )

        # FREEZE
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass - produces logits."""
        logits = self.decoder(z)
        return logits.view(-1, self.output_dim, 3)

    @classmethod
    def from_v5_5_checkpoint(
        cls,
        checkpoint_path: Path,
        decoder_prefix: str = 'decoder_A',
        device: str = 'cpu'
    ) -> 'FrozenDecoder':
        """Load frozen decoder from v5.5 checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint['model']

        decoder = cls()

        decoder_state = {}
        for key, value in model_state.items():
            if key.startswith(f'{decoder_prefix}.'):
                new_key = key[len(decoder_prefix) + 1:]
                decoder_state[new_key] = value

        decoder.load_state_dict(decoder_state)
        decoder.to(device)

        for param in decoder.parameters():
            param.requires_grad = False

        return decoder


class TernaryVAEV5_11(nn.Module):
    """Ternary VAE v5.11 with frozen coverage and trainable hyperbolic projection.

    This model:
    1. Uses frozen v5.5 encoder for 100% coverage (no training)
    2. Projects Euclidean latents to Poincaré ball (trainable)
    3. Uses differentiable controller for loss weighting (trainable)
    4. Trains only for hyperbolic structure (geodesic loss)
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        max_radius: float = 0.95,
        curvature: float = 1.0,
        use_controller: bool = True,
        use_dual_projection: bool = False,
        n_projection_layers: int = 1,
        projection_dropout: float = 0.0
    ):
        """Initialize TernaryVAEV5_11.

        Args:
            latent_dim: Latent space dimension (must match v5.5)
            hidden_dim: Hidden dimension for projection networks
            max_radius: Maximum radius in Poincaré ball
            curvature: Hyperbolic curvature parameter
            use_controller: Whether to use differentiable controller
            use_dual_projection: Whether to use separate A/B projections
            n_projection_layers: Number of hidden layers in projection (1=shallow, 2+=deep)
            projection_dropout: Dropout rate for projection networks (default: 0.0)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_radius = max_radius
        self.curvature = curvature
        self.use_controller = use_controller
        self.use_dual_projection = use_dual_projection
        self.n_projection_layers = n_projection_layers
        self.projection_dropout = projection_dropout

        # Frozen encoders (will be loaded from checkpoint)
        self.encoder_A = FrozenEncoder(latent_dim=latent_dim)
        self.encoder_B = FrozenEncoder(latent_dim=latent_dim)

        # Frozen decoder (for verification only)
        self.decoder_A = FrozenDecoder(latent_dim=latent_dim)

        # Trainable hyperbolic projection
        if use_dual_projection:
            self.projection = DualHyperbolicProjection(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                max_radius=max_radius,
                curvature=curvature,
                n_layers=n_projection_layers,
                dropout=projection_dropout
            )
        else:
            self.projection = HyperbolicProjection(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                max_radius=max_radius,
                curvature=curvature,
                n_layers=n_projection_layers,
                dropout=projection_dropout
            )

        # Trainable controller
        if use_controller:
            self.controller = DifferentiableController(
                input_dim=8,
                hidden_dim=32
            )
        else:
            self.controller = None

        # Default control values (used when controller is disabled)
        self.default_control = {
            'rho': 0.0,  # No cross-injection for frozen model
            'weight_geodesic': 1.0,
            'weight_radial': 0.5,
            'beta_A': 1.0,
            'beta_B': 1.0,
            'tau': 0.5
        }

    def load_v5_5_checkpoint(self, checkpoint_path: Path, device: str = 'cpu'):
        """Load frozen components from v5.5 checkpoint.

        Args:
            checkpoint_path: Path to v5.5 checkpoint file
            device: Device to load to
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint['model']

        # Load encoder_A
        enc_A_state = {
            k.replace('encoder_A.', ''): v
            for k, v in model_state.items()
            if k.startswith('encoder_A.')
        }
        self.encoder_A.load_state_dict(enc_A_state)

        # Load encoder_B
        enc_B_state = {
            k.replace('encoder_B.', ''): v
            for k, v in model_state.items()
            if k.startswith('encoder_B.')
        }
        self.encoder_B.load_state_dict(enc_B_state)

        # Load decoder_A
        dec_A_state = {
            k.replace('decoder_A.', ''): v
            for k, v in model_state.items()
            if k.startswith('decoder_A.')
        }
        self.decoder_A.load_state_dict(dec_A_state)

        # Move to device
        self.to(device)

        # Ensure frozen components stay frozen
        for param in self.encoder_A.parameters():
            param.requires_grad = False
        for param in self.encoder_B.parameters():
            param.requires_grad = False
        for param in self.decoder_A.parameters():
            param.requires_grad = False

        print(f"Loaded v5.5 checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        compute_control: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x: Input ternary operations (batch, 9)
            compute_control: Whether to compute controller outputs

        Returns:
            Dict with all outputs
        """
        # Frozen encoding (no gradients)
        with torch.no_grad():
            mu_A, logvar_A = self.encoder_A(x)
            mu_B, logvar_B = self.encoder_B(x)
            z_A_euc = self.reparameterize(mu_A, logvar_A)
            z_B_euc = self.reparameterize(mu_B, logvar_B)

        # Trainable projection to Poincaré ball
        if self.use_dual_projection:
            z_A_hyp, z_B_hyp = self.projection(z_A_euc, z_B_euc)
        else:
            z_A_hyp = self.projection(z_A_euc)
            z_B_hyp = self.projection(z_B_euc)

        # Compute control signals (if enabled)
        if compute_control and self.controller is not None:
            # Compute batch statistics (all tensors for gradient flow)
            radius_A = torch.norm(z_A_hyp, dim=-1).mean()
            radius_B = torch.norm(z_B_hyp, dim=-1).mean()

            # Use mean embeddings for other stats
            kl_A = -0.5 * (1 + logvar_A - mu_A.pow(2) - logvar_A.exp()).sum(dim=-1).mean()
            kl_B = -0.5 * (1 + logvar_B - mu_B.pow(2) - logvar_B.exp()).sum(dim=-1).mean()

            # Placeholder for loss-based stats (will be filled during training)
            geo_loss_placeholder = torch.tensor(0.0, device=x.device)
            rad_loss_placeholder = torch.tensor(0.0, device=x.device)

            batch_stats = torch.stack([
                radius_A, radius_B,
                torch.tensor(1.0, device=x.device),  # H_A placeholder
                torch.tensor(1.0, device=x.device),  # H_B placeholder
                kl_A, kl_B,
                geo_loss_placeholder,
                rad_loss_placeholder
            ])

            control = self.controller(batch_stats)
            # Squeeze batch dimension
            control = {k: v.squeeze(0) for k, v in control.items()}
        else:
            control = {
                k: torch.tensor(v, device=x.device)
                for k, v in self.default_control.items()
            }

        # Verification reconstruction (no gradients, for monitoring only)
        with torch.no_grad():
            logits_A = self.decoder_A(z_A_euc)

        return {
            # Euclidean latents (from frozen encoder)
            'z_A_euc': z_A_euc,
            'z_B_euc': z_B_euc,
            'mu_A': mu_A,
            'mu_B': mu_B,
            'logvar_A': logvar_A,
            'logvar_B': logvar_B,

            # Hyperbolic latents (from trainable projection)
            'z_A_hyp': z_A_hyp,
            'z_B_hyp': z_B_hyp,

            # Control signals (from trainable controller)
            'control': control,

            # Reconstruction logits (for verification)
            'logits_A': logits_A
        }

    def get_trainable_parameters(self):
        """Get only trainable parameters (projection + controller)."""
        params = list(self.projection.parameters())
        if self.controller is not None:
            params.extend(self.controller.parameters())
        return params

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        frozen_params = sum(
            p.numel() for p in self.encoder_A.parameters()
        ) + sum(
            p.numel() for p in self.encoder_B.parameters()
        ) + sum(
            p.numel() for p in self.decoder_A.parameters()
        )

        projection_params = sum(p.numel() for p in self.projection.parameters())

        controller_params = 0
        if self.controller is not None:
            controller_params = sum(p.numel() for p in self.controller.parameters())

        return {
            'frozen': frozen_params,
            'projection': projection_params,
            'controller': controller_params,
            'trainable': projection_params + controller_params,
            'total': frozen_params + projection_params + controller_params
        }


class TernaryVAEV5_11_OptionC(TernaryVAEV5_11):
    """V5.11 Option C: Partial freeze variant with homeostatic control.

    Inherits from V5.11 but allows dynamic freeze/unfreeze of components
    based on training metrics (coverage, hierarchy, gradient norms).

    V5.11.7: Hierarchical homeostasis where:
    - encoder_A: coverage-gated (freeze on drop, unfreeze on recovery)
    - encoder_B: hierarchy-gated (freeze when VAE-B hierarchy plateaus)
    - controller: gradient-gated (freeze when weights stabilize)
    - projections: always trainable (fast adaptation layer)

    This implements complementary learning systems theory:
    - Slow components (encoders) consolidate when objectives met
    - Fast components (projections) continuously adapt
    """

    def __init__(
        self,
        freeze_encoder_b: bool = False,
        encoder_b_lr_scale: float = 0.1,
        encoder_a_lr_scale: float = 0.05,
        controller_lr_scale: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.freeze_encoder_b = freeze_encoder_b
        self.encoder_b_lr_scale = encoder_b_lr_scale
        self.controller_lr_scale = controller_lr_scale
        # Homeostatic freeze states
        self.freeze_encoder_a = True  # Starts frozen
        self.freeze_controller = False  # Starts trainable
        self.encoder_a_lr_scale = encoder_a_lr_scale

    def load_v5_5_checkpoint(self, checkpoint_path: Path, device: str = 'cpu'):
        """Load checkpoint with optional encoder_B unfreezing."""
        super().load_v5_5_checkpoint(checkpoint_path, device)

        if not self.freeze_encoder_b:
            # Unfreeze encoder_B for exploration
            for param in self.encoder_B.parameters():
                param.requires_grad = True
            print("  encoder_B UNFROZEN for Option C training")
            print(f"  encoder_B LR scale: {self.encoder_b_lr_scale}")

    def set_encoder_a_unfreeze(self, lr_scale: float):
        """Progressively unfreeze encoder_A with given learning rate scale.

        V5.11.6: Allows gradual unfreezing of encoder_A during training.
        Call this during training to update the unfreeze state.

        Args:
            lr_scale: Learning rate multiplier (0.0 = frozen, >0 = trainable)
        """
        self.encoder_a_lr_scale = lr_scale
        self.freeze_encoder_a = (lr_scale == 0.0)

        if not self.freeze_encoder_a:
            # Enable gradients for encoder_A
            for param in self.encoder_A.parameters():
                param.requires_grad = True
        else:
            # Disable gradients for encoder_A
            for param in self.encoder_A.parameters():
                param.requires_grad = False

    def set_encoder_a_frozen(self, frozen: bool):
        """Set encoder_A freeze state (homeostatic control).

        Args:
            frozen: True to freeze, False to unfreeze
        """
        self.freeze_encoder_a = frozen
        for param in self.encoder_A.parameters():
            param.requires_grad = not frozen

    def set_encoder_b_frozen(self, frozen: bool):
        """Set encoder_B freeze state (homeostatic control).

        Args:
            frozen: True to freeze, False to unfreeze
        """
        self.freeze_encoder_b = frozen
        for param in self.encoder_B.parameters():
            param.requires_grad = not frozen

    def set_controller_frozen(self, frozen: bool):
        """Set controller freeze state (homeostatic control).

        Args:
            frozen: True to freeze, False to unfreeze
        """
        if self.controller is None:
            return
        self.freeze_controller = frozen
        for param in self.controller.parameters():
            param.requires_grad = not frozen

    def apply_homeostasis_state(self, state: dict):
        """Apply freeze states from HomeostasisController.

        Args:
            state: Dict with encoder_a_frozen, encoder_b_frozen, controller_frozen
        """
        if 'encoder_a_frozen' in state:
            self.set_encoder_a_frozen(state['encoder_a_frozen'])
        if 'encoder_b_frozen' in state:
            self.set_encoder_b_frozen(state['encoder_b_frozen'])
        if 'controller_frozen' in state:
            self.set_controller_frozen(state['controller_frozen'])

    def get_freeze_state_summary(self) -> str:
        """Get human-readable freeze state summary."""
        states = []
        states.append(f"enc_A:{'F' if self.freeze_encoder_a else 'T'}")
        states.append(f"enc_B:{'F' if self.freeze_encoder_b else 'T'}")
        ctrl_state = 'F' if getattr(self, 'freeze_controller', False) else 'T'
        states.append(f"ctrl:{ctrl_state}")
        return " ".join(states)

    def forward(
        self,
        x: torch.Tensor,
        compute_control: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with conditional gradient flow for both encoders.

        Override parent to allow gradients through encoder_A and encoder_B
        based on freeze state. V5.11.6 enables progressive unfreezing.
        """
        # Encoder A: frozen by default, can be progressively unfrozen
        if self.freeze_encoder_a:
            with torch.no_grad():
                mu_A, logvar_A = self.encoder_A(x)
                z_A_euc = self.reparameterize(mu_A, logvar_A)
        else:
            # Allow gradients through encoder_A for progressive unfreezing
            mu_A, logvar_A = self.encoder_A(x)
            z_A_euc = self.reparameterize(mu_A, logvar_A)

        # Encoder B: frozen or trainable depending on config
        if self.freeze_encoder_b:
            with torch.no_grad():
                mu_B, logvar_B = self.encoder_B(x)
                z_B_euc = self.reparameterize(mu_B, logvar_B)
        else:
            # Allow gradients through encoder_B for Option C
            mu_B, logvar_B = self.encoder_B(x)
            z_B_euc = self.reparameterize(mu_B, logvar_B)

        # Trainable projection to Poincaré ball
        if self.use_dual_projection:
            z_A_hyp, z_B_hyp = self.projection(z_A_euc, z_B_euc)
        else:
            z_A_hyp = self.projection(z_A_euc)
            z_B_hyp = self.projection(z_B_euc)

        # Compute control signals (if enabled)
        if compute_control and self.controller is not None:
            radius_A = torch.norm(z_A_hyp, dim=-1).mean()
            radius_B = torch.norm(z_B_hyp, dim=-1).mean()
            kl_A = -0.5 * (1 + logvar_A - mu_A.pow(2) - logvar_A.exp()).sum(dim=-1).mean()
            kl_B = -0.5 * (1 + logvar_B - mu_B.pow(2) - logvar_B.exp()).sum(dim=-1).mean()

            batch_stats = torch.stack([
                radius_A, radius_B,
                torch.tensor(1.0, device=x.device),
                torch.tensor(1.0, device=x.device),
                kl_A, kl_B,
                torch.tensor(0.0, device=x.device),
                torch.tensor(0.0, device=x.device)
            ])

            control = self.controller(batch_stats)
            control = {k: v.squeeze(0) for k, v in control.items()}
        else:
            control = {
                k: torch.tensor(v, device=x.device)
                for k, v in self.default_control.items()
            }

        # Verification reconstruction
        with torch.no_grad():
            logits_A = self.decoder_A(mu_A)  # Use mean for deterministic verification

        return {
            'z_A_euc': z_A_euc,
            'z_B_euc': z_B_euc,
            'mu_A': mu_A,
            'mu_B': mu_B,
            'logvar_A': logvar_A,
            'logvar_B': logvar_B,
            'z_A_hyp': z_A_hyp,
            'z_B_hyp': z_B_hyp,
            'control': control,
            'logits_A': logits_A
        }

    def get_trainable_parameters(self):
        """Get trainable parameters including optionally unfrozen encoders."""
        params = super().get_trainable_parameters()

        if not self.freeze_encoder_a:
            params.extend(self.encoder_A.parameters())

        if not self.freeze_encoder_b:
            params.extend(self.encoder_B.parameters())

        return params

    def get_param_groups(self, base_lr: float):
        """Get parameter groups with different learning rates.

        Returns param groups for optimizer with encoders at lower LR.
        For dual projection, proj_A and proj_B get separate groups.
        Frozen components are excluded from param groups.
        """
        param_groups = []

        # Projections always trainable (fast adaptation layer)
        if self.use_dual_projection:
            param_groups.append({
                'params': list(self.projection.proj_A.parameters()),
                'lr': base_lr,
                'name': 'proj_A'
            })
            if hasattr(self.projection, 'proj_B'):
                proj_b_params = list(self.projection.proj_B.parameters())
            else:
                proj_b_params = list(self.projection.proj_B_radius.parameters())
            param_groups.append({
                'params': proj_b_params,
                'lr': base_lr,
                'name': 'proj_B'
            })
        else:
            param_groups.append({
                'params': list(self.projection.parameters()),
                'lr': base_lr,
                'name': 'projection'
            })

        # Controller (can be frozen by homeostasis)
        if self.controller is not None and not getattr(self, 'freeze_controller', False):
            param_groups.append({
                'params': list(self.controller.parameters()),
                'lr': base_lr * self.controller_lr_scale,
                'name': 'controller'
            })

        # Encoder A (can be frozen by homeostasis)
        if not self.freeze_encoder_a:
            param_groups.append({
                'params': list(self.encoder_A.parameters()),
                'lr': base_lr * self.encoder_a_lr_scale,
                'name': 'encoder_A'
            })

        # Encoder B (can be frozen by homeostasis)
        if not self.freeze_encoder_b:
            param_groups.append({
                'params': list(self.encoder_B.parameters()),
                'lr': base_lr * self.encoder_b_lr_scale,
                'name': 'encoder_B'
            })

        return param_groups

    def rebuild_optimizer(self, optimizer, base_lr: float):
        """Rebuild optimizer with current freeze states.

        Called when homeostasis changes freeze states to update optimizer.
        Returns new optimizer with correct param groups.
        """
        import torch.optim as optim
        param_groups = self.get_param_groups(base_lr)
        new_optimizer = optim.AdamW(
            param_groups,
            weight_decay=optimizer.defaults.get('weight_decay', 1e-4)
        )
        return new_optimizer

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = super().count_parameters()

        if not self.freeze_encoder_b:
            encoder_b_params = sum(p.numel() for p in self.encoder_B.parameters())
            counts['encoder_b_trainable'] = encoder_b_params
            counts['trainable'] += encoder_b_params
            counts['frozen'] -= encoder_b_params

        return counts


__all__ = [
    'FrozenEncoder',
    'FrozenDecoder',
    'TernaryVAEV5_11',
    'TernaryVAEV5_11_OptionC'
]
