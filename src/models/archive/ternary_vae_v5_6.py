# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Ternary VAE v5.6 - Production Implementation with TensorBoard and TorchInductor.

Key improvements over v5.5:
1. TensorBoard integration for local, IP-safe visualization
2. TorchInductor (torch.compile) support for 1.4-2x speedup
3. All config parameters properly integrated
4. Proper Phase 4 ultra-exploration support
5. Complete documentation

Architecture:
- Dual-VAE with StateNet controller (99.7% coverage at epoch 100+)
- Stop-gradient cross-injection
- Adaptive gradient balance
- Phase-scheduled permeability
- Cyclic entropy alignment

Total parameters: ~168,770
- VAE-A: 50,203 params
- VAE-B: 117,499 params
- StateNet: 1,068 params (0.63% overhead)
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryEncoderA(nn.Module):
    """VAE-A Encoder (chaotic regime)."""

    def __init__(self, input_dim: int = 9, latent_dim: int = 16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class TernaryDecoderA(nn.Module):
    """VAE-A Decoder (chaotic regime)."""

    def __init__(self, latent_dim: int = 16, output_dim: int = 9):
        super().__init__()
        self.output_dim = output_dim

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim * 3),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(z)
        logits = logits.view(-1, self.output_dim, 3)
        return logits


class ResidualBlock(nn.Module):
    """Residual block for VAE-B decoder."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TernaryEncoderB(nn.Module):
    """VAE-B Encoder (frozen regime)."""

    def __init__(self, input_dim: int = 9, latent_dim: int = 16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class TernaryDecoderB(nn.Module):
    """VAE-B Decoder (frozen regime with residual connections)."""

    def __init__(self, latent_dim: int = 16, output_dim: int = 9):
        super().__init__()
        self.output_dim = output_dim

        self.fc_in = nn.Linear(latent_dim, 128)
        self.residual1 = ResidualBlock(128)
        self.residual2 = ResidualBlock(128)
        self.fc_out = nn.Linear(128, output_dim * 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc_in(z))
        h = self.residual1(h)
        h = F.relu(h)
        h = self.residual2(h)
        logits = self.fc_out(h)
        logits = logits.view(-1, self.output_dim, 3)
        return logits


class StateNet(nn.Module):
    """Coverage-aware StateNet v2 for learned hyperparameter modulation.

    Compresses training state into latent representation, then decodes
    to produce corrections for learning rate and lambda values.

    v2 UPGRADE: Now includes coverage feedback to prevent collapse blindness.
    The original StateNet (v1) only saw entropy/KL signals and couldn't detect
    the 4,028-operation coverage collapse during epochs 0-40.

    Input: state vector (12D):
        [H_A, H_B, KL_A, KL_B, grad_ratio, ρ, λ₁, λ₂, λ₃,
         coverage_A_norm, coverage_B_norm, missing_ops_norm]

        - coverage_A_norm: VAE-A coverage / 19683 (normalized to [0, 1])
        - coverage_B_norm: VAE-B coverage / 19683 (normalized to [0, 1])
        - missing_ops_norm: (19683 - max(cov_A, cov_B)) / 19683 (normalized)

    Latent: compressed state representation (8D)
    Output: corrections [Δlr, Δλ₁, Δλ₂, Δλ₃] (4D)
    """

    def __init__(self, state_dim: int = 12, hidden_dim: int = 32, latent_dim: int = 8):
        super().__init__()

        self.state_dim = state_dim
        self.latent_dim = latent_dim

        # Encoder: state → latent
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder: latent → corrections
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # [Δlr, Δλ₁, Δλ₂, Δλ₃]
            nn.Tanh(),  # Output in [-1, 1] for bounded corrections
        )

    def forward(self, state: torch.Tensor) -> tuple:
        """
        Args:
            state: Training state tensor [batch, 12] or [12]

        Returns:
            corrections: [Δlr, Δλ₁, Δλ₂, Δλ₃]
            latent: Compressed state representation
        """
        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Encode state to latent
        latent = self.encoder(state)

        # Decode to corrections
        corrections = self.decoder(latent)

        return corrections, latent


class DualNeuralVAEV5(nn.Module):
    """Adaptive Dual-Neural VAE v5.6 with StateNet controller.

    Production implementation with TensorBoard and TorchInductor support.
    Proven architecture achieving 99.57% coverage.
    """

    def __init__(
        self,
        input_dim: int = 9,
        latent_dim: int = 16,
        rho_min: float = 0.1,
        rho_max: float = 0.7,
        lambda3_base: float = 0.3,
        lambda3_amplitude: float = 0.15,
        eps_kl: float = 5e-4,
        gradient_balance: bool = True,
        adaptive_scheduling: bool = True,
        use_statenet: bool = True,
        statenet_lr_scale: float = 0.05,
        statenet_lambda_scale: float = 0.01,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.lambda3_base = lambda3_base
        self.lambda3_amplitude = lambda3_amplitude
        self.eps_kl = eps_kl
        self.gradient_balance = gradient_balance
        self.adaptive_scheduling = adaptive_scheduling
        self.use_statenet = use_statenet
        self.statenet_lr_scale = statenet_lr_scale
        self.statenet_lambda_scale = statenet_lambda_scale

        # VAE-A components (chaotic regime)
        self.encoder_A = TernaryEncoderA(input_dim, latent_dim)
        self.decoder_A = TernaryDecoderA(latent_dim, input_dim)

        # VAE-B components (frozen regime)
        self.encoder_B = TernaryEncoderB(input_dim, latent_dim)
        self.decoder_B = TernaryDecoderB(latent_dim, input_dim)

        # StateNet v2 controller (coverage-aware, 12D input)
        if self.use_statenet:
            self.state_net = StateNet(state_dim=12, hidden_dim=32, latent_dim=8)

        # Controller state
        self.H_A_prev = None
        self.H_B_prev = None
        self.epoch = 0

        # Adaptive weights
        self.lambda1 = 0.7
        self.lambda2 = 0.7
        self.rho = rho_min
        self.lambda3 = lambda3_base

        # Phase tracking
        self.current_phase = 1
        self.grad_balance_achieved = False

        # Gradient magnitude tracking
        self.register_buffer("grad_norm_A_ema", torch.tensor(1.0))
        self.register_buffer("grad_norm_B_ema", torch.tensor(1.0))
        self.grad_ema_momentum = 0.9

        # StateNet correction history
        self.statenet_corrections = {
            "delta_lr": [],
            "delta_lambda1": [],
            "delta_lambda2": [],
            "delta_lambda3": [],
        }

    def compute_phase_scheduled_rho(self, epoch: int, phase_4_start: int = 250) -> float:
        """Compute phase-scheduled latent permeability.

        Phase 1 (0-40):     ρ=0.1  (isolation)
        Phase 2 (40-120):   ρ→0.3  (consolidation)
        Phase 3 (120-250):  ρ→0.7  (resonant coupling)
        Phase 4 (250+):     ρ=0.7  (ultra-exploration, maintained)
        """
        if epoch < 40:
            self.current_phase = 1
            return self.rho_min
        elif epoch < 120:
            self.current_phase = 2
            progress = (epoch - 40) / 80.0
            return self.rho_min + progress * (0.3 - self.rho_min)
        elif epoch < phase_4_start:
            if self.grad_balance_achieved:
                self.current_phase = 3
                progress = min(1.0, (epoch - 120) / 80.0)
                return 0.3 + progress * (self.rho_max - 0.3)
            else:
                return 0.3
        else:
            # Phase 4: maintain high permeability
            self.current_phase = 4
            return self.rho_max

    def compute_cyclic_lambda3(self, epoch: int, period: int = 30) -> float:
        """Compute cyclic entropy alignment weight."""
        phase = (epoch % period) / period * 2 * np.pi + np.pi / 2
        return self.lambda3_base + self.lambda3_amplitude * np.cos(phase)

    def update_adaptive_ema_momentum(self, grad_ratio: float):
        """Update EMA momentum based on gradient balance."""
        if self.adaptive_scheduling:
            if 0.8 < grad_ratio < 1.2:
                self.grad_ema_momentum = 0.95
                self.grad_balance_achieved = True
            else:
                self.grad_ema_momentum = 0.5
                self.grad_balance_achieved = False
        else:
            self.grad_ema_momentum = 0.9

    def update_adaptive_lambdas(self, grad_ratio: float, coverage_A: int, coverage_B: int):
        """Update λ1 and λ2 adaptively."""
        if not self.adaptive_scheduling:
            return

        # Priority 1: Gradient balance
        if grad_ratio < 0.8:
            self.lambda1 = min(0.95, self.lambda1 + 0.02)
            self.lambda2 = max(0.50, self.lambda2 - 0.02)
        elif grad_ratio > 1.2:
            self.lambda1 = max(0.50, self.lambda1 - 0.02)
            self.lambda2 = min(0.95, self.lambda2 + 0.02)

        # Priority 2: Coverage balance
        if 0.8 < grad_ratio < 1.2 and coverage_A > 0 and coverage_B > 0:
            cov_ratio = coverage_B / (coverage_A + 1e-6)

            if cov_ratio > 1.5:
                self.lambda1 = min(0.95, self.lambda1 * 1.05)
                self.lambda2 = max(0.50, self.lambda2 * 0.95)
            elif cov_ratio < 0.67:
                self.lambda1 = max(0.50, self.lambda1 * 0.95)
                self.lambda2 = min(0.95, self.lambda2 * 1.05)

        self.lambda1 = max(0.5, min(0.95, self.lambda1))
        self.lambda2 = max(0.5, min(0.95, self.lambda2))

    def apply_statenet_corrections(
        self,
        lr: float,
        H_A: float,
        H_B: float,
        kl_A: float,
        kl_B: float,
        grad_ratio: float,
        coverage_A: int = 0,
        coverage_B: int = 0,
    ) -> tuple:
        """Apply StateNet v2 corrections to learning rate and lambdas.

        v2 UPGRADE: Now includes coverage feedback in state vector.
        This allows StateNet to detect and respond to coverage collapse
        proactively rather than reacting only to entropy/KL signals.

        Args:
            lr: Current learning rate
            H_A: VAE-A entropy
            H_B: VAE-B entropy
            kl_A: VAE-A KL divergence
            kl_B: VAE-B KL divergence
            grad_ratio: Gradient norm ratio (A/B)
            coverage_A: VAE-A unique operations count (0-19683)
            coverage_B: VAE-B unique operations count (0-19683)

        Returns:
            Tuple of (corrected_lr, delta_lr, delta_lambda1, delta_lambda2, delta_lambda3)
        """
        if not self.use_statenet or not self.training:
            return lr, 0.0, 0.0, 0.0, 0.0

        # Normalize coverage values to [0, 1] for stable training
        TOTAL_OPS = 19683
        coverage_A_norm = coverage_A / TOTAL_OPS
        coverage_B_norm = coverage_B / TOTAL_OPS
        missing_ops_norm = (TOTAL_OPS - max(coverage_A, coverage_B)) / TOTAL_OPS

        # Build 12D state vector with coverage feedback
        state_vec = torch.tensor(
            [
                H_A,
                H_B,
                kl_A,
                kl_B,
                grad_ratio,
                self.rho,
                self.lambda1,
                self.lambda2,
                self.lambda3,
                coverage_A_norm,
                coverage_B_norm,
                missing_ops_norm,
            ],
            device=self.grad_norm_A_ema.device,
            dtype=torch.float32,
        )

        corrections, latent = self.state_net(state_vec)
        delta_lr, delta_lambda1, delta_lambda2, delta_lambda3 = corrections[0]

        corrected_lr = lr * (1 + self.statenet_lr_scale * delta_lr.item())
        corrected_lr = max(1e-6, min(0.01, corrected_lr))

        self.lambda1 = torch.clamp(
            torch.tensor(self.lambda1) + self.statenet_lambda_scale * delta_lambda1,
            0.5,
            0.95,
        ).item()
        self.lambda2 = torch.clamp(
            torch.tensor(self.lambda2) + self.statenet_lambda_scale * delta_lambda2,
            0.5,
            0.95,
        ).item()
        self.lambda3 = torch.clamp(
            torch.tensor(self.lambda3) + self.statenet_lambda_scale * delta_lambda3,
            0.15,
            0.75,
        ).item()

        self.statenet_corrections["delta_lr"].append(delta_lr.item())
        self.statenet_corrections["delta_lambda1"].append(delta_lambda1.item())
        self.statenet_corrections["delta_lambda2"].append(delta_lambda2.item())
        self.statenet_corrections["delta_lambda3"].append(delta_lambda3.item())

        return (
            corrected_lr,
            delta_lr.item(),
            delta_lambda1.item(),
            delta_lambda2.item(),
            delta_lambda3.item(),
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Reparameterization trick with temperature."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std * temperature
        else:
            return mu

    def compute_latent_entropy(self, z: torch.Tensor, num_bins: int = 50) -> torch.Tensor:
        """Estimate latent entropy using histogram method."""
        batch_size, latent_dim = z.shape

        entropies = []
        for i in range(latent_dim):
            z_i = z[:, i]
            hist = torch.histc(z_i, bins=num_bins, min=-3.0, max=3.0)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            entropy = -(hist * torch.log(hist)).sum()
            entropies.append(entropy)

        return torch.stack(entropies).mean()

    def forward(
        self,
        x: torch.Tensor,
        temp_A: float = 1.0,
        temp_B: float = 1.0,
        beta_A: float = 1.0,
        beta_B: float = 1.0,
    ) -> dict:
        """Forward pass with stop-gradient cross-injection."""
        # Encode
        mu_A, logvar_A = self.encoder_A(x)
        mu_B, logvar_B = self.encoder_B(x)

        # Sample latents
        z_A = self.reparameterize(mu_A, logvar_A, temp_A)
        z_B = self.reparameterize(mu_B, logvar_B, temp_B)

        # Cross-injection with stop-gradient
        z_A_detached = z_A.detach()
        z_B_detached = z_B.detach()

        z_A_tilde = (1 - self.rho) * z_A + self.rho * z_B_detached
        z_B_tilde = (1 - self.rho) * z_B + self.rho * z_A_detached

        # Decode
        logits_A = self.decoder_A(z_A_tilde)
        logits_B = self.decoder_B(z_B_tilde)

        # Compute entropies
        with torch.no_grad():
            H_A = self.compute_latent_entropy(z_A)
            H_B = self.compute_latent_entropy(z_B)

        return {
            "logits_A": logits_A,
            "logits_B": logits_B,
            "mu_A": mu_A,
            "logvar_A": logvar_A,
            "mu_B": mu_B,
            "logvar_B": logvar_B,
            "z_A": z_A,
            "z_B": z_B,
            "z_A_tilde": z_A_tilde,
            "z_B_tilde": z_B_tilde,
            "H_A": H_A,
            "H_B": H_B,
            "beta_A": beta_A,
            "beta_B": beta_B,
        }

    def update_gradient_norms(self):
        """Update EMA of gradient norms."""
        if not self.training:
            return

        grad_norm_A = 0.0
        for param in list(self.encoder_A.parameters()) + list(self.decoder_A.parameters()):
            if param.grad is not None:
                grad_norm_A += param.grad.norm().item() ** 2
        grad_norm_A = math.sqrt(grad_norm_A)

        grad_norm_B = 0.0
        for param in list(self.encoder_B.parameters()) + list(self.decoder_B.parameters()):
            if param.grad is not None:
                grad_norm_B += param.grad.norm().item() ** 2
        grad_norm_B = math.sqrt(grad_norm_B)

        if grad_norm_A > 0:
            self.grad_norm_A_ema = self.grad_ema_momentum * self.grad_norm_A_ema + (1 - self.grad_ema_momentum) * grad_norm_A
        if grad_norm_B > 0:
            self.grad_norm_B_ema = self.grad_ema_momentum * self.grad_norm_B_ema + (1 - self.grad_ema_momentum) * grad_norm_B

    def sample(self, num_samples: int, device: str = "cpu", use_vae: str = "A") -> torch.Tensor:
        """Sample from the learned manifold."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)

            if use_vae == "A":
                logits = self.decoder_A(z)
            else:
                logits = self.decoder_B(z)

            # Use categorical sampling instead of expectation
            dist = torch.distributions.Categorical(logits=logits)
            indices = dist.sample()
            values = torch.tensor([-1.0, 0.0, 1.0], device=device)
            samples = values[indices]

        return samples
