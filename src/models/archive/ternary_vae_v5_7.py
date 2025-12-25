# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Ternary VAE v5.7 - Metric-Aware StateNet with Unified r + Coverage Optimization.

Key improvements over v5.6:
1. StateNet v3: 14D input (adds r_A, r_B), 5D output (adds ranking_weight)
2. Dynamic ranking loss weight modulation based on metric-coverage balance
3. Unified optimization for both 3-adic correlation and coverage

Architecture:
- Dual-VAE with StateNet v3 controller (metric-aware)
- Stop-gradient cross-injection
- Adaptive gradient balance
- Phase-scheduled permeability
- Cyclic entropy alignment
- DYNAMIC ranking loss weight

The insight: StateNet v2 solved coverage collapse by adding coverage signals.
StateNet v3 solves the r-coverage tradeoff by adding ranking correlation signals.
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


class StateNetV3(nn.Module):
    """Metric-Aware StateNet v3 for unified r + coverage optimization.

    v3 UPGRADE: Adds ranking correlation feedback to enable dynamic balancing
    of metric structure vs coverage spread.

    The same architectural pattern that solved coverage collapse (adding coverage
    signals in v2) now solves the r-coverage tradeoff (adding ranking signals).

    Input: state vector (14D):
        [H_A, H_B, KL_A, KL_B, grad_ratio, rho, lambda1, lambda2, lambda3,
         coverage_A_norm, coverage_B_norm, missing_ops_norm,
         r_A, r_B]  <- NEW: ranking correlation signals

        - r_A: VAE-A 3-adic ranking correlation (0 to 1)
        - r_B: VAE-B 3-adic ranking correlation (0 to 1)

    Latent: compressed state representation (10D, up from 8D)
    Output: corrections [delta_lr, delta_lambda1, delta_lambda2, delta_lambda3, delta_ranking_weight] (5D)
        - delta_ranking_weight: NEW - modulates ranking loss weight dynamically
    """

    def __init__(self, state_dim: int = 14, hidden_dim: int = 48, latent_dim: int = 10):
        super().__init__()

        self.state_dim = state_dim
        self.latent_dim = latent_dim

        # Encoder: state -> latent (wider for richer representations)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder: latent -> corrections (5D output now)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),  # [delta_lr, delta_lambda1-3, delta_ranking_weight]
            nn.Tanh(),  # Output in [-1, 1] for bounded corrections
        )

        # Metric-specific attention head
        # Learns which state dimensions most affect r vs coverage
        self.metric_attention = nn.Sequential(nn.Linear(state_dim, 8), nn.Softmax(dim=-1))

    def forward(self, state: torch.Tensor) -> tuple:
        """
        Args:
            state: Training state tensor [batch, 14] or [14]

        Returns:
            corrections: [delta_lr, delta_lambda1, delta_lambda2, delta_lambda3, delta_ranking_weight]
            latent: Compressed state representation
            attention: Attention weights over state dimensions
        """
        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Normalize state for stable gradients
        # Scale entropy values (indices 0,1) and correlations (indices 12,13)
        normalized_state = state.clone()
        normalized_state[:, 0:2] = state[:, 0:2] / 3.0  # H_A, H_B typically 0-3
        # r_A, r_B already in [0,1] range

        # Compute attention over state dimensions
        attention = self.metric_attention(normalized_state)

        # Expand attention to match state dimensions (8 -> 14)
        # Use attention to weight different parts of the state
        attention_expanded = torch.cat(
            [
                attention[:, 0:2].repeat(1, 1),  # For entropy (H_A, H_B)
                attention[:, 2:4].repeat(1, 1),  # For KL (kl_A, kl_B)
                attention[:, 4:5],  # For grad_ratio
                attention[:, 4:5],  # For rho
                attention[:, 5:6].repeat(1, 3),  # For lambdas
                attention[:, 6:7].repeat(1, 3),  # For coverage
                attention[:, 7:8].repeat(1, 2),  # For ranking (r_A, r_B)
            ],
            dim=1,
        )

        # Apply attention-weighted encoding
        attended_state = normalized_state * attention_expanded

        # Encode attended state to latent (FIX: was using unattended state)
        latent = self.encoder(attended_state)

        # Decode to corrections
        corrections = self.decoder(latent)

        return corrections, latent, attention


class DualNeuralVAEV5_7(nn.Module):
    """Adaptive Dual-Neural VAE v5.7 with Metric-Aware StateNet v3.

    Key innovation: StateNet now sees ranking correlation and can dynamically
    adjust ranking loss weight to balance metric structure vs coverage spread.

    This creates a unified optimization objective where the model learns to:
    - Increase ranking weight when coverage is stable but r is low
    - Decrease ranking weight when r is high but coverage is dropping
    - Find the optimal operating point that maximizes both
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
        statenet_ranking_scale: float = 0.1,  # NEW: scale for ranking weight modulation
        base_ranking_weight: float = 0.5,  # NEW: base ranking loss weight
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
        self.statenet_ranking_scale = statenet_ranking_scale
        self.base_ranking_weight = base_ranking_weight

        # VAE-A components (chaotic regime)
        self.encoder_A = TernaryEncoderA(input_dim, latent_dim)
        self.decoder_A = TernaryDecoderA(latent_dim, input_dim)

        # VAE-B components (frozen regime)
        self.encoder_B = TernaryEncoderB(input_dim, latent_dim)
        self.decoder_B = TernaryDecoderB(latent_dim, input_dim)

        # StateNet v3 controller (metric-aware, 14D input, 5D output)
        if self.use_statenet:
            self.state_net = StateNetV3(state_dim=14, hidden_dim=48, latent_dim=10)

        # Controller state
        self.H_A_prev = None
        self.H_B_prev = None
        self.epoch = 0

        # Adaptive weights
        self.lambda1 = 0.7
        self.lambda2 = 0.7
        self.rho = rho_min
        self.lambda3 = lambda3_base
        self.ranking_weight = base_ranking_weight  # NEW: dynamic ranking weight

        # Phase tracking
        self.current_phase = 1
        self.grad_balance_achieved = False

        # Gradient magnitude tracking
        self.register_buffer("grad_norm_A_ema", torch.tensor(1.0))
        self.register_buffer("grad_norm_B_ema", torch.tensor(1.0))
        self.grad_ema_momentum = 0.9

        # Ranking correlation tracking (for trend analysis)
        self.register_buffer("r_A_ema", torch.tensor(0.5))
        self.register_buffer("r_B_ema", torch.tensor(0.5))
        self.r_ema_momentum = 0.9

        # StateNet correction history
        self.statenet_corrections = {
            "delta_lr": [],
            "delta_lambda1": [],
            "delta_lambda2": [],
            "delta_lambda3": [],
            "delta_ranking_weight": [],  # NEW
        }

    def compute_phase_scheduled_rho(self, epoch: int, phase_4_start: int = 250) -> float:
        """Compute phase-scheduled latent permeability."""
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
        """Update lambda1 and lambda2 adaptively."""
        if not self.adaptive_scheduling:
            return

        if grad_ratio < 0.8:
            self.lambda1 = min(0.95, self.lambda1 + 0.02)
            self.lambda2 = max(0.50, self.lambda2 - 0.02)
        elif grad_ratio > 1.2:
            self.lambda1 = max(0.50, self.lambda1 - 0.02)
            self.lambda2 = min(0.95, self.lambda2 + 0.02)

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

    def update_ranking_ema(self, r_A: float, r_B: float):
        """Update EMA of ranking correlations for trend tracking."""
        self.r_A_ema = self.r_ema_momentum * self.r_A_ema + (1 - self.r_ema_momentum) * r_A
        self.r_B_ema = self.r_ema_momentum * self.r_B_ema + (1 - self.r_ema_momentum) * r_B

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
        r_A: float = 0.5,  # NEW: ranking correlation VAE-A
        r_B: float = 0.5,  # NEW: ranking correlation VAE-B
    ) -> tuple:
        """Apply StateNet v3 corrections including ranking weight modulation.

        v3 UPGRADE: Now includes ranking correlation feedback in state vector.
        This allows StateNet to dynamically balance metric structure vs coverage.

        Args:
            lr: Current learning rate
            H_A: VAE-A entropy
            H_B: VAE-B entropy
            kl_A: VAE-A KL divergence
            kl_B: VAE-B KL divergence
            grad_ratio: Gradient norm ratio (A/B)
            coverage_A: VAE-A unique operations count (0-19683)
            coverage_B: VAE-B unique operations count (0-19683)
            r_A: VAE-A 3-adic ranking correlation (0-1)
            r_B: VAE-B 3-adic ranking correlation (0-1)

        Returns:
            Tuple of (corrected_lr, delta_lr, delta_lambda1, delta_lambda2,
                     delta_lambda3, delta_ranking_weight, effective_ranking_weight)
        """
        if not self.use_statenet or not self.training:
            return lr, 0.0, 0.0, 0.0, 0.0, 0.0, self.ranking_weight

        # Update ranking EMA
        self.update_ranking_ema(r_A, r_B)

        # Normalize coverage values to [0, 1]
        TOTAL_OPS = 19683
        coverage_A_norm = coverage_A / TOTAL_OPS
        coverage_B_norm = coverage_B / TOTAL_OPS
        missing_ops_norm = (TOTAL_OPS - max(coverage_A, coverage_B)) / TOTAL_OPS

        # Build 14D state vector with coverage AND ranking feedback
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
                r_A,
                r_B,
            ],  # NEW: ranking signals
            device=self.grad_norm_A_ema.device,
            dtype=torch.float32,
        )

        corrections, latent, attention = self.state_net(state_vec)
        delta_lr = corrections[0, 0]
        delta_lambda1 = corrections[0, 1]
        delta_lambda2 = corrections[0, 2]
        delta_lambda3 = corrections[0, 3]
        delta_ranking_weight = corrections[0, 4]  # NEW

        # Apply learning rate correction
        corrected_lr = lr * (1 + self.statenet_lr_scale * delta_lr.item())
        corrected_lr = max(1e-6, min(0.01, corrected_lr))

        # Apply lambda corrections
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

        # Apply ranking weight correction (NEW)
        # Base weight is modulated by StateNet output
        # With scale=0.5 and delta in [-1,1], range is [0.25, 0.75] from base 0.5
        effective_ranking_weight = self.base_ranking_weight * (1 + self.statenet_ranking_scale * delta_ranking_weight.item())
        # Clamp to reasonable range [0.1, 1.5] to allow significant modulation
        effective_ranking_weight = max(0.1, min(1.5, effective_ranking_weight))
        self.ranking_weight = effective_ranking_weight

        # Record corrections
        self.statenet_corrections["delta_lr"].append(delta_lr.item())
        self.statenet_corrections["delta_lambda1"].append(delta_lambda1.item())
        self.statenet_corrections["delta_lambda2"].append(delta_lambda2.item())
        self.statenet_corrections["delta_lambda3"].append(delta_lambda3.item())
        self.statenet_corrections["delta_ranking_weight"].append(delta_ranking_weight.item())

        return (
            corrected_lr,
            delta_lr.item(),
            delta_lambda1.item(),
            delta_lambda2.item(),
            delta_lambda3.item(),
            delta_ranking_weight.item(),
            effective_ranking_weight,
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
            "ranking_weight": self.ranking_weight,  # NEW: expose dynamic weight
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

            dist = torch.distributions.Categorical(logits=logits)
            indices = dist.sample()
            values = torch.tensor([-1.0, 0.0, 1.0], device=device)
            samples = values[indices]

        return samples

    def get_ranking_weight(self) -> float:
        """Get current effective ranking loss weight."""
        return self.ranking_weight

    def get_metric_state(self) -> dict:
        """Get current metric-related state for logging."""
        return {
            "r_A_ema": self.r_A_ema.item(),
            "r_B_ema": self.r_B_ema.item(),
            "ranking_weight": self.ranking_weight,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "lambda3": self.lambda3,
            "rho": self.rho,
            "phase": self.current_phase,
        }
