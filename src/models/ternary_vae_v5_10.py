"""Ternary VAE v5.10 - Pure Hyperbolic Geometry with Homeostatic Emergence.

Key innovations over v5.7/v5.9:
1. StateNet v4: 18D input (adds hyperbolic state), 7D output (adds hyp params)
2. Pure hyperbolic geometry: No Euclidean contamination
3. Homeostatic emergence: Both VAEs self-regulate for algebraic convergence
4. Inherits ALL v5.7 features: metric attention, dynamic ranking weight

Architecture:
- Dual-VAE with StateNet v4 controller (hyperbolic-aware)
- Stop-gradient cross-injection (from v5.6)
- Adaptive gradient balance (from v5.6)
- Phase-scheduled permeability (from v5.6)
- Cyclic entropy alignment (from v5.6)
- Metric attention head (from v5.7)
- Dynamic ranking weight modulation (from v5.7)
- Hyperbolic parameter modulation (NEW in v5.10)

StateNet Evolution:
- v2 (v5.6): 12D input [H, KL, grad, rho, lambda, coverage] -> 4D [lr, lambda1-3]
- v3 (v5.7): 14D input [+r_A, r_B] -> 5D [+ranking_weight]
- v4 (v5.10): 18D input [+mean_radius_A/B, prior_sigma, curvature] -> 7D [+sigma, curvature]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


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
            nn.ReLU()
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
            nn.Linear(64, output_dim * 3)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(z)
        logits = logits.view(-1, self.output_dim, 3)
        return logits


class ResidualBlock(nn.Module):
    """Residual block for VAE-B decoder."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

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
            nn.ReLU()
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


class StateNetV4(nn.Module):
    """Hyperbolic-Aware StateNet v4 for Pure Hyperbolic VAE.

    v4 UPGRADE: Adds hyperbolic state feedback to enable dynamic modulation
    of hyperbolic geometry parameters (prior_sigma, curvature).

    Inherits from v3: metric attention, ranking correlation feedback
    Adds: hyperbolic state awareness for homeostatic emergence

    Input: state vector (18D):
        [H_A, H_B, KL_A, KL_B, grad_ratio, rho, lambda1, lambda2, lambda3,
         coverage_A_norm, coverage_B_norm, missing_ops_norm,
         r_A, r_B,                          <- v3: ranking correlations
         mean_radius_A, mean_radius_B,      <- v4: hyperbolic radii
         prior_sigma, curvature]            <- v4: current hyperbolic params

    Latent: compressed state representation (12D, up from 10D)
    Output: corrections (7D):
        [delta_lr, delta_lambda1, delta_lambda2, delta_lambda3,
         delta_ranking_weight,              <- v3
         delta_sigma, delta_curvature]      <- v4: hyperbolic param corrections
    """

    def __init__(self, state_dim: int = 18, hidden_dim: int = 64, latent_dim: int = 12):
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
            nn.Linear(hidden_dim, latent_dim)
        )

        # Decoder: latent -> corrections (7D output)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 7),  # [delta_lr, delta_lambda1-3, delta_ranking, delta_sigma, delta_curvature]
            nn.Tanh()  # Output in [-1, 1] for bounded corrections
        )

        # Metric-specific attention head (inherited from v3)
        # Learns which state dimensions most affect metrics
        self.metric_attention = nn.Sequential(
            nn.Linear(state_dim, 10),  # Expanded for hyperbolic dims
            nn.Softmax(dim=-1)
        )

        # Hyperbolic-specific attention head (new in v4)
        # Learns which state dimensions affect hyperbolic params
        self.hyperbolic_attention = nn.Sequential(
            nn.Linear(state_dim, 6),  # For hyperbolic-specific weighting
            nn.Softmax(dim=-1)
        )

    def forward(self, state: torch.Tensor) -> tuple:
        """
        Args:
            state: Training state tensor [batch, 18] or [18]

        Returns:
            corrections: [delta_lr, delta_lambda1-3, delta_ranking, delta_sigma, delta_curvature]
            latent: Compressed state representation
            attention: Dict with metric and hyperbolic attention weights
        """
        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Normalize state for stable gradients
        normalized_state = state.clone()
        normalized_state[:, 0:2] = state[:, 0:2] / 3.0  # H_A, H_B typically 0-3
        # r_A, r_B already in [0,1]
        # mean_radius_A, mean_radius_B already in [0,1] (within Poincare ball)
        normalized_state[:, 16:17] = state[:, 16:17] / 2.0  # prior_sigma typically 0.3-2.0
        normalized_state[:, 17:18] = state[:, 17:18] / 4.0  # curvature typically 0.5-4.0

        # Compute metric attention (v3 style)
        metric_attention = self.metric_attention(normalized_state)

        # Compute hyperbolic attention (v4 new)
        hyp_attention = self.hyperbolic_attention(normalized_state)

        # Expand attention to match state dimensions (10 -> 18 for metric)
        metric_attention_expanded = torch.cat([
            metric_attention[:, 0:2].repeat(1, 1),   # For entropy (H_A, H_B)
            metric_attention[:, 2:4].repeat(1, 1),   # For KL (kl_A, kl_B)
            metric_attention[:, 4:5],                # For grad_ratio
            metric_attention[:, 4:5],                # For rho
            metric_attention[:, 5:6].repeat(1, 3),   # For lambdas
            metric_attention[:, 6:7].repeat(1, 3),   # For coverage
            metric_attention[:, 7:8].repeat(1, 2),   # For ranking (r_A, r_B)
            metric_attention[:, 8:9].repeat(1, 2),   # For radii (mean_radius_A, mean_radius_B)
            metric_attention[:, 9:10].repeat(1, 2),  # For hyperbolic params
        ], dim=1)

        # Apply attention-weighted encoding
        attended_state = normalized_state * metric_attention_expanded

        # Encode attended state to latent
        latent = self.encoder(attended_state)

        # Decode to corrections
        corrections = self.decoder(latent)

        attention = {
            'metric': metric_attention,
            'hyperbolic': hyp_attention
        }

        return corrections, latent, attention


class DualNeuralVAEV5_10(nn.Module):
    """Adaptive Dual-Neural VAE v5.10 with Hyperbolic-Aware StateNet v4.

    Inherits ALL features from v5.6/v5.7:
    - Dual-VAE structure (VAE-A chaotic, VAE-B frozen)
    - Stop-gradient cross-injection
    - Adaptive gradient balance
    - Phase-scheduled permeability (rho)
    - Cyclic entropy alignment (lambda3)
    - Coverage-aware adaptation
    - Metric attention (v5.7)
    - Dynamic ranking weight modulation (v5.7)

    NEW in v5.10:
    - StateNet v4 with hyperbolic state feedback
    - Hyperbolic parameter modulation (sigma, curvature)
    - Pure hyperbolic geometry support (no Euclidean contamination)
    """

    def __init__(
        self,
        input_dim: int = 9,
        latent_dim: int = 16,
        rho_min: float = 0.1,
        rho_max: float = 0.7,
        lambda3_base: float = 0.3,
        lambda3_amplitude: float = 0.15,
        eps_kl: float = 0.0005,
        gradient_balance: bool = True,
        adaptive_scheduling: bool = True,
        use_statenet: bool = True,
        statenet_lr_scale: float = 0.1,
        statenet_lambda_scale: float = 0.02,
        statenet_ranking_scale: float = 0.3,
        statenet_hyp_sigma_scale: float = 0.05,
        statenet_hyp_curvature_scale: float = 0.02
    ):
        """Initialize Dual-Neural VAE v5.10.

        Args:
            input_dim: Input dimension (9 for ternary operations)
            latent_dim: Latent space dimension
            rho_min: Minimum permeability (phase 1)
            rho_max: Maximum permeability (phase 4)
            lambda3_base: Base entropy alignment weight
            lambda3_amplitude: Amplitude for cyclic modulation
            eps_kl: KL divergence threshold for collapse detection
            gradient_balance: Enable adaptive gradient balancing
            adaptive_scheduling: Enable phase-scheduled adaptation
            use_statenet: Enable StateNet v4 controller
            statenet_lr_scale: Scale for StateNet learning rate correction
            statenet_lambda_scale: Scale for StateNet lambda corrections
            statenet_ranking_scale: Scale for ranking weight correction (v5.7)
            statenet_hyp_sigma_scale: Scale for hyperbolic sigma correction (v5.10)
            statenet_hyp_curvature_scale: Scale for curvature correction (v5.10)
        """
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
        self.statenet_hyp_sigma_scale = statenet_hyp_sigma_scale
        self.statenet_hyp_curvature_scale = statenet_hyp_curvature_scale

        # VAE-A: Chaotic regime (explores boundary)
        self.encoder_A = TernaryEncoderA(input_dim, latent_dim)
        self.decoder_A = TernaryDecoderA(latent_dim, input_dim)

        # VAE-B: Frozen regime (anchors near origin)
        self.encoder_B = TernaryEncoderB(input_dim, latent_dim)
        self.decoder_B = TernaryDecoderB(latent_dim, input_dim)

        # StateNet v4: Hyperbolic-aware controller
        if use_statenet:
            self.statenet = StateNetV4(state_dim=18, hidden_dim=64, latent_dim=12)
        else:
            self.statenet = None

        # Gradient norm tracking (for adaptive balance)
        self.register_buffer('grad_norm_A_ema', torch.tensor(1.0))
        self.register_buffer('grad_norm_B_ema', torch.tensor(1.0))

        # Ranking correlation EMA (for StateNet v3/v4)
        self.register_buffer('r_A_ema', torch.tensor(0.5))
        self.register_buffer('r_B_ema', torch.tensor(0.5))

        # Hyperbolic state tracking (for StateNet v4)
        self.register_buffer('mean_radius_A', torch.tensor(0.5))
        self.register_buffer('mean_radius_B', torch.tensor(0.5))
        self.register_buffer('prior_sigma', torch.tensor(1.0))
        self.register_buffer('curvature', torch.tensor(2.0))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute output distribution entropy."""
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
        return entropy

    def get_phase_params(self, epoch: int, total_epochs: int = 300) -> dict:
        """Get phase-scheduled parameters (from v5.6)."""
        # Phase transitions
        phase1_end = 40    # Entropy expansion
        phase2_end = 120   # Consolidation
        phase3_end = 250   # Resonant coupling
        # Phase 4: Ultra-exploration (250+)

        if epoch < phase1_end:
            phase = 1
            rho = self.rho_min
            lambda1, lambda2 = 1.0, 1.0
        elif epoch < phase2_end:
            phase = 2
            progress = (epoch - phase1_end) / (phase2_end - phase1_end)
            rho = self.rho_min + 0.2 * progress
            lambda1, lambda2 = 1.0, 1.0
        elif epoch < phase3_end:
            phase = 3
            progress = (epoch - phase2_end) / (phase3_end - phase2_end)
            rho = self.rho_min + 0.2 + 0.4 * progress
            lambda1, lambda2 = 1.0, 1.0
        else:
            phase = 4
            rho = self.rho_max
            lambda1, lambda2 = 1.0, 1.0

        # Cyclic lambda3 (entropy alignment)
        cycle_progress = (epoch % 30) / 30.0
        lambda3 = self.lambda3_base + self.lambda3_amplitude * math.cos(2 * math.pi * cycle_progress)

        return {
            'phase': phase,
            'rho': rho,
            'lambda1': lambda1,
            'lambda2': lambda2,
            'lambda3': lambda3
        }

    def build_state_vector(
        self,
        H_A: torch.Tensor,
        H_B: torch.Tensor,
        kl_A: torch.Tensor,
        kl_B: torch.Tensor,
        rho: float,
        lambda1: float,
        lambda2: float,
        lambda3: float,
        coverage_A: float = 0.0,
        coverage_B: float = 0.0,
        r_A: float = 0.5,
        r_B: float = 0.5,
        mean_radius_A: float = 0.5,
        mean_radius_B: float = 0.5,
        prior_sigma: float = 1.0,
        curvature: float = 2.0
    ) -> torch.Tensor:
        """Build 18D state vector for StateNet v4."""
        device = H_A.device

        # Compute gradient ratio
        grad_ratio = self.grad_norm_A_ema / (self.grad_norm_B_ema + 1e-8)

        # Normalize coverage to [0, 1]
        coverage_A_norm = coverage_A / 19683.0
        coverage_B_norm = coverage_B / 19683.0
        missing_ops = 19683 - max(coverage_A, coverage_B)
        missing_ops_norm = missing_ops / 19683.0

        state = torch.tensor([
            H_A.item() if torch.is_tensor(H_A) else H_A,
            H_B.item() if torch.is_tensor(H_B) else H_B,
            kl_A.item() if torch.is_tensor(kl_A) else kl_A,
            kl_B.item() if torch.is_tensor(kl_B) else kl_B,
            grad_ratio.item() if torch.is_tensor(grad_ratio) else grad_ratio,
            rho,
            lambda1,
            lambda2,
            lambda3,
            coverage_A_norm,
            coverage_B_norm,
            missing_ops_norm,
            r_A,                 # v3: ranking correlation A
            r_B,                 # v3: ranking correlation B
            mean_radius_A,       # v4: hyperbolic radius A
            mean_radius_B,       # v4: hyperbolic radius B
            prior_sigma,         # v4: current prior sigma
            curvature            # v4: current curvature
        ], device=device)

        return state

    def apply_statenet_corrections(
        self,
        corrections: torch.Tensor,
        base_lr: float,
        lambda1: float,
        lambda2: float,
        lambda3: float,
        ranking_weight: float,
        prior_sigma: float,
        curvature: float
    ) -> dict:
        """Apply StateNet v4 corrections to hyperparameters.

        Returns:
            Dict with corrected values for all modulated params
        """
        corrections = corrections.squeeze()

        # Extract corrections (7D output)
        delta_lr = corrections[0].item() * self.statenet_lr_scale
        delta_lambda1 = corrections[1].item() * self.statenet_lambda_scale
        delta_lambda2 = corrections[2].item() * self.statenet_lambda_scale
        delta_lambda3 = corrections[3].item() * self.statenet_lambda_scale
        delta_ranking = corrections[4].item() * self.statenet_ranking_scale
        delta_sigma = corrections[5].item() * self.statenet_hyp_sigma_scale
        delta_curvature = corrections[6].item() * self.statenet_hyp_curvature_scale

        # Apply corrections with bounds
        corrected_lr = base_lr * (1 + delta_lr)
        corrected_lr = max(1e-5, min(0.01, corrected_lr))

        corrected_lambda1 = max(0.1, min(2.0, lambda1 + delta_lambda1))
        corrected_lambda2 = max(0.1, min(2.0, lambda2 + delta_lambda2))
        corrected_lambda3 = max(0.0, min(1.0, lambda3 + delta_lambda3))
        corrected_ranking = max(0.1, min(0.8, ranking_weight + delta_ranking))

        # Hyperbolic param corrections (v4)
        corrected_sigma = max(0.3, min(2.0, prior_sigma + delta_sigma))
        corrected_curvature = max(0.5, min(4.0, curvature + delta_curvature))

        return {
            'lr': corrected_lr,
            'lambda1': corrected_lambda1,
            'lambda2': corrected_lambda2,
            'lambda3': corrected_lambda3,
            'ranking_weight': corrected_ranking,
            'prior_sigma': corrected_sigma,
            'curvature': corrected_curvature,
            'delta_lr': delta_lr,
            'delta_ranking': delta_ranking,
            'delta_sigma': delta_sigma,
            'delta_curvature': delta_curvature
        }

    def update_ranking_ema(self, r_A: float, r_B: float, alpha: float = 0.95):
        """Update ranking correlation EMA (for StateNet v3/v4)."""
        self.r_A_ema = alpha * self.r_A_ema + (1 - alpha) * r_A
        self.r_B_ema = alpha * self.r_B_ema + (1 - alpha) * r_B

    def update_hyperbolic_state(
        self,
        mean_radius_A: float,
        mean_radius_B: float,
        prior_sigma: float,
        curvature: float,
        alpha: float = 0.95
    ):
        """Update hyperbolic state tracking (for StateNet v4)."""
        self.mean_radius_A = alpha * self.mean_radius_A + (1 - alpha) * mean_radius_A
        self.mean_radius_B = alpha * self.mean_radius_B + (1 - alpha) * mean_radius_B
        self.prior_sigma = torch.tensor(prior_sigma)
        self.curvature = torch.tensor(curvature)

    def forward(
        self,
        x: torch.Tensor,
        beta_A: float,
        beta_B: float,
        temp_A: float,
        temp_B: float,
        rho: float = 0.5
    ) -> dict:
        """Forward pass through both VAEs with cross-injection.

        Args:
            x: Input ternary operations (batch_size, 9)
            beta_A: KL weight for VAE-A
            beta_B: KL weight for VAE-B
            temp_A: Temperature for VAE-A
            temp_B: Temperature for VAE-B
            rho: Permeability for cross-injection

        Returns:
            Dict with all outputs and intermediate values
        """
        batch_size = x.size(0)

        # Encode
        mu_A, logvar_A = self.encoder_A(x)
        mu_B, logvar_B = self.encoder_B(x)

        # Sample latent
        z_A = self.reparameterize(mu_A, logvar_A)
        z_B = self.reparameterize(mu_B, logvar_B)

        # Cross-injection with stop-gradient
        z_A_injected = (1 - rho) * z_A + rho * z_B.detach()
        z_B_injected = (1 - rho) * z_B + rho * z_A.detach()

        # Decode
        logits_A = self.decoder_A(z_A_injected) / temp_A
        logits_B = self.decoder_B(z_B_injected) / temp_B

        # Compute entropies
        H_A = self.compute_entropy(logits_A)
        H_B = self.compute_entropy(logits_B)

        return {
            'logits_A': logits_A,
            'logits_B': logits_B,
            'mu_A': mu_A,
            'mu_B': mu_B,
            'logvar_A': logvar_A,
            'logvar_B': logvar_B,
            'z_A': z_A,
            'z_B': z_B,
            'z_A_injected': z_A_injected,
            'z_B_injected': z_B_injected,
            'H_A': H_A,
            'H_B': H_B,
            'beta_A': beta_A,
            'beta_B': beta_B,
            'temp_A': temp_A,
            'temp_B': temp_B,
            'rho': rho
        }


# Backward compatibility alias
DualNeuralVAEV5 = DualNeuralVAEV5_10
