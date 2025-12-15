"""Ternary VAE v5.10 - Pure Hyperbolic Geometry with Homeostatic Emergence.

Key innovations over v5.7/v5.9:
1. StateNet v4/v5: Hyperbolic + curriculum-aware state feedback
2. Pure hyperbolic geometry: No Euclidean contamination
3. Homeostatic emergence: Both VAEs self-regulate for algebraic convergence
4. Curriculum learning: Radial-first training via StateNet-controlled tau
5. Inherits ALL v5.7 features: metric attention, dynamic ranking weight

Architecture:
- Dual-VAE with StateNet v4/v5 controller (hyperbolic + curriculum aware)
- Stop-gradient cross-injection (from v5.6)
- Adaptive gradient balance (from v5.6)
- Phase-scheduled permeability (from v5.6)
- Cyclic entropy alignment (from v5.6)
- Metric attention head (from v5.7)
- Dynamic ranking weight modulation (from v5.7)
- Hyperbolic parameter modulation (v5.10)
- Curriculum control for radial→ranking transition (v5.10 + StateNet v5)

StateNet Evolution:
- v2 (v5.6): 12D input [H, KL, grad, rho, lambda, coverage] -> 4D [lr, lambda1-3]
- v3 (v5.7): 14D input [+r_A, r_B] -> 5D [+ranking_weight]
- v4 (v5.10): 18D input [+mean_radius_A/B, prior_sigma, curvature] -> 7D [+sigma, curvature]
- v5 (v5.10+): 20D input [+radial_loss, curriculum_tau] -> 8D [+delta_curriculum]
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


class StateNetV5(nn.Module):
    """Curriculum-Aware StateNet v5 for Radial-First Learning.

    v5 UPGRADE: Adds curriculum control for coordinated radial→ranking learning.
    v5.1 UPGRADE (Gap 6 fix): Adds cross-VAE correlation r_AB to detect redundancy.

    Inherits from v4: hyperbolic state feedback, metric attention
    Adds: radial_loss feedback, curriculum_tau observation, delta_curriculum output

    Input: state vector (21D):
        [H_A, H_B, KL_A, KL_B, grad_ratio, rho, lambda1, lambda2, lambda3,
         coverage_A_norm, coverage_B_norm, missing_ops_norm,
         r_A, r_B,                          <- v3: ranking correlations
         r_AB,                              <- v5.1: cross-VAE embedding correlation
         mean_radius_A, mean_radius_B,      <- v4: hyperbolic radii
         prior_sigma, curvature,            <- v4: hyperbolic params
         radial_loss_norm, curriculum_tau]  <- v5: curriculum state

    Latent: compressed state representation (14D, up from 12D)
    Output: corrections (8D):
        [delta_lr, delta_lambda1, delta_lambda2, delta_lambda3,
         delta_ranking_weight,              <- v3
         delta_sigma, delta_curvature,      <- v4
         delta_curriculum]                  <- v5: curriculum advancement signal
    """

    def __init__(self, state_dim: int = 21, hidden_dim: int = 64, latent_dim: int = 14):
        super().__init__()

        self.state_dim = state_dim
        self.latent_dim = latent_dim

        # Encoder: state -> latent (wider for curriculum awareness)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Decoder: latent -> corrections (8D output)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),  # +1 for delta_curriculum
            nn.Tanh()  # Output in [-1, 1] for bounded corrections
        )

        # Metric-specific attention head (inherited from v4)
        # v5.1: Expanded from 12 to 13 for r_AB dimension
        self.metric_attention = nn.Sequential(
            nn.Linear(state_dim, 13),  # +1 for r_AB
            nn.Softmax(dim=-1)
        )

        # Hyperbolic-specific attention head (inherited from v4)
        self.hyperbolic_attention = nn.Sequential(
            nn.Linear(state_dim, 6),
            nn.Softmax(dim=-1)
        )

        # Curriculum-specific attention head (new in v5)
        # Learns which state dimensions most affect curriculum decisions
        self.curriculum_attention = nn.Sequential(
            nn.Linear(state_dim, 4),  # Focus: radial_loss, tau, coverage, ranking
            nn.Softmax(dim=-1)
        )

        # GAP 5 FIX: Learnable attention head scaling (dynamic architecture)
        # These scalars learn which attention heads are most important
        # Initialized to equal importance (1.0 each), learned through backprop
        self.attention_head_scales = nn.ParameterDict({
            'metric': nn.Parameter(torch.tensor(1.0)),
            'hyperbolic': nn.Parameter(torch.tensor(1.0)),
            'curriculum': nn.Parameter(torch.tensor(1.0))
        })

    def forward(self, state: torch.Tensor) -> tuple:
        """
        Args:
            state: Training state tensor [batch, 21] or [21] (v5.1: includes r_AB)

        Returns:
            corrections: [delta_lr, delta_lambda1-3, delta_ranking, delta_sigma, delta_curvature, delta_curriculum]
            latent: Compressed state representation
            attention: Dict with metric, hyperbolic, and curriculum attention weights
        """
        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Normalize state for stable gradients (21D state vector for v5.1)
        # Indices: [0-1: H_A/B, 2-3: kl_A/B, 4: grad_ratio, 5: rho, 6-8: lambdas,
        #           9-11: coverage, 12-13: r_A/B, 14: r_AB, 15-16: radii, 17-18: hyp_params, 19-20: curriculum]
        normalized_state = state.clone()
        normalized_state[:, 0:2] = state[:, 0:2] / 3.0  # H_A, H_B typically 0-3
        # r_A, r_B, r_AB already in [0,1] (indices 12-14)
        # mean_radius_A, mean_radius_B already in [0,1] (indices 15-16)
        normalized_state[:, 17:18] = state[:, 17:18] / 2.0  # prior_sigma typically 0.3-2.0
        normalized_state[:, 18:19] = state[:, 18:19] / 4.0  # curvature typically 0.5-4.0
        # radial_loss_norm: normalize by expected range (0-1), at index 19
        normalized_state[:, 19:20] = torch.clamp(state[:, 19:20], 0, 2) / 2.0
        # curriculum_tau already in [0,1] at index 20

        # Compute attention weights
        metric_attention = self.metric_attention(normalized_state)
        hyp_attention = self.hyperbolic_attention(normalized_state)
        curriculum_attention = self.curriculum_attention(normalized_state)

        # GAP 5 FIX: Apply learnable attention head scaling (dynamic architecture)
        # Scales are learned through backprop, allowing network to focus on important heads
        metric_scale = torch.sigmoid(self.attention_head_scales['metric']) * 2.0  # Range [0, 2]
        hyp_scale = torch.sigmoid(self.attention_head_scales['hyperbolic']) * 2.0
        curriculum_scale = torch.sigmoid(self.attention_head_scales['curriculum']) * 2.0

        # Scale attention outputs
        metric_attention = metric_attention * metric_scale
        hyp_attention = hyp_attention * hyp_scale
        curriculum_attention = curriculum_attention * curriculum_scale

        # Expand metric attention to match state dimensions (13 -> 21 for v5.1)
        # 21D: [H_A, H_B, kl_A, kl_B, grad_ratio, rho, lambda1-3, coverage_A/B/missing,
        #       r_A, r_B, r_AB, mean_radius_A/B, prior_sigma, curvature, radial_loss, tau]
        metric_attention_expanded = torch.cat([
            metric_attention[:, 0:2].repeat(1, 1),   # For entropy (H_A, H_B) - indices 0-1
            metric_attention[:, 2:4].repeat(1, 1),   # For KL (kl_A, kl_B) - indices 2-3
            metric_attention[:, 4:5],                # For grad_ratio - index 4
            metric_attention[:, 4:5],                # For rho - index 5
            metric_attention[:, 5:6].repeat(1, 3),   # For lambdas - indices 6-8
            metric_attention[:, 6:7].repeat(1, 3),   # For coverage - indices 9-11
            metric_attention[:, 7:8].repeat(1, 2),   # For ranking (r_A, r_B) - indices 12-13
            metric_attention[:, 8:9],                # For r_AB - index 14 (v5.1 Gap 6)
            metric_attention[:, 9:10].repeat(1, 2),  # For radii (mean_radius_A/B) - indices 15-16
            metric_attention[:, 10:11].repeat(1, 2), # For hyperbolic params - indices 17-18
            metric_attention[:, 11:13],              # For curriculum (radial_loss, tau) - indices 19-20
        ], dim=1)

        # Apply attention-weighted encoding
        attended_state = normalized_state * metric_attention_expanded

        # Encode attended state to latent
        latent = self.encoder(attended_state)

        # Decode to corrections
        corrections = self.decoder(latent)

        attention = {
            'metric': metric_attention,
            'hyperbolic': hyp_attention,
            'curriculum': curriculum_attention,
            # GAP 5 FIX: Include learned attention head scales for monitoring
            'metric_scale': metric_scale.item(),
            'hyperbolic_scale': hyp_scale.item(),
            'curriculum_scale': curriculum_scale.item()
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
        statenet_version: int = 4,
        statenet_lr_scale: float = 0.1,
        statenet_lambda_scale: float = 0.02,
        statenet_ranking_scale: float = 0.3,
        statenet_hyp_sigma_scale: float = 0.05,
        statenet_hyp_curvature_scale: float = 0.02,
        statenet_curriculum_scale: float = 0.1
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
            use_statenet: Enable StateNet controller
            statenet_version: StateNet version (4 or 5, v5 adds curriculum control)
            statenet_lr_scale: Scale for StateNet learning rate correction
            statenet_lambda_scale: Scale for StateNet lambda corrections
            statenet_ranking_scale: Scale for ranking weight correction (v5.7)
            statenet_hyp_sigma_scale: Scale for hyperbolic sigma correction (v5.10)
            statenet_hyp_curvature_scale: Scale for curvature correction (v5.10)
            statenet_curriculum_scale: Scale for curriculum correction (v5, StateNet v5)
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
        self.statenet_version = statenet_version
        self.statenet_lr_scale = statenet_lr_scale
        self.statenet_lambda_scale = statenet_lambda_scale
        self.statenet_ranking_scale = statenet_ranking_scale
        self.statenet_hyp_sigma_scale = statenet_hyp_sigma_scale
        self.statenet_hyp_curvature_scale = statenet_hyp_curvature_scale
        self.statenet_curriculum_scale = statenet_curriculum_scale

        # VAE-A: Chaotic regime (explores boundary)
        self.encoder_A = TernaryEncoderA(input_dim, latent_dim)
        self.decoder_A = TernaryDecoderA(latent_dim, input_dim)

        # VAE-B: Frozen regime (anchors near origin)
        self.encoder_B = TernaryEncoderB(input_dim, latent_dim)
        self.decoder_B = TernaryDecoderB(latent_dim, input_dim)

        # StateNet v4/v5: Hyperbolic-aware controller with optional curriculum
        # v5.1: Gap 6 fix - added r_AB cross-VAE correlation (21D total)
        if use_statenet:
            if statenet_version == 5:
                self.state_net = StateNetV5(state_dim=21, hidden_dim=64, latent_dim=14)
            else:
                self.state_net = StateNetV4(state_dim=18, hidden_dim=64, latent_dim=12)
        else:
            self.state_net = None

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

        # GAP 4 FIX: Loss plateau detection for adaptive StateNet LR
        # When VAEs converge (loss plateaus), StateNet needs higher LR to explore
        self.register_buffer('loss_ema', torch.tensor(10.0))  # EMA of training loss
        self.register_buffer('loss_prev', torch.tensor(10.0))  # Previous loss (for gradient)
        self.register_buffer('loss_grad_ema', torch.tensor(1.0))  # EMA of |loss change|
        self.statenet_lr_boost = 1.0  # Current adaptive boost (1.0 = no boost)

        # Adaptive parameters (inherited from v5.6/v5.7)
        self.rho = rho_min
        self.lambda1 = 0.7
        self.lambda2 = 0.7
        self.lambda3 = lambda3_base
        self.ranking_weight = 0.4  # Dynamic ranking weight (v5.7)

        # EMA momentum parameters
        self.r_ema_momentum = 0.95
        self.grad_ema_momentum = 0.9

        # State tracking
        self.grad_balance_achieved = False
        self.current_phase = 1

        # StateNet correction history
        self.statenet_corrections = {
            'delta_lr': [],
            'delta_lambda1': [],
            'delta_lambda2': [],
            'delta_lambda3': [],
            'delta_ranking_weight': [],
            'delta_sigma': [],
            'delta_curvature': []
        }

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
            self.grad_norm_A_ema = (self.grad_ema_momentum * self.grad_norm_A_ema +
                                   (1 - self.grad_ema_momentum) * grad_norm_A)
        if grad_norm_B > 0:
            self.grad_norm_B_ema = (self.grad_ema_momentum * self.grad_norm_B_ema +
                                   (1 - self.grad_ema_momentum) * grad_norm_B)

    def get_ranking_weight(self) -> float:
        """Get current effective ranking loss weight."""
        return self.ranking_weight

    def get_metric_state(self) -> dict:
        """Get current metric-related state for logging."""
        return {
            'r_A_ema': self.r_A_ema.item() if torch.is_tensor(self.r_A_ema) else self.r_A_ema,
            'r_B_ema': self.r_B_ema.item() if torch.is_tensor(self.r_B_ema) else self.r_B_ema,
            'ranking_weight': self.ranking_weight,
            'lambda1': self.lambda1,
            'lambda2': self.lambda2,
            'lambda3': self.lambda3,
            'rho': self.rho,
            'phase': self.current_phase,
            'mean_radius_A': self.mean_radius_A.item() if torch.is_tensor(self.mean_radius_A) else self.mean_radius_A,
            'mean_radius_B': self.mean_radius_B.item() if torch.is_tensor(self.mean_radius_B) else self.mean_radius_B,
            'prior_sigma': self.prior_sigma.item() if torch.is_tensor(self.prior_sigma) else self.prior_sigma,
            'curvature': self.curvature.item() if torch.is_tensor(self.curvature) else self.curvature
        }

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
        lr: float,
        H_A: float,
        H_B: float,
        kl_A: float,
        kl_B: float,
        grad_ratio: float,
        coverage_A: int = 0,
        coverage_B: int = 0,
        r_A: float = 0.5,
        r_B: float = 0.5,
        mean_radius_A: float = 0.5,
        mean_radius_B: float = 0.5,
        prior_sigma: float = 1.0,
        curvature: float = 2.0
    ) -> tuple:
        """Apply StateNet v4 corrections including hyperbolic parameter modulation.

        v4 UPGRADE: Extends v3 with hyperbolic state feedback for sigma/curvature.

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
            mean_radius_A: VAE-A mean hyperbolic radius
            mean_radius_B: VAE-B mean hyperbolic radius
            prior_sigma: Current prior sigma
            curvature: Current curvature

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

        # Build 18D state vector (v4: with hyperbolic state)
        state_vec = torch.tensor(
            [H_A, H_B, kl_A, kl_B, grad_ratio, self.rho,
             self.lambda1, self.lambda2, self.lambda3,
             coverage_A_norm, coverage_B_norm, missing_ops_norm,
             r_A, r_B,
             mean_radius_A, mean_radius_B,
             prior_sigma, curvature],
            device=self.grad_norm_A_ema.device,
            dtype=torch.float32
        )

        corrections, latent, attention = self.state_net(state_vec)
        delta_lr = corrections[0, 0]
        delta_lambda1 = corrections[0, 1]
        delta_lambda2 = corrections[0, 2]
        delta_lambda3 = corrections[0, 3]
        delta_ranking_weight = corrections[0, 4]
        delta_sigma = corrections[0, 5]
        delta_curvature = corrections[0, 6]

        # Apply learning rate correction
        corrected_lr = lr * (1 + self.statenet_lr_scale * delta_lr.item())
        corrected_lr = max(1e-6, min(0.01, corrected_lr))

        # Apply lambda corrections
        self.lambda1 = max(0.5, min(0.95, self.lambda1 + self.statenet_lambda_scale * delta_lambda1.item()))
        self.lambda2 = max(0.5, min(0.95, self.lambda2 + self.statenet_lambda_scale * delta_lambda2.item()))
        self.lambda3 = max(0.15, min(0.75, self.lambda3 + self.statenet_lambda_scale * delta_lambda3.item()))

        # Apply ranking weight correction
        effective_ranking_weight = self.ranking_weight * (
            1 + self.statenet_ranking_scale * delta_ranking_weight.item()
        )
        effective_ranking_weight = max(0.1, min(1.5, effective_ranking_weight))
        self.ranking_weight = effective_ranking_weight

        # Record corrections
        self.statenet_corrections['delta_lr'].append(delta_lr.item())
        self.statenet_corrections['delta_lambda1'].append(delta_lambda1.item())
        self.statenet_corrections['delta_lambda2'].append(delta_lambda2.item())
        self.statenet_corrections['delta_lambda3'].append(delta_lambda3.item())
        self.statenet_corrections['delta_ranking_weight'].append(delta_ranking_weight.item())
        self.statenet_corrections['delta_sigma'].append(delta_sigma.item())
        self.statenet_corrections['delta_curvature'].append(delta_curvature.item())

        return (corrected_lr, delta_lr.item(), delta_lambda1.item(),
                delta_lambda2.item(), delta_lambda3.item(),
                delta_ranking_weight.item(), effective_ranking_weight)

    def build_state_vector_v5(
        self,
        H_A: float,
        H_B: float,
        kl_A: float,
        kl_B: float,
        rho: float,
        lambda1: float,
        lambda2: float,
        lambda3: float,
        coverage_A: float = 0.0,
        coverage_B: float = 0.0,
        r_A: float = 0.5,
        r_B: float = 0.5,
        r_AB: float = 0.0,
        mean_radius_A: float = 0.5,
        mean_radius_B: float = 0.5,
        prior_sigma: float = 1.0,
        curvature: float = 2.0,
        radial_loss: float = 0.0,
        curriculum_tau: float = 0.0
    ) -> torch.Tensor:
        """Build 21D state vector for StateNet v5.1.

        v5.1 (Gap 6 fix): Adds r_AB for cross-VAE embedding correlation.
        High r_AB means both VAEs learn similar structure (redundancy).
        Low r_AB means VAEs learn complementary representations.
        """
        device = self.grad_norm_A_ema.device

        # Compute gradient ratio
        grad_ratio = self.grad_norm_A_ema / (self.grad_norm_B_ema + 1e-8)

        # Normalize coverage to [0, 1]
        TOTAL_OPS = 19683
        coverage_A_norm = coverage_A / TOTAL_OPS
        coverage_B_norm = coverage_B / TOTAL_OPS
        missing_ops = TOTAL_OPS - max(coverage_A, coverage_B)
        missing_ops_norm = missing_ops / TOTAL_OPS

        # Normalize radial_loss (expected range 0-1, clamp at 2)
        radial_loss_norm = min(radial_loss, 2.0) / 2.0

        state = torch.tensor([
            H_A,
            H_B,
            kl_A,
            kl_B,
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
            r_AB,                # v5.1: cross-VAE embedding correlation (Gap 6 fix)
            mean_radius_A,       # v4: hyperbolic radius A
            mean_radius_B,       # v4: hyperbolic radius B
            prior_sigma,         # v4: current prior sigma
            curvature,           # v4: current curvature
            radial_loss_norm,    # v5: normalized radial stratification loss
            curriculum_tau       # v5: current curriculum position
        ], device=device, dtype=torch.float32)

        return state

    def apply_statenet_v5_corrections(
        self,
        lr: float,
        H_A: float,
        H_B: float,
        kl_A: float,
        kl_B: float,
        grad_ratio: float,
        coverage_A: int = 0,
        coverage_B: int = 0,
        r_A: float = 0.5,
        r_B: float = 0.5,
        r_AB: float = 0.0,
        mean_radius_A: float = 0.5,
        mean_radius_B: float = 0.5,
        prior_sigma: float = 1.0,
        curvature: float = 2.0,
        radial_loss: float = 0.0,
        curriculum_tau: float = 0.0
    ) -> dict:
        """Apply StateNet v5.1 corrections with curriculum control.

        v5 UPGRADE: Extends v4 with radial_loss observation and delta_curriculum output.
        v5.1 UPGRADE (Gap 6 fix): Adds r_AB for cross-VAE embedding correlation.

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
            r_AB: Cross-VAE embedding correlation (0-1), high=redundancy
            mean_radius_A: VAE-A mean hyperbolic radius
            mean_radius_B: VAE-B mean hyperbolic radius
            prior_sigma: Current prior sigma
            curvature: Current curvature
            radial_loss: Current radial stratification loss
            curriculum_tau: Current curriculum position (0-1)

        Returns:
            Dict with all corrections including delta_curriculum
        """
        if not self.use_statenet or not self.training:
            return {
                'corrected_lr': lr,
                'delta_lr': 0.0,
                'delta_lambda1': 0.0,
                'delta_lambda2': 0.0,
                'delta_lambda3': 0.0,
                'delta_ranking_weight': 0.0,
                'effective_ranking_weight': self.ranking_weight,
                'delta_sigma': 0.0,
                'delta_curvature': 0.0,
                'delta_curriculum': 0.0
            }

        if self.statenet_version != 5:
            # Fall back to v4 corrections if not using v5
            result = self.apply_statenet_corrections(
                lr, H_A, H_B, kl_A, kl_B, grad_ratio,
                coverage_A, coverage_B, r_A, r_B,
                mean_radius_A, mean_radius_B, prior_sigma, curvature
            )
            return {
                'corrected_lr': result[0],
                'delta_lr': result[1],
                'delta_lambda1': result[2],
                'delta_lambda2': result[3],
                'delta_lambda3': result[4],
                'delta_ranking_weight': result[5],
                'effective_ranking_weight': result[6],
                'delta_sigma': 0.0,
                'delta_curvature': 0.0,
                'delta_curriculum': 0.0
            }

        # Update ranking EMA
        self.update_ranking_ema(r_A, r_B)

        # Build 21D state vector (v5.1: includes r_AB for cross-VAE correlation)
        state_vec = self.build_state_vector_v5(
            H_A, H_B, kl_A, kl_B,
            self.rho, self.lambda1, self.lambda2, self.lambda3,
            coverage_A, coverage_B, r_A, r_B, r_AB,
            mean_radius_A, mean_radius_B, prior_sigma, curvature,
            radial_loss, curriculum_tau
        )

        corrections, latent, attention = self.state_net(state_vec)
        delta_lr = corrections[0, 0]
        delta_lambda1 = corrections[0, 1]
        delta_lambda2 = corrections[0, 2]
        delta_lambda3 = corrections[0, 3]
        delta_ranking_weight = corrections[0, 4]
        delta_sigma = corrections[0, 5]
        delta_curvature = corrections[0, 6]
        delta_curriculum = corrections[0, 7]

        # GAP 4 FIX: Apply learning rate correction with adaptive boost
        # statenet_lr_boost increases when loss plateaus (more aggressive corrections)
        effective_lr_scale = self.statenet_lr_scale * self.statenet_lr_boost
        corrected_lr = lr * (1 + effective_lr_scale * delta_lr.item())
        corrected_lr = max(1e-6, min(0.01, corrected_lr))

        # Apply lambda corrections
        self.lambda1 = max(0.5, min(0.95, self.lambda1 + self.statenet_lambda_scale * delta_lambda1.item()))
        self.lambda2 = max(0.5, min(0.95, self.lambda2 + self.statenet_lambda_scale * delta_lambda2.item()))
        self.lambda3 = max(0.15, min(0.75, self.lambda3 + self.statenet_lambda_scale * delta_lambda3.item()))

        # Apply ranking weight correction
        effective_ranking_weight = self.ranking_weight * (
            1 + self.statenet_ranking_scale * delta_ranking_weight.item()
        )
        effective_ranking_weight = max(0.1, min(1.5, effective_ranking_weight))
        self.ranking_weight = effective_ranking_weight

        # Record corrections (include new curriculum correction)
        self.statenet_corrections['delta_lr'].append(delta_lr.item())
        self.statenet_corrections['delta_lambda1'].append(delta_lambda1.item())
        self.statenet_corrections['delta_lambda2'].append(delta_lambda2.item())
        self.statenet_corrections['delta_lambda3'].append(delta_lambda3.item())
        self.statenet_corrections['delta_ranking_weight'].append(delta_ranking_weight.item())
        self.statenet_corrections['delta_sigma'].append(delta_sigma.item())
        self.statenet_corrections['delta_curvature'].append(delta_curvature.item())

        # Add delta_curriculum to tracking if not present
        if 'delta_curriculum' not in self.statenet_corrections:
            self.statenet_corrections['delta_curriculum'] = []
        self.statenet_corrections['delta_curriculum'].append(delta_curriculum.item())

        return {
            'corrected_lr': corrected_lr,
            'delta_lr': delta_lr.item(),
            'delta_lambda1': delta_lambda1.item(),
            'delta_lambda2': delta_lambda2.item(),
            'delta_lambda3': delta_lambda3.item(),
            'delta_ranking_weight': delta_ranking_weight.item(),
            'effective_ranking_weight': effective_ranking_weight,
            'delta_sigma': delta_sigma.item(),
            'delta_curvature': delta_curvature.item(),
            'delta_curriculum': delta_curriculum.item(),
            'statenet_lr_boost': self.statenet_lr_boost  # GAP 4 FIX: Adaptive LR boost
        }

    def update_ranking_ema(self, r_A: float, r_B: float, alpha: float = 0.7):
        """Update ranking correlation EMA (for StateNet v3/v4).

        THREE-BODY FIX: Reduced alpha from 0.95 to 0.7 for faster response.
        With alpha=0.7, correlation changes propagate in ~5 epochs vs ~20.
        """
        self.r_A_ema = alpha * self.r_A_ema + (1 - alpha) * r_A
        self.r_B_ema = alpha * self.r_B_ema + (1 - alpha) * r_B

    def update_hyperbolic_state(
        self,
        mean_radius_A: float,
        mean_radius_B: float,
        prior_sigma: float,
        curvature: float,
        alpha: float = 0.7
    ):
        """Update hyperbolic state tracking (for StateNet v4)."""
        self.mean_radius_A = alpha * self.mean_radius_A + (1 - alpha) * mean_radius_A
        self.mean_radius_B = alpha * self.mean_radius_B + (1 - alpha) * mean_radius_B
        self.prior_sigma = torch.tensor(prior_sigma)
        self.curvature = torch.tensor(curvature)

    def update_loss_plateau_detection(
        self,
        current_loss: float,
        alpha: float = 0.9,
        plateau_threshold: float = 0.05,
        max_boost: float = 3.0
    ):
        """GAP 4 FIX: Update loss plateau detection and adaptive StateNet LR boost.

        When loss plateaus (gradient near zero), increases StateNet's effective LR
        to allow more aggressive exploration of hyperparameter corrections.

        Args:
            current_loss: Current epoch's training loss
            alpha: EMA momentum for loss tracking
            plateau_threshold: Loss gradient below which we consider plateau
            max_boost: Maximum LR boost factor when fully plateaued

        Returns:
            Current boost factor for StateNet LR
        """
        # Update loss EMA
        self.loss_ema = alpha * self.loss_ema + (1 - alpha) * current_loss

        # Compute loss gradient (absolute change)
        loss_grad = abs(current_loss - self.loss_prev.item())
        self.loss_grad_ema = alpha * self.loss_grad_ema + (1 - alpha) * loss_grad
        self.loss_prev = torch.tensor(current_loss, device=self.loss_prev.device)

        # Compute plateau factor (0 = changing, 1 = fully plateaued)
        # Normalized by loss magnitude to be scale-invariant
        normalized_grad = self.loss_grad_ema.item() / (self.loss_ema.item() + 1e-6)
        plateau_factor = max(0.0, 1.0 - normalized_grad / plateau_threshold)

        # Boost StateNet LR when plateaued (linear interpolation)
        # At plateau_factor=0: boost=1.0 (no boost)
        # At plateau_factor=1: boost=max_boost
        self.statenet_lr_boost = 1.0 + (max_boost - 1.0) * plateau_factor

        return self.statenet_lr_boost

    def forward(self, x: torch.Tensor, temp_A: float = 1.0, temp_B: float = 1.0,
                beta_A: float = 1.0, beta_B: float = 1.0) -> dict:
        """Forward pass through both VAEs with cross-injection.

        Args:
            x: Input ternary operations (batch_size, 9)
            temp_A: Temperature for VAE-A
            temp_B: Temperature for VAE-B
            beta_A: KL weight for VAE-A
            beta_B: KL weight for VAE-B

        Returns:
            Dict with all outputs and intermediate values
        """
        # Encode
        mu_A, logvar_A = self.encoder_A(x)
        mu_B, logvar_B = self.encoder_B(x)

        # Sample latent
        z_A = self.reparameterize(mu_A, logvar_A)
        z_B = self.reparameterize(mu_B, logvar_B)

        # Cross-injection with stop-gradient (using self.rho)
        z_A_injected = (1 - self.rho) * z_A + self.rho * z_B.detach()
        z_B_injected = (1 - self.rho) * z_B + self.rho * z_A.detach()

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
            'rho': self.rho
        }


    def sample(self, num_samples: int, device: str = 'cpu',
               use_vae: str = 'A') -> torch.Tensor:
        """Sample from the learned manifold.

        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            use_vae: Which VAE to use ('A' or 'B')

        Returns:
            Sampled ternary operations (num_samples, 9)
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=device)

            if use_vae == 'A':
                logits = self.decoder_A(z)
            else:
                logits = self.decoder_B(z)

            # Use categorical sampling instead of expectation
            dist = torch.distributions.Categorical(logits=logits)
            indices = dist.sample()
            values = torch.tensor([-1.0, 0.0, 1.0], device=device)
            samples = values[indices]

        return samples


# Backward compatibility alias
DualNeuralVAEV5 = DualNeuralVAEV5_10
