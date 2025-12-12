"""Training script for Ternary VAE v5.10 - Pure Hyperbolic Geometry.

Key innovations:
1. HyperbolicPrior: Wrapped Normal on Poincare ball (replaces Gaussian KL)
2. HyperbolicReconLoss: Radius-weighted CE (replaces flat MSE)
3. HyperbolicCentroidLoss: Frechet mean tree structure enforcement
4. HomeostaticAdaptation: Both VAEs self-regulate for algebraic convergence
5. No Euclidean contamination: pure hyperbolic geometry throughout

Usage:
    python scripts/train/train_ternary_v5_10.py --config configs/ternary_v5_10.yaml
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import yaml
import argparse
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.ternary_vae_v5_10 import DualNeuralVAEV5_10, StateNetV4
from src.training import TernaryVAETrainer
from src.data import generate_all_ternary_operations, TernaryOperationDataset
from src.losses.padic_losses import PAdicRankingLossHyperbolic
from src.losses.hyperbolic_prior import HomeostaticHyperbolicPrior
from src.losses.hyperbolic_recon import HomeostaticReconLoss, HyperbolicCentroidLoss


def setup_logging(config_path: str, log_dir: str = "logs") -> logging.Logger:
    """Setup dual logging to console and file with timestamp."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = Path(config_path).stem
    log_file = log_path / f"{config_name}_{timestamp}.log"

    # Create logger
    logger = logging.getLogger("ternary_vae")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging to: {log_file}")
    return logger


def compute_ranking_correlation_hyperbolic(model, device, n_samples=5000, max_norm=0.95, curvature=2.0):
    """Compute 3-adic ranking correlation using Poincare distance.

    Returns the concordance rate between 3-adic distance ordering
    and Poincare distance ordering in the latent space.
    """
    was_training = model.training
    model.eval()

    def project_to_poincare(z, max_norm=0.95):
        """Project points onto Poincare ball."""
        norm = torch.norm(z, dim=1, keepdim=True)
        return z / (1 + norm) * max_norm

    def poincare_distance(x, y, c=1.0):
        """Compute Poincare distance with curvature c."""
        x_norm_sq = torch.sum(x ** 2, dim=1)
        y_norm_sq = torch.sum(y ** 2, dim=1)
        diff_norm_sq = torch.sum((x - y) ** 2, dim=1)
        denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
        denom = torch.clamp(denom, min=1e-10)
        arg = 1 + 2 * c * diff_norm_sq / denom
        arg = torch.clamp(arg, min=1.0 + 1e-7)
        return (1 / np.sqrt(c)) * torch.acosh(arg)

    with torch.no_grad():
        indices = torch.randint(0, 19683, (n_samples,), device=device)

        # Convert to ternary
        ternary_data = torch.zeros(n_samples, 9, device=device)
        for i in range(9):
            ternary_data[:, i] = ((indices // (3**i)) % 3) - 1

        # Forward pass
        outputs = model(ternary_data.float(), 1.0, 1.0, 0.5, 0.5)
        z_A = outputs['z_A']
        z_B = outputs['z_B']

        # Project to Poincare ball
        z_A_hyp = project_to_poincare(z_A, max_norm)
        z_B_hyp = project_to_poincare(z_B, max_norm)

        # Compute mean radius (for homeostatic monitoring)
        mean_radius_A = torch.norm(z_A_hyp, dim=1).mean().item()
        mean_radius_B = torch.norm(z_B_hyp, dim=1).mean().item()

        # Sample triplets
        n_triplets = 1000
        i_idx = torch.randint(n_samples, (n_triplets,), device=device)
        j_idx = torch.randint(n_samples, (n_triplets,), device=device)
        k_idx = torch.randint(n_samples, (n_triplets,), device=device)

        # Filter distinct
        valid = (i_idx != j_idx) & (i_idx != k_idx) & (j_idx != k_idx)
        i_idx, j_idx, k_idx = i_idx[valid], j_idx[valid], k_idx[valid]

        if len(i_idx) < 100:
            if was_training:
                model.train()
            return 0.5, 0.5, 0.5, 0.5, mean_radius_A, mean_radius_B

        # Compute 3-adic valuations
        def compute_valuation(diff):
            val = torch.zeros_like(diff, dtype=torch.float32)
            remaining = diff.clone()
            for _ in range(10):
                mask = (remaining % 3 == 0) & (remaining > 0)
                val[mask] += 1
                remaining[mask] = remaining[mask] // 3
            val[diff == 0] = 10.0
            return val

        diff_ij = torch.abs(indices[i_idx] - indices[j_idx])
        diff_ik = torch.abs(indices[i_idx] - indices[k_idx])

        v_ij = compute_valuation(diff_ij)
        v_ik = compute_valuation(diff_ik)

        # 3-adic: larger valuation = smaller distance
        padic_closer_ij = (v_ij > v_ik).float()

        # Poincare distances with curvature
        d_A_ij = poincare_distance(z_A_hyp[i_idx], z_A_hyp[j_idx], curvature)
        d_A_ik = poincare_distance(z_A_hyp[i_idx], z_A_hyp[k_idx], curvature)
        d_B_ij = poincare_distance(z_B_hyp[i_idx], z_B_hyp[j_idx], curvature)
        d_B_ik = poincare_distance(z_B_hyp[i_idx], z_B_hyp[k_idx], curvature)

        # Also compute Euclidean for comparison
        d_A_ij_euc = torch.norm(z_A[i_idx] - z_A[j_idx], dim=1)
        d_A_ik_euc = torch.norm(z_A[i_idx] - z_A[k_idx], dim=1)
        d_B_ij_euc = torch.norm(z_B[i_idx] - z_B[j_idx], dim=1)
        d_B_ik_euc = torch.norm(z_B[i_idx] - z_B[k_idx], dim=1)

        # Hyperbolic correlations
        latent_A_closer_hyp = (d_A_ij < d_A_ik).float()
        latent_B_closer_hyp = (d_B_ij < d_B_ik).float()
        corr_A_hyp = (padic_closer_ij == latent_A_closer_hyp).float().mean().item()
        corr_B_hyp = (padic_closer_ij == latent_B_closer_hyp).float().mean().item()

        # Euclidean correlations
        latent_A_closer_euc = (d_A_ij_euc < d_A_ik_euc).float()
        latent_B_closer_euc = (d_B_ij_euc < d_B_ik_euc).float()
        corr_A_euc = (padic_closer_ij == latent_A_closer_euc).float().mean().item()
        corr_B_euc = (padic_closer_ij == latent_B_closer_euc).float().mean().item()

    if was_training:
        model.train()

    return corr_A_hyp, corr_B_hyp, corr_A_euc, corr_B_euc, mean_radius_A, mean_radius_B


class PureHyperbolicTrainer:
    """Trainer for v5.10 Pure Hyperbolic geometry with homeostatic adaptation."""

    def __init__(self, base_trainer, model, device, config):
        self.base_trainer = base_trainer
        self.model = model
        self.device = device
        self.config = config

        # Continuous feedback config
        self.feedback_config = config.get('continuous_feedback', {})
        self.feedback_enabled = self.feedback_config.get('enabled', True)

        if self.feedback_enabled:
            self.base_ranking_weight = self.feedback_config.get('base_ranking_weight', 0.5)
            self.coverage_threshold = self.feedback_config.get('coverage_threshold', 90.0)
            self.coverage_sensitivity = self.feedback_config.get('coverage_sensitivity', 0.1)
            self.coverage_trend_sensitivity = self.feedback_config.get('coverage_trend_sensitivity', 2.0)
            self.min_ranking_weight = self.feedback_config.get('min_ranking_weight', 0.0)
            self.max_ranking_weight = self.feedback_config.get('max_ranking_weight', 1.0)
            self.coverage_ema_alpha = self.feedback_config.get('coverage_ema_alpha', 0.9)
        else:
            self.base_ranking_weight = 0.5

        # PAdicRankingLossHyperbolic (v5.9)
        padic_config = config.get('padic_losses', {})
        if padic_config.get('enable_ranking_loss_hyperbolic', False):
            hyp_config = padic_config.get('ranking_hyperbolic', {})
            self.ranking_loss_hyp = PAdicRankingLossHyperbolic(
                base_margin=hyp_config.get('base_margin', 0.05),
                margin_scale=hyp_config.get('margin_scale', 0.15),
                n_triplets=hyp_config.get('n_triplets', 500),
                hard_negative_ratio=hyp_config.get('hard_negative_ratio', 0.5),
                curvature=hyp_config.get('curvature', 2.0),
                radial_weight=hyp_config.get('radial_weight', 0.4),
                max_norm=hyp_config.get('max_norm', 0.95)
            )
            self.max_norm = hyp_config.get('max_norm', 0.95)
            self.curvature = hyp_config.get('curvature', 2.0)
        else:
            self.ranking_loss_hyp = None
            self.max_norm = 0.95
            self.curvature = 2.0

        # v5.10: Pure Hyperbolic Modules
        hyp_v10 = padic_config.get('hyperbolic_v10', {})

        # Hyperbolic Prior (replaces Gaussian KL)
        if hyp_v10.get('use_hyperbolic_prior', False):
            prior_config = hyp_v10.get('prior', {})
            self.hyperbolic_prior_A = HomeostaticHyperbolicPrior(
                latent_dim=prior_config.get('latent_dim', 16),
                curvature=prior_config.get('curvature', 2.0),
                prior_sigma=prior_config.get('prior_sigma', 1.0),
                max_norm=prior_config.get('max_norm', 0.95)
            ).to(device)
            self.hyperbolic_prior_B = HomeostaticHyperbolicPrior(
                latent_dim=prior_config.get('latent_dim', 16),
                curvature=prior_config.get('curvature', 2.0),
                prior_sigma=prior_config.get('prior_sigma', 1.0),
                max_norm=prior_config.get('max_norm', 0.95)
            ).to(device)
            self.use_hyperbolic_prior = True
        else:
            self.hyperbolic_prior_A = None
            self.hyperbolic_prior_B = None
            self.use_hyperbolic_prior = False

        # Hyperbolic Reconstruction
        if hyp_v10.get('use_hyperbolic_recon', False):
            recon_config = hyp_v10.get('recon', {})
            self.hyperbolic_recon_A = HomeostaticReconLoss(
                mode=recon_config.get('mode', 'weighted_ce'),
                curvature=recon_config.get('curvature', 2.0),
                max_norm=recon_config.get('max_norm', 0.95),
                radius_weighting=recon_config.get('radius_weighting', True),
                radius_power=recon_config.get('radius_power', 2.0)
            ).to(device)
            self.hyperbolic_recon_B = HomeostaticReconLoss(
                mode=recon_config.get('mode', 'weighted_ce'),
                curvature=recon_config.get('curvature', 2.0),
                max_norm=recon_config.get('max_norm', 0.95),
                radius_weighting=recon_config.get('radius_weighting', True),
                radius_power=recon_config.get('radius_power', 2.0)
            ).to(device)
            self.hyperbolic_recon_weight = recon_config.get('weight', 0.5)
            self.use_hyperbolic_recon = True
        else:
            self.hyperbolic_recon_A = None
            self.hyperbolic_recon_B = None
            self.use_hyperbolic_recon = False

        # Hyperbolic Centroid Loss
        if hyp_v10.get('use_centroid_loss', False):
            centroid_config = hyp_v10.get('centroid', {})
            self.centroid_loss = HyperbolicCentroidLoss(
                max_level=centroid_config.get('max_level', 4),
                curvature=centroid_config.get('curvature', 2.0),
                max_norm=centroid_config.get('max_norm', 0.95)
            ).to(device)
            self.centroid_loss_weight = centroid_config.get('weight', 0.2)
            self.use_centroid_loss = True
        else:
            self.centroid_loss = None
            self.use_centroid_loss = False

        # EMA for coverage tracking
        self.coverage_ema = None
        self.prev_coverage = None

        # Evaluation intervals (now actually used!)
        self.coverage_check_interval = config.get('coverage_check_interval', 5)
        self.eval_interval = config.get('eval_interval', 20)

        # Cached values for non-evaluation epochs
        self.cached_cov_A = 0.0
        self.cached_cov_B = 0.0
        self.cached_unique_A = 0
        self.cached_unique_B = 0
        self.cached_corr_A_hyp = 0.5
        self.cached_corr_B_hyp = 0.5
        self.cached_corr_A_euc = 0.5
        self.cached_corr_B_euc = 0.5
        self.cached_mean_radius_A = 0.0
        self.cached_mean_radius_B = 0.0

        # Tracking
        self.correlation_history_hyp = []
        self.correlation_history_euc = []
        self.coverage_history = []
        self.ranking_weight_history = []
        self.radial_loss_history = []
        self.homeostatic_history = []
        self.best_corr_hyp = 0.0
        self.best_corr_euc = 0.0
        self.best_coverage = 0.0

    def compute_ranking_weight(self, current_coverage):
        """Compute ranking weight using sigmoid-based continuous feedback."""
        if not self.feedback_enabled:
            return self.base_ranking_weight

        # Update coverage EMA
        if self.coverage_ema is None:
            self.coverage_ema = current_coverage
        else:
            self.coverage_ema = (self.coverage_ema_alpha * self.coverage_ema +
                                (1 - self.coverage_ema_alpha) * current_coverage)

        # Compute coverage trend
        if self.prev_coverage is None:
            coverage_trend = 0.0
        else:
            coverage_trend = current_coverage - self.prev_coverage

        self.prev_coverage = current_coverage

        # Sigmoid modulation
        coverage_gap = current_coverage - self.coverage_threshold
        signal = (self.coverage_sensitivity * coverage_gap +
                  self.coverage_trend_sensitivity * coverage_trend)

        modulation = torch.sigmoid(torch.tensor(signal)).item()

        # Scale to [min, max] range
        weight = self.min_ranking_weight + modulation * (
            self.max_ranking_weight - self.min_ranking_weight
        )

        return weight

    def train_epoch(self, train_loader, val_loader, epoch):
        """Train one epoch with pure hyperbolic geometry and homeostatic adaptation."""
        # Coverage evaluation: only every coverage_check_interval epochs (or epoch 0)
        should_check_coverage = (epoch == 0) or (epoch % self.coverage_check_interval == 0)

        if should_check_coverage:
            unique_A, cov_A = self.base_trainer.monitor.evaluate_coverage(
                self.model, self.config['eval_num_samples'], self.device, 'A'
            )
            unique_B, cov_B = self.base_trainer.monitor.evaluate_coverage(
                self.model, self.config['eval_num_samples'], self.device, 'B'
            )
            # Cache for non-evaluation epochs
            self.cached_cov_A = cov_A
            self.cached_cov_B = cov_B
            self.cached_unique_A = unique_A
            self.cached_unique_B = unique_B
        else:
            # Use cached values
            cov_A = self.cached_cov_A
            cov_B = self.cached_cov_B
            unique_A = self.cached_unique_A
            unique_B = self.cached_unique_B

        current_coverage = (cov_A + cov_B) / 2

        # Compute adaptive ranking weight
        ranking_weight = self.compute_ranking_weight(current_coverage)
        self.ranking_weight_history.append(ranking_weight)

        # Base training
        train_losses = self.base_trainer.train_epoch(train_loader)
        val_losses = self.base_trainer.validate(val_loader)

        # Compute hyperbolic losses and metrics
        ranking_loss = 0.0
        radial_loss = 0.0
        hyp_kl_A = 0.0
        hyp_kl_B = 0.0
        hyp_recon_A = 0.0
        hyp_recon_B = 0.0
        centroid_loss_val = 0.0
        ranking_metrics = {}
        homeostatic_metrics = {}

        if ranking_weight > 0:
            self.model.eval()
            with torch.no_grad():
                # Sample for loss computation
                n_samples = min(2000, len(train_loader.dataset))
                indices = torch.randint(0, 19683, (n_samples,), device=self.device)

                ternary_data = torch.zeros(n_samples, 9, device=self.device)
                for i in range(9):
                    ternary_data[:, i] = ((indices // (3**i)) % 3) - 1

                outputs = self.model(ternary_data.float(), 1.0, 1.0, 0.5, 0.5)
                z_A = outputs['z_A']
                z_B = outputs['z_B']
                mu_A = outputs['mu_A']
                mu_B = outputs['mu_B']
                logvar_A = outputs['logvar_A']
                logvar_B = outputs['logvar_B']
                logits_A = outputs['logits_A']
                logits_B = outputs['logits_B']

            self.model.train()

            # Hyperbolic Ranking Loss (v5.9)
            if self.ranking_loss_hyp is not None:
                loss_A, metrics_A = self.ranking_loss_hyp(z_A, indices)
                loss_B, metrics_B = self.ranking_loss_hyp(z_B, indices)

                ranking_loss = ranking_weight * (loss_A.item() + loss_B.item()) / 2
                radial_loss = (metrics_A.get('radial_loss', 0) + metrics_B.get('radial_loss', 0)) / 2

                ranking_metrics = {
                    'hard_ratio': (metrics_A.get('hard_ratio', 0) + metrics_B.get('hard_ratio', 0)) / 2,
                    'violations': metrics_A.get('violations', 0) + metrics_B.get('violations', 0),
                    'radial_loss': radial_loss
                }

            # v5.10: Hyperbolic Prior KL
            if self.use_hyperbolic_prior:
                kl_A, z_hyp_A = self.hyperbolic_prior_A(mu_A, logvar_A)
                kl_B, z_hyp_B = self.hyperbolic_prior_B(mu_B, logvar_B)
                hyp_kl_A = kl_A.item()
                hyp_kl_B = kl_B.item()

                # Update homeostatic state
                self.hyperbolic_prior_A.update_homeostatic_state(z_hyp_A, kl_A, current_coverage)
                self.hyperbolic_prior_B.update_homeostatic_state(z_hyp_B, kl_B, current_coverage)

                homeostatic_metrics.update({
                    'prior_sigma_A': self.hyperbolic_prior_A.adaptive_sigma.item(),
                    'prior_sigma_B': self.hyperbolic_prior_B.adaptive_sigma.item(),
                    'prior_curvature_A': self.hyperbolic_prior_A.adaptive_curvature.item(),
                    'prior_curvature_B': self.hyperbolic_prior_B.adaptive_curvature.item()
                })

            # v5.10: Hyperbolic Reconstruction
            if self.use_hyperbolic_recon:
                recon_A, recon_metrics_A = self.hyperbolic_recon_A(logits_A, ternary_data, z_A)
                recon_B, recon_metrics_B = self.hyperbolic_recon_B(logits_B, ternary_data, z_B)
                hyp_recon_A = recon_A.item()
                hyp_recon_B = recon_B.item()

                homeostatic_metrics.update({
                    'recon_radius_A': recon_metrics_A.get('mean_radius', 0),
                    'recon_radius_B': recon_metrics_B.get('mean_radius', 0)
                })

            # v5.10: Centroid Loss
            if self.use_centroid_loss:
                cent_A, cent_metrics_A = self.centroid_loss(z_A, indices)
                cent_B, cent_metrics_B = self.centroid_loss(z_B, indices)
                centroid_loss_val = (cent_A.item() + cent_B.item()) / 2

                homeostatic_metrics.update({
                    'centroid_loss': centroid_loss_val
                })

        self.radial_loss_history.append(radial_loss)
        self.homeostatic_history.append(homeostatic_metrics)

        # Correlation evaluation: only every eval_interval epochs (or epoch 0)
        should_check_correlation = (epoch == 0) or (epoch % self.eval_interval == 0)

        if should_check_correlation:
            corr_A_hyp, corr_B_hyp, corr_A_euc, corr_B_euc, mean_radius_A, mean_radius_B = \
                compute_ranking_correlation_hyperbolic(
                    self.model, self.device,
                    max_norm=self.max_norm,
                    curvature=self.curvature
                )
            # Cache for non-evaluation epochs
            self.cached_corr_A_hyp = corr_A_hyp
            self.cached_corr_B_hyp = corr_B_hyp
            self.cached_corr_A_euc = corr_A_euc
            self.cached_corr_B_euc = corr_B_euc
            self.cached_mean_radius_A = mean_radius_A
            self.cached_mean_radius_B = mean_radius_B
        else:
            # Use cached values
            corr_A_hyp = self.cached_corr_A_hyp
            corr_B_hyp = self.cached_corr_B_hyp
            corr_A_euc = self.cached_corr_A_euc
            corr_B_euc = self.cached_corr_B_euc
            mean_radius_A = self.cached_mean_radius_A
            mean_radius_B = self.cached_mean_radius_B

        corr_mean_hyp = (corr_A_hyp + corr_B_hyp) / 2
        corr_mean_euc = (corr_A_euc + corr_B_euc) / 2

        # Track best
        if corr_mean_hyp > self.best_corr_hyp:
            self.best_corr_hyp = corr_mean_hyp
        if corr_mean_euc > self.best_corr_euc:
            self.best_corr_euc = corr_mean_euc
        if current_coverage > self.best_coverage:
            self.best_coverage = current_coverage

        # Update histories
        self.correlation_history_hyp.append(corr_mean_hyp)
        self.correlation_history_euc.append(corr_mean_euc)
        self.coverage_history.append(current_coverage)

        return {
            **train_losses,
            'ranking_weight': ranking_weight,
            'ranking_loss_hyp': ranking_loss,
            'radial_loss': radial_loss,
            'hyp_kl_A': hyp_kl_A,
            'hyp_kl_B': hyp_kl_B,
            'hyp_recon_A': hyp_recon_A,
            'hyp_recon_B': hyp_recon_B,
            'centroid_loss': centroid_loss_val,
            'corr_A_hyp': corr_A_hyp,
            'corr_B_hyp': corr_B_hyp,
            'corr_mean_hyp': corr_mean_hyp,
            'corr_A_euc': corr_A_euc,
            'corr_B_euc': corr_B_euc,
            'corr_mean_euc': corr_mean_euc,
            'cov_A': cov_A,
            'cov_B': cov_B,
            'cov_mean': current_coverage,
            'unique_A': unique_A,
            'unique_B': unique_B,
            'coverage_ema': self.coverage_ema or current_coverage,
            'mean_radius_A': mean_radius_A,
            'mean_radius_B': mean_radius_B,
            'coverage_evaluated': should_check_coverage,
            'correlation_evaluated': should_check_correlation,
            **{f'ranking_{k}': v for k, v in ranking_metrics.items()},
            **{f'homeo_{k}': v for k, v in homeostatic_metrics.items()}
        }


def main():
    parser = argparse.ArgumentParser(description='Train Ternary VAE v5.10 - Pure Hyperbolic')
    parser.add_argument('--config', type=str, default='configs/ternary_v5_10.yaml',
                        help='Path to config file')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for log files')
    args = parser.parse_args()

    # Setup logging first
    logger = setup_logging(args.config, args.log_dir)

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"{'='*80}")
    logger.info("Ternary VAE v5.10 Training - PURE HYPERBOLIC GEOMETRY")
    logger.info("Homeostatic Emergence for Algebraic Convergence")
    logger.info(f"{'='*80}")
    logger.info(f"Config: {args.config}")

    # Print v5.10 module status
    padic = config.get('padic_losses', {})
    hyp_v10 = padic.get('hyperbolic_v10', {})

    logger.info(f"\nv5.10 Modules:")
    logger.info(f"  Hyperbolic Prior: {'ENABLED' if hyp_v10.get('use_hyperbolic_prior', False) else 'DISABLED'}")
    logger.info(f"  Hyperbolic Recon: {'ENABLED' if hyp_v10.get('use_hyperbolic_recon', False) else 'DISABLED'}")
    logger.info(f"  Centroid Loss: {'ENABLED' if hyp_v10.get('use_centroid_loss', False) else 'DISABLED'}")

    if hyp_v10.get('use_hyperbolic_prior', False):
        prior = hyp_v10.get('prior', {})
        logger.info(f"    Prior curvature: {prior.get('curvature', 2.0)}")
        logger.info(f"    Prior sigma: {prior.get('prior_sigma', 1.0)}")
        logger.info(f"    Homeostatic: {prior.get('homeostatic', True)}")

    logger.info(f"\nEuclidean Contamination:")
    logger.info(f"  norm_loss: DISABLED" if not padic.get('enable_norm_loss', False) else "  norm_loss: WARNING - ENABLED")
    logger.info(f"  metric_loss: DISABLED" if not padic.get('enable_metric_loss', False) else "  metric_loss: WARNING - ENABLED")

    # Check hyperbolic ranking
    if padic.get('enable_ranking_loss_hyperbolic', False):
        hyp = padic.get('ranking_hyperbolic', {})
        logger.info(f"\nHyperbolic Ranking Loss:")
        logger.info(f"  Curvature: {hyp.get('curvature', 2.0)}")
        logger.info(f"  Radial weight: {hyp.get('radial_weight', 0.4)}")
        logger.info(f"  Max norm: {hyp.get('max_norm', 0.95)}")

    # Set seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"\nDevice: {device}")

    # Generate dataset
    logger.info("\nGenerating dataset...")
    operations = generate_all_ternary_operations()
    dataset = TernaryOperationDataset(operations)
    logger.info(f"Total operations: {len(dataset):,}")

    # Split dataset
    train_size = int(config['train_split'] * len(dataset))
    val_size = int(config['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    logger.info(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    # Initialize model (v5.10 with StateNet v4)
    model_config = config['model']
    model = DualNeuralVAEV5_10(
        input_dim=model_config['input_dim'],
        latent_dim=model_config['latent_dim'],
        rho_min=model_config['rho_min'],
        rho_max=model_config['rho_max'],
        lambda3_base=model_config['lambda3_base'],
        lambda3_amplitude=model_config['lambda3_amplitude'],
        eps_kl=model_config['eps_kl'],
        gradient_balance=model_config.get('gradient_balance', True),
        adaptive_scheduling=model_config.get('adaptive_scheduling', True),
        use_statenet=model_config.get('use_statenet', True),
        statenet_lr_scale=model_config.get('statenet_lr_scale', 0.1),
        statenet_lambda_scale=model_config.get('statenet_lambda_scale', 0.02),
        statenet_ranking_scale=model_config.get('statenet_ranking_scale', 0.3),
        statenet_hyp_sigma_scale=model_config.get('statenet_hyp_sigma_scale', 0.05),
        statenet_hyp_curvature_scale=model_config.get('statenet_hyp_curvature_scale', 0.02)
    )

    # Initialize base trainer
    base_trainer = TernaryVAETrainer(model, config, device)

    # Wrap with pure hyperbolic trainer
    trainer = PureHyperbolicTrainer(base_trainer, model, device, config)

    logger.info(f"\n{'='*80}")
    logger.info("Starting Pure Hyperbolic Training with Homeostatic Adaptation")
    logger.info(f"{'='*80}")
    logger.info(f"\nEvaluation Intervals (reduces ~10min/epoch to ~1-2min/epoch):")
    logger.info(f"  Coverage check: every {config.get('coverage_check_interval', 5)} epochs")
    logger.info(f"  Correlation check: every {config.get('eval_interval', 20)} epochs")
    logger.info(f"  Samples per check: {config.get('eval_num_samples', 1000)}")
    logger.info(f"  Training loss logged every batch (free convergence signal)\n")

    total_epochs = config['total_epochs']

    for epoch in range(total_epochs):
        base_trainer.epoch = epoch

        # Train with pure hyperbolic geometry
        losses = trainer.train_epoch(train_loader, val_loader, epoch)

        # Check for best model
        is_best = base_trainer.monitor.check_best(losses['loss'])

        # Update histories
        base_trainer.monitor.update_histories(
            losses['H_A'], losses['H_B'], losses['unique_A'], losses['unique_B']
        )

        # Print epoch summary
        cov_status = "FRESH" if losses.get('coverage_evaluated', True) else "cached"
        corr_status = "FRESH" if losses.get('correlation_evaluated', True) else "cached"

        logger.info(f"\nEpoch {epoch}/{total_epochs}")
        logger.info(f"  Loss: {losses['loss']:.4f} | Ranking Weight: {losses['ranking_weight']:.3f}")
        logger.info(f"  Coverage [{cov_status}]: A={losses['cov_A']:.1f}% B={losses['cov_B']:.1f}% (best={trainer.best_coverage:.1f}%)")
        logger.info(f"  3-Adic Correlation [{corr_status}] (Hyp): A={losses['corr_A_hyp']:.3f} B={losses['corr_B_hyp']:.3f} (best={trainer.best_corr_hyp:.3f})")
        if losses.get('correlation_evaluated', True):
            logger.info(f"  3-Adic Correlation (Euclidean):  A={losses['corr_A_euc']:.3f} B={losses['corr_B_euc']:.3f}")
        logger.info(f"  Mean Radius: A={losses['mean_radius_A']:.3f} B={losses['mean_radius_B']:.3f}")

        if losses.get('radial_loss', 0) > 0:
            logger.info(f"  Radial Loss: {losses['radial_loss']:.4f}")

        # v5.10 metrics
        if losses.get('hyp_kl_A', 0) > 0:
            logger.info(f"  Hyperbolic KL: A={losses['hyp_kl_A']:.4f} B={losses['hyp_kl_B']:.4f}")
        if losses.get('centroid_loss', 0) > 0:
            logger.info(f"  Centroid Loss: {losses['centroid_loss']:.4f}")

        # Homeostatic metrics
        if 'homeo_prior_sigma_A' in losses:
            logger.info(f"  Homeostatic Prior Sigma: A={losses['homeo_prior_sigma_A']:.3f} B={losses['homeo_prior_sigma_B']:.3f}")
            logger.info(f"  Homeostatic Curvature: A={losses['homeo_prior_curvature_A']:.3f} B={losses['homeo_prior_curvature_B']:.3f}")

        # Log to TensorBoard
        if base_trainer.monitor.writer is not None:
            writer = base_trainer.monitor.writer

            # Correlation metrics
            writer.add_scalars('Hyperbolic/CorrelationHyp', {
                'VAE_A': losses['corr_A_hyp'],
                'VAE_B': losses['corr_B_hyp'],
                'Mean': losses['corr_mean_hyp']
            }, epoch)

            writer.add_scalars('Hyperbolic/Coverage', {
                'VAE_A': losses['cov_A'],
                'VAE_B': losses['cov_B'],
                'Mean': losses['cov_mean']
            }, epoch)

            writer.add_scalars('Hyperbolic/MeanRadius', {
                'VAE_A': losses['mean_radius_A'],
                'VAE_B': losses['mean_radius_B']
            }, epoch)

            writer.add_scalar('Hyperbolic/RankingWeight', losses['ranking_weight'], epoch)
            writer.add_scalar('Hyperbolic/RadialLoss', losses.get('radial_loss', 0), epoch)

            # v5.10 metrics
            if losses.get('hyp_kl_A', 0) > 0:
                writer.add_scalars('v5.10/HyperbolicKL', {
                    'VAE_A': losses['hyp_kl_A'],
                    'VAE_B': losses['hyp_kl_B']
                }, epoch)

            if losses.get('centroid_loss', 0) > 0:
                writer.add_scalar('v5.10/CentroidLoss', losses['centroid_loss'], epoch)

            if 'homeo_prior_sigma_A' in losses:
                writer.add_scalars('v5.10/HomeostaticSigma', {
                    'VAE_A': losses['homeo_prior_sigma_A'],
                    'VAE_B': losses['homeo_prior_sigma_B']
                }, epoch)
                writer.add_scalars('v5.10/HomeostaticCurvature', {
                    'VAE_A': losses['homeo_prior_curvature_A'],
                    'VAE_B': losses['homeo_prior_curvature_B']
                }, epoch)

            writer.flush()

        # Checkpoint
        if epoch % config.get('checkpoint_freq', 10) == 0:
            checkpoint_dir = Path(config.get('checkpoint_dir', 'sandbox-training/checkpoints/v5_10'))
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': base_trainer.optimizer.state_dict(),
                'best_corr_hyp': trainer.best_corr_hyp,
                'best_coverage': trainer.best_coverage,
                'config': config
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("Training Complete - Pure Hyperbolic v5.10")
    logger.info(f"{'='*80}")
    logger.info(f"Best Hyperbolic Correlation: {trainer.best_corr_hyp:.4f}")
    logger.info(f"Best Euclidean Correlation: {trainer.best_corr_euc:.4f}")
    logger.info(f"Best Coverage: {trainer.best_coverage:.2f}%")
    logger.info(f"Target: r > 0.99, coverage > 99.7%")

    # Save final model
    checkpoint_dir = Path(config.get('checkpoint_dir', 'sandbox-training/checkpoints/v5_10'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': base_trainer.optimizer.state_dict(),
        'best_corr_hyp': trainer.best_corr_hyp,
        'best_corr_euc': trainer.best_corr_euc,
        'best_coverage': trainer.best_coverage,
        'correlation_history_hyp': trainer.correlation_history_hyp,
        'correlation_history_euc': trainer.correlation_history_euc,
        'coverage_history': trainer.coverage_history,
        'config': config
    }, checkpoint_dir / 'final_model.pt')

    logger.info(f"\nFinal model saved to: {checkpoint_dir / 'final_model.pt'}")


if __name__ == '__main__':
    main()
