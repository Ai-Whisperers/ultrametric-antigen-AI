"""Training script for Ternary VAE v5.9.

Key innovations:
1. Hyperbolic geometry: PAdicRankingLossHyperbolic with Poincare distance
2. Radial hierarchy: High valuation near origin, low valuation at boundary
3. Continuous feedback: Sigmoid-based ranking weight modulation
4. No discrete phases - smooth continuous adaptation

Usage:
    python scripts/train/train_ternary_v5_9.py --config configs/ternary_v5_9.yaml
"""

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

from src.models.ternary_vae_v5_6 import DualNeuralVAEV5
from src.training import TernaryVAETrainer
from src.data import generate_all_ternary_operations, TernaryOperationDataset
from src.losses.padic_losses import PAdicRankingLossHyperbolic


def compute_ranking_correlation_hyperbolic(model, device, n_samples=5000, max_norm=0.95):
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

    def poincare_distance(x, y):
        """Compute Poincare distance."""
        x_norm_sq = torch.sum(x ** 2, dim=1)
        y_norm_sq = torch.sum(y ** 2, dim=1)
        diff_norm_sq = torch.sum((x - y) ** 2, dim=1)
        denom = (1 - x_norm_sq) * (1 - y_norm_sq)
        denom = torch.clamp(denom, min=1e-10)
        arg = 1 + 2 * diff_norm_sq / denom
        arg = torch.clamp(arg, min=1.0 + 1e-7)
        return torch.log(arg + torch.sqrt(arg ** 2 - 1))

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
            return 0.5, 0.5, 0.5, 0.5

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

        # Poincare distances
        d_A_ij = poincare_distance(z_A_hyp[i_idx], z_A_hyp[j_idx])
        d_A_ik = poincare_distance(z_A_hyp[i_idx], z_A_hyp[k_idx])
        d_B_ij = poincare_distance(z_B_hyp[i_idx], z_B_hyp[j_idx])
        d_B_ik = poincare_distance(z_B_hyp[i_idx], z_B_hyp[k_idx])

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

    return corr_A_hyp, corr_B_hyp, corr_A_euc, corr_B_euc


class ContinuousFeedbackTrainer:
    """Trainer with continuous feedback for ranking weight modulation."""

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

        # PAdicRankingLossHyperbolic
        padic_config = config.get('padic_losses', {})
        if padic_config.get('enable_ranking_loss_hyperbolic', False):
            hyp_config = padic_config.get('ranking_hyperbolic', {})
            self.ranking_loss_hyp = PAdicRankingLossHyperbolic(
                base_margin=hyp_config.get('base_margin', 0.05),
                margin_scale=hyp_config.get('margin_scale', 0.15),
                n_triplets=hyp_config.get('n_triplets', 500),
                hard_negative_ratio=hyp_config.get('hard_negative_ratio', 0.5),
                curvature=hyp_config.get('curvature', 1.0),
                radial_weight=hyp_config.get('radial_weight', 0.1),
                max_norm=hyp_config.get('max_norm', 0.95)
            )
            self.max_norm = hyp_config.get('max_norm', 0.95)
        else:
            self.ranking_loss_hyp = None
            self.max_norm = 0.95

        # EMA for coverage tracking
        self.coverage_ema = None
        self.prev_coverage = None

        # Tracking
        self.correlation_history_hyp = []
        self.correlation_history_euc = []
        self.coverage_history = []
        self.ranking_weight_history = []
        self.radial_loss_history = []
        self.best_corr_hyp = 0.0
        self.best_corr_euc = 0.0
        self.best_coverage = 0.0

    def compute_ranking_weight(self, current_coverage):
        """Compute ranking weight using sigmoid-based continuous feedback.

        ranking_weight = base_weight * sigmoid(k * (coverage - threshold) + m * d_coverage/dt)

        When coverage is high and stable: weight increases (focus on correlation)
        When coverage is dropping: weight decreases (protect coverage)
        """
        if not self.feedback_enabled:
            return self.base_ranking_weight

        # Update coverage EMA
        if self.coverage_ema is None:
            self.coverage_ema = current_coverage
        else:
            self.coverage_ema = (self.coverage_ema_alpha * self.coverage_ema +
                                (1 - self.coverage_ema_alpha) * current_coverage)

        # Compute coverage trend (d_coverage/dt)
        if self.prev_coverage is None:
            coverage_trend = 0.0
        else:
            coverage_trend = current_coverage - self.prev_coverage

        self.prev_coverage = current_coverage

        # Sigmoid modulation
        # When coverage > threshold and trend >= 0: high weight (correlation focus)
        # When coverage < threshold or trend < 0: low weight (coverage protection)
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
        """Train one epoch with continuous feedback and hyperbolic loss."""
        # Get current coverage
        unique_A, cov_A = self.base_trainer.monitor.evaluate_coverage(
            self.model, self.config['eval_num_samples'], self.device, 'A'
        )
        unique_B, cov_B = self.base_trainer.monitor.evaluate_coverage(
            self.model, self.config['eval_num_samples'], self.device, 'B'
        )
        current_coverage = (cov_A + cov_B) / 2

        # Compute adaptive ranking weight
        ranking_weight = self.compute_ranking_weight(current_coverage)
        self.ranking_weight_history.append(ranking_weight)

        # Base training
        train_losses = self.base_trainer.train_epoch(train_loader)
        val_losses = self.base_trainer.validate(val_loader)

        # Compute hyperbolic ranking loss
        ranking_loss = 0.0
        radial_loss = 0.0
        ranking_metrics = {}

        if self.ranking_loss_hyp is not None and ranking_weight > 0:
            self.model.eval()
            with torch.no_grad():
                # Sample for ranking computation
                n_samples = min(2000, len(train_loader.dataset))
                indices = torch.randint(0, 19683, (n_samples,), device=self.device)

                ternary_data = torch.zeros(n_samples, 9, device=self.device)
                for i in range(9):
                    ternary_data[:, i] = ((indices // (3**i)) % 3) - 1

                outputs = self.model(ternary_data.float(), 1.0, 1.0, 0.5, 0.5)
                z_A = outputs['z_A']
                z_B = outputs['z_B']

            self.model.train()

            # Compute hyperbolic ranking loss
            loss_A, metrics_A = self.ranking_loss_hyp(z_A, indices)
            loss_B, metrics_B = self.ranking_loss_hyp(z_B, indices)

            ranking_loss = ranking_weight * (loss_A.item() + loss_B.item()) / 2
            radial_loss = (metrics_A.get('radial_loss', 0) + metrics_B.get('radial_loss', 0)) / 2

            ranking_metrics = {
                'hard_ratio': (metrics_A.get('hard_ratio', 0) + metrics_B.get('hard_ratio', 0)) / 2,
                'violations': metrics_A.get('violations', 0) + metrics_B.get('violations', 0),
                'mean_margin': (metrics_A.get('mean_margin', 0) + metrics_B.get('mean_margin', 0)) / 2,
                'total_triplets': metrics_A.get('total_triplets', 0) + metrics_B.get('total_triplets', 0),
                'radial_loss': radial_loss
            }

        self.radial_loss_history.append(radial_loss)

        # Compute ranking correlations (both hyperbolic and Euclidean)
        corr_A_hyp, corr_B_hyp, corr_A_euc, corr_B_euc = compute_ranking_correlation_hyperbolic(
            self.model, self.device, max_norm=self.max_norm
        )
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
            **{f'ranking_{k}': v for k, v in ranking_metrics.items()}
        }


def main():
    parser = argparse.ArgumentParser(description='Train Ternary VAE v5.9')
    parser.add_argument('--config', type=str, default='configs/ternary_v5_9.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"{'='*80}")
    print("Ternary VAE v5.9 Training")
    print("Hyperbolic Geometry + Continuous Feedback")
    print(f"{'='*80}")
    print(f"Config: {args.config}")

    # Check continuous feedback config
    feedback = config.get('continuous_feedback', {})
    if feedback.get('enabled', True):
        print(f"Continuous Feedback: ENABLED")
        print(f"  Coverage threshold: {feedback.get('coverage_threshold', 90.0)}%")
        print(f"  Base ranking weight: {feedback.get('base_ranking_weight', 0.5)}")
    else:
        print("Continuous Feedback: DISABLED")

    # Check hyperbolic config
    padic = config.get('padic_losses', {})
    if padic.get('enable_ranking_loss_hyperbolic', False):
        hyp = padic.get('ranking_hyperbolic', {})
        print(f"Hyperbolic Loss: ENABLED")
        print(f"  Curvature: {hyp.get('curvature', 1.0)}")
        print(f"  Radial weight: {hyp.get('radial_weight', 0.1)}")
        print(f"  Max norm: {hyp.get('max_norm', 0.95)}")
    else:
        print("Hyperbolic Loss: DISABLED")

    # Set seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Generate dataset
    print("\nGenerating dataset...")
    operations = generate_all_ternary_operations()
    dataset = TernaryOperationDataset(operations)
    print(f"Total operations: {len(dataset):,}")

    # Split dataset
    train_size = int(config['train_split'] * len(dataset))
    val_size = int(config['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

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

    # Initialize model (v5.6 backbone)
    model_config = config['model']
    model = DualNeuralVAEV5(
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
        statenet_lr_scale=model_config.get('statenet_lr_scale', 0.05),
        statenet_lambda_scale=model_config.get('statenet_lambda_scale', 0.01)
    )

    # Initialize base trainer
    base_trainer = TernaryVAETrainer(model, config, device)

    # Wrap with continuous feedback trainer
    trainer = ContinuousFeedbackTrainer(base_trainer, model, device, config)

    print(f"\n{'='*80}")
    print("Starting Hyperbolic Training with Continuous Feedback")
    print(f"{'='*80}\n")

    total_epochs = config['total_epochs']

    for epoch in range(total_epochs):
        base_trainer.epoch = epoch

        # Train with continuous feedback
        losses = trainer.train_epoch(train_loader, val_loader, epoch)

        # Check for best model
        is_best = base_trainer.monitor.check_best(losses['loss'])

        # Update histories
        base_trainer.monitor.update_histories(
            losses['H_A'], losses['H_B'], losses['unique_A'], losses['unique_B']
        )

        # Print epoch summary
        print(f"\nEpoch {epoch}/{total_epochs}")
        print(f"  Loss: {losses['loss']:.4f} | Ranking Weight: {losses['ranking_weight']:.3f}")
        print(f"  Coverage: A={losses['cov_A']:.1f}% B={losses['cov_B']:.1f}% (best={trainer.best_coverage:.1f}%)")
        print(f"  3-Adic Correlation (Hyperbolic): A={losses['corr_A_hyp']:.3f} B={losses['corr_B_hyp']:.3f} (best={trainer.best_corr_hyp:.3f})")
        print(f"  3-Adic Correlation (Euclidean):  A={losses['corr_A_euc']:.3f} B={losses['corr_B_euc']:.3f} (best={trainer.best_corr_euc:.3f})")

        if losses.get('radial_loss', 0) > 0:
            print(f"  Radial Loss: {losses['radial_loss']:.4f}")

        if losses.get('ranking_violations', 0) > 0:
            print(f"  Ranking: violations={losses['ranking_violations']}, "
                  f"hard_ratio={losses.get('ranking_hard_ratio', 0):.2f}")

        # Log to TensorBoard if available
        if base_trainer.monitor.writer is not None:
            base_trainer.monitor.writer.add_scalars('Hyperbolic/CorrelationHyp', {
                'VAE_A': losses['corr_A_hyp'],
                'VAE_B': losses['corr_B_hyp'],
                'Mean': losses['corr_mean_hyp']
            }, epoch)
            base_trainer.monitor.writer.add_scalars('Hyperbolic/CorrelationEuc', {
                'VAE_A': losses['corr_A_euc'],
                'VAE_B': losses['corr_B_euc'],
                'Mean': losses['corr_mean_euc']
            }, epoch)
            base_trainer.monitor.writer.add_scalars('Hyperbolic/Coverage', {
                'VAE_A': losses['cov_A'],
                'VAE_B': losses['cov_B'],
                'Mean': losses['cov_mean']
            }, epoch)
            base_trainer.monitor.writer.add_scalar(
                'Hyperbolic/RankingWeight', losses['ranking_weight'], epoch
            )
            base_trainer.monitor.writer.add_scalar(
                'Hyperbolic/RadialLoss', losses.get('radial_loss', 0), epoch
            )
            base_trainer.monitor.writer.add_scalar(
                'Hyperbolic/CoverageEMA', losses['coverage_ema'], epoch
            )
            if losses.get('ranking_violations', 0) > 0:
                base_trainer.monitor.writer.add_scalar(
                    'Hyperbolic/RankingViolations', losses['ranking_violations'], epoch
                )
            base_trainer.monitor.writer.flush()

        # Save checkpoint
        if epoch % config['checkpoint_freq'] == 0:
            base_trainer.checkpoint_manager.save_checkpoint(
                epoch, model, base_trainer.optimizer,
                {
                    **base_trainer.monitor.get_metadata(),
                    'correlation_history_hyp': trainer.correlation_history_hyp,
                    'correlation_history_euc': trainer.correlation_history_euc,
                    'coverage_history': trainer.coverage_history,
                    'ranking_weight_history': trainer.ranking_weight_history,
                    'radial_loss_history': trainer.radial_loss_history,
                    'best_corr_hyp': trainer.best_corr_hyp,
                    'best_corr_euc': trainer.best_corr_euc,
                    'best_coverage': trainer.best_coverage
                },
                is_best
            )

        # Early stopping
        if base_trainer.monitor.should_stop(config['patience']):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # Summary
    print(f"\n{'='*80}")
    print("Training Complete")
    print(f"{'='*80}")
    print(f"Best 3-adic correlation (Hyperbolic): {trainer.best_corr_hyp:.4f}")
    print(f"Best 3-adic correlation (Euclidean):  {trainer.best_corr_euc:.4f}")
    print(f"Best coverage: {trainer.best_coverage:.1f}%")
    print(f"Final correlation (Hyperbolic): {trainer.correlation_history_hyp[-1]:.4f}")
    print(f"Final correlation (Euclidean):  {trainer.correlation_history_euc[-1]:.4f}")
    print(f"Final coverage: {trainer.coverage_history[-1]:.1f}%")

    # Compare hyperbolic vs Euclidean
    hyp_improvement = trainer.best_corr_hyp - trainer.best_corr_euc
    print(f"\nHyperbolic vs Euclidean improvement: {hyp_improvement:+.4f}")

    base_trainer.monitor.close()


if __name__ == '__main__':
    main()
