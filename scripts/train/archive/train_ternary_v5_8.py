"""Training script for Ternary VAE v5.8.

Key innovations:
1. Two-phase training: Coverage first, then correlation
2. PAdicRankingLossV2 with hard negative mining and hierarchical margin
3. Coverage protection in Phase 2

Usage:
    python scripts/train/train_ternary_v5_8.py --config configs/ternary_v5_8.yaml
"""

import torch
import torch.nn as nn
import torch.optim as optim
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
from src.losses.padic_losses import PAdicRankingLossV2


def compute_ranking_correlation(model, device, n_samples=5000):
    """Compute 3-adic ranking correlation.

    Returns the concordance rate between 3-adic distance ordering
    and latent space distance ordering.
    """
    was_training = model.training
    model.eval()

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
            return 0.5, 0.5

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

        # Latent distances
        d_A_ij = torch.norm(z_A[i_idx] - z_A[j_idx], dim=1)
        d_A_ik = torch.norm(z_A[i_idx] - z_A[k_idx], dim=1)
        d_B_ij = torch.norm(z_B[i_idx] - z_B[j_idx], dim=1)
        d_B_ik = torch.norm(z_B[i_idx] - z_B[k_idx], dim=1)

        latent_A_closer = (d_A_ij < d_A_ik).float()
        latent_B_closer = (d_B_ij < d_B_ik).float()

        corr_A = (padic_closer_ij == latent_A_closer).float().mean().item()
        corr_B = (padic_closer_ij == latent_B_closer).float().mean().item()

    if was_training:
        model.train()

    return corr_A, corr_B


class TwoPhaseTrainer:
    """Trainer with two-phase training strategy."""

    def __init__(self, base_trainer, model, device, config):
        self.base_trainer = base_trainer
        self.model = model
        self.device = device
        self.config = config

        # Two-phase config
        self.two_phase_config = config.get('two_phase_training', {})
        self.two_phase_enabled = self.two_phase_config.get('enabled', False)

        if self.two_phase_enabled:
            self.phase1_end = self.two_phase_config['phase1']['end_epoch']
            self.phase1_ranking_weight = self.two_phase_config['phase1']['ranking_weight']
            self.phase1_coverage_target = self.two_phase_config['phase1']['coverage_target']

            self.phase2_ranking_start = self.two_phase_config['phase2']['ranking_weight_start']
            self.phase2_ranking_end = self.two_phase_config['phase2']['ranking_weight_end']
            self.phase2_ramp_epochs = self.two_phase_config['phase2']['ranking_ramp_epochs']
            self.coverage_floor = self.two_phase_config['phase2']['coverage_floor']
            self.coverage_penalty_scale = self.two_phase_config['phase2']['coverage_penalty_scale']
        else:
            self.phase1_end = 0

        # PAdicRankingLossV2
        padic_config = config.get('padic_losses', {})
        if padic_config.get('enable_ranking_loss_v2', False):
            v2_config = padic_config.get('ranking_v2', {})
            self.ranking_loss_v2 = PAdicRankingLossV2(
                base_margin=v2_config.get('base_margin', 0.05),
                margin_scale=v2_config.get('margin_scale', 0.15),
                n_triplets=v2_config.get('n_triplets', 500),
                hard_negative_ratio=v2_config.get('hard_negative_ratio', 0.5),
                semi_hard=v2_config.get('semi_hard', True)
            )
        else:
            self.ranking_loss_v2 = None

        # Tracking
        self.correlation_history = []
        self.coverage_history = []
        self.ranking_metrics_history = []
        self.best_corr = 0.0
        self.best_coverage = 0.0

    def get_current_phase(self, epoch):
        """Determine current training phase."""
        if not self.two_phase_enabled:
            return 2  # Always Phase 2 if disabled
        return 1 if epoch < self.phase1_end else 2

    def get_ranking_weight(self, epoch, current_coverage):
        """Compute dynamic ranking weight based on phase and coverage."""
        phase = self.get_current_phase(epoch)

        if phase == 1:
            return self.phase1_ranking_weight

        # Phase 2: Ramp up ranking weight
        epochs_in_phase2 = epoch - self.phase1_end
        ramp_progress = min(1.0, epochs_in_phase2 / max(1, self.phase2_ramp_epochs))
        base_weight = self.phase2_ranking_start + ramp_progress * (
            self.phase2_ranking_end - self.phase2_ranking_start
        )

        # Coverage protection: reduce ranking weight if coverage drops below floor
        if current_coverage < self.coverage_floor:
            coverage_violation = (self.coverage_floor - current_coverage) / 100.0
            penalty = coverage_violation * self.coverage_penalty_scale
            base_weight = max(0.0, base_weight - penalty)

        return base_weight

    def train_epoch(self, train_loader, val_loader, epoch):
        """Train one epoch with two-phase logic."""
        phase = self.get_current_phase(epoch)

        # Get current coverage for weight calculation
        unique_A, cov_A = self.base_trainer.monitor.evaluate_coverage(
            self.model, self.config['eval_num_samples'], self.device, 'A'
        )
        unique_B, cov_B = self.base_trainer.monitor.evaluate_coverage(
            self.model, self.config['eval_num_samples'], self.device, 'B'
        )
        current_coverage = (cov_A + cov_B) / 2

        # Compute ranking weight
        ranking_weight = self.get_ranking_weight(epoch, current_coverage)

        # Base training
        train_losses = self.base_trainer.train_epoch(train_loader)
        val_losses = self.base_trainer.validate(val_loader)

        # Compute ranking loss with V2 if enabled
        ranking_loss = 0.0
        ranking_metrics = {}
        if self.ranking_loss_v2 is not None and ranking_weight > 0:
            # Get a batch for ranking computation
            self.model.eval()
            with torch.no_grad():
                # Sample from full dataset for ranking
                n_samples = min(2000, len(train_loader.dataset))
                indices = torch.randint(0, 19683, (n_samples,), device=self.device)

                ternary_data = torch.zeros(n_samples, 9, device=self.device)
                for i in range(9):
                    ternary_data[:, i] = ((indices // (3**i)) % 3) - 1

                outputs = self.model(ternary_data.float(), 1.0, 1.0, 0.5, 0.5)
                z_A = outputs['z_A']
                z_B = outputs['z_B']

            self.model.train()

            # Compute V2 ranking loss
            loss_A, metrics_A = self.ranking_loss_v2(z_A, indices)
            loss_B, metrics_B = self.ranking_loss_v2(z_B, indices)

            ranking_loss = ranking_weight * (loss_A.item() + loss_B.item()) / 2
            ranking_metrics = {
                'hard_ratio': (metrics_A['hard_ratio'] + metrics_B['hard_ratio']) / 2,
                'violations': metrics_A['violations'] + metrics_B['violations'],
                'mean_margin': (metrics_A['mean_margin'] + metrics_B['mean_margin']) / 2,
                'total_triplets': metrics_A['total_triplets'] + metrics_B['total_triplets']
            }

        # Compute ranking correlation
        corr_A, corr_B = compute_ranking_correlation(self.model, self.device)
        corr_mean = (corr_A + corr_B) / 2

        # Track best
        if corr_mean > self.best_corr:
            self.best_corr = corr_mean
        if current_coverage > self.best_coverage:
            self.best_coverage = current_coverage

        # Update histories
        self.correlation_history.append(corr_mean)
        self.coverage_history.append(current_coverage)
        self.ranking_metrics_history.append(ranking_metrics)

        return {
            **train_losses,
            'phase': phase,
            'ranking_weight': ranking_weight,
            'ranking_loss_v2': ranking_loss,
            'corr_A': corr_A,
            'corr_B': corr_B,
            'corr_mean': corr_mean,
            'cov_A': cov_A,
            'cov_B': cov_B,
            'cov_mean': current_coverage,
            'unique_A': unique_A,
            'unique_B': unique_B,
            **{f'ranking_{k}': v for k, v in ranking_metrics.items()}
        }


def main():
    parser = argparse.ArgumentParser(description='Train Ternary VAE v5.8')
    parser.add_argument('--config', type=str, default='configs/ternary_v5_8.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"{'='*80}")
    print("Ternary VAE v5.8 Training")
    print("Two-Phase Training + Hard Negative Mining + Hierarchical Margin")
    print(f"{'='*80}")
    print(f"Config: {args.config}")

    # Check two-phase config
    two_phase = config.get('two_phase_training', {})
    if two_phase.get('enabled', False):
        print(f"Phase 1: Epochs 0-{two_phase['phase1']['end_epoch']} (Coverage focus)")
        print(f"Phase 2: Epochs {two_phase['phase1']['end_epoch']}+ (Correlation optimization)")
    else:
        print("Two-phase training: DISABLED")

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

    # Wrap with two-phase trainer
    trainer = TwoPhaseTrainer(base_trainer, model, device, config)

    print(f"\n{'='*80}")
    print("Starting Two-Phase Training")
    print(f"{'='*80}\n")

    total_epochs = config['total_epochs']

    for epoch in range(total_epochs):
        base_trainer.epoch = epoch

        # Train with two-phase logic
        losses = trainer.train_epoch(train_loader, val_loader, epoch)

        # Check for best model
        is_best = base_trainer.monitor.check_best(losses['loss'])

        # Update histories
        base_trainer.monitor.update_histories(
            losses['H_A'], losses['H_B'], losses['unique_A'], losses['unique_B']
        )

        # Phase transition notification
        if epoch == trainer.phase1_end and trainer.two_phase_enabled:
            print(f"\n{'='*80}")
            print("PHASE TRANSITION: Coverage -> Correlation Optimization")
            print(f"Phase 1 Coverage: A={losses['cov_A']:.1f}% B={losses['cov_B']:.1f}%")
            print(f"{'='*80}\n")

        # Print epoch summary
        phase_str = f"Phase {losses['phase']}" if trainer.two_phase_enabled else "Training"
        print(f"\nEpoch {epoch}/{total_epochs} [{phase_str}]")
        print(f"  Loss: {losses['loss']:.4f} | Ranking Weight: {losses['ranking_weight']:.3f}")
        print(f"  Coverage: A={losses['cov_A']:.1f}% B={losses['cov_B']:.1f}% (best={trainer.best_coverage:.1f}%)")
        print(f"  3-Adic Correlation: A={losses['corr_A']:.3f} B={losses['corr_B']:.3f} (best={trainer.best_corr:.3f})")

        if losses.get('ranking_violations', 0) > 0:
            print(f"  Ranking V2: violations={losses['ranking_violations']}, "
                  f"hard_ratio={losses.get('ranking_hard_ratio', 0):.2f}, "
                  f"margin={losses.get('ranking_mean_margin', 0):.3f}")

        # Log to TensorBoard if available
        if base_trainer.monitor.writer is not None:
            base_trainer.monitor.writer.add_scalars('TwoPhase/Correlation', {
                'VAE_A': losses['corr_A'],
                'VAE_B': losses['corr_B'],
                'Mean': losses['corr_mean']
            }, epoch)
            base_trainer.monitor.writer.add_scalars('TwoPhase/Coverage', {
                'VAE_A': losses['cov_A'],
                'VAE_B': losses['cov_B'],
                'Mean': losses['cov_mean']
            }, epoch)
            base_trainer.monitor.writer.add_scalar(
                'TwoPhase/RankingWeight', losses['ranking_weight'], epoch
            )
            base_trainer.monitor.writer.add_scalar(
                'TwoPhase/Phase', losses['phase'], epoch
            )
            if losses.get('ranking_violations', 0) > 0:
                base_trainer.monitor.writer.add_scalar(
                    'TwoPhase/RankingViolations', losses['ranking_violations'], epoch
                )
            base_trainer.monitor.writer.flush()

        # Save checkpoint
        if epoch % config['checkpoint_freq'] == 0:
            base_trainer.checkpoint_manager.save_checkpoint(
                epoch, model, base_trainer.optimizer,
                {
                    **base_trainer.monitor.get_metadata(),
                    'correlation_history': trainer.correlation_history,
                    'coverage_history': trainer.coverage_history,
                    'best_corr': trainer.best_corr,
                    'best_coverage': trainer.best_coverage,
                    'phase': losses['phase']
                },
                is_best
            )

        # Early stopping (only in Phase 2)
        if losses['phase'] == 2:
            if base_trainer.monitor.should_stop(config['patience']):
                print(f"\nEarly stopping triggered in Phase 2")
                break

    # Summary
    print(f"\n{'='*80}")
    print("Training Complete")
    print(f"{'='*80}")
    print(f"Best 3-adic correlation: {trainer.best_corr:.4f}")
    print(f"Best coverage: {trainer.best_coverage:.1f}%")
    print(f"Final correlation: {trainer.correlation_history[-1]:.4f}")
    print(f"Final coverage: {trainer.coverage_history[-1]:.1f}%")

    base_trainer.monitor.close()


if __name__ == '__main__':
    main()
