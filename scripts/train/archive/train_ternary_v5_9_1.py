"""Training script for Ternary VAE v5.9.1 - FIXED GRADIENT FLOW.

Key fixes from v5.9:
1. Hyperbolic loss now properly integrated into gradient flow
2. Continuous feedback ranking weight actually modulates the loss
3. Full TensorBoard logging for hyperbolic metrics
4. StateNet corrections applied

Usage:
    python scripts/train/train_ternary_v5_9_1.py --config configs/ternary_v5_9.yaml
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
from src.losses import DualVAELoss


def compute_ranking_correlation_hyperbolic(model, device, n_samples=5000, max_norm=0.95):
    """Compute 3-adic ranking correlation using Poincare distance."""
    was_training = model.training
    model.eval()

    def project_to_poincare(z, max_norm=0.95):
        norm = torch.norm(z, dim=1, keepdim=True)
        return z / (1 + norm) * max_norm

    def poincare_distance(x, y):
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
        ternary_data = torch.zeros(n_samples, 9, device=device)
        for i in range(9):
            ternary_data[:, i] = ((indices // (3**i)) % 3) - 1

        outputs = model(ternary_data.float(), 1.0, 1.0, 0.5, 0.5)
        z_A = outputs['z_A']
        z_B = outputs['z_B']

        z_A_hyp = project_to_poincare(z_A, max_norm)
        z_B_hyp = project_to_poincare(z_B, max_norm)

        n_triplets = 1000
        i_idx = torch.randint(n_samples, (n_triplets,), device=device)
        j_idx = torch.randint(n_samples, (n_triplets,), device=device)
        k_idx = torch.randint(n_samples, (n_triplets,), device=device)

        valid = (i_idx != j_idx) & (i_idx != k_idx) & (j_idx != k_idx)
        i_idx, j_idx, k_idx = i_idx[valid], j_idx[valid], k_idx[valid]

        if len(i_idx) < 100:
            if was_training:
                model.train()
            return 0.5, 0.5, 0.5, 0.5

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

        padic_closer_ij = (v_ij > v_ik).float()

        d_A_ij = poincare_distance(z_A_hyp[i_idx], z_A_hyp[j_idx])
        d_A_ik = poincare_distance(z_A_hyp[i_idx], z_A_hyp[k_idx])
        d_B_ij = poincare_distance(z_B_hyp[i_idx], z_B_hyp[j_idx])
        d_B_ik = poincare_distance(z_B_hyp[i_idx], z_B_hyp[k_idx])

        d_A_ij_euc = torch.norm(z_A[i_idx] - z_A[j_idx], dim=1)
        d_A_ik_euc = torch.norm(z_A[i_idx] - z_A[k_idx], dim=1)
        d_B_ij_euc = torch.norm(z_B[i_idx] - z_B[j_idx], dim=1)
        d_B_ik_euc = torch.norm(z_B[i_idx] - z_B[k_idx], dim=1)

        latent_A_closer_hyp = (d_A_ij < d_A_ik).float()
        latent_B_closer_hyp = (d_B_ij < d_B_ik).float()
        corr_A_hyp = (padic_closer_ij == latent_A_closer_hyp).float().mean().item()
        corr_B_hyp = (padic_closer_ij == latent_B_closer_hyp).float().mean().item()

        latent_A_closer_euc = (d_A_ij_euc < d_A_ik_euc).float()
        latent_B_closer_euc = (d_B_ij_euc < d_B_ik_euc).float()
        corr_A_euc = (padic_closer_ij == latent_A_closer_euc).float().mean().item()
        corr_B_euc = (padic_closer_ij == latent_B_closer_euc).float().mean().item()

    if was_training:
        model.train()

    return corr_A_hyp, corr_B_hyp, corr_A_euc, corr_B_euc


class HyperbolicTrainerV2:
    """Fixed trainer with proper gradient flow for hyperbolic loss.

    Uses DualVAELoss with ranking_weight_override for continuous feedback.
    """

    def __init__(self, model, optimizer, device, config):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config

        # Initialize DualVAELoss with p-adic config (includes hyperbolic loss)
        padic_config = config.get('padic_losses', {})
        self.loss_fn = DualVAELoss(
            free_bits=config.get('free_bits', 0.5),
            repulsion_sigma=0.5,
            padic_config=padic_config
        ).to(device)

        # TorchInductor compilation
        compile_config = config.get('torch_compile', {})
        self.compiled = False
        if hasattr(torch, 'compile') and compile_config.get('enabled', False):
            try:
                backend = compile_config.get('backend', 'inductor')
                mode = compile_config.get('mode', 'default')
                fullgraph = compile_config.get('fullgraph', False)

                self.model = torch.compile(
                    self.model,
                    backend=backend,
                    mode=mode,
                    fullgraph=fullgraph
                )
                self.compiled = True
                print(f"torch.compile enabled: backend={backend}, mode={mode}")
            except Exception as e:
                print(f"Warning: torch.compile failed ({e}), falling back to eager mode")

        # Continuous feedback config
        self.feedback_config = config.get('continuous_feedback', {})
        self.feedback_enabled = self.feedback_config.get('enabled', True)
        self.base_ranking_weight = self.feedback_config.get('base_ranking_weight', 0.5)
        self.coverage_threshold = self.feedback_config.get('coverage_threshold', 90.0)
        self.coverage_sensitivity = self.feedback_config.get('coverage_sensitivity', 0.1)
        self.coverage_trend_sensitivity = self.feedback_config.get('coverage_trend_sensitivity', 2.0)
        self.min_ranking_weight = self.feedback_config.get('min_ranking_weight', 0.0)
        self.max_ranking_weight = self.feedback_config.get('max_ranking_weight', 1.0)
        self.coverage_ema_alpha = self.feedback_config.get('coverage_ema_alpha', 0.9)

        # Get max_norm for correlation computation
        hyp_config = padic_config.get('ranking_hyperbolic', {})
        self.max_norm = hyp_config.get('max_norm', 0.95)

        # EMA tracking
        self.coverage_ema = None
        self.prev_coverage = None

        # Best tracking
        self.best_corr_hyp = 0.0
        self.best_corr_euc = 0.0
        self.best_coverage = 0.0

        # Histories
        self.correlation_history_hyp = []
        self.correlation_history_euc = []
        self.coverage_history = []
        self.ranking_weight_history = []

    def compute_ranking_weight(self, current_coverage):
        """Sigmoid-based continuous feedback."""
        if not self.feedback_enabled:
            return self.base_ranking_weight

        if self.coverage_ema is None:
            self.coverage_ema = current_coverage
        else:
            self.coverage_ema = (self.coverage_ema_alpha * self.coverage_ema +
                                (1 - self.coverage_ema_alpha) * current_coverage)

        if self.prev_coverage is None:
            coverage_trend = 0.0
        else:
            coverage_trend = current_coverage - self.prev_coverage

        self.prev_coverage = current_coverage

        coverage_gap = current_coverage - self.coverage_threshold
        signal = (self.coverage_sensitivity * coverage_gap +
                  self.coverage_trend_sensitivity * coverage_trend)

        modulation = torch.sigmoid(torch.tensor(signal)).item()
        weight = self.min_ranking_weight + modulation * (
            self.max_ranking_weight - self.min_ranking_weight
        )

        return weight

    def train_epoch_with_hyperbolic(self, train_loader, epoch, current_coverage):
        """Train one epoch with PROPER hyperbolic loss gradient flow.

        Uses DualVAELoss with ranking_weight_override for continuous feedback.
        This ensures hyperbolic loss participates in backpropagation.
        """
        self.model.train()

        # Get adaptive ranking weight from continuous feedback
        ranking_weight = self.compute_ranking_weight(current_coverage)
        self.ranking_weight_history.append(ranking_weight)

        # Get schedule parameters
        total_epochs = self.config['total_epochs']
        progress = epoch / total_epochs

        # Temperature and beta scheduling
        vae_a_config = self.config['vae_a']
        vae_b_config = self.config['vae_b']

        temp_A = vae_a_config['temp_start'] + progress * (vae_a_config['temp_end'] - vae_a_config['temp_start'])
        temp_B = vae_b_config['temp_start'] + progress * (vae_b_config['temp_end'] - vae_b_config['temp_start'])

        beta_warmup_A = vae_a_config['beta_warmup_epochs']
        beta_warmup_B = vae_b_config['beta_warmup_epochs']
        beta_A = min(1.0, epoch / beta_warmup_A) * vae_a_config['beta_end'] if epoch < beta_warmup_A else vae_a_config['beta_end']
        beta_B = min(1.0, epoch / beta_warmup_B) * vae_b_config['beta_end'] if epoch < beta_warmup_B else vae_b_config['beta_end']

        entropy_weight = vae_b_config.get('entropy_weight', 0.05)
        repulsion_weight = vae_b_config.get('repulsion_weight', 0.01)

        epoch_losses = {
            'total': 0.0, 'recon_A': 0.0, 'recon_B': 0.0,
            'kl_A': 0.0, 'kl_B': 0.0, 'hyperbolic': 0.0, 'radial': 0.0
        }
        n_batches = 0

        for batch in train_loader:
            # Dataset returns only x tensor - compute indices from ternary representation
            x = batch.to(self.device)

            # Convert ternary to integer indices: sum((trit+1) * 3^i)
            trits = (x + 1).long()  # {-1,0,1} -> {0,1,2}
            powers = torch.tensor([3**i for i in range(9)], device=self.device)
            indices = (trits * powers).sum(dim=1)

            self.optimizer.zero_grad()

            # Forward pass through model
            outputs = self.model(x, temp_A, temp_B, beta_A, beta_B)

            # Compute ALL losses using DualVAELoss (including hyperbolic with override)
            losses = self.loss_fn(
                x, outputs,
                self.model.lambda1, self.model.lambda2, self.model.lambda3,
                entropy_weight, repulsion_weight,
                self.model.grad_norm_A_ema, self.model.grad_norm_B_ema,
                self.model.gradient_balance, self.model.training,
                batch_indices=indices,
                ranking_weight_override=ranking_weight  # Continuous feedback modulation
            )

            # Backward and optimize - FULL GRADIENT FLOW including hyperbolic
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))
            self.model.update_gradient_norms()
            self.optimizer.step()

            # Accumulate losses
            epoch_losses['total'] += losses['loss'].item()
            epoch_losses['recon_A'] += losses['ce_A'].item()
            epoch_losses['recon_B'] += losses['ce_B'].item()
            epoch_losses['kl_A'] += losses['kl_A'].item()
            epoch_losses['kl_B'] += losses['kl_B'].item()
            epoch_losses['hyperbolic'] += losses.get('padic_hyp_A', 0) + losses.get('padic_hyp_B', 0)
            epoch_losses['radial'] += losses.get('hyp_radial_loss', 0)
            n_batches += 1

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(1, n_batches)

        return {
            **epoch_losses,
            'ranking_weight': ranking_weight,
            'temp_A': temp_A,
            'temp_B': temp_B,
            'beta_A': beta_A,
            'beta_B': beta_B,
            'H_A': losses.get('H_A', 0),
            'H_B': losses.get('H_B', 0)
        }


def main():
    parser = argparse.ArgumentParser(description='Train Ternary VAE v5.9.1 (Fixed)')
    parser.add_argument('--config', type=str, default='configs/ternary_v5_9.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"{'='*80}")
    print("Ternary VAE v5.9.1 Training (FIXED GRADIENT FLOW)")
    print("Hyperbolic Geometry + Continuous Feedback + Proper Backprop")
    print(f"{'='*80}")
    print(f"Config: {args.config}")

    feedback = config.get('continuous_feedback', {})
    print(f"Continuous Feedback: {'ENABLED' if feedback.get('enabled', True) else 'DISABLED'}")
    print(f"  Coverage threshold: {feedback.get('coverage_threshold', 90.0)}%")

    padic = config.get('padic_losses', {})
    hyp = padic.get('ranking_hyperbolic', {})
    print(f"Hyperbolic Loss: ENABLED (with gradient flow)")
    print(f"  Curvature: {hyp.get('curvature', 1.0)}")
    print(f"  Radial weight: {hyp.get('radial_weight', 0.1)}")

    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\nGenerating dataset...")
    operations = generate_all_ternary_operations()
    dataset = TernaryOperationDataset(operations)
    print(f"Total operations: {len(dataset):,}")

    train_size = int(config['train_split'] * len(dataset))
    val_size = int(config['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

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

    # Initialize model
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
    ).to(device)

    # Initialize optimizer
    opt_config = config['optimizer']
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt_config['lr_start'],
        weight_decay=opt_config.get('weight_decay', 0.0001)
    )

    # LR scheduler
    lr_schedule = opt_config.get('lr_schedule', [])

    # Initialize base trainer for monitoring/checkpointing
    base_trainer = TernaryVAETrainer(model, config, device)

    # Initialize hyperbolic trainer
    hyp_trainer = HyperbolicTrainerV2(model, optimizer, device, config)

    print(f"\n{'='*80}")
    print("Starting Training with FIXED Gradient Flow")
    print(f"{'='*80}\n")

    total_epochs = config['total_epochs']

    for epoch in range(total_epochs):
        # Update learning rate based on schedule
        for sched in lr_schedule:
            if epoch >= sched['epoch']:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = sched['lr']

        # Evaluate coverage for continuous feedback
        unique_A, cov_A = base_trainer.monitor.evaluate_coverage(
            model, config['eval_num_samples'], device, 'A'
        )
        unique_B, cov_B = base_trainer.monitor.evaluate_coverage(
            model, config['eval_num_samples'], device, 'B'
        )
        current_coverage = (cov_A + cov_B) / 2

        # Train with hyperbolic loss (WITH GRADIENTS)
        losses = hyp_trainer.train_epoch_with_hyperbolic(train_loader, epoch, current_coverage)

        # Track coverage
        hyp_trainer.coverage_history.append(current_coverage)
        if current_coverage > hyp_trainer.best_coverage:
            hyp_trainer.best_coverage = current_coverage

        # Compute correlations
        corr_A_hyp, corr_B_hyp, corr_A_euc, corr_B_euc = compute_ranking_correlation_hyperbolic(
            model, device, max_norm=hyp_trainer.max_norm
        )
        corr_mean_hyp = (corr_A_hyp + corr_B_hyp) / 2
        corr_mean_euc = (corr_A_euc + corr_B_euc) / 2

        hyp_trainer.correlation_history_hyp.append(corr_mean_hyp)
        hyp_trainer.correlation_history_euc.append(corr_mean_euc)

        if corr_mean_hyp > hyp_trainer.best_corr_hyp:
            hyp_trainer.best_corr_hyp = corr_mean_hyp
        if corr_mean_euc > hyp_trainer.best_corr_euc:
            hyp_trainer.best_corr_euc = corr_mean_euc

        # Check best model
        is_best = base_trainer.monitor.check_best(losses['total'])

        # Update histories
        base_trainer.monitor.update_histories(
            losses.get('H_A', 0), losses.get('H_B', 0), unique_A, unique_B
        )

        # Print epoch summary
        print(f"\nEpoch {epoch}/{total_epochs}")
        print(f"  Loss: {losses['total']:.4f} | Hyperbolic: {losses['hyperbolic']:.4f} | Radial: {losses['radial']:.4f}")
        print(f"  Ranking Weight: {losses['ranking_weight']:.3f}")
        print(f"  Coverage: A={cov_A:.1f}% B={cov_B:.1f}% (best={hyp_trainer.best_coverage:.1f}%)")
        print(f"  3-Adic Correlation (Hyperbolic): {corr_mean_hyp:.3f} (best={hyp_trainer.best_corr_hyp:.3f})")
        print(f"  3-Adic Correlation (Euclidean):  {corr_mean_euc:.3f} (best={hyp_trainer.best_corr_euc:.3f})")

        # TensorBoard logging
        if base_trainer.monitor.writer is not None:
            base_trainer.monitor.writer.add_scalars('Hyperbolic/Correlation', {
                'Hyperbolic': corr_mean_hyp,
                'Euclidean': corr_mean_euc
            }, epoch)
            base_trainer.monitor.writer.add_scalars('Hyperbolic/Coverage', {
                'VAE_A': cov_A,
                'VAE_B': cov_B,
                'Mean': current_coverage
            }, epoch)
            base_trainer.monitor.writer.add_scalar('Hyperbolic/RankingWeight', losses['ranking_weight'], epoch)
            base_trainer.monitor.writer.add_scalar('Hyperbolic/HyperbolicLoss', losses['hyperbolic'], epoch)
            base_trainer.monitor.writer.add_scalar('Hyperbolic/RadialLoss', losses['radial'], epoch)
            base_trainer.monitor.writer.add_scalar('Loss/Total', losses['total'], epoch)
            base_trainer.monitor.writer.flush()

        # Save checkpoint
        if epoch % config['checkpoint_freq'] == 0:
            base_trainer.checkpoint_manager.save_checkpoint(
                epoch, model, optimizer,
                {
                    **base_trainer.monitor.get_metadata(),
                    'correlation_history_hyp': hyp_trainer.correlation_history_hyp,
                    'correlation_history_euc': hyp_trainer.correlation_history_euc,
                    'coverage_history': hyp_trainer.coverage_history,
                    'ranking_weight_history': hyp_trainer.ranking_weight_history,
                    'best_corr_hyp': hyp_trainer.best_corr_hyp,
                    'best_corr_euc': hyp_trainer.best_corr_euc,
                    'best_coverage': hyp_trainer.best_coverage
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
    print(f"Best 3-adic correlation (Hyperbolic): {hyp_trainer.best_corr_hyp:.4f}")
    print(f"Best 3-adic correlation (Euclidean):  {hyp_trainer.best_corr_euc:.4f}")
    print(f"Best coverage: {hyp_trainer.best_coverage:.1f}%")

    hyp_improvement = hyp_trainer.best_corr_hyp - hyp_trainer.best_corr_euc
    print(f"\nHyperbolic vs Euclidean improvement: {hyp_improvement:+.4f}")

    base_trainer.monitor.close()


if __name__ == '__main__':
    main()
