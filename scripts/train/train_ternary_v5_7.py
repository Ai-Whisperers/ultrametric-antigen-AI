"""Training script for Ternary VAE v5.7 - Metric-Aware StateNet.

Key innovation: StateNet v3 sees ranking correlation and dynamically adjusts
ranking loss weight to balance metric structure vs coverage spread.

This script extends v5.6 training with:
- Per-epoch ranking correlation computation
- r_A, r_B passed to StateNet v3
- Dynamic ranking_weight from StateNet (replaces static config value)
- Logging of metric-related state
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
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.ternary_vae_v5_7 import DualNeuralVAEV5_7
from src.training.schedulers import TemperatureScheduler, BetaScheduler, LearningRateScheduler
from src.training.monitor import TrainingMonitor
from src.artifacts import CheckpointManager
from src.losses import DualVAELoss
from src.data import generate_all_ternary_operations, TernaryOperationDataset


def compute_ranking_correlation(model, device, n_samples=5000):
    """Compute 3-adic ranking correlation for both VAEs.

    Returns the concordance rate between 3-adic distance ordering
    and latent space distance ordering.

    Args:
        model: VAE model
        device: Device
        n_samples: Number of samples for evaluation

    Returns:
        (r_A, r_B): Ranking correlations for VAE-A and VAE-B
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

        r_A = (padic_closer_ij == latent_A_closer).float().mean().item()
        r_B = (padic_closer_ij == latent_B_closer).float().mean().item()

    # Restore training mode
    if was_training:
        model.train()

    return r_A, r_B


class MetricAwareTrainer:
    """Trainer with StateNet v3 metric awareness.

    Extends v5.6 training with ranking correlation feedback to StateNet.
    """

    def __init__(self, model: nn.Module, config: dict, device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.epoch = 0

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['optimizer']['lr_start'],
            weight_decay=config['optimizer'].get('weight_decay', 0.0001)
        )

        # Initialize schedulers
        self.temp_scheduler = TemperatureScheduler(
            config,
            config['phase_transitions']['ultra_exploration_start'],
            config['controller']['temp_lag']
        )

        self.beta_scheduler = BetaScheduler(
            config,
            config['controller']['beta_phase_lag']
        )

        self.lr_scheduler = LearningRateScheduler(
            config['optimizer']['lr_schedule']
        )

        # Initialize monitor
        self.monitor = TrainingMonitor(
            eval_num_samples=config['eval_num_samples'],
            tensorboard_dir=config.get('tensorboard_dir'),
            experiment_name=config.get('experiment_name')
        )

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            Path(config['checkpoint_dir']),
            config['checkpoint_freq']
        )

        # Initialize loss function with p-adic losses
        # NOTE: ranking_loss_weight will be overridden by model.ranking_weight
        self.loss_fn = DualVAELoss(
            free_bits=config.get('free_bits', 0.0),
            repulsion_sigma=0.5,
            padic_config=config.get('padic_losses', {})
        )

        # Cache phase 4 start
        self.phase_4_start = config['phase_transitions']['ultra_exploration_start']

        # Base-3 weights for index computation
        self._base3_weights = torch.tensor([3**i for i in range(9)], dtype=torch.long)

        # Ranking correlation history
        self.r_A_history = []
        self.r_B_history = []
        self.ranking_weight_history = []
        self.best_r = 0.0

        self._print_init_summary()

    def _print_init_summary(self):
        """Print initialization summary."""
        print(f"\n{'='*80}")
        print("DN-VAE v5.7 - Metric-Aware StateNet")
        print(f"{'='*80}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        if self.config['model'].get('use_statenet', True):
            statenet_params = sum(p.numel() for p in self.model.state_net.parameters())
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"StateNet v3 parameters: {statenet_params:,} ({statenet_params/total_params*100:.2f}%)")

        print(f"Device: {self.device}")
        print(f"StateNet v3 enabled: {self.config['model'].get('use_statenet', True)}")
        print(f"Base ranking weight: {self.config['model'].get('base_ranking_weight', 0.5)}")
        print(f"Ranking scale: {self.config['model'].get('statenet_ranking_scale', 0.1)}")

        # p-Adic losses
        padic_config = self.config.get('padic_losses', {})
        if padic_config.get('enable_ranking_loss', False):
            print(f"\np-Adic Ranking Loss:")
            print(f"  Base weight: {padic_config.get('ranking_loss_weight', 0.5)} (dynamically modulated)")
            print(f"  Margin: {padic_config.get('ranking_margin', 0.1)}")
            print(f"  Triplets/batch: {padic_config.get('ranking_n_triplets', 500)}")

    def _compute_batch_indices(self, batch_data: torch.Tensor) -> torch.Tensor:
        """Compute operation indices from ternary data."""
        digits = (batch_data + 1).long()
        weights = self._base3_weights.to(batch_data.device)
        indices = (digits * weights).sum(dim=1)
        return indices

    def _update_model_parameters(self, epoch: int):
        """Update model's adaptive parameters."""
        self.model.epoch = epoch
        self.model.rho = self.model.compute_phase_scheduled_rho(epoch, self.phase_4_start)
        self.model.lambda3 = self.model.compute_cyclic_lambda3(epoch, period=30)

        grad_ratio = (self.model.grad_norm_B_ema / (self.model.grad_norm_A_ema + 1e-8)).item()
        self.model.update_adaptive_ema_momentum(grad_ratio)

        if len(self.monitor.coverage_A_history) > 0:
            coverage_A = self.monitor.coverage_A_history[-1]
            coverage_B = self.monitor.coverage_B_history[-1]
            self.model.update_adaptive_lambdas(grad_ratio, coverage_A, coverage_B)

    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch with ranking correlation feedback."""
        self.model.train()
        self._update_model_parameters(self.epoch)

        # Get scheduled parameters
        temp_A = self.temp_scheduler.get_temperature(self.epoch, 'A')
        temp_B = self.temp_scheduler.get_temperature(self.epoch, 'B')
        beta_A = self.beta_scheduler.get_beta(self.epoch, 'A')
        beta_B = self.beta_scheduler.get_beta(self.epoch, 'B')
        lr_scheduled = self.lr_scheduler.get_lr(self.epoch)

        entropy_weight = self.config['vae_b']['entropy_weight']
        repulsion_weight = self.config['vae_b']['repulsion_weight']

        epoch_losses = defaultdict(float)
        num_batches = 0

        # Compute ranking correlation BEFORE training (for StateNet input)
        r_A, r_B = compute_ranking_correlation(self.model, self.device)
        self.r_A_history.append(r_A)
        self.r_B_history.append(r_B)

        # Track best correlation
        r_mean = (r_A + r_B) / 2
        if r_mean > self.best_r:
            self.best_r = r_mean

        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(self.device)
            batch_indices = self._compute_batch_indices(batch_data)

            # Forward pass
            outputs = self.model(batch_data, temp_A, temp_B, beta_A, beta_B)

            # Override ranking weight with model's dynamic weight
            dynamic_ranking_weight = self.model.get_ranking_weight()

            # Compute losses (with dynamic ranking weight)
            # Override the loss_fn's ranking weight with StateNet's dynamic value
            if self.loss_fn.enable_ranking_loss:
                original_weight = self.loss_fn.ranking_loss_weight
                self.loss_fn.ranking_loss_weight = dynamic_ranking_weight

            losses = self.loss_fn(
                batch_data, outputs,
                self.model.lambda1, self.model.lambda2, self.model.lambda3,
                entropy_weight, repulsion_weight,
                self.model.grad_norm_A_ema, self.model.grad_norm_B_ema,
                self.model.gradient_balance, self.model.training,
                batch_indices=batch_indices
            )

            # Restore original weight (for consistency in logging)
            if self.loss_fn.enable_ranking_loss:
                self.loss_fn.ranking_loss_weight = original_weight

            # Add rho and phase
            losses['rho'] = self.model.rho
            losses['phase'] = self.model.current_phase

            # Apply StateNet v3 corrections (with ranking correlation feedback)
            if self.model.use_statenet and batch_idx == 0:
                coverage_A = self.monitor.coverage_A_history[-1] if self.monitor.coverage_A_history else 0
                coverage_B = self.monitor.coverage_B_history[-1] if self.monitor.coverage_B_history else 0

                # v3: Pass r_A and r_B to StateNet
                result = self.model.apply_statenet_corrections(
                    lr_scheduled,
                    losses['H_A'].item() if torch.is_tensor(losses['H_A']) else losses['H_A'],
                    losses['H_B'].item() if torch.is_tensor(losses['H_B']) else losses['H_B'],
                    losses['kl_A'].item(),
                    losses['kl_B'].item(),
                    (self.model.grad_norm_B_ema / (self.model.grad_norm_A_ema + 1e-8)).item(),
                    coverage_A=coverage_A,
                    coverage_B=coverage_B,
                    r_A=r_A,  # NEW: ranking correlation
                    r_B=r_B   # NEW: ranking correlation
                )

                corrected_lr = result[0]
                delta_lr = result[1]
                delta_lambda1 = result[2]
                delta_lambda2 = result[3]
                delta_lambda3 = result[4]
                delta_ranking_weight = result[5]
                effective_ranking_weight = result[6]

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = corrected_lr

                epoch_losses['lr_corrected'] = corrected_lr
                epoch_losses['delta_lr'] = delta_lr
                epoch_losses['delta_lambda1'] = delta_lambda1
                epoch_losses['delta_lambda2'] = delta_lambda2
                epoch_losses['delta_lambda3'] = delta_lambda3
                epoch_losses['delta_ranking_weight'] = delta_ranking_weight
                epoch_losses['effective_ranking_weight'] = effective_ranking_weight

                self.ranking_weight_history.append(effective_ranking_weight)
            else:
                if batch_idx == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_scheduled

            # Backward and optimize
            self.optimizer.zero_grad()
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.model.update_gradient_norms()
            self.optimizer.step()

            # Accumulate losses
            for key, val in losses.items():
                if isinstance(val, torch.Tensor):
                    epoch_losses[key] += val.item()
                else:
                    epoch_losses[key] += val

            num_batches += 1

        # Average losses
        skip_keys = ['lr_corrected', 'delta_lr', 'delta_lambda1', 'delta_lambda2',
                     'delta_lambda3', 'delta_ranking_weight', 'effective_ranking_weight']
        for key in epoch_losses:
            if key not in skip_keys:
                epoch_losses[key] /= num_batches

        # Store schedule info and ranking correlation
        epoch_losses['temp_A'] = temp_A
        epoch_losses['temp_B'] = temp_B
        epoch_losses['beta_A'] = beta_A
        epoch_losses['beta_B'] = beta_B
        epoch_losses['lr_scheduled'] = lr_scheduled
        epoch_losses['grad_ratio'] = (self.model.grad_norm_B_ema / (self.model.grad_norm_A_ema + 1e-8)).item()
        epoch_losses['ema_momentum'] = self.model.grad_ema_momentum
        epoch_losses['r_A'] = r_A
        epoch_losses['r_B'] = r_B
        epoch_losses['r_mean'] = r_mean
        epoch_losses['best_r'] = self.best_r

        return epoch_losses

    def validate(self, val_loader: DataLoader) -> dict:
        """Validation pass."""
        self.model.eval()
        epoch_losses = defaultdict(float)
        num_batches = 0

        temp_A = self.temp_scheduler.get_temperature(self.epoch, 'A')
        temp_B = self.temp_scheduler.get_temperature(self.epoch, 'B')
        beta_A = self.beta_scheduler.get_beta(self.epoch, 'A')
        beta_B = self.beta_scheduler.get_beta(self.epoch, 'B')
        entropy_weight = self.config['vae_b']['entropy_weight']
        repulsion_weight = self.config['vae_b']['repulsion_weight']

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(self.device)
                batch_indices = self._compute_batch_indices(batch_data)

                outputs = self.model(batch_data, temp_A, temp_B, beta_A, beta_B)

                losses = self.loss_fn(
                    batch_data, outputs,
                    self.model.lambda1, self.model.lambda2, self.model.lambda3,
                    entropy_weight, repulsion_weight,
                    self.model.grad_norm_A_ema, self.model.grad_norm_B_ema,
                    self.model.gradient_balance, False,
                    batch_indices=batch_indices
                )

                losses['rho'] = self.model.rho
                losses['phase'] = self.model.current_phase

                for key, val in losses.items():
                    if isinstance(val, torch.Tensor):
                        epoch_losses[key] += val.item()
                    else:
                        epoch_losses[key] += val

                num_batches += 1

        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        print(f"\n{'='*80}")
        print("Starting DN-VAE v5.7 Training (Metric-Aware StateNet)")
        print(f"{'='*80}\n")

        total_epochs = self.config['total_epochs']

        for epoch in range(total_epochs):
            self.epoch = epoch

            # Train and validate
            train_losses = self.train_epoch(train_loader)
            val_losses = self.validate(val_loader)

            # Check for best model
            is_best = self.monitor.check_best(val_losses['loss'])

            # Evaluate coverage
            unique_A, cov_A = self.monitor.evaluate_coverage(
                self.model, self.config['eval_num_samples'], self.device, 'A'
            )
            unique_B, cov_B = self.monitor.evaluate_coverage(
                self.model, self.config['eval_num_samples'], self.device, 'B'
            )

            # Update histories
            self.monitor.update_histories(
                train_losses['H_A'], train_losses['H_B'],
                unique_A, unique_B
            )

            # Print epoch summary with ranking info
            print(f"\nEpoch {epoch}/{total_epochs}")
            print(f"  Loss: {train_losses['loss']:.4f} | Coverage: A={cov_A:.1f}% B={cov_B:.1f}%")
            print(f"  3-Adic Correlation: A={train_losses['r_A']:.3f} B={train_losses['r_B']:.3f} (best={train_losses['best_r']:.3f})")
            if 'effective_ranking_weight' in train_losses:
                delta = train_losses.get('delta_ranking_weight', 0)
                print(f"  Dynamic Ranking Weight: {train_losses['effective_ranking_weight']:.4f} (delta={delta:+.4f})")
            print(f"  Phase: {self.model.current_phase} | rho: {self.model.rho:.3f}")

            # Log to TensorBoard
            self.monitor.log_tensorboard(
                epoch, train_losses, val_losses,
                unique_A, unique_B, cov_A, cov_B
            )

            # Log ranking-specific metrics to TensorBoard
            if self.monitor.writer is not None:
                self.monitor.writer.add_scalars('Metric/RankingCorrelation', {
                    'VAE_A': train_losses['r_A'],
                    'VAE_B': train_losses['r_B'],
                    'Mean': train_losses['r_mean']
                }, epoch)
                if 'effective_ranking_weight' in train_losses:
                    self.monitor.writer.add_scalar(
                        'Metric/DynamicRankingWeight',
                        train_losses['effective_ranking_weight'],
                        epoch
                    )
                self.monitor.writer.flush()

            # Log weight histograms every 10 epochs
            if epoch % 10 == 0:
                self.monitor.log_histograms(epoch, self.model)

            # Save checkpoint with ranking metadata
            metadata = {
                **self.monitor.get_metadata(),
                'lambda1': self.model.lambda1,
                'lambda2': self.model.lambda2,
                'lambda3': self.model.lambda3,
                'rho': self.model.rho,
                'phase': self.model.current_phase,
                'grad_balance_achieved': self.model.grad_balance_achieved,
                'grad_norm_A_ema': self.model.grad_norm_A_ema.item(),
                'grad_norm_B_ema': self.model.grad_norm_B_ema.item(),
                'statenet_enabled': self.model.use_statenet,
                # v5.7 additions
                'r_A': train_losses['r_A'],
                'r_B': train_losses['r_B'],
                'best_r': self.best_r,
                'ranking_weight': self.model.ranking_weight,
                'r_A_history': self.r_A_history,
                'r_B_history': self.r_B_history,
                'ranking_weight_history': self.ranking_weight_history
            }

            if self.model.use_statenet:
                metadata['statenet_corrections'] = self.model.statenet_corrections

            self.checkpoint_manager.save_checkpoint(
                epoch, self.model, self.optimizer, metadata, is_best
            )

            # Early stopping
            if self.monitor.should_stop(self.config['patience']):
                print(f"\nEarly stopping triggered (patience={self.config['patience']})")
                break

        # Print summary
        print(f"\n{'='*80}")
        print("Training Complete")
        print(f"{'='*80}")
        print(f"Best 3-adic correlation: {self.best_r:.4f}")
        print(f"Final coverage: A={cov_A:.1f}% B={cov_B:.1f}%")
        print(f"Final ranking weight: {self.model.ranking_weight:.3f}")

        self.monitor.close()


def main():
    parser = argparse.ArgumentParser(description='Train Ternary VAE v5.7')
    parser.add_argument('--config', type=str, default='configs/ternary_v5_7.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"{'='*80}")
    print("Ternary VAE v5.7 - Metric-Aware StateNet")
    print(f"{'='*80}")
    print(f"Config: {args.config}")

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

    # Initialize model
    model_config = config['model']
    model = DualNeuralVAEV5_7(
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
        statenet_lambda_scale=model_config.get('statenet_lambda_scale', 0.01),
        statenet_ranking_scale=model_config.get('statenet_ranking_scale', 0.1),
        base_ranking_weight=model_config.get('base_ranking_weight', 0.5)
    )

    # Initialize trainer
    trainer = MetricAwareTrainer(model, config, device)

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
