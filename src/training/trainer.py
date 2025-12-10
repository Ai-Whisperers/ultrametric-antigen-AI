"""Refactored trainer using modular components.

This trainer delegates to specialized components for:
- Scheduling: TemperatureScheduler, BetaScheduler, LearningRateScheduler
- Monitoring: TrainingMonitor
- Checkpointing: CheckpointManager
- Compilation: torch.compile (TorchInductor) for 1.4-2x speedup

Single responsibility: Orchestrate training loop only.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict
import sys

from .schedulers import TemperatureScheduler, BetaScheduler, LearningRateScheduler
from .monitor import TrainingMonitor
from ..artifacts import CheckpointManager
from ..losses import DualVAELoss


class TernaryVAETrainer:
    """Refactored trainer with single responsibility: orchestrate training loop.

    All scheduling, monitoring, and checkpoint management delegated to components.
    """

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], device: str = 'cuda'):
        """Initialize trainer with model and config.

        Args:
            model: DualNeuralVAE model
            config: Training configuration dict
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.epoch = 0

        # TorchInductor compilation (PyTorch 2.x)
        self.compiled = False
        compile_config = config.get('torch_compile', {})
        if compile_config.get('enabled', False) and hasattr(torch, 'compile'):
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
        elif compile_config.get('enabled', False):
            print("Warning: torch.compile requested but not available (PyTorch < 2.0)")

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['optimizer']['lr_start'],
            weight_decay=config['optimizer'].get('weight_decay', 0.0001)
        )

        # Initialize components
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

        self.monitor = TrainingMonitor(
            eval_num_samples=config['eval_num_samples'],
            tensorboard_dir=config.get('tensorboard_dir'),
            experiment_name=config.get('experiment_name')
        )

        self.checkpoint_manager = CheckpointManager(
            Path(config['checkpoint_dir']),
            config['checkpoint_freq']
        )

        # Initialize loss function (with p-adic losses if configured)
        self.loss_fn = DualVAELoss(
            free_bits=config.get('free_bits', 0.0),
            repulsion_sigma=0.5,
            padic_config=config.get('padic_losses', {})
        )

        # Cache phase 4 start for model updates
        self.phase_4_start = config['phase_transitions']['ultra_exploration_start']

        # Precompute base-3 weights for index computation
        self._base3_weights = torch.tensor([3**i for i in range(9)], dtype=torch.long)

        # Print initialization summary
        self._print_init_summary()

    def _print_init_summary(self) -> None:
        """Print initialization summary."""
        print(f"\n{'='*80}")
        print("DN-VAE v5.6 Initialized")
        print(f"{'='*80}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        if self.config['model'].get('use_statenet', True):
            statenet_params = sum(p.numel() for p in self.model.state_net.parameters())
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"StateNet parameters: {statenet_params:,} ({statenet_params/total_params*100:.2f}%)")

        print(f"Device: {self.device}")
        print(f"Gradient balance: {self.config['model'].get('gradient_balance', True)}")
        print(f"Adaptive scheduling: {self.config['model'].get('adaptive_scheduling', True)}")
        print(f"StateNet enabled: {self.config['model'].get('use_statenet', True)}")
        print(f"torch.compile: {'enabled' if self.compiled else 'disabled'}")

        # p-Adic losses (Phase 1A/1B)
        padic_config = self.config.get('padic_losses', {})
        has_padic = (padic_config.get('enable_metric_loss', False) or
                     padic_config.get('enable_ranking_loss', False) or
                     padic_config.get('enable_norm_loss', False))
        if has_padic:
            print(f"\np-Adic Losses (implement.md Phase 1):")
            if padic_config.get('enable_metric_loss', False):
                print(f"  Metric Loss: weight={padic_config.get('metric_loss_weight', 0.1)}, scale={padic_config.get('metric_loss_scale', 1.0)}")
            if padic_config.get('enable_ranking_loss', False):
                print(f"  Ranking Loss: weight={padic_config.get('ranking_loss_weight', 0.5)}, margin={padic_config.get('ranking_margin', 0.1)}")
            if padic_config.get('enable_norm_loss', False):
                print(f"  Norm Loss: weight={padic_config.get('norm_loss_weight', 0.05)}")

    def _compute_batch_indices(self, batch_data: torch.Tensor) -> torch.Tensor:
        """Compute operation indices from ternary data.

        Each ternary operation is encoded as 9 digits in {-1, 0, 1}.
        The index is computed as: Σ (digit + 1) * 3^i for i in 0..8

        Args:
            batch_data: Ternary data (batch_size, 9) with values in {-1, 0, 1}

        Returns:
            Operation indices (batch_size,) in range [0, 19682]
        """
        # Convert {-1, 0, 1} to {0, 1, 2}
        digits = (batch_data + 1).long()
        # Compute index as base-3 number
        weights = self._base3_weights.to(batch_data.device)
        indices = (digits * weights).sum(dim=1)
        return indices

    def _update_model_parameters(self, epoch: int) -> None:
        """Update model's adaptive parameters for current epoch.

        Args:
            epoch: Current epoch
        """
        self.model.epoch = epoch
        self.model.rho = self.model.compute_phase_scheduled_rho(epoch, self.phase_4_start)
        self.model.lambda3 = self.model.compute_cyclic_lambda3(epoch, period=30)

        grad_ratio = (self.model.grad_norm_B_ema / (self.model.grad_norm_A_ema + 1e-8)).item()
        self.model.update_adaptive_ema_momentum(grad_ratio)

        if len(self.monitor.coverage_A_history) > 0:
            coverage_A = self.monitor.coverage_A_history[-1]
            coverage_B = self.monitor.coverage_B_history[-1]
            self.model.update_adaptive_lambdas(grad_ratio, coverage_A, coverage_B)

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, Any]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dict of epoch losses and metrics
        """
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
        free_bits = self.config.get('free_bits', 0.0)

        epoch_losses = defaultdict(float)
        num_batches = 0

        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(self.device)

            # Compute batch indices for p-adic losses
            batch_indices = self._compute_batch_indices(batch_data)

            # Forward pass
            outputs = self.model(batch_data, temp_A, temp_B, beta_A, beta_B)

            # Compute losses using DualVAELoss (with p-adic losses)
            losses = self.loss_fn(
                batch_data, outputs,
                self.model.lambda1, self.model.lambda2, self.model.lambda3,
                entropy_weight, repulsion_weight,
                self.model.grad_norm_A_ema, self.model.grad_norm_B_ema,
                self.model.gradient_balance, self.model.training,
                batch_indices=batch_indices
            )

            # Add rho and phase to loss dict for logging
            losses['rho'] = self.model.rho
            losses['phase'] = self.model.current_phase

            # Apply StateNet v2 corrections once per epoch (with coverage feedback)
            if self.model.use_statenet and batch_idx == 0:
                # Get latest coverage from monitor history for StateNet v2
                coverage_A = self.monitor.coverage_A_history[-1] if self.monitor.coverage_A_history else 0
                coverage_B = self.monitor.coverage_B_history[-1] if self.monitor.coverage_B_history else 0

                corrected_lr, *deltas = self.model.apply_statenet_corrections(
                    lr_scheduled,
                    losses['H_A'].item() if torch.is_tensor(losses['H_A']) else losses['H_A'],
                    losses['H_B'].item() if torch.is_tensor(losses['H_B']) else losses['H_B'],
                    losses['kl_A'].item(),
                    losses['kl_B'].item(),
                    (self.model.grad_norm_B_ema / (self.model.grad_norm_A_ema + 1e-8)).item(),
                    coverage_A=coverage_A,
                    coverage_B=coverage_B
                )

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = corrected_lr

                epoch_losses['lr_corrected'] = corrected_lr
                epoch_losses['delta_lr'] = deltas[0]
                epoch_losses['delta_lambda1'] = deltas[1]
                epoch_losses['delta_lambda2'] = deltas[2]
                epoch_losses['delta_lambda3'] = deltas[3]
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
        for key in epoch_losses:
            if key not in ['lr_corrected', 'delta_lr', 'delta_lambda1', 'delta_lambda2', 'delta_lambda3']:
                epoch_losses[key] /= num_batches

        # Store schedule info
        epoch_losses['temp_A'] = temp_A
        epoch_losses['temp_B'] = temp_B
        epoch_losses['beta_A'] = beta_A
        epoch_losses['beta_B'] = beta_B
        epoch_losses['lr_scheduled'] = lr_scheduled
        epoch_losses['grad_ratio'] = (self.model.grad_norm_B_ema / (self.model.grad_norm_A_ema + 1e-8)).item()
        epoch_losses['ema_momentum'] = self.model.grad_ema_momentum

        return epoch_losses

    def validate(self, val_loader: DataLoader) -> Dict[str, Any]:
        """Validation pass.

        Args:
            val_loader: Validation data loader

        Returns:
            Dict of validation losses
        """
        self.model.eval()
        epoch_losses = defaultdict(float)
        num_batches = 0

        temp_A = self.temp_scheduler.get_temperature(self.epoch, 'A')
        temp_B = self.temp_scheduler.get_temperature(self.epoch, 'B')
        beta_A = self.beta_scheduler.get_beta(self.epoch, 'A')
        beta_B = self.beta_scheduler.get_beta(self.epoch, 'B')
        entropy_weight = self.config['vae_b']['entropy_weight']
        repulsion_weight = self.config['vae_b']['repulsion_weight']
        free_bits = self.config.get('free_bits', 0.0)

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(self.device)

                # Compute batch indices for p-adic losses
                batch_indices = self._compute_batch_indices(batch_data)

                outputs = self.model(batch_data, temp_A, temp_B, beta_A, beta_B)

                # Compute losses using DualVAELoss (with p-adic losses)
                losses = self.loss_fn(
                    batch_data, outputs,
                    self.model.lambda1, self.model.lambda2, self.model.lambda3,
                    entropy_weight, repulsion_weight,
                    self.model.grad_norm_A_ema, self.model.grad_norm_B_ema,
                    self.model.gradient_balance, False,  # training=False in validation
                    batch_indices=batch_indices
                )

                # Add rho and phase to loss dict for logging
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

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print(f"\n{'='*80}")
        print("Starting DN-VAE v5.6 Training")
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

            # Log epoch (console)
            self.monitor.log_epoch(
                epoch, total_epochs,
                train_losses, val_losses,
                unique_A, cov_A, unique_B, cov_B,
                is_best,
                self.model.use_statenet,
                self.model.grad_balance_achieved
            )

            # Log to TensorBoard (if enabled)
            self.monitor.log_tensorboard(
                epoch, train_losses, val_losses,
                unique_A, unique_B, cov_A, cov_B
            )

            # Log weight histograms every 10 epochs (expensive)
            if epoch % 10 == 0:
                self.monitor.log_histograms(epoch, self.model)

            # Save checkpoint
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
                'grad_ema_momentum': self.model.grad_ema_momentum,
                'statenet_enabled': self.model.use_statenet
            }

            if self.model.use_statenet:
                metadata['statenet_corrections'] = self.model.statenet_corrections

            self.checkpoint_manager.save_checkpoint(
                epoch, self.model, self.optimizer, metadata, is_best
            )

            # Early stopping
            if self.monitor.should_stop(self.config['patience']):
                print(f"\n⚠️  Early stopping triggered (patience={self.config['patience']})")
                break

        # Print summary and cleanup
        self.monitor.print_training_summary()
        self.monitor.close()
