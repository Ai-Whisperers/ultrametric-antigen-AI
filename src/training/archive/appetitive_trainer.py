"""Appetitive VAE Trainer with bio-inspired loss modules.

This trainer extends the base training loop with:
- AdaptiveRankingLoss for ultrametric structure
- HierarchicalNormLoss for MSB/LSB variance
- CuriosityModule for density-based exploration
- SymbioticBridge for VAE-A/VAE-B mutual information
- AlgebraicClosureLoss for homomorphism constraint
- Metric-gated phase transitions (not epoch-based)

Single responsibility: Training loop for appetitive dual VAE.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

from .schedulers import TemperatureScheduler, BetaScheduler, LearningRateScheduler
from .monitor import TrainingMonitor
from ..artifacts import CheckpointManager
from ..losses import DualVAELoss, evaluate_addition_accuracy
from ..models.appetitive_vae import AppetitiveDualVAE


class AppetitiveVAETrainer:
    """Trainer for Appetitive Dual-VAE with emergent drives.

    Orchestrates training with appetite losses and metric-gated phase transitions.
    """

    def __init__(
        self,
        model: AppetitiveDualVAE,
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        """Initialize appetitive trainer.

        Args:
            model: AppetitiveDualVAE model
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

        # Initialize optimizer (includes appetite module parameters)
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

        # Initialize base loss function
        self.loss_fn = DualVAELoss(
            free_bits=config.get('free_bits', 0.0),
            repulsion_sigma=0.5,
            padic_config=config.get('padic_losses', {})
        )

        # Cache phase 4 start for model updates
        self.phase_4_start = config['phase_transitions']['ultra_exploration_start']

        # Precompute base-3 weights for index computation
        self._base3_weights = torch.tensor([3**i for i in range(9)], dtype=torch.long)

        # 3-adic correlation tracking for phase transitions
        self.correlation_history = []
        self.mi_history = []
        self.addition_accuracy_history = []

        # Print initialization summary
        self._print_init_summary()

    def _print_init_summary(self) -> None:
        """Print initialization summary."""
        print(f"\n{'='*80}")
        print("Appetitive Dual-VAE Initialized")
        print(f"{'='*80}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        base_model = self.model.base
        if hasattr(base_model, 'state_net') and base_model.use_statenet:
            statenet_params = sum(p.numel() for p in base_model.state_net.parameters())
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"StateNet parameters: {statenet_params:,} ({statenet_params/total_params*100:.2f}%)")

        # Appetite module parameters
        appetite_params = sum(
            p.numel() for name, p in self.model.named_parameters()
            if any(x in name for x in ['ranking', 'hierarchy', 'curiosity', 'symbiosis', 'closure'])
        )
        print(f"Appetite module parameters: {appetite_params:,}")

        print(f"Device: {self.device}")
        print(f"torch.compile: {'enabled' if self.compiled else 'disabled'}")
        print(f"\nPhase: {self.model.get_phase_description()}")
        print("Appetite weights:")
        print(f"  Ranking: {self.model.appetite_ranking.item():.3f}")
        print(f"  Hierarchy: {self.model.appetite_hierarchy.item():.3f}")
        print(f"  Curiosity: {self.model.appetite_curiosity.item():.3f}")
        print(f"  Symbiosis: {self.model.appetite_symbiosis.item():.3f}")
        print(f"  Closure: {self.model.appetite_closure.item():.3f}")

    def _compute_batch_indices(self, batch_data: torch.Tensor) -> torch.Tensor:
        """Compute operation indices from ternary data.

        Args:
            batch_data: Ternary data (batch_size, 9) with values in {-1, 0, 1}

        Returns:
            Operation indices (batch_size,) in range [0, 19682]
        """
        digits = (batch_data + 1).long()
        weights = self._base3_weights.to(batch_data.device)
        indices = (digits * weights).sum(dim=1)
        return indices

    def _update_model_parameters(self, epoch: int) -> None:
        """Update base model's adaptive parameters for current epoch.

        Args:
            epoch: Current epoch
        """
        base = self.model.base
        base.epoch = epoch
        base.rho = base.compute_phase_scheduled_rho(epoch, self.phase_4_start)
        base.lambda3 = base.compute_cyclic_lambda3(epoch, period=30)

        grad_ratio = (base.grad_norm_B_ema / (base.grad_norm_A_ema + 1e-8)).item()
        base.update_adaptive_ema_momentum(grad_ratio)

        if len(self.monitor.coverage_A_history) > 0:
            coverage_A = self.monitor.coverage_A_history[-1]
            coverage_B = self.monitor.coverage_B_history[-1]
            base.update_adaptive_lambdas(grad_ratio, coverage_A, coverage_B)

    def _compute_3adic_valuation(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute 3-adic valuation for operation indices.

        The 3-adic valuation v_3(n) is the highest power of 3 dividing n.
        For n=0, we use a large value (represents infinity).

        Args:
            indices: Operation indices (batch_size,)

        Returns:
            3-adic valuations (batch_size,)
        """
        valuations = torch.zeros_like(indices, dtype=torch.float32)
        remaining = indices.clone()

        for k in range(10):  # At most log_3(19683) = 9 iterations needed
            mask = (remaining % 3 == 0) & (remaining > 0)
            valuations[mask] += 1
            remaining[mask] = remaining[mask] // 3

        # Handle zero (infinite valuation)
        valuations[indices == 0] = 10.0  # Large value for "infinity"

        return valuations

    def _compute_3adic_correlation(
        self,
        model: AppetitiveDualVAE,
        device: str,
        n_samples: int = 5000
    ) -> Dict[str, float]:
        """Compute 3-adic correlation for phase gating.

        Args:
            model: Appetitive model
            device: Device
            n_samples: Number of samples for estimation

        Returns:
            Dict with 'correlation_A', 'correlation_B', 'correlation_mean'
        """
        model.eval()

        # Generate samples and compute latents
        with torch.no_grad():
            # Sample random operation indices
            indices = torch.randint(0, 19683, (n_samples,), device=device)

            # Generate ternary representations
            ternary_data = torch.zeros(n_samples, 9, device=device)
            for i in range(9):
                ternary_data[:, i] = ((indices // (3**i)) % 3) - 1

            # Forward pass
            outputs = model(ternary_data.float(), indices=indices, compute_appetites=False)
            z_A = outputs['z_A']
            z_B = outputs['z_B']

            # Sample triplets and measure concordance
            n_triplets = 1000
            i_idx = torch.randint(n_samples, (n_triplets,), device=device)
            j_idx = torch.randint(n_samples, (n_triplets,), device=device)
            k_idx = torch.randint(n_samples, (n_triplets,), device=device)

            # Filter distinct triplets
            valid = (i_idx != j_idx) & (i_idx != k_idx) & (j_idx != k_idx)
            i_idx, j_idx, k_idx = i_idx[valid], j_idx[valid], k_idx[valid]

            if len(i_idx) < 100:
                return {'correlation_A': 0.0, 'correlation_B': 0.0, 'correlation_mean': 0.0}

            # Compute 3-adic distances using valuation
            # d_3(a, b) = 3^(-v_3(a - b)) where v_3 is 3-adic valuation
            diff_ij = torch.abs(indices[i_idx] - indices[j_idx])
            diff_ik = torch.abs(indices[i_idx] - indices[k_idx])

            v_ij = self._compute_3adic_valuation(diff_ij)
            v_ik = self._compute_3adic_valuation(diff_ik)

            # 3-adic distance: d = 3^(-v), so larger v means smaller distance
            # For ordering: v_ij > v_ik means d_ij < d_ik
            padic_closer_ij = (v_ij > v_ik).float()  # 1 if d(i,j) < d(i,k) in 3-adic

            # Get latent distances
            d_A_ij = torch.norm(z_A[i_idx] - z_A[j_idx], dim=1)
            d_A_ik = torch.norm(z_A[i_idx] - z_A[k_idx], dim=1)
            d_B_ij = torch.norm(z_B[i_idx] - z_B[j_idx], dim=1)
            d_B_ik = torch.norm(z_B[i_idx] - z_B[k_idx], dim=1)

            # Compute rank correlation (concordance)
            latent_A_closer_ij = (d_A_ij < d_A_ik).float()
            latent_B_closer_ij = (d_B_ij < d_B_ik).float()

            # Concordance rate: how often ordering agrees
            corr_A = (padic_closer_ij == latent_A_closer_ij).float().mean().item()
            corr_B = (padic_closer_ij == latent_B_closer_ij).float().mean().item()

        return {
            'correlation_A': corr_A,
            'correlation_B': corr_B,
            'correlation_mean': (corr_A + corr_B) / 2
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, Any]:
        """Train for one epoch with appetite losses.

        Args:
            train_loader: Training data loader

        Returns:
            Dict of epoch losses and metrics
        """
        self.model.train()
        self._update_model_parameters(self.epoch)

        base = self.model.base

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

        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(self.device)

            # Compute batch indices for losses
            batch_indices = self._compute_batch_indices(batch_data)

            # Forward pass through appetitive model
            outputs = self.model(batch_data, indices=batch_indices, compute_appetites=True)

            # Compute base losses using DualVAELoss
            base_losses = self.loss_fn(
                batch_data, outputs,
                base.lambda1, base.lambda2, base.lambda3,
                entropy_weight, repulsion_weight,
                base.grad_norm_A_ema, base.grad_norm_B_ema,
                base.gradient_balance, base.training,
                batch_indices=batch_indices
            )

            # Add appetite loss to total
            appetite_loss = outputs.get('appetite_loss', torch.tensor(0.0, device=self.device))
            total_loss = base_losses['loss'] + appetite_loss

            # Store all losses
            base_losses['loss'] = total_loss
            base_losses['appetite_loss'] = appetite_loss
            base_losses['ranking_loss'] = outputs.get('ranking_loss', 0.0)
            base_losses['hierarchy_loss'] = outputs.get('hierarchy_loss', 0.0)
            base_losses['curiosity_loss'] = outputs.get('curiosity_loss', 0.0)
            base_losses['symbiosis_loss'] = outputs.get('symbiosis_loss', 0.0)
            base_losses['closure_loss'] = outputs.get('closure_loss', 0.0)
            base_losses['adaptive_rho'] = outputs.get('adaptive_rho', 0.0)
            base_losses['estimated_mi'] = outputs.get('estimated_mi', 0.0)
            base_losses['appetitive_phase'] = outputs.get('current_phase', 1)

            # Add base model phase info
            base_losses['rho'] = base.rho
            base_losses['phase'] = base.current_phase

            # Apply StateNet v2 corrections once per epoch
            if base.use_statenet and batch_idx == 0:
                coverage_A = self.monitor.coverage_A_history[-1] if self.monitor.coverage_A_history else 0
                coverage_B = self.monitor.coverage_B_history[-1] if self.monitor.coverage_B_history else 0

                corrected_lr, *deltas = base.apply_statenet_corrections(
                    lr_scheduled,
                    base_losses['H_A'].item() if torch.is_tensor(base_losses['H_A']) else base_losses['H_A'],
                    base_losses['H_B'].item() if torch.is_tensor(base_losses['H_B']) else base_losses['H_B'],
                    base_losses['kl_A'].item(),
                    base_losses['kl_B'].item(),
                    (base.grad_norm_B_ema / (base.grad_norm_A_ema + 1e-8)).item(),
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
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            base.update_gradient_norms()
            self.optimizer.step()

            # Accumulate losses
            for key, val in base_losses.items():
                if isinstance(val, torch.Tensor):
                    epoch_losses[key] += val.item()
                else:
                    epoch_losses[key] += val

            num_batches += 1

        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                if key not in ['lr_corrected', 'delta_lr', 'delta_lambda1', 'delta_lambda2', 'delta_lambda3']:
                    epoch_losses[key] /= num_batches

        # Store schedule info
        epoch_losses['temp_A'] = temp_A
        epoch_losses['temp_B'] = temp_B
        epoch_losses['beta_A'] = beta_A
        epoch_losses['beta_B'] = beta_B
        epoch_losses['lr_scheduled'] = lr_scheduled
        epoch_losses['grad_ratio'] = (base.grad_norm_B_ema / (base.grad_norm_A_ema + 1e-8)).item()
        epoch_losses['ema_momentum'] = base.grad_ema_momentum

        return epoch_losses

    def validate(self, val_loader: DataLoader) -> Dict[str, Any]:
        """Validation pass.

        Args:
            val_loader: Validation data loader

        Returns:
            Dict of validation losses
        """
        self.model.eval()
        base = self.model.base
        epoch_losses = defaultdict(float)
        num_batches = 0

        self.temp_scheduler.get_temperature(self.epoch, 'A')
        self.temp_scheduler.get_temperature(self.epoch, 'B')
        self.beta_scheduler.get_beta(self.epoch, 'A')
        self.beta_scheduler.get_beta(self.epoch, 'B')
        entropy_weight = self.config['vae_b']['entropy_weight']
        repulsion_weight = self.config['vae_b']['repulsion_weight']

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(self.device)
                batch_indices = self._compute_batch_indices(batch_data)

                outputs = self.model(batch_data, indices=batch_indices, compute_appetites=True)

                # Compute base losses
                losses = self.loss_fn(
                    batch_data, outputs,
                    base.lambda1, base.lambda2, base.lambda3,
                    entropy_weight, repulsion_weight,
                    base.grad_norm_A_ema, base.grad_norm_B_ema,
                    base.gradient_balance, False,
                    batch_indices=batch_indices
                )

                # Add appetite losses
                losses['appetite_loss'] = outputs.get('appetite_loss', 0.0)
                losses['ranking_loss'] = outputs.get('ranking_loss', 0.0)
                losses['rho'] = base.rho
                losses['phase'] = base.current_phase

                for key, val in losses.items():
                    if isinstance(val, torch.Tensor):
                        epoch_losses[key] += val.item()
                    else:
                        epoch_losses[key] += val

                num_batches += 1

        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches

        return epoch_losses

    def _log_appetitive_metrics(
        self,
        epoch: int,
        train_losses: Dict[str, Any],
        correlation_metrics: Dict[str, float]
    ) -> None:
        """Log appetitive-specific metrics.

        Args:
            epoch: Current epoch
            train_losses: Training losses dict
            correlation_metrics: 3-adic correlation metrics
        """
        print(f"  Appetitive Phase {self.model.current_phase}: {self.model.get_phase_description()}")
        print("  Appetite Losses:")
        print(f"    Ranking: {train_losses.get('ranking_loss', 0):.4f}")
        print(f"    Hierarchy: {train_losses.get('hierarchy_loss', 0):.4f}")
        print(f"    Curiosity: {train_losses.get('curiosity_loss', 0):.4f}")
        print(f"    Symbiosis: {train_losses.get('symbiosis_loss', 0):.4f} (MI={train_losses.get('estimated_mi', 0):.3f})")
        print(f"    Closure: {train_losses.get('closure_loss', 0):.4f}")
        print(f"  3-Adic Correlation: A={correlation_metrics['correlation_A']:.3f} B={correlation_metrics['correlation_B']:.3f}")

        # Log to TensorBoard if available
        if self.monitor.writer is not None:
            self.monitor.writer.add_scalars('Appetitive/Losses', {
                'ranking': train_losses.get('ranking_loss', 0),
                'hierarchy': train_losses.get('hierarchy_loss', 0),
                'curiosity': train_losses.get('curiosity_loss', 0),
                'symbiosis': train_losses.get('symbiosis_loss', 0),
                'closure': train_losses.get('closure_loss', 0)
            }, epoch)

            self.monitor.writer.add_scalars('Appetitive/Correlation', {
                'VAE_A': correlation_metrics['correlation_A'],
                'VAE_B': correlation_metrics['correlation_B']
            }, epoch)

            self.monitor.writer.add_scalar('Appetitive/Phase', self.model.current_phase, epoch)
            self.monitor.writer.add_scalar('Appetitive/EstimatedMI', train_losses.get('estimated_mi', 0), epoch)

            self.monitor.writer.flush()

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Main training loop with metric-gated phase transitions.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print(f"\n{'='*80}")
        print("Starting Appetitive Dual-VAE Training")
        print(f"{'='*80}\n")

        total_epochs = self.config['total_epochs']
        base = self.model.base

        for epoch in range(total_epochs):
            self.epoch = epoch

            # Train and validate
            train_losses = self.train_epoch(train_loader)

            # Validate only if val_loader is provided (not in manifold approach)
            if val_loader is not None:
                val_losses = self.validate(val_loader)
                is_best = self.monitor.check_best(val_losses['loss'])
            else:
                val_losses = train_losses  # Use train losses for logging compatibility
                is_best = self.monitor.check_best(train_losses['loss'])

            # Evaluate coverage
            unique_A, cov_A = self.monitor.evaluate_coverage(
                self.model, self.config['eval_num_samples'], self.device, 'A'
            )
            unique_B, cov_B = self.monitor.evaluate_coverage(
                self.model, self.config['eval_num_samples'], self.device, 'B'
            )

            # Compute 3-adic correlation for phase gating
            correlation_metrics = self._compute_3adic_correlation(
                self.model, self.device, n_samples=5000
            )
            self.correlation_history.append(correlation_metrics['correlation_mean'])

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
                base.use_statenet,
                base.grad_balance_achieved
            )

            # Log appetitive-specific metrics
            self._log_appetitive_metrics(epoch, train_losses, correlation_metrics)

            # Log to TensorBoard
            self.monitor.log_tensorboard(
                epoch, train_losses, val_losses,
                unique_A, unique_B, cov_A, cov_B
            )

            # Log weight histograms every 10 epochs
            if epoch % 10 == 0:
                self.monitor.log_histograms(epoch, self.model)

            # Log embedding projections for 3D visualization
            embedding_interval = self.config.get('embedding_interval', 50)
            if embedding_interval > 0 and epoch % embedding_interval == 0:
                self.monitor.log_manifold_embedding(
                    self.model, epoch, self.device,
                    n_samples=self.config.get('embedding_n_samples', 5000)
                )

            # Check phase transitions (metric-gated)
            # Only compute addition_accuracy when in phase 4+ (needed for 4->5 transition)
            add_acc = 0.0
            if self.model.current_phase >= 4:
                add_acc = evaluate_addition_accuracy(
                    self.model, self.device, n_samples=100
                )
            phase_metrics = {
                'correlation': correlation_metrics['correlation_mean'],
                'mi': train_losses.get('estimated_mi', 0.0),
                'addition_accuracy': add_acc
            }
            old_phase = self.model.current_phase
            self.model.update_phase(phase_metrics)
            if self.model.current_phase != old_phase:
                print(f"\n{'*'*40}")
                print(f"  Phase Transition: {old_phase} -> {self.model.current_phase}")
                print(f"  New Phase: {self.model.get_phase_description()}")
                print(f"{'*'*40}\n")

            # Save checkpoint
            metadata = {
                **self.monitor.get_metadata(),
                'lambda1': base.lambda1,
                'lambda2': base.lambda2,
                'lambda3': base.lambda3,
                'rho': base.rho,
                'phase': base.current_phase,
                'appetitive_phase': self.model.current_phase,
                'grad_balance_achieved': base.grad_balance_achieved,
                'grad_norm_A_ema': base.grad_norm_A_ema.item(),
                'grad_norm_B_ema': base.grad_norm_B_ema.item(),
                'grad_ema_momentum': base.grad_ema_momentum,
                'statenet_enabled': base.use_statenet,
                'correlation_history': self.correlation_history,
                'appetite_weights': {
                    'ranking': self.model.appetite_ranking.item(),
                    'hierarchy': self.model.appetite_hierarchy.item(),
                    'curiosity': self.model.appetite_curiosity.item(),
                    'symbiosis': self.model.appetite_symbiosis.item(),
                    'closure': self.model.appetite_closure.item()
                }
            }

            if base.use_statenet:
                metadata['statenet_corrections'] = base.statenet_corrections

            self.checkpoint_manager.save_checkpoint(
                epoch, self.model, self.optimizer, metadata, is_best
            )

            # Early stopping (loss-based)
            if self.monitor.should_stop(self.config['patience']):
                print(f"\nEarly stopping triggered (patience={self.config['patience']})")
                break

            # Coverage plateau detection (for manifold approach)
            coverage_plateau_patience = self.config.get('coverage_plateau_patience', 100)
            coverage_plateau_delta = self.config.get('coverage_plateau_min_delta', 0.0005)
            if self.monitor.has_coverage_plateaued(coverage_plateau_patience, coverage_plateau_delta):
                current_cov = max(
                    self.monitor.coverage_A_history[-1] if self.monitor.coverage_A_history else 0,
                    self.monitor.coverage_B_history[-1] if self.monitor.coverage_B_history else 0
                )
                print(f"\nCoverage plateaued at {current_cov/19683*100:.2f}% (no improvement for {coverage_plateau_patience} epochs)")
                break

        # Print summary and cleanup
        self.monitor.print_training_summary()
        print(f"\nFinal 3-adic correlation: {self.correlation_history[-1]:.4f}")
        print(f"Final appetitive phase: {self.model.get_phase_description()}")
        self.monitor.close()
