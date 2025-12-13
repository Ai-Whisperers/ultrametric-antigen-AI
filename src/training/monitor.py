"""Training monitoring and logging.

This module handles training progress monitoring:
- Loss and metrics logging (batch-level and epoch-level)
- Coverage evaluation and tracking
- Training history management
- TensorBoard visualization (local, IP-safe)
- Persistent file logging
- v5.10 hyperbolic metrics support

Single responsibility: Monitoring and logging only.
"""

import torch
import logging
import sys
from typing import Dict, Any, List, Optional
from collections import defaultdict
import datetime
from pathlib import Path

# TensorBoard integration (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class TrainingMonitor:
    """Monitors and logs training progress with unified observability."""

    def __init__(
        self,
        eval_num_samples: int = 100000,
        tensorboard_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        log_dir: Optional[str] = "logs",
        log_to_file: bool = True
    ):
        """Initialize training monitor.

        Args:
            eval_num_samples: Number of samples for coverage evaluation
            tensorboard_dir: Base directory for TensorBoard logs (default: runs/)
            experiment_name: Name for this experiment run (auto-generated if None)
            log_dir: Directory for persistent log files
            log_to_file: Whether to enable file logging
        """
        self.eval_num_samples = eval_num_samples

        # Generate experiment name if not provided
        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = experiment_name

        # Training history
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Coverage tracking
        self.coverage_A_history: List[int] = []
        self.coverage_B_history: List[int] = []

        # Entropy tracking
        self.H_A_history: List[float] = []
        self.H_B_history: List[float] = []

        # v5.10 Hyperbolic metrics tracking
        self.correlation_hyp_history: List[float] = []
        self.correlation_euc_history: List[float] = []
        self.best_corr_hyp = 0.0
        self.best_corr_euc = 0.0
        self.best_coverage = 0.0

        # Batch/step counter for TensorBoard
        self.global_step = 0
        self.batches_per_epoch = 0

        # Setup file logging
        self.logger = self._setup_file_logging(log_dir, experiment_name) if log_to_file else None

        # TensorBoard setup
        self.writer: Optional[SummaryWriter] = None
        if TENSORBOARD_AVAILABLE and tensorboard_dir is not None:
            log_path = Path(tensorboard_dir) / f"ternary_vae_{experiment_name}"
            self.writer = SummaryWriter(str(log_path))
            self._log(f"TensorBoard logging to: {log_path}")
        elif tensorboard_dir is not None and not TENSORBOARD_AVAILABLE:
            self._log("Warning: TensorBoard requested but not installed (pip install tensorboard)")

    def _setup_file_logging(self, log_dir: str, experiment_name: str) -> logging.Logger:
        """Setup persistent file logging."""
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        log_file = log_path / f"training_{experiment_name}.log"

        logger = logging.getLogger(f"ternary_vae_{experiment_name}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        # File handler with timestamps
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)

        # Console handler without timestamps (cleaner output)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info(f"Logging to: {log_file}")
        return logger

    def _log(self, message: str) -> None:
        """Log message to both file and console."""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def update_histories(
        self,
        H_A: float,
        H_B: float,
        coverage_A: int,
        coverage_B: int
    ) -> None:
        """Update all tracked histories.

        Args:
            H_A: VAE-A entropy
            H_B: VAE-B entropy
            coverage_A: VAE-A coverage count
            coverage_B: VAE-B coverage count
        """
        self.H_A_history.append(H_A)
        self.H_B_history.append(H_B)
        self.coverage_A_history.append(coverage_A)
        self.coverage_B_history.append(coverage_B)

    def log_batch(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        loss: float,
        ce_A: float = 0.0,
        ce_B: float = 0.0,
        kl_A: float = 0.0,
        kl_B: float = 0.0,
        log_interval: int = 10
    ) -> None:
        """Log batch-level metrics for real-time observability.

        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            total_batches: Total batches in epoch
            loss: Current batch loss
            ce_A: VAE-A cross-entropy
            ce_B: VAE-B cross-entropy
            kl_A: VAE-A KL divergence
            kl_B: VAE-B KL divergence
            log_interval: Log every N batches
        """
        self.global_step += 1

        # Log to TensorBoard every batch for real-time graphs
        if self.writer is not None:
            self.writer.add_scalar('Batch/Loss', loss, self.global_step)
            self.writer.add_scalar('Batch/CE_A', ce_A, self.global_step)
            self.writer.add_scalar('Batch/CE_B', ce_B, self.global_step)
            self.writer.add_scalar('Batch/KL_A', kl_A, self.global_step)
            self.writer.add_scalar('Batch/KL_B', kl_B, self.global_step)

        # Log to console/file at intervals
        if batch_idx % log_interval == 0 or batch_idx == total_batches - 1:
            progress = (batch_idx + 1) / total_batches * 100
            self._log(f"  [Epoch {epoch}] Batch {batch_idx+1}/{total_batches} ({progress:.0f}%) | Loss: {loss:.4f}")

    def log_hyperbolic_batch(
        self,
        ranking_loss: float = 0.0,
        radial_loss: float = 0.0,
        hyp_kl_A: float = 0.0,
        hyp_kl_B: float = 0.0,
        centroid_loss: float = 0.0
    ) -> None:
        """Log v5.10 hyperbolic metrics at batch level.

        Args:
            ranking_loss: Hyperbolic ranking loss
            radial_loss: Radial hierarchy loss
            hyp_kl_A: Hyperbolic KL for VAE-A
            hyp_kl_B: Hyperbolic KL for VAE-B
            centroid_loss: Frechet centroid loss
        """
        if self.writer is not None:
            self.writer.add_scalar('Batch/HypRankingLoss', ranking_loss, self.global_step)
            self.writer.add_scalar('Batch/RadialLoss', radial_loss, self.global_step)
            self.writer.add_scalar('Batch/HypKL_A', hyp_kl_A, self.global_step)
            self.writer.add_scalar('Batch/HypKL_B', hyp_kl_B, self.global_step)
            self.writer.add_scalar('Batch/CentroidLoss', centroid_loss, self.global_step)
            self.writer.flush()

    def log_hyperbolic_epoch(
        self,
        epoch: int,
        corr_A_hyp: float,
        corr_B_hyp: float,
        corr_A_euc: float,
        corr_B_euc: float,
        mean_radius_A: float,
        mean_radius_B: float,
        ranking_weight: float,
        ranking_loss: float = 0.0,
        radial_loss: float = 0.0,
        hyp_kl_A: float = 0.0,
        hyp_kl_B: float = 0.0,
        centroid_loss: float = 0.0,
        homeostatic_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Log v5.10 hyperbolic metrics at epoch level.

        Args:
            epoch: Current epoch
            corr_A_hyp: VAE-A hyperbolic correlation
            corr_B_hyp: VAE-B hyperbolic correlation
            corr_A_euc: VAE-A Euclidean correlation
            corr_B_euc: VAE-B Euclidean correlation
            mean_radius_A: VAE-A mean latent radius
            mean_radius_B: VAE-B mean latent radius
            ranking_weight: Current ranking loss weight
            ranking_loss: Hyperbolic ranking loss
            radial_loss: Radial hierarchy loss
            hyp_kl_A: Hyperbolic KL for VAE-A
            hyp_kl_B: Hyperbolic KL for VAE-B
            centroid_loss: Frechet centroid loss
            homeostatic_metrics: Dict of homeostatic adaptation metrics
        """
        corr_mean_hyp = (corr_A_hyp + corr_B_hyp) / 2
        corr_mean_euc = (corr_A_euc + corr_B_euc) / 2

        # Update tracking
        self.correlation_hyp_history.append(corr_mean_hyp)
        self.correlation_euc_history.append(corr_mean_euc)

        if corr_mean_hyp > self.best_corr_hyp:
            self.best_corr_hyp = corr_mean_hyp
        if corr_mean_euc > self.best_corr_euc:
            self.best_corr_euc = corr_mean_euc

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalars('Hyperbolic/Correlation_Hyp', {
                'VAE_A': corr_A_hyp,
                'VAE_B': corr_B_hyp,
                'Mean': corr_mean_hyp
            }, epoch)

            self.writer.add_scalars('Hyperbolic/Correlation_Euc', {
                'VAE_A': corr_A_euc,
                'VAE_B': corr_B_euc,
                'Mean': corr_mean_euc
            }, epoch)

            self.writer.add_scalars('Hyperbolic/MeanRadius', {
                'VAE_A': mean_radius_A,
                'VAE_B': mean_radius_B
            }, epoch)

            self.writer.add_scalar('Hyperbolic/RankingWeight', ranking_weight, epoch)
            self.writer.add_scalar('Hyperbolic/RankingLoss', ranking_loss, epoch)
            self.writer.add_scalar('Hyperbolic/RadialLoss', radial_loss, epoch)

            # v5.10 specific
            self.writer.add_scalars('v5.10/HyperbolicKL', {
                'VAE_A': hyp_kl_A,
                'VAE_B': hyp_kl_B
            }, epoch)
            self.writer.add_scalar('v5.10/CentroidLoss', centroid_loss, epoch)

            # Homeostatic metrics
            if homeostatic_metrics:
                if 'prior_sigma_A' in homeostatic_metrics:
                    self.writer.add_scalars('v5.10/HomeostaticSigma', {
                        'VAE_A': homeostatic_metrics.get('prior_sigma_A', 1.0),
                        'VAE_B': homeostatic_metrics.get('prior_sigma_B', 1.0)
                    }, epoch)
                if 'prior_curvature_A' in homeostatic_metrics:
                    self.writer.add_scalars('v5.10/HomeostaticCurvature', {
                        'VAE_A': homeostatic_metrics.get('prior_curvature_A', 2.0),
                        'VAE_B': homeostatic_metrics.get('prior_curvature_B', 2.0)
                    }, epoch)

            self.writer.flush()

    def log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        loss: float,
        cov_A: float,
        cov_B: float,
        corr_A_hyp: float,
        corr_B_hyp: float,
        corr_A_euc: float,
        corr_B_euc: float,
        mean_radius_A: float,
        mean_radius_B: float,
        ranking_weight: float,
        coverage_evaluated: bool = True,
        correlation_evaluated: bool = True,
        hyp_kl_A: float = 0.0,
        hyp_kl_B: float = 0.0,
        centroid_loss: float = 0.0,
        radial_loss: float = 0.0,
        homeostatic_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Log comprehensive epoch summary to file and console.

        Args:
            epoch: Current epoch
            total_epochs: Total epochs
            loss: Total loss
            cov_A: VAE-A coverage percentage
            cov_B: VAE-B coverage percentage
            corr_A_hyp: VAE-A hyperbolic correlation
            corr_B_hyp: VAE-B hyperbolic correlation
            corr_A_euc: VAE-A Euclidean correlation
            corr_B_euc: VAE-B Euclidean correlation
            mean_radius_A: VAE-A mean latent radius
            mean_radius_B: VAE-B mean latent radius
            ranking_weight: Current ranking loss weight
            coverage_evaluated: Whether coverage was freshly evaluated
            correlation_evaluated: Whether correlation was freshly evaluated
            hyp_kl_A: Hyperbolic KL for VAE-A
            hyp_kl_B: Hyperbolic KL for VAE-B
            centroid_loss: Frechet centroid loss
            radial_loss: Radial hierarchy loss
            homeostatic_metrics: Dict of homeostatic adaptation metrics
        """
        current_coverage = (cov_A + cov_B) / 2
        if current_coverage > self.best_coverage:
            self.best_coverage = current_coverage

        cov_status = "FRESH" if coverage_evaluated else "cached"
        corr_status = "FRESH" if correlation_evaluated else "cached"

        self._log(f"\nEpoch {epoch}/{total_epochs}")
        self._log(f"  Loss: {loss:.4f} | Ranking Weight: {ranking_weight:.3f}")
        self._log(f"  Coverage [{cov_status}]: A={cov_A:.1f}% B={cov_B:.1f}% (best={self.best_coverage:.1f}%)")
        self._log(f"  3-Adic Correlation [{corr_status}] (Hyp): A={corr_A_hyp:.3f} B={corr_B_hyp:.3f} (best={self.best_corr_hyp:.3f})")

        if correlation_evaluated:
            self._log(f"  3-Adic Correlation (Euclidean): A={corr_A_euc:.3f} B={corr_B_euc:.3f}")

        self._log(f"  Mean Radius: A={mean_radius_A:.3f} B={mean_radius_B:.3f}")

        if radial_loss > 0:
            self._log(f"  Radial Loss: {radial_loss:.4f}")

        if hyp_kl_A > 0:
            self._log(f"  Hyperbolic KL: A={hyp_kl_A:.4f} B={hyp_kl_B:.4f}")

        if centroid_loss > 0:
            self._log(f"  Centroid Loss: {centroid_loss:.4f}")

        if homeostatic_metrics:
            if 'prior_sigma_A' in homeostatic_metrics:
                self._log(f"  Homeostatic Sigma: A={homeostatic_metrics['prior_sigma_A']:.3f} B={homeostatic_metrics['prior_sigma_B']:.3f}")
            if 'prior_curvature_A' in homeostatic_metrics:
                self._log(f"  Homeostatic Curvature: A={homeostatic_metrics['prior_curvature_A']:.3f} B={homeostatic_metrics['prior_curvature_B']:.3f}")

    def check_best(self, val_loss: float) -> bool:
        """Check if current validation loss is best.

        Args:
            val_loss: Current validation loss

        Returns:
            True if this is the best loss so far
        """
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return is_best

    def should_stop(self, patience: int) -> bool:
        """Check if early stopping criterion is met.

        Args:
            patience: Patience threshold

        Returns:
            True if should stop training
        """
        return self.patience_counter >= patience

    def has_coverage_plateaued(
        self,
        patience: int = 50,
        min_delta: float = 0.001
    ) -> bool:
        """Check if coverage improvement has plateaued.

        Useful for manifold approach where 100% coverage is the goal.
        Triggers when coverage improvement over `patience` epochs is below threshold.

        Args:
            patience: Number of epochs to check for improvement
            min_delta: Minimum improvement fraction required (relative to 19683 total ops)

        Returns:
            True if coverage has plateaued, False otherwise
        """
        if len(self.coverage_A_history) < patience:
            return False

        # Use max of A and B as coverage metric
        recent_A = self.coverage_A_history[-patience:]
        recent_B = self.coverage_B_history[-patience:]
        recent_max = [max(a, b) for a, b in zip(recent_A, recent_B)]

        # Compute improvement as fraction of total operations
        improvement = (recent_max[-1] - recent_max[0]) / 19683

        return improvement < min_delta

    def evaluate_coverage(
        self,
        model: torch.nn.Module,
        num_samples: int,
        device: str,
        vae: str = 'A'
    ) -> tuple[int, float]:
        """Evaluate operation coverage.

        Args:
            model: Model to evaluate
            num_samples: Number of samples to generate
            device: Device to run on
            vae: Which VAE to evaluate ('A' or 'B')

        Returns:
            Tuple of (unique_count, coverage_percentage)
        """
        model.eval()
        unique_ops = set()

        with torch.no_grad():
            batch_size = 1000
            num_batches = num_samples // batch_size

            for _ in range(num_batches):
                samples = model.sample(batch_size, device, vae)
                samples_rounded = torch.round(samples).long()

                for i in range(batch_size):
                    lut = samples_rounded[i]
                    lut_tuple = tuple(lut.cpu().tolist())
                    unique_ops.add(lut_tuple)

        coverage_pct = (len(unique_ops) / 19683) * 100
        return len(unique_ops), coverage_pct

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_losses: Dict[str, Any],
        val_losses: Dict[str, Any],
        unique_A: int,
        cov_A: float,
        unique_B: int,
        cov_B: float,
        is_best: bool,
        use_statenet: bool,
        grad_balance_achieved: bool
    ) -> None:
        """Log epoch results to console and file.

        Args:
            epoch: Current epoch
            total_epochs: Total epochs
            train_losses: Training losses dict
            val_losses: Validation losses dict
            unique_A: VAE-A unique operations
            cov_A: VAE-A coverage percentage
            unique_B: VAE-B unique operations
            cov_B: VAE-B coverage percentage
            is_best: Whether this is best validation loss
            use_statenet: Whether StateNet is enabled
            grad_balance_achieved: Whether gradient balance is achieved
        """
        self._log(f"\nEpoch {epoch}/{total_epochs}")
        self._log(f"  Loss: Train={train_losses['loss']:.4f} Val={val_losses['loss']:.4f}")
        self._log(f"  VAE-A: CE={train_losses['ce_A']:.4f} KL={train_losses['kl_A']:.4f} H={train_losses['H_A']:.3f}")
        self._log(f"  VAE-B: CE={train_losses['ce_B']:.4f} KL={train_losses['kl_B']:.4f} H={train_losses['H_B']:.3f}")
        self._log(f"  Weights: l1={train_losses['lambda1']:.3f} l2={train_losses['lambda2']:.3f} l3={train_losses['lambda3']:.3f}")
        self._log(f"  Phase {train_losses['phase']}: rho={train_losses['rho']:.3f} (balance: {'Y' if grad_balance_achieved else 'N'})")
        self._log(f"  Grad: ratio={train_losses['grad_ratio']:.3f} EMA_a={train_losses['ema_momentum']:.2f}")
        self._log(f"  Temp: A={train_losses['temp_A']:.3f} B={train_losses['temp_B']:.3f} | beta: A={train_losses['beta_A']:.3f} B={train_losses['beta_B']:.3f}")

        if use_statenet and 'lr_corrected' in train_losses:
            self._log(f"  LR: {train_losses['lr_scheduled']:.6f} -> {train_losses['lr_corrected']:.6f} (d={train_losses.get('delta_lr', 0):+.3f})")
            self._log(f"  StateNet: dl1={train_losses.get('delta_lambda1', 0):+.3f} dl2={train_losses.get('delta_lambda2', 0):+.3f} dl3={train_losses.get('delta_lambda3', 0):+.3f}")
        else:
            self._log(f"  LR: {train_losses['lr_scheduled']:.6f}")

        self._log(f"  Coverage: A={unique_A} ({cov_A:.2f}%) | B={unique_B} ({cov_B:.2f}%)")

        # p-Adic losses (Phase 1A/1B)
        has_padic = (train_losses.get('padic_metric_A', 0) > 0 or
                     train_losses.get('padic_ranking_A', 0) > 0 or
                     train_losses.get('padic_norm_A', 0) > 0)
        if has_padic:
            parts = []
            if train_losses.get('padic_metric_A', 0) > 0:
                parts.append(f"metric={train_losses.get('padic_metric_A', 0):.4f}/{train_losses.get('padic_metric_B', 0):.4f}")
            if train_losses.get('padic_ranking_A', 0) > 0:
                parts.append(f"rank={train_losses.get('padic_ranking_A', 0):.4f}/{train_losses.get('padic_ranking_B', 0):.4f}")
            if train_losses.get('padic_norm_A', 0) > 0:
                parts.append(f"norm={train_losses.get('padic_norm_A', 0):.4f}/{train_losses.get('padic_norm_B', 0):.4f}")
            self._log(f"  p-Adic: {' '.join(parts)}")

        if is_best:
            self._log(f"  Best val loss: {self.best_val_loss:.4f}")

    def log_tensorboard(
        self,
        epoch: int,
        train_losses: Dict[str, Any],
        val_losses: Dict[str, Any],
        unique_A: int,
        unique_B: int,
        cov_A: float,
        cov_B: float
    ) -> None:
        """Log metrics to TensorBoard.

        Args:
            epoch: Current epoch
            train_losses: Training losses dict
            val_losses: Validation losses dict
            unique_A: VAE-A unique operations
            unique_B: VAE-B unique operations
            cov_A: VAE-A coverage percentage
            cov_B: VAE-B coverage percentage
        """
        if self.writer is None:
            return

        # Primary losses (grouped comparison)
        self.writer.add_scalars('Loss/Total', {
            'train': train_losses['loss'],
            'val': val_losses['loss']
        }, epoch)

        # VAE-A metrics
        self.writer.add_scalar('VAE_A/CrossEntropy', train_losses['ce_A'], epoch)
        self.writer.add_scalar('VAE_A/KL_Divergence', train_losses['kl_A'], epoch)
        self.writer.add_scalar('VAE_A/Entropy', train_losses['H_A'], epoch)
        self.writer.add_scalar('VAE_A/Coverage_Count', unique_A, epoch)
        self.writer.add_scalar('VAE_A/Coverage_Pct', cov_A, epoch)

        # VAE-B metrics
        self.writer.add_scalar('VAE_B/CrossEntropy', train_losses['ce_B'], epoch)
        self.writer.add_scalar('VAE_B/KL_Divergence', train_losses['kl_B'], epoch)
        self.writer.add_scalar('VAE_B/Entropy', train_losses['H_B'], epoch)
        self.writer.add_scalar('VAE_B/Coverage_Count', unique_B, epoch)
        self.writer.add_scalar('VAE_B/Coverage_Pct', cov_B, epoch)

        # Comparative metrics
        self.writer.add_scalars('Compare/Entropy', {
            'VAE_A': train_losses['H_A'],
            'VAE_B': train_losses['H_B']
        }, epoch)
        self.writer.add_scalars('Compare/Coverage', {
            'VAE_A': cov_A,
            'VAE_B': cov_B
        }, epoch)

        # Training dynamics
        self.writer.add_scalar('Dynamics/Phase', train_losses['phase'], epoch)
        self.writer.add_scalar('Dynamics/Rho', train_losses['rho'], epoch)
        self.writer.add_scalar('Dynamics/GradRatio', train_losses['grad_ratio'], epoch)
        self.writer.add_scalar('Dynamics/EMA_Momentum', train_losses['ema_momentum'], epoch)

        # Lambda weights
        self.writer.add_scalars('Lambdas', {
            'lambda1': train_losses['lambda1'],
            'lambda2': train_losses['lambda2'],
            'lambda3': train_losses['lambda3']
        }, epoch)

        # Temperature scheduling
        self.writer.add_scalars('Temperature', {
            'VAE_A': train_losses['temp_A'],
            'VAE_B': train_losses['temp_B']
        }, epoch)

        # Beta scheduling
        self.writer.add_scalars('Beta', {
            'VAE_A': train_losses['beta_A'],
            'VAE_B': train_losses['beta_B']
        }, epoch)

        # Learning rate
        self.writer.add_scalar('LR/Scheduled', train_losses['lr_scheduled'], epoch)
        if 'lr_corrected' in train_losses:
            self.writer.add_scalar('LR/Corrected', train_losses['lr_corrected'], epoch)
            self.writer.add_scalar('LR/Delta', train_losses.get('delta_lr', 0), epoch)

        # StateNet corrections (if enabled)
        if 'delta_lambda1' in train_losses:
            self.writer.add_scalars('StateNet/Deltas', {
                'delta_lr': train_losses.get('delta_lr', 0),
                'delta_lambda1': train_losses.get('delta_lambda1', 0),
                'delta_lambda2': train_losses.get('delta_lambda2', 0),
                'delta_lambda3': train_losses.get('delta_lambda3', 0)
            }, epoch)

        # p-Adic losses (Phase 1A/1B from implement.md)
        has_padic = (train_losses.get('padic_metric_A', 0) > 0 or
                     train_losses.get('padic_ranking_A', 0) > 0 or
                     train_losses.get('padic_norm_A', 0) > 0)
        if has_padic:
            if train_losses.get('padic_metric_A', 0) > 0:
                self.writer.add_scalars('PAdicLoss/Metric', {
                    'VAE_A': train_losses.get('padic_metric_A', 0),
                    'VAE_B': train_losses.get('padic_metric_B', 0)
                }, epoch)
            if train_losses.get('padic_ranking_A', 0) > 0:
                self.writer.add_scalars('PAdicLoss/Ranking', {
                    'VAE_A': train_losses.get('padic_ranking_A', 0),
                    'VAE_B': train_losses.get('padic_ranking_B', 0)
                }, epoch)
            if train_losses.get('padic_norm_A', 0) > 0:
                self.writer.add_scalars('PAdicLoss/Norm', {
                    'VAE_A': train_losses.get('padic_norm_A', 0),
                    'VAE_B': train_losses.get('padic_norm_B', 0)
                }, epoch)

        # Flush to ensure real-time visibility in dashboard
        self.writer.flush()

    def log_histograms(
        self,
        epoch: int,
        model: torch.nn.Module
    ) -> None:
        """Log model weight histograms to TensorBoard.

        Args:
            epoch: Current epoch
            model: Model to log weights from
        """
        if self.writer is None:
            return

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'Weights/{name}', param.data, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

        # Flush histograms immediately
        self.writer.flush()

    def log_manifold_embedding(
        self,
        model: torch.nn.Module,
        epoch: int,
        device: str,
        n_samples: int = 5000,
        include_all: bool = False
    ) -> None:
        """Log latent embeddings to TensorBoard for 3D visualization.

        Uses TensorBoard's embedding projector for interactive PCA/t-SNE/UMAP
        visualization of the latent space with 3-adic structure metadata.

        Args:
            model: The VAE model to encode samples
            epoch: Current epoch for step tracking
            device: Device to run inference on
            n_samples: Number of samples to embed (default 5000 for performance)
            include_all: If True, embed all 19,683 operations (slower)
        """
        if self.writer is None:
            return

        from src.data.generation import generate_all_ternary_operations

        model.eval()

        # Generate operations
        all_operations = generate_all_ternary_operations()
        total_ops = len(all_operations)

        # Sample or use all
        if include_all or n_samples >= total_ops:
            indices = list(range(total_ops))
        else:
            import random
            indices = sorted(random.sample(range(total_ops), n_samples))

        import numpy as np
        operations = all_operations[np.array(indices)]  # Efficient numpy indexing
        x = torch.from_numpy(operations).float().to(device)

        with torch.no_grad():
            # Forward pass - handle both v5.6 and v5.10 models
            outputs = model(x, 1.0, 1.0, 0.5, 0.5)
            z_A = outputs['z_A']  # (n_samples, latent_dim)
            z_B = outputs['z_B']

            # Project to Poincaré ball for visualization
            z_A_norm = torch.norm(z_A, dim=1, keepdim=True)
            z_A_poincare = z_A / (1 + z_A_norm) * 0.95

            z_B_norm = torch.norm(z_B, dim=1, keepdim=True)
            z_B_poincare = z_B / (1 + z_B_norm) * 0.95

        # Compute 3-adic metadata for each operation
        metadata = []
        metadata_header = [
            'index',
            'prefix_1',    # First trit (3 values)
            'prefix_2',    # First 2 trits (9 values)
            'prefix_3',    # First 3 trits (27 values)
            'tree_depth',  # 3-adic depth from origin
            'radius_A',    # Poincaré radius
            'radius_B'
        ]

        for idx, op_idx in enumerate(indices):
            # 3-adic prefix hierarchy
            prefix_1 = op_idx % 3
            prefix_2 = op_idx % 9
            prefix_3 = op_idx % 27

            # 3-adic valuation from 0 (tree depth)
            depth = self._compute_3adic_depth(op_idx)

            # Poincaré radii
            r_A = z_A_norm[idx, 0].item()
            r_B = z_B_norm[idx, 0].item()

            metadata.append([
                str(op_idx),
                str(prefix_1),
                str(prefix_2),
                str(prefix_3),
                str(depth),
                f'{r_A:.3f}',
                f'{r_B:.3f}'
            ])

        # Log VAE-A embeddings (Euclidean)
        self.writer.add_embedding(
            z_A.cpu(),
            metadata=metadata,
            metadata_header=metadata_header,
            global_step=epoch,
            tag='Embedding/VAE_A_Euclidean'
        )

        # Log VAE-A embeddings (Poincaré projected)
        self.writer.add_embedding(
            z_A_poincare.cpu(),
            metadata=metadata,
            metadata_header=metadata_header,
            global_step=epoch,
            tag='Embedding/VAE_A_Poincare'
        )

        # Log VAE-B embeddings (Euclidean)
        self.writer.add_embedding(
            z_B.cpu(),
            metadata=metadata,
            metadata_header=metadata_header,
            global_step=epoch,
            tag='Embedding/VAE_B_Euclidean'
        )

        # Log VAE-B embeddings (Poincaré projected)
        self.writer.add_embedding(
            z_B_poincare.cpu(),
            metadata=metadata,
            metadata_header=metadata_header,
            global_step=epoch,
            tag='Embedding/VAE_B_Poincare'
        )

        self.writer.flush()
        self._log(f"Logged {len(indices)} embeddings to TensorBoard (epoch {epoch})")

    def _compute_3adic_depth(self, n: int) -> int:
        """Compute 3-adic valuation (tree depth) of integer n.

        Returns the largest k such that 3^k divides n.
        For n=0, returns 9 (maximum depth for 3^9 space).

        Args:
            n: Integer index

        Returns:
            3-adic valuation (depth in tree)
        """
        if n == 0:
            return 9  # Origin has maximum depth

        depth = 0
        while n % 3 == 0:
            depth += 1
            n //= 3
        return depth

    def close(self) -> None:
        """Close TensorBoard writer and flush all pending events."""
        if self.writer is not None:
            self.writer.close()
            self._log("TensorBoard writer closed")

    def get_metadata(self) -> Dict[str, Any]:
        """Get all tracked metadata for checkpointing.

        Returns:
            Dict of all tracked metrics and history
        """
        return {
            'best_val_loss': self.best_val_loss,
            'H_A_history': self.H_A_history,
            'H_B_history': self.H_B_history,
            'coverage_A_history': self.coverage_A_history,
            'coverage_B_history': self.coverage_B_history,
            'correlation_hyp_history': self.correlation_hyp_history,
            'correlation_euc_history': self.correlation_euc_history,
            'best_corr_hyp': self.best_corr_hyp,
            'best_corr_euc': self.best_corr_euc,
            'best_coverage': self.best_coverage,
            'global_step': self.global_step
        }

    def print_training_summary(self) -> None:
        """Print training completion summary."""
        self._log(f"\n{'='*80}")
        self._log("Training Complete")
        self._log(f"{'='*80}")
        self._log(f"Best val loss: {self.best_val_loss:.4f}")
        self._log(f"Best hyperbolic correlation: {self.best_corr_hyp:.4f}")
        self._log(f"Best Euclidean correlation: {self.best_corr_euc:.4f}")
        self._log(f"Best coverage: {self.best_coverage:.2f}%")

        if self.coverage_A_history:
            final_cov_A = self.coverage_A_history[-1]
            final_cov_B = self.coverage_B_history[-1]
            self._log(f"Final Coverage: A={final_cov_A} ({final_cov_A/19683*100:.2f}%)")
            self._log(f"                B={final_cov_B} ({final_cov_B/19683*100:.2f}%)")

        self._log(f"Target: r > 0.99, coverage > 99.7%")
