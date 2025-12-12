"""Hyperbolic VAE Trainer for v5.10 Pure Hyperbolic Geometry.

This trainer implements pure hyperbolic geometry training with homeostatic
adaptation for the Dual Neural VAE. It wraps a base trainer and adds:

1. Hyperbolic Prior (wrapped normal on Poincare ball)
2. Hyperbolic Reconstruction Loss (radius-weighted)
3. Hyperbolic Centroid Loss (Frechet mean clustering)
4. Continuous feedback for ranking weight adaptation
5. Homeostatic parameter adaptation
6. Unified TensorBoard observability (batch + epoch level)

Key principle: No Euclidean contamination - all geometry is hyperbolic.

Single responsibility: Hyperbolic training orchestration and observability.
"""

import torch
from typing import Dict, Any, Optional, List

from ..losses.padic_losses import PAdicRankingLossHyperbolic
from ..losses.hyperbolic_prior import HomeostaticHyperbolicPrior
from ..losses.hyperbolic_recon import HomeostaticReconLoss, HyperbolicCentroidLoss
from ..metrics.hyperbolic import compute_ranking_correlation_hyperbolic
from .monitor import TrainingMonitor


class HyperbolicVAETrainer:
    """Trainer for v5.10 Pure Hyperbolic geometry with homeostatic adaptation.

    This trainer wraps a base TernaryVAETrainer and adds hyperbolic geometry
    losses and metrics. It implements continuous feedback for adaptive ranking
    weight modulation based on coverage.

    Args:
        base_trainer: Base TernaryVAETrainer instance
        model: DualNeuralVAEV5_10 model
        device: Device to train on
        config: Training configuration dict
        monitor: Optional TrainingMonitor for logging
    """

    def __init__(
        self,
        base_trainer,
        model: torch.nn.Module,
        device: str,
        config: Dict[str, Any],
        monitor: Optional[TrainingMonitor] = None
    ):
        self.base_trainer = base_trainer
        self.model = model
        self.device = device
        self.config = config
        self.monitor = monitor or base_trainer.monitor
        self.total_epochs = config.get('total_epochs', 100)

        # Observability config
        self.histogram_interval = config.get('histogram_interval', 10)
        self.log_interval = config.get('log_interval', 10)

        # Initialize continuous feedback
        self._init_continuous_feedback(config)

        # Initialize hyperbolic loss modules
        self._init_hyperbolic_losses(config, device)

        # Initialize evaluation intervals and caching
        self._init_evaluation_config(config)

        # Tracking histories
        self.correlation_history_hyp: List[float] = []
        self.correlation_history_euc: List[float] = []
        self.coverage_history: List[float] = []
        self.ranking_weight_history: List[float] = []
        self.radial_loss_history: List[float] = []
        self.homeostatic_history: List[Dict[str, float]] = []

        # Best metrics
        self.best_corr_hyp = 0.0
        self.best_corr_euc = 0.0
        self.best_coverage = 0.0

    def _init_continuous_feedback(self, config: Dict[str, Any]) -> None:
        """Initialize continuous feedback parameters."""
        feedback_config = config.get('continuous_feedback', {})
        self.feedback_enabled = feedback_config.get('enabled', True)

        if self.feedback_enabled:
            self.base_ranking_weight = feedback_config.get('base_ranking_weight', 0.5)
            self.coverage_threshold = feedback_config.get('coverage_threshold', 90.0)
            self.coverage_sensitivity = feedback_config.get('coverage_sensitivity', 0.1)
            self.coverage_trend_sensitivity = feedback_config.get('coverage_trend_sensitivity', 2.0)
            self.min_ranking_weight = feedback_config.get('min_ranking_weight', 0.0)
            self.max_ranking_weight = feedback_config.get('max_ranking_weight', 1.0)
            self.coverage_ema_alpha = feedback_config.get('coverage_ema_alpha', 0.9)
        else:
            self.base_ranking_weight = 0.5

        # EMA tracking
        self.coverage_ema = None
        self.prev_coverage = None

    def _init_hyperbolic_losses(self, config: Dict[str, Any], device: str) -> None:
        """Initialize hyperbolic loss modules based on config."""
        padic_config = config.get('padic_losses', {})

        # PAdicRankingLossHyperbolic (v5.9 foundation)
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

        # v5.10 Hyperbolic modules
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

    def _init_evaluation_config(self, config: Dict[str, Any]) -> None:
        """Initialize evaluation intervals and cached values."""
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

    def compute_ranking_weight(self, current_coverage: float) -> float:
        """Compute ranking weight using sigmoid-based continuous feedback.

        The ranking weight modulates how strongly the hyperbolic ranking loss
        affects training. It increases when coverage is high (can focus on
        structure) and decreases when coverage is low (focus on exploration).

        Args:
            current_coverage: Current mean coverage percentage

        Returns:
            Ranking weight in [min_ranking_weight, max_ranking_weight]
        """
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

    def train_epoch(self, train_loader, val_loader, epoch: int) -> Dict[str, Any]:
        """Train one epoch with pure hyperbolic geometry and homeostatic adaptation.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Dict containing all losses and metrics for the epoch
        """
        # Coverage evaluation (only at intervals)
        should_check_coverage = (epoch == 0) or (epoch % self.coverage_check_interval == 0)

        if should_check_coverage:
            unique_A, cov_A = self.base_trainer.monitor.evaluate_coverage(
                self.model, self.config['eval_num_samples'], self.device, 'A'
            )
            unique_B, cov_B = self.base_trainer.monitor.evaluate_coverage(
                self.model, self.config['eval_num_samples'], self.device, 'B'
            )
            self.cached_cov_A = cov_A
            self.cached_cov_B = cov_B
            self.cached_unique_A = unique_A
            self.cached_unique_B = unique_B
        else:
            cov_A = self.cached_cov_A
            cov_B = self.cached_cov_B
            unique_A = self.cached_unique_A
            unique_B = self.cached_unique_B

        current_coverage = (cov_A + cov_B) / 2

        # Compute adaptive ranking weight
        ranking_weight = self.compute_ranking_weight(current_coverage)
        self.ranking_weight_history.append(ranking_weight)

        # Base training (uses DualVAELoss internally, logs batch metrics)
        train_losses = self.base_trainer.train_epoch(train_loader, log_interval=self.log_interval)
        val_losses = self.base_trainer.validate(val_loader)

        # Compute hyperbolic losses and metrics
        hyperbolic_metrics = self._compute_hyperbolic_losses(
            train_loader, ranking_weight, current_coverage
        )

        # Log hyperbolic metrics at batch level for TensorBoard
        self.log_hyperbolic_batch(hyperbolic_metrics)

        # Correlation evaluation (only at intervals)
        should_check_correlation = (epoch == 0) or (epoch % self.eval_interval == 0)

        if should_check_correlation:
            corr_results = compute_ranking_correlation_hyperbolic(
                self.model, self.device,
                max_norm=self.max_norm,
                curvature=self.curvature
            )
            corr_A_hyp, corr_B_hyp, corr_A_euc, corr_B_euc, mean_radius_A, mean_radius_B = corr_results

            self.cached_corr_A_hyp = corr_A_hyp
            self.cached_corr_B_hyp = corr_B_hyp
            self.cached_corr_A_euc = corr_A_euc
            self.cached_corr_B_euc = corr_B_euc
            self.cached_mean_radius_A = mean_radius_A
            self.cached_mean_radius_B = mean_radius_B
        else:
            corr_A_hyp = self.cached_corr_A_hyp
            corr_B_hyp = self.cached_corr_B_hyp
            corr_A_euc = self.cached_corr_A_euc
            corr_B_euc = self.cached_corr_B_euc
            mean_radius_A = self.cached_mean_radius_A
            mean_radius_B = self.cached_mean_radius_B

        corr_mean_hyp = (corr_A_hyp + corr_B_hyp) / 2
        corr_mean_euc = (corr_A_euc + corr_B_euc) / 2

        # Update best metrics
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

        # Build return dict
        return {
            **train_losses,
            'ranking_weight': ranking_weight,
            'ranking_loss_hyp': hyperbolic_metrics['ranking_loss'],
            'radial_loss': hyperbolic_metrics['radial_loss'],
            'hyp_kl_A': hyperbolic_metrics['hyp_kl_A'],
            'hyp_kl_B': hyperbolic_metrics['hyp_kl_B'],
            'hyp_recon_A': hyperbolic_metrics['hyp_recon_A'],
            'hyp_recon_B': hyperbolic_metrics['hyp_recon_B'],
            'centroid_loss': hyperbolic_metrics['centroid_loss'],
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
            **{f'ranking_{k}': v for k, v in hyperbolic_metrics['ranking_metrics'].items()},
            **{f'homeo_{k}': v for k, v in hyperbolic_metrics['homeostatic_metrics'].items()}
        }

    def _compute_hyperbolic_losses(
        self,
        train_loader,
        ranking_weight: float,
        current_coverage: float
    ) -> Dict[str, Any]:
        """Compute all hyperbolic geometry losses.

        Args:
            train_loader: Training data loader
            ranking_weight: Current ranking loss weight
            current_coverage: Current coverage for homeostatic adaptation

        Returns:
            Dict of hyperbolic loss values and metrics
        """
        ranking_loss = 0.0
        radial_loss = 0.0
        hyp_kl_A = 0.0
        hyp_kl_B = 0.0
        hyp_recon_A = 0.0
        hyp_recon_B = 0.0
        centroid_loss_val = 0.0
        ranking_metrics = {}
        homeostatic_metrics = {}

        if ranking_weight <= 0:
            self.radial_loss_history.append(0.0)
            self.homeostatic_history.append({})
            return {
                'ranking_loss': 0.0,
                'radial_loss': 0.0,
                'hyp_kl_A': 0.0,
                'hyp_kl_B': 0.0,
                'hyp_recon_A': 0.0,
                'hyp_recon_B': 0.0,
                'centroid_loss': 0.0,
                'ranking_metrics': {},
                'homeostatic_metrics': {}
            }

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

        # Hyperbolic Ranking Loss
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

        # Hyperbolic Prior KL
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

        # Hyperbolic Reconstruction
        if self.use_hyperbolic_recon:
            recon_A, recon_metrics_A = self.hyperbolic_recon_A(logits_A, ternary_data, z_A)
            recon_B, recon_metrics_B = self.hyperbolic_recon_B(logits_B, ternary_data, z_B)
            hyp_recon_A = recon_A.item()
            hyp_recon_B = recon_B.item()

            homeostatic_metrics.update({
                'recon_radius_A': recon_metrics_A.get('mean_radius', 0),
                'recon_radius_B': recon_metrics_B.get('mean_radius', 0)
            })

        # Centroid Loss
        if self.use_centroid_loss:
            cent_A, _ = self.centroid_loss(z_A, indices)
            cent_B, _ = self.centroid_loss(z_B, indices)
            centroid_loss_val = (cent_A.item() + cent_B.item()) / 2

            homeostatic_metrics.update({
                'centroid_loss': centroid_loss_val
            })

        self.radial_loss_history.append(radial_loss)
        self.homeostatic_history.append(homeostatic_metrics)

        return {
            'ranking_loss': ranking_loss,
            'radial_loss': radial_loss,
            'hyp_kl_A': hyp_kl_A,
            'hyp_kl_B': hyp_kl_B,
            'hyp_recon_A': hyp_recon_A,
            'hyp_recon_B': hyp_recon_B,
            'centroid_loss': centroid_loss_val,
            'ranking_metrics': ranking_metrics,
            'homeostatic_metrics': homeostatic_metrics
        }

    def log_epoch(self, epoch: int, losses: Dict[str, Any]) -> None:
        """Unified epoch-level logging to TensorBoard and console/file.

        This method centralizes ALL observability for the epoch:
        - Console/file summary via log_epoch_summary()
        - TensorBoard hyperbolic metrics via log_hyperbolic_epoch()
        - TensorBoard standard VAE metrics via log_tensorboard()
        - Weight/gradient histograms at intervals

        Args:
            epoch: Current epoch number
            losses: Dict containing all losses and metrics from train_epoch()
        """
        if self.monitor is None:
            return

        # Extract homeostatic metrics
        homeo = {k.replace('homeo_', ''): v for k, v in losses.items() if k.startswith('homeo_')}
        homeo_dict = homeo if homeo else None

        # 1. Console/file epoch summary
        self.monitor.log_epoch_summary(
            epoch=epoch,
            total_epochs=self.total_epochs,
            loss=losses['loss'],
            cov_A=losses['cov_A'],
            cov_B=losses['cov_B'],
            corr_A_hyp=losses['corr_A_hyp'],
            corr_B_hyp=losses['corr_B_hyp'],
            corr_A_euc=losses['corr_A_euc'],
            corr_B_euc=losses['corr_B_euc'],
            mean_radius_A=losses['mean_radius_A'],
            mean_radius_B=losses['mean_radius_B'],
            ranking_weight=losses['ranking_weight'],
            coverage_evaluated=losses.get('coverage_evaluated', True),
            correlation_evaluated=losses.get('correlation_evaluated', True),
            hyp_kl_A=losses.get('hyp_kl_A', 0),
            hyp_kl_B=losses.get('hyp_kl_B', 0),
            centroid_loss=losses.get('centroid_loss', 0),
            radial_loss=losses.get('radial_loss', 0),
            homeostatic_metrics=homeo_dict
        )

        # 2. TensorBoard hyperbolic metrics (epoch-level)
        self.monitor.log_hyperbolic_epoch(
            epoch=epoch,
            corr_A_hyp=losses['corr_A_hyp'],
            corr_B_hyp=losses['corr_B_hyp'],
            corr_A_euc=losses['corr_A_euc'],
            corr_B_euc=losses['corr_B_euc'],
            mean_radius_A=losses['mean_radius_A'],
            mean_radius_B=losses['mean_radius_B'],
            ranking_weight=losses['ranking_weight'],
            ranking_loss=losses.get('ranking_loss_hyp', 0),
            radial_loss=losses.get('radial_loss', 0),
            hyp_kl_A=losses.get('hyp_kl_A', 0),
            hyp_kl_B=losses.get('hyp_kl_B', 0),
            centroid_loss=losses.get('centroid_loss', 0),
            homeostatic_metrics=homeo_dict
        )

        # 3. TensorBoard standard VAE metrics (epoch-level)
        self._log_standard_tensorboard(epoch, losses)

        # 4. Weight/gradient histograms at intervals
        if epoch % self.histogram_interval == 0:
            self.monitor.log_histograms(epoch, self.model)

    def _log_standard_tensorboard(self, epoch: int, losses: Dict[str, Any]) -> None:
        """Log standard VAE metrics to TensorBoard.

        Args:
            epoch: Current epoch
            losses: Losses dict from train_epoch()
        """
        if self.monitor.writer is None:
            return

        writer = self.monitor.writer

        # Total loss
        writer.add_scalar('Loss/Total', losses['loss'], epoch)

        # VAE-A metrics
        writer.add_scalar('VAE_A/CrossEntropy', losses.get('ce_A', 0), epoch)
        writer.add_scalar('VAE_A/KL_Divergence', losses.get('kl_A', 0), epoch)
        writer.add_scalar('VAE_A/Entropy', losses.get('H_A', 0), epoch)
        writer.add_scalar('VAE_A/Coverage_Pct', losses['cov_A'], epoch)

        # VAE-B metrics
        writer.add_scalar('VAE_B/CrossEntropy', losses.get('ce_B', 0), epoch)
        writer.add_scalar('VAE_B/KL_Divergence', losses.get('kl_B', 0), epoch)
        writer.add_scalar('VAE_B/Entropy', losses.get('H_B', 0), epoch)
        writer.add_scalar('VAE_B/Coverage_Pct', losses['cov_B'], epoch)

        # Comparative metrics
        writer.add_scalars('Compare/Coverage', {
            'VAE_A': losses['cov_A'],
            'VAE_B': losses['cov_B']
        }, epoch)

        writer.add_scalars('Compare/Entropy', {
            'VAE_A': losses.get('H_A', 0),
            'VAE_B': losses.get('H_B', 0)
        }, epoch)

        # Training dynamics
        writer.add_scalar('Dynamics/Phase', losses.get('phase', 0), epoch)
        writer.add_scalar('Dynamics/Rho', losses.get('rho', 0), epoch)
        writer.add_scalar('Dynamics/GradRatio', losses.get('grad_ratio', 0), epoch)

        # Lambda weights
        writer.add_scalars('Lambdas', {
            'lambda1': losses.get('lambda1', 0),
            'lambda2': losses.get('lambda2', 0),
            'lambda3': losses.get('lambda3', 0)
        }, epoch)

        # Temperature and beta
        writer.add_scalars('Temperature', {
            'VAE_A': losses.get('temp_A', 1.0),
            'VAE_B': losses.get('temp_B', 1.0)
        }, epoch)

        writer.add_scalars('Beta', {
            'VAE_A': losses.get('beta_A', 1.0),
            'VAE_B': losses.get('beta_B', 1.0)
        }, epoch)

        # Learning rate
        writer.add_scalar('LR/Scheduled', losses.get('lr_scheduled', 0), epoch)
        if 'lr_corrected' in losses:
            writer.add_scalar('LR/Corrected', losses['lr_corrected'], epoch)

        # Coverage EMA for continuous feedback
        if 'coverage_ema' in losses:
            writer.add_scalar('Feedback/CoverageEMA', losses['coverage_ema'], epoch)

        writer.flush()

    def log_hyperbolic_batch(self, hyperbolic_metrics: Dict[str, Any]) -> None:
        """Log hyperbolic metrics at batch level to TensorBoard.

        Called after computing hyperbolic losses for real-time observability.

        Args:
            hyperbolic_metrics: Dict from _compute_hyperbolic_losses()
        """
        if self.monitor is None:
            return

        self.monitor.log_hyperbolic_batch(
            ranking_loss=hyperbolic_metrics.get('ranking_loss', 0),
            radial_loss=hyperbolic_metrics.get('radial_loss', 0),
            hyp_kl_A=hyperbolic_metrics.get('hyp_kl_A', 0),
            hyp_kl_B=hyperbolic_metrics.get('hyp_kl_B', 0),
            centroid_loss=hyperbolic_metrics.get('centroid_loss', 0)
        )

    def update_monitor_state(self, losses: Dict[str, Any]) -> None:
        """Update monitor's internal tracking state.

        Args:
            losses: Losses dict from train_epoch()
        """
        if self.monitor is None:
            return

        self.monitor.check_best(losses['loss'])
        self.monitor.update_histories(
            H_A=losses.get('H_A', 0),
            H_B=losses.get('H_B', 0),
            coverage_A=losses['unique_A'],
            coverage_B=losses['unique_B']
        )

    def print_summary(self) -> None:
        """Print training completion summary."""
        if self.monitor:
            self.monitor.print_training_summary()

    def close(self) -> None:
        """Close TensorBoard writer and cleanup."""
        if self.monitor:
            self.monitor.close()


__all__ = ['HyperbolicVAETrainer']
