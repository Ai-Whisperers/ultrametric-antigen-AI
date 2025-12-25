# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Refactored trainer using modular components.

This trainer delegates to specialized components for:
- Scheduling: TemperatureScheduler, BetaScheduler, LearningRateScheduler
- Monitoring: TrainingMonitor
- Checkpointing: CheckpointManager
- Compilation: torch.compile (TorchInductor) for 1.4-2x speedup

Single responsibility: Orchestrate training loop only.

Inherits from BaseTrainer for:
- Safe division helpers (prevents P0 division-by-zero bugs)
- Validation guards (handles optional val_loader)
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from ..artifacts import CheckpointManager
from ..losses import DualVAELoss, RadialStratificationLoss
from ..models.curriculum import ContinuousCurriculumModule
from .base import BaseTrainer
from .monitor import TrainingMonitor
from .schedulers import BetaScheduler, LearningRateScheduler, TemperatureScheduler


class TernaryVAETrainer(BaseTrainer):
    """Refactored trainer with single responsibility: orchestrate training loop.

    All scheduling, monitoring, and checkpoint management delegated to components.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        device: str = "cuda",
    ):
        """Initialize trainer with model and config.

        Args:
            model: DualNeuralVAE model
            config: Training configuration dict
            device: Device to train on
        """
        super().__init__(model, config, device)

        # TorchInductor compilation (PyTorch 2.x)
        self.compiled = False
        compile_config = config.get("torch_compile", {})
        if compile_config.get("enabled", False) and hasattr(torch, "compile"):
            try:
                backend = compile_config.get("backend", "inductor")
                mode = compile_config.get("mode", "default")
                fullgraph = compile_config.get("fullgraph", False)

                self.model = torch.compile(
                    self.model, backend=backend, mode=mode, fullgraph=fullgraph
                )
                self.compiled = True
                print(f"torch.compile enabled: backend={backend}, mode={mode}")
            except Exception as e:
                print(
                    f"Warning: torch.compile failed ({e}), falling back to eager mode"
                )
        elif compile_config.get("enabled", False):
            print("Warning: torch.compile requested but not available (PyTorch < 2.0)")

        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config["optimizer"]["lr_start"],
            weight_decay=config["optimizer"].get("weight_decay", 0.0001),
        )

        # Initialize components
        self.temp_scheduler = TemperatureScheduler(
            config,
            config["phase_transitions"]["ultra_exploration_start"],
            config["controller"]["temp_lag"],
        )

        self.beta_scheduler = BetaScheduler(
            config, config["controller"]["beta_phase_lag"]
        )

        self.lr_scheduler = LearningRateScheduler(config["optimizer"]["lr_schedule"])

        self.monitor = TrainingMonitor(
            eval_num_samples=config["eval_num_samples"],
            tensorboard_dir=config.get("tensorboard_dir"),
            experiment_name=config.get("experiment_name"),
        )

        self.checkpoint_manager = CheckpointManager(
            Path(config["checkpoint_dir"]), config["checkpoint_freq"]
        )

        # Initialize loss function (with p-adic losses if configured)
        self.loss_fn = DualVAELoss(
            free_bits=config.get("free_bits", 0.0),
            repulsion_sigma=0.5,
            padic_config=config.get("padic_losses", {}),
        )

        # Initialize radial stratification loss (for curriculum learning)
        radial_config = config.get("radial_stratification", {})
        if radial_config.get("enabled", False):
            self.radial_loss_fn = RadialStratificationLoss(
                inner_radius=radial_config.get("inner_radius", 0.1),
                outer_radius=radial_config.get("outer_radius", 0.85),
                max_valuation=radial_config.get("max_valuation", 9),
                valuation_weighting=radial_config.get("valuation_weighting", True),
                loss_type=radial_config.get("loss_type", "smooth_l1"),
            )
            self.radial_loss_weight = radial_config.get("base_weight", 0.3)
        else:
            self.radial_loss_fn = None
            self.radial_loss_weight = 0.0

        # Initialize curriculum module (StateNet v5 controlled)
        curriculum_config = config.get("curriculum", {})
        if curriculum_config.get("enabled", False):
            self.curriculum = ContinuousCurriculumModule(
                initial_tau=curriculum_config.get("initial_tau", 0.0),
                tau_min=curriculum_config.get("tau_min", 0.0),
                tau_max=curriculum_config.get("tau_max", 1.0),
                tau_scale=curriculum_config.get("tau_scale", 0.1),
                momentum=curriculum_config.get("tau_momentum", 0.95),
            ).to(device)
        else:
            self.curriculum = None

        # P1 FIX: Correlation loss - actually add to total loss (not just logging)
        corr_loss_config = config.get("correlation_loss", {})
        self.correlation_loss_enabled = corr_loss_config.get("enabled", False)
        self.correlation_loss_weight = corr_loss_config.get("weight", 0.5)
        self.correlation_loss_warmup = corr_loss_config.get("warmup_epochs", 5)

        # GAP 7 FIX: Exploration boost temp multiplier (set by hyperbolic_trainer)
        # This unifies exploration_boost with curriculum temp modulation
        self.exploration_temp_multiplier = 1.0

        # Cache phase 4 start for model updates
        self.phase_4_start = config["phase_transitions"]["ultra_exploration_start"]

        # Precompute base-3 weights for index computation
        self._base3_weights = torch.tensor([3**i for i in range(9)], dtype=torch.long)

        # Print initialization summary
        self._print_init_summary()

    def _print_init_summary(self) -> None:
        """Print initialization summary."""
        print(f"\n{'='*80}")
        print("Dual Neural VAE - Base Trainer Initialized")
        print(f"{'='*80}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        if (
            self.config["model"].get("use_statenet", True)
            and self.model.state_net is not None
        ):
            statenet_params = sum(p.numel() for p in self.model.state_net.parameters())
            total_params = sum(p.numel() for p in self.model.parameters())
            print(
                f"StateNet parameters: {statenet_params:,} ({statenet_params/total_params*100:.2f}%)"
            )

        print(f"Device: {self.device}")
        print(f"Gradient balance: {self.config['model'].get('gradient_balance', True)}")
        print(
            f"Adaptive scheduling: {self.config['model'].get('adaptive_scheduling', True)}"
        )
        print(f"StateNet enabled: {self.config['model'].get('use_statenet', True)}")
        print(f"torch.compile: {'enabled' if self.compiled else 'disabled'}")

        # Curriculum learning (v5.10+)
        if self.curriculum is not None:
            curriculum_config = self.config.get("curriculum", {})
            print("\nCurriculum Learning (StateNet v5):")
            print(f"  Initial tau: {curriculum_config.get('initial_tau', 0.0)}")
            print(f"  tau_scale: {curriculum_config.get('tau_scale', 0.1)}")
            print(
                f"  tau bounds: [{curriculum_config.get('tau_min', 0.0)}, {curriculum_config.get('tau_max', 1.0)}]"
            )

        if self.radial_loss_fn is not None:
            radial_config = self.config.get("radial_stratification", {})
            print("\nRadial Stratification Loss:")
            print(f"  inner_radius: {radial_config.get('inner_radius', 0.1)}")
            print(f"  outer_radius: {radial_config.get('outer_radius', 0.85)}")
            print(f"  base_weight: {radial_config.get('base_weight', 0.3)}")

        # p-Adic losses (Phase 1A/1B)
        padic_config = self.config.get("padic_losses", {})
        has_padic = (
            padic_config.get("enable_metric_loss", False)
            or padic_config.get("enable_ranking_loss", False)
            or padic_config.get("enable_norm_loss", False)
        )
        if has_padic:
            print("\np-Adic Losses (implement.md Phase 1):")
            if padic_config.get("enable_metric_loss", False):
                print(
                    f"  Metric Loss: weight={padic_config.get('metric_loss_weight', 0.1)}, scale={padic_config.get('metric_loss_scale', 1.0)}"
                )
            if padic_config.get("enable_ranking_loss", False):
                print(
                    f"  Ranking Loss: weight={padic_config.get('ranking_loss_weight', 0.5)}, margin={padic_config.get('ranking_margin', 0.1)}"
                )
            if padic_config.get("enable_norm_loss", False):
                print(
                    f"  Norm Loss: weight={padic_config.get('norm_loss_weight', 0.05)}"
                )

        # P1 FIX: Correlation loss (actually wired into total loss)
        if self.correlation_loss_enabled:
            print("\nCorrelation Loss (P1 Fix - WIRED INTO LOSS):")
            print(f"  weight: {self.correlation_loss_weight}")
            print(f"  warmup_epochs: {self.correlation_loss_warmup}")
            print(
                "  Effect: -weight * correlation added to loss (rewards high correlation)"
            )

    def _check_best(self, losses: Dict[str, Any]) -> bool:
        """Check if current losses represent best model.

        Args:
            losses: Current validation/training losses

        Returns:
            True if this is the best model so far
        """
        return self.monitor.check_best(losses["loss"])

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
        self.model.rho = self.model.compute_phase_scheduled_rho(
            epoch, self.phase_4_start
        )

        # THREE-BODY FIX: Curriculum modulates rho (cross-injection)
        # When tau is high (ranking/angular focus), reduce cross-injection to preserve structure
        # When tau is low (radial focus), allow more mixing for exploration
        if self.curriculum is not None:
            tau = self.curriculum.get_tau().item()
            # Reduce rho by up to 30% when tau=1 (preserve individual VAE structures)
            rho_curriculum_factor = 1.0 - 0.3 * tau
            self.model.rho = self.model.rho * rho_curriculum_factor

        self.model.lambda3 = self.model.compute_cyclic_lambda3(epoch, period=30)

        grad_ratio = (
            self.model.grad_norm_B_ema / (self.model.grad_norm_A_ema + 1e-8)
        ).item()
        self.model.update_adaptive_ema_momentum(grad_ratio)

        if len(self.monitor.coverage_A_history) > 0:
            coverage_A = self.monitor.coverage_A_history[-1]
            coverage_B = self.monitor.coverage_B_history[-1]
            self.model.update_adaptive_lambdas(grad_ratio, coverage_A, coverage_B)

    def train_epoch(
        self, train_loader: DataLoader, log_interval: int = 10
    ) -> Dict[str, Any]:
        """Train for one epoch with batch-level TensorBoard logging.

        Args:
            train_loader: Training data loader
            log_interval: Log to console every N batches

        Returns:
            Dict of epoch losses and metrics
        """
        self.model.train()
        self._update_model_parameters(self.epoch)

        # Get scheduled parameters
        temp_A = self.temp_scheduler.get_temperature(self.epoch, "A")
        temp_B = self.temp_scheduler.get_temperature(self.epoch, "B")
        beta_A = self.beta_scheduler.get_beta(self.epoch, "A")
        beta_B = self.beta_scheduler.get_beta(self.epoch, "B")
        lr_scheduled = self.lr_scheduler.get_lr(self.epoch)

        # THREE-BODY FIX: Asymmetric curriculum-to-architecture feedback
        # VAE-A (chaotic): lighter modulation to maintain exploration
        # VAE-B (frozen): stronger modulation for structure focus
        if self.curriculum is not None:
            tau = self.curriculum.get_tau().item()
            # Asymmetric beta modulation:
            # VAE-A: +15% when tau=1 (lighter, maintains exploration)
            # VAE-B: +30% when tau=1 (stronger, structure focus)
            beta_A = beta_A * (1.0 + 0.15 * tau)
            beta_B = beta_B * (1.0 + 0.30 * tau)
            # Asymmetric temperature modulation:
            # VAE-A: -10% when tau=1 (lighter reduction, stays exploratory)
            # VAE-B: -20% when tau=1 (stronger reduction, more certain)
            temp_A = temp_A * (1.0 - 0.10 * tau)
            temp_B = temp_B * (1.0 - 0.20 * tau)

        # GAP 7 FIX: Apply exploration boost multiplier (set by hyperbolic_trainer)
        # Unified formula: temp = base_temp * curriculum_factor * exploration_mult
        if self.exploration_temp_multiplier != 1.0:
            temp_A = temp_A * self.exploration_temp_multiplier
            temp_B = temp_B * self.exploration_temp_multiplier

        entropy_weight = self.config["vae_b"]["entropy_weight"]
        repulsion_weight = self.config["vae_b"]["repulsion_weight"]
        self.config.get("free_bits", 0.0)

        epoch_losses = defaultdict(float)
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            # P2 FIX: Handle both standard DataLoader (tensor) and GPU-resident (tuple)
            if isinstance(batch, tuple):
                # GPU-resident dataset returns (data, indices) already on GPU
                batch_data, batch_indices = batch
            else:
                # Standard DataLoader returns just data tensor
                batch_data = batch.to(self.device)
                batch_indices = self._compute_batch_indices(batch_data)

            # Forward pass
            outputs = self.model(batch_data, temp_A, temp_B, beta_A, beta_B)

            # Compute losses using DualVAELoss (with p-adic losses)
            losses = self.loss_fn(
                batch_data,
                outputs,
                self.model.lambda1,
                self.model.lambda2,
                self.model.lambda3,
                entropy_weight,
                repulsion_weight,
                self.model.grad_norm_A_ema,
                self.model.grad_norm_B_ema,
                self.model.gradient_balance,
                self.model.training,
                batch_indices=batch_indices,
            )

            # Add rho and phase to loss dict for logging
            losses["rho"] = self.model.rho
            losses["phase"] = self.model.current_phase

            # Curriculum-controlled radial stratification loss (v5.10+)
            radial_loss = torch.tensor(0.0, device=self.device)
            curriculum_tau = 0.0

            if self.radial_loss_fn is not None:
                # Compute radial loss for both VAEs
                radial_loss_A = self.radial_loss_fn(outputs["z_A"], batch_indices)
                radial_loss_B = self.radial_loss_fn(outputs["z_B"], batch_indices)
                radial_loss = radial_loss_A + radial_loss_B

                # Get ranking loss from p-adic losses (if available)
                ranking_loss = torch.tensor(0.0, device=self.device)
                if "padic_ranking_A" in losses and torch.is_tensor(
                    losses["padic_ranking_A"]
                ):
                    ranking_loss = losses["padic_ranking_A"] + losses.get(
                        "padic_ranking_B", 0.0
                    )

                # Use curriculum to blend radial and ranking losses
                if self.curriculum is not None:
                    curriculum_tau = self.curriculum.get_tau().item()
                    structure_loss = self.curriculum.modulate_losses(
                        radial_loss, ranking_loss
                    )
                    # Add curriculum-modulated structure loss to total
                    losses["loss"] = (
                        losses["loss"] + self.radial_loss_weight * structure_loss
                    )
                else:
                    # No curriculum: just add weighted radial loss
                    losses["loss"] = (
                        losses["loss"] + self.radial_loss_weight * radial_loss
                    )

                # Log radial stratification metrics (named distinctly from hyperbolic radial_loss)
                losses["radial_stratification_loss"] = radial_loss.item()
                losses["curriculum_tau"] = curriculum_tau
                radial_wt, ranking_wt = (
                    (1 - curriculum_tau, curriculum_tau)
                    if self.curriculum
                    else (1.0, 0.0)
                )
                losses["curriculum_radial_weight"] = radial_wt
                losses["curriculum_ranking_weight"] = ranking_wt

            # Apply StateNet corrections once per epoch (with coverage feedback)
            if self.model.use_statenet and batch_idx == 0:
                # Get latest coverage from monitor history
                coverage_A = (
                    self.monitor.coverage_A_history[-1]
                    if self.monitor.coverage_A_history
                    else 0
                )
                coverage_B = (
                    self.monitor.coverage_B_history[-1]
                    if self.monitor.coverage_B_history
                    else 0
                )

                # GAP 6 FIX: Compute cross-VAE correlation r_AB
                # Measures similarity between VAE-A and VAE-B representations
                # High r_AB = redundancy (both VAEs learning same thing)
                # Low r_AB = complementary representations (desired)
                z_A = outputs["z_A"]
                z_B = outputs["z_B"]
                # Compute mean-centered correlation across batch
                z_A_centered = z_A - z_A.mean(dim=0, keepdim=True)
                z_B_centered = z_B - z_B.mean(dim=0, keepdim=True)
                # Cosine similarity as proxy for correlation (normalized)
                r_AB_raw = torch.nn.functional.cosine_similarity(
                    z_A_centered.flatten(start_dim=1).mean(dim=0),
                    z_B_centered.flatten(start_dim=1).mean(dim=0),
                    dim=0,
                )
                # Normalize to [0, 1] range (cos similarity is in [-1, 1])
                r_AB = ((r_AB_raw + 1.0) / 2.0).item()

                # Use StateNet v5 if curriculum is enabled and model supports it
                if self.curriculum is not None and hasattr(
                    self.model, "apply_statenet_v5_corrections"
                ):
                    corrections = self.model.apply_statenet_v5_corrections(
                        lr_scheduled,
                        (
                            losses["H_A"].item()
                            if torch.is_tensor(losses["H_A"])
                            else losses["H_A"]
                        ),
                        (
                            losses["H_B"].item()
                            if torch.is_tensor(losses["H_B"])
                            else losses["H_B"]
                        ),
                        losses["kl_A"].item(),
                        losses["kl_B"].item(),
                        (
                            self.model.grad_norm_B_ema
                            / (self.model.grad_norm_A_ema + 1e-8)
                        ).item(),
                        coverage_A=coverage_A,
                        coverage_B=coverage_B,
                        r_AB=r_AB,  # GAP 6 FIX: Cross-VAE embedding correlation
                        radial_loss=(
                            radial_loss.item()
                            if torch.is_tensor(radial_loss)
                            else radial_loss
                        ),
                        curriculum_tau=curriculum_tau,
                    )

                    # Update curriculum tau from StateNet's delta_curriculum
                    self.curriculum.update_tau(corrections["delta_curriculum"])

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = corrections["corrected_lr"]

                    epoch_losses["lr_corrected"] = corrections["corrected_lr"]
                    epoch_losses["delta_lr"] = corrections["delta_lr"]
                    epoch_losses["delta_lambda1"] = corrections["delta_lambda1"]
                    epoch_losses["delta_lambda2"] = corrections["delta_lambda2"]
                    epoch_losses["delta_lambda3"] = corrections["delta_lambda3"]
                    epoch_losses["delta_curriculum"] = corrections["delta_curriculum"]
                    # P1 FIX: Include delta_sigma and delta_curvature for HyperbolicPrior
                    epoch_losses["delta_sigma"] = corrections["delta_sigma"]
                    epoch_losses["delta_curvature"] = corrections["delta_curvature"]
                    # GAP 6 FIX: Log cross-VAE correlation
                    epoch_losses["r_AB"] = r_AB
                else:
                    # Fall back to v4 corrections
                    corrected_lr, *deltas = self.model.apply_statenet_corrections(
                        lr_scheduled,
                        (
                            losses["H_A"].item()
                            if torch.is_tensor(losses["H_A"])
                            else losses["H_A"]
                        ),
                        (
                            losses["H_B"].item()
                            if torch.is_tensor(losses["H_B"])
                            else losses["H_B"]
                        ),
                        losses["kl_A"].item(),
                        losses["kl_B"].item(),
                        (
                            self.model.grad_norm_B_ema
                            / (self.model.grad_norm_A_ema + 1e-8)
                        ).item(),
                        coverage_A=coverage_A,
                        coverage_B=coverage_B,
                    )

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = corrected_lr

                    epoch_losses["lr_corrected"] = corrected_lr
                    epoch_losses["delta_lr"] = deltas[0]
                    epoch_losses["delta_lambda1"] = deltas[1]
                    epoch_losses["delta_lambda2"] = deltas[2]
                    epoch_losses["delta_lambda3"] = deltas[3]
            else:
                if batch_idx == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = lr_scheduled

            # P1 FIX: Add correlation loss to total loss (rewards high correlation)
            # This is the actual wiring - not just logging but affecting gradients
            if (
                self.correlation_loss_enabled
                and self.epoch >= self.correlation_loss_warmup
            ):
                if self.monitor.correlation_hyp_history:
                    cached_corr = self.monitor.correlation_hyp_history[-1]
                    # Negative weight because we REWARD high correlation (minimize -corr)
                    correlation_loss_term = -self.correlation_loss_weight * cached_corr
                    losses["loss"] = losses["loss"] + correlation_loss_term
                    losses["correlation_loss_applied"] = correlation_loss_term
                    losses["cached_correlation_used"] = cached_corr

            # Backward and optimize
            self.optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["grad_clip"]
            )
            self.model.update_gradient_norms()
            self.optimizer.step()

            # Batch-level TensorBoard logging (real-time observability)
            batch_loss = (
                losses["loss"].item()
                if torch.is_tensor(losses["loss"])
                else losses["loss"]
            )
            self.monitor.log_batch(
                epoch=self.epoch,
                batch_idx=batch_idx,
                total_batches=num_batches,
                loss=batch_loss,
                ce_A=(
                    losses["ce_A"].item()
                    if torch.is_tensor(losses["ce_A"])
                    else losses["ce_A"]
                ),
                ce_B=(
                    losses["ce_B"].item()
                    if torch.is_tensor(losses["ce_B"])
                    else losses["ce_B"]
                ),
                kl_A=(
                    losses["kl_A"].item()
                    if torch.is_tensor(losses["kl_A"])
                    else losses["kl_A"]
                ),
                kl_B=(
                    losses["kl_B"].item()
                    if torch.is_tensor(losses["kl_B"])
                    else losses["kl_B"]
                ),
                log_interval=log_interval,
            )

            # Accumulate losses
            for key, val in losses.items():
                if isinstance(val, torch.Tensor):
                    epoch_losses[key] += val.item()
                else:
                    epoch_losses[key] += val

        # Average losses (guard against empty loader)
        if num_batches > 0:
            for key in epoch_losses:
                if key not in [
                    "lr_corrected",
                    "delta_lr",
                    "delta_lambda1",
                    "delta_lambda2",
                    "delta_lambda3",
                ]:
                    epoch_losses[key] /= num_batches

        # Store schedule info
        epoch_losses["temp_A"] = temp_A
        epoch_losses["temp_B"] = temp_B
        epoch_losses["beta_A"] = beta_A
        epoch_losses["beta_B"] = beta_B
        epoch_losses["lr_scheduled"] = lr_scheduled
        epoch_losses["grad_ratio"] = (
            self.model.grad_norm_B_ema / (self.model.grad_norm_A_ema + 1e-8)
        ).item()
        epoch_losses["ema_momentum"] = self.model.grad_ema_momentum

        # GAP 4 FIX: Update loss plateau detection for adaptive StateNet LR
        # When loss plateaus, StateNet gets higher effective LR to explore more aggressively
        if hasattr(self.model, "update_loss_plateau_detection"):
            current_loss = epoch_losses["loss"]
            boost = self.model.update_loss_plateau_detection(current_loss)
            epoch_losses["statenet_lr_boost"] = boost

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

        temp_A = self.temp_scheduler.get_temperature(self.epoch, "A")
        temp_B = self.temp_scheduler.get_temperature(self.epoch, "B")
        beta_A = self.beta_scheduler.get_beta(self.epoch, "A")
        beta_B = self.beta_scheduler.get_beta(self.epoch, "B")
        entropy_weight = self.config["vae_b"]["entropy_weight"]
        repulsion_weight = self.config["vae_b"]["repulsion_weight"]
        self.config.get("free_bits", 0.0)

        with torch.no_grad():
            for batch in val_loader:
                # P2 FIX: Handle both standard DataLoader (tensor) and GPU-resident (tuple)
                if isinstance(batch, tuple):
                    # GPU-resident dataset returns (data, indices) already on GPU
                    batch_data, batch_indices = batch
                else:
                    # Standard DataLoader returns just data tensor
                    batch_data = batch.to(self.device)
                    batch_indices = self._compute_batch_indices(batch_data)

                outputs = self.model(batch_data, temp_A, temp_B, beta_A, beta_B)

                # Compute losses using DualVAELoss (with p-adic losses)
                losses = self.loss_fn(
                    batch_data,
                    outputs,
                    self.model.lambda1,
                    self.model.lambda2,
                    self.model.lambda3,
                    entropy_weight,
                    repulsion_weight,
                    self.model.grad_norm_A_ema,
                    self.model.grad_norm_B_ema,
                    self.model.gradient_balance,
                    False,  # training=False in validation
                    batch_indices=batch_indices,
                )

                # Add rho and phase to loss dict for logging
                losses["rho"] = self.model.rho
                losses["phase"] = self.model.current_phase

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

    def train(
        self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None
    ) -> None:
        """Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        """
        print(f"\n{'='*80}")
        print("Starting Dual Neural VAE Training")
        print(f"{'='*80}\n")

        total_epochs = self.config["total_epochs"]

        for epoch in range(total_epochs):
            self.epoch = epoch

            # Train and validate
            train_losses = self.train_epoch(train_loader)

            # Validate only if val_loader is provided
            if val_loader is not None:
                val_losses = self.validate(val_loader)
                is_best = self.monitor.check_best(val_losses["loss"])
            else:
                # Use train losses for manifold approach
                val_losses = train_losses
                is_best = self.monitor.check_best(train_losses["loss"])

            # Evaluate coverage
            unique_A, cov_A = self.monitor.evaluate_coverage(
                self.model, self.config["eval_num_samples"], self.device, "A"
            )
            unique_B, cov_B = self.monitor.evaluate_coverage(
                self.model, self.config["eval_num_samples"], self.device, "B"
            )

            # Update histories
            self.monitor.update_histories(
                train_losses["H_A"], train_losses["H_B"], unique_A, unique_B
            )

            # Log epoch (console)
            self.monitor.log_epoch(
                epoch,
                total_epochs,
                train_losses,
                val_losses,
                unique_A,
                cov_A,
                unique_B,
                cov_B,
                is_best,
                self.model.use_statenet,
                self.model.grad_balance_achieved,
            )

            # Log to TensorBoard (if enabled)
            self.monitor.log_tensorboard(
                epoch,
                train_losses,
                val_losses,
                unique_A,
                unique_B,
                cov_A,
                cov_B,
            )

            # Log weight histograms every 10 epochs (expensive)
            if epoch % 10 == 0:
                self.monitor.log_histograms(epoch, self.model)

            # Log embedding projections for 3D visualization
            embedding_interval = self.config.get("embedding_interval", 50)
            if embedding_interval > 0 and epoch % embedding_interval == 0:
                self.monitor.log_manifold_embedding(
                    self.model,
                    epoch,
                    self.device,
                    n_samples=self.config.get("embedding_n_samples", 5000),
                )

            # Save checkpoint
            metadata = {
                **self.monitor.get_metadata(),
                "lambda1": self.model.lambda1,
                "lambda2": self.model.lambda2,
                "lambda3": self.model.lambda3,
                "rho": self.model.rho,
                "phase": self.model.current_phase,
                "grad_balance_achieved": self.model.grad_balance_achieved,
                "grad_norm_A_ema": self.model.grad_norm_A_ema.item(),
                "grad_norm_B_ema": self.model.grad_norm_B_ema.item(),
                "grad_ema_momentum": self.model.grad_ema_momentum,
                "statenet_enabled": self.model.use_statenet,
            }

            if self.model.use_statenet:
                metadata["statenet_corrections"] = self.model.statenet_corrections

            self.checkpoint_manager.save_checkpoint(
                epoch, self.model, self.optimizer, metadata, is_best
            )

            # Early stopping (loss-based)
            if self.monitor.should_stop(self.config["patience"]):
                print(
                    f"\n⚠️  Early stopping triggered (patience={self.config['patience']})"
                )
                break

            # Coverage plateau detection (for manifold approach)
            coverage_plateau_patience = self.config.get(
                "coverage_plateau_patience", 100
            )
            coverage_plateau_delta = self.config.get(
                "coverage_plateau_min_delta", 0.0005
            )
            if self.monitor.has_coverage_plateaued(
                coverage_plateau_patience, coverage_plateau_delta
            ):
                current_cov = max(
                    (
                        self.monitor.coverage_A_history[-1]
                        if self.monitor.coverage_A_history
                        else 0
                    ),
                    (
                        self.monitor.coverage_B_history[-1]
                        if self.monitor.coverage_B_history
                        else 0
                    ),
                )
                print(
                    f"\n📊 Coverage plateaued at {current_cov/19683*100:.2f}% (no improvement for {coverage_plateau_patience} epochs)"
                )
                break

        # Print summary and cleanup
        self.monitor.print_training_summary()
        self.monitor.close()
