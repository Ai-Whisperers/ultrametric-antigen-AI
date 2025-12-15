"""Training script for Ternary VAE v5.10 - Pure Hyperbolic Geometry.

This is a THIN orchestration script that wires together components from src/.
All logic is delegated to src modules:
- src.training: Trainers, monitoring, config validation
- src.data: Data loading (standard or GPU-resident)
- src.utils: Reproducibility
- src.artifacts: Checkpoint management
- src.models: Model architecture
- src.losses: Consequence predictor for purpose-aware training

Usage:
    python scripts/train/train_ternary_v5_10.py --config configs/ternary_v5_10.yaml
    python scripts/train/train_ternary_v5_10.py --config configs/ternary_v5_10.yaml --gpu-resident
    python scripts/train/train_ternary_v5_10.py --config configs/ternary_v5_10.yaml --strict
"""

import yaml
import argparse
from pathlib import Path
import sys
import torch
import torch.optim as optim

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.ternary_vae_v5_10 import DualNeuralVAEV5_10
from src.models.curriculum import ContinuousCurriculumModule  # v5.10.1: StateNet-driven curriculum
from src.training import (
    TernaryVAETrainer,
    HyperbolicVAETrainer,
    TrainingMonitor,
    validate_config,
    ConfigValidationError,
    validate_environment
)
from src.data import create_ternary_data_loaders, get_data_loader_info
from src.data import create_gpu_resident_loaders  # P2: GPU-resident dataset
from src.utils import set_seed
from src.artifacts import CheckpointManager
from src.losses import (
    ConsequencePredictor,           # Consequence awareness
    evaluate_addition_accuracy,
    RadialStratificationLoss,       # v5.10.1: 3-adic radial hierarchy
    PurposefulRankingLoss           # Consequence-aware ranking
)


def main():
    args = parse_args()

    # Load and validate config
    raw_config = load_config(args.config)
    try:
        validated = validate_config(raw_config)
        config = raw_config  # Use raw dict for backward compatibility
    except ConfigValidationError as e:
        print(f"Configuration error:\n{e}")
        sys.exit(1)

    # Initialize monitor (centralized observability)
    monitor = TrainingMonitor(
        eval_num_samples=config.get('eval_num_samples', 1000),
        tensorboard_dir=config.get('tensorboard_dir', 'runs'),
        log_dir=args.log_dir,
        log_to_file=True
    )
    log_startup_info(monitor, config, args.config)

    # Validate environment
    env_status = validate_environment(config, monitor, strict=args.strict)
    if not env_status.is_valid:
        monitor._log("\nAborting due to environment validation failure")
        sys.exit(1)

    # Set reproducibility (from src.utils)
    set_seed(config.get('seed', 42))
    device = 'cuda' if env_status.cuda_available else 'cpu'
    monitor._log(f"\nDevice: {device}")

    # Create data loaders (from src.data)
    # P2 FIX: Support GPU-resident dataset for zero CPU-GPU transfers
    use_gpu_resident = args.gpu_resident or config.get('gpu_resident', False)
    monitor._log(f"\nCreating data loaders (GPU-resident: {use_gpu_resident})...")

    if use_gpu_resident and device == 'cuda':
        train_loader, val_loader, _ = create_gpu_resident_loaders(
            device=device,
            batch_size=config['batch_size'],
            train_split=config['train_split'],
            val_split=config['val_split'],
            seed=config.get('seed', 42)
        )
        monitor._log(f"Train: {len(train_loader.dataset.train_indices):,} samples (GPU-resident)")
        monitor._log(f"Val: {len(train_loader.dataset.val_indices):,} samples (GPU-resident)")
    else:
        train_loader, val_loader, _ = create_ternary_data_loaders(
            batch_size=config['batch_size'],
            train_split=config['train_split'],
            val_split=config['val_split'],
            test_split=config.get('test_split', 0.1),
            num_workers=config['num_workers'],
            seed=config.get('seed', 42)
        )
        monitor._log(f"Train: {get_data_loader_info(train_loader)['size']:,} samples")
        monitor._log(f"Val: {get_data_loader_info(val_loader)['size']:,} samples")

    # Create model
    model = create_model(config)

    # Create trainer
    base_trainer = TernaryVAETrainer(model, config, device)
    trainer = HyperbolicVAETrainer(base_trainer, model, device, config, monitor)

    # Create checkpoint manager (from src.artifacts)
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=Path(config.get('checkpoint_dir', 'sandbox-training/checkpoints/v5_10')),
        checkpoint_freq=config.get('checkpoint_freq', 10)
    )

    # Run training
    run_training_loop(trainer, model, train_loader, val_loader, config, checkpoint_manager)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Ternary VAE v5.10 - Pure Hyperbolic')
    parser.add_argument('--config', type=str, default='configs/ternary_v5_10.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory for log files')
    parser.add_argument('--strict', action='store_true',
                        help='Treat environment warnings as errors')
    parser.add_argument('--gpu-resident', action='store_true',
                        help='Use GPU-resident dataset (zero CPU-GPU transfers)')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def log_startup_info(monitor: TrainingMonitor, config: dict, config_path: str) -> None:
    """Log configuration summary at startup."""
    monitor._log(f"{'='*80}")
    monitor._log("Ternary VAE v5.10.1 Training - RADIAL-FIRST CURRICULUM LEARNING")
    monitor._log(f"{'='*80}")
    monitor._log(f"Config: {config_path} (validated)")

    # Curriculum and Radial Stratification (v5.10.1)
    radial_config = config.get('radial_stratification', {})
    curriculum_config = config.get('curriculum', {})
    mc = config.get('model', {})

    monitor._log(f"\nv5.10.1 Curriculum Learning:")
    monitor._log(f"  StateNet Version: v{mc.get('statenet_version', 4)}")
    monitor._log(f"  Radial Stratification: {'ENABLED' if radial_config.get('enabled') else 'DISABLED'}")
    if radial_config.get('enabled'):
        monitor._log(f"    inner_radius: {radial_config.get('inner_radius', 0.1)}")
        monitor._log(f"    outer_radius: {radial_config.get('outer_radius', 0.85)}")
        monitor._log(f"    base_weight: {radial_config.get('base_weight', 0.3)}")
    monitor._log(f"  Curriculum Module: {'ENABLED' if curriculum_config.get('enabled') else 'DISABLED'}")
    if curriculum_config.get('enabled'):
        monitor._log(f"    initial_tau: {curriculum_config.get('initial_tau', 0.0)}")
        monitor._log(f"    tau_scale: {curriculum_config.get('tau_scale', 0.1)}")

    # GPU-resident dataset (P2 optimization)
    monitor._log(f"\nData Pipeline:")
    monitor._log(f"  GPU-Resident: {'ENABLED' if config.get('gpu_resident', False) else 'DISABLED'}")
    if config.get('gpu_resident', False):
        monitor._log(f"    Memory: ~865 KB (all 19,683 samples on GPU)")
        monitor._log(f"    Benefit: Zero CPU→GPU transfers per batch")

    # Hyperbolic modules (v5.10 base)
    padic = config.get('padic_losses', {})
    hyp_v10 = padic.get('hyperbolic_v10', {})

    monitor._log(f"\nv5.10 Hyperbolic Modules:")
    monitor._log(f"  Hyperbolic Prior: {'ENABLED' if hyp_v10.get('use_hyperbolic_prior') else 'DISABLED'}")
    monitor._log(f"  Hyperbolic Recon: {'ENABLED' if hyp_v10.get('use_hyperbolic_recon') else 'DISABLED'}")
    monitor._log(f"  Centroid Loss: {'ENABLED' if hyp_v10.get('use_centroid_loss') else 'DISABLED'}")

    # Consequence predictor (purpose-aware training)
    monitor._log(f"\nPurpose-Aware Training:")
    monitor._log(f"  Consequence Predictor: ENABLED")
    monitor._log(f"    Eval interval: every {config.get('consequence_eval_interval', 50)} epochs")
    monitor._log(f"    Purpose: Predicts addition accuracy from ranking correlation")

    monitor._log(f"\nObservability:")
    monitor._log(f"  TensorBoard: {config.get('tensorboard_dir', 'runs')}/")
    monitor._log(f"  Histogram interval: every {config.get('histogram_interval', 10)} epochs")
    monitor._log(f"  Embedding interval: every {config.get('embedding_interval', 50)} epochs")
    monitor._log(f"  Batch log interval: every {config.get('log_interval', 10)} batches")

    monitor._log(f"\nEvaluation Intervals:")
    monitor._log(f"  Coverage: every {config.get('coverage_check_interval', 5)} epochs")
    monitor._log(f"  Correlation: every {config.get('eval_interval', 20)} epochs")


def create_model(config: dict) -> DualNeuralVAEV5_10:
    """Create model from config."""
    mc = config['model']
    return DualNeuralVAEV5_10(
        input_dim=mc['input_dim'],
        latent_dim=mc['latent_dim'],
        rho_min=mc['rho_min'],
        rho_max=mc['rho_max'],
        lambda3_base=mc['lambda3_base'],
        lambda3_amplitude=mc['lambda3_amplitude'],
        eps_kl=mc['eps_kl'],
        gradient_balance=mc.get('gradient_balance', True),
        adaptive_scheduling=mc.get('adaptive_scheduling', True),
        use_statenet=mc.get('use_statenet', True),
        statenet_version=mc.get('statenet_version', 4),
        statenet_lr_scale=mc.get('statenet_lr_scale', 0.1),
        statenet_lambda_scale=mc.get('statenet_lambda_scale', 0.02),
        statenet_ranking_scale=mc.get('statenet_ranking_scale', 0.3),
        statenet_hyp_sigma_scale=mc.get('statenet_hyp_sigma_scale', 0.05),
        statenet_hyp_curvature_scale=mc.get('statenet_hyp_curvature_scale', 0.02),
        statenet_curriculum_scale=mc.get('statenet_curriculum_scale', 0.1)
    )


def run_training_loop(
    trainer: HyperbolicVAETrainer,
    model,
    train_loader,
    val_loader,
    config: dict,
    checkpoint_manager: CheckpointManager
) -> None:
    """Execute the training loop with unified observability."""
    total_epochs = config['total_epochs']

    for epoch in range(total_epochs):
        # Set epoch on base trainer
        trainer.base_trainer.epoch = epoch

        # Train epoch (all batch-level TensorBoard logging happens inside)
        losses = trainer.train_epoch(train_loader, val_loader, epoch)

        # Update monitor state and log epoch (all epoch-level logging happens inside)
        trainer.update_monitor_state(losses)
        trainer.log_epoch(epoch, losses)

        # Save checkpoint using CheckpointManager (from src.artifacts)
        is_best = losses['loss'] < trainer.monitor.best_val_loss
        checkpoint_manager.save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=trainer.base_trainer.optimizer,
            metadata={
                'best_corr_hyp': trainer.best_corr_hyp,
                'best_corr_euc': trainer.best_corr_euc,
                'best_coverage': trainer.best_coverage,
                'correlation_history_hyp': trainer.correlation_history_hyp,
                'correlation_history_euc': trainer.correlation_history_euc,
                'coverage_history': trainer.coverage_history,
                'ranking_weight_history': trainer.ranking_weight_history,
                'config': config
            },
            is_best=is_best
        )

        # P2 FIX: Log exploration boost when triggered
        if losses.get('exploration_boosted', False):
            trainer.monitor._log(
                f"  [P2] Exploration boost: temp_mult={losses['temp_multiplier']:.2f}, "
                f"ranking_mult={losses['ranking_multiplier']:.2f} (stall_counter={losses['coverage_stall_counter']})"
            )

        # P1 FIX: Check correlation early stopping
        if losses.get('should_stop_correlation', False):
            trainer.monitor._log(
                f"\n[P1] Correlation early stopping triggered at epoch {epoch}"
            )
            trainer.monitor._log(
                f"  Correlation dropped from {losses['best_correlation_for_stopping']:.4f} "
                f"to {losses['corr_mean_hyp']:.4f} (threshold: {config.get('correlation_feedback', {}).get('correlation_drop_threshold', 0.05)})"
            )
            break

    # Training complete
    trainer.print_summary()
    trainer.monitor._log(f"\nCheckpoints saved to: {checkpoint_manager.checkpoint_dir}")
    trainer.close()


if __name__ == '__main__':
    main()
