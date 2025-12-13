"""Training script for Ternary VAE v5.10 - Pure Hyperbolic Geometry.

This is a THIN orchestration script that wires together components from src/.
All logic is delegated to src modules:
- src.training: Trainers, monitoring, config validation
- src.data: Data loading
- src.utils: Reproducibility
- src.artifacts: Checkpoint management
- src.models: Model architecture

Usage:
    python scripts/train/train_ternary_v5_10.py --config configs/ternary_v5_10.yaml
    python scripts/train/train_ternary_v5_10.py --config configs/ternary_v5_10.yaml --strict
"""

import yaml
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.ternary_vae_v5_10 import DualNeuralVAEV5_10
from src.training import (
    TernaryVAETrainer,
    HyperbolicVAETrainer,
    TrainingMonitor,
    validate_config,
    ConfigValidationError,
    validate_environment
)
from src.data import create_ternary_data_loaders, get_data_loader_info
from src.utils import set_seed
from src.artifacts import CheckpointManager


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
    monitor._log("\nCreating data loaders...")
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
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def log_startup_info(monitor: TrainingMonitor, config: dict, config_path: str) -> None:
    """Log configuration summary at startup."""
    monitor._log(f"{'='*80}")
    monitor._log("Ternary VAE v5.10 Training - PURE HYPERBOLIC GEOMETRY")
    monitor._log(f"{'='*80}")
    monitor._log(f"Config: {config_path} (validated)")

    padic = config.get('padic_losses', {})
    hyp_v10 = padic.get('hyperbolic_v10', {})

    monitor._log(f"\nv5.10 Modules:")
    monitor._log(f"  Hyperbolic Prior: {'ENABLED' if hyp_v10.get('use_hyperbolic_prior') else 'DISABLED'}")
    monitor._log(f"  Hyperbolic Recon: {'ENABLED' if hyp_v10.get('use_hyperbolic_recon') else 'DISABLED'}")
    monitor._log(f"  Centroid Loss: {'ENABLED' if hyp_v10.get('use_centroid_loss') else 'DISABLED'}")

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
        statenet_lr_scale=mc.get('statenet_lr_scale', 0.1),
        statenet_lambda_scale=mc.get('statenet_lambda_scale', 0.02),
        statenet_ranking_scale=mc.get('statenet_ranking_scale', 0.3),
        statenet_hyp_sigma_scale=mc.get('statenet_hyp_sigma_scale', 0.05),
        statenet_hyp_curvature_scale=mc.get('statenet_hyp_curvature_scale', 0.02)
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

    # Training complete
    trainer.print_summary()
    trainer.monitor._log(f"\nCheckpoints saved to: {checkpoint_manager.checkpoint_dir}")
    trainer.close()


if __name__ == '__main__':
    main()
