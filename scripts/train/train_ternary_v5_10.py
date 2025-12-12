"""Training script for Ternary VAE v5.10 - Pure Hyperbolic Geometry.

This is a THIN orchestration script that wires together components from src/.
All training logic and observability is delegated to HyperbolicVAETrainer.

The script only handles:
- CLI argument parsing
- Config loading
- Component instantiation
- Training loop orchestration
- Checkpoint saving

All TensorBoard logging, console output, and metrics tracking happen in src/.

Usage:
    python scripts/train/train_ternary_v5_10.py --config configs/ternary_v5_10.yaml
"""

import torch
import yaml
import argparse
import numpy as np
from pathlib import Path
import sys
from torch.utils.data import DataLoader, random_split

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.ternary_vae_v5_10 import DualNeuralVAEV5_10
from src.training import TernaryVAETrainer, HyperbolicVAETrainer, TrainingMonitor
from src.data import generate_all_ternary_operations, TernaryOperationDataset


def main():
    args = parse_args()
    config = load_config(args.config)

    # Initialize monitor (centralized observability)
    monitor = create_monitor(config, args.log_dir)
    log_startup_info(monitor, config, args.config)

    # Set reproducibility
    set_seed(config.get('seed', 42))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    monitor._log(f"\nDevice: {device}")

    # Create components
    train_loader, val_loader = create_data_loaders(config, monitor)
    model = create_model(config)
    trainer = create_trainer(model, config, device, monitor)

    # Run training
    run_training_loop(trainer, model, train_loader, val_loader, config)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Ternary VAE v5.10 - Pure Hyperbolic')
    parser.add_argument('--config', type=str, default='configs/ternary_v5_10.yaml')
    parser.add_argument('--log-dir', type=str, default='logs')
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_monitor(config: dict, log_dir: str) -> TrainingMonitor:
    return TrainingMonitor(
        eval_num_samples=config.get('eval_num_samples', 1000),
        tensorboard_dir=config.get('tensorboard_dir', 'runs'),
        log_dir=log_dir,
        log_to_file=True
    )


def log_startup_info(monitor: TrainingMonitor, config: dict, config_path: str) -> None:
    """Log configuration summary at startup."""
    monitor._log(f"{'='*80}")
    monitor._log("Ternary VAE v5.10 Training - PURE HYPERBOLIC GEOMETRY")
    monitor._log(f"{'='*80}")
    monitor._log(f"Config: {config_path}")

    padic = config.get('padic_losses', {})
    hyp_v10 = padic.get('hyperbolic_v10', {})

    monitor._log(f"\nv5.10 Modules:")
    monitor._log(f"  Hyperbolic Prior: {'ENABLED' if hyp_v10.get('use_hyperbolic_prior') else 'DISABLED'}")
    monitor._log(f"  Hyperbolic Recon: {'ENABLED' if hyp_v10.get('use_hyperbolic_recon') else 'DISABLED'}")
    monitor._log(f"  Centroid Loss: {'ENABLED' if hyp_v10.get('use_centroid_loss') else 'DISABLED'}")

    monitor._log(f"\nObservability:")
    monitor._log(f"  TensorBoard: {config.get('tensorboard_dir', 'runs')}/")
    monitor._log(f"  Histogram interval: every {config.get('histogram_interval', 10)} epochs")
    monitor._log(f"  Batch log interval: every {config.get('log_interval', 10)} batches")

    monitor._log(f"\nEvaluation Intervals:")
    monitor._log(f"  Coverage: every {config.get('coverage_check_interval', 5)} epochs")
    monitor._log(f"  Correlation: every {config.get('eval_interval', 20)} epochs")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_data_loaders(config: dict, monitor: TrainingMonitor):
    """Create train and validation data loaders."""
    monitor._log("\nGenerating dataset...")
    operations = generate_all_ternary_operations()
    dataset = TernaryOperationDataset(operations)
    monitor._log(f"Total operations: {len(dataset):,}")

    seed = config.get('seed', 42)
    train_size = int(config['train_split'] * len(dataset))
    val_size = int(config['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, _ = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    monitor._log(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

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

    return train_loader, val_loader


def create_model(config: dict) -> DualNeuralVAEV5_10:
    """Create and return the model."""
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


def create_trainer(model, config: dict, device: str, monitor: TrainingMonitor) -> HyperbolicVAETrainer:
    """Create the hyperbolic trainer with all observability wired up."""
    base_trainer = TernaryVAETrainer(model, config, device)
    return HyperbolicVAETrainer(base_trainer, model, device, config, monitor)


def run_training_loop(trainer: HyperbolicVAETrainer, model, train_loader, val_loader, config: dict) -> None:
    """Execute the training loop with unified observability."""
    checkpoint_dir = Path(config.get('checkpoint_dir', 'sandbox-training/checkpoints/v5_10'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    total_epochs = config['total_epochs']
    checkpoint_freq = config.get('checkpoint_freq', 10)

    for epoch in range(total_epochs):
        # Set epoch on base trainer
        trainer.base_trainer.epoch = epoch

        # Train epoch (all batch-level TensorBoard logging happens inside)
        losses = trainer.train_epoch(train_loader, val_loader, epoch)

        # Update monitor state and log epoch (all epoch-level logging happens inside)
        trainer.update_monitor_state(losses)
        trainer.log_epoch(epoch, losses)

        # Save checkpoint at intervals
        if epoch % checkpoint_freq == 0:
            save_checkpoint(checkpoint_dir, epoch, model, trainer, config)

    # Training complete
    trainer.print_summary()
    save_final_model(checkpoint_dir, model, trainer, config)
    trainer.close()


def save_checkpoint(checkpoint_dir: Path, epoch: int, model, trainer: HyperbolicVAETrainer, config: dict) -> None:
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.base_trainer.optimizer.state_dict(),
        'best_corr_hyp': trainer.best_corr_hyp,
        'best_corr_euc': trainer.best_corr_euc,
        'best_coverage': trainer.best_coverage,
        'config': config
    }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')


def save_final_model(checkpoint_dir: Path, model, trainer: HyperbolicVAETrainer, config: dict) -> None:
    """Save final model with full training history."""
    final_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'epoch': config['total_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.base_trainer.optimizer.state_dict(),
        'best_corr_hyp': trainer.best_corr_hyp,
        'best_corr_euc': trainer.best_corr_euc,
        'best_coverage': trainer.best_coverage,
        'correlation_history_hyp': trainer.correlation_history_hyp,
        'correlation_history_euc': trainer.correlation_history_euc,
        'coverage_history': trainer.coverage_history,
        'ranking_weight_history': trainer.ranking_weight_history,
        'config': config
    }, final_path)
    trainer.monitor._log(f"\nFinal model saved to: {final_path}")


if __name__ == '__main__':
    main()
