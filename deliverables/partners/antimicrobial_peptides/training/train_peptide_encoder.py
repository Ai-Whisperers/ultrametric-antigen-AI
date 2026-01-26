#!/usr/bin/env python3
"""Training Script for PeptideVAE.

This script trains the PeptideVAE model on AMP activity data using
stratified cross-validation and curriculum learning.

Usage:
    python training/train_peptide_encoder.py
    python training/train_peptide_encoder.py --epochs 100 --fold 0
    python training/train_peptide_encoder.py --config training/config.yaml

Features:
    - Stratified 5-fold cross-validation
    - 6-component loss with curriculum learning
    - Early stopping on validation loss
    - Checkpoint saving with best model
    - TensorBoard logging
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add paths - repo_root must be first to avoid shadowing by local src/
_script_dir = Path(__file__).resolve().parent
_package_dir = _script_dir.parent
_deliverables_dir = _package_dir.parent.parent
_repo_root = _deliverables_dir.parent
# Insert in reverse priority order (last insert = highest priority)
sys.path.insert(0, str(_package_dir))
sys.path.insert(0, str(_deliverables_dir))
sys.path.insert(0, str(_repo_root))  # Must be last to take precedence

from src.encoders.peptide_encoder import PeptideVAE
from src.losses.peptide_losses import PeptideLossManager, CurriculumSchedule
from training.dataset import create_stratified_dataloaders

try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Model
    latent_dim: int = 16
    hidden_dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    max_radius: float = 0.95
    curvature: float = 1.0

    # Training
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Loss weights
    reconstruction_weight: float = 1.0
    mic_weight: float = 2.0
    property_weight: float = 1.0
    radial_weight: float = 0.5
    cohesion_weight: float = 0.3
    separation_weight: float = 0.3

    # Curriculum
    use_curriculum: bool = True

    # Validation
    n_folds: int = 5
    fold_idx: int = 0
    val_every: int = 1
    early_stop_patience: int = 10

    # I/O
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    seed: int = 42


# =============================================================================
# Training Functions
# =============================================================================


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: PeptideVAE,
    loss_manager: PeptideLossManager,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: PeptideVAE model
        loss_manager: Loss manager
        train_loader: Training data loader
        optimizer: Optimizer
        device: Training device
        epoch: Current epoch
        grad_clip: Gradient clipping value

    Returns:
        Dictionary of epoch metrics
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_metrics = {}

    for batch in train_loader:
        sequences = batch['sequences']
        mic_targets = batch['mic'].to(device)
        pathogen_labels = batch['pathogen_labels'].to(device)
        properties = batch['properties'].to(device)

        # Forward pass
        outputs = model(sequences, teacher_forcing=True)

        # Compute loss
        loss, metrics = loss_manager.compute_total_loss(
            outputs=outputs,
            target_tokens=outputs['tokens'],
            mic_targets=mic_targets,
            pathogen_labels=pathogen_labels,
            peptide_properties=properties,
            epoch=epoch,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        n_batches += 1
        for key, value in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = 0.0
            all_metrics[key] += value

    # Average metrics
    avg_metrics = {k: v / n_batches for k, v in all_metrics.items()}
    avg_metrics['epoch'] = epoch
    avg_metrics['train_loss'] = total_loss / n_batches

    return avg_metrics


@torch.no_grad()
def validate(
    model: PeptideVAE,
    loss_manager: PeptideLossManager,
    val_loader,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Validate the model.

    Args:
        model: PeptideVAE model
        loss_manager: Loss manager
        val_loader: Validation data loader
        device: Device
        epoch: Current epoch

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_metrics = {}

    all_mic_preds = []
    all_mic_targets = []

    for batch in val_loader:
        sequences = batch['sequences']
        mic_targets = batch['mic'].to(device)
        pathogen_labels = batch['pathogen_labels'].to(device)
        properties = batch['properties'].to(device)

        # Forward pass
        outputs = model(sequences, teacher_forcing=True)

        # Compute loss
        loss, metrics = loss_manager.compute_total_loss(
            outputs=outputs,
            target_tokens=outputs['tokens'],
            mic_targets=mic_targets,
            pathogen_labels=pathogen_labels,
            peptide_properties=properties,
            epoch=epoch,
        )

        # Collect predictions for correlation
        all_mic_preds.append(outputs['mic_pred'].cpu())
        all_mic_targets.append(mic_targets.cpu())

        # Accumulate metrics
        total_loss += loss.item()
        n_batches += 1
        for key, value in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = 0.0
            all_metrics[key] += value

    # Average metrics
    avg_metrics = {f'val_{k}': v / n_batches for k, v in all_metrics.items()}
    avg_metrics['val_loss'] = total_loss / n_batches

    # Compute correlations
    if HAS_SCIPY:
        mic_preds = torch.cat(all_mic_preds).squeeze().numpy()
        mic_targets = torch.cat(all_mic_targets).numpy()

        try:
            pearson_r, pearson_p = pearsonr(mic_targets, mic_preds)
            spearman_r, spearman_p = spearmanr(mic_targets, mic_preds)
            avg_metrics['val_pearson_r'] = pearson_r
            avg_metrics['val_pearson_p'] = pearson_p
            avg_metrics['val_spearman_r'] = spearman_r
            avg_metrics['val_spearman_p'] = spearman_p
        except Exception:
            pass

    return avg_metrics


def train_fold(
    config: TrainingConfig,
    fold_idx: int,
    device: torch.device,
    writer: Optional["SummaryWriter"] = None,
) -> Tuple[PeptideVAE, Dict[str, float]]:
    """Train on a single fold.

    Args:
        config: Training configuration
        fold_idx: Fold index
        device: Training device
        writer: Optional TensorBoard writer

    Returns:
        Tuple of (best_model, best_metrics)
    """
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx + 1}/{config.n_folds}")
    print('='*60)

    # Create data loaders
    train_loader, val_loader = create_stratified_dataloaders(
        fold_idx=fold_idx,
        n_folds=config.n_folds,
        batch_size=config.batch_size,
        random_state=config.seed,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    model = PeptideVAE(
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        dropout=config.dropout,
        max_radius=config.max_radius,
        curvature=config.curvature,
    ).to(device)

    # Create loss manager
    curriculum = CurriculumSchedule() if config.use_curriculum else None
    loss_manager = PeptideLossManager(
        reconstruction_weight=config.reconstruction_weight,
        mic_weight=config.mic_weight,
        property_weight=config.property_weight,
        radial_weight=config.radial_weight,
        cohesion_weight=config.cohesion_weight,
        separation_weight=config.separation_weight,
        use_curriculum=config.use_curriculum,
        curriculum=curriculum,
    )

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Training loop - track by Spearman correlation (our main metric)
    best_spearman = -float('inf')
    best_metrics = {}
    best_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        start_time = time.time()

        # Train
        train_metrics = train_epoch(
            model, loss_manager, train_loader, optimizer,
            device, epoch, config.grad_clip
        )

        # Validate
        if (epoch + 1) % config.val_every == 0:
            val_metrics = validate(
                model, loss_manager, val_loader, device, epoch
            )

            # Log to TensorBoard
            if writer is not None:
                for key, value in train_metrics.items():
                    writer.add_scalar(f'fold{fold_idx}/train/{key}', value, epoch)
                for key, value in val_metrics.items():
                    writer.add_scalar(f'fold{fold_idx}/val/{key}', value, epoch)

            # Check for improvement by Spearman correlation (higher is better)
            current_spearman = val_metrics.get('val_spearman_r', 0)
            if current_spearman > best_spearman:
                best_spearman = current_spearman
                best_metrics = val_metrics.copy()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            # Print progress
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d}/{config.epochs}: "
                  f"train={train_metrics['train_loss']:.4f}, "
                  f"val={val_metrics['val_loss']:.4f}, "
                  f"r={current_spearman:.3f}, "
                  f"best_r={best_spearman:.3f} "
                  f"[{elapsed:.1f}s]")

            # Early stopping based on Spearman plateau
            if patience_counter >= config.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        scheduler.step()

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metrics


def train_cross_validation(
    config: TrainingConfig,
) -> Dict[str, float]:
    """Run full cross-validation training.

    Args:
        config: Training configuration

    Returns:
        Dictionary of aggregated metrics across folds
    """
    set_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create TensorBoard writer
    writer = None
    if HAS_TENSORBOARD:
        log_dir = Path(config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir / f"run_{int(time.time())}")

    # Train each fold
    all_fold_metrics = []
    best_models = []

    for fold_idx in range(config.n_folds):
        model, fold_metrics = train_fold(config, fold_idx, device, writer)
        all_fold_metrics.append(fold_metrics)
        best_models.append(model)

        # Save fold checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': asdict(config),
            'fold_idx': fold_idx,
            'metrics': fold_metrics,
        }, checkpoint_dir / f"fold_{fold_idx}_best.pt")

    # Aggregate metrics across folds
    aggregated = {}
    for key in all_fold_metrics[0].keys():
        values = [m[key] for m in all_fold_metrics if key in m]
        if values:
            aggregated[f'{key}_mean'] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)

    # Print summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION SUMMARY")
    print("="*60)
    print(f"Folds: {config.n_folds}")
    print(f"Val Loss: {aggregated.get('val_loss_mean', 0):.4f} +/- {aggregated.get('val_loss_std', 0):.4f}")
    print(f"Pearson r: {aggregated.get('val_pearson_r_mean', 0):.4f} +/- {aggregated.get('val_pearson_r_std', 0):.4f}")
    print(f"Spearman r: {aggregated.get('val_spearman_r_mean', 0):.4f} +/- {aggregated.get('val_spearman_r_std', 0):.4f}")

    # Save aggregated results
    results = {
        'config': asdict(config),
        'fold_metrics': all_fold_metrics,
        'aggregated': aggregated,
    }
    with open(checkpoint_dir / 'cv_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    if writer is not None:
        writer.close()

    return aggregated


def train_single_fold(config: TrainingConfig) -> Dict[str, float]:
    """Train a single fold (for quick testing).

    Args:
        config: Training configuration

    Returns:
        Dictionary of metrics
    """
    set_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model, metrics = train_fold(config, config.fold_idx, device, None)

    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': asdict(config),
        'fold_idx': config.fold_idx,
        'metrics': metrics,
    }, checkpoint_dir / f"fold_{config.fold_idx}_best.pt")

    return metrics


# =============================================================================
# Main
# =============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train PeptideVAE")

    # Training options
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--fold', type=int, default=None,
                        help="Single fold to train (None = all folds)")
    parser.add_argument('--n-folds', type=int, default=5)

    # Model options
    parser.add_argument('--latent-dim', type=int, default=16)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=2)

    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--seed', type=int, default=42)

    # Curriculum
    parser.add_argument('--no-curriculum', action='store_true')

    args = parser.parse_args()

    # Build config
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_folds=args.n_folds,
        fold_idx=args.fold if args.fold is not None else 0,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        use_curriculum=not args.no_curriculum,
    )

    print("Training Configuration:")
    print(json.dumps(asdict(config), indent=2))
    print()

    if args.fold is not None:
        # Single fold
        metrics = train_single_fold(config)
    else:
        # Full cross-validation
        metrics = train_cross_validation(config)

    print("\nTraining complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
