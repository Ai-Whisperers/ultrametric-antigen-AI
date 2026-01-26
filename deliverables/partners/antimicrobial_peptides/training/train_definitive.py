#!/usr/bin/env python3
"""PeptideVAE Training - Definitive Fix Version.

Key fixes over train_improved.py:
1. DISABLED CURRICULUM - All losses active from epoch 0 (fixes collapse)
2. MIN_EPOCHS guard - Don't early stop before epoch 30
3. LR WARMUP - 5 epochs warmup prevents early instability
4. COLLAPSE DETECTION - Monitor prediction variance
5. FULL REPRODUCIBILITY - Deterministic CUDA ops

Memory: ~2GB VRAM, ~4GB RAM (fits 3-4GB VRAM constraint)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

# Add paths
# Path: training/ -> carlos_brizuela/ -> partners/ -> deliverables/ -> PROJECT_ROOT
_script_dir = Path(__file__).resolve().parent
_package_dir = _script_dir.parent  # carlos_brizuela/
BRIZUELA_ROOT = _package_dir  # Alias for consistency
_deliverables_dir = _package_dir.parent.parent
PROJECT_ROOT = _deliverables_dir.parent
sys.path.insert(0, str(_package_dir))
sys.path.insert(0, str(_deliverables_dir))
sys.path.insert(0, str(PROJECT_ROOT))

from src.encoders.peptide_encoder import PeptideVAE
from src.losses.peptide_losses import PeptideLossManager
from training.dataset import create_stratified_dataloaders

try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class DefinitiveConfig:
    """Definitive configuration with curriculum FIX."""

    # Model - smaller for 272 samples
    latent_dim: int = 16
    hidden_dim: int = 64
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.15
    max_radius: float = 0.95
    curvature: float = 1.0

    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Loss weights - MIC focused
    reconstruction_weight: float = 0.5
    mic_weight: float = 5.0
    property_weight: float = 0.5
    radial_weight: float = 0.3
    cohesion_weight: float = 0.2
    separation_weight: float = 0.2

    # FIX 1: DISABLE CURRICULUM - This is the critical fix!
    use_curriculum: bool = False

    # Validation
    n_folds: int = 5
    fold_idx: int = 0
    val_every: int = 1

    # FIX 2: Don't early stop too early
    early_stop_patience: int = 20
    min_epochs: int = 30  # NEW: minimum epochs before early stopping allowed

    # FIX 3: Learning rate warmup
    warmup_epochs: int = 5

    # FIX 4: Collapse detection
    min_pred_std: float = 0.05  # Minimum prediction std (collapse = predicting constant)

    # I/O - Local checkpoints directory (consolidated)
    checkpoint_dir: str = "../checkpoints_definitive"
    log_dir: str = "../logs"
    seed: int = 42


def set_seed(seed: int):
    """Set random seeds for full reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # FIX 5: Deterministic operations for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_warmup_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Create LR scheduler with linear warmup then cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: 0.1 → 1.0
            return 0.1 + 0.9 * epoch / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: PeptideVAE,
    loss_manager: PeptideLossManager,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    all_metrics = {}

    for batch in train_loader:
        sequences = batch['sequences']
        mic_targets = batch['mic'].to(device)
        pathogen_labels = batch['pathogen_labels'].to(device)
        properties = batch['properties'].to(device)

        outputs = model(sequences, teacher_forcing=True)

        loss, metrics = loss_manager.compute_total_loss(
            outputs=outputs,
            target_tokens=outputs['tokens'],
            mic_targets=mic_targets,
            pathogen_labels=pathogen_labels,
            peptide_properties=properties,
            epoch=epoch,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        for key, value in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = 0.0
            all_metrics[key] += value

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
    """Validate the model."""
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

        outputs = model(sequences, teacher_forcing=True)

        loss, metrics = loss_manager.compute_total_loss(
            outputs=outputs,
            target_tokens=outputs['tokens'],
            mic_targets=mic_targets,
            pathogen_labels=pathogen_labels,
            peptide_properties=properties,
            epoch=epoch,
        )

        all_mic_preds.append(outputs['mic_pred'].cpu())
        all_mic_targets.append(mic_targets.cpu())

        total_loss += loss.item()
        n_batches += 1
        for key, value in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = 0.0
            all_metrics[key] += value

    avg_metrics = {f'val_{k}': v / n_batches for k, v in all_metrics.items()}
    avg_metrics['val_loss'] = total_loss / n_batches

    # FIX 4: Compute prediction statistics for collapse detection
    mic_preds = torch.cat(all_mic_preds).squeeze().numpy()
    mic_targets = torch.cat(all_mic_targets).numpy()

    avg_metrics['val_pred_std'] = float(np.std(mic_preds))
    avg_metrics['val_pred_mean'] = float(np.mean(mic_preds))
    avg_metrics['val_target_std'] = float(np.std(mic_targets))

    if HAS_SCIPY:
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
    config: DefinitiveConfig,
    fold_idx: int,
    device: torch.device,
) -> Tuple[PeptideVAE, Dict[str, float]]:
    """Train on a single fold with DEFINITIVE FIX."""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx + 1}/{config.n_folds} (DEFINITIVE FIX)")
    print('='*60)
    print(f"FIX 1: use_curriculum = {config.use_curriculum} (should be False)")
    print(f"FIX 2: min_epochs = {config.min_epochs} (no early stop before this)")
    print(f"FIX 3: warmup_epochs = {config.warmup_epochs}")
    print(f"FIX 4: min_pred_std = {config.min_pred_std} (collapse threshold)")

    train_loader, val_loader = create_stratified_dataloaders(
        fold_idx=fold_idx,
        n_folds=config.n_folds,
        batch_size=config.batch_size,
        random_state=config.seed,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    model = PeptideVAE(
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        dropout=config.dropout,
        max_radius=config.max_radius,
        curvature=config.curvature,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # FIX 1: DISABLE CURRICULUM - All losses active from epoch 0
    loss_manager = PeptideLossManager(
        reconstruction_weight=config.reconstruction_weight,
        mic_weight=config.mic_weight,
        property_weight=config.property_weight,
        radial_weight=config.radial_weight,
        cohesion_weight=config.cohesion_weight,
        separation_weight=config.separation_weight,
        use_curriculum=False,  # CRITICAL FIX
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # FIX 3: Use warmup scheduler
    scheduler = get_warmup_scheduler(optimizer, config.warmup_epochs, config.epochs)

    best_spearman = -float('inf')
    best_metrics = {}
    best_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        start_time = time.time()

        train_metrics = train_epoch(
            model, loss_manager, train_loader, optimizer,
            device, epoch, config.grad_clip
        )

        if (epoch + 1) % config.val_every == 0:
            val_metrics = validate(
                model, loss_manager, val_loader, device, epoch
            )

            current_spearman = val_metrics.get('val_spearman_r', 0)
            if current_spearman > best_spearman:
                best_spearman = current_spearman
                best_metrics = val_metrics.copy()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                improved = " *NEW BEST*"
            else:
                patience_counter += 1
                improved = ""

            elapsed = time.time() - start_time

            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0]

            # FIX 4: Collapse detection warning
            pred_std = val_metrics.get('val_pred_std', 0)
            collapse_warning = ""
            if pred_std < config.min_pred_std:
                collapse_warning = " ⚠️ LOW VARIANCE"

            # Log with LR and prediction std
            print(f"Epoch {epoch+1:3d}/{config.epochs}: "
                  f"loss={train_metrics['train_loss']:.4f}, "
                  f"val={val_metrics['val_loss']:.4f}, "
                  f"r={current_spearman:.3f}, "
                  f"std={pred_std:.3f}, "
                  f"lr={current_lr:.2e} "
                  f"[{elapsed:.1f}s]{improved}{collapse_warning}")

            # FIX 2: Don't early stop before min_epochs
            if epoch >= config.min_epochs and patience_counter >= config.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} (after min_epochs={config.min_epochs})")
                break

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final verification
    print(f"\n--- Fold {fold_idx} Final Verification ---")

    # Check curriculum fix
    curriculum_mic = best_metrics.get('val_curriculum_mic', 'N/A')
    if curriculum_mic == 0.0:
        print(f"❌ WARNING: curriculum_mic=0.0 - FIX MAY NOT HAVE WORKED!")
    else:
        print(f"✓ Curriculum: curriculum_mic={curriculum_mic}")

    # Check collapse
    final_spearman = best_metrics.get('val_spearman_r', 0)
    final_pred_std = best_metrics.get('val_pred_std', 0)
    if final_spearman < 0.40:
        print(f"❌ COLLAPSE DETECTED: r={final_spearman:.4f} < 0.40")
    else:
        print(f"✓ No collapse: r={final_spearman:.4f}")

    if final_pred_std < config.min_pred_std:
        print(f"❌ LOW VARIANCE: pred_std={final_pred_std:.4f} < {config.min_pred_std}")
    else:
        print(f"✓ Variance OK: pred_std={final_pred_std:.4f}")

    # Check baseline
    if final_spearman >= 0.56:
        print(f"✓ BEATS BASELINE: r={final_spearman:.4f} >= 0.56")
    else:
        print(f"⚠️ Below baseline: r={final_spearman:.4f} < 0.56")

    return model, best_metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Definitive PeptideVAE")
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--checkpoint-dir', type=str, default='../checkpoints_definitive')
    args = parser.parse_args()

    config = DefinitiveConfig(
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        fold_idx=args.fold if args.fold is not None else 0,
    )

    set_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nDEFINITIVE Configuration:")
    print(f"  use_curriculum: {config.use_curriculum} (FIX 1)")
    print(f"  min_epochs: {config.min_epochs} (FIX 2)")
    print(f"  mic_weight: {config.mic_weight}")
    print(f"  hidden_dim: {config.hidden_dim}")
    print()

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.fold is not None:
        # Single fold
        model, metrics = train_fold(config, args.fold, device)

        torch.save({
            'model_state_dict': model.state_dict(),
            'config': asdict(config),
            'fold_idx': args.fold,
            'metrics': metrics,
        }, checkpoint_dir / f"fold_{args.fold}_definitive.pt")

        print(f"\nFold {args.fold} Results:")
        print(f"  Spearman r: {metrics.get('val_spearman_r', 0):.4f}")
        print(f"  Pearson r: {metrics.get('val_pearson_r', 0):.4f}")
        print(f"  Target: Beat sklearn baseline of 0.56")
    else:
        # All folds
        all_metrics = []
        for fold_idx in range(config.n_folds):
            model, metrics = train_fold(config, fold_idx, device)
            all_metrics.append(metrics)

            torch.save({
                'model_state_dict': model.state_dict(),
                'config': asdict(config),
                'fold_idx': fold_idx,
                'metrics': metrics,
            }, checkpoint_dir / f"fold_{fold_idx}_definitive.pt")

        # Summary
        spearman_vals = [m.get('val_spearman_r', 0) for m in all_metrics]
        pearson_vals = [m.get('val_pearson_r', 0) for m in all_metrics]

        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY (DEFINITIVE FIX)")
        print('='*60)
        for i, r in enumerate(spearman_vals):
            status = "PASSED" if r >= 0.56 else "BELOW"
            print(f"  Fold {i}: r={r:.4f} [{status}]")
        print()
        print(f"Spearman r: {np.mean(spearman_vals):.4f} +/- {np.std(spearman_vals):.4f}")
        print(f"Pearson r: {np.mean(pearson_vals):.4f} +/- {np.std(pearson_vals):.4f}")
        print(f"Min fold: {min(spearman_vals):.4f}")
        print(f"Max fold: {max(spearman_vals):.4f}")
        print(f"Target: Beat sklearn baseline of 0.56")
        print(f"Status: {'PASSED' if np.mean(spearman_vals) >= 0.56 else 'NEEDS MORE WORK'}")
        print(f"Collapse check: {'NO COLLAPSE' if min(spearman_vals) > 0.40 else 'COLLAPSE DETECTED'}")

        # Save summary
        def to_native(obj):
            """Convert numpy types to native Python for JSON serialization."""
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_native(v) for v in obj]
            return obj

        with open(checkpoint_dir / 'cv_results_definitive.json', 'w') as f:
            json.dump({
                'config': asdict(config),
                'fold_metrics': [to_native(m) for m in all_metrics],
                'mean_spearman': float(np.mean(spearman_vals)),
                'std_spearman': float(np.std(spearman_vals)),
                'min_spearman': float(min(spearman_vals)),
                'max_spearman': float(max(spearman_vals)),
                'mean_pearson': float(np.mean(pearson_vals)),
                'std_pearson': float(np.std(pearson_vals)),
                'passed_baseline': bool(np.mean(spearman_vals) >= 0.56),
                'no_collapse': bool(min(spearman_vals) > 0.40),
            }, f, indent=2)

    print("\nTraining complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
