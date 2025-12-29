#!/usr/bin/env python3
"""Train with HomeostasisController + richness preservation.

The v5.11 architecture demonstrated that high coverage AND high hierarchy
can coexist with geometric richness. This script uses the homeostatic
controller to dynamically balance objectives while preserving within-level
variance (geometric richness).

Key differences from previous approaches:
1. Uses HomeostasisController for dynamic freeze management
2. Includes richness in the loss function
3. Targets high hierarchy WITHOUT collapsing the geometric manifold

Usage:
    python scripts/epsilon_vae/train_homeostatic_rich.py
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.models import TernaryVAEV5_11_PartialFreeze
from src.models.homeostasis import HomeostasisController, compute_Q
from src.utils.checkpoint import load_checkpoint_compat, get_model_state_dict


class RichHierarchyLoss(nn.Module):
    """Loss function balancing hierarchy, coverage, and richness.

    Unlike previous approaches that collapsed richness to maximize hierarchy,
    this loss actively preserves within-level variance while pushing toward
    better hierarchy.
    """

    def __init__(
        self,
        inner_radius: float = 0.1,
        outer_radius: float = 0.9,
        hierarchy_weight: float = 5.0,
        coverage_weight: float = 1.0,
        richness_weight: float = 2.0,
        separation_weight: float = 3.0,
        min_richness_ratio: float = 0.5,  # Keep at least 50% of original variance
    ):
        super().__init__()
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.hierarchy_weight = hierarchy_weight
        self.coverage_weight = coverage_weight
        self.richness_weight = richness_weight
        self.separation_weight = separation_weight
        self.min_richness_ratio = min_richness_ratio
        self.max_valuation = 9

        # Target radii - less aggressive than before
        self.register_buffer(
            'target_radii',
            torch.tensor([
                outer_radius - (v / self.max_valuation) * (outer_radius - inner_radius)
                for v in range(10)
            ])
        )

    def forward(
        self,
        z_hyp: torch.Tensor,
        indices: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        original_radii: torch.Tensor = None,
    ) -> dict:
        device = z_hyp.device
        radii = z_hyp.norm(dim=-1)
        valuations = TERNARY.valuation(indices).long().to(device)

        # === 1. Hierarchy: Push mean radius per level toward target ===
        # But allow variance within each level
        hierarchy_loss = torch.tensor(0.0, device=device)
        unique_vals = torch.unique(valuations)

        for v in unique_vals:
            mask = valuations == v
            if mask.sum() > 0:
                mean_r = radii[mask].mean()
                target_r = self.target_radii[v]
                hierarchy_loss = hierarchy_loss + (mean_r - target_r) ** 2

        hierarchy_loss = hierarchy_loss / len(unique_vals)

        # === 2. Coverage: Reconstruction accuracy ===
        coverage_loss = nn.functional.cross_entropy(
            logits.view(-1, 3),
            (targets + 1).long().view(-1),
        )

        # === 3. Richness: Preserve within-level variance ===
        richness_loss = torch.tensor(0.0, device=device)

        if original_radii is not None:
            for v in unique_vals:
                mask = valuations == v
                if mask.sum() > 1:
                    new_var = radii[mask].var()
                    orig_var = original_radii[mask].var() + 1e-8

                    # Penalize if variance drops below threshold
                    ratio = new_var / orig_var
                    if ratio < self.min_richness_ratio:
                        # Strong penalty for collapsing variance
                        richness_loss = richness_loss + (self.min_richness_ratio - ratio) ** 2

            richness_loss = richness_loss / max(len(unique_vals), 1)

        # === 4. Separation: Ensure levels don't overlap ===
        separation_loss = torch.tensor(0.0, device=device)
        mean_radii_list = []

        for v in sorted(unique_vals.tolist()):
            mask = valuations == v
            if mask.sum() > 0:
                mean_radii_list.append((v, radii[mask].mean(), radii[mask].std()))

        for i in range(len(mean_radii_list) - 1):
            v1, r1, std1 = mean_radii_list[i]
            v2, r2, std2 = mean_radii_list[i + 1]
            # v2 > v1, so r2 should be < r1
            # Allow some overlap based on std, but enforce ordering
            margin = 0.01
            violation = torch.relu(r2 - r1 + margin)
            separation_loss = separation_loss + violation

        total = (
            self.hierarchy_weight * hierarchy_loss +
            self.coverage_weight * coverage_loss +
            self.richness_weight * richness_loss +
            self.separation_weight * separation_loss
        )

        return {
            'total': total,
            'hierarchy_loss': hierarchy_loss,
            'coverage_loss': coverage_loss,
            'richness_loss': richness_loss,
            'separation_loss': separation_loss,
        }


def compute_metrics(model, all_ops, indices, device):
    """Compute full metrics including dist_corr for Q."""
    model.eval()
    batch_size = 4096

    all_radii = []
    all_correct = []
    all_z = []

    with torch.no_grad():
        for i in range(0, len(all_ops), batch_size):
            batch_ops = all_ops[i:i+batch_size].to(device)

            out = model(batch_ops, compute_control=False)
            z_A = out['z_A_hyp']

            all_radii.append(z_A.norm(dim=-1).cpu().numpy())
            all_z.append(z_A.cpu().numpy())

            logits = model.decoder_A(out['mu_A'])
            preds = torch.argmax(logits, dim=-1) - 1
            correct = (preds == batch_ops.long()).float().mean(dim=1).cpu().numpy()
            all_correct.append(correct)

    all_radii = np.concatenate(all_radii)
    all_z = np.concatenate(all_z)
    all_correct = np.concatenate(all_correct)
    valuations = TERNARY.valuation(indices).numpy()

    coverage = (all_correct == 1.0).mean()
    hierarchy = spearmanr(valuations, all_radii)[0]

    # Compute within-level variance (richness)
    richness = 0
    for v in range(10):
        mask = valuations == v
        if mask.sum() > 1:
            richness += all_radii[mask].var()
    richness /= 10

    # Simple distance correlation proxy (correlation of pairwise distances)
    # Using a sample for speed
    sample_idx = np.random.choice(len(all_z), min(1000, len(all_z)), replace=False)
    z_sample = all_z[sample_idx]
    val_sample = valuations[sample_idx]

    # Compute pairwise distances
    z_dists = np.sqrt(((z_sample[:, None] - z_sample[None, :]) ** 2).sum(-1))
    val_dists = np.abs(val_sample[:, None] - val_sample[None, :]).astype(float)

    # Flatten upper triangle
    triu_idx = np.triu_indices(len(sample_idx), k=1)
    z_flat = z_dists[triu_idx]
    val_flat = val_dists[triu_idx]

    dist_corr = spearmanr(z_flat, val_flat)[0]

    model.train()

    return {
        'coverage': coverage,
        'hierarchy': hierarchy,
        'richness': richness,
        'dist_corr': dist_corr,
        'Q': compute_Q(dist_corr, hierarchy),
        'r_v0': all_radii[valuations == 0].mean(),
        'r_v9': all_radii[valuations == 9].mean() if (valuations == 9).any() else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description="Homeostatic training with richness")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--checkpoint", type=str,
                        default="sandbox-training/checkpoints/v5_11_homeostasis/best.pt")
    parser.add_argument("--save_dir", type=str,
                        default="sandbox-training/checkpoints/homeostatic_rich")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    save_dir = PROJECT_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # === Load Model ===
    print("\n=== Loading Model ===")
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.99,
        curvature=1.0,
        use_controller=False,
        use_dual_projection=True,
        freeze_encoder_b=False,  # Start unfrozen for Option C
        encoder_b_lr_scale=0.1,
        encoder_a_lr_scale=0.05,
    )

    ckpt_path = PROJECT_ROOT / args.checkpoint
    if ckpt_path.exists():
        print(f"Loading from: {ckpt_path}")
        ckpt = load_checkpoint_compat(ckpt_path, map_location=device)
        model_state = get_model_state_dict(ckpt)
        model.load_state_dict(model_state, strict=False)
        print(f"Loaded checkpoint (keys: {list(ckpt.keys())[:5]}...)")

    model = model.to(device)

    # Start with encoder_A frozen (homeostatic default)
    model.set_encoder_a_frozen(True)
    model.set_encoder_b_frozen(False)

    print(f"Freeze state: {model.get_freeze_state_summary()}")

    # === Dataset ===
    print("\n=== Loading Dataset ===")
    all_ops_np = generate_all_ternary_operations()
    all_ops = torch.tensor(all_ops_np, dtype=torch.float32)
    indices = torch.arange(len(all_ops))
    dataset = TensorDataset(all_ops, indices)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Get original radii for richness reference
    with torch.no_grad():
        model.eval()
        original_radii_list = []
        for i in range(0, len(all_ops), 4096):
            batch = all_ops[i:i+4096].to(device)
            out = model(batch, compute_control=False)
            original_radii_list.append(out['z_A_hyp'].norm(dim=-1))
        original_radii = torch.cat(original_radii_list)
        model.train()

    print(f"Original radii: {original_radii.min():.4f} - {original_radii.max():.4f}")

    # Initial metrics
    init_metrics = compute_metrics(model, all_ops, indices, device)
    print(f"Initial: cov={init_metrics['coverage']*100:.1f}%, hier={init_metrics['hierarchy']:.4f}, rich={init_metrics['richness']:.6f}, Q={init_metrics['Q']:.3f}")
    initial_richness = init_metrics['richness']

    # === Homeostasis Controller ===
    homeostasis = HomeostasisController(
        coverage_freeze_threshold=0.995,
        coverage_unfreeze_threshold=0.999,
        enable_annealing=True,
    )

    # === Loss & Optimizer ===
    loss_fn = RichHierarchyLoss(
        hierarchy_weight=5.0,
        coverage_weight=1.0,
        richness_weight=2.0,
        separation_weight=3.0,
        min_richness_ratio=0.5,
    ).to(device)

    # === Training Loop ===
    print("\n=== Starting Homeostatic Training ===")

    best_Q = 0.0
    best_hierarchy = 0.0

    for epoch in range(args.epochs):
        model.train()

        # Rebuild optimizer based on current freeze states
        param_groups = model.get_param_groups(args.lr)
        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

        epoch_losses = {'total': 0, 'hierarchy': 0, 'coverage': 0, 'richness': 0}
        n_batches = 0

        for batch_ops, batch_idx in dataloader:
            batch_ops = batch_ops.to(device)
            batch_idx = batch_idx.to(device)
            orig_radii_batch = original_radii[batch_idx]

            out = model(batch_ops, compute_control=False)
            z_A = out['z_A_hyp']
            mu_A = out['mu_A']
            logits = model.decoder_A(mu_A)

            losses = loss_fn(z_A, batch_idx, logits, batch_ops, orig_radii_batch)

            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses['total'] += losses['total'].item()
            epoch_losses['hierarchy'] += losses['hierarchy_loss'].item()
            epoch_losses['coverage'] += losses['coverage_loss'].item()
            epoch_losses['richness'] += losses['richness_loss'].item()
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        # Compute metrics and update homeostasis
        if epoch % 3 == 0 or epoch == args.epochs - 1:
            metrics = compute_metrics(model, all_ops, indices, device)
            richness_ratio = metrics['richness'] / (initial_richness + 1e-10)

            # Update homeostasis
            homeo_state = homeostasis.update(
                epoch=epoch,
                coverage=metrics['coverage'],
                hierarchy_A=metrics['hierarchy'],
                hierarchy_B=metrics['hierarchy'],  # Same for now
                dist_corr_A=metrics['dist_corr'],
            )

            # Apply freeze states
            model.apply_homeostasis_state(homeo_state)

            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Loss: {epoch_losses['total']:.4f}")
            print(f"  Coverage: {metrics['coverage']*100:.2f}%")
            print(f"  Hierarchy: {metrics['hierarchy']:.4f}")
            print(f"  Richness ratio: {richness_ratio:.4f} (target: >0.5)")
            print(f"  Q: {metrics['Q']:.3f} (best: {best_Q:.3f})")
            print(f"  Freeze: {model.get_freeze_state_summary()}")

            if homeo_state['events']:
                print(f"  Events: {homeo_state['events']}")

            # Track best
            is_best_Q = metrics['Q'] > best_Q
            is_best_hier = metrics['hierarchy'] < best_hierarchy

            if is_best_Q:
                best_Q = metrics['Q']
                print(f"  [NEW BEST Q: {best_Q:.3f}]")

            if is_best_hier and metrics['coverage'] > 0.99:
                best_hierarchy = metrics['hierarchy']
                print(f"  [NEW BEST HIERARCHY: {best_hierarchy:.4f}]")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': metrics,
                    'richness_ratio': richness_ratio,
                    'homeostasis_state': homeostasis.get_state_summary(),
                    'config': vars(args),
                }, save_dir / 'best.pt')

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
            }, save_dir / 'latest.pt')

    print("\n=== Training Complete ===")
    print(f"Best Q: {best_Q:.3f}")
    print(f"Best Hierarchy: {best_hierarchy:.4f}")
    print(f"Saved to: {save_dir}")


if __name__ == '__main__':
    main()
