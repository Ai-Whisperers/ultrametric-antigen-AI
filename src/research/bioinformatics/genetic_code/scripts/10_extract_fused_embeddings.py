#!/usr/bin/env python3
"""
Extract Fused Embeddings for Codon Encoder Training

This script creates embeddings from homeostatic_rich (best hierarchy + richness)
which already uses frozen v5_5 encoders internally.

The homeostatic_rich checkpoint provides:
- Coverage: 100% (from frozen v5_5 encoders)
- Hierarchy: ~-0.83 (from trained hyperbolic projection)
- Richness: High (from homeostatic training)

Output: research/genetic_code/data/fused_embeddings.pt
"""

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from scipy.stats import spearmanr

from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.models.ternary_vae import TernaryVAEV5_11_PartialFreeze

# Use sandbox-training checkpoints directory
CHECKPOINTS_DIR = PROJECT_ROOT / "sandbox-training" / "checkpoints"


# =============================================================================
# POINCARE DISTANCE FUNCTIONS
# =============================================================================


def poincare_distance(x, y, c=1.0, eps=1e-10):
    """Compute Poincare ball geodesic distance."""
    norm_x_sq = torch.sum(x**2, dim=-1, keepdim=True)
    norm_y_sq = torch.sum(y**2, dim=-1, keepdim=True)
    diff_sq = torch.sum((x - y) ** 2, dim=-1, keepdim=True)

    denom = (1 - c * norm_x_sq) * (1 - c * norm_y_sq)
    denom = torch.clamp(denom, min=eps)

    arg = 1 + 2 * c * diff_sq / denom
    arg = torch.clamp(arg, min=1.0 + eps)

    dist = (1 / np.sqrt(c)) * torch.acosh(arg)
    return dist.squeeze(-1)


# =============================================================================
# MODEL LOADING
# =============================================================================


def load_model(checkpoint_name, device="cpu"):
    """Load TernaryVAEV5_11_PartialFreeze with proper weight loading.

    Args:
        checkpoint_name: Name of checkpoint directory (e.g., "homeostatic_rich", "v5_11_homeostasis")
        device: Device to load to
    """
    print(f"Loading model from {checkpoint_name}...")

    # Create model with correct architecture
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.99,
        curvature=1.0,
        use_dual_projection=True,
        n_projection_layers=1,
        projection_dropout=0.0,
        freeze_encoder_b=False,
        encoder_b_lr_scale=0.1,
    )

    # Load v5.5 base (frozen encoders)
    v5_5_path = CHECKPOINTS_DIR / "v5_5" / "best.pt"
    if not v5_5_path.exists():
        v5_5_path = CHECKPOINTS_DIR / "v5_5" / "latest.pt"
    if not v5_5_path.exists():
        raise FileNotFoundError(f"v5_5 checkpoint not found")

    print(f"  Loading v5_5 base from: {v5_5_path}")
    model.load_v5_5_checkpoint(v5_5_path, device=device)

    # Load trained checkpoint
    ckpt_path = CHECKPOINTS_DIR / checkpoint_name / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINTS_DIR / checkpoint_name / "latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_name}")

    print(f"  Loading trained weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Get model state
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        state = checkpoint["model_state"]
    else:
        state = checkpoint

    # Load trainable components (projection + controller)
    trainable_keys = [k for k in state.keys()
                      if "projection" in k or "controller" in k or "encoder_B" in k]
    trainable_state = {k: v for k, v in state.items() if k in trainable_keys}

    missing, unexpected = model.load_state_dict(trainable_state, strict=False)
    print(f"  Loaded {len(trainable_state)} trainable weights")

    # Print metrics
    if "metrics" in checkpoint:
        metrics = checkpoint["metrics"]
        print("  Checkpoint metrics:")
        for k, v in metrics.items():
            if isinstance(v, (float, np.floating)):
                print(f"    {k}: {float(v):.4f}")

    model.to(device)
    model.eval()

    return model, checkpoint.get("metrics", {})


def extract_embeddings(model, device="cpu"):
    """Extract hyperbolic embeddings for all 19,683 ternary operations."""
    print("\nExtracting embeddings...")

    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)
    indices = torch.arange(len(operations), device=device)

    valuations = TERNARY.valuation(indices)

    with torch.no_grad():
        outputs = model(x, compute_control=False)

    z_A_hyp = outputs["z_A_hyp"].cpu()
    z_B_hyp = outputs["z_B_hyp"].cpu()
    z_A_euc = outputs["z_A_euc"].cpu()
    z_B_euc = outputs["z_B_euc"].cpu()

    radii_A = torch.norm(z_A_hyp, dim=1)
    radii_B = torch.norm(z_B_hyp, dim=1)

    # Compute hierarchy correlation
    vals = valuations.cpu().numpy()
    corr_A = spearmanr(vals, radii_A.numpy())[0]
    corr_B = spearmanr(vals, radii_B.numpy())[0]

    print(f"  Shape: {z_B_hyp.shape}")
    print(f"  Radii (A): [{radii_A.min():.4f}, {radii_A.max():.4f}]")
    print(f"  Radii (B): [{radii_B.min():.4f}, {radii_B.max():.4f}]")
    print(f"  Hierarchy: A={corr_A:.4f}, B={corr_B:.4f}")

    # Compute richness (within-valuation variance)
    richness_values = []
    for v in range(10):
        mask = valuations.cpu() == v
        if mask.sum() > 1:
            var = radii_B[mask].var().item()
            richness_values.append(var)
    richness = np.mean(richness_values) if richness_values else 0.0
    print(f"  Richness: {richness:.6f}")

    # Radii by valuation
    print("\n  Radii by valuation (VAE-B):")
    for v in range(10):
        mask = vals == v
        if mask.sum() > 0:
            mean_r = radii_B[mask].mean()
            std_r = radii_B[mask].std()
            print(f"    v={v}: n={mask.sum():5d}, radius={mean_r:.4f} +/- {std_r:.4f}")

    return {
        "z_A_hyp": z_A_hyp,
        "z_B_hyp": z_B_hyp,
        "z_A_euc": z_A_euc,
        "z_B_euc": z_B_euc,
        "valuations": valuations.cpu(),
        "radii_A": radii_A,
        "radii_B": radii_B,
        "operations": torch.tensor(operations),
        "indices": indices.cpu(),
        "hierarchy_A": corr_A,
        "hierarchy_B": corr_B,
        "richness": richness,
    }


def compute_coverage(model, x, device="cpu"):
    """Compute reconstruction coverage."""
    with torch.no_grad():
        outputs = model(x, compute_control=False)
        logits = outputs.get("logits_A")
        if logits is None:
            return None
        preds = logits.argmax(dim=-1) - 1  # Back to {-1, 0, 1}
        correct = (preds == x.long()).all(dim=1).float().mean().item()
    return correct


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("EXTRACT EMBEDDINGS FOR CODON ENCODER")
    print("Using homeostatic_rich / v5_11_homeostasis")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Try checkpoints in order of preference
    checkpoints_to_try = ["homeostatic_rich", "v5_11_homeostasis", "v5_11_structural"]

    model = None
    metrics = {}
    checkpoint_used = None

    for ckpt_name in checkpoints_to_try:
        ckpt_path = CHECKPOINTS_DIR / ckpt_name / "best.pt"
        if ckpt_path.exists() or (CHECKPOINTS_DIR / ckpt_name / "latest.pt").exists():
            try:
                model, metrics = load_model(ckpt_name, device)
                checkpoint_used = ckpt_name
                break
            except Exception as e:
                print(f"  Warning: Failed to load {ckpt_name}: {e}")
                continue

    if model is None:
        print("ERROR: No valid checkpoint found")
        return 1

    # Extract embeddings
    embeddings = extract_embeddings(model, device)

    # Check coverage
    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)
    coverage = compute_coverage(model, x, device)

    if coverage is not None:
        print(f"\nCoverage: {coverage * 100:.2f}%")

    # Save embeddings
    output = {
        # Main embeddings
        "z_A_hyp": embeddings["z_A_hyp"],
        "z_B_hyp": embeddings["z_B_hyp"],
        "z_A_euc": embeddings["z_A_euc"],
        "z_B_euc": embeddings["z_B_euc"],

        # For fusion compatibility
        "z_fused": embeddings["z_B_hyp"],  # Use z_B_hyp as fused (already has hierarchy)

        # Metadata
        "valuations": embeddings["valuations"],
        "radii_A": embeddings["radii_A"],
        "radii_B": embeddings["radii_B"],
        "operations": embeddings["operations"],
        "indices": embeddings["indices"],

        # Metrics
        "metadata": {
            "checkpoint": checkpoint_used,
            "hierarchy_correlation": embeddings["hierarchy_B"],
            "hierarchy_A": embeddings["hierarchy_A"],
            "hierarchy_B": embeddings["hierarchy_B"],
            "richness": embeddings["richness"],
            "coverage": coverage,
            "n_operations": 19683,
            "latent_dim": 16,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_metrics": metrics,
        }
    }

    # Save as fused_embeddings.pt
    fused_path = data_dir / "fused_embeddings.pt"
    torch.save(output, fused_path)
    print(f"\nSaved to: {fused_path}")

    # Also update v5_11_3_embeddings.pt for compatibility
    compat_path = data_dir / "v5_11_3_embeddings.pt"
    torch.save(output, compat_path)
    print(f"Updated: {compat_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Checkpoint: {checkpoint_used}")
    print(f"  Operations: 19,683")
    print(f"  Hierarchy (A): {embeddings['hierarchy_A']:.4f}")
    print(f"  Hierarchy (B): {embeddings['hierarchy_B']:.4f}")
    print(f"  Richness: {embeddings['richness']:.6f}")
    if coverage:
        print(f"  Coverage: {coverage * 100:.2f}%")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
