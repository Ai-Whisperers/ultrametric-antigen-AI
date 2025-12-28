#!/usr/bin/env python3
"""
Extract V5.11.3 Hyperbolic Embeddings for Codon Encoder Training

This script loads the production-ready V5.11.3 Ternary VAE model and extracts
native hyperbolic embeddings from the Poincaré ball for training the
codon-encoder-3-adic.

The V5.11.3 model has:
- Hierarchy correlation: -0.730
- Pairwise ordering accuracy: 92.4%
- Ultrametric compliance: 90%+
- v9 radius: ~0.10, v0 radius: ~0.90

Output: research/genetic_code/data/v5_11_3_embeddings.pt
"""

import sys
from pathlib import Path

import torch
from scipy.stats import spearmanr

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR
from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.models.ternary_vae import TernaryVAEV5_11_PartialFreeze


def load_v5_11_3_model(device="cpu"):
    """Load the V5.11.3 model with proper architecture settings.

    The model uses:
    - Frozen encoder_A from v5.5
    - Trainable encoder_B (PartialFreeze variant)
    - Dual hyperbolic projection
    - 2 projection layers with dropout
    """
    print("Loading V5.11.3 model...")

    # V5.11.3 architecture parameters (from training config)
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=128,
        max_radius=0.95,
        curvature=1.0,
        use_dual_projection=True,
        n_projection_layers=2,
        projection_dropout=0.1,
        freeze_encoder_b=False,  # Option C
        encoder_b_lr_scale=0.1,
    )

    # Load v5.5 base checkpoint (frozen encoders)
    v5_5_path = CHECKPOINTS_DIR / "v5_5" / "latest.pt"
    if not v5_5_path.exists():
        raise FileNotFoundError(f"V5.5 checkpoint not found: {v5_5_path}")

    print(f"  Loading v5.5 base from: {v5_5_path}")
    model.load_v5_5_checkpoint(v5_5_path, device=device)

    # Load V5.11.3 trained weights (projection layer + encoder_B)
    v5_11_3_path = CHECKPOINTS_DIR / "v5_11_structural" / "best.pt"
    if not v5_11_3_path.exists():
        raise FileNotFoundError(f"V5.11.3 checkpoint not found: {v5_11_3_path}")

    print(f"  Loading V5.11.3 weights from: {v5_11_3_path}")
    checkpoint = torch.load(v5_11_3_path, map_location=device, weights_only=False)

    # Extract trainable state (projection + encoder_B)
    if "model_state" in checkpoint:
        state = checkpoint["model_state"]
    else:
        state = checkpoint

    # Load only trainable components
    trainable_keys = [k for k in state.keys() if "projection" in k or "encoder_B" in k or "controller" in k]
    trainable_state = {k: v for k, v in state.items() if k in trainable_keys}

    missing, unexpected = model.load_state_dict(trainable_state, strict=False)
    print(f"  Loaded {len(trainable_state)} trainable parameters")

    # Print checkpoint metadata if available
    if "metrics" in checkpoint:
        metrics = checkpoint["metrics"]
        print("  Checkpoint metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")

    model.to(device)
    model.eval()

    return model


def extract_embeddings(model, device="cpu"):
    """Extract hyperbolic embeddings for all 19,683 ternary operations.

    Returns:
        dict with:
            - z_A_hyp: VAE-A hyperbolic embeddings (19683, 16)
            - z_B_hyp: VAE-B hyperbolic embeddings (19683, 16)
            - z_A_euc: VAE-A Euclidean embeddings (19683, 16)
            - z_B_euc: VAE-B Euclidean embeddings (19683, 16)
            - valuations: 3-adic valuations (19683,)
            - radii_A: VAE-A radii (19683,)
            - radii_B: VAE-B radii (19683,)
            - operations: raw ternary operations (19683, 9)
    """
    print("\nExtracting embeddings for all 19,683 ternary operations...")

    # Generate all operations
    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)
    indices = torch.arange(len(operations), device=device)

    # Compute valuations
    valuations = TERNARY.valuation(indices)

    # Forward pass
    with torch.no_grad():
        outputs = model(x, compute_control=False)

    # Extract embeddings
    z_A_hyp = outputs["z_A_hyp"].cpu()
    z_B_hyp = outputs["z_B_hyp"].cpu()
    z_A_euc = outputs["z_A_euc"].cpu()
    z_B_euc = outputs["z_B_euc"].cpu()

    # Compute radii
    radii_A = torch.norm(z_A_hyp, dim=1)
    radii_B = torch.norm(z_B_hyp, dim=1)

    print(f"  z_A_hyp shape: {z_A_hyp.shape}")
    print(f"  z_B_hyp shape: {z_B_hyp.shape}")
    print(f"  Radii range (A): [{radii_A.min():.4f}, {radii_A.max():.4f}]")
    print(f"  Radii range (B): [{radii_B.min():.4f}, {radii_B.max():.4f}]")

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
    }


def validate_embeddings(embeddings):
    """Validate that embeddings have expected properties.

    Checks:
    - Hierarchy correlation (should be <= -0.70)
    - Radii distribution (v9 ~0.10, v0 ~0.90)
    - All operations embedded
    """
    print("\nValidating extracted embeddings...")

    valuations = embeddings["valuations"].numpy()
    radii_A = embeddings["radii_A"].numpy()
    radii_B = embeddings["radii_B"].numpy()

    # Check completeness
    n_ops = len(valuations)
    assert n_ops == 19683, f"Expected 19683 operations, got {n_ops}"
    print(f"  Total operations: {n_ops} (expected 19683)")

    # Hierarchy correlation (VAE-B is the one we train on)
    corr_A, p_A = spearmanr(valuations, radii_A)
    corr_B, p_B = spearmanr(valuations, radii_B)

    print("\n  Hierarchy correlation (valuation vs radius):")
    print(f"    VAE-A: r = {corr_A:.4f} (p = {p_A:.2e})")
    print(f"    VAE-B: r = {corr_B:.4f} (p = {p_B:.2e})")

    # Check radii by valuation
    print("\n  Radii by valuation (VAE-B):")
    for v in range(10):
        mask = valuations == v
        if mask.sum() > 0:
            mean_r = radii_B[mask].mean()
            std_r = radii_B[mask].std()
            print(f"    v={v}: n={mask.sum():5d}, radius={mean_r:.4f} ± {std_r:.4f}")

    # Validation checks
    passed = True

    if corr_B > -0.60:
        print(f"\n  WARNING: Hierarchy correlation ({corr_B:.4f}) is weaker than expected (< -0.60)")
        passed = False
    else:
        print(f"\n  PASS: Hierarchy correlation is strong ({corr_B:.4f})")

    # Check v9 (should be near center)
    v9_mask = valuations == 9
    if v9_mask.sum() > 0:
        v9_radius = radii_B[v9_mask].mean()
        if v9_radius > 0.20:
            print(f"  WARNING: v9 radius ({v9_radius:.4f}) is too large (expected ~0.10)")
            passed = False
        else:
            print(f"  PASS: v9 radius ({v9_radius:.4f}) is near center")

    # Check v0 (should be near boundary)
    v0_mask = valuations == 0
    v0_radius = radii_B[v0_mask].mean()
    if v0_radius < 0.70:
        print(f"  WARNING: v0 radius ({v0_radius:.4f}) is too small (expected ~0.90)")
        passed = False
    else:
        print(f"  PASS: v0 radius ({v0_radius:.4f}) is near boundary")

    return passed, corr_B


def main():
    print("=" * 70)
    print("EXTRACT V5.11.3 HYPERBOLIC EMBEDDINGS")
    print("=" * 70)

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_v5_11_3_model(device)

    # Extract embeddings
    embeddings = extract_embeddings(model, device)

    # Validate
    passed, hierarchy_corr = validate_embeddings(embeddings)

    # Save
    output_path = output_dir / "v5_11_3_embeddings.pt"

    # Add metadata
    embeddings["metadata"] = {
        "model_version": "V5.11.3",
        "checkpoint": "v5_11_structural/best.pt",
        "hierarchy_correlation": float(hierarchy_corr),
        "n_operations": 19683,
        "latent_dim": 16,
        "max_radius": 0.95,
        "device": device,
    }

    torch.save(embeddings, output_path)
    print(f"\nSaved embeddings to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Operations embedded: {len(embeddings['valuations'])}")
    print(f"  Hierarchy correlation: {hierarchy_corr:.4f}")
    print(f"  Validation: {'PASSED' if passed else 'WARNINGS (see above)'}")
    print(f"  Output file: {output_path}")
    print("=" * 70)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
