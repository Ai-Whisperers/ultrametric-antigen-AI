#!/usr/bin/env python3
"""
Extract embeddings from checkpoints for contact prediction experiments.

Extracts z_A_hyp and z_B_hyp embeddings for all 19,683 ternary operations
from specified checkpoints.

Usage:
    python extract_embeddings.py --checkpoint final_rich_lr5e5_best.pt
"""

import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ternary_vae import TernaryVAEV5_11_PartialFreeze
from src.core import TERNARY


def extract_embeddings(checkpoint_path: Path, output_path: Path):
    """Extract embeddings from a checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Determine model configuration from checkpoint
    state_dict = ckpt.get('model_state_dict', ckpt)

    # Infer dimensions from state dict
    encoder_a_weight = state_dict.get('encoder_A.net.0.weight')
    if encoder_a_weight is not None:
        hidden_dim = encoder_a_weight.shape[0]
    else:
        hidden_dim = 64  # default

    latent_dim = 16  # standard for this project

    print(f"  Inferred hidden_dim={hidden_dim}, latent_dim={latent_dim}")

    # Create model
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        max_radius=0.99,
        curvature=1.0,
        use_controller=False,
        use_dual_projection=True,
    )

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"  Model loaded successfully")

    # Generate all operations
    all_ops = torch.arange(TERNARY.n_ops)  # 19,683 operations
    print(f"  Extracting embeddings for {len(all_ops)} operations...")

    with torch.no_grad():
        out = model(all_ops, compute_control=False)

    z_A_hyp = out['z_A_hyp'].numpy()
    z_B_hyp = out['z_B_hyp'].numpy()

    # Compute metrics
    from scipy.stats import spearmanr
    valuations = TERNARY.valuation(all_ops).numpy()

    # V5.12.2: Use hyperbolic distance for radii
    def hyperbolic_radius(z, c=1.0):
        norm = np.linalg.norm(z, axis=-1)
        norm = np.clip(norm, 0, 1 - 1e-7)
        return (2.0 / np.sqrt(c)) * np.arctanh(norm)

    radii_A = hyperbolic_radius(z_A_hyp)
    radii_B = hyperbolic_radius(z_B_hyp)

    hier_A = spearmanr(valuations, radii_A)[0]
    hier_B = spearmanr(valuations, radii_B)[0]

    # Compute richness (within-level variance)
    richness = 0.0
    for v in range(10):
        mask = valuations == v
        if mask.sum() > 1:
            richness += radii_B[mask].var()
    richness /= 10

    print(f"  Hierarchy A: {hier_A:.4f}")
    print(f"  Hierarchy B: {hier_B:.4f}")
    print(f"  Richness: {richness:.6f}")

    # Save embeddings
    output_data = {
        'z_A_hyp': torch.tensor(z_A_hyp),
        'z_B_hyp': torch.tensor(z_B_hyp),
        'radii_A': torch.tensor(radii_A),
        'radii_B': torch.tensor(radii_B),
        'valuations': torch.tensor(valuations),
        'metrics': {
            'hierarchy_A': hier_A,
            'hierarchy_B': hier_B,
            'richness': richness,
            'n_operations': len(all_ops),
        },
        'source_checkpoint': str(checkpoint_path.name),
    }

    torch.save(output_data, output_path)
    print(f"  Saved to: {output_path}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description='Extract embeddings from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint filename in checkpoints/ folder')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: derived from checkpoint)')
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent
    checkpoint_path = script_dir / 'checkpoints' / args.checkpoint

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if args.output:
        output_path = script_dir / 'embeddings' / args.output
    else:
        output_name = args.checkpoint.replace('.pt', '_embeddings.pt')
        output_path = script_dir / 'embeddings' / output_name

    extract_embeddings(checkpoint_path, output_path)


if __name__ == '__main__':
    main()
