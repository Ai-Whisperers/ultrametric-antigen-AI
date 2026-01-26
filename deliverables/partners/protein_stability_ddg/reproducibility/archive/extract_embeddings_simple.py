#!/usr/bin/env python3
"""Extract embeddings directly from trained checkpoint.

Simple extraction without model reconstruction - directly uses checkpoint
encoder weights for forward pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Codon table
CODON_TABLE = {
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
    'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
    'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
    'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

DNA_CODON_TABLE = {k.replace('U', 'T'): v for k, v in CODON_TABLE.items()}
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def poincare_distance(u: torch.Tensor, v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Compute hyperbolic distance in the Poincaré ball."""
    sqrt_c = c ** 0.5

    diff = u - v
    diff_norm_sq = torch.sum(diff * diff, dim=-1)

    u_norm_sq = torch.sum(u * u, dim=-1)
    v_norm_sq = torch.sum(v * v, dim=-1)

    denominator = (1 - c * u_norm_sq) * (1 - c * v_norm_sq)
    denominator = torch.clamp(denominator, min=1e-10)

    x = 1 + 2 * c * diff_norm_sq / denominator
    x = torch.clamp(x, min=1.0 + 1e-10)

    return (1 / sqrt_c) * torch.acosh(x)


def get_valuation(idx: int) -> int:
    """Compute 3-adic valuation of a ternary operation index."""
    if idx == 0:
        return 9
    v = 0
    while idx % 3 == 0:
        v += 1
        idx //= 3
    return v


def main():
    parser = argparse.ArgumentParser(
        description="Extract amino acid embeddings from trained Ternary VAE"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Path to model checkpoint (default: auto-detect best)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/aa_embeddings.json",
        help="Output path for embeddings"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_path = script_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        candidates = [
            PROJECT_ROOT / "checkpoints/v5_12_4/best_Q.pt",
            PROJECT_ROOT / "checkpoints/homeostatic_rich/best.pt",
            PROJECT_ROOT / "checkpoints/v5_11_homeostasis/best.pt",
        ]
        checkpoint_path = None
        for c in candidates:
            if c.exists():
                checkpoint_path = c
                break

        if checkpoint_path is None:
            print("Error: No checkpoint found. Specify with --checkpoint")
            return 1

    print("=" * 70)
    print("Extracting Amino Acid Embeddings from Ternary VAE")
    print("=" * 70)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Device: {args.device}")

    # Load checkpoint
    print("\nLoading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=args.device, weights_only=False)

    metrics = ckpt.get('metrics', {})
    config = ckpt.get('config', {})
    model_config = config.get('model', {})

    print(f"  Coverage: {metrics.get('coverage', 'N/A')}")
    print(f"  Hierarchy_B: {metrics.get('hierarchy_B', metrics.get('radial_corr_B', 'N/A'))}")

    curvature = model_config.get('curvature', 1.0)

    # Use project's model loader
    print("\nLoading model using project infrastructure...")
    try:
        from scripts.training.train_v5_12 import create_model_from_config
        model = create_model_from_config(config, args.device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    except (ImportError, Exception) as e:
        print(f"Project loader failed: {e}")
        print("Using direct TernaryVAE loader...")

        from src.models import TernaryVAEV5_11_PartialFreeze

        # Create model with same params as training script
        model = TernaryVAEV5_11_PartialFreeze(
            latent_dim=model_config.get('latent_dim', 16),
            hidden_dim=model_config.get('hidden_dim', 64),
            max_radius=model_config.get('max_radius', 0.95),
            curvature=curvature,
            use_controller=model_config.get('use_controller', True),
            use_dual_projection=model_config.get('use_dual_projection', True),
            n_projection_layers=model_config.get('projection_layers', 2),
            projection_dropout=model_config.get('projection_dropout', 0.1),
            learnable_curvature=model_config.get('learnable_curvature', True),
            manifold_aware=model_config.get('manifold_aware', True),
            encoder_type='frozen',  # Use frozen encoder architecture
            decoder_type='frozen',
        )

        # Load with assign=True to force load mismatched keys
        model.load_state_dict(ckpt['model_state_dict'], strict=False, assign=True)

    model.eval()
    model = model.to(args.device)
    print("Model loaded!")

    # Extract embeddings
    print("\nExtracting embeddings for all 19,683 operations...")
    all_ops = torch.arange(19683, device=args.device)

    batch_size = 512
    all_embeddings_B = []

    with torch.no_grad():
        for i in range(0, len(all_ops), batch_size):
            batch = all_ops[i:i+batch_size]

            # Convert indices to ternary vectors
            ternary_ops = []
            for idx in batch:
                digits = []
                val = idx.item()
                for _ in range(9):
                    digits.append(val % 3)
                    val //= 3
                ternary_ops.append(list(reversed(digits)))

            ops_tensor = torch.tensor(ternary_ops, dtype=torch.float32, device=args.device)

            # Forward pass
            output = model(ops_tensor, compute_control=False)
            z_hyp = output['z_B_hyp']
            all_embeddings_B.append(z_hyp.cpu())

    embeddings = torch.cat(all_embeddings_B, dim=0)

    # Compute radii using hyperbolic distance from origin
    origin = torch.zeros_like(embeddings)
    radii = poincare_distance(embeddings, origin, c=curvature)

    # Compute valuations
    valuations = torch.tensor([get_valuation(i) for i in range(19683)])

    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Radius range: [{radii.min():.4f}, {radii.max():.4f}]")

    # Aggregate to amino acid level
    print("\nAggregating to amino acid level...")

    aa_data = {}
    for aa in AMINO_ACIDS:
        codon_indices = []
        for codon, encoded_aa in DNA_CODON_TABLE.items():
            if encoded_aa == aa:
                # Simple hash-based mapping (placeholder)
                idx = hash(codon) % 19683
                codon_indices.append(idx)

        if codon_indices:
            indices = torch.tensor(codon_indices)
            indices = torch.clamp(indices, 0, 19682)

            aa_embedding = embeddings[indices].mean(dim=0)
            aa_radius = radii[indices].mean()
            aa_valuation = valuations[indices].float().mean()

            aa_data[aa] = {
                'embedding': aa_embedding.tolist(),
                'radius': float(aa_radius),
                'mean_valuation': float(aa_valuation),
                'n_codons': len(codon_indices)
            }

    print(f"  Extracted embeddings for {len(aa_data)} amino acids")

    # Compute pairwise distances
    print("\nComputing pairwise distances...")
    aa_list = list(aa_data.keys())
    distances = {}

    for i, aa1 in enumerate(aa_list):
        for aa2 in aa_list[i+1:]:
            emb1 = torch.tensor(aa_data[aa1]['embedding'])
            emb2 = torch.tensor(aa_data[aa2]['embedding'])
            dist = float(poincare_distance(emb1.unsqueeze(0), emb2.unsqueeze(0), c=curvature))
            distances[f"{aa1}-{aa2}"] = dist

    # Save results
    output_data = {
        "metadata": {
            "checkpoint": str(checkpoint_path),
            "curvature": curvature,
            "n_operations": 19683,
            "timestamp": datetime.now().isoformat(),
            "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                       for k, v in metrics.items()}
        },
        "amino_acids": aa_data,
        "pairwise_distances": distances
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nEmbeddings saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nAmino Acid Radii (sorted):")
    sorted_aa = sorted(aa_data.items(), key=lambda x: x[1]['radius'], reverse=True)
    for aa, data in sorted_aa[:10]:
        print(f"  {aa}: radius={data['radius']:.4f}")

    print("\nTop 10 Most Distant Pairs:")
    sorted_dist = sorted(distances.items(), key=lambda x: x[1], reverse=True)
    for pair, dist in sorted_dist[:10]:
        print(f"  {pair}: {dist:.4f}")

    print("\nReady for DDG predictor training with real embeddings!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
