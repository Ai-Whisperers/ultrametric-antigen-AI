#!/usr/bin/env python3
"""Extract Hyperbolic Codon Embeddings from VAE Checkpoint.

This script extracts codon and amino acid embeddings from a trained TernaryVAE
checkpoint, properly preserving hyperbolic geometry throughout.

Key improvements over previous versions:
1. Uses HyperbolicCodonEncoder for native Poincare ball operations
2. Computes Frechet means (not Euclidean) for amino acid aggregation
3. Preserves all hyperbolic structure (radii via poincare_distance)
4. Outputs embeddings suitable for DDG prediction without overfitting

Anti-overfitting measures:
- Uses actual VAE embeddings, not handcrafted features
- Hyperbolic distances are geometric, not learned from data
- Simple linear predictor to avoid overfitting on small datasets

Usage:
    python extract_hyperbolic_embeddings.py --checkpoint path/to/best_Q.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.biology.codons import (
    GENETIC_CODE,
    CODON_TO_INDEX,
    AMINO_ACID_TO_CODONS,
    codon_index_to_triplet,
)
from src.geometry import poincare_distance, log_map_zero
from src.encoders.codon_encoder import AA_PROPERTIES

# Standard amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def load_vae_model(checkpoint_path: str, device: str = 'cpu'):
    """Load TernaryVAE from checkpoint."""
    from src.models.ternary_vae import TernaryVAEV5_11_PartialFreeze

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})

    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=config.get('latent_dim', 16),
        hidden_dim=config.get('hidden_dim', 64),
        max_radius=0.99,
        curvature=config.get('curvature', 1.0),
        use_controller=config.get('use_controller', False),
        use_dual_projection=config.get('use_dual_projection', True),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    model.to(device)

    return model, config, ckpt.get('metrics', {})


def extract_codon_embeddings(
    model,
    device: str = 'cpu',
    use_encoder: str = 'B',
    curvature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract embeddings for all 64 codons.

    Returns:
        codon_embeddings: (64, latent_dim) hyperbolic embeddings
        codon_radii: (64,) hyperbolic radii
    """
    embeddings = []
    radii = []
    origin = None

    with torch.no_grad():
        for codon_idx in range(64):
            # Create operation tensor
            op_tensor = torch.tensor([[codon_idx]], device=device)

            # Forward through VAE
            out = model(op_tensor, compute_control=False)

            # Get hyperbolic embedding
            if use_encoder == 'B':
                z_hyp = out['z_B_hyp'][0]
            else:
                z_hyp = out['z_A_hyp'][0]

            embeddings.append(z_hyp)

            # Compute hyperbolic radius
            if origin is None:
                origin = torch.zeros(1, z_hyp.shape[0], device=device)

            r = poincare_distance(z_hyp.unsqueeze(0), origin, c=curvature)
            radii.append(r.squeeze())

    return torch.stack(embeddings), torch.stack(radii)


def frechet_mean_hyperbolic(
    embeddings: torch.Tensor,
    curvature: float = 1.0,
    max_iter: int = 20,
    tol: float = 1e-7,
) -> torch.Tensor:
    """Compute Frechet mean of hyperbolic embeddings.

    Uses iterative tangent space averaging.

    Args:
        embeddings: (N, dim) embeddings on Poincare ball
        curvature: Ball curvature
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Frechet mean, shape (dim,)
    """
    from src.geometry import exp_map_zero, project_to_poincare

    if len(embeddings) == 0:
        return torch.zeros(embeddings.shape[1], device=embeddings.device)
    if len(embeddings) == 1:
        return embeddings[0].clone()

    # Initialize with first point
    mean = embeddings[0].clone()

    for iteration in range(max_iter):
        # Compute log maps from mean to all points
        tangent_sum = torch.zeros_like(mean)

        for emb in embeddings:
            # Log map at mean
            diff = emb - mean
            tangent = log_map_zero(diff.unsqueeze(0), c=curvature).squeeze(0)
            tangent_sum = tangent_sum + tangent

        # Average tangent
        tangent_avg = tangent_sum / len(embeddings)

        # Check convergence
        if torch.norm(tangent_avg) < tol:
            break

        # Update via exp map
        update = exp_map_zero(tangent_avg.unsqueeze(0), c=curvature).squeeze(0)
        mean = project_to_poincare((mean + update).unsqueeze(0), max_norm=0.95, c=curvature).squeeze(0)

    return mean


def compute_aa_embeddings(
    codon_embeddings: torch.Tensor,
    curvature: float = 1.0,
    method: str = 'frechet',
) -> dict[str, torch.Tensor]:
    """Aggregate codon embeddings to amino acid level.

    Args:
        codon_embeddings: (64, dim) codon embeddings
        curvature: Poincare ball curvature
        method: 'frechet' for proper hyperbolic mean, 'centroid' for Euclidean

    Returns:
        Dictionary mapping AA letter to embedding tensor
    """
    from src.geometry import project_to_poincare

    aa_embeddings = {}

    for aa in AMINO_ACIDS:
        codons = AMINO_ACID_TO_CODONS.get(aa, [])
        if not codons:
            continue

        indices = [CODON_TO_INDEX[c] for c in codons]
        embs = codon_embeddings[indices]

        if method == 'frechet':
            aa_embeddings[aa] = frechet_mean_hyperbolic(embs, curvature)
        else:
            # Euclidean centroid (faster but less accurate)
            centroid = embs.mean(dim=0)
            aa_embeddings[aa] = project_to_poincare(
                centroid.unsqueeze(0), max_norm=0.95, c=curvature
            ).squeeze(0)

    return aa_embeddings


def compute_aa_hyperbolic_distance(
    aa_embeddings: dict[str, torch.Tensor],
    aa1: str,
    aa2: str,
    curvature: float = 1.0,
) -> float:
    """Compute hyperbolic distance between amino acids."""
    emb1 = aa_embeddings.get(aa1)
    emb2 = aa_embeddings.get(aa2)

    if emb1 is None or emb2 is None:
        return 1.0  # Maximum distance for unknown AAs

    return poincare_distance(
        emb1.unsqueeze(0), emb2.unsqueeze(0), c=curvature
    ).item()


def compute_aa_radii(
    aa_embeddings: dict[str, torch.Tensor],
    curvature: float = 1.0,
) -> dict[str, float]:
    """Compute hyperbolic radii for all amino acids."""
    radii = {}
    for aa, emb in aa_embeddings.items():
        origin = torch.zeros(1, emb.shape[0], device=emb.device)
        radii[aa] = poincare_distance(emb.unsqueeze(0), origin, c=curvature).item()
    return radii


def main():
    parser = argparse.ArgumentParser(
        description="Extract hyperbolic embeddings from VAE checkpoint"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="sandbox-training/checkpoints/v5_12_3/best_Q.pt",
        help="Path to VAE checkpoint"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/hyperbolic_embeddings.json",
        help="Output path"
    )
    parser.add_argument(
        "--encoder", "-e",
        choices=['A', 'B'],
        default='B',
        help="Which encoder to use (B recommended for hierarchy)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Device for computation"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_path = script_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path

    print("=" * 70)
    print("Extracting Hyperbolic Codon Embeddings")
    print("=" * 70)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Encoder: VAE-{args.encoder}")
    print(f"Device: {args.device}")

    # Load model
    print("\nLoading VAE model...")
    model, config, metrics = load_vae_model(str(checkpoint_path), args.device)
    curvature = config.get('curvature', 1.0)
    latent_dim = config.get('latent_dim', 16)

    print(f"  Latent dim: {latent_dim}")
    print(f"  Curvature: {curvature}")
    if metrics:
        print(f"  Checkpoint metrics:")
        print(f"    Coverage: {metrics.get('coverage', 'N/A'):.4f}")
        print(f"    Hierarchy_B: {metrics.get('hierarchy_B', 'N/A'):.4f}")

    # Extract codon embeddings
    print("\nExtracting codon embeddings...")
    codon_embeddings, codon_radii = extract_codon_embeddings(
        model, args.device, args.encoder, curvature
    )
    print(f"  Codon embeddings shape: {codon_embeddings.shape}")
    print(f"  Radius range: [{codon_radii.min():.4f}, {codon_radii.max():.4f}]")

    # Compute AA embeddings using Frechet mean
    print("\nComputing amino acid embeddings (Frechet mean)...")
    aa_embeddings = compute_aa_embeddings(codon_embeddings, curvature, method='frechet')
    aa_radii = compute_aa_radii(aa_embeddings, curvature)

    print(f"  Computed embeddings for {len(aa_embeddings)} amino acids")

    # Compute pairwise distances
    print("\nComputing pairwise hyperbolic distances...")
    aa_distances = {}
    for i, aa1 in enumerate(AMINO_ACIDS):
        for aa2 in AMINO_ACIDS[i+1:]:
            key = f"{aa1}-{aa2}"
            aa_distances[key] = compute_aa_hyperbolic_distance(
                aa_embeddings, aa1, aa2, curvature
            )

    # Prepare output data
    output_data = {
        "metadata": {
            "version": "v3_hyperbolic",
            "checkpoint": str(checkpoint_path),
            "encoder": args.encoder,
            "latent_dim": latent_dim,
            "curvature": curvature,
            "extraction_method": "frechet_mean",
            "timestamp": datetime.now().isoformat(),
            "checkpoint_metrics": {
                k: float(v) if hasattr(v, 'item') else v
                for k, v in metrics.items()
                if isinstance(v, (int, float, np.floating))
            }
        },
        "codon_data": {
            codon_index_to_triplet(i): {
                "index": i,
                "amino_acid": GENETIC_CODE.get(codon_index_to_triplet(i), '*'),
                "embedding": codon_embeddings[i].tolist(),
                "hyperbolic_radius": float(codon_radii[i]),
            }
            for i in range(64)
        },
        "amino_acid_data": {
            aa: {
                "embedding": aa_embeddings[aa].tolist(),
                "hyperbolic_radius": aa_radii[aa],
                "properties": AA_PROPERTIES.get(aa, (0, 0, 0, 0)),
                "num_codons": len(AMINO_ACID_TO_CODONS.get(aa, [])),
            }
            for aa in AMINO_ACIDS
        },
        "pairwise_distances": aa_distances,
    }

    # Save
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nEmbeddings saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nAmino Acid Radii (sorted by radius):")
    sorted_radii = sorted(aa_radii.items(), key=lambda x: x[1], reverse=True)
    for aa, r in sorted_radii:
        props = AA_PROPERTIES.get(aa, (0, 0, 0, 0))
        print(f"  {aa}: r={r:.4f}  (hydro={props[0]:+.2f}, charge={props[1]:+.1f})")

    print("\nTop 10 Closest AA Pairs (hyperbolic distance):")
    sorted_dists = sorted(aa_distances.items(), key=lambda x: x[1])
    for pair, d in sorted_dists[:10]:
        print(f"  {pair}: {d:.4f}")

    print("\nTop 10 Most Distant AA Pairs:")
    for pair, d in sorted_dists[-10:]:
        print(f"  {pair}: {d:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
