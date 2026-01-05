#!/usr/bin/env python3
"""Extract Codon Embeddings from v5.12.4 Ternary VAE.

This script extracts embeddings from the improved v5.12.4 checkpoint with:
- ImprovedEncoder/Decoder (SiLU, LayerNorm, Dropout)
- FrozenEncoder from v5.5 for coverage preservation
- Better radial hierarchy (Hier_B = -0.82, Q = 1.96)

The extracted embeddings will be used to train a generalized p-adic codon
encoder with segment-based support for long sequences.

Usage:
    python extract_v5_12_4_embeddings.py [--use-encoder B] [--output results/]
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
from src.core.ternary import TERNARY

# Standard amino acids
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def load_v5_12_4_model(checkpoint_path: str, device: str = 'cpu'):
    """Load v5.12.4 TernaryVAE from checkpoint.

    v5.12.4 uses ImprovedEncoder/Decoder with:
    - SiLU activation instead of ReLU
    - LayerNorm for stable training
    - Dropout (default 0.1) for regularization
    - logvar clamped to [-10, 2]
    - Controller and 2 projection layers
    """
    from src.models.ternary_vae import TernaryVAEV5_11_PartialFreeze

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = ckpt.get('config', {})

    # Extract model config from nested structure
    model_config = raw_config.get('model', raw_config)

    print(f"Loading v5.12.4 model with config:")
    print(f"  latent_dim: {model_config.get('latent_dim', 16)}")
    print(f"  hidden_dim: {model_config.get('hidden_dim', 64)}")
    print(f"  curvature: {model_config.get('curvature', 1.0)}")
    print(f"  use_controller: {model_config.get('use_controller', True)}")
    print(f"  projection_layers: {model_config.get('projection_layers', 2)}")

    # Try to match checkpoint architecture
    # The v5.12.4 checkpoint may have different layer counts
    state_dict = ckpt['model_state_dict']

    # Create model with default settings first
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=model_config.get('latent_dim', 16),
        hidden_dim=model_config.get('hidden_dim', 64),
        max_radius=model_config.get('max_radius', 0.95),
        curvature=model_config.get('curvature', 1.0),
        use_controller=model_config.get('use_controller', True),
        use_dual_projection=model_config.get('use_dual_projection', True),
    )

    # Filter state_dict to only include matching keys with matching shapes
    model_state = model.state_dict()
    filtered_state = {}
    skipped = []

    for key, value in state_dict.items():
        if key in model_state:
            if model_state[key].shape == value.shape:
                filtered_state[key] = value
            else:
                skipped.append(f"{key}: {value.shape} vs {model_state[key].shape}")
        else:
            skipped.append(f"{key}: not in model")

    print(f"  Loading {len(filtered_state)}/{len(state_dict)} state_dict entries")
    if skipped:
        print(f"  Skipped {len(skipped)} mismatched entries (projection layers)")

    # Load filtered weights
    model.load_state_dict(filtered_state, strict=False)

    model.eval()
    model.to(device)

    # Print checkpoint metrics
    metrics = ckpt.get('metrics', {})
    print(f"\nCheckpoint metrics:")
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            print(f"  {key}: {val:.4f}")

    return model, model_config, metrics


def extract_codon_embeddings(
    model,
    device: str = 'cpu',
    use_encoder: str = 'B',
    curvature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Extract embeddings for all 64 codons.

    Returns:
        codon_embeddings: (64, latent_dim) hyperbolic embeddings
        codon_radii: (64,) hyperbolic radii
        metadata: dict with extraction details
    """
    embeddings = []
    radii = []
    origin = None

    with torch.no_grad():
        for codon_idx in range(64):
            # Convert codon index to 9-dim ternary representation
            # TernaryVAE expects input of shape (batch, 9) with values in {-1, 0, 1}
            idx_tensor = torch.tensor([codon_idx], dtype=torch.long, device=device)
            ternary_repr = TERNARY.to_ternary(idx_tensor).float()  # (1, 9)

            # Forward through VAE
            out = model(ternary_repr, compute_control=False)

            # Get hyperbolic embedding
            if use_encoder == 'B':
                z_hyp = out['z_B_hyp'][0]
            else:
                z_hyp = out['z_A_hyp'][0]

            if origin is None:
                origin = torch.zeros_like(z_hyp)

            # Compute hyperbolic radius
            r = poincare_distance(z_hyp.unsqueeze(0), origin.unsqueeze(0), c=curvature)

            embeddings.append(z_hyp.cpu())
            radii.append(r.item())

    embeddings = torch.stack(embeddings)
    radii = torch.tensor(radii)

    # Compute hierarchy correlation
    from scipy.stats import spearmanr
    valuations = []
    for i in range(64):
        triplet = codon_index_to_triplet(i)
        aa = GENETIC_CODE.get(triplet, '*')
        # Compute simple hierarchy: degeneracy of codon
        n_codons = len(AMINO_ACID_TO_CODONS.get(aa, [triplet]))
        valuations.append(np.log2(n_codons) if n_codons > 0 else 0)

    hier_corr, _ = spearmanr(valuations, radii.numpy())

    metadata = {
        'use_encoder': use_encoder,
        'curvature': curvature,
        'latent_dim': embeddings.shape[1],
        'hierarchy_correlation': float(hier_corr),
        'mean_radius': float(radii.mean()),
        'std_radius': float(radii.std()),
        'min_radius': float(radii.min()),
        'max_radius': float(radii.max()),
    }

    return embeddings, radii, metadata


def compute_aa_embeddings(
    codon_embeddings: torch.Tensor,
    curvature: float = 1.0,
    method: str = 'frechet',
) -> dict[str, torch.Tensor]:
    """Aggregate codon embeddings to amino acid embeddings.

    Args:
        codon_embeddings: (64, latent_dim) hyperbolic embeddings
        curvature: Poincare ball curvature
        method: 'frechet' for hyperbolic mean, 'euclidean' for simple mean

    Returns:
        Dictionary mapping AA to embedding tensor
    """
    aa_embeddings = {}

    for aa in AMINO_ACIDS + ['*']:
        codons = AMINO_ACID_TO_CODONS.get(aa, [])
        if not codons:
            continue

        # Get codon indices
        indices = [CODON_TO_INDEX[c] for c in codons if c in CODON_TO_INDEX]
        if not indices:
            continue

        group_embs = codon_embeddings[indices]

        if method == 'frechet':
            # Frechet mean via log-map averaging (approximate)
            # Map to tangent space at origin, average, map back
            tangent_vecs = log_map_zero(group_embs, c=curvature)
            mean_tangent = tangent_vecs.mean(dim=0)
            from src.geometry import exp_map_zero
            aa_embedding = exp_map_zero(mean_tangent.unsqueeze(0), c=curvature).squeeze(0)
        else:
            # Simple Euclidean mean
            aa_embedding = group_embs.mean(dim=0)

        aa_embeddings[aa] = aa_embedding

    return aa_embeddings


def analyze_embeddings(
    codon_embeddings: torch.Tensor,
    radii: torch.Tensor,
    aa_embeddings: dict,
    curvature: float = 1.0,
) -> dict:
    """Analyze embedding quality.

    Returns:
        Dictionary with analysis metrics
    """
    analysis = {}

    # 1. Synonymous codon clustering
    intra_aa_dists = []
    for aa, codons in AMINO_ACID_TO_CODONS.items():
        if len(codons) < 2:
            continue
        indices = [CODON_TO_INDEX[c] for c in codons if c in CODON_TO_INDEX]
        if len(indices) < 2:
            continue

        embs = codon_embeddings[indices]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                d = poincare_distance(
                    embs[i].unsqueeze(0),
                    embs[j].unsqueeze(0),
                    c=curvature
                ).item()
                intra_aa_dists.append(d)

    analysis['mean_intra_aa_distance'] = float(np.mean(intra_aa_dists))
    analysis['std_intra_aa_distance'] = float(np.std(intra_aa_dists))

    # 2. Inter-AA distances
    inter_aa_dists = []
    aa_list = list(aa_embeddings.keys())
    for i, aa1 in enumerate(aa_list):
        for aa2 in aa_list[i + 1:]:
            if aa1 == '*' or aa2 == '*':
                continue
            d = poincare_distance(
                aa_embeddings[aa1].unsqueeze(0),
                aa_embeddings[aa2].unsqueeze(0),
                c=curvature
            ).item()
            inter_aa_dists.append(d)

    analysis['mean_inter_aa_distance'] = float(np.mean(inter_aa_dists))
    analysis['std_inter_aa_distance'] = float(np.std(inter_aa_dists))

    # 3. Separation ratio (should be > 1)
    if analysis['mean_intra_aa_distance'] > 0:
        analysis['separation_ratio'] = (
            analysis['mean_inter_aa_distance'] / analysis['mean_intra_aa_distance']
        )

    # 4. Property correlations
    from scipy.stats import spearmanr
    aa_radii = []
    aa_hydro = []
    aa_mass = []

    origin = torch.zeros(1, codon_embeddings.shape[1])
    for aa, emb in aa_embeddings.items():
        if aa == '*':
            continue
        r = poincare_distance(emb.unsqueeze(0), origin, c=curvature).item()
        props = AA_PROPERTIES.get(aa, (0, 0, 0, 0))
        aa_radii.append(r)
        aa_hydro.append(props[0])
        aa_mass.append(props[1])

    analysis['radius_hydrophobicity_corr'] = float(spearmanr(aa_radii, aa_hydro)[0])
    analysis['radius_mass_corr'] = float(spearmanr(aa_radii, aa_mass)[0])

    return analysis


def save_embeddings(
    output_dir: Path,
    codon_embeddings: torch.Tensor,
    radii: torch.Tensor,
    aa_embeddings: dict,
    metadata: dict,
    analysis: dict,
):
    """Save embeddings to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save codon embeddings as numpy
    np.save(output_dir / 'codon_embeddings.npy', codon_embeddings.numpy())
    np.save(output_dir / 'codon_radii.npy', radii.numpy())

    # Save AA embeddings
    aa_emb_dict = {aa: emb.numpy().tolist() for aa, emb in aa_embeddings.items()}
    with open(output_dir / 'aa_embeddings.json', 'w') as f:
        json.dump(aa_emb_dict, f, indent=2)

    # Save metadata and analysis
    combined = {
        'metadata': metadata,
        'analysis': analysis,
        'extraction_timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'extraction_report.json', 'w') as f:
        json.dump(combined, f, indent=2)

    print(f"\nSaved embeddings to {output_dir}")
    print(f"  - codon_embeddings.npy: shape {codon_embeddings.shape}")
    print(f"  - codon_radii.npy: shape {radii.shape}")
    print(f"  - aa_embeddings.json: {len(aa_embeddings)} amino acids")
    print(f"  - extraction_report.json: metadata and analysis")


def main():
    parser = argparse.ArgumentParser(description='Extract v5.12.4 embeddings')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='sandbox-training/checkpoints/v5_12_4/best_Q.pt',
        help='Path to v5.12.4 checkpoint',
    )
    parser.add_argument(
        '--use-encoder',
        type=str,
        default='B',
        choices=['A', 'B'],
        help='Which VAE encoder to use (A or B)',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='research/codon-encoder/results/v5_12_4_embeddings',
        help='Output directory',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    args = parser.parse_args()

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    output_dir = PROJECT_ROOT / args.output

    print("=" * 70)
    print("V5.12.4 EMBEDDING EXTRACTION")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Using encoder: VAE-{args.use_encoder}")
    print(f"Device: {args.device}")
    print()

    # Load model
    model, config, ckpt_metrics = load_v5_12_4_model(
        str(checkpoint_path),
        device=args.device
    )
    curvature = config.get('curvature', 1.0)

    # Extract codon embeddings
    print("\nExtracting codon embeddings...")
    codon_embeddings, radii, metadata = extract_codon_embeddings(
        model,
        device=args.device,
        use_encoder=args.use_encoder,
        curvature=curvature,
    )

    print(f"  Extracted {len(codon_embeddings)} codon embeddings")
    print(f"  Latent dim: {codon_embeddings.shape[1]}")
    print(f"  Hierarchy correlation: {metadata['hierarchy_correlation']:.4f}")
    print(f"  Radius range: [{metadata['min_radius']:.4f}, {metadata['max_radius']:.4f}]")

    # Compute AA embeddings
    print("\nComputing AA embeddings (Frechet mean)...")
    aa_embeddings = compute_aa_embeddings(codon_embeddings, curvature, method='frechet')
    print(f"  Computed embeddings for {len(aa_embeddings)} amino acids")

    # Analyze embeddings
    print("\nAnalyzing embedding quality...")
    analysis = analyze_embeddings(codon_embeddings, radii, aa_embeddings, curvature)

    print(f"  Intra-AA distance: {analysis['mean_intra_aa_distance']:.4f} ± {analysis['std_intra_aa_distance']:.4f}")
    print(f"  Inter-AA distance: {analysis['mean_inter_aa_distance']:.4f} ± {analysis['std_inter_aa_distance']:.4f}")
    print(f"  Separation ratio: {analysis.get('separation_ratio', 0):.4f}")
    print(f"  Radius-Hydrophobicity corr: {analysis['radius_hydrophobicity_corr']:.4f}")
    print(f"  Radius-Mass corr: {analysis['radius_mass_corr']:.4f}")

    # Save
    save_embeddings(output_dir, codon_embeddings, radii, aa_embeddings, metadata, analysis)

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
