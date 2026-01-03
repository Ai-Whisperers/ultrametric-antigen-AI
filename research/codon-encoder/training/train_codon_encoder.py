#!/usr/bin/env python3
"""Train the 12-dim Codon Encoder and Evaluate on DDG Prediction.

This script:
1. Trains the TrainableCodonEncoder on the 64 genetic codons
2. Learns hyperbolic embeddings with p-adic + AA property structure
3. Evaluates on DDG prediction using the S669 dataset
4. Compares with baseline methods

Architecture:
- Input: 12-dim one-hot (4 bases × 3 positions)
- Encoder: MLP with LayerNorm, SiLU, Dropout
- Output: 16-dim embeddings on Poincaré ball

Loss Components:
- Radial: Target radius by hierarchy level
- P-adic: Hyperbolic distances match p-adic distances
- Cohesion: Synonymous codons cluster
- Separation: Different AAs separate
- Property: AA distances correlate with property distances

Usage:
    python train_codon_encoder.py [--epochs 1000] [--lr 0.001]
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

from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from src.encoders.trainable_codon_encoder import (
    TrainableCodonEncoder,
    train_codon_encoder,
)
from src.encoders.codon_encoder import AA_PROPERTIES
from src.biology.codons import AMINO_ACID_TO_CODONS
from src.geometry import poincare_distance


def load_s669(filepath: Path) -> list[dict]:
    """Load S669 dataset."""
    mutations = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 6:
            try:
                mutations.append({
                    'pdb_id': parts[0],
                    'position': int(parts[2]),
                    'wild_type': parts[3].upper(),
                    'mutant': parts[4].upper(),
                    'ddg_exp': float(parts[5])
                })
            except (ValueError, IndexError):
                continue

    return mutations


def evaluate_ddg(
    encoder: TrainableCodonEncoder,
    mutations: list[dict],
    device: str = 'cpu',
) -> dict:
    """Evaluate DDG prediction with trained encoder.

    Features:
    1. Hyperbolic distance between WT and MUT AA
    2. Delta embedding norm
    3. Delta hydrophobicity
    4. Delta charge
    """
    encoder.eval()

    # Get AA embeddings
    aa_embeddings = encoder.get_all_amino_acid_embeddings()

    # Build features
    X = []
    y = []

    for mut in mutations:
        wt = mut['wild_type']
        mt = mut['mutant']

        if wt not in aa_embeddings or mt not in aa_embeddings:
            continue

        wt_emb = aa_embeddings[wt]
        mt_emb = aa_embeddings[mt]

        # Hyperbolic distance
        hyp_dist = poincare_distance(
            wt_emb.unsqueeze(0), mt_emb.unsqueeze(0), c=encoder.curvature
        ).item()

        # Embedding norms
        origin = torch.zeros(1, encoder.latent_dim, device=wt_emb.device)
        wt_norm = poincare_distance(wt_emb.unsqueeze(0), origin, c=encoder.curvature).item()
        mt_norm = poincare_distance(mt_emb.unsqueeze(0), origin, c=encoder.curvature).item()
        delta_norm = mt_norm - wt_norm

        # Physicochemical features
        wt_props = AA_PROPERTIES.get(wt, (0, 0, 0, 0))
        mt_props = AA_PROPERTIES.get(mt, (0, 0, 0, 0))

        delta_hydro = mt_props[0] - wt_props[0]
        delta_charge = abs(mt_props[1] - wt_props[1])

        X.append([hyp_dist, delta_norm, delta_hydro, delta_charge])
        y.append(mut['ddg_exp'])

    X = np.array(X)
    y = np.array(y)

    # Leave-one-out cross-validation
    loo = LeaveOneOut()
    best_alpha = None
    best_loo_r = -1

    for alpha in [0.01, 0.1, 1.0, 10.0]:
        model = Ridge(alpha=alpha)
        y_pred_loo = cross_val_predict(model, X, y, cv=loo)
        pearson_loo, _ = pearsonr(y_pred_loo, y)

        if pearson_loo > best_loo_r:
            best_loo_r = pearson_loo
            best_alpha = alpha
            best_y_pred = y_pred_loo

    # Final metrics
    model = Ridge(alpha=best_alpha)
    model.fit(X, y)
    y_pred_train = model.predict(X)

    pearson_loo, pearson_p = pearsonr(best_y_pred, y)
    spearman_loo, spearman_p = spearmanr(best_y_pred, y)
    mae_loo = np.mean(np.abs(best_y_pred - y))
    rmse_loo = np.sqrt(np.mean((best_y_pred - y) ** 2))

    pearson_train, _ = pearsonr(y_pred_train, y)
    spearman_train, _ = spearmanr(y_pred_train, y)

    return {
        'n_samples': len(y),
        'best_alpha': best_alpha,
        'loo_pearson_r': float(pearson_loo),
        'loo_spearman_r': float(spearman_loo),
        'loo_mae': float(mae_loo),
        'loo_rmse': float(rmse_loo),
        'train_pearson_r': float(pearson_train),
        'train_spearman_r': float(spearman_train),
        'coefficients': dict(zip(
            ['hyp_distance', 'delta_norm', 'delta_hydro', 'delta_charge'],
            model.coef_.tolist()
        )),
        'intercept': float(model.intercept_),
    }


def analyze_embeddings(encoder: TrainableCodonEncoder) -> dict:
    """Analyze learned codon embeddings."""
    encoder.eval()

    z_hyp = encoder.encode_all()
    radii = encoder.get_hyperbolic_radii(z_hyp).detach().cpu().numpy()
    hierarchies = encoder.hierarchies.cpu().numpy()

    # Correlation between hierarchy and radius
    corr_hier, p_hier = spearmanr(hierarchies, radii)

    # P-adic distance correlation
    hyp_dists = []
    padic_dists = []
    padic_matrix = encoder.padic_distances.cpu().numpy()

    for i in range(64):
        for j in range(i + 1, 64):
            d_hyp = poincare_distance(
                z_hyp[i:i+1], z_hyp[j:j+1], c=encoder.curvature
            ).item()
            hyp_dists.append(d_hyp)
            padic_dists.append(padic_matrix[i, j])

    corr_padic, p_padic = spearmanr(hyp_dists, padic_dists)

    # AA clustering quality
    aa_embeddings = encoder.get_all_amino_acid_embeddings()
    aa_radii = {}
    origin = torch.zeros(1, encoder.latent_dim)
    for aa, emb in aa_embeddings.items():
        aa_radii[aa] = poincare_distance(emb.unsqueeze(0), origin, c=encoder.curvature).item()

    return {
        'radius_range': [float(radii.min()), float(radii.max())],
        'radius_mean': float(radii.mean()),
        'radius_std': float(radii.std()),
        'hierarchy_radius_corr': float(corr_hier),
        'padic_distance_corr': float(corr_padic),
        'aa_radii': {aa: round(r, 4) for aa, r in sorted(aa_radii.items(), key=lambda x: x[1])},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train 12-dim codon encoder and evaluate on DDG"
    )
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--device", type=str, default='cpu', help="Device")
    parser.add_argument("--output", type=str, default="results/trained_codon_encoder.json")
    parser.add_argument("--save-model", type=str, default="results/trained_codon_encoder.pt")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_path = script_dir / args.output
    model_path = script_dir / args.save_model
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_path = PROJECT_ROOT / "deliverables/partners/jose_colbes/reproducibility/data/s669.csv"

    print("=" * 70)
    print("Training 12-dim Codon Encoder")
    print("=" * 70)
    print(f"\nArchitecture:")
    print(f"  Input: 12-dim one-hot (4 bases × 3 positions)")
    print(f"  Hidden: {args.hidden_dim}-dim MLP")
    print(f"  Output: {args.latent_dim}-dim Poincaré ball")
    print(f"\nTraining:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")

    # Train encoder
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    encoder = train_codon_encoder(
        epochs=args.epochs,
        lr=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        device=args.device,
        radial_weight=1.0,
        padic_weight=1.0,
        cohesion_weight=0.5,
        separation_weight=0.3,
        property_weight=0.5,
        print_every=args.epochs // 10,
    )

    # Analyze embeddings
    print("\n" + "=" * 70)
    print("EMBEDDING ANALYSIS")
    print("=" * 70)

    analysis = analyze_embeddings(encoder)
    print(f"\nRadius range: [{analysis['radius_range'][0]:.4f}, {analysis['radius_range'][1]:.4f}]")
    print(f"Radius mean: {analysis['radius_mean']:.4f}")
    print(f"Hierarchy-Radius correlation: {analysis['hierarchy_radius_corr']:.4f}")
    print(f"P-adic distance correlation: {analysis['padic_distance_corr']:.4f}")

    print("\nAA radii (sorted):")
    for aa, r in analysis['aa_radii'].items():
        print(f"  {aa}: {r:.4f}")

    # DDG evaluation
    print("\n" + "=" * 70)
    print("DDG PREDICTION EVALUATION")
    print("=" * 70)

    mutations = load_s669(data_path)
    print(f"\nLoaded {len(mutations)} mutations")

    ddg_results = evaluate_ddg(encoder, mutations, args.device)

    print(f"\nLeave-One-Out Metrics (HONEST):")
    print(f"  Pearson r:  {ddg_results['loo_pearson_r']:.4f}")
    print(f"  Spearman r: {ddg_results['loo_spearman_r']:.4f}")
    print(f"  MAE:        {ddg_results['loo_mae']:.4f}")
    print(f"  RMSE:       {ddg_results['loo_rmse']:.4f}")

    print(f"\nTraining Metrics (optimistic):")
    print(f"  Pearson r:  {ddg_results['train_pearson_r']:.4f}")
    print(f"  Spearman r: {ddg_results['train_spearman_r']:.4f}")

    print(f"\nCoefficients:")
    for name, coef in ddg_results['coefficients'].items():
        print(f"  {name}: {coef:.4f}")

    # Save results
    results = {
        'metadata': {
            'version': 'v1_12dim_trainable',
            'timestamp': datetime.now().isoformat(),
            'architecture': {
                'input_dim': 12,
                'hidden_dim': args.hidden_dim,
                'latent_dim': args.latent_dim,
            },
            'training': {
                'epochs': args.epochs,
                'lr': args.lr,
            },
        },
        'embedding_analysis': analysis,
        'ddg_evaluation': ddg_results,
        'comparison': {
            'Rosetta_ddg_monomer': 0.69,
            'FoldX': 0.48,
            'ELASPIC-2': 0.50,
            'Baseline_padic': 0.30,
            'This_work_LOO': ddg_results['loo_spearman_r'],
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Save model
    torch.save({
        'model_state_dict': encoder.state_dict(),
        'config': {
            'latent_dim': args.latent_dim,
            'hidden_dim': args.hidden_dim,
        },
        'analysis': analysis,
        'ddg_results': ddg_results,
    }, model_path)
    print(f"Model saved to: {model_path}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON WITH LITERATURE")
    print("=" * 70)

    print("\n| Method | Spearman r | Notes |")
    print("|--------|------------|-------|")
    print(f"| Rosetta ddg_monomer | 0.69 | Structure-based |")
    print(f"| FoldX | 0.48 | Structure-based |")
    print(f"| ELASPIC-2 (2024) | 0.50 | Sequence-based |")
    print(f"| Baseline (p-adic) | 0.30 | Sequence-based |")
    print(f"| **This work (12-dim)** | **{ddg_results['loo_spearman_r']:.2f}** | Trained codon encoder |")

    improvement = (ddg_results['loo_spearman_r'] - 0.30) / 0.30 * 100
    if improvement > 0:
        print(f"\nImprovement over baseline: +{improvement:.1f}%")
    else:
        print(f"\nChange from baseline: {improvement:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
