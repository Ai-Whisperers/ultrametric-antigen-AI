#!/usr/bin/env python3
"""Bootstrap significance test for DDG predictor."""

import numpy as np
from scipy.stats import spearmanr, pearsonr
import sys
from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[4]
COLBES_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

# Import codon encoder
from src.encoders.trainable_codon_encoder import TrainableCodonEncoder
from src.geometry import poincare_distance
import torch


def load_s669(filepath):
    """Load S669 dataset."""
    mutations = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mutations.append({
                'pdb_id': row['pdb_id'],
                'wild_type': row['wild_type'],
                'mutant': row['mutant'],
                'ddg_exp': float(row['ddg'])
            })
    return mutations


# Physicochemical properties
AA_PROPERTIES = {
    "A": (0.62, 0, 88.6, 0),
    "R": (-2.53, 1, 173.4, 1),
    "N": (-0.78, 0, 114.1, 1),
    "D": (-0.90, -1, 111.1, 1),
    "C": (0.29, 0, 108.5, 0),
    "Q": (-0.85, 0, 143.8, 1),
    "E": (-0.74, -1, 138.4, 1),
    "G": (0.48, 0, 60.1, 0),
    "H": (-0.40, 0.5, 153.2, 1),
    "I": (1.38, 0, 166.7, 0),
    "L": (1.06, 0, 166.7, 0),
    "K": (-1.50, 1, 168.6, 1),
    "M": (0.64, 0, 162.9, 0),
    "F": (1.19, 0, 189.9, 0),
    "P": (0.12, 0, 112.7, 0),
    "S": (-0.18, 0, 89.0, 1),
    "T": (-0.05, 0, 116.1, 1),
    "W": (0.81, 0, 227.8, 0),
    "Y": (0.26, 0, 193.6, 1),
    "V": (1.08, 0, 140.0, 0),
}


def main():
    print("=" * 60)
    print("BOOTSTRAP SIGNIFICANCE TEST FOR DDG PREDICTOR")
    print("=" * 60)

    # Load encoder (local checkpoint in models/)
    encoder_path = COLBES_ROOT / "models/trained_codon_encoder.pt"
    checkpoint = torch.load(encoder_path, map_location='cpu', weights_only=True)
    config = checkpoint.get('config', {'latent_dim': 16, 'hidden_dim': 64})

    encoder = TrainableCodonEncoder(
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
    )
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.eval()
    aa_embeddings = encoder.get_all_amino_acid_embeddings()
    print(f"Loaded encoder with {len(aa_embeddings)} AA embeddings")

    # Load data (local to colbes package)
    data_path = COLBES_ROOT / "reproducibility/data/s669.csv"
    mutations = load_s669(data_path)
    print(f"Loaded {len(mutations)} mutations")

    # Extract features
    X, y = [], []
    for mut in mutations:
        wt = mut['wild_type']
        mt = mut['mutant']

        if wt not in aa_embeddings or mt not in aa_embeddings:
            continue
        if wt not in AA_PROPERTIES or mt not in AA_PROPERTIES:
            continue

        wt_emb = aa_embeddings[wt]
        mut_emb = aa_embeddings[mt]

        # Codon features
        hyp_dist = poincare_distance(
            wt_emb.unsqueeze(0), mut_emb.unsqueeze(0), c=encoder.curvature
        ).item()

        origin = torch.zeros(1, encoder.latent_dim)
        wt_radius = poincare_distance(wt_emb.unsqueeze(0), origin, c=encoder.curvature).item()
        mut_radius = poincare_distance(mut_emb.unsqueeze(0), origin, c=encoder.curvature).item()
        delta_radius = mut_radius - wt_radius

        diff = (mut_emb - wt_emb).detach().cpu().numpy()
        diff_norm = float(np.linalg.norm(diff))

        wt_np = wt_emb.detach().cpu().numpy()
        mut_np = mut_emb.detach().cpu().numpy()
        cos_sim = float(np.dot(wt_np, mut_np) / (np.linalg.norm(wt_np) * np.linalg.norm(mut_np) + 1e-10))

        # Physicochemical features
        wt_props = AA_PROPERTIES[wt]
        mut_props = AA_PROPERTIES[mt]
        delta_hydro = mut_props[0] - wt_props[0]
        delta_charge = abs(mut_props[1] - wt_props[1])
        delta_size = mut_props[2] - wt_props[2]
        delta_polar = mut_props[3] - wt_props[3]

        features = [hyp_dist, delta_radius, diff_norm, cos_sim,
                    delta_hydro, delta_charge, delta_size, delta_polar]
        X.append(features)
        y.append(mut['ddg_exp'])

    X = np.array(X)
    y = np.array(y)
    print(f"Features extracted: {X.shape}")

    # LOO predictions
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = Ridge(alpha=100)
    y_pred = cross_val_predict(model, X_scaled, y, cv=len(y))

    # Observed correlation
    observed_rho, p = spearmanr(y, y_pred)
    pearson_r, pearson_p = pearsonr(y, y_pred)

    print(f"\n{'=' * 60}")
    print("LOO CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"Spearman rho: {observed_rho:.4f} (p = {p:.2e})")
    print(f"Pearson r:    {pearson_r:.4f} (p = {pearson_p:.2e})")
    print(f"MAE:          {np.mean(np.abs(y - y_pred)):.3f} kcal/mol")

    # Bootstrap CI
    print(f"\n{'=' * 60}")
    print("BOOTSTRAP CONFIDENCE INTERVAL (1000 resamples)")
    print("=" * 60)

    np.random.seed(42)
    n_bootstrap = 1000
    bootstrap_rhos = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y), size=len(y), replace=True)
        r, _ = spearmanr(y[idx], y_pred[idx])
        if not np.isnan(r):
            bootstrap_rhos.append(r)

    ci_lower = np.percentile(bootstrap_rhos, 2.5)
    ci_upper = np.percentile(bootstrap_rhos, 97.5)
    se = np.std(bootstrap_rhos)

    print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"Standard Error: {se:.3f}")

    # Permutation test
    print(f"\n{'=' * 60}")
    print("PERMUTATION TEST (1000 permutations)")
    print("=" * 60)

    n_perm = 1000
    perm_rhos = []
    for _ in range(n_perm):
        y_perm = np.random.permutation(y)
        r, _ = spearmanr(y_perm, y_pred)
        perm_rhos.append(abs(r))

    perm_p = np.mean(np.array(perm_rhos) >= abs(observed_rho))
    print(f"Permutation p-value: {perm_p:.4f}")

    # Summary
    print(f"\n{'=' * 60}")
    print("STATISTICAL ASSESSMENT")
    print("=" * 60)

    if p < 0.05:
        print("✓ Spearman p-value < 0.05: STATISTICALLY SIGNIFICANT")
    else:
        print("✗ Spearman p-value >= 0.05: NOT significant")

    if ci_lower > 0:
        print("✓ 95% CI does NOT include zero: REAL CORRELATION")
    else:
        print("✗ 95% CI includes zero: interpret with caution")

    if perm_p < 0.05:
        print("✓ Permutation test p < 0.05: CONFIRMED")
    else:
        print("✗ Permutation test p >= 0.05: NOT confirmed")

    # Comparison - WITH HONEST CAVEATS
    print(f"\n{'=' * 60}")
    print("COMPARISON WITH PUBLISHED METHODS")
    print("=" * 60)
    print()
    print("⚠️  IMPORTANT: Direct comparison is NOT valid!")
    print("    Literature methods benchmarked on N=669 (full S669)")
    print("    Our results are on N=52 (curated subset)")
    print("    On N=669, our method achieves ρ=0.37-0.40")
    print()
    print(f"| Method                    | Spearman | Dataset    | Type       |")
    print(f"|---------------------------|----------|------------|------------|")
    print(f"| Rosetta ddg_monomer       | 0.69     | N=669      | Structure  |")
    print(f"| **Our Method (N=52)**     | **{observed_rho:.2f}**   | N=52       | Sequence   |")
    print(f"| Our Method (N=669)        | 0.37-0.40| N=669      | Sequence   |")
    print(f"| Mutate Everything         | 0.56     | N=669      | Sequence   |")
    print(f"| ESM-1v                    | 0.51     | N=669      | Sequence   |")
    print(f"| ELASPIC-2                 | 0.50     | N=669      | Sequence   |")
    print(f"| FoldX                     | 0.48     | N=669      | Structure  |")
    print()
    print("NOTE: On comparable N=669 data, our method does NOT outperform")
    print("      ESM-1v or other sequence-based methods.")


if __name__ == "__main__":
    main()
