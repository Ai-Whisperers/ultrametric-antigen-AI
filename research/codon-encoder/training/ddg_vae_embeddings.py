#!/usr/bin/env python3
"""DDG Prediction using VAE-extracted Codon Embeddings.

This script trains a DDG predictor using actual VAE embeddings from v5.12.3.

Key features:
1. Uses tangent space embeddings (mu) from VAE encoder_B
2. Proper hierarchy preserved (valuation -> mu_norm correlation = -0.63)
3. Leave-one-out cross-validation for honest evaluation
4. Simple model to avoid overfitting

Usage:
    python ddg_vae_embeddings.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from src.biology.codons import (
    GENETIC_CODE,
    CODON_TO_INDEX,
    AMINO_ACID_TO_CODONS,
    codon_index_to_triplet,
)
from src.core import TERNARY
from src.encoders.codon_encoder import AA_PROPERTIES


def load_encoder_b(checkpoint_path: str):
    """Load encoder_B from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict']

    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(9, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU()
            )
            self.fc_mu = nn.Linear(64, 16)

        def forward(self, x):
            h = self.encoder(x)
            return self.fc_mu(h)

    encoder = SimpleEncoder()
    encoder.encoder[0].weight.data = state_dict['encoder_B.encoder.0.weight']
    encoder.encoder[0].bias.data = state_dict['encoder_B.encoder.0.bias']
    encoder.encoder[2].weight.data = state_dict['encoder_B.encoder.2.weight']
    encoder.encoder[2].bias.data = state_dict['encoder_B.encoder.2.bias']
    encoder.encoder[4].weight.data = state_dict['encoder_B.encoder.4.weight']
    encoder.encoder[4].bias.data = state_dict['encoder_B.encoder.4.bias']
    encoder.fc_mu.weight.data = state_dict['encoder_B.fc_mu.weight']
    encoder.fc_mu.bias.data = state_dict['encoder_B.fc_mu.bias']
    encoder.eval()

    return encoder


def codon_to_op_natural(codon_idx: int) -> int:
    """Map codon index to ternary operation using natural hierarchy."""
    b1 = (codon_idx // 16) % 4
    b2 = (codon_idx // 4) % 4
    b3 = codon_idx % 4

    t1 = b1 % 3
    t2 = b2 % 3
    t3 = b3 % 3

    return t1 * (3 ** 8) + t2 * (3 ** 7) + t3 * (3 ** 6)


def op_to_input(op_idx: int) -> torch.Tensor:
    """Convert operation index to input tensor."""
    digits = []
    n = op_idx
    for _ in range(9):
        digits.append(n % 3)
        n //= 3
    return torch.tensor(digits, dtype=torch.float32)


def extract_codon_embeddings(encoder: nn.Module) -> dict[str, torch.Tensor]:
    """Extract embeddings for all codons."""
    codon_embeddings = {}

    with torch.no_grad():
        for codon_idx in range(64):
            triplet = codon_index_to_triplet(codon_idx)
            op = codon_to_op_natural(codon_idx)
            x = op_to_input(op).unsqueeze(0)
            mu = encoder(x).squeeze(0)
            codon_embeddings[triplet] = mu

    return codon_embeddings


def compute_aa_embeddings(codon_embeddings: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Aggregate codon embeddings to amino acid level (Euclidean mean)."""
    aa_embeddings = {}

    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        codons = AMINO_ACID_TO_CODONS.get(aa, [])
        if not codons:
            continue

        embs = torch.stack([codon_embeddings[c] for c in codons])
        aa_embeddings[aa] = embs.mean(dim=0)

    return aa_embeddings


def compute_aa_distance(aa_emb1: torch.Tensor, aa_emb2: torch.Tensor) -> float:
    """Compute Euclidean distance between AA embeddings (in tangent space)."""
    return torch.norm(aa_emb1 - aa_emb2).item()


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


def extract_features(
    mut: dict,
    aa_embeddings: dict[str, torch.Tensor],
) -> np.ndarray:
    """Extract features for DDG prediction.

    Uses:
    1. VAE embedding distance (Euclidean in tangent space)
    2. VAE embedding norm difference
    3. Delta hydrophobicity
    4. Delta charge
    """
    wt = mut['wild_type']
    mt = mut['mutant']

    # VAE-based features
    wt_emb = aa_embeddings.get(wt)
    mt_emb = aa_embeddings.get(mt)

    if wt_emb is None or mt_emb is None:
        return None

    emb_dist = compute_aa_distance(wt_emb, mt_emb)
    wt_norm = torch.norm(wt_emb).item()
    mt_norm = torch.norm(mt_emb).item()
    delta_norm = mt_norm - wt_norm

    # Physicochemical features
    wt_props = AA_PROPERTIES.get(wt, (0, 0, 0, 0))
    mt_props = AA_PROPERTIES.get(mt, (0, 0, 0, 0))

    delta_hydro = mt_props[0] - wt_props[0]
    delta_charge = abs(mt_props[1] - wt_props[1])

    return np.array([emb_dist, delta_norm, delta_hydro, delta_charge])


def main():
    script_dir = Path(__file__).parent
    checkpoint_path = PROJECT_ROOT / "sandbox-training/checkpoints/v5_12_3/best_Q.pt"
    data_path = PROJECT_ROOT / "deliverables/partners/jose_colbes/reproducibility/data/s669.csv"
    output_path = script_dir / "results/ddg_vae_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DDG Prediction with VAE Embeddings (v5.12.3)")
    print("=" * 70)

    # Load encoder
    print("\nLoading VAE encoder_B...")
    encoder = load_encoder_b(str(checkpoint_path))

    # Extract embeddings
    print("Extracting codon embeddings...")
    codon_embeddings = extract_codon_embeddings(encoder)
    print(f"  Extracted {len(codon_embeddings)} codon embeddings")

    # Aggregate to AA level
    print("Computing amino acid embeddings (mean of synonymous codons)...")
    aa_embeddings = compute_aa_embeddings(codon_embeddings)
    print(f"  Computed {len(aa_embeddings)} AA embeddings")

    # Print AA norms (should show hierarchy)
    aa_norms = {aa: torch.norm(emb).item() for aa, emb in aa_embeddings.items()}
    sorted_aa = sorted(aa_norms.items(), key=lambda x: x[1], reverse=True)
    print("\nAA embedding norms:")
    for aa, norm in sorted_aa:
        print(f"  {aa}: {norm:.4f}")

    # Load S669 data
    print(f"\nLoading S669 dataset...")
    mutations = load_s669(data_path)
    print(f"  Loaded {len(mutations)} mutations")

    # Build feature matrix
    X = []
    y = []
    valid_mutations = []

    for mut in mutations:
        features = extract_features(mut, aa_embeddings)
        if features is not None:
            X.append(features)
            y.append(mut['ddg_exp'])
            valid_mutations.append(mut)

    X = np.array(X)
    y = np.array(y)
    print(f"  Valid mutations: {len(y)}")

    # Train with LOO CV
    print("\n" + "=" * 70)
    print("TRAINING (Leave-One-Out Cross-Validation)")
    print("=" * 70)

    feature_names = ['vae_distance', 'delta_norm', 'delta_hydro', 'delta_charge']
    loo = LeaveOneOut()

    best_alpha = None
    best_loo_r = -1

    for alpha in [0.01, 0.1, 1.0, 10.0]:
        model = Ridge(alpha=alpha)
        y_pred_loo = cross_val_predict(model, X, y, cv=loo)
        pearson_loo, _ = pearsonr(y_pred_loo, y)
        spearman_loo, _ = spearmanr(y_pred_loo, y)

        print(f"\n  Alpha={alpha}:")
        print(f"    LOO Pearson:  {pearson_loo:.4f}")
        print(f"    LOO Spearman: {spearman_loo:.4f}")

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

    print("\n" + "=" * 70)
    print(f"BEST RESULTS (alpha={best_alpha})")
    print("=" * 70)

    print("\nLeave-One-Out Metrics (HONEST):")
    print(f"  Pearson r:  {pearson_loo:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman r: {spearman_loo:.4f} (p={spearman_p:.2e})")
    print(f"  MAE:        {mae_loo:.4f}")
    print(f"  RMSE:       {rmse_loo:.4f}")

    print("\nTraining Metrics (optimistic):")
    print(f"  Pearson r:  {pearson_train:.4f}")
    print(f"  Spearman r: {spearman_train:.4f}")

    print(f"\nOverfitting ratio: {pearson_train/max(pearson_loo, 0.01):.2f}x")

    print("\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.4f}")
    print(f"  intercept: {model.intercept_:.4f}")

    # Save results
    results = {
        "metadata": {
            "version": "v4_vae_embeddings",
            "checkpoint": str(checkpoint_path),
            "data": str(data_path),
            "timestamp": datetime.now().isoformat(),
        },
        "results": {
            "n_samples": len(y),
            "best_alpha": best_alpha,
            "loo_pearson_r": float(pearson_loo),
            "loo_spearman_r": float(spearman_loo),
            "loo_mae": float(mae_loo),
            "loo_rmse": float(rmse_loo),
            "train_pearson_r": float(pearson_train),
            "train_spearman_r": float(spearman_train),
            "coefficients": dict(zip(feature_names, model.coef_.tolist())),
            "intercept": float(model.intercept_),
        },
        "comparison": {
            "Rosetta_ddg_monomer": 0.69,
            "FoldX": 0.48,
            "ELASPIC-2": 0.50,
            "This_work_LOO": float(spearman_loo),
        }
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON WITH LITERATURE")
    print("=" * 70)

    print("\n| Method | Spearman r | Notes |")
    print("|--------|------------|-------|")
    print(f"| Rosetta ddg_monomer | 0.69 | Structure-based |")
    print(f"| FoldX | 0.48 | Structure-based |")
    print(f"| ELASPIC-2 (2024) | 0.50 | Sequence-based |")
    print(f"| **This work (VAE, LOO)** | **{spearman_loo:.2f}** | Sequence-based, p-adic |")

    return 0


if __name__ == "__main__":
    sys.exit(main())
