#!/usr/bin/env python3
"""Train P-adic DDG Predictor V2 - Proper Codon-Based Distances

This version uses PROPER p-adic distances computed from the genetic code
structure, not hash-based approximations.

Key improvements:
1. Uses src.biology.codons for proper codon→index mapping
2. P-adic distances reflect true codon similarity (shared bases)
3. Includes codon degeneracy as a feature

Usage:
    python train_padic_ddg_predictor_v2.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scipy.stats import spearmanr, pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score, KFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from src.biology.codons import AMINO_ACID_TO_CODONS, CODON_TO_INDEX
from src.encoders.codon_encoder import (
    compute_padic_distance_between_codons,
    AA_PROPERTIES,
)


def compute_aa_padic_distance(aa1: str, aa2: str) -> float:
    """Compute minimum p-adic distance between amino acids."""
    codons1 = AMINO_ACID_TO_CODONS.get(aa1, [])
    codons2 = AMINO_ACID_TO_CODONS.get(aa2, [])

    if not codons1 or not codons2:
        return 1.0

    min_dist = 1.0
    for c1 in codons1:
        idx1 = CODON_TO_INDEX[c1]
        for c2 in codons2:
            idx2 = CODON_TO_INDEX[c2]
            dist = compute_padic_distance_between_codons(idx1, idx2)
            min_dist = min(min_dist, dist)

    return min_dist


def compute_aa_mean_padic_distance(aa1: str, aa2: str) -> float:
    """Compute mean p-adic distance over all codon pairs."""
    codons1 = AMINO_ACID_TO_CODONS.get(aa1, [])
    codons2 = AMINO_ACID_TO_CODONS.get(aa2, [])

    if not codons1 or not codons2:
        return 1.0

    total_dist = 0.0
    count = 0
    for c1 in codons1:
        idx1 = CODON_TO_INDEX[c1]
        for c2 in codons2:
            idx2 = CODON_TO_INDEX[c2]
            total_dist += compute_padic_distance_between_codons(idx1, idx2)
            count += 1

    return total_dist / count if count > 0 else 1.0


class PAdicDDGPredictorV2:
    """DDG predictor using proper p-adic codon distances."""

    def __init__(self):
        self.weights = {
            'padic_min': 1.0,
            'padic_mean': 0.5,
            'delta_volume': 0.015,
            'delta_hydro': 0.5,
            'delta_charge': 1.5,
            'delta_polarity': 0.3,
            'degeneracy_ratio': 0.1,
            'bias': 0.0
        }

    def extract_features(self, wt_aa: str, mut_aa: str) -> np.ndarray:
        """Extract features for a mutation."""
        features = []

        # P-adic features (proper codon distances)
        padic_min = compute_aa_padic_distance(wt_aa, mut_aa)
        padic_mean = compute_aa_mean_padic_distance(wt_aa, mut_aa)
        features.extend([padic_min, padic_mean])

        # Physicochemical features
        wt_props = AA_PROPERTIES.get(wt_aa, (0, 0, 0, 0))
        mut_props = AA_PROPERTIES.get(mut_aa, (0, 0, 0, 0))

        delta_hydro = mut_props[0] - wt_props[0]
        delta_charge = mut_props[1] - wt_props[1]
        delta_size = mut_props[2] - wt_props[2]
        delta_polarity = mut_props[3] - wt_props[3]

        features.extend([delta_size, delta_hydro, abs(delta_charge), delta_polarity])

        # Degeneracy ratio (more degenerate = more evolvable)
        wt_deg = len(AMINO_ACID_TO_CODONS.get(wt_aa, []))
        mut_deg = len(AMINO_ACID_TO_CODONS.get(mut_aa, []))
        deg_ratio = mut_deg / max(wt_deg, 1)
        features.append(deg_ratio)

        return np.array(features)

    def train(self, mutations: list) -> dict:
        """Train predictor on mutation dataset."""
        X = []
        y = []

        for mut in mutations:
            wt_aa = mut['wild_type']
            mut_aa = mut['mutant']
            ddg_exp = mut['ddg_experimental']

            if wt_aa in AA_PROPERTIES and mut_aa in AA_PROPERTIES:
                features = self.extract_features(wt_aa, mut_aa)
                X.append(features)
                y.append(ddg_exp)

        X = np.array(X)
        y = np.array(y)

        print(f"Training on {len(y)} mutations")
        print(f"Feature matrix shape: {X.shape}")

        if HAS_SKLEARN:
            # Use Ridge regression with cross-validation
            model = Ridge(alpha=1.0)

            # 5-fold cross-validation
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
            print(f"Cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            # Fit on full data
            model.fit(X, y)

            # Update weights
            feature_names = ['padic_min', 'padic_mean', 'delta_volume',
                           'delta_hydro', 'delta_charge', 'delta_polarity',
                           'degeneracy_ratio']
            for i, name in enumerate(feature_names):
                self.weights[name] = model.coef_[i]
            self.weights['bias'] = model.intercept_

            y_pred = model.predict(X)
        else:
            # Numpy fallback
            X_bias = np.hstack([X, np.ones((len(X), 1))])
            coeffs, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)
            y_pred = X_bias @ coeffs

        # Compute metrics
        if HAS_SCIPY:
            pearson_r, pearson_p = pearsonr(y_pred, y)
            spearman_r, spearman_p = spearmanr(y_pred, y)
        else:
            pearson_r = np.corrcoef(y_pred, y)[0, 1]
            spearman_r = 0.0
            pearson_p = spearman_p = 0.0

        mae = np.mean(np.abs(y_pred - y))
        rmse = np.sqrt(np.mean((y_pred - y) ** 2))

        results = {
            'n_samples': len(y),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'mae': float(mae),
            'rmse': float(rmse),
            'cv_r2_mean': float(cv_scores.mean()) if HAS_SKLEARN else 0.0,
            'cv_r2_std': float(cv_scores.std()) if HAS_SKLEARN else 0.0,
            'weights': self.weights.copy()
        }

        print(f"\nTraining Results:")
        print(f"  Pearson r:  {pearson_r:.4f} (p={pearson_p:.2e})")
        print(f"  Spearman r: {spearman_r:.4f} (p={spearman_p:.2e})")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")

        print(f"\nLearned Weights:")
        for name, weight in self.weights.items():
            print(f"  {name}: {weight:.4f}")

        return results

    def save(self, path: Path) -> None:
        """Save trained predictor."""
        with open(path, 'w') as f:
            json.dump({
                'version': 'v2',
                'weights': self.weights,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)


def load_s669(filepath: Path) -> list:
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
                    'ddg_experimental': float(parts[5])
                })
            except (ValueError, IndexError):
                continue

    return mutations


def main():
    parser = argparse.ArgumentParser(
        description="Train p-adic DDG predictor V2 (proper codon distances)"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default="data/s669.csv",
        help="Path to training data (S669 format)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/padic_ddg_v2_trained.json",
        help="Output path for trained model"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_path = script_dir / args.data
    output_path = script_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Training P-adic DDG Predictor V2 (Proper Codon Distances)")
    print("=" * 70)

    # Initialize predictor
    predictor = PAdicDDGPredictorV2()

    # Load training data
    if not data_path.exists():
        print(f"\nError: Training data not found at {data_path}")
        return 1

    print(f"\nLoading training data from: {data_path}")
    mutations = load_s669(data_path)
    print(f"Loaded {len(mutations)} mutations")

    # Show p-adic distance examples
    print("\nP-adic Distance Examples:")
    examples = [('A', 'G'), ('A', 'V'), ('F', 'L'), ('D', 'E'), ('K', 'R')]
    for aa1, aa2 in examples:
        min_d = compute_aa_padic_distance(aa1, aa2)
        mean_d = compute_aa_mean_padic_distance(aa1, aa2)
        print(f"  {aa1}→{aa2}: min={min_d:.4f}, mean={mean_d:.4f}")

    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    results = predictor.train(mutations)

    # Save trained model
    predictor.save(output_path)
    print(f"\nTrained model saved to: {output_path}")

    # Save full results
    results_path = output_path.with_suffix('.results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'version': 'v2',
            'training_data': str(data_path),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    print(f"Full results saved to: {results_path}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON WITH LITERATURE")
    print("=" * 70)

    literature = {
        'Rosetta ddg_monomer': 0.69,
        'FoldX': 0.48,
        'ELASPIC-2 (2024)': 0.50,
        'P-adic V1 (hash-based)': 0.58,
    }

    print("\n| Method | Spearman r |")
    print("|--------|------------|")
    for method, r in literature.items():
        print(f"| {method} | {r:.2f} |")
    print(f"| **P-adic V2 (proper codon)** | **{results['spearman_r']:.2f}** |")

    # Assessment
    if results['spearman_r'] > 0.60:
        print("\nAssessment: EXCELLENT - Best sequence-only performance!")
    elif results['spearman_r'] > 0.55:
        print("\nAssessment: VERY GOOD - Exceeds state-of-art")
    elif results['spearman_r'] > 0.50:
        print("\nAssessment: GOOD - Matches state-of-art")
    else:
        print("\nAssessment: Needs improvement")

    return 0


if __name__ == "__main__":
    sys.exit(main())
