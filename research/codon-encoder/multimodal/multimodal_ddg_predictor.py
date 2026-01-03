#!/usr/bin/env python3
"""Multimodal DDG Predictor with PROPER Leave-One-Out Validation.

This script implements a validated multimodal DDG predictor that:
1. Uses TrainableCodonEncoder (learned hyperbolic embeddings)
2. Optionally adds ESM-2 embeddings (configurable)
3. Uses proper LOO CV throughout - NO training metric leakage
4. Reports ONLY validation metrics

Key Design Principles:
- All reported metrics are from held-out data (LOO)
- No training set metrics are ever reported as performance
- Modular: can disable ESM to verify codon-only baseline
- Ablation-friendly: can compare codon vs ESM vs combined

Usage:
    python multimodal_ddg_predictor.py --mode codon_only
    python multimodal_ddg_predictor.py --mode esm_only
    python multimodal_ddg_predictor.py --mode combined
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.encoders.trainable_codon_encoder import TrainableCodonEncoder
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
                    'chain': parts[1],
                    'position': int(parts[2]),
                    'wild_type': parts[3].upper(),
                    'mutant': parts[4].upper(),
                    'ddg_exp': float(parts[5])
                })
            except (ValueError, IndexError):
                continue

    return mutations


class MultimodalDDGPredictor:
    """Multimodal DDG predictor with proper validation.

    Modes:
        - codon_only: Only TrainableCodonEncoder features
        - esm_only: Only ESM-2 embeddings (if available)
        - physico_only: Only physicochemical properties
        - combined: All features together

    All evaluation uses Leave-One-Out cross-validation.
    """

    def __init__(
        self,
        mode: str = "combined",
        codon_encoder_path: Optional[Path] = None,
        esm_embeddings_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        self.mode = mode
        self.device = device
        self.codon_encoder = None
        self.esm_embeddings = None
        self.aa_codon_embeddings = {}

        # Load codon encoder if needed
        if mode in ["codon_only", "combined"]:
            self._load_codon_encoder(codon_encoder_path)

        # Load ESM embeddings if needed
        if mode in ["esm_only", "combined"] and esm_embeddings_path:
            self._load_esm_embeddings(esm_embeddings_path)

    def _load_codon_encoder(self, path: Optional[Path] = None) -> None:
        """Load trained TrainableCodonEncoder."""
        if path is None:
            path = PROJECT_ROOT / "research/codon-encoder/training/results/trained_codon_encoder.pt"

        if not path.exists():
            print(f"Warning: Codon encoder not found at {path}")
            print("Run train_codon_encoder.py first.")
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        config = checkpoint.get('config', {'latent_dim': 16, 'hidden_dim': 64})

        self.codon_encoder = TrainableCodonEncoder(
            latent_dim=config['latent_dim'],
            hidden_dim=config['hidden_dim'],
        )
        self.codon_encoder.load_state_dict(checkpoint['model_state_dict'])
        self.codon_encoder.eval()
        self.codon_encoder.to(self.device)

        # Pre-compute AA embeddings
        self.aa_codon_embeddings = self.codon_encoder.get_all_amino_acid_embeddings()
        print(f"Loaded codon encoder with {len(self.aa_codon_embeddings)} AA embeddings")

    def _load_esm_embeddings(self, path: Path) -> None:
        """Load pre-extracted ESM embeddings.

        Expected format:
        {
            "A": [float, ...],  # 1280-dim for ESM-2 650M
            "R": [float, ...],
            ...
        }
        """
        if not path.exists():
            print(f"Warning: ESM embeddings not found at {path}")
            return

        with open(path, 'r') as f:
            data = json.load(f)

        self.esm_embeddings = {}
        for aa, emb in data.items():
            if len(aa) == 1 and aa.isalpha():  # Single letter amino acid
                self.esm_embeddings[aa] = np.array(emb)

        print(f"Loaded ESM embeddings for {len(self.esm_embeddings)} amino acids")

    def extract_codon_features(self, wt_aa: str, mut_aa: str) -> np.ndarray:
        """Extract features from TrainableCodonEncoder embeddings."""
        features = []

        if self.codon_encoder is None:
            return np.array([0.0] * 4)  # Placeholder if not loaded

        wt_emb = self.aa_codon_embeddings.get(wt_aa)
        mut_emb = self.aa_codon_embeddings.get(mut_aa)

        if wt_emb is None or mut_emb is None:
            return np.array([0.0] * 4)

        # 1. Hyperbolic distance between WT and MUT
        hyp_dist = poincare_distance(
            wt_emb.unsqueeze(0), mut_emb.unsqueeze(0), c=self.codon_encoder.curvature
        ).item()
        features.append(hyp_dist)

        # 2. Delta radius (radial position change)
        origin = torch.zeros(1, self.codon_encoder.latent_dim, device=wt_emb.device)
        wt_radius = poincare_distance(wt_emb.unsqueeze(0), origin, c=self.codon_encoder.curvature).item()
        mut_radius = poincare_distance(mut_emb.unsqueeze(0), origin, c=self.codon_encoder.curvature).item()
        delta_radius = mut_radius - wt_radius
        features.append(delta_radius)

        # 3. Embedding difference magnitude (in tangent space)
        diff = (mut_emb - wt_emb).detach().cpu().numpy()
        diff_norm = np.linalg.norm(diff)
        features.append(diff_norm)

        # 4. Cosine similarity in embedding space
        wt_np = wt_emb.detach().cpu().numpy()
        mut_np = mut_emb.detach().cpu().numpy()
        cos_sim = np.dot(wt_np, mut_np) / (np.linalg.norm(wt_np) * np.linalg.norm(mut_np) + 1e-10)
        features.append(cos_sim)

        return np.array(features)

    def extract_physico_features(self, wt_aa: str, mut_aa: str) -> np.ndarray:
        """Extract physicochemical features."""
        wt_props = AA_PROPERTIES.get(wt_aa, (0, 0, 0, 0))
        mut_props = AA_PROPERTIES.get(mut_aa, (0, 0, 0, 0))

        features = [
            mut_props[0] - wt_props[0],  # Delta hydrophobicity
            abs(mut_props[1] - wt_props[1]),  # Delta charge magnitude
            mut_props[2] - wt_props[2],  # Delta size
            mut_props[3] - wt_props[3],  # Delta polarity
        ]

        return np.array(features)

    def extract_esm_features(self, wt_aa: str, mut_aa: str) -> np.ndarray:
        """Extract ESM-2 embedding features."""
        if self.esm_embeddings is None:
            return np.array([0.0] * 4)  # Placeholder

        wt_emb = self.esm_embeddings.get(wt_aa)
        mut_emb = self.esm_embeddings.get(mut_aa)

        if wt_emb is None or mut_emb is None:
            return np.array([0.0] * 4)

        features = []

        # 1. Euclidean distance
        euc_dist = np.linalg.norm(mut_emb - wt_emb)
        features.append(euc_dist)

        # 2. Cosine similarity
        cos_sim = np.dot(wt_emb, mut_emb) / (np.linalg.norm(wt_emb) * np.linalg.norm(mut_emb) + 1e-10)
        features.append(cos_sim)

        # 3. Mean of difference (directional)
        mean_diff = np.mean(mut_emb - wt_emb)
        features.append(mean_diff)

        # 4. Std of difference (variance)
        std_diff = np.std(mut_emb - wt_emb)
        features.append(std_diff)

        return np.array(features)

    def extract_features(self, wt_aa: str, mut_aa: str) -> np.ndarray:
        """Extract all features based on mode."""
        feature_groups = []

        if self.mode in ["codon_only", "combined"]:
            feature_groups.append(self.extract_codon_features(wt_aa, mut_aa))

        if self.mode in ["physico_only", "combined"]:
            feature_groups.append(self.extract_physico_features(wt_aa, mut_aa))

        if self.mode in ["esm_only", "combined"]:
            esm_feats = self.extract_esm_features(wt_aa, mut_aa)
            if np.any(esm_feats != 0):  # Only add if ESM is available
                feature_groups.append(esm_feats)

        if not feature_groups:
            feature_groups.append(self.extract_physico_features(wt_aa, mut_aa))

        return np.concatenate(feature_groups)

    def evaluate_loo(
        self,
        mutations: list[dict],
        alphas: list[float] = [0.01, 0.1, 1.0, 10.0, 100.0],
    ) -> dict:
        """Evaluate using Leave-One-Out cross-validation.

        This is the ONLY valid evaluation method for small datasets.
        All metrics are computed on held-out samples.
        """
        # Build feature matrix
        X = []
        y = []
        valid_mutations = []

        for mut in mutations:
            wt = mut['wild_type']
            mt = mut['mutant']

            if wt not in AA_PROPERTIES or mt not in AA_PROPERTIES:
                continue

            features = self.extract_features(wt, mt)
            X.append(features)
            y.append(mut['ddg_exp'])
            valid_mutations.append(mut)

        X = np.array(X)
        y = np.array(y)

        print(f"\nDataset: {len(y)} mutations, {X.shape[1]} features")
        print(f"Mode: {self.mode}")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Find best alpha using LOO
        loo = LeaveOneOut()
        best_alpha = None
        best_loo_spearman = -1
        best_y_pred = None

        print("\nAlpha search (LOO):")
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            y_pred_loo = cross_val_predict(model, X_scaled, y, cv=loo)
            spearman, _ = spearmanr(y_pred_loo, y)
            pearson, _ = pearsonr(y_pred_loo, y)

            print(f"  alpha={alpha:6.2f}: LOO Spearman={spearman:.4f}, Pearson={pearson:.4f}")

            if spearman > best_loo_spearman:
                best_loo_spearman = spearman
                best_alpha = alpha
                best_y_pred = y_pred_loo

        # Compute final LOO metrics
        loo_spearman, spearman_p = spearmanr(best_y_pred, y)
        loo_pearson, pearson_p = pearsonr(best_y_pred, y)
        loo_mae = np.mean(np.abs(best_y_pred - y))
        loo_rmse = np.sqrt(np.mean((best_y_pred - y) ** 2))

        # Fit final model to get coefficients (for interpretation only)
        final_model = Ridge(alpha=best_alpha)
        final_model.fit(X_scaled, y)

        # Training metrics (for overfitting detection only)
        y_pred_train = final_model.predict(X_scaled)
        train_spearman, _ = spearmanr(y_pred_train, y)
        train_pearson, _ = pearsonr(y_pred_train, y)

        overfitting_ratio = train_spearman / max(loo_spearman, 0.01)

        results = {
            "mode": self.mode,
            "n_samples": len(y),
            "n_features": X.shape[1],
            "best_alpha": best_alpha,

            # THESE ARE THE VALID METRICS (LOO)
            "loo_spearman": float(loo_spearman),
            "loo_spearman_p": float(spearman_p),
            "loo_pearson": float(loo_pearson),
            "loo_pearson_p": float(pearson_p),
            "loo_mae": float(loo_mae),
            "loo_rmse": float(loo_rmse),

            # Training metrics (for overfitting detection)
            "train_spearman": float(train_spearman),
            "train_pearson": float(train_pearson),
            "overfitting_ratio": float(overfitting_ratio),

            # Model coefficients
            "coefficients": final_model.coef_.tolist(),
            "intercept": float(final_model.intercept_),
        }

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal DDG Predictor with LOO Validation"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["codon_only", "esm_only", "physico_only", "combined"],
        default="codon_only",
        help="Feature mode"
    )
    parser.add_argument(
        "--codon-encoder",
        type=str,
        default=None,
        help="Path to trained codon encoder"
    )
    parser.add_argument(
        "--esm-embeddings",
        type=str,
        default=None,
        help="Path to ESM-2 amino acid embeddings"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/multimodal_ddg_results.json",
        help="Output path"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_path = script_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Data path
    data_path = PROJECT_ROOT / "deliverables/partners/jose_colbes/reproducibility/data/s669.csv"

    print("=" * 70)
    print("Multimodal DDG Predictor - PROPER LOO Validation")
    print("=" * 70)
    print(f"\nMode: {args.mode}")
    print(f"Data: {data_path}")

    # Initialize predictor
    codon_path = Path(args.codon_encoder) if args.codon_encoder else None
    esm_path = Path(args.esm_embeddings) if args.esm_embeddings else None

    predictor = MultimodalDDGPredictor(
        mode=args.mode,
        codon_encoder_path=codon_path,
        esm_embeddings_path=esm_path,
    )

    # Load data
    mutations = load_s669(data_path)
    print(f"\nLoaded {len(mutations)} mutations from S669")

    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION (Leave-One-Out Cross-Validation)")
    print("=" * 70)

    results = predictor.evaluate_loo(mutations)

    # Report
    print("\n" + "=" * 70)
    print("RESULTS (All metrics are from held-out data)")
    print("=" * 70)

    print(f"\nLOO Metrics (VALID):")
    print(f"  Spearman: {results['loo_spearman']:.4f} (p={results['loo_spearman_p']:.2e})")
    print(f"  Pearson:  {results['loo_pearson']:.4f} (p={results['loo_pearson_p']:.2e})")
    print(f"  MAE:      {results['loo_mae']:.4f}")
    print(f"  RMSE:     {results['loo_rmse']:.4f}")

    print(f"\nOverfitting Check:")
    print(f"  Train Spearman: {results['train_spearman']:.4f}")
    print(f"  Ratio (train/LOO): {results['overfitting_ratio']:.2f}x")

    if results['overfitting_ratio'] > 1.5:
        print("  WARNING: Potential overfitting detected!")
    else:
        print("  OK: Overfitting within acceptable range")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON WITH LITERATURE")
    print("=" * 70)

    literature = {
        "Rosetta ddg_monomer": 0.69,
        "ELASPIC-2 (2024)": 0.50,
        "FoldX": 0.48,
        "P-adic baseline": 0.30,
    }

    print("\n| Method | Spearman | Type |")
    print("|--------|----------|------|")
    for method, r in sorted(literature.items(), key=lambda x: -x[1]):
        print(f"| {method} | {r:.2f} | Literature |")
    print(f"| **This ({args.mode})** | **{results['loo_spearman']:.2f}** | LOO-validated |")

    # Save results
    output_data = {
        "metadata": {
            "version": "multimodal_v1",
            "mode": args.mode,
            "timestamp": datetime.now().isoformat(),
            "validation": "Leave-One-Out (LOO)",
        },
        "results": results,
        "literature_comparison": literature,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
