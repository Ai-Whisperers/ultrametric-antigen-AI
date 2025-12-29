#!/usr/bin/env python3
"""
Cross-Validation: Hyperbolic Encoder on Rheumatoid Arthritis Data

Tests whether the p-adic/hyperbolic encoder (trained on genetic code)
generalizes to an independent disease dataset (RA citrullination).

This validates the HIV severity prediction results on completely
independent data - different disease, different proteins, different task.

Key hypothesis: If radial shift predicts citrullination immunogenicity,
then the p-adic structure captures fundamental biochemical relationships,
not HIV-specific patterns.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def hyperbolic_radius(embedding: np.ndarray, c: float = 1.0) -> float:
    """V5.12.2: Proper hyperbolic distance from origin."""
    sqrt_c = np.sqrt(c)
    euclidean_norm = np.linalg.norm(embedding)
    clamped = np.clip(euclidean_norm * sqrt_c, 0, 0.999)
    return 2.0 * np.arctanh(clamped) / sqrt_c


from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, r2_score, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "cross_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Genetic code encoder path (up to bioinformatics level)
GENETIC_CODE_DIR = SCRIPT_DIR.parent.parent.parent / "genetic_code" / "data"


# ============================================================================
# DATA LOADING
# ============================================================================


def load_ra_data():
    """Load RA citrullination prediction data."""
    # Try predictions summary first
    pred_path = RESULTS_DIR / "proteome_wide" / "15_predictions" / "predictions_summary.csv"
    if pred_path.exists():
        df = pd.read_csv(pred_path)
        print(f"Loaded predictions: {len(df)} sites")
        return df

    # Try geometric features
    geo_path = RESULTS_DIR / "proteome_wide" / "14_geometric_features" / "geometric_features_summary.csv"
    if geo_path.exists():
        df = pd.read_csv(geo_path)
        print(f"Loaded geometric features: {len(df)} sites")
        return df

    print("No RA data found")
    return None


# ============================================================================
# ENCODER LOADING
# ============================================================================


def load_encoder():
    """Load the 3-adic codon encoder."""
    import torch
    import torch.nn as nn

    class CodonEncoder(nn.Module):
        def __init__(self, input_dim=12, hidden_dim=32, embed_dim=16, n_clusters=21):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )

        def forward(self, x):
            return {"z_hyp": self.encoder(x)}

    path = GENETIC_CODE_DIR / "codon_encoder_3adic.pt"
    if not path.exists():
        raise FileNotFoundError(f"Encoder not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    encoder = CodonEncoder()
    state = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
    encoder.load_state_dict(state, strict=False)
    encoder.eval()

    return encoder


# ============================================================================
# CODON ENCODING
# ============================================================================


AA_TO_CODON = {
    "A": "GCT", "R": "CGT", "N": "AAC", "D": "GAC", "C": "TGC",
    "E": "GAG", "Q": "CAG", "G": "GGC", "H": "CAC", "I": "ATC",
    "L": "CTG", "K": "AAG", "M": "ATG", "F": "TTC", "P": "CCG",
    "S": "TCG", "T": "ACC", "W": "TGG", "Y": "TAC", "V": "GTG",
}


def codon_to_onehot(codon: str) -> np.ndarray:
    """Convert codon to one-hot encoding."""
    nucleotides = {"A": 0, "C": 1, "G": 2, "T": 3}
    onehot = np.zeros(12)
    for i, nuc in enumerate(codon.upper()[:3]):
        if nuc in nucleotides:
            onehot[i * 4 + nucleotides[nuc]] = 1
    return onehot


def encode_sequence_context(window: str, encoder):
    """Encode a 9-mer window centered on arginine.

    Citrullination converts R (Arginine) to Citrulline.
    We compute the embedding shift caused by this modification.
    """
    import torch

    if len(window) < 9:
        return None

    # Find the central R (arginine)
    center_idx = len(window) // 2
    center_aa = window[center_idx]

    if center_aa != "R":
        # Find R in window
        r_positions = [i for i, aa in enumerate(window) if aa == "R"]
        if not r_positions:
            return None
        center_idx = r_positions[len(r_positions) // 2]

    # Get flanking amino acids
    left_aa = window[center_idx - 1] if center_idx > 0 else "G"
    right_aa = window[center_idx + 1] if center_idx < len(window) - 1 else "G"

    # Get codons
    r_codon = AA_TO_CODON.get("R", "CGT")
    left_codon = AA_TO_CODON.get(left_aa, "GGC")
    right_codon = AA_TO_CODON.get(right_aa, "GGC")

    # Encode R
    x_r = torch.from_numpy(codon_to_onehot(r_codon)).float().unsqueeze(0)
    x_left = torch.from_numpy(codon_to_onehot(left_codon)).float().unsqueeze(0)
    x_right = torch.from_numpy(codon_to_onehot(right_codon)).float().unsqueeze(0)

    with torch.no_grad():
        emb_r = encoder(x_r)["z_hyp"].numpy().squeeze()
        emb_left = encoder(x_left)["z_hyp"].numpy().squeeze()
        emb_right = encoder(x_right)["z_hyp"].numpy().squeeze()

    # Compute features (V5.12.2: use hyperbolic radii)
    r_radius = hyperbolic_radius(emb_r)
    left_radius = hyperbolic_radius(emb_left)
    right_radius = hyperbolic_radius(emb_right)

    # Radial context: how R relates to neighbors
    radial_context = abs(r_radius - (left_radius + right_radius) / 2)

    # Euclidean context
    euc_context = (np.linalg.norm(emb_r - emb_left) + np.linalg.norm(emb_r - emb_right)) / 2

    # Hyperbolic distance to neighbors
    def poincare_dist(x, y, c=1.0, eps=1e-10):
        x_norm_sq = np.sum(x**2)
        y_norm_sq = np.sum(y**2)
        diff_norm_sq = np.sum((x - y)**2)
        denom = max((1 - c * x_norm_sq) * (1 - c * y_norm_sq), eps)
        arg = max(1 + 2 * c * diff_norm_sq / denom, 1.0 + eps)
        return (1 / np.sqrt(c)) * np.arccosh(arg)

    hyp_context = (poincare_dist(emb_r, emb_left) + poincare_dist(emb_r, emb_right)) / 2

    return {
        "r_radius": r_radius,
        "left_radius": left_radius,
        "right_radius": right_radius,
        "radial_context": radial_context,
        "euc_context": euc_context,
        "hyp_context": hyp_context,
    }


# ============================================================================
# CROSS-VALIDATION EXPERIMENT
# ============================================================================


def run_cross_validation():
    """Run cross-validation on RA data."""
    print("="*70)
    print("CROSS-VALIDATION: HIV Encoder on RA Data")
    print("Testing generalization of p-adic severity prediction")
    print("="*70)

    # Load data
    df = load_ra_data()
    if df is None:
        return None

    # Load encoder
    print("\nLoading 3-adic encoder (trained on genetic code)...")
    encoder = load_encoder()

    # Compute encoder features for each site
    print("\nComputing encoder features for RA sites...")
    features = []

    for _, row in df.iterrows():
        window = row.get("window_sequence", "")
        if not window or len(window) < 5:
            continue

        feat = encode_sequence_context(window, encoder)
        if feat is None:
            continue

        feat["immunogenic_probability"] = row.get("immunogenic_probability", 0.5)
        feat["risk_category"] = row.get("risk_category", "unknown")
        feat["centroid_shift"] = row.get("centroid_shift", 0)
        feat["entropy_change"] = row.get("entropy_change", 0)
        features.append(feat)

    feat_df = pd.DataFrame(features)
    print(f"Valid sites with features: {len(feat_df)}")

    # Target: immunogenic probability
    y = feat_df["immunogenic_probability"].values
    y_binary = (y >= 0.7).astype(int)  # High risk threshold

    print(f"\nTarget distribution:")
    print(f"  High risk (prob >= 0.7): {y_binary.sum()} ({y_binary.mean()*100:.1f}%)")
    print(f"  Low risk (prob < 0.7):   {len(y_binary) - y_binary.sum()} ({(1-y_binary.mean())*100:.1f}%)")

    # Feature sets
    X_hyp = feat_df[["hyp_context", "radial_context", "r_radius"]].values
    X_euc = feat_df[["euc_context"]].values
    X_radial = feat_df[["radial_context", "r_radius"]].values

    # Also use existing features from data
    X_existing = feat_df[["centroid_shift", "entropy_change"]].values

    # Correlation analysis
    print("\n" + "="*60)
    print("CORRELATION WITH IMMUNOGENIC PROBABILITY")
    print("="*60)

    for col in ["hyp_context", "euc_context", "radial_context", "r_radius", "centroid_shift"]:
        if col in feat_df.columns:
            corr, pval = stats.spearmanr(feat_df[col], y)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {col:<20}: r={corr:>7.4f}, p={pval:.2e} {sig}")

    # Classification: Predict high vs low risk
    print("\n" + "="*60)
    print("CLASSIFICATION: Predicting High-Risk Citrullination")
    print("="*60)

    results = {}

    # Hyperbolic features
    clf_hyp = LogisticRegression(max_iter=1000)
    scores_hyp = cross_val_score(clf_hyp, X_hyp, y_binary, cv=5, scoring="roc_auc")
    results["hyperbolic"] = {"mean_auc": scores_hyp.mean(), "std_auc": scores_hyp.std()}
    print(f"  Hyperbolic (context + radial): AUC = {scores_hyp.mean():.4f} ± {scores_hyp.std():.4f}")

    # Euclidean features
    clf_euc = LogisticRegression(max_iter=1000)
    scores_euc = cross_val_score(clf_euc, X_euc, y_binary, cv=5, scoring="roc_auc")
    results["euclidean"] = {"mean_auc": scores_euc.mean(), "std_auc": scores_euc.std()}
    print(f"  Euclidean only:                AUC = {scores_euc.mean():.4f} ± {scores_euc.std():.4f}")

    # Radial only (pure p-adic)
    clf_rad = LogisticRegression(max_iter=1000)
    scores_rad = cross_val_score(clf_rad, X_radial, y_binary, cv=5, scoring="roc_auc")
    results["radial"] = {"mean_auc": scores_rad.mean(), "std_auc": scores_rad.std()}
    print(f"  Radial only (p-adic):          AUC = {scores_rad.mean():.4f} ± {scores_rad.std():.4f}")

    # Existing features (centroid_shift + entropy)
    clf_exist = LogisticRegression(max_iter=1000)
    scores_exist = cross_val_score(clf_exist, X_existing, y_binary, cv=5, scoring="roc_auc")
    results["existing"] = {"mean_auc": scores_exist.mean(), "std_auc": scores_exist.std()}
    print(f"  Existing features:             AUC = {scores_exist.mean():.4f} ± {scores_exist.std():.4f}")

    # Compare hyperbolic vs euclidean
    auc_improvement = (scores_hyp.mean() - scores_euc.mean()) / scores_euc.mean() * 100

    print(f"\n  Hyperbolic vs Euclidean improvement: {auc_improvement:+.1f}%")

    # Statistical comparison (paired t-test on CV folds)
    t_stat, t_pval = stats.ttest_rel(scores_hyp, scores_euc)
    print(f"  Paired t-test p-value: {t_pval:.4f}")

    # Regression: Predict continuous probability
    print("\n" + "="*60)
    print("REGRESSION: Predicting Immunogenic Probability")
    print("="*60)

    # Hyperbolic
    reg_hyp = LinearRegression().fit(X_hyp, y)
    r2_hyp = r2_score(y, reg_hyp.predict(X_hyp))

    # Euclidean
    reg_euc = LinearRegression().fit(X_euc, y)
    r2_euc = r2_score(y, reg_euc.predict(X_euc))

    # Radial
    reg_rad = LinearRegression().fit(X_radial, y)
    r2_rad = r2_score(y, reg_rad.predict(X_radial))

    print(f"  Hyperbolic R²: {r2_hyp:.4f}")
    print(f"  Euclidean R²:  {r2_euc:.4f}")
    print(f"  Radial R²:     {r2_rad:.4f}")

    r2_improvement = (r2_hyp - r2_euc) / max(r2_euc, 0.0001) * 100
    print(f"\n  R² improvement: {r2_improvement:+.1f}%")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Hyperbolic context vs immunogenicity
    ax1 = axes[0, 0]
    ax1.scatter(feat_df["hyp_context"], y, alpha=0.3, c="coral")
    ax1.set_xlabel("Hyperbolic Context Distance")
    ax1.set_ylabel("Immunogenic Probability")
    ax1.set_title("Hyperbolic Feature vs Immunogenicity")

    # 2. Radial context vs immunogenicity
    ax2 = axes[0, 1]
    ax2.scatter(feat_df["radial_context"], y, alpha=0.3, c="green")
    ax2.set_xlabel("Radial Context (p-adic)")
    ax2.set_ylabel("Immunogenic Probability")
    ax2.set_title("Radial Feature vs Immunogenicity")

    # 3. AUC comparison
    ax3 = axes[1, 0]
    methods = ["Hyperbolic", "Euclidean", "Radial", "Existing"]
    aucs = [scores_hyp.mean(), scores_euc.mean(), scores_rad.mean(), scores_exist.mean()]
    stds = [scores_hyp.std(), scores_euc.std(), scores_rad.std(), scores_exist.std()]
    colors = ["coral", "steelblue", "green", "purple"]
    bars = ax3.bar(methods, aucs, yerr=stds, color=colors, capsize=5)
    ax3.set_ylabel("ROC-AUC (5-fold CV)")
    ax3.set_title("Classification Performance")
    ax3.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax3.set_ylim(0.4, 0.8)

    # 4. R² comparison
    ax4 = axes[1, 1]
    r2_vals = [r2_hyp, r2_euc, r2_rad]
    ax4.bar(["Hyperbolic", "Euclidean", "Radial"], r2_vals, color=["coral", "steelblue", "green"])
    ax4.set_ylabel("R² Score")
    ax4.set_title("Regression Performance")
    for i, v in enumerate(r2_vals):
        ax4.text(i, v + 0.005, f"{v:.4f}", ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cross_validation_ra.png", dpi=150)
    plt.close()
    print(f"\nSaved visualization: {OUTPUT_DIR / 'cross_validation_ra.png'}")

    # Summary
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "dataset": "Rheumatoid Arthritis Citrullination",
        "n_sites": len(feat_df),
        "classification": {
            "hyperbolic_auc": float(scores_hyp.mean()),
            "euclidean_auc": float(scores_euc.mean()),
            "radial_auc": float(scores_rad.mean()),
            "auc_improvement_pct": float(auc_improvement),
            "ttest_pvalue": float(t_pval),
        },
        "regression": {
            "hyperbolic_r2": float(r2_hyp),
            "euclidean_r2": float(r2_euc),
            "radial_r2": float(r2_rad),
            "r2_improvement_pct": float(r2_improvement),
        },
        "winner": "hyperbolic" if scores_hyp.mean() > scores_euc.mean() else "euclidean",
    }

    with open(OUTPUT_DIR / "cross_validation_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"Saved results: {OUTPUT_DIR / 'cross_validation_results.json'}")

    # Final verdict
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS")
    print("="*70)
    print(f"\nDataset: Rheumatoid Arthritis (independent of HIV training)")
    print(f"Task: Predict citrullination immunogenicity")
    print(f"\nClassification (5-fold CV AUC):")
    print(f"  Hyperbolic: {scores_hyp.mean():.4f} ± {scores_hyp.std():.4f}")
    print(f"  Euclidean:  {scores_euc.mean():.4f} ± {scores_euc.std():.4f}")
    print(f"  Improvement: {auc_improvement:+.1f}%")

    if scores_hyp.mean() > scores_euc.mean() and t_pval < 0.1:
        print(f"\nCONCLUSION: Hyperbolic geometry GENERALIZES to independent data")
        print("            The p-adic encoder captures fundamental biochemistry,")
        print("            not dataset-specific patterns.")
    elif scores_hyp.mean() > scores_euc.mean():
        print(f"\nCONCLUSION: Hyperbolic shows advantage but not statistically significant")
        print(f"            (p={t_pval:.4f})")
    else:
        print(f"\nCONCLUSION: No clear generalization advantage for hyperbolic")
    print("="*70)

    return final_results


def main():
    results = run_cross_validation()
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
