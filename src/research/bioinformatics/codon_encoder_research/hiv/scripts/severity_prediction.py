#!/usr/bin/env python3
"""
Mutation Severity Prediction: Hyperbolic vs Euclidean

Tests whether p-adic/hyperbolic radial distance predicts mutation severity
(fold-change in drug resistance) better than Euclidean distance.

Key insight: The p-adic valuation encodes codon structure - third position
changes (synonymous) should have small radial shifts, while first position
changes (non-synonymous, typically severe) should have large radial shifts.

This is where hyperbolic geometry should excel over Euclidean.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Path setup
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Genetic code data path
GENETIC_CODE_DIR = SCRIPT_DIR.parent.parent.parent / "genetic_code" / "data"
OUTPUT_DIR = SCRIPT_DIR.parent / "results" / "severity_prediction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CODON ENCODING
# ============================================================================

AA_TO_CODON = {
    "A": "GCT", "R": "CGG", "N": "AAC", "D": "GAC", "C": "TGC",
    "E": "GAG", "Q": "CAG", "G": "GGC", "H": "CAC", "I": "ATC",
    "L": "CTG", "K": "AAG", "M": "ATG", "F": "TTC", "P": "CCG",
    "S": "TCG", "T": "ACC", "W": "TGG", "Y": "TAC", "V": "GTG",
    "*": "TAA",
}


def codon_to_onehot(codon: str) -> np.ndarray:
    """Convert codon to one-hot encoding."""
    nucleotides = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
    onehot = np.zeros(12)
    for i, nuc in enumerate(codon.upper()[:3]):
        if nuc in nucleotides:
            onehot[i * 4 + nucleotides[nuc]] = 1
    return onehot


def parse_mutation(mutation: str):
    """Parse mutation string like 'K103N'."""
    if len(mutation) < 3:
        return None, None, None
    wt_aa = mutation[0]
    mut_aa = mutation[-1]
    try:
        position = int(mutation[1:-1])
    except ValueError:
        return None, None, None
    return wt_aa, position, mut_aa


# ============================================================================
# DISTANCE METRICS
# ============================================================================


def poincare_distance(x: np.ndarray, y: np.ndarray, c: float = 1.0, eps: float = 1e-10) -> float:
    """Poincare ball geodesic distance."""
    x_norm_sq = np.sum(x**2)
    y_norm_sq = np.sum(y**2)
    diff_norm_sq = np.sum((x - y)**2)

    denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    denom = max(denom, eps)

    arg = 1 + 2 * c * diff_norm_sq / denom
    arg = max(arg, 1.0 + eps)

    return (1 / np.sqrt(c)) * np.arccosh(arg)


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Standard Euclidean distance."""
    return np.linalg.norm(x - y)


def hyperbolic_radius(embedding: np.ndarray, c: float = 1.0) -> float:
    """Compute hyperbolic distance from origin for a Poincare ball embedding.

    V5.12.2: Use proper hyperbolic distance formula instead of Euclidean norm.

    Args:
        embedding: Array of shape (dim,) in Poincare ball
        c: Curvature parameter (default 1.0)

    Returns:
        Hyperbolic radius (scalar)
    """
    sqrt_c = np.sqrt(c)
    euclidean_norm = np.linalg.norm(embedding)
    clamped = np.clip(euclidean_norm * sqrt_c, 0, 0.999)
    return 2.0 * np.arctanh(clamped) / sqrt_c


def compute_radial_shift(x: np.ndarray, y: np.ndarray) -> float:
    """Compute absolute radial shift (key p-adic feature).

    V5.12.2: Uses hyperbolic radius for proper Poincare ball geometry.
    """
    return abs(hyperbolic_radius(x) - hyperbolic_radius(y))


# ============================================================================
# ENCODER LOADING
# ============================================================================


def load_encoder(version: str = "3adic"):
    """Load codon encoder."""
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
            self.cluster_head = nn.Linear(embed_dim, n_clusters)

        def forward(self, x):
            emb = self.encoder(x)
            return {"z_hyp": emb}

    if version == "3adic":
        path = GENETIC_CODE_DIR / "codon_encoder_3adic.pt"
    else:
        path = GENETIC_CODE_DIR / "codon_encoder_fused.pt"

    if not path.exists():
        raise FileNotFoundError(f"Encoder not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    encoder = CodonEncoder(input_dim=12, hidden_dim=32, embed_dim=16, n_clusters=21)

    state = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint))
    encoder.load_state_dict(state, strict=False)
    encoder.eval()

    return encoder


def encode_mutation(mutation: str, encoder):
    """Encode a mutation and return (wt_emb, mut_emb)."""
    import torch

    wt_aa, _, mut_aa = parse_mutation(mutation)
    if wt_aa is None or mut_aa is None:
        return None, None

    wt_codon = AA_TO_CODON.get(wt_aa)
    mut_codon = AA_TO_CODON.get(mut_aa)

    if wt_codon is None or mut_codon is None:
        return None, None

    x_wt = torch.from_numpy(codon_to_onehot(wt_codon)).float().unsqueeze(0)
    x_mut = torch.from_numpy(codon_to_onehot(mut_codon)).float().unsqueeze(0)

    with torch.no_grad():
        out_wt = encoder(x_wt)
        out_mut = encoder(x_mut)
        emb_wt = out_wt["z_hyp"].numpy().squeeze()
        emb_mut = out_mut["z_hyp"].numpy().squeeze()

    return emb_wt, emb_mut


# ============================================================================
# DATA LOADING
# ============================================================================


def load_severity_data():
    """Load mutation severity data (fold-change from cross-resistance)."""
    cross_res_path = SCRIPT_DIR.parent / "results" / "stanford_resistance" / "cross_resistance.csv"

    if not cross_res_path.exists():
        print(f"Data not found: {cross_res_path}")
        return None

    df = pd.read_csv(cross_res_path)

    # Filter for valid mutations and fold-changes
    df = df[df["mean_fc"].notna() & (df["mean_fc"] > 0)]

    return df


# ============================================================================
# SEVERITY PREDICTION EXPERIMENT
# ============================================================================


def run_severity_experiment(encoder):
    """Test if hyperbolic geometry predicts mutation severity."""
    print("="*70)
    print("SEVERITY PREDICTION: HYPERBOLIC vs EUCLIDEAN")
    print("="*70)

    # Load data
    df = load_severity_data()
    if df is None:
        return None

    print(f"\nLoaded {len(df)} mutations with fold-change data")

    # Compute features for each mutation
    results = []

    for _, row in df.iterrows():
        mutation = row["mutation"]
        fold_change = row["mean_fc"]
        n_drugs = row["n_drugs_resistant"]

        emb_wt, emb_mut = encode_mutation(mutation, encoder)
        if emb_wt is None:
            continue

        # Compute distances
        hyp_dist = poincare_distance(emb_wt, emb_mut)
        euc_dist = euclidean_distance(emb_wt, emb_mut)
        radial_shift = compute_radial_shift(emb_wt, emb_mut)

        # Radii (V5.12.2: hyperbolic distance from origin)
        wt_radius = hyperbolic_radius(emb_wt)
        mut_radius = hyperbolic_radius(emb_mut)

        results.append({
            "mutation": mutation,
            "fold_change": fold_change,
            "log_fc": np.log10(fold_change + 1),
            "n_drugs": n_drugs,
            "hyp_dist": hyp_dist,
            "euc_dist": euc_dist,
            "radial_shift": radial_shift,
            "wt_radius": wt_radius,
            "mut_radius": mut_radius,
        })

    result_df = pd.DataFrame(results)
    print(f"Valid mutations: {len(result_df)}")

    # Correlation analysis
    print("\n" + "="*60)
    print("CORRELATION WITH FOLD-CHANGE (log10)")
    print("="*60)

    features = ["hyp_dist", "euc_dist", "radial_shift", "wt_radius", "mut_radius"]
    correlations = {}

    for feat in features:
        corr, pval = stats.spearmanr(result_df[feat], result_df["log_fc"])
        correlations[feat] = {"spearman": corr, "pvalue": pval}
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {feat:<15}: r={corr:>7.4f}, p={pval:.2e} {sig}")

    # Regression analysis
    print("\n" + "="*60)
    print("LINEAR REGRESSION: R² for predicting log(fold-change)")
    print("="*60)

    X_hyp = result_df[["hyp_dist", "radial_shift"]].values
    X_euc = result_df[["euc_dist"]].values
    X_radial = result_df[["radial_shift"]].values
    y = result_df["log_fc"].values

    # Hyperbolic features (distance + radial shift)
    reg_hyp = LinearRegression().fit(X_hyp, y)
    r2_hyp = r2_score(y, reg_hyp.predict(X_hyp))
    mae_hyp = mean_absolute_error(y, reg_hyp.predict(X_hyp))

    # Euclidean features
    reg_euc = LinearRegression().fit(X_euc, y)
    r2_euc = r2_score(y, reg_euc.predict(X_euc))
    mae_euc = mean_absolute_error(y, reg_euc.predict(X_euc))

    # Radial shift only (p-adic feature)
    reg_radial = LinearRegression().fit(X_radial, y)
    r2_radial = r2_score(y, reg_radial.predict(X_radial))
    mae_radial = mean_absolute_error(y, reg_radial.predict(X_radial))

    print(f"  Hyperbolic (dist + radial): R²={r2_hyp:.4f}, MAE={mae_hyp:.4f}")
    print(f"  Euclidean only:             R²={r2_euc:.4f}, MAE={mae_euc:.4f}")
    print(f"  Radial shift only:          R²={r2_radial:.4f}, MAE={mae_radial:.4f}")

    # Statistical comparison
    r2_improvement = (r2_hyp - r2_euc) / r2_euc * 100 if r2_euc > 0 else 0

    print(f"\n  R² improvement (Hyp vs Euc): {r2_improvement:+.1f}%")

    # Binned analysis: high vs low severity
    print("\n" + "="*60)
    print("BINNED ANALYSIS: HIGH vs LOW SEVERITY")
    print("="*60)

    median_fc = result_df["log_fc"].median()
    high_sev = result_df[result_df["log_fc"] >= median_fc]
    low_sev = result_df[result_df["log_fc"] < median_fc]

    print(f"\n  High severity (n={len(high_sev)}, FC≥{10**median_fc:.1f}):")
    print(f"    Mean hyperbolic dist:  {high_sev['hyp_dist'].mean():.4f}")
    print(f"    Mean euclidean dist:   {high_sev['euc_dist'].mean():.4f}")
    print(f"    Mean radial shift:     {high_sev['radial_shift'].mean():.4f}")

    print(f"\n  Low severity (n={len(low_sev)}, FC<{10**median_fc:.1f}):")
    print(f"    Mean hyperbolic dist:  {low_sev['hyp_dist'].mean():.4f}")
    print(f"    Mean euclidean dist:   {low_sev['euc_dist'].mean():.4f}")
    print(f"    Mean radial shift:     {low_sev['radial_shift'].mean():.4f}")

    # Mann-Whitney tests
    stat_hyp, p_hyp = stats.mannwhitneyu(high_sev["hyp_dist"], low_sev["hyp_dist"])
    stat_euc, p_euc = stats.mannwhitneyu(high_sev["euc_dist"], low_sev["euc_dist"])
    stat_rad, p_rad = stats.mannwhitneyu(high_sev["radial_shift"], low_sev["radial_shift"])

    print(f"\n  Mann-Whitney p-values (high vs low):")
    print(f"    Hyperbolic:   p={p_hyp:.2e}")
    print(f"    Euclidean:    p={p_euc:.2e}")
    print(f"    Radial shift: p={p_rad:.2e}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scatter: hyperbolic distance vs fold-change
    ax1 = axes[0, 0]
    ax1.scatter(result_df["hyp_dist"], result_df["log_fc"], alpha=0.5, c="coral")
    ax1.set_xlabel("Hyperbolic Distance")
    ax1.set_ylabel("log10(Fold-Change)")
    ax1.set_title(f"Hyperbolic Distance vs Severity\nr={correlations['hyp_dist']['spearman']:.3f}")

    # Scatter: euclidean distance vs fold-change
    ax2 = axes[0, 1]
    ax2.scatter(result_df["euc_dist"], result_df["log_fc"], alpha=0.5, c="steelblue")
    ax2.set_xlabel("Euclidean Distance")
    ax2.set_ylabel("log10(Fold-Change)")
    ax2.set_title(f"Euclidean Distance vs Severity\nr={correlations['euc_dist']['spearman']:.3f}")

    # Scatter: radial shift vs fold-change
    ax3 = axes[1, 0]
    ax3.scatter(result_df["radial_shift"], result_df["log_fc"], alpha=0.5, c="green")
    ax3.set_xlabel("Radial Shift (p-adic feature)")
    ax3.set_ylabel("log10(Fold-Change)")
    ax3.set_title(f"Radial Shift vs Severity\nr={correlations['radial_shift']['spearman']:.3f}")

    # R² comparison bar chart
    ax4 = axes[1, 1]
    methods = ["Hyperbolic\n(dist+radial)", "Euclidean", "Radial Only"]
    r2_vals = [r2_hyp, r2_euc, r2_radial]
    colors = ["coral", "steelblue", "green"]
    ax4.bar(methods, r2_vals, color=colors)
    ax4.set_ylabel("R² Score")
    ax4.set_title("Severity Prediction Performance")
    ax4.set_ylim(0, max(r2_vals) * 1.2)
    for i, v in enumerate(r2_vals):
        ax4.text(i, v + 0.01, f"{v:.4f}", ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "severity_prediction.png", dpi=150)
    plt.close()
    print(f"\nSaved visualization to: {OUTPUT_DIR / 'severity_prediction.png'}")

    # Compile results
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "n_mutations": len(result_df),
        "correlations": correlations,
        "regression": {
            "hyperbolic_r2": r2_hyp,
            "euclidean_r2": r2_euc,
            "radial_r2": r2_radial,
            "hyperbolic_mae": mae_hyp,
            "euclidean_mae": mae_euc,
            "r2_improvement_pct": r2_improvement,
        },
        "binned_analysis": {
            "hyp_pvalue": p_hyp,
            "euc_pvalue": p_euc,
            "radial_pvalue": p_rad,
        },
        "winner": "hyperbolic" if r2_hyp > r2_euc else "euclidean",
    }

    with open(OUTPUT_DIR / "severity_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"Saved results to: {OUTPUT_DIR / 'severity_results.json'}")

    # Final verdict
    print("\n" + "="*70)
    if r2_hyp > r2_euc * 1.1:  # 10% improvement threshold
        print("CONCLUSION: HYPERBOLIC geometry significantly outperforms Euclidean")
        print(f"            for mutation severity prediction ({r2_improvement:+.1f}% R² improvement)")
    elif r2_hyp > r2_euc:
        print("CONCLUSION: Hyperbolic geometry shows modest advantage")
        print(f"            ({r2_improvement:+.1f}% R² improvement)")
    else:
        print("CONCLUSION: No clear advantage for hyperbolic geometry")
        print("            Euclidean baseline performs comparably or better")
    print("="*70)

    return final_results


def main():
    print("\nLoading encoder...")
    try:
        encoder = load_encoder("3adic")
    except Exception as e:
        print(f"Error loading encoder: {e}")
        return 1

    results = run_severity_experiment(encoder)

    if results is None:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
