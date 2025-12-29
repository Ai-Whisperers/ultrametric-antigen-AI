#!/usr/bin/env python3
"""
Epistasis Prediction: Hyperbolic vs Euclidean Geometry

Tests whether p-adic/hyperbolic distance predicts mutation interactions
(epistasis) better than standard Euclidean distance.

Hypothesis: Mutations at similar p-adic levels (small hyperbolic distance)
show synergistic epistasis, while mutations across levels show independence
or antagonism.

Test case: Cross-resistance mutations - if a mutation confers resistance to
multiple drugs, it likely has epistatic interactions with other resistance
mutations.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Path setup
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from unified_data_loader import load_stanford_hivdb

# Genetic code data path
GENETIC_CODE_DIR = SCRIPT_DIR.parent.parent.parent / "genetic_code" / "data"
OUTPUT_DIR = SCRIPT_DIR.parent / "results" / "epistasis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CODON ENCODING
# ============================================================================

AA_TO_CODON = {
    "A": "GCT", "R": "CGG", "N": "AAC", "D": "GAC", "C": "TGC",
    "E": "GAG", "Q": "CAG", "G": "GGC", "H": "CAC", "I": "ATC",
    "L": "CTG", "K": "AAG", "M": "ATG", "F": "TTC", "P": "CCG",
    "S": "TCG", "T": "ACC", "W": "TGG", "Y": "TAC", "V": "GTG",
    "*": "TAA",  # Stop codon
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
    """Parse mutation string like 'K103N' into (wt_aa, position, mut_aa)."""
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


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Standard Euclidean distance."""
    return np.linalg.norm(x - y)


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


def compute_mutation_distance(mut1: str, mut2: str, encoder, metric: str = "hyperbolic"):
    """Compute distance between two mutations using encoder."""
    import torch

    _, _, aa1 = parse_mutation(mut1)
    _, _, aa2 = parse_mutation(mut2)

    if aa1 is None or aa2 is None:
        return None

    codon1 = AA_TO_CODON.get(aa1)
    codon2 = AA_TO_CODON.get(aa2)

    if codon1 is None or codon2 is None:
        return None

    # Encode codons
    x1 = torch.from_numpy(codon_to_onehot(codon1)).float().unsqueeze(0)
    x2 = torch.from_numpy(codon_to_onehot(codon2)).float().unsqueeze(0)

    with torch.no_grad():
        out1 = encoder(x1)
        out2 = encoder(x2)
        emb1 = out1["z_hyp"] if isinstance(out1, dict) else out1
        emb2 = out2["z_hyp"] if isinstance(out2, dict) else out2

    emb1 = emb1.numpy().squeeze()
    emb2 = emb2.numpy().squeeze()

    if metric == "hyperbolic":
        return poincare_distance(emb1, emb2)
    else:
        return euclidean_distance(emb1, emb2)


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
            logits = self.cluster_head(emb)
            return {"z_hyp": emb, "cluster_logits": logits}

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


# ============================================================================
# EPISTASIS DATA PREPARATION
# ============================================================================


def load_cross_resistance_data():
    """Load cross-resistance data as epistasis proxy."""
    cross_res_path = SCRIPT_DIR.parent / "results" / "stanford_resistance" / "cross_resistance.csv"

    if not cross_res_path.exists():
        print(f"Cross-resistance data not found at {cross_res_path}")
        return None

    df = pd.read_csv(cross_res_path)
    return df


def create_mutation_pairs(df: pd.DataFrame, min_drugs: int = 4):
    """Create mutation pairs for epistasis testing.

    Hypothesis: Mutations resistant to many drugs (n_drugs >= min_drugs)
    are likely epistatic with each other (synergistic for cross-resistance).
    """
    # High cross-resistance mutations (likely epistatic)
    high_cr = df[df["n_drugs_resistant"] >= min_drugs]["mutation"].tolist()

    # Low cross-resistance mutations (likely independent)
    low_cr = df[df["n_drugs_resistant"] <= 2]["mutation"].tolist()

    # Create pairs
    epistatic_pairs = []
    for i, m1 in enumerate(high_cr[:50]):  # Limit for efficiency
        for m2 in high_cr[i+1:50]:
            epistatic_pairs.append((m1, m2, 1))  # 1 = epistatic

    non_epistatic_pairs = []
    for m1 in high_cr[:30]:
        for m2 in low_cr[:30]:
            non_epistatic_pairs.append((m1, m2, 0))  # 0 = non-epistatic

    return epistatic_pairs, non_epistatic_pairs


# ============================================================================
# EPISTASIS PREDICTION EXPERIMENT
# ============================================================================


def run_epistasis_experiment(encoder, metric: str = "hyperbolic"):
    """Run epistasis prediction experiment."""
    print(f"\n{'='*60}")
    print(f"EPISTASIS PREDICTION EXPERIMENT ({metric.upper()})")
    print(f"{'='*60}")

    # Load data
    df = load_cross_resistance_data()
    if df is None:
        return None

    epistatic_pairs, non_epistatic_pairs = create_mutation_pairs(df)
    print(f"\nData: {len(epistatic_pairs)} epistatic pairs, {len(non_epistatic_pairs)} non-epistatic pairs")

    # Compute distances
    all_pairs = epistatic_pairs + non_epistatic_pairs
    distances = []
    labels = []
    valid_pairs = []

    for m1, m2, label in all_pairs:
        dist = compute_mutation_distance(m1, m2, encoder, metric)
        if dist is not None:
            distances.append(dist)
            labels.append(label)
            valid_pairs.append((m1, m2))

    distances = np.array(distances)
    labels = np.array(labels)

    print(f"Valid pairs: {len(distances)}")

    if len(distances) < 10:
        print("Not enough valid pairs for analysis")
        return None

    # Hypothesis: Epistatic pairs have SMALLER distance (same p-adic level)
    # So we invert for ROC: higher score = more likely epistatic = smaller distance
    scores = -distances  # Negative distance as epistasis score

    # ROC-AUC
    try:
        auc_score = roc_auc_score(labels, scores)
    except ValueError:
        auc_score = 0.5

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    # Statistical test: are distances different between groups?
    epistatic_dists = distances[labels == 1]
    non_epistatic_dists = distances[labels == 0]

    stat, pvalue = stats.mannwhitneyu(
        epistatic_dists, non_epistatic_dists, alternative="less"
    )

    results = {
        "metric": metric,
        "n_epistatic_pairs": int(sum(labels)),
        "n_non_epistatic_pairs": int(len(labels) - sum(labels)),
        "epistatic_mean_dist": float(np.mean(epistatic_dists)),
        "epistatic_std_dist": float(np.std(epistatic_dists)),
        "non_epistatic_mean_dist": float(np.mean(non_epistatic_dists)),
        "non_epistatic_std_dist": float(np.std(non_epistatic_dists)),
        "roc_auc": float(auc_score),
        "pr_auc": float(pr_auc),
        "mann_whitney_stat": float(stat),
        "mann_whitney_pvalue": float(pvalue),
        "distance_ratio": float(np.mean(non_epistatic_dists) / np.mean(epistatic_dists)) if np.mean(epistatic_dists) > 0 else 0,
    }

    print(f"\nResults:")
    print(f"  Epistatic pairs mean distance:     {results['epistatic_mean_dist']:.4f} ± {results['epistatic_std_dist']:.4f}")
    print(f"  Non-epistatic pairs mean distance: {results['non_epistatic_mean_dist']:.4f} ± {results['non_epistatic_std_dist']:.4f}")
    print(f"  Distance ratio (non-epi/epi):      {results['distance_ratio']:.4f}")
    print(f"  ROC-AUC:                           {results['roc_auc']:.4f}")
    print(f"  PR-AUC:                            {results['pr_auc']:.4f}")
    print(f"  Mann-Whitney p-value:              {results['mann_whitney_pvalue']:.2e}")

    return results, distances, labels


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("="*70)
    print("EPISTASIS PREDICTION: HYPERBOLIC vs EUCLIDEAN")
    print("Testing if p-adic geometry predicts mutation interactions")
    print("="*70)

    # Load encoder
    print("\nLoading encoder...")
    try:
        encoder = load_encoder("3adic")
    except Exception as e:
        print(f"Error loading encoder: {e}")
        return 1

    # Run experiments
    results_hyp, dist_hyp, labels = run_epistasis_experiment(encoder, "hyperbolic")
    results_euc, dist_euc, _ = run_epistasis_experiment(encoder, "euclidean")

    if results_hyp is None or results_euc is None:
        print("\nExperiment failed - insufficient data")
        return 1

    # Compare
    print("\n" + "="*70)
    print("COMPARISON: HYPERBOLIC vs EUCLIDEAN")
    print("="*70)

    print(f"\n{'Metric':<25} {'Hyperbolic':>15} {'Euclidean':>15} {'Winner':>15}")
    print("-"*70)

    metrics = [
        ("ROC-AUC", results_hyp["roc_auc"], results_euc["roc_auc"], "higher"),
        ("PR-AUC", results_hyp["pr_auc"], results_euc["pr_auc"], "higher"),
        ("Distance Ratio", results_hyp["distance_ratio"], results_euc["distance_ratio"], "higher"),
        ("p-value (log10)", np.log10(max(results_hyp["mann_whitney_pvalue"], 1e-100)),
         np.log10(max(results_euc["mann_whitney_pvalue"], 1e-100)), "lower"),
    ]

    hyperbolic_wins = 0
    for name, hyp_val, euc_val, direction in metrics:
        if direction == "higher":
            winner = "HYPERBOLIC" if hyp_val > euc_val else "EUCLIDEAN"
            if hyp_val > euc_val:
                hyperbolic_wins += 1
        else:
            winner = "HYPERBOLIC" if hyp_val < euc_val else "EUCLIDEAN"
            if hyp_val < euc_val:
                hyperbolic_wins += 1
        print(f"{name:<25} {hyp_val:>15.4f} {euc_val:>15.4f} {winner:>15}")

    print("-"*70)
    print(f"Hyperbolic wins: {hyperbolic_wins}/4 metrics")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Distribution comparison
    ax1 = axes[0]
    ax1.hist(dist_hyp[labels == 1], bins=20, alpha=0.7, label="Epistatic", color="coral")
    ax1.hist(dist_hyp[labels == 0], bins=20, alpha=0.7, label="Non-epistatic", color="steelblue")
    ax1.set_xlabel("Hyperbolic Distance")
    ax1.set_ylabel("Count")
    ax1.set_title("Hyperbolic: Distance Distribution")
    ax1.legend()

    ax2 = axes[1]
    ax2.hist(dist_euc[labels == 1], bins=20, alpha=0.7, label="Epistatic", color="coral")
    ax2.hist(dist_euc[labels == 0], bins=20, alpha=0.7, label="Non-epistatic", color="steelblue")
    ax2.set_xlabel("Euclidean Distance")
    ax2.set_ylabel("Count")
    ax2.set_title("Euclidean: Distance Distribution")
    ax2.legend()

    # ROC comparison
    ax3 = axes[2]
    metrics_compare = ["ROC-AUC", "PR-AUC", "Dist. Ratio"]
    hyp_vals = [results_hyp["roc_auc"], results_hyp["pr_auc"], min(results_hyp["distance_ratio"], 2)]
    euc_vals = [results_euc["roc_auc"], results_euc["pr_auc"], min(results_euc["distance_ratio"], 2)]

    x = np.arange(len(metrics_compare))
    width = 0.35
    ax3.bar(x - width/2, hyp_vals, width, label="Hyperbolic", color="coral")
    ax3.bar(x + width/2, euc_vals, width, label="Euclidean", color="steelblue")
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics_compare)
    ax3.set_ylabel("Score")
    ax3.set_title("Epistasis Prediction Performance")
    ax3.legend()
    ax3.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "epistasis_comparison.png", dpi=150)
    plt.close()
    print(f"\nSaved visualization to: {OUTPUT_DIR / 'epistasis_comparison.png'}")

    # Save results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "hyperbolic": results_hyp,
        "euclidean": results_euc,
        "hyperbolic_wins": hyperbolic_wins,
        "conclusion": "Hyperbolic geometry outperforms" if hyperbolic_wins >= 3 else "Mixed results",
    }

    with open(OUTPUT_DIR / "epistasis_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved results to: {OUTPUT_DIR / 'epistasis_results.json'}")

    # Final verdict
    print("\n" + "="*70)
    if hyperbolic_wins >= 3:
        print("CONCLUSION: Hyperbolic/p-adic geometry OUTPERFORMS Euclidean")
        print("for epistasis prediction in HIV drug resistance mutations.")
    elif hyperbolic_wins >= 2:
        print("CONCLUSION: Hyperbolic geometry shows ADVANTAGE over Euclidean")
        print("but results are mixed. Further validation recommended.")
    else:
        print("CONCLUSION: No clear advantage for hyperbolic geometry.")
        print("Euclidean baseline performs comparably or better.")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
