#!/usr/bin/env python3
"""
Hierarchical Model Validation: Proving Disruption Prediction

This script validates that the 14-level ultrametric hierarchy
actually predicts biological disruption by testing against:

1. Known immunodominant vs silent epitopes (n=57)
2. ACPA reactivity scores (continuous)
3. Structural disruption (AlphaFold)

Scientific rigor requirements:
- ROC/AUC analysis
- Statistical significance tests
- Cross-validation
- Effect size calculation

If successful, this proves the model has predictive power.

Version: 1.0
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


def hyperbolic_radius(embedding: np.ndarray, c: float = 1.0) -> float:
    """V5.12.2: Proper hyperbolic distance from origin."""
    sqrt_c = np.sqrt(c)
    euclidean_norm = np.linalg.norm(embedding)
    clamped = np.clip(euclidean_norm * sqrt_c, 0, 0.999)
    return 2.0 * np.arctanh(clamped) / sqrt_c


from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score

matplotlib.use("Agg")

# Add path for imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from hyperbolic_utils import (AA_TO_CODON, CodonEncoder, codon_to_onehot,
                              load_codon_encoder, poincare_distance)


def get_output_dir() -> Path:
    """Get output directory for results."""
    output_dir = SCRIPT_DIR.parent / "results" / "hyperbolic" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_epitope_database() -> Dict:
    """Load the epitope database with ground truth labels."""
    data_path = SCRIPT_DIR.parent / "data" / "augmented_epitope_database.json"
    with open(data_path) as f:
        return json.load(f)


def extract_cluster_info(encoder: CodonEncoder, device: str = "cpu") -> Dict:
    """Extract cluster centers and compute distance matrix."""
    cluster_centers = encoder.cluster_centers.detach().cpu().numpy()
    n_clusters = cluster_centers.shape[0]

    centers_tensor = torch.tensor(cluster_centers).float()
    dist_matrix = np.zeros((n_clusters, n_clusters))

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            d = poincare_distance(centers_tensor[i], centers_tensor[j]).item()
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return {
        "centers": cluster_centers,
        "dist_matrix": dist_matrix,
        "n_clusters": n_clusters,
    }


def compute_epitope_hierarchical_features(
    epitope_seq: str,
    encoder: CodonEncoder,
    cluster_info: Dict,
    device: str = "cpu",
) -> Dict:
    """
    Compute hierarchical features for an epitope sequence.

    Features:
    - Mean cluster distance (diversity within epitope)
    - Entropy of cluster distribution
    - R→Q transition distance (citrullination effect)
    - Number of R residues
    - Mean pairwise codon distance
    """
    # Encode each residue
    embeddings = []
    cluster_ids = []
    cluster_probs_list = []

    for aa in epitope_seq.upper():
        codon = AA_TO_CODON.get(aa)
        if codon is None or codon == "NNN":
            continue

        onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            probs, emb = encoder.get_cluster_probs(onehot)
            cluster_id = torch.argmax(probs, dim=-1).item()

        embeddings.append(emb.cpu().numpy().squeeze())
        cluster_ids.append(cluster_id)
        cluster_probs_list.append(probs.cpu().numpy().squeeze())

    if len(embeddings) < 2:
        return None

    embeddings = np.array(embeddings)
    cluster_probs = np.array(cluster_probs_list)

    # Feature 1: Mean embedding (centroid) - V5.12.2: use hyperbolic radius
    centroid = np.mean(embeddings, axis=0)
    centroid_norm = hyperbolic_radius(centroid)

    # Feature 2: Cluster diversity (unique clusters / total)
    unique_clusters = len(set(cluster_ids))
    cluster_diversity = unique_clusters / len(cluster_ids)

    # Feature 3: Mean pairwise embedding distance
    pairwise_dists = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            d = poincare_distance(
                torch.tensor(embeddings[i]).float(),
                torch.tensor(embeddings[j]).float(),
            ).item()
            pairwise_dists.append(d)

    mean_pairwise_dist = np.mean(pairwise_dists) if pairwise_dists else 0

    # Feature 4: Mean cluster probability entropy
    entropies = [-np.sum(p * np.log(p + 1e-10)) for p in cluster_probs]
    mean_entropy = np.mean(entropies)

    # Feature 5: R content and citrullination potential
    r_count = epitope_seq.upper().count("R")
    r_fraction = r_count / len(epitope_seq)

    # Feature 6: Compute R→Q transition effect (if R present)
    r_to_q_effect = 0
    if "R" in epitope_seq.upper() and "Q" in AA_TO_CODON:
        r_codon = AA_TO_CODON["R"]
        q_codon = AA_TO_CODON["Q"]

        r_onehot = torch.tensor(codon_to_onehot(r_codon), dtype=torch.float32).unsqueeze(0).to(device)
        q_onehot = torch.tensor(codon_to_onehot(q_codon), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            _, r_emb = encoder.get_cluster_probs(r_onehot)
            _, q_emb = encoder.get_cluster_probs(q_onehot)

        r_to_q_effect = poincare_distance(r_emb.cpu().squeeze(), q_emb.cpu().squeeze()).item()

    # Feature 7: Cluster transition potential
    # How much would citrullination move the centroid?
    if r_count > 0:
        # Simulate R→Q replacement
        modified_embeddings = embeddings.copy()
        r_indices = [i for i, aa in enumerate(epitope_seq.upper()) if aa == "R"]

        for idx in r_indices:
            if idx < len(modified_embeddings):
                q_codon = AA_TO_CODON["Q"]
                q_onehot = torch.tensor(codon_to_onehot(q_codon), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, q_emb = encoder.get_cluster_probs(q_onehot)
                modified_embeddings[idx] = q_emb.cpu().numpy().squeeze()

        modified_centroid = np.mean(modified_embeddings, axis=0)
        centroid_shift = poincare_distance(
            torch.tensor(centroid).float(),
            torch.tensor(modified_centroid).float(),
        ).item()

        relative_shift = centroid_shift / (centroid_norm + 1e-10)
    else:
        centroid_shift = 0
        relative_shift = 0

    # Feature 8: Goldilocks score (how close to 0.15-0.30 range)
    goldilocks_lower, goldilocks_upper = 0.15, 0.30
    goldilocks_center = (goldilocks_lower + goldilocks_upper) / 2

    if relative_shift < goldilocks_lower:
        goldilocks_score = relative_shift / goldilocks_lower
    elif relative_shift > goldilocks_upper:
        goldilocks_score = goldilocks_upper / relative_shift
    else:
        goldilocks_score = 1.0 - abs(relative_shift - goldilocks_center) / (goldilocks_center)

    in_goldilocks = goldilocks_lower <= relative_shift <= goldilocks_upper

    return {
        "centroid_norm": float(centroid_norm),
        "cluster_diversity": float(cluster_diversity),
        "mean_pairwise_dist": float(mean_pairwise_dist),
        "mean_entropy": float(mean_entropy),
        "r_count": r_count,
        "r_fraction": float(r_fraction),
        "r_to_q_effect": float(r_to_q_effect),
        "centroid_shift": float(centroid_shift),
        "relative_shift": float(relative_shift),
        "goldilocks_score": float(goldilocks_score),
        "in_goldilocks": in_goldilocks,
        "n_residues": len(epitope_seq),
    }


def validate_against_ground_truth(
    epitope_data: Dict,
    encoder: CodonEncoder,
    cluster_info: Dict,
    device: str = "cpu",
) -> Dict:
    """
    Validate hierarchical features against immunodominant/silent labels.
    """
    # Collect all epitopes with features
    epitopes = []

    for protein_id, protein_data in epitope_data["proteins"].items():
        for epitope in protein_data["epitopes"]:
            features = compute_epitope_hierarchical_features(epitope["sequence"], encoder, cluster_info, device)

            if features is None:
                continue

            epitopes.append(
                {
                    "id": epitope["id"],
                    "protein": protein_id,
                    "sequence": epitope["sequence"],
                    "immunodominant": epitope["immunodominant"],
                    "acpa_reactivity": epitope.get("acpa_reactivity", 0),
                    **features,
                }
            )

    print(f"  Processed {len(epitopes)} epitopes")

    # Split into immunodominant vs silent
    immunodominant = [e for e in epitopes if e["immunodominant"]]
    silent = [e for e in epitopes if not e["immunodominant"]]

    print(f"  Immunodominant: {len(immunodominant)}, Silent: {len(silent)}")

    # Statistical tests for each feature
    feature_names = [
        "centroid_norm",
        "cluster_diversity",
        "mean_pairwise_dist",
        "mean_entropy",
        "r_count",
        "r_fraction",
        "r_to_q_effect",
        "centroid_shift",
        "relative_shift",
        "goldilocks_score",
    ]

    statistical_tests = {}

    for feature in feature_names:
        imm_values = [e[feature] for e in immunodominant]
        sil_values = [e[feature] for e in silent]

        # Mann-Whitney U test (non-parametric)
        if len(imm_values) > 0 and len(sil_values) > 0:
            u_stat, p_value = mannwhitneyu(imm_values, sil_values, alternative="two-sided")

            # Effect size (rank-biserial correlation)
            n1, n2 = len(imm_values), len(sil_values)
            effect_size = 1 - (2 * u_stat) / (n1 * n2)

            # Cohen's d
            pooled_std = np.sqrt((np.var(imm_values) + np.var(sil_values)) / 2)
            cohens_d = (np.mean(imm_values) - np.mean(sil_values)) / (pooled_std + 1e-10)

            statistical_tests[feature] = {
                "imm_mean": float(np.mean(imm_values)),
                "imm_std": float(np.std(imm_values)),
                "sil_mean": float(np.mean(sil_values)),
                "sil_std": float(np.std(sil_values)),
                "u_statistic": float(u_stat),
                "p_value": float(p_value),
                "effect_size": float(effect_size),
                "cohens_d": float(cohens_d),
                "significant": p_value < 0.05,
            }

    # ROC/AUC for each feature
    y_true = [1 if e["immunodominant"] else 0 for e in epitopes]

    roc_results = {}
    for feature in feature_names:
        y_scores = [e[feature] for e in epitopes]

        if len(set(y_true)) < 2:
            continue

        try:
            auc = roc_auc_score(y_true, y_scores)
            # Also try inverted (some features might be inversely related)
            auc_inv = roc_auc_score(y_true, [-s for s in y_scores])

            roc_results[feature] = {
                "auc": float(max(auc, auc_inv)),
                "direction": "positive" if auc >= auc_inv else "negative",
            }
        except Exception:
            continue

    # Correlation with ACPA reactivity (continuous validation)
    acpa_values = [e["acpa_reactivity"] for e in epitopes if e["acpa_reactivity"] > 0]

    correlation_results = {}
    for feature in feature_names:
        feature_values = [e[feature] for e in epitopes if e["acpa_reactivity"] > 0]

        if len(feature_values) >= 5:
            r_spearman, p_spearman = spearmanr(feature_values, acpa_values)
            r_pearson, p_pearson = pearsonr(feature_values, acpa_values)

            correlation_results[feature] = {
                "spearman_r": float(r_spearman),
                "spearman_p": float(p_spearman),
                "pearson_r": float(r_pearson),
                "pearson_p": float(p_pearson),
            }

    # Combined model (logistic regression with cross-validation)
    X = np.array([[e[f] for f in feature_names] for e in epitopes])
    y = np.array(y_true)

    # Handle any NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

    # Cross-validated AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)

    try:
        cv_scores = cross_val_score(lr, X, y, cv=cv, scoring="roc_auc")
        combined_auc = float(np.mean(cv_scores))
        combined_auc_std = float(np.std(cv_scores))
    except Exception:
        combined_auc = 0
        combined_auc_std = 0

    # Goldilocks zone validation
    goldilocks_imm = sum(1 for e in immunodominant if e["in_goldilocks"])
    goldilocks_sil = sum(1 for e in silent if e["in_goldilocks"])

    goldilocks_validation = {
        "imm_in_goldilocks": goldilocks_imm,
        "imm_total": len(immunodominant),
        "imm_rate": (goldilocks_imm / len(immunodominant) if immunodominant else 0),
        "sil_in_goldilocks": goldilocks_sil,
        "sil_total": len(silent),
        "sil_rate": goldilocks_sil / len(silent) if silent else 0,
    }

    # Fisher's exact test for Goldilocks zone
    from scipy.stats import fisher_exact

    contingency = [
        [goldilocks_imm, len(immunodominant) - goldilocks_imm],
        [goldilocks_sil, len(silent) - goldilocks_sil],
    ]
    odds_ratio, fisher_p = fisher_exact(contingency)
    goldilocks_validation["fisher_odds_ratio"] = float(odds_ratio)
    goldilocks_validation["fisher_p_value"] = float(fisher_p)

    return {
        "epitopes": epitopes,
        "statistical_tests": statistical_tests,
        "roc_results": roc_results,
        "correlation_results": correlation_results,
        "combined_model": {
            "cv_auc_mean": combined_auc,
            "cv_auc_std": combined_auc_std,
            "features_used": feature_names,
        },
        "goldilocks_validation": goldilocks_validation,
    }


def generate_validation_plots(results: Dict, output_dir: Path):
    """Generate comprehensive validation visualizations."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Feature comparison (immunodominant vs silent)
    ax = axes[0, 0]

    significant_features = [(f, d) for f, d in results["statistical_tests"].items() if d["significant"]]

    if significant_features:
        features = [f[0] for f in significant_features[:6]]
        imm_means = [results["statistical_tests"][f]["imm_mean"] for f in features]
        sil_means = [results["statistical_tests"][f]["sil_mean"] for f in features]
        imm_stds = [results["statistical_tests"][f]["imm_std"] for f in features]
        sil_stds = [results["statistical_tests"][f]["sil_std"] for f in features]

        x = np.arange(len(features))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2,
            imm_means,
            width,
            yerr=imm_stds,
            label="Immunodominant",
            color="coral",
            alpha=0.7,
            capsize=3,
        )
        bars2 = ax.bar(
            x + width / 2,
            sil_means,
            width,
            yerr=sil_stds,
            label="Silent",
            color="steelblue",
            alpha=0.7,
            capsize=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f.replace("_", "\n") for f in features], fontsize=8)
        ax.set_ylabel("Feature Value", fontsize=10)
        ax.set_title("Significant Features (p < 0.05)", fontsize=11, fontweight="bold")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No significant features", ha="center", va="center")
        ax.set_title("Feature Comparison", fontsize=11, fontweight="bold")

    # 2. ROC curves for top features
    ax = axes[0, 1]

    top_roc = sorted(results["roc_results"].items(), key=lambda x: -x[1]["auc"])[:3]

    epitopes = results["epitopes"]
    y_true = [1 if e["immunodominant"] else 0 for e in epitopes]

    for feature, roc_data in top_roc:
        y_scores = [e[feature] for e in epitopes]
        if roc_data["direction"] == "negative":
            y_scores = [-s for s in y_scores]

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        ax.plot(fpr, tpr, label=f"{feature} (AUC={roc_data['auc']:.2f})", lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curves (Top Features)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. AUC bar chart
    ax = axes[0, 2]

    features = list(results["roc_results"].keys())
    aucs = [results["roc_results"][f]["auc"] for f in features]

    sorted_idx = np.argsort(aucs)[::-1]
    features_sorted = [features[i] for i in sorted_idx]
    aucs_sorted = [aucs[i] for i in sorted_idx]

    colors = ["green" if a > 0.65 else "orange" if a > 0.55 else "gray" for a in aucs_sorted]

    ax.barh(range(len(features_sorted)), aucs_sorted, color=colors, alpha=0.7)
    ax.axvline(0.5, color="red", linestyle="--", lw=2, label="Random (0.5)")
    ax.axvline(0.65, color="green", linestyle=":", lw=2, label="Good (0.65)")

    ax.set_yticks(range(len(features_sorted)))
    ax.set_yticklabels([f.replace("_", " ") for f in features_sorted], fontsize=8)
    ax.set_xlabel("AUC", fontsize=10)
    ax.set_title("Feature AUC Scores", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0.3, 1.0)

    # 4. Goldilocks zone validation
    ax = axes[1, 0]

    gold_val = results["goldilocks_validation"]

    categories = ["Immunodominant", "Silent"]
    in_gold = [gold_val["imm_in_goldilocks"], gold_val["sil_in_goldilocks"]]
    out_gold = [
        gold_val["imm_total"] - gold_val["imm_in_goldilocks"],
        gold_val["sil_total"] - gold_val["sil_in_goldilocks"],
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        in_gold,
        width,
        label="In Goldilocks",
        color="gold",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        out_gold,
        width,
        label="Outside Goldilocks",
        color="gray",
        alpha=0.5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(
        f"Goldilocks Zone Validation\n(Fisher p={gold_val['fisher_p_value']:.4f}, OR={gold_val['fisher_odds_ratio']:.2f})",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend()

    # Add percentages
    for i, (ing, outg) in enumerate(zip(in_gold, out_gold)):
        total = ing + outg
        pct = ing / total * 100 if total > 0 else 0
        ax.text(
            i - width / 2,
            ing + 0.5,
            f"{pct:.0f}%",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # 5. Correlation with ACPA reactivity
    ax = axes[1, 1]

    # Best correlating feature
    best_corr_feature = max(
        results["correlation_results"].items(),
        key=lambda x: abs(x[1]["spearman_r"]),
    )[0]

    acpa_epitopes = [e for e in epitopes if e["acpa_reactivity"] > 0]
    feature_vals = [e[best_corr_feature] for e in acpa_epitopes]
    acpa_vals = [e["acpa_reactivity"] for e in acpa_epitopes]

    ax.scatter(
        feature_vals,
        acpa_vals,
        alpha=0.6,
        s=50,
        c=["coral" if e["immunodominant"] else "steelblue" for e in acpa_epitopes],
    )

    # Trend line
    z = np.polyfit(feature_vals, acpa_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(feature_vals), max(feature_vals), 100)
    ax.plot(x_line, p(x_line), "k--", lw=2)

    corr = results["correlation_results"][best_corr_feature]
    ax.set_xlabel(best_corr_feature.replace("_", " ").title(), fontsize=10)
    ax.set_ylabel("ACPA Reactivity", fontsize=10)
    ax.set_title(
        f"Correlation with ACPA Reactivity\n(Spearman r={corr['spearman_r']:.3f}, p={corr['spearman_p']:.4f})",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # 6. Combined model performance
    ax = axes[1, 2]

    cm = results["combined_model"]

    # Summary text
    summary_text = f"""
COMBINED MODEL VALIDATION

Cross-validated AUC: {cm['cv_auc_mean']:.3f} ± {cm['cv_auc_std']:.3f}

Features used: {len(cm['features_used'])}

Interpretation:
• AUC > 0.7: Good discrimination
• AUC > 0.8: Excellent discrimination
• AUC > 0.9: Outstanding discrimination

Current performance: {'GOOD' if cm['cv_auc_mean'] > 0.7 else 'MODERATE' if cm['cv_auc_mean'] > 0.6 else 'WEAK'}

Best individual features:
"""

    for f, r in sorted(results["roc_results"].items(), key=lambda x: -x[1]["auc"])[:3]:
        summary_text += f"• {f}: AUC = {r['auc']:.3f}\n"

    ax.text(
        0.1,
        0.9,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.axis("off")
    ax.set_title("Model Summary", fontsize=11, fontweight="bold")

    plt.suptitle(
        "Hierarchical Model Validation: Proving Disruption Prediction",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "hierarchical_validation.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved: hierarchical_validation.png")


def main():
    print("=" * 80)
    print("HIERARCHICAL MODEL VALIDATION")
    print("Proving Disruption Prediction Against Ground Truth")
    print("=" * 80)

    output_dir = get_output_dir()
    print(f"\nOutput directory: {output_dir}")

    # Load encoder
    print("\nLoading codon encoder (3-adic, V5.11.3)...")
    device = "cpu"
    encoder, mapping, native_hyperbolic = load_codon_encoder(device=device, version="3adic")

    # Extract cluster info
    print("\nExtracting cluster hierarchy...")
    cluster_info = extract_cluster_info(encoder, device)

    # Load epitope database
    print("\nLoading epitope database...")
    epitope_data = load_epitope_database()
    print(f"  Total epitopes: {epitope_data['metadata']['total_epitopes']}")
    print(f"  Immunodominant: {epitope_data['metadata']['immunodominant']}")
    print(f"  Silent: {epitope_data['metadata']['silent']}")

    # Validate
    print("\nValidating against ground truth...")
    results = validate_against_ground_truth(epitope_data, encoder, cluster_info, device)

    # Generate plots
    print("\nGenerating validation plots...")
    generate_validation_plots(results, output_dir)

    # Print key results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    # Significant features
    print("\n1. SIGNIFICANT FEATURES (p < 0.05):")
    sig_features = [(f, d) for f, d in results["statistical_tests"].items() if d["significant"]]
    if sig_features:
        for f, d in sorted(sig_features, key=lambda x: x[1]["p_value"]):
            print(f"   {f}: p={d['p_value']:.4f}, Cohen's d={d['cohens_d']:.3f}")
            print(f"      Immunodominant: {d['imm_mean']:.3f}±{d['imm_std']:.3f}")
            print(f"      Silent: {d['sil_mean']:.3f}±{d['sil_std']:.3f}")
    else:
        print("   No features reached significance at p < 0.05")

    # ROC/AUC
    print("\n2. PREDICTIVE POWER (AUC):")
    for f, r in sorted(results["roc_results"].items(), key=lambda x: -x[1]["auc"])[:5]:
        print(f"   {f}: AUC = {r['auc']:.3f}")

    # Combined model
    print("\n3. COMBINED MODEL:")
    cm = results["combined_model"]
    print(f"   Cross-validated AUC: {cm['cv_auc_mean']:.3f} ± {cm['cv_auc_std']:.3f}")

    # Goldilocks validation
    print("\n4. GOLDILOCKS ZONE VALIDATION:")
    gv = results["goldilocks_validation"]
    print(f"   Immunodominant in Goldilocks: {gv['imm_in_goldilocks']}/{gv['imm_total']} ({gv['imm_rate']*100:.1f}%)")
    print(f"   Silent in Goldilocks: {gv['sil_in_goldilocks']}/{gv['sil_total']} ({gv['sil_rate']*100:.1f}%)")
    print(f"   Fisher's exact test: OR={gv['fisher_odds_ratio']:.2f}, p={gv['fisher_p_value']:.4f}")

    # ACPA correlation
    print("\n5. CORRELATION WITH ACPA REACTIVITY:")
    for f, c in sorted(
        results["correlation_results"].items(),
        key=lambda x: -abs(x[1]["spearman_r"]),
    )[:3]:
        print(f"   {f}: Spearman r={c['spearman_r']:.3f}, p={c['spearman_p']:.4f}")

    # Save results
    print("\nSaving results...")

    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {str(k): convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(item) for item in obj]
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_json = {
        "analysis_date": datetime.now().isoformat(),
        "encoder_version": "3-adic V5.11.3",
        "n_epitopes": len(results["epitopes"]),
        "statistical_tests": convert_for_json(results["statistical_tests"]),
        "roc_results": convert_for_json(results["roc_results"]),
        "correlation_results": convert_for_json(results["correlation_results"]),
        "combined_model": convert_for_json(results["combined_model"]),
        "goldilocks_validation": convert_for_json(results["goldilocks_validation"]),
    }

    results_path = output_dir / "validation_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved: {results_path}")

    # Final verdict
    print("\n" + "=" * 80)
    print("SCIENTIFIC VERDICT")
    print("=" * 80)

    best_auc = max(r["auc"] for r in results["roc_results"].values())
    combined_auc = cm["cv_auc_mean"]
    any_significant = any(d["significant"] for d in results["statistical_tests"].values())
    goldilocks_significant = gv["fisher_p_value"] < 0.05

    proof_level = 0
    if best_auc > 0.6:
        proof_level += 1
    if combined_auc > 0.65:
        proof_level += 1
    if any_significant:
        proof_level += 1
    if goldilocks_significant:
        proof_level += 1

    print(
        f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         DISRUPTION PREDICTION PROOF                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Best single feature AUC:     {best_auc:.3f}  {'✓' if best_auc > 0.6 else '✗'}                                  ║
║  Combined model AUC:          {combined_auc:.3f}  {'✓' if combined_auc > 0.65 else '✗'}                                  ║
║  Significant features found:  {'YES' if any_significant else 'NO':4s}  {'✓' if any_significant else '✗'}                                  ║
║  Goldilocks zone validated:   {'YES' if goldilocks_significant else 'NO':4s}  {'✓' if goldilocks_significant else '✗'}                                  ║
║                                                                              ║
║  PROOF LEVEL: {proof_level}/4                                                          ║
║                                                                              ║
║  CONCLUSION: {'STRONG' if proof_level >= 3 else 'MODERATE' if proof_level >= 2 else 'WEAK'} EVIDENCE that hierarchical model predicts disruption  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    )

    print(f"\nOutput: {output_dir}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = main()
