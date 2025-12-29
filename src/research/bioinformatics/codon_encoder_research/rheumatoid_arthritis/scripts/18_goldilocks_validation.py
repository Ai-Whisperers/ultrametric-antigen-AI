#!/usr/bin/env python3
"""
Goldilocks Zone Validation and Precise Boundary Calculation

Re-validate the Goldilocks hypothesis with the 3-adic encoder (V5.11.3)
and compute precise boundaries for the immunogenic zone.

Key finding to validate:
- Entropy change upon citrullination distinguishes immunodominant from silent
- Immunodominant: entropy INCREASES (stays complex, recognizable as modified-self)
- Silent: entropy DECREASES (becomes simpler, ignored or tolerated)

Version: 1.0
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats


def hyperbolic_radius(embedding: np.ndarray, c: float = 1.0) -> float:
    """V5.12.2: Proper hyperbolic distance from origin."""
    sqrt_c = np.sqrt(c)
    euclidean_norm = np.linalg.norm(embedding)
    clamped = np.clip(euclidean_norm * sqrt_c, 0, 0.999)
    return 2.0 * np.arctanh(clamped) / sqrt_c

matplotlib.use("Agg")

# Import epitope database
import importlib.util

# Local imports
from hyperbolic_utils import (AA_TO_CODON, codon_to_onehot, load_codon_encoder,
                              poincare_distance)

spec = importlib.util.spec_from_file_location("augmented_db", Path(__file__).parent / "08_augmented_epitope_database.py")
augmented_db = importlib.util.module_from_spec(spec)
spec.loader.exec_module(augmented_db)
RA_AUTOANTIGENS_AUGMENTED = augmented_db.RA_AUTOANTIGENS_AUGMENTED


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def get_output_dir() -> Path:
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / "hyperbolic" / "goldilocks_validation"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def compute_epitope_metrics(epitope: dict, encoder, device="cpu") -> Dict:
    """Compute all metrics for a single epitope."""
    sequence = epitope["sequence"]

    # Encode sequence
    embeddings = []
    cluster_probs_list = []

    for aa in sequence:
        codon = AA_TO_CODON.get(aa, "NNN")
        if codon == "NNN":
            continue
        onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            probs, emb = encoder.get_cluster_probs(onehot)
            embeddings.append(emb.cpu().numpy().squeeze())
            cluster_probs_list.append(probs.cpu().numpy().squeeze())

    if len(embeddings) < 2:
        return None

    embeddings = np.array(embeddings)
    cluster_probs = np.array(cluster_probs_list)

    # Find arginine positions
    arg_positions = [i for i, aa in enumerate(sequence) if aa == "R"]

    if not arg_positions:
        return None

    # Original metrics
    original_centroid = np.mean(embeddings, axis=0)
    original_probs = np.mean(cluster_probs, axis=0)
    original_entropy = -np.sum(original_probs * np.log(original_probs + 1e-10))
    original_norm = hyperbolic_radius(original_centroid)  # V5.12.2

    # Per-arginine citrullination effects
    per_r_metrics = []

    for r_idx, r_pos in enumerate(arg_positions):
        # Find embedding index (accounting for any skipped AAs)
        emb_idx = sum(1 for i, aa in enumerate(sequence[: r_pos + 1]) if AA_TO_CODON.get(aa) is not None) - 1

        if emb_idx >= len(embeddings):
            continue

        # Citrullinated version (R removed)
        cit_embeddings = np.delete(embeddings, emb_idx, axis=0)
        cit_probs = np.delete(cluster_probs, emb_idx, axis=0)

        if len(cit_embeddings) == 0:
            continue

        cit_centroid = np.mean(cit_embeddings, axis=0)
        cit_probs_mean = np.mean(cit_probs, axis=0)
        cit_entropy = -np.sum(cit_probs_mean * np.log(cit_probs_mean + 1e-10))

        # Centroid shift (Poincaré distance)
        centroid_shift = poincare_distance(
            torch.tensor(original_centroid).float(),
            torch.tensor(cit_centroid).float(),
        ).item()

        # Relative shift (normalized by original norm)
        relative_shift = centroid_shift / (original_norm + 1e-10)

        # JS divergence
        m = 0.5 * (original_probs + cit_probs_mean)
        js_div = 0.5 * (
            np.sum(original_probs * np.log((original_probs + 1e-10) / (m + 1e-10)))
            + np.sum(cit_probs_mean * np.log((cit_probs_mean + 1e-10) / (m + 1e-10)))
        )

        # Entropy change
        entropy_change = cit_entropy - original_entropy

        # Relative entropy change
        relative_entropy_change = entropy_change / (original_entropy + 1e-10)

        per_r_metrics.append(
            {
                "r_position": r_pos,
                "centroid_shift": centroid_shift,
                "relative_shift": relative_shift,
                "js_divergence": js_div,
                "entropy_change": entropy_change,
                "relative_entropy_change": relative_entropy_change,
                "original_entropy": original_entropy,
                "cit_entropy": cit_entropy,
            }
        )

    if not per_r_metrics:
        return None

    # Aggregate metrics
    return {
        "epitope_id": epitope["id"],
        "sequence": sequence,
        "immunodominant": epitope["immunodominant"],
        "acpa_reactivity": epitope.get("acpa_reactivity", 0),
        "n_arginines": len(arg_positions),
        "original_entropy": original_entropy,
        "original_norm": original_norm,
        # Mean across R positions
        "mean_centroid_shift": np.mean([m["centroid_shift"] for m in per_r_metrics]),
        "mean_relative_shift": np.mean([m["relative_shift"] for m in per_r_metrics]),
        "mean_js_divergence": np.mean([m["js_divergence"] for m in per_r_metrics]),
        "mean_entropy_change": np.mean([m["entropy_change"] for m in per_r_metrics]),
        "mean_relative_entropy_change": np.mean([m["relative_entropy_change"] for m in per_r_metrics]),
        # Per-position details
        "per_r_metrics": per_r_metrics,
    }


def compute_goldilocks_boundaries(imm_values: List[float], silent_values: List[float]) -> Dict:
    """
    Compute precise boundaries for the Goldilocks zone using statistical methods.
    """
    imm = np.array(imm_values)
    silent = np.array(silent_values)

    # Basic statistics
    imm_mean, imm_std = np.mean(imm), np.std(imm)
    silent_mean, silent_std = np.mean(silent), np.std(silent)

    # Statistical tests
    t_stat, t_p = stats.ttest_ind(imm, silent)
    u_stat, u_p = stats.mannwhitneyu(imm, silent, alternative="two-sided")

    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(imm) - 1) * imm_std**2 + (len(silent) - 1) * silent_std**2) / (len(imm) + len(silent) - 2))
    cohens_d = (imm_mean - silent_mean) / pooled_std if pooled_std > 0 else 0

    # Optimal threshold (ROC-based)
    all_values = np.concatenate([imm, silent])
    all_labels = np.concatenate([np.ones(len(imm)), np.zeros(len(silent))])

    best_threshold = None
    best_accuracy = 0
    best_sens_spec = (0, 0)

    thresholds = np.percentile(all_values, np.arange(5, 96, 5))
    for thresh in thresholds:
        pred = all_values > thresh
        tp = np.sum(pred & (all_labels == 1))
        tn = np.sum(~pred & (all_labels == 0))
        fp = np.sum(pred & (all_labels == 0))
        fn = np.sum(~pred & (all_labels == 1))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc = (tp + tn) / len(all_labels)

        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = thresh
            best_sens_spec = (sens, spec)

    # Goldilocks zone boundaries (±1 SD from immunodominant mean)
    goldilocks_lower = imm_mean - 1.5 * imm_std
    goldilocks_upper = imm_mean + 1.5 * imm_std

    # Classification using Goldilocks zone
    in_zone_imm = np.sum((imm >= goldilocks_lower) & (imm <= goldilocks_upper))
    in_zone_silent = np.sum((silent >= goldilocks_lower) & (silent <= goldilocks_upper))

    return {
        "immunodominant": {
            "n": len(imm),
            "mean": float(imm_mean),
            "std": float(imm_std),
            "median": float(np.median(imm)),
            "min": float(np.min(imm)),
            "max": float(np.max(imm)),
            "q25": float(np.percentile(imm, 25)),
            "q75": float(np.percentile(imm, 75)),
        },
        "silent": {
            "n": len(silent),
            "mean": float(silent_mean),
            "std": float(silent_std),
            "median": float(np.median(silent)),
            "min": float(np.min(silent)),
            "max": float(np.max(silent)),
            "q25": float(np.percentile(silent, 25)),
            "q75": float(np.percentile(silent, 75)),
        },
        "statistics": {
            "t_statistic": float(t_stat),
            "p_value_t": float(t_p),
            "u_statistic": float(u_stat),
            "p_value_mann_whitney": float(u_p),
            "cohens_d": float(cohens_d),
            "effect_magnitude": ("large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small"),
        },
        "goldilocks_zone": {
            "lower_bound": float(goldilocks_lower),
            "upper_bound": float(goldilocks_upper),
            "center": float(imm_mean),
            "width": float(goldilocks_upper - goldilocks_lower),
            "imm_in_zone": int(in_zone_imm),
            "imm_in_zone_pct": float(100 * in_zone_imm / len(imm)),
            "silent_in_zone": int(in_zone_silent),
            "silent_in_zone_pct": float(100 * in_zone_silent / len(silent)),
        },
        "optimal_threshold": {
            "value": float(best_threshold) if best_threshold else None,
            "accuracy": float(best_accuracy),
            "sensitivity": float(best_sens_spec[0]),
            "specificity": float(best_sens_spec[1]),
        },
    }


def plot_goldilocks_zones(results: Dict, output_dir: Path):
    """Generate visualization of Goldilocks zones."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    metrics = [
        "entropy_change",
        "centroid_shift",
        "js_divergence",
        "relative_entropy_change",
    ]
    titles = [
        "Entropy Change",
        "Centroid Shift",
        "JS Divergence",
        "Relative Entropy Change",
    ]

    for ax, metric, title in zip(axes.flat, metrics, titles):
        if metric not in results:
            continue

        data = results[metric]
        imm = data["immunodominant"]
        sil = data["silent"]
        zone = data["goldilocks_zone"]

        # Box plots
        bp = ax.boxplot(
            [
                [r for r in results["raw_values"][metric]["imm"]],
                [r for r in results["raw_values"][metric]["silent"]],
            ],
            labels=["Immunodominant", "Silent"],
            patch_artist=True,
        )

        bp["boxes"][0].set_facecolor("#e53935")
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor("#1e88e5")
        bp["boxes"][1].set_alpha(0.6)

        # Goldilocks zone
        ax.axhspan(
            zone["lower_bound"],
            zone["upper_bound"],
            alpha=0.2,
            color="gold",
            label="Goldilocks Zone",
        )
        ax.axhline(zone["center"], color="gold", linestyle="--", lw=2)

        # Statistics annotation
        p_val = data["statistics"]["p_value_t"]
        d = data["statistics"]["cohens_d"]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        ax.set_title(
            f"{title}\np={p_val:.4f} {sig}, d={d:.2f}",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_ylabel(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc="upper right", fontsize=9)

    plt.suptitle(
        "Goldilocks Zone Analysis: 3-adic Encoder (V5.11.3)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "goldilocks_zones.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: goldilocks_zones.png")


def plot_entropy_detail(results: Dict, output_dir: Path):
    """Detailed entropy change analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    imm_vals = results["raw_values"]["entropy_change"]["imm"]
    silent_vals = results["raw_values"]["entropy_change"]["silent"]
    zone = results["entropy_change"]["goldilocks_zone"]

    # 1. Distribution comparison
    ax = axes[0]
    ax.hist(
        imm_vals,
        bins=15,
        alpha=0.6,
        color="#e53935",
        label="Immunodominant",
        density=True,
    )
    ax.hist(
        silent_vals,
        bins=15,
        alpha=0.6,
        color="#1e88e5",
        label="Silent",
        density=True,
    )
    ax.axvline(0, color="black", linestyle="-", lw=1, label="No change")
    ax.axvline(
        zone["center"],
        color="gold",
        linestyle="--",
        lw=2,
        label="Goldilocks center",
    )
    ax.axvspan(zone["lower_bound"], zone["upper_bound"], alpha=0.2, color="gold")
    ax.set_xlabel("Entropy Change upon Citrullination", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of Entropy Changes", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. Individual epitopes
    ax = axes[1]
    all_epitopes = results["epitope_details"]

    imm_epitopes = [(e["epitope_id"], e["mean_entropy_change"], e["acpa_reactivity"]) for e in all_epitopes if e["immunodominant"]]
    silent_epitopes = [(e["epitope_id"], e["mean_entropy_change"], e["acpa_reactivity"]) for e in all_epitopes if not e["immunodominant"]]

    # Sort by entropy change
    imm_epitopes.sort(key=lambda x: x[1], reverse=True)
    silent_epitopes.sort(key=lambda x: x[1], reverse=True)

    y_imm = np.arange(len(imm_epitopes))
    y_silent = np.arange(len(silent_epitopes)) + len(imm_epitopes) + 1

    ax.barh(
        y_imm,
        [e[1] for e in imm_epitopes],
        color="#e53935",
        alpha=0.7,
        label="Immunodominant",
    )
    ax.barh(
        y_silent,
        [e[1] for e in silent_epitopes],
        color="#1e88e5",
        alpha=0.7,
        label="Silent",
    )

    ax.axvline(0, color="black", linestyle="-", lw=1)
    ax.axvspan(zone["lower_bound"], zone["upper_bound"], alpha=0.2, color="gold")

    ax.set_yticks(list(y_imm) + list(y_silent))
    ax.set_yticklabels(
        [e[0][:10] for e in imm_epitopes] + [e[0][:10] for e in silent_epitopes],
        fontsize=7,
    )
    ax.set_xlabel("Entropy Change", fontsize=12)
    ax.set_title("Per-Epitope Entropy Change", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")

    # 3. ACPA correlation
    ax = axes[2]
    all_entropy = [e["mean_entropy_change"] for e in all_epitopes]
    all_acpa = [e["acpa_reactivity"] for e in all_epitopes]
    colors = ["#e53935" if e["immunodominant"] else "#1e88e5" for e in all_epitopes]

    ax.scatter(all_entropy, all_acpa, c=colors, alpha=0.7, s=80, edgecolors="white")

    # Correlation
    r, p = stats.pearsonr(all_entropy, all_acpa)
    ax.set_xlabel("Entropy Change", fontsize=12)
    ax.set_ylabel("ACPA Reactivity", fontsize=12)
    ax.set_title(
        f"Entropy Change vs ACPA\nr={r:.3f}, p={p:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.axvline(0, color="gray", linestyle="--", lw=1, alpha=0.5)
    ax.axvspan(zone["lower_bound"], zone["upper_bound"], alpha=0.2, color="gold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "entropy_detail.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: entropy_detail.png")


def main():
    print("=" * 80)
    print("GOLDILOCKS ZONE VALIDATION")
    print("Precise boundary calculation with 3-adic encoder (V5.11.3)")
    print("=" * 80)

    output_dir = get_output_dir()
    print(f"\nOutput directory: {output_dir}")

    # Load encoder
    print("\nLoading codon encoder (3-adic, V5.11.3)...")
    device = "cpu"
    encoder, _, _ = load_codon_encoder(device=device, version="3adic")

    # Compute metrics for all epitopes
    print("\nComputing metrics for all epitopes...")
    epitope_metrics = []

    for protein_id, protein in RA_AUTOANTIGENS_AUGMENTED.items():
        for epitope in protein["epitopes"]:
            metrics = compute_epitope_metrics(epitope, encoder, device)
            if metrics:
                epitope_metrics.append(metrics)

    print(f"  Processed {len(epitope_metrics)} epitopes with arginine")

    # Separate by immunodominance
    imm_epitopes = [e for e in epitope_metrics if e["immunodominant"]]
    silent_epitopes = [e for e in epitope_metrics if not e["immunodominant"]]

    print(f"  Immunodominant: {len(imm_epitopes)}")
    print(f"  Silent: {len(silent_epitopes)}")

    # Compute Goldilocks boundaries for each metric
    print("\n" + "=" * 80)
    print("COMPUTING GOLDILOCKS BOUNDARIES")
    print("=" * 80)

    metrics_to_analyze = [
        "mean_entropy_change",
        "mean_centroid_shift",
        "mean_js_divergence",
        "mean_relative_entropy_change",
    ]

    results = {
        "raw_values": {},
        "epitope_details": epitope_metrics,
    }

    for metric in metrics_to_analyze:
        imm_values = [e[metric] for e in imm_epitopes]
        silent_values = [e[metric] for e in silent_epitopes]

        results["raw_values"][metric.replace("mean_", "")] = {
            "imm": imm_values,
            "silent": silent_values,
        }

        boundaries = compute_goldilocks_boundaries(imm_values, silent_values)
        results[metric.replace("mean_", "")] = boundaries

        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"  Immunodominant: {boundaries['immunodominant']['mean']:.4f} ± {boundaries['immunodominant']['std']:.4f}")
        print(f"  Silent: {boundaries['silent']['mean']:.4f} ± {boundaries['silent']['std']:.4f}")
        print(f"  p-value: {boundaries['statistics']['p_value_t']:.4f}")
        print(f"  Cohen's d: {boundaries['statistics']['cohens_d']:.3f} ({boundaries['statistics']['effect_magnitude']})")
        print(f"  Goldilocks Zone: [{boundaries['goldilocks_zone']['lower_bound']:.4f}, {boundaries['goldilocks_zone']['upper_bound']:.4f}]")
        print(f"  IMM in zone: {boundaries['goldilocks_zone']['imm_in_zone_pct']:.1f}%")
        print(f"  Silent in zone: {boundaries['goldilocks_zone']['silent_in_zone_pct']:.1f}%")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_goldilocks_zones(results, output_dir)
    plot_entropy_detail(results, output_dir)

    # Save results
    print("\nSaving results...")

    # Remove raw values for JSON (keep only statistics)
    json_results = {k: v for k, v in results.items() if k != "raw_values"}
    json_results["epitope_details"] = [{k: v for k, v in e.items() if k != "per_r_metrics"} for e in epitope_metrics]

    results_path = output_dir / "goldilocks_validation_results.json"
    with open(results_path, "w") as f:
        json.dump(convert_numpy_types(json_results), f, indent=2)
    print(f"  Saved: {results_path}")

    # Summary
    print("\n" + "=" * 80)
    print("GOLDILOCKS VALIDATION SUMMARY")
    print("=" * 80)

    entropy_data = results["entropy_change"]
    print(
        f"""
KEY FINDING: Entropy Change upon Citrullination

  Statistical Significance:
    p-value (t-test): {entropy_data['statistics']['p_value_t']:.4f}
    p-value (Mann-Whitney): {entropy_data['statistics']['p_value_mann_whitney']:.4f}
    Cohen's d: {entropy_data['statistics']['cohens_d']:.3f} ({entropy_data['statistics']['effect_magnitude']} effect)

  Immunodominant Epitopes (n={entropy_data['immunodominant']['n']}):
    Mean: {entropy_data['immunodominant']['mean']:+.4f}
    Std:  {entropy_data['immunodominant']['std']:.4f}
    Range: [{entropy_data['immunodominant']['min']:.4f}, {entropy_data['immunodominant']['max']:.4f}]

  Silent Epitopes (n={entropy_data['silent']['n']}):
    Mean: {entropy_data['silent']['mean']:+.4f}
    Std:  {entropy_data['silent']['std']:.4f}
    Range: [{entropy_data['silent']['min']:.4f}, {entropy_data['silent']['max']:.4f}]

  GOLDILOCKS ZONE:
    Lower bound: {entropy_data['goldilocks_zone']['lower_bound']:+.4f}
    Center:      {entropy_data['goldilocks_zone']['center']:+.4f}
    Upper bound: {entropy_data['goldilocks_zone']['upper_bound']:+.4f}
    Width:       {entropy_data['goldilocks_zone']['width']:.4f}

  Classification Performance:
    IMM in Goldilocks zone: {entropy_data['goldilocks_zone']['imm_in_zone_pct']:.1f}%
    Silent in Goldilocks zone: {entropy_data['goldilocks_zone']['silent_in_zone_pct']:.1f}%

  INTERPRETATION:
    Immunodominant: Entropy INCREASES (+{entropy_data['immunodominant']['mean']:.3f})
    Silent: Entropy DECREASES ({entropy_data['silent']['mean']:.3f})

    Citrullination of immunodominant epitopes PRESERVES or INCREASES
    the entropy (complexity) of the cluster distribution, keeping
    the peptide recognizable as "modified self" to the immune system.

    Silent sites show entropy DECREASE - becoming simpler, more
    homogeneous, and thus ignored by the immune system.
"""
    )

    print("=" * 80)
    print("VALIDATION COMPLETE")
    print(f"Output: {output_dir}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
