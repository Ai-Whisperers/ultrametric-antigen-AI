#!/usr/bin/env python3
"""
Citrullination Shift Analysis - HYPERBOLIC GEOMETRY

Follow-up analysis focusing on the CHANGE in Poincaré ball space caused by citrullination,
rather than the absolute position of arginine residues.

Key insight from 06_autoantigen_epitope_analysis.py:
- Immunodominant sites have LOWER boundary crossing potential
- This suggests the immune trigger isn't raw "foreignness"
- Instead, it may be the MAGNITUDE of change from self → modified

Hypothesis: Immunodominant citrullination sites cause a specific magnitude of
hyperbolic shift that crosses a recognition threshold - not too small (ignored)
and not too large (tolerized as completely foreign).

Version: 2.0 - Updated to use Poincaré ball geometry
"""

import json
from collections import defaultdict

import numpy as np
import torch
# Import hyperbolic utilities
from hyperbolic_utils import (AA_TO_CODON, codon_to_onehot, get_results_dir,
                              load_codon_encoder)
from scipy import stats

# ============================================================================
# AUTOANTIGEN DATA (same as 06)
# ============================================================================

# Epitope database with ACPA reactivity percentages
EPITOPE_DATABASE = [
    # Vimentin
    {
        "id": "VIM_R71",
        "seq": "RLRSSVPGVR",
        "arg_pos": [0, 2, 9],
        "immunodominant": True,
        "acpa": 0.85,
    },
    {
        "id": "VIM_R257",
        "seq": "SSLNLRETNL",
        "arg_pos": [5],
        "immunodominant": True,
        "acpa": 0.72,
    },
    {
        "id": "VIM_R45",
        "seq": "SSRSFRTYSF",
        "arg_pos": [2, 5],
        "immunodominant": False,
        "acpa": 0.15,
    },
    {
        "id": "VIM_R201",
        "seq": "ARLRSSLAGS",
        "arg_pos": [0, 2],
        "immunodominant": True,
        "acpa": 0.68,
    },
    # Fibrinogen
    {
        "id": "FGA_R38",
        "seq": "GPRVVERHQS",
        "arg_pos": [2, 6],
        "immunodominant": True,
        "acpa": 0.78,
    },
    {
        "id": "FGA_R573",
        "seq": "MELERPGGNEI",
        "arg_pos": [4],
        "immunodominant": True,
        "acpa": 0.65,
    },
    {
        "id": "FGA_R84",
        "seq": "RHPDEAAFFDT",
        "arg_pos": [0],
        "immunodominant": False,
        "acpa": 0.22,
    },
    {
        "id": "FGB_R74",
        "seq": "HARPAKAATN",
        "arg_pos": [2],
        "immunodominant": True,
        "acpa": 0.71,
    },
    {
        "id": "FGB_R44",
        "seq": "NEEGFFRHNDK",
        "arg_pos": [7],
        "immunodominant": False,
        "acpa": 0.18,
    },
    # Alpha-enolase
    {
        "id": "ENO1_CEP1",
        "seq": "KIREEIFDSRGNP",
        "arg_pos": [2, 9],
        "immunodominant": True,
        "acpa": 0.62,
    },
    {
        "id": "ENO1_R400",
        "seq": "SFRSGKYKSV",
        "arg_pos": [2],
        "immunodominant": False,
        "acpa": 0.12,
    },
    # Collagen II
    {
        "id": "CII_259",
        "seq": "GARGLTGRPGDAGK",
        "arg_pos": [2, 7],
        "immunodominant": True,
        "acpa": 0.45,
    },
    {
        "id": "CII_511",
        "seq": "PGERGAPGFRGPAG",
        "arg_pos": [3, 10],
        "immunodominant": True,
        "acpa": 0.38,
    },
    # Filaggrin
    {
        "id": "FLG_CCP",
        "seq": "SHQESTRGRS",
        "arg_pos": [6, 8],
        "immunodominant": True,
        "acpa": 0.75,
    },
    {
        "id": "FLG_SEC",
        "seq": "DSHRGSSSSS",
        "arg_pos": [3],
        "immunodominant": False,
        "acpa": 0.20,
    },
    # Histones
    {
        "id": "H2A_R3",
        "seq": "SGRGKQGGKAR",
        "arg_pos": [2, 10],
        "immunodominant": True,
        "acpa": 0.35,
    },
]


# ============================================================================
# CITRULLINATION SHIFT ANALYSIS
# ============================================================================


def compute_epitope_shift_profile(epitope, encoder, device="cpu"):
    """
    Compute how citrullination shifts the epitope in p-adic space.

    For each arginine position, we calculate:
    1. Original epitope centroid
    2. Citrullinated epitope centroid (R removed)
    3. Magnitude of shift
    4. Direction of shift (toward which cluster?)
    5. Change in cluster probability distribution
    """
    seq = epitope["seq"]
    codons = [AA_TO_CODON.get(aa, "NNN") for aa in seq]

    # Get all embeddings
    embeddings = []
    cluster_probs = []

    for codon in codons:
        if codon != "NNN":
            onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                probs, emb = encoder.get_cluster_probs(onehot)
                embeddings.append(emb.cpu().numpy().squeeze())
                cluster_probs.append(probs.cpu().numpy().squeeze())

    embeddings = np.array(embeddings)
    cluster_probs = np.array(cluster_probs)

    if len(embeddings) == 0:
        return None

    # Original centroid and cluster distribution
    original_centroid = np.mean(embeddings, axis=0)
    original_cluster_dist = np.mean(cluster_probs, axis=0)
    original_entropy = -np.sum(original_cluster_dist * np.log(original_cluster_dist + 1e-10))

    # Analyze each arginine
    arg_shifts = []

    for arg_idx in epitope["arg_pos"]:
        if arg_idx >= len(embeddings):
            continue

        # Create mask excluding this arginine
        mask = np.ones(len(embeddings), dtype=bool)
        mask[arg_idx] = False
        remaining = embeddings[mask]
        remaining_probs = cluster_probs[mask]

        if len(remaining) == 0:
            continue

        # Citrullinated centroid
        cit_centroid = np.mean(remaining, axis=0)
        cit_cluster_dist = np.mean(remaining_probs, axis=0)
        cit_entropy = -np.sum(cit_cluster_dist * np.log(cit_cluster_dist + 1e-10))

        # Compute shifts
        centroid_shift = np.linalg.norm(cit_centroid - original_centroid)
        relative_shift = centroid_shift / np.linalg.norm(original_centroid) if np.linalg.norm(original_centroid) > 0 else 0

        # KL divergence between original and citrullinated cluster distributions
        kl_div = np.sum(original_cluster_dist * np.log((original_cluster_dist + 1e-10) / (cit_cluster_dist + 1e-10)))

        # Jensen-Shannon divergence (symmetric)
        m = 0.5 * (original_cluster_dist + cit_cluster_dist)
        js_div = 0.5 * np.sum(original_cluster_dist * np.log((original_cluster_dist + 1e-10) / (m + 1e-10))) + 0.5 * np.sum(
            cit_cluster_dist * np.log((cit_cluster_dist + 1e-10) / (m + 1e-10))
        )

        # Entropy change (does citrullination make the epitope more/less defined?)
        entropy_change = cit_entropy - original_entropy

        # Which cluster gains the most probability after citrullination?
        cluster_shift = cit_cluster_dist - original_cluster_dist
        max_gain_cluster = np.argmax(cluster_shift)
        max_loss_cluster = np.argmin(cluster_shift)

        # Original cluster of the arginine
        arg_cluster = np.argmax(cluster_probs[arg_idx])

        arg_shifts.append(
            {
                "position": arg_idx,
                "centroid_shift": centroid_shift,
                "relative_shift": relative_shift,
                "kl_divergence": kl_div,
                "js_divergence": js_div,
                "entropy_change": entropy_change,
                "original_entropy": original_entropy,
                "cit_entropy": cit_entropy,
                "arg_cluster": int(arg_cluster),
                "max_gain_cluster": int(max_gain_cluster),
                "max_loss_cluster": int(max_loss_cluster),
                "max_cluster_gain": float(cluster_shift[max_gain_cluster]),
            }
        )

    return {
        "epitope_id": epitope["id"],
        "immunodominant": epitope["immunodominant"],
        "acpa": epitope["acpa"],
        "arg_shifts": arg_shifts,
    }


def analyze_shift_patterns(all_shifts):
    """
    Compare shift magnitudes between immunodominant and silent epitopes.
    """
    imm_data = defaultdict(list)
    sil_data = defaultdict(list)

    for result in all_shifts:
        if result is None:
            continue

        target = imm_data if result["immunodominant"] else sil_data

        for shift in result["arg_shifts"]:
            for key, value in shift.items():
                if key not in [
                    "position",
                    "arg_cluster",
                    "max_gain_cluster",
                    "max_loss_cluster",
                ]:
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        target[key].append(float(value))

    # Statistical comparison
    comparisons = {}
    metrics = [
        "centroid_shift",
        "relative_shift",
        "js_divergence",
        "entropy_change",
    ]

    for metric in metrics:
        if imm_data[metric] and sil_data[metric]:
            t_stat, p_val = stats.ttest_ind(imm_data[metric], sil_data[metric])

            imm_mean = np.mean(imm_data[metric])
            sil_mean = np.mean(sil_data[metric])
            pooled_std = np.sqrt((np.var(imm_data[metric]) + np.var(sil_data[metric])) / 2)
            effect = (imm_mean - sil_mean) / pooled_std if pooled_std > 0 else 0

            comparisons[metric] = {
                "imm_mean": imm_mean,
                "imm_std": np.std(imm_data[metric]),
                "sil_mean": sil_mean,
                "sil_std": np.std(sil_data[metric]),
                "t_stat": t_stat,
                "p_value": p_val,
                "effect_size": effect,
            }

    return comparisons, imm_data, sil_data


def correlate_shift_with_acpa(all_shifts):
    """
    Correlate shift magnitudes with ACPA reactivity percentage.
    """
    shift_values = []
    acpa_values = []

    for result in all_shifts:
        if result is None or not result["arg_shifts"]:
            continue

        # Average shift across all R in this epitope
        mean_shift = np.mean([s["relative_shift"] for s in result["arg_shifts"]])
        shift_values.append(mean_shift)
        acpa_values.append(result["acpa"])

    if len(shift_values) > 3:
        r, p = stats.pearsonr(shift_values, acpa_values)
        return {"r": r, "p": p, "n": len(shift_values)}

    return None


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 80)
    print("CITRULLINATION SHIFT ANALYSIS - HYPERBOLIC SPACE PERTURBATION")
    print("=" * 80)

    # Setup - use hyperbolic results directory
    results_dir = get_results_dir(hyperbolic=True)
    print(f"\nResults will be saved to: {results_dir}")

    # Load codon encoder using utility function
    # Using '3adic' version (native hyperbolic from V5.11.3)
    print("\nLoading codon encoder (3-adic, V5.11.3)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, mapping, _ = load_codon_encoder(device=device, version="3adic")

    # =========================================================================
    # ANALYZE ALL EPITOPES
    # =========================================================================

    print("\n" + "=" * 80)
    print("EPITOPE SHIFT ANALYSIS")
    print("=" * 80)

    all_shifts = []

    for epitope in EPITOPE_DATABASE:
        result = compute_epitope_shift_profile(epitope, encoder, device)
        all_shifts.append(result)

        if result:
            status = "IMMUNODOMINANT" if result["immunodominant"] else "Silent"
            print(f"\n{result['epitope_id']} ({status}, ACPA={result['acpa']*100:.0f}%)")

            for shift in result["arg_shifts"]:
                print(
                    f"  R@{shift['position']}: "
                    f"centroid_shift={shift['centroid_shift']:.4f}, "
                    f"relative={shift['relative_shift']*100:.1f}%, "
                    f"JS_div={shift['js_divergence']:.4f}, "
                    f"ΔS={shift['entropy_change']:.4f}"
                )

    # =========================================================================
    # STATISTICAL ANALYSIS
    # =========================================================================

    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON: IMMUNODOMINANT vs SILENT")
    print("=" * 80)

    comparisons, imm_data, sil_data = analyze_shift_patterns(all_shifts)

    for metric, stats_data in comparisons.items():
        sig = "***" if stats_data["p_value"] < 0.001 else ("**" if stats_data["p_value"] < 0.01 else "*" if stats_data["p_value"] < 0.05 else "")

        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"  Immunodominant: {stats_data['imm_mean']:.4f} ± {stats_data['imm_std']:.4f}")
        print(f"  Silent:         {stats_data['sil_mean']:.4f} ± {stats_data['sil_std']:.4f}")
        print(f"  t = {stats_data['t_stat']:.3f}, p = {stats_data['p_value']:.4f} {sig}")
        print(f"  Cohen's d = {stats_data['effect_size']:.3f}")

    # =========================================================================
    # ACPA CORRELATION
    # =========================================================================

    print("\n" + "=" * 80)
    print("CORRELATION: SHIFT MAGNITUDE vs ACPA REACTIVITY")
    print("=" * 80)

    corr = correlate_shift_with_acpa(all_shifts)
    if corr:
        sig = "***" if corr["p"] < 0.001 else "**" if corr["p"] < 0.01 else "*" if corr["p"] < 0.05 else ""
        print(f"\nPearson correlation: r = {corr['r']:.3f}, p = {corr['p']:.4f} {sig}")
        print(f"Sample size: n = {corr['n']}")

        if corr["r"] < 0:
            print("\nNEGATIVE correlation: Smaller shifts → Higher ACPA reactivity")
            print("This supports the 'Goldilocks zone' hypothesis!")
        else:
            print("\nPOSITIVE correlation: Larger shifts → Higher ACPA reactivity")

    # =========================================================================
    # KEY FINDINGS
    # =========================================================================

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Check for Goldilocks pattern
    if "relative_shift" in comparisons:
        rs = comparisons["relative_shift"]
        if rs["imm_mean"] < rs["sil_mean"] and rs["p_value"] < 0.1:
            print("\n✓ GOLDILOCKS ZONE HYPOTHESIS SUPPORTED:")
            print(f"  Immunodominant epitopes show SMALLER shifts ({rs['imm_mean']*100:.1f}%)")
            print(f"  than silent epitopes ({rs['sil_mean']*100:.1f}%)")
            print("\n  Interpretation: Citrullination at immunodominant sites causes")
            print("  a moderate perturbation - enough to break self-tolerance but not")
            print("  so large that the epitope appears completely foreign and is ignored.")

    if "js_divergence" in comparisons:
        js = comparisons["js_divergence"]
        print("\n✓ CLUSTER DISTRIBUTION SHIFT:")
        print(f"  Immunodominant JS divergence: {js['imm_mean']:.4f}")
        print(f"  Silent JS divergence:         {js['sil_mean']:.4f}")
        print(f"  The cluster probability distribution changes {'less' if js['imm_mean'] < js['sil_mean'] else 'more'} " f"in immunodominant sites")

    if "entropy_change" in comparisons:
        ec = comparisons["entropy_change"]
        direction = "increases" if ec["imm_mean"] > 0 else "decreases"
        print("\n✓ ENTROPY EFFECT:")
        print(f"  Citrullination {direction} epitope entropy by {abs(ec['imm_mean']):.4f}")
        print("  in immunodominant sites")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    output = {
        "all_shifts": [
            {
                **r,
                "arg_shifts": [{k: float(v) if isinstance(v, np.floating) else v for k, v in s.items()} for s in r["arg_shifts"]],
            }
            for r in all_shifts
            if r
        ],
        "comparisons": {k: {kk: float(vv) if isinstance(vv, np.floating) else vv for kk, vv in v.items()} for k, v in comparisons.items()},
        "acpa_correlation": corr,
    }

    output_path = results_dir / "citrullination_shift_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
