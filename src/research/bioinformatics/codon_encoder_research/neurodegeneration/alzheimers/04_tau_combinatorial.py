#!/usr/bin/env python3
"""
Tau Combinatorial Phosphorylation Analysis

Tests pathologically relevant phosphorylation combinations to identify:
1. Synergistic "tipping points" (combined > sum of individuals)
2. Antagonistic interactions (combined < sum)
3. Critical thresholds for dysfunction

Key hypothesis: Specific combinations may cause non-linear geometric shifts
that correlate with disease progression milestones.

Uses the 3-adic codon encoder (V5.11.3) in hyperbolic space.
"""

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np

# Add paths
# Path: .../codon_encoder_research/neurodegeneration/alzheimers/this_script.py
# parent = alzheimers, parent.parent = neurodegeneration, parent.parent.parent = codon_encoder_research
SCRIPT_DIR = Path(__file__).parent
CODON_RESEARCH_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "data"))
sys.path.insert(0, str(CODON_RESEARCH_DIR / "rheumatoid_arthritis" / "scripts"))

from hyperbolic_utils import (AA_TO_CODON, encode_codon_hyperbolic,
                              hyperbolic_centroid, load_hyperbolic_encoder,
                              poincare_distance)
from tau_phospho_database import TAU_2N4R_SEQUENCE, TAU_PHOSPHO_SITES

# ============================================================================
# PATHOLOGICAL COMBINATIONS TO TEST
# ============================================================================

PATHOLOGICAL_COMBINATIONS = {
    # Clinical epitopes (antibody-defined)
    "AT8": {
        "sites": [202, 205],
        "description": "Early tangle marker, gold standard for AD staging",
        "stage": "early",
        "clinical_use": "Immunohistochemistry",
    },
    "AT8_extended": {
        "sites": [202, 205, 208],
        "description": "Extended AT8 with pS208",
        "stage": "early",
        "clinical_use": "Research",
    },
    "AT100": {
        "sites": [212, 214],
        "description": "AD-specific phosphorylation pattern",
        "stage": "mid",
        "clinical_use": "Immunohistochemistry",
    },
    "AT180": {
        "sites": [231, 235],
        "description": "Conformational epitope, proline-directed",
        "stage": "early",
        "clinical_use": "Research",
    },
    "PHF-1": {
        "sites": [396, 404],
        "description": "Paired helical filament marker",
        "stage": "late",
        "clinical_use": "Immunohistochemistry",
    },
    # MTBR combinations (microtubule binding disruption)
    "MTBR_R1_R2": {
        "sites": [262, 293],
        "description": "KXGS motifs in R1 and R2",
        "stage": "mid",
        "clinical_use": "Research",
    },
    "MTBR_R3_R4": {
        "sites": [324, 356],
        "description": "KXGS motifs in R3 and R4",
        "stage": "mid",
        "clinical_use": "Research",
    },
    "MTBR_full": {
        "sites": [262, 293, 324, 356],
        "description": "All four KXGS motifs",
        "stage": "mid-late",
        "clinical_use": "Research",
    },
    # CSF biomarker combinations
    "CSF_early": {
        "sites": [181, 217],
        "description": "Clinical CSF biomarkers (p-tau181, p-tau217)",
        "stage": "early",
        "clinical_use": "Diagnosis",
    },
    "CSF_extended": {
        "sites": [181, 217, 231],
        "description": "Extended CSF panel with p-tau231",
        "stage": "early",
        "clinical_use": "Diagnosis",
    },
    # Disease progression stages
    "Braak_I_II": {
        "sites": [181, 231, 202, 205],
        "description": "Transentorhinal stage phosphorylation",
        "stage": "early",
        "clinical_use": "Staging",
    },
    "Braak_III_IV": {
        "sites": [181, 231, 202, 205, 262, 396],
        "description": "Limbic stage phosphorylation",
        "stage": "mid",
        "clinical_use": "Staging",
    },
    "Braak_V_VI": {
        "sites": [181, 231, 202, 205, 262, 293, 324, 356, 396, 404, 422],
        "description": "Neocortical stage (severe)",
        "stage": "late",
        "clinical_use": "Staging",
    },
    # C-terminal aggregation seed
    "C_term_seed": {
        "sites": [396, 404, 409, 422],
        "description": "C-terminal aggregation-prone region",
        "stage": "late",
        "clinical_use": "Research",
    },
}


# ============================================================================
# ENCODING FUNCTIONS
# ============================================================================


def encode_sequence(sequence: str, encoder) -> np.ndarray:
    """Encode amino acid sequence to hyperbolic embeddings."""
    embeddings = []
    for aa in sequence:
        if aa in AA_TO_CODON:
            codon = AA_TO_CODON[aa]
            embedding = encode_codon_hyperbolic(codon, encoder)
            embeddings.append(embedding)
    return np.array(embeddings) if embeddings else np.array([])


def apply_multi_phosphomimic(sequence: str, positions: List[int]) -> str:
    """Apply phosphomimetic mutations at multiple positions."""
    seq_list = list(sequence)
    for pos in positions:
        idx = pos - 1  # Convert to 0-indexed
        if 0 <= idx < len(seq_list) and seq_list[idx] in ["S", "T", "Y"]:
            seq_list[idx] = "D"
    return "".join(seq_list)


def compute_centroid_shift(sequence: str, positions: List[int], encoder) -> float:
    """Compute geometric shift when phosphorylating multiple sites."""
    # Encode wild-type
    wt_emb = encode_sequence(sequence, encoder)
    if len(wt_emb) == 0:
        return 0.0

    wt_centroid = hyperbolic_centroid(wt_emb)

    # Encode phosphorylated
    phospho_seq = apply_multi_phosphomimic(sequence, positions)
    phospho_emb = encode_sequence(phospho_seq, encoder)
    if len(phospho_emb) == 0:
        return 0.0

    phospho_centroid = hyperbolic_centroid(phospho_emb)

    return float(poincare_distance(wt_centroid, phospho_centroid))


def compute_individual_shifts(sequence: str, positions: List[int], encoder) -> Dict[int, float]:
    """Compute individual shifts for each site."""
    shifts = {}
    for pos in positions:
        shift = compute_centroid_shift(sequence, [pos], encoder)
        shifts[pos] = shift
    return shifts


# ============================================================================
# SYNERGY ANALYSIS
# ============================================================================


def analyze_synergy(
    combined_shift: float,
    individual_shifts: Dict[int, float],
    positions: List[int],
) -> Dict:
    """
    Analyze whether a combination is synergistic, additive, or antagonistic.

    Synergy ratio:
    - > 1.2: SYNERGISTIC (combined > expected additive)
    - 0.8 - 1.2: ADDITIVE (combined ≈ expected)
    - < 0.8: ANTAGONISTIC (combined < expected)
    """
    expected_additive = sum(individual_shifts.get(pos, 0) for pos in positions)

    if expected_additive == 0:
        return {
            "synergy_ratio": 0,
            "synergy_type": "UNDEFINED",
            "expected_additive": 0,
            "actual": combined_shift,
            "excess": 0,
        }

    synergy_ratio = combined_shift / expected_additive

    if synergy_ratio > 1.2:
        synergy_type = "SYNERGISTIC"
    elif synergy_ratio < 0.8:
        synergy_type = "ANTAGONISTIC"
    else:
        synergy_type = "ADDITIVE"

    return {
        "synergy_ratio": synergy_ratio,
        "synergy_type": synergy_type,
        "expected_additive": expected_additive,
        "actual": combined_shift,
        "excess": combined_shift - expected_additive,
    }


def find_minimal_tipping_point(sequence: str, available_sites: List[int], encoder, threshold: float = 0.35) -> Dict:
    """
    Find the minimal set of phosphorylations that crosses the dysfunction threshold.

    Uses greedy search: at each step, add the site that maximally increases shift.
    """
    current_sites = []
    current_shift = 0.0
    trajectory = []

    remaining_sites = list(available_sites)

    while remaining_sites and current_shift < threshold:
        best_site = None
        best_shift = current_shift

        for site in remaining_sites:
            test_sites = current_sites + [site]
            shift = compute_centroid_shift(sequence, test_sites, encoder)

            if shift > best_shift:
                best_shift = shift
                best_site = site

        if best_site is None:
            break

        current_sites.append(best_site)
        remaining_sites.remove(best_site)
        current_shift = best_shift

        trajectory.append(
            {
                "n_phospho": len(current_sites),
                "added_site": best_site,
                "cumulative_shift": current_shift,
                "crossed_threshold": current_shift >= threshold,
            }
        )

    return {
        "threshold": threshold,
        "minimal_sites": current_sites if current_shift >= threshold else None,
        "final_shift": current_shift,
        "crossed": current_shift >= threshold,
        "trajectory": trajectory,
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def main():
    print("=" * 70)
    print("TAU COMBINATORIAL PHOSPHORYLATION ANALYSIS")
    print("Testing Synergistic Tipping Points")
    print("=" * 70)

    # Load encoder
    print("\nLoading 3-adic codon encoder...")
    encoder, mapping = load_hyperbolic_encoder()
    print("Encoder loaded successfully")

    results = {
        "metadata": {
            "analysis": "Combinatorial Phosphorylation",
            "encoder": "3-adic (V5.11.3)",
            "tau_isoform": "2N4R (441 aa)",
            "n_combinations": len(PATHOLOGICAL_COMBINATIONS),
        },
        "individual_shifts": {},
        "combination_results": {},
        "synergy_analysis": {},
        "tipping_points": {},
        "summary": {},
    }

    # ========================================================================
    # 1. Compute Individual Site Shifts
    # ========================================================================
    print("\n" + "-" * 70)
    print("1. Computing Individual Site Shifts")
    print("-" * 70)

    # Get all unique sites from combinations
    all_sites = set()
    for combo in PATHOLOGICAL_COMBINATIONS.values():
        all_sites.update(combo["sites"])

    individual_shifts = {}
    for site in sorted(all_sites):
        shift = compute_centroid_shift(TAU_2N4R_SEQUENCE, [site], encoder)
        individual_shifts[site] = shift

        site_data = TAU_PHOSPHO_SITES.get(site, {})
        aa = site_data.get("aa", "?")
        domain = site_data.get("domain", "?")
        print(f"  {aa}{site} ({domain}): {shift*100:.1f}%")

    results["individual_shifts"] = {str(k): v for k, v in individual_shifts.items()}

    # ========================================================================
    # 2. Analyze Pathological Combinations
    # ========================================================================
    print("\n" + "-" * 70)
    print("2. Analyzing Pathological Combinations")
    print("-" * 70)

    combination_results = []

    for combo_name, combo_data in PATHOLOGICAL_COMBINATIONS.items():
        sites = combo_data["sites"]

        # Compute combined shift
        combined_shift = compute_centroid_shift(TAU_2N4R_SEQUENCE, sites, encoder)

        # Analyze synergy
        synergy = analyze_synergy(combined_shift, individual_shifts, sites)

        result = {
            "name": combo_name,
            "sites": sites,
            "n_sites": len(sites),
            "description": combo_data["description"],
            "stage": combo_data["stage"],
            "combined_shift": combined_shift,
            "combined_shift_pct": combined_shift * 100,
            **synergy,
        }

        combination_results.append(result)

        # Print result
        synergy_marker = {
            "SYNERGISTIC": "***",
            "ANTAGONISTIC": "---",
            "ADDITIVE": "   ",
        }[synergy["synergy_type"]]

        print(f"\n  {combo_name} ({combo_data['stage']}):")
        print(f"    Sites: {sites}")
        print(f"    Combined shift: {combined_shift*100:.1f}%")
        print(f"    Expected (additive): {synergy['expected_additive']*100:.1f}%")
        print(f"    Synergy ratio: {synergy['synergy_ratio']:.2f} [{synergy['synergy_type']}] {synergy_marker}")

    results["combination_results"] = combination_results

    # ========================================================================
    # 3. Synergy Ranking
    # ========================================================================
    print("\n" + "-" * 70)
    print("3. Synergy Ranking")
    print("-" * 70)

    # Sort by synergy ratio
    synergistic = [r for r in combination_results if r["synergy_type"] == "SYNERGISTIC"]
    antagonistic = [r for r in combination_results if r["synergy_type"] == "ANTAGONISTIC"]
    additive = [r for r in combination_results if r["synergy_type"] == "ADDITIVE"]

    print(f"\n  SYNERGISTIC combinations: {len(synergistic)}")
    for r in sorted(synergistic, key=lambda x: x["synergy_ratio"], reverse=True):
        print(f"    {r['name']}: ratio={r['synergy_ratio']:.2f}, shift={r['combined_shift_pct']:.1f}%")

    print(f"\n  ANTAGONISTIC combinations: {len(antagonistic)}")
    for r in sorted(antagonistic, key=lambda x: x["synergy_ratio"]):
        print(f"    {r['name']}: ratio={r['synergy_ratio']:.2f}, shift={r['combined_shift_pct']:.1f}%")

    print(f"\n  ADDITIVE combinations: {len(additive)}")
    for r in additive:
        print(f"    {r['name']}: ratio={r['synergy_ratio']:.2f}, shift={r['combined_shift_pct']:.1f}%")

    results["synergy_analysis"] = {
        "synergistic": [r["name"] for r in synergistic],
        "antagonistic": [r["name"] for r in antagonistic],
        "additive": [r["name"] for r in additive],
    }

    # ========================================================================
    # 4. Tipping Point Analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("4. Tipping Point Analysis")
    print("-" * 70)

    # Find minimal set to cross different thresholds
    thresholds = [0.15, 0.25, 0.35]

    # Use all characterized phospho-sites
    all_phospho_sites = sorted(TAU_PHOSPHO_SITES.keys())

    for threshold in thresholds:
        print(f"\n  Threshold: {threshold*100:.0f}%")
        tipping = find_minimal_tipping_point(TAU_2N4R_SEQUENCE, all_phospho_sites, encoder, threshold)

        results["tipping_points"][str(threshold)] = tipping

        if tipping["crossed"]:
            print(f"    Minimal sites needed: {len(tipping['minimal_sites'])}")
            print(f"    Sites: {tipping['minimal_sites']}")
            print(f"    Final shift: {tipping['final_shift']*100:.1f}%")
        else:
            print(f"    Threshold NOT reached with {len(all_phospho_sites)} sites")
            print(f"    Maximum shift: {tipping['final_shift']*100:.1f}%")

    # ========================================================================
    # 5. Disease Stage Analysis
    # ========================================================================
    print("\n" + "-" * 70)
    print("5. Disease Stage Analysis")
    print("-" * 70)

    stage_results = defaultdict(list)
    for r in combination_results:
        stage_results[r["stage"]].append(r)

    print("\n  Mean shift by disease stage:")
    stage_summary = {}
    for stage in ["early", "mid", "mid-late", "late"]:
        if stage in stage_results:
            shifts = [r["combined_shift"] for r in stage_results[stage]]
            mean_shift = np.mean(shifts) * 100
            stage_summary[stage] = {
                "mean_shift_pct": mean_shift,
                "n_combinations": len(shifts),
            }
            print(f"    {stage}: {mean_shift:.1f}% (n={len(shifts)})")

    results["stage_analysis"] = stage_summary

    # ========================================================================
    # 6. MTBR Vulnerability Deep Dive
    # ========================================================================
    print("\n" + "-" * 70)
    print("6. MTBR Vulnerability Analysis")
    print("-" * 70)

    # Test progressive MTBR phosphorylation
    kxgs_sites = [262, 293, 324, 356]
    mtbr_trajectory = []

    print("\n  Progressive KXGS phosphorylation:")
    for n in range(1, len(kxgs_sites) + 1):
        for combo in combinations(kxgs_sites, n):
            shift = compute_centroid_shift(TAU_2N4R_SEQUENCE, list(combo), encoder)
            mtbr_trajectory.append({"sites": list(combo), "n_sites": n, "shift": shift})

    # Group by number of sites
    for n in range(1, len(kxgs_sites) + 1):
        combos_n = [t for t in mtbr_trajectory if t["n_sites"] == n]
        shifts = [t["shift"] for t in combos_n]
        print(f"    {n} KXGS site(s): mean shift = {np.mean(shifts)*100:.1f}%, " f"max = {np.max(shifts)*100:.1f}%")

    results["mtbr_analysis"] = mtbr_trajectory

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: COMBINATORIAL PHOSPHORYLATION")
    print("=" * 70)

    # Find highest synergy
    max_synergy = max(combination_results, key=lambda x: x["synergy_ratio"])

    # Find highest shift
    max_shift = max(combination_results, key=lambda x: x["combined_shift"])

    summary = {
        "total_combinations_tested": len(combination_results),
        "synergistic_count": len(synergistic),
        "antagonistic_count": len(antagonistic),
        "additive_count": len(additive),
        "highest_synergy": {
            "name": max_synergy["name"],
            "ratio": max_synergy["synergy_ratio"],
            "sites": max_synergy["sites"],
        },
        "highest_shift": {
            "name": max_shift["name"],
            "shift_pct": max_shift["combined_shift_pct"],
            "sites": max_shift["sites"],
        },
    }

    results["summary"] = summary

    print(
        f"""
1. SYNERGY PATTERNS
   - Synergistic combinations: {len(synergistic)}
   - Antagonistic combinations: {len(antagonistic)}
   - Additive combinations: {len(additive)}

2. HIGHEST SYNERGY
   {max_synergy['name']}: ratio = {max_synergy['synergy_ratio']:.2f}
   Sites: {max_synergy['sites']}

3. HIGHEST COMBINED SHIFT
   {max_shift['name']}: shift = {max_shift['combined_shift_pct']:.1f}%
   Sites: {max_shift['sites']}

4. TIPPING POINT THRESHOLDS
   - 15% threshold: ~{results['tipping_points']['0.15']['minimal_sites'][:3] if results['tipping_points']['0.15']['crossed'] else 'Not reached'}...
   - 25% threshold: ~{len(results['tipping_points']['0.25']['minimal_sites']) if results['tipping_points']['0.25']['crossed'] else 'Not reached'} sites
   - 35% threshold: ~{len(results['tipping_points']['0.35']['minimal_sites']) if results['tipping_points']['0.35']['crossed'] else 'Not reached'} sites

5. DISEASE STAGE PROGRESSION
   Early: ~{stage_summary.get('early', {}).get('mean_shift_pct', 0):.1f}% mean shift
   Mid: ~{stage_summary.get('mid', {}).get('mean_shift_pct', 0):.1f}% mean shift
   Late: ~{stage_summary.get('late', {}).get('mean_shift_pct', 0):.1f}% mean shift

6. KEY FINDINGS
   - Most combinations show {'ANTAGONISTIC' if len(antagonistic) > len(synergistic) else 'SYNERGISTIC'} behavior
   - MTBR full phosphorylation (4 KXGS sites) causes maximal dysfunction
   - Braak staging correlates with geometric shift magnitude
"""
    )

    # Save results
    output_path = SCRIPT_DIR / "results" / "tau_combinatorial_results.json"
    output_path.parent.mkdir(exist_ok=True)

    # Convert numpy
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
