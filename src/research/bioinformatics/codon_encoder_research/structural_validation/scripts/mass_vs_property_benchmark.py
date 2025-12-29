#!/usr/bin/env python3
"""
Mass vs Property-Based Prediction Benchmark

Tests whether MASS (the strongest p-adic correlate at ρ=0.76) outperforms
traditional property-based approaches (hydropathy + volume).

Hypothesis: The genetic code's p-adic structure encodes molecular dynamics
(mass-dependent) rather than just static biochemistry. If true, mass-based
predictions should outperform property-based on tasks involving:
1. Protein folding kinetics
2. Aggregation propensity
3. Evolutionary substitution rates
4. PTM effects

Author: Research Team
Date: December 2025
"""

from __future__ import annotations

import json
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
VALIDATION_DIR = SCRIPT_DIR.parent
RESULTS_DIR = VALIDATION_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

GENETIC_CODE_DIR = SCRIPT_DIR.parent.parent.parent / "genetic_code" / "data"

# ============================================================================
# AMINO ACID DATA
# ============================================================================

AA_PROPERTIES = {
    'A': {'hydropathy': 1.8, 'volume': 88.6, 'mass': 89.09, 'charge': 0},
    'R': {'hydropathy': -4.5, 'volume': 173.4, 'mass': 174.20, 'charge': 1},
    'N': {'hydropathy': -3.5, 'volume': 114.1, 'mass': 132.12, 'charge': 0},
    'D': {'hydropathy': -3.5, 'volume': 111.1, 'mass': 133.10, 'charge': -1},
    'C': {'hydropathy': 2.5, 'volume': 108.5, 'mass': 121.16, 'charge': 0},
    'Q': {'hydropathy': -3.5, 'volume': 143.8, 'mass': 146.15, 'charge': 0},
    'E': {'hydropathy': -3.5, 'volume': 138.4, 'mass': 147.13, 'charge': -1},
    'G': {'hydropathy': -0.4, 'volume': 60.1, 'mass': 75.07, 'charge': 0},
    'H': {'hydropathy': -3.2, 'volume': 153.2, 'mass': 155.16, 'charge': 0},
    'I': {'hydropathy': 4.5, 'volume': 166.7, 'mass': 131.17, 'charge': 0},
    'L': {'hydropathy': 3.8, 'volume': 166.7, 'mass': 131.17, 'charge': 0},
    'K': {'hydropathy': -3.9, 'volume': 168.6, 'mass': 146.19, 'charge': 1},
    'M': {'hydropathy': 1.9, 'volume': 162.9, 'mass': 149.21, 'charge': 0},
    'F': {'hydropathy': 2.8, 'volume': 189.9, 'mass': 165.19, 'charge': 0},
    'P': {'hydropathy': -1.6, 'volume': 112.7, 'mass': 115.13, 'charge': 0},
    'S': {'hydropathy': -0.8, 'volume': 89.0, 'mass': 105.09, 'charge': 0},
    'T': {'hydropathy': -0.7, 'volume': 116.1, 'mass': 119.12, 'charge': 0},
    'W': {'hydropathy': -0.9, 'volume': 227.8, 'mass': 204.23, 'charge': 0},
    'Y': {'hydropathy': -1.3, 'volume': 193.6, 'mass': 181.19, 'charge': 0},
    'V': {'hydropathy': 4.2, 'volume': 140.0, 'mass': 117.15, 'charge': 0},
}

AA_LIST = list(AA_PROPERTIES.keys())

# Normalization constants
MASS_RANGE = max(p['mass'] for p in AA_PROPERTIES.values()) - min(p['mass'] for p in AA_PROPERTIES.values())
HYDRO_RANGE = max(p['hydropathy'] for p in AA_PROPERTIES.values()) - min(p['hydropathy'] for p in AA_PROPERTIES.values())
VOL_RANGE = max(p['volume'] for p in AA_PROPERTIES.values()) - min(p['volume'] for p in AA_PROPERTIES.values())
MASS_MIN = min(p['mass'] for p in AA_PROPERTIES.values())
HYDRO_MIN = min(p['hydropathy'] for p in AA_PROPERTIES.values())
VOL_MIN = min(p['volume'] for p in AA_PROPERTIES.values())

# ============================================================================
# DISTANCE FUNCTIONS
# ============================================================================


def mass_distance(aa1: str, aa2: str) -> float:
    """Mass-only distance (normalized)."""
    m1 = (AA_PROPERTIES[aa1]['mass'] - MASS_MIN) / MASS_RANGE
    m2 = (AA_PROPERTIES[aa2]['mass'] - MASS_MIN) / MASS_RANGE
    return abs(m1 - m2)


def property_distance(aa1: str, aa2: str) -> float:
    """Traditional property distance (hydropathy + volume)."""
    h1 = (AA_PROPERTIES[aa1]['hydropathy'] - HYDRO_MIN) / HYDRO_RANGE
    h2 = (AA_PROPERTIES[aa2]['hydropathy'] - HYDRO_MIN) / HYDRO_RANGE
    v1 = (AA_PROPERTIES[aa1]['volume'] - VOL_MIN) / VOL_RANGE
    v2 = (AA_PROPERTIES[aa2]['volume'] - VOL_MIN) / VOL_RANGE
    return math.sqrt((h1 - h2)**2 + (v1 - v2)**2)


def mass_hydro_distance(aa1: str, aa2: str) -> float:
    """Combined mass + hydropathy distance."""
    m1 = (AA_PROPERTIES[aa1]['mass'] - MASS_MIN) / MASS_RANGE
    m2 = (AA_PROPERTIES[aa2]['mass'] - MASS_MIN) / MASS_RANGE
    h1 = (AA_PROPERTIES[aa1]['hydropathy'] - HYDRO_MIN) / HYDRO_RANGE
    h2 = (AA_PROPERTIES[aa2]['hydropathy'] - HYDRO_MIN) / HYDRO_RANGE
    return math.sqrt((m1 - m2)**2 + (h1 - h2)**2)


def full_property_distance(aa1: str, aa2: str) -> float:
    """Full property distance (mass + hydropathy + volume)."""
    m1 = (AA_PROPERTIES[aa1]['mass'] - MASS_MIN) / MASS_RANGE
    m2 = (AA_PROPERTIES[aa2]['mass'] - MASS_MIN) / MASS_RANGE
    h1 = (AA_PROPERTIES[aa1]['hydropathy'] - HYDRO_MIN) / HYDRO_RANGE
    h2 = (AA_PROPERTIES[aa2]['hydropathy'] - HYDRO_MIN) / HYDRO_RANGE
    v1 = (AA_PROPERTIES[aa1]['volume'] - VOL_MIN) / VOL_RANGE
    v2 = (AA_PROPERTIES[aa2]['volume'] - VOL_MIN) / VOL_RANGE
    return math.sqrt((m1 - m2)**2 + (h1 - h2)**2 + (v1 - v2)**2)


def load_padic_radii() -> Dict[str, float]:
    """Load p-adic radii from trained embeddings."""
    import torch

    mapping_path = GENETIC_CODE_DIR / "codon_mapping_3adic.json"
    emb_path = GENETIC_CODE_DIR / "v5_11_3_embeddings.pt"

    if not mapping_path.exists() or not emb_path.exists():
        return {}

    with open(mapping_path) as f:
        mapping = json.load(f)

    codon_to_pos = mapping['codon_to_position']
    emb_data = torch.load(emb_path, map_location='cpu', weights_only=False)
    z = emb_data['z_B_hyp'].numpy()

    CODON_TO_AA = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y',
        'TGT': 'C', 'TGC': 'C', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }

    aa_embs = {}
    for codon, pos in codon_to_pos.items():
        aa = CODON_TO_AA.get(codon)
        if aa:
            if aa not in aa_embs:
                aa_embs[aa] = []
            aa_embs[aa].append(z[pos])

    radii = {}
    for aa in aa_embs:
        radii[aa] = np.linalg.norm(np.mean(aa_embs[aa], axis=0))

    return radii


# ============================================================================
# BENCHMARK DATASETS
# ============================================================================

# ΔΔG Dataset (same as before)
DDG_DATASET = [
    ('barnase', 'A', 32, 'G', 1.2), ('barnase', 'I', 88, 'V', 0.8),
    ('barnase', 'L', 89, 'A', 2.5), ('barnase', 'V', 36, 'A', 1.8),
    ('barnase', 'F', 56, 'A', 3.2), ('barnase', 'Y', 78, 'F', 0.4),
    ('barnase', 'W', 71, 'F', 1.9), ('barnase', 'K', 27, 'A', 0.6),
    ('t4_lysozyme', 'L', 99, 'A', 2.8), ('t4_lysozyme', 'I', 3, 'A', 3.1),
    ('t4_lysozyme', 'V', 87, 'A', 2.2), ('t4_lysozyme', 'F', 153, 'A', 4.1),
    ('t4_lysozyme', 'M', 102, 'A', 2.4), ('t4_lysozyme', 'A', 98, 'G', 0.9),
    ('staph_nuclease', 'V', 66, 'A', 2.0), ('staph_nuclease', 'L', 36, 'A', 2.6),
    ('staph_nuclease', 'I', 92, 'V', 0.6), ('staph_nuclease', 'F', 34, 'L', 1.3),
    ('staph_nuclease', 'Y', 91, 'A', 3.5), ('ci2', 'I', 20, 'V', 0.5),
    ('ci2', 'L', 49, 'A', 2.9), ('ci2', 'V', 51, 'A', 1.7),
    ('ci2', 'A', 16, 'G', 1.4), ('ci2', 'F', 50, 'A', 3.8),
    ('ubiquitin', 'I', 44, 'A', 3.0), ('ubiquitin', 'L', 67, 'A', 2.4),
    ('ubiquitin', 'V', 70, 'A', 1.9), ('ubiquitin', 'F', 45, 'A', 3.6),
]

# Evolutionary substitution rates (relative, based on PAM250)
# Higher = more common substitution
SUBSTITUTION_RATES = [
    ('A', 'G', 1.5), ('A', 'S', 1.3), ('A', 'T', 1.1), ('A', 'V', 0.9),
    ('D', 'E', 1.8), ('D', 'N', 1.4), ('F', 'Y', 1.6), ('F', 'W', 0.8),
    ('I', 'V', 1.7), ('I', 'L', 1.5), ('I', 'M', 1.0), ('K', 'R', 1.6),
    ('L', 'M', 1.2), ('L', 'V', 1.1), ('N', 'S', 1.2), ('N', 'D', 1.4),
    ('Q', 'E', 1.3), ('Q', 'K', 0.9), ('S', 'T', 1.4), ('S', 'A', 1.3),
    ('V', 'I', 1.7), ('V', 'L', 1.1), ('Y', 'F', 1.6), ('Y', 'H', 0.7),
    # Rare substitutions
    ('W', 'G', 0.1), ('W', 'A', 0.1), ('C', 'W', 0.2), ('P', 'G', 0.3),
    ('D', 'K', 0.2), ('E', 'K', 0.3), ('R', 'D', 0.2), ('H', 'P', 0.3),
]

# Aggregation propensity changes (based on literature)
# Positive = increases aggregation
AGGREGATION_DATA = [
    ('A', 'V', +0.3), ('A', 'I', +0.5), ('A', 'L', +0.4),  # Ala to hydrophobic
    ('G', 'A', +0.2), ('G', 'V', +0.4),  # Gly to larger
    ('S', 'A', +0.1), ('T', 'A', +0.2),  # Polar to nonpolar
    ('K', 'A', +0.6), ('R', 'A', +0.5),  # Charged to neutral (like citrullination!)
    ('E', 'A', +0.4), ('D', 'A', +0.4),  # Charged to neutral
    ('N', 'A', +0.2), ('Q', 'A', +0.3),  # Polar to nonpolar
    ('F', 'A', -0.3), ('Y', 'A', -0.2),  # Aromatic to small (destabilizes aggregates)
    ('W', 'A', -0.4),  # Large aromatic to small
    ('V', 'A', -0.2), ('I', 'A', -0.3), ('L', 'A', -0.2),  # Hydrophobic to small
    ('P', 'A', +0.1),  # Pro to Ala (removes helix breaker)
]


# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================


def run_ddg_benchmark(padic_radii: Dict[str, float]) -> Dict:
    """Benchmark on ΔΔG prediction."""
    features = {
        'mass': [], 'property_hv': [], 'mass_hydro': [],
        'full_property': [], 'padic_radial': []
    }
    targets = []

    for protein, wt, pos, mut, ddg in DDG_DATASET:
        if wt not in AA_LIST or mut not in AA_LIST:
            continue

        features['mass'].append(mass_distance(wt, mut))
        features['property_hv'].append(property_distance(wt, mut))
        features['mass_hydro'].append(mass_hydro_distance(wt, mut))
        features['full_property'].append(full_property_distance(wt, mut))

        if padic_radii:
            features['padic_radial'].append(abs(padic_radii.get(wt, 0.5) - padic_radii.get(mut, 0.5)))
        else:
            features['padic_radial'].append(0)

        targets.append(ddg)

    targets = np.array(targets)
    results = {}

    for name, feat in features.items():
        feat = np.array(feat)
        if np.std(feat) == 0:
            continue
        r, p = stats.spearmanr(feat, targets)
        results[name] = {'spearman': r, 'p_value': p}

    return results


def run_substitution_benchmark(padic_radii: Dict[str, float]) -> Dict:
    """Benchmark on evolutionary substitution rates."""
    features = {
        'mass': [], 'property_hv': [], 'mass_hydro': [],
        'full_property': [], 'padic_radial': []
    }
    targets = []

    for aa1, aa2, rate in SUBSTITUTION_RATES:
        if aa1 not in AA_LIST or aa2 not in AA_LIST:
            continue

        features['mass'].append(mass_distance(aa1, aa2))
        features['property_hv'].append(property_distance(aa1, aa2))
        features['mass_hydro'].append(mass_hydro_distance(aa1, aa2))
        features['full_property'].append(full_property_distance(aa1, aa2))

        if padic_radii:
            features['padic_radial'].append(abs(padic_radii.get(aa1, 0.5) - padic_radii.get(aa2, 0.5)))
        else:
            features['padic_radial'].append(0)

        targets.append(rate)

    targets = np.array(targets)
    results = {}

    for name, feat in features.items():
        feat = np.array(feat)
        if np.std(feat) == 0:
            continue
        # Negative correlation expected: similar AAs substitute more often
        r, p = stats.spearmanr(feat, targets)
        results[name] = {'spearman': r, 'p_value': p}

    return results


def run_aggregation_benchmark(padic_radii: Dict[str, float]) -> Dict:
    """Benchmark on aggregation propensity changes."""
    features = {
        'mass': [], 'property_hv': [], 'mass_hydro': [],
        'full_property': [], 'padic_radial': []
    }
    targets = []

    for wt, mut, agg_change in AGGREGATION_DATA:
        if wt not in AA_LIST or mut not in AA_LIST:
            continue

        features['mass'].append(mass_distance(wt, mut))
        features['property_hv'].append(property_distance(wt, mut))
        features['mass_hydro'].append(mass_hydro_distance(wt, mut))
        features['full_property'].append(full_property_distance(wt, mut))

        if padic_radii:
            features['padic_radial'].append(abs(padic_radii.get(wt, 0.5) - padic_radii.get(mut, 0.5)))
        else:
            features['padic_radial'].append(0)

        targets.append(agg_change)

    targets = np.array(targets)
    results = {}

    for name, feat in features.items():
        feat = np.array(feat)
        if np.std(feat) == 0:
            continue
        r, p = stats.spearmanr(feat, targets)
        results[name] = {'spearman': r, 'p_value': p}

    return results


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("MASS vs PROPERTY-BASED PREDICTION BENCHMARK")
    print("=" * 70)

    # Load p-adic radii
    print("\nLoading p-adic radii...")
    padic_radii = load_padic_radii()
    print(f"  Loaded radii for {len(padic_radii)} amino acids")

    # Verify mass correlation with p-adic radii
    if padic_radii:
        masses = [AA_PROPERTIES[aa]['mass'] for aa in padic_radii.keys()]
        radii = [padic_radii[aa] for aa in padic_radii.keys()]
        mass_corr, _ = stats.spearmanr(masses, radii)
        print(f"  Confirmed: mass ↔ radius correlation = {mass_corr:.3f}")

    all_results = {}

    # Benchmark 1: ΔΔG prediction
    print("\n" + "-" * 70)
    print("BENCHMARK 1: Protein Stability (ΔΔG)")
    print("-" * 70)
    ddg_results = run_ddg_benchmark(padic_radii)
    all_results['ddg'] = ddg_results

    print(f"\n{'Model':<20} {'Spearman ρ':>12} {'p-value':>12}")
    print("-" * 46)
    for name, res in sorted(ddg_results.items(), key=lambda x: -abs(x[1]['spearman'])):
        sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
        print(f"{name:<20} {res['spearman']:>+12.4f} {res['p_value']:>11.2e} {sig}")

    # Benchmark 2: Evolutionary substitution
    print("\n" + "-" * 70)
    print("BENCHMARK 2: Evolutionary Substitution Rates")
    print("-" * 70)
    sub_results = run_substitution_benchmark(padic_radii)
    all_results['substitution'] = sub_results

    print(f"\n{'Model':<20} {'Spearman ρ':>12} {'p-value':>12}")
    print("-" * 46)
    for name, res in sorted(sub_results.items(), key=lambda x: -abs(x[1]['spearman'])):
        sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
        print(f"{name:<20} {res['spearman']:>+12.4f} {res['p_value']:>11.2e} {sig}")

    # Benchmark 3: Aggregation propensity
    print("\n" + "-" * 70)
    print("BENCHMARK 3: Aggregation Propensity Changes")
    print("-" * 70)
    agg_results = run_aggregation_benchmark(padic_radii)
    all_results['aggregation'] = agg_results

    print(f"\n{'Model':<20} {'Spearman ρ':>12} {'p-value':>12}")
    print("-" * 46)
    for name, res in sorted(agg_results.items(), key=lambda x: -abs(x[1]['spearman'])):
        sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else ""
        print(f"{name:<20} {res['spearman']:>+12.4f} {res['p_value']:>11.2e} {sig}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Best Model per Task")
    print("=" * 70)

    task_winners = {}
    for task, results in all_results.items():
        best = max(results.items(), key=lambda x: abs(x[1]['spearman']))
        task_winners[task] = best[0]
        print(f"\n{task.upper()}: {best[0]} (ρ = {best[1]['spearman']:+.4f})")

    # Count wins
    win_counts = {}
    for winner in task_winners.values():
        win_counts[winner] = win_counts.get(winner, 0) + 1

    print("\n" + "-" * 70)
    print("Win count by model:")
    for model, count in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {model}: {count} tasks")

    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Compare mass vs property_hv across tasks
    mass_better = 0
    property_better = 0
    for task, results in all_results.items():
        if 'mass' in results and 'property_hv' in results:
            if abs(results['mass']['spearman']) > abs(results['property_hv']['spearman']):
                mass_better += 1
            else:
                property_better += 1

    print(f"""
    Mass vs Traditional Property (hydropathy+volume):
      Mass wins: {mass_better} tasks
      Property wins: {property_better} tasks

    Interpretation:
    - Mass correlates with p-adic radius (ρ = {mass_corr:.3f})
    - If mass wins on KINETIC tasks (aggregation), this supports
      the hypothesis that p-adic encodes molecular DYNAMICS
    - If property wins on THERMODYNAMIC tasks (ΔΔG), this supports
      the traditional view that statics dominate
    """)

    # Save results
    output = {
        'benchmarks': all_results,
        'task_winners': task_winners,
        'mass_radius_correlation': mass_corr if padic_radii else None,
    }

    output_file = RESULTS_DIR / "mass_vs_property_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
