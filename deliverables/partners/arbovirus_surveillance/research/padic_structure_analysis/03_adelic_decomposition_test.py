#!/usr/bin/env python3
"""Adelic Decomposition Test: Is viral space a product of multiple p-adic structures?

Mathematical Background:
The ADELES are the restricted product of all p-adic completions:
    A_Q = R × ∏_p Q_p

Perhaps viral evolution doesn't operate in a single p-adic space, but in an
ADELIC space where different primes capture different aspects:
    - 2-adic: purine/pyrimidine transitions (binary)
    - 3-adic: codon position effects
    - 5-adic: amino acid group properties
    - 7-adic: codon table structure (64 ≈ 9 × 7)

This script tests the ADELIC HYPOTHESIS:
    viral_distance ≈ ∏_p d_p(x,y)^{w_p}  (weighted product)

Or in log space:
    log(viral_distance) ≈ Σ_p w_p × log(d_p(x,y))

If the adelic model fits better than any single prime, the viral space is
GENUINELY MULTI-PRIME, requiring an output module that combines multiple
p-adic projections.

Usage:
    python 03_adelic_decomposition_test.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

# Setup paths
_script_dir = Path(__file__).resolve().parent
_package_root = _script_dir.parents[1]
_project_root = _package_root.parents[3]

sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_package_root))

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19]


@dataclass
class AdelicAnalysis:
    """Results from adelic decomposition."""
    # Model comparison
    best_single_prime: int
    best_single_r2: float
    adelic_r2: float
    improvement_over_single: float

    # Adelic weights (importance of each prime)
    adelic_weights: dict  # prime -> weight

    # Statistical tests
    f_statistic: float  # F-test for nested models
    f_pvalue: float
    aic_single: float
    aic_adelic: float

    # Interpretation
    is_genuinely_adelic: bool
    dominant_primes: list  # Primes with significant weights


def padic_valuation(n: int, p: int) -> int:
    """Compute p-adic valuation."""
    if n == 0:
        return 50  # Large but not infinite for numerical stability
    val = 0
    while n % p == 0:
        val += 1
        n //= p
    return val


def padic_log_distance(x: int, y: int, p: int) -> float:
    """Compute log of p-adic distance (for linear combination).

    log(d_p(x,y)) = -v_p(x-y) × log(p)
    """
    diff = abs(x - y)
    if diff == 0:
        return -50 * np.log(p)  # Very negative (close points)
    val = padic_valuation(diff, p)
    return -val * np.log(p)


def compute_sequence_distance(seq1: str, seq2: str) -> float:
    """Compute normalized Hamming distance."""
    if len(seq1) != len(seq2):
        min_len = min(len(seq1), len(seq2))
        seq1, seq2 = seq1[:min_len], seq2[:min_len]
    if len(seq1) == 0:
        return 0.0
    return sum(1 for a, b in zip(seq1.upper(), seq2.upper()) if a != b) / len(seq1)


def sequence_to_index(seq: str) -> int:
    """Convert sequence to integer (base-4 encoding)."""
    base_map = {'A': 0, 'T': 1, 'U': 1, 'G': 2, 'C': 3}
    idx = 0
    for base in seq.upper():
        if base in base_map:
            idx = idx * 4 + base_map[base]
    return idx


def compute_aic(n: int, k: int, rss: float) -> float:
    """Compute Akaike Information Criterion.

    AIC = n × log(RSS/n) + 2k
    where k = number of parameters
    """
    if rss <= 0:
        return float('inf')
    return n * np.log(rss / n) + 2 * k


def test_adelic_structure(
    sequences: list[str],
    primes: list[int] = PRIMES
) -> AdelicAnalysis:
    """Test if viral distance follows adelic (multi-prime) structure."""

    n = len(sequences)
    print(f"Testing adelic structure for {n} sequences...")

    # Compute pairwise indices
    indices = [sequence_to_index(seq) for seq in sequences]

    # Get upper triangular pairs
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_pairs = len(pairs)
    print(f"  Analyzing {n_pairs} sequence pairs")

    # Compute viral distances
    viral_dists = np.array([
        compute_sequence_distance(sequences[i], sequences[j])
        for i, j in pairs
    ])

    # Compute log p-adic distances for each prime
    log_padic = {}
    for p in primes:
        log_padic[p] = np.array([
            padic_log_distance(indices[i], indices[j], p)
            for i, j in pairs
        ])

    # Handle log(0) for viral distances
    viral_log = np.log(viral_dists + 1e-10)

    # === SINGLE PRIME MODELS ===
    print("\n  Testing single-prime models...")
    single_results = {}

    for p in primes:
        X = log_padic[p].reshape(-1, 1)
        model = Ridge(alpha=0.1)
        model.fit(X, viral_log)
        pred = model.predict(X)
        r2 = r2_score(viral_log, pred)
        rss = np.sum((viral_log - pred) ** 2)
        aic = compute_aic(n_pairs, 2, rss)  # 2 params: slope + intercept
        single_results[p] = {'r2': r2, 'aic': aic, 'rss': rss}
        print(f"    {p}-adic: R² = {r2:.4f}, AIC = {aic:.2f}")

    best_single = max(single_results.items(), key=lambda x: x[1]['r2'])
    best_prime, best_stats = best_single
    print(f"\n  Best single prime: {best_prime}-adic (R² = {best_stats['r2']:.4f})")

    # === ADELIC MODEL (all primes) ===
    print("\n  Testing adelic (multi-prime) model...")

    X_adelic = np.column_stack([log_padic[p] for p in primes])

    # Use Lasso for automatic feature selection (sparse weights)
    adelic_model = Lasso(alpha=0.01, max_iter=10000)
    adelic_model.fit(X_adelic, viral_log)

    adelic_pred = adelic_model.predict(X_adelic)
    adelic_r2 = r2_score(viral_log, adelic_pred)
    adelic_rss = np.sum((viral_log - adelic_pred) ** 2)
    adelic_aic = compute_aic(n_pairs, len(primes) + 1, adelic_rss)

    print(f"  Adelic R² = {adelic_r2:.4f}, AIC = {adelic_aic:.2f}")

    # Extract weights
    adelic_weights = {p: float(w) for p, w in zip(primes, adelic_model.coef_)}
    print("\n  Adelic weights (importance of each prime):")
    for p, w in sorted(adelic_weights.items(), key=lambda x: abs(x[1]), reverse=True):
        if abs(w) > 0.01:
            print(f"    {p}-adic: {w:.4f}")

    # === STATISTICAL COMPARISON ===
    print("\n  Statistical comparison...")

    # F-test for nested models
    rss_reduced = best_stats['rss']  # Single prime
    rss_full = adelic_rss  # Adelic
    df_reduced = n_pairs - 2
    df_full = n_pairs - len(primes) - 1

    if rss_full < rss_reduced and df_full > 0:
        f_stat = ((rss_reduced - rss_full) / (df_reduced - df_full)) / (rss_full / df_full)
        f_pvalue = 1 - stats.f.cdf(f_stat, df_reduced - df_full, df_full)
    else:
        f_stat = 0.0
        f_pvalue = 1.0

    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  F p-value: {f_pvalue:.4e}")
    print(f"  AIC improvement: {best_stats['aic'] - adelic_aic:.2f}")

    # === INTERPRETATION ===
    improvement = adelic_r2 - best_stats['r2']
    is_adelic = (improvement > 0.05) and (f_pvalue < 0.05) and (adelic_aic < best_stats['aic'])

    # Dominant primes (significant weights)
    dominant = [p for p, w in adelic_weights.items() if abs(w) > 0.05]

    return AdelicAnalysis(
        best_single_prime=best_prime,
        best_single_r2=best_stats['r2'],
        adelic_r2=adelic_r2,
        improvement_over_single=improvement,
        adelic_weights=adelic_weights,
        f_statistic=f_stat,
        f_pvalue=f_pvalue,
        aic_single=best_stats['aic'],
        aic_adelic=adelic_aic,
        is_genuinely_adelic=is_adelic,
        dominant_primes=dominant
    )


def test_chinese_remainder_structure(sequences: list[str]) -> dict:
    """Test if sequence space factors via Chinese Remainder Theorem.

    CRT: Z/mnZ ≅ Z/mZ × Z/nZ when gcd(m,n) = 1

    For codons: 64 = 2^6, so factors as (Z/2Z)^6
    But also: 64 = 8 × 8 = 4 × 16, etc.

    Test if different factorizations capture different aspects.
    """
    print("\n  Testing Chinese Remainder Theorem factorizations...")

    indices = [sequence_to_index(seq) for seq in sequences]
    n = len(sequences)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    viral_dists = np.array([
        compute_sequence_distance(sequences[i], sequences[j])
        for i, j in pairs
    ])

    # Different modular representations
    factorizations = {
        'mod_2': 2,
        'mod_3': 3,
        'mod_4': 4,
        'mod_5': 5,
        'mod_7': 7,
        'mod_8': 8,
        'mod_9': 9,
        'mod_16': 16,
        'mod_64': 64,
    }

    results = {}
    for name, mod in factorizations.items():
        # Compute modular distance
        mod_dists = np.array([
            abs((indices[i] % mod) - (indices[j] % mod)) / mod
            for i, j in pairs
        ])

        # Correlation with viral distance
        rho, pval = stats.spearmanr(viral_dists, mod_dists)
        results[name] = {
            'modulus': mod,
            'spearman_rho': float(rho) if not np.isnan(rho) else 0.0,
            'pvalue': float(pval) if not np.isnan(pval) else 1.0
        }
        print(f"    {name}: ρ = {rho:.4f}, p = {pval:.4e}")

    return results


def load_sequences() -> list[str]:
    """Load DENV-4 sequence windows."""
    genome_file = _package_root / "results" / "ml_ready" / "denv4_genome_sequences.json"

    if not genome_file.exists():
        np.random.seed(42)
        return [''.join(np.random.choice(['A', 'T', 'G', 'C'], 75)) for _ in range(80)]

    with open(genome_file) as f:
        data = json.load(f)

    sequences = []
    window_size = 75

    # Handle schema: data['data'][accession] = sequence_string
    if isinstance(data, dict) and 'data' in data:
        genomes = list(data['data'].values())
    elif isinstance(data, dict):
        genomes = list(data.values())
    else:
        genomes = data

    print(f"Found {len(genomes)} genomes")

    for genome_seq in genomes[:25]:
        if isinstance(genome_seq, dict):
            seq = genome_seq.get('sequence', '')
        else:
            seq = str(genome_seq)

        for pos in range(0, min(len(seq) - window_size, 1500), 150):
            window = seq[pos:pos + window_size]
            if len(window) == window_size and all(b in 'ATGCU' for b in window.upper()):
                sequences.append(window.upper().replace('U', 'T'))

    return sequences[:120]


def main():
    """Run adelic decomposition analysis."""

    print("=" * 70)
    print("ADELIC DECOMPOSITION ANALYSIS")
    print("Is viral space a product of multiple p-adic spaces?")
    print("=" * 70)

    # Load data
    print("\nLoading sequences...")
    sequences = load_sequences()
    print(f"Loaded {len(sequences)} sequence windows")

    # Test adelic structure
    print("\n" + "-" * 70)
    result = test_adelic_structure(sequences)

    # Test CRT factorizations
    print("\n" + "-" * 70)
    crt_results = test_chinese_remainder_structure(sequences)

    # Final interpretation
    print("\n" + "=" * 70)
    print("FINAL INTERPRETATION")
    print("=" * 70)

    if result.is_genuinely_adelic:
        print(f"""
ADELIC STRUCTURE DETECTED!

The viral combinatorial space appears to be GENUINELY MULTI-PRIME:
    - Single best prime ({result.best_single_prime}-adic): R² = {result.best_single_r2:.4f}
    - Adelic model: R² = {result.adelic_r2:.4f}
    - Improvement: +{result.improvement_over_single:.4f} (p = {result.f_pvalue:.4e})

Dominant primes: {result.dominant_primes}

IMPLICATION FOR OUTPUT MODULE:
    An output module SHOULD combine multiple p-adic projections:

    adjusted_distance = f(d_2, d_3, d_5, d_7, ...)

    with learned weights:
""")
        for p in result.dominant_primes:
            w = result.adelic_weights[p]
            print(f"        {p}-adic: weight = {w:.4f}")

    else:
        if result.best_single_r2 > 0.3:
            print(f"""
SINGLE-PRIME STRUCTURE (NOT ADELIC)

The viral space is best described by {result.best_single_prime}-adic geometry alone.
    - R² = {result.best_single_r2:.4f}
    - Adelic model does NOT significantly improve (Δ = {result.improvement_over_single:.4f})

IMPLICATION:
    Use {result.best_single_prime}-adic distance as the primary metric.
    Multi-prime combination is NOT justified by the data.
""")
        else:
            print(f"""
NO P-ADIC STRUCTURE DETECTED

Neither single-prime nor adelic models explain viral evolutionary distance:
    - Best single prime ({result.best_single_prime}-adic): R² = {result.best_single_r2:.4f}
    - Adelic model: R² = {result.adelic_r2:.4f}

HONEST CONCLUSION:
    The viral combinatorial space does NOT exhibit p-adic geometry
    for any prime or combination of primes tested.

    This is a NEGATIVE RESULT that must be reported honestly.

    The TrainableCodonEncoder captures CODON GRAMMAR (universal)
    but viral EVOLUTIONARY DISTANCE operates under different mathematics.

RECOMMENDATIONS:
    1. Do NOT build an output module claiming p-adic projection adjustment
    2. Acknowledge the orthogonality as a scientific finding
    3. Consider alternative geometries (Riemannian, hyperbolic, etc.)
    4. Or accept that two independent signals (grammar + evolution) is valuable
""")

    # Save results
    output_dir = _script_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'n_sequences': len(sequences),
        'primes_tested': PRIMES,
        'adelic_analysis': {
            'best_single_prime': result.best_single_prime,
            'best_single_r2': result.best_single_r2,
            'adelic_r2': result.adelic_r2,
            'improvement': result.improvement_over_single,
            'f_statistic': result.f_statistic,
            'f_pvalue': result.f_pvalue,
            'aic_single': result.aic_single,
            'aic_adelic': result.aic_adelic,
            'is_genuinely_adelic': result.is_genuinely_adelic,
            'dominant_primes': result.dominant_primes,
            'weights': result.adelic_weights
        },
        'crt_factorizations': crt_results,
        'conclusion': 'adelic' if result.is_genuinely_adelic else (
            f'{result.best_single_prime}-adic' if result.best_single_r2 > 0.3 else 'no_padic_structure'
        )
    }

    output_file = output_dir / "adelic_decomposition_results.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return result


if __name__ == "__main__":
    main()
