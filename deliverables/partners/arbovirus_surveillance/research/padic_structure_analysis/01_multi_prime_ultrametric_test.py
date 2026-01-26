#!/usr/bin/env python3
"""Multi-Prime Ultrametric Structure Analysis for Viral Combinatorial Space.

The core hypothesis: If 3-adic structure (codon grammar) shows ρ ≈ 0 with viral
conservation, perhaps the viral combinatorial space is NOT 3-adic but operates
under a different prime base (5-adic for AA groups? 7-adic? or no p-adic at all).

The DEFINING property of p-adic metrics is the ULTRAMETRIC INEQUALITY:
    d(x,z) ≤ max(d(x,y), d(y,z))   [strong triangle inequality]

This script tests ultrametric compliance across multiple primes to determine
if viral sequence space exhibits p-adic structure for ANY prime.

Mathematical Background:
- 3-adic: Natural for codons (3 nucleotide positions)
- 5-adic: Natural for amino acid groups (5 physicochemical classes)
- 7-adic: 64 codons ≈ 7² (could capture codon table structure differently)
- 2-adic: Binary (purine/pyrimidine) structure

If ultrametric compliance is HIGH for some prime p but LOW for others,
that prime captures the "native" geometry of the space.

Usage:
    python 01_multi_prime_ultrametric_test.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Setup paths
_script_dir = Path(__file__).resolve().parent
_package_root = _script_dir.parents[1]
_project_root = _package_root.parents[3]

sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_package_root))

# Test primes - mathematical candidates for viral space structure
PRIMES_TO_TEST = [2, 3, 5, 7, 11, 13]

# Amino acid groupings (5 physicochemical classes - motivates 5-adic)
AA_GROUPS = {
    'hydrophobic': ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P'],
    'polar': ['S', 'T', 'N', 'Q', 'Y', 'C'],
    'positive': ['K', 'R', 'H'],
    'negative': ['D', 'E'],
    'special': ['G', '*']  # Glycine + stop
}

# Reverse mapping
AA_TO_GROUP_IDX = {}
for idx, (group, aas) in enumerate(AA_GROUPS.items()):
    for aa in aas:
        AA_TO_GROUP_IDX[aa] = idx


@dataclass
class UltrametricTestResult:
    """Results from ultrametric compliance testing."""
    prime: int
    n_triples_tested: int
    n_violations: int
    compliance_rate: float
    mean_violation_magnitude: float
    max_violation_magnitude: float
    # Breakdown by violation type
    violations_by_magnitude: dict = field(default_factory=dict)


@dataclass
class PrimeStructureAnalysis:
    """Complete analysis for one prime."""
    prime: int
    ultrametric_result: UltrametricTestResult
    hierarchy_correlation: float  # Correlation with conservation
    hierarchy_pvalue: float
    distance_distribution: dict  # Statistics of pairwise distances


def padic_valuation(n: int, p: int) -> int:
    """Compute p-adic valuation v_p(n) = max power of p dividing n.

    v_p(0) = infinity (we return a large number)
    v_p(n) = k where p^k | n but p^(k+1) does not divide n
    """
    if n == 0:
        return 100  # Represent infinity
    if p <= 1:
        raise ValueError(f"p must be prime > 1, got {p}")

    val = 0
    while n % p == 0:
        val += 1
        n //= p
    return val


def padic_distance(x: int, y: int, p: int) -> float:
    """Compute p-adic distance d_p(x, y) = p^(-v_p(x-y)).

    Properties:
    - d_p(x,x) = 0
    - d_p(x,y) = d_p(y,x)
    - d_p(x,z) ≤ max(d_p(x,y), d_p(y,z))  [ULTRAMETRIC!]
    """
    diff = abs(x - y)
    if diff == 0:
        return 0.0
    val = padic_valuation(diff, p)
    return float(p) ** (-val)


def sequence_to_padic_index(sequence: str, p: int, method: str = 'positional') -> int:
    """Convert nucleotide sequence to integer for p-adic distance.

    Methods:
    - 'positional': Treat as base-4 number (A=0, T=1, G=2, C=3)
    - 'codon_sum': Sum of codon indices
    - 'aa_group': Sum of amino acid group indices (for 5-adic)
    """
    base_map = {'A': 0, 'T': 1, 'U': 1, 'G': 2, 'C': 3}

    if method == 'positional':
        # Treat sequence as base-4 number
        idx = 0
        for i, base in enumerate(sequence.upper()):
            if base in base_map:
                idx = idx * 4 + base_map[base]
        return idx

    elif method == 'codon_sum':
        # Sum codon indices (each codon 0-63)
        from src.biology.codons import CODON_TO_INDEX
        total = 0
        seq = sequence.upper().replace('U', 'T')
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i+3]
            if codon in CODON_TO_INDEX:
                total += CODON_TO_INDEX[codon]
        return total

    elif method == 'aa_group':
        # Sum of amino acid group indices (0-4)
        from src.biology.codons import GENETIC_CODE
        total = 0
        seq = sequence.upper().replace('U', 'T')
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i:i+3]
            if codon in GENETIC_CODE:
                aa = GENETIC_CODE[codon]
                if aa in AA_TO_GROUP_IDX:
                    total += AA_TO_GROUP_IDX[aa]
        return total

    else:
        raise ValueError(f"Unknown method: {method}")


def test_ultrametric_compliance(
    distances: np.ndarray,
    n_samples: int = 10000,
    seed: int = 42
) -> UltrametricTestResult:
    """Test ultrametric inequality compliance on a distance matrix.

    For ultrametric: d(x,z) ≤ max(d(x,y), d(y,z)) for all x,y,z

    We sample random triples and check violation rate.
    """
    np.random.seed(seed)
    n = distances.shape[0]

    if n < 3:
        return UltrametricTestResult(
            prime=0, n_triples_tested=0, n_violations=0,
            compliance_rate=1.0, mean_violation_magnitude=0.0,
            max_violation_magnitude=0.0
        )

    # Sample random triples
    n_triples = min(n_samples, n * (n-1) * (n-2) // 6)

    violations = []
    n_tested = 0

    # Generate unique triples
    tested_triples = set()
    attempts = 0
    max_attempts = n_triples * 10

    while len(tested_triples) < n_triples and attempts < max_attempts:
        i, j, k = np.random.choice(n, 3, replace=False)
        triple = tuple(sorted([i, j, k]))
        if triple not in tested_triples:
            tested_triples.add(triple)

            # Check ultrametric inequality for all 3 orderings
            d_ij = distances[i, j]
            d_jk = distances[j, k]
            d_ik = distances[i, k]

            # d(i,k) ≤ max(d(i,j), d(j,k))
            max_side = max(d_ij, d_jk)
            if d_ik > max_side + 1e-10:  # Small tolerance for floating point
                violation = d_ik - max_side
                violations.append(violation)

            n_tested += 1

        attempts += 1

    n_violations = len(violations)
    compliance_rate = 1.0 - (n_violations / n_tested) if n_tested > 0 else 1.0

    # Violation magnitude statistics
    if violations:
        violations = np.array(violations)
        mean_viol = float(np.mean(violations))
        max_viol = float(np.max(violations))

        # Binned violations
        bins = [(0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, float('inf'))]
        violations_by_mag = {}
        for lo, hi in bins:
            count = int(np.sum((violations >= lo) & (violations < hi)))
            violations_by_mag[f"{lo}-{hi}"] = count
    else:
        mean_viol = 0.0
        max_viol = 0.0
        violations_by_mag = {}

    return UltrametricTestResult(
        prime=0,  # Set by caller
        n_triples_tested=n_tested,
        n_violations=n_violations,
        compliance_rate=compliance_rate,
        mean_violation_magnitude=mean_viol,
        max_violation_magnitude=max_viol,
        violations_by_magnitude=violations_by_mag
    )


def compute_padic_distance_matrix(
    sequences: list[str],
    prime: int,
    method: str = 'positional'
) -> np.ndarray:
    """Compute pairwise p-adic distances for sequences."""
    n = len(sequences)

    # Convert sequences to indices
    indices = [sequence_to_padic_index(seq, prime, method) for seq in sequences]

    # Compute pairwise distances
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = padic_distance(indices[i], indices[j], prime)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


def analyze_prime_structure(
    sequences: list[str],
    conservation_scores: list[float],
    prime: int,
    method: str = 'positional'
) -> PrimeStructureAnalysis:
    """Complete p-adic structure analysis for one prime."""

    # Compute distance matrix
    dist_matrix = compute_padic_distance_matrix(sequences, prime, method)

    # Test ultrametric compliance
    ultra_result = test_ultrametric_compliance(dist_matrix)
    ultra_result.prime = prime

    # Compute hierarchy (mean distance from "center")
    mean_dists = np.mean(dist_matrix, axis=1)

    # Correlate with conservation
    if len(conservation_scores) == len(sequences):
        corr, pval = stats.spearmanr(mean_dists, conservation_scores)
    else:
        corr, pval = 0.0, 1.0

    # Distance distribution statistics
    upper_tri = dist_matrix[np.triu_indices(len(sequences), k=1)]
    dist_stats = {
        'mean': float(np.mean(upper_tri)),
        'std': float(np.std(upper_tri)),
        'min': float(np.min(upper_tri)) if len(upper_tri) > 0 else 0,
        'max': float(np.max(upper_tri)) if len(upper_tri) > 0 else 0,
        'median': float(np.median(upper_tri)) if len(upper_tri) > 0 else 0,
    }

    return PrimeStructureAnalysis(
        prime=prime,
        ultrametric_result=ultra_result,
        hierarchy_correlation=float(corr) if not np.isnan(corr) else 0.0,
        hierarchy_pvalue=float(pval) if not np.isnan(pval) else 1.0,
        distance_distribution=dist_stats
    )


def load_denv4_data() -> tuple[list[str], list[float]]:
    """Load DENV-4 genome windows and conservation scores."""

    # Load genome sequences
    genome_file = _package_root / "results" / "ml_ready" / "denv4_genome_sequences.json"

    if not genome_file.exists():
        print(f"WARNING: {genome_file} not found, using synthetic data")
        # Generate synthetic test data
        np.random.seed(42)
        sequences = []
        for _ in range(100):
            seq = ''.join(np.random.choice(['A', 'T', 'G', 'C'], 75))
            sequences.append(seq)
        conservation = np.random.random(100).tolist()
        return sequences, conservation

    with open(genome_file) as f:
        data = json.load(f)

    # Extract windows from genomes
    sequences = []
    window_size = 75
    step = 300

    # Handle schema: data['data'][accession] = sequence_string
    if isinstance(data, dict) and 'data' in data:
        genomes = list(data['data'].values())
    elif isinstance(data, dict):
        genomes = list(data.values())
    else:
        genomes = data

    print(f"Found {len(genomes)} genomes")

    for genome_seq in genomes[:50]:  # Limit for computational feasibility
        if isinstance(genome_seq, dict):
            seq = genome_seq.get('sequence', '')
        else:
            seq = str(genome_seq)

        # Extract windows
        for pos in range(0, min(len(seq) - window_size, 3000), step):
            window = seq[pos:pos + window_size]
            if len(window) == window_size and all(b in 'ATGCU' for b in window.upper()):
                sequences.append(window.upper().replace('U', 'T'))

    # Use entropy as conservation proxy (lower entropy = more conserved)
    conservation = []
    for seq in sequences:
        # Simple entropy calculation
        counts = defaultdict(int)
        for b in seq:
            counts[b] += 1
        total = len(seq)
        entropy = -sum((c/total) * np.log2(c/total) for c in counts.values() if c > 0)
        conservation.append(entropy)

    print(f"Loaded {len(sequences)} windows from DENV-4 genomes")
    return sequences, conservation


def main():
    """Run multi-prime ultrametric analysis."""

    print("=" * 70)
    print("MULTI-PRIME ULTRAMETRIC STRUCTURE ANALYSIS")
    print("Testing if viral combinatorial space is p-adic for p ∈ {2,3,5,7,11,13}")
    print("=" * 70)

    # Load data
    print("\nLoading DENV-4 sequence data...")
    sequences, conservation = load_denv4_data()

    # Limit sequences for computational feasibility
    max_seqs = 200
    if len(sequences) > max_seqs:
        np.random.seed(42)
        indices = np.random.choice(len(sequences), max_seqs, replace=False)
        sequences = [sequences[i] for i in indices]
        conservation = [conservation[i] for i in indices]

    print(f"Analyzing {len(sequences)} sequence windows")

    # Test each prime with different encoding methods
    results = {}
    methods = ['positional', 'codon_sum']

    # Add aa_group method only if biology module available
    try:
        from src.biology.codons import GENETIC_CODE
        methods.append('aa_group')
    except ImportError:
        print("WARNING: src.biology.codons not available, skipping aa_group method")

    for method in methods:
        print(f"\n{'='*60}")
        print(f"ENCODING METHOD: {method}")
        print('='*60)

        results[method] = {}

        for prime in PRIMES_TO_TEST:
            print(f"\n  Testing p = {prime}...")

            try:
                analysis = analyze_prime_structure(
                    sequences, conservation, prime, method
                )
                results[method][prime] = analysis

                ur = analysis.ultrametric_result
                print(f"    Ultrametric compliance: {ur.compliance_rate:.4f} "
                      f"({ur.n_violations}/{ur.n_triples_tested} violations)")
                print(f"    Conservation correlation: ρ = {analysis.hierarchy_correlation:.4f} "
                      f"(p = {analysis.hierarchy_pvalue:.4e})")

            except Exception as e:
                print(f"    ERROR: {e}")
                results[method][prime] = None

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: ULTRAMETRIC COMPLIANCE BY PRIME")
    print("=" * 70)
    print(f"{'Method':<15} {'Prime':<8} {'Compliance':<12} {'Cons. ρ':<12} {'p-value':<12}")
    print("-" * 70)

    best_compliance = (None, None, 0.0)
    best_correlation = (None, None, 0.0, 1.0)

    for method in methods:
        for prime in PRIMES_TO_TEST:
            if results[method].get(prime):
                r = results[method][prime]
                comp = r.ultrametric_result.compliance_rate
                corr = r.hierarchy_correlation
                pval = r.hierarchy_pvalue

                print(f"{method:<15} {prime:<8} {comp:<12.4f} {corr:<12.4f} {pval:<12.4e}")

                if comp > best_compliance[2]:
                    best_compliance = (method, prime, comp)

                if abs(corr) > abs(best_correlation[2]) and pval < 0.05:
                    best_correlation = (method, prime, corr, pval)

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(f"\nBest ultrametric compliance: {best_compliance[0]}/{best_compliance[1]}-adic "
          f"({best_compliance[2]:.4f})")

    if best_compliance[2] > 0.95:
        print("  → STRONG p-adic structure detected!")
        print(f"  → Viral space appears to be {best_compliance[1]}-adic")
    elif best_compliance[2] > 0.85:
        print("  → MODERATE p-adic structure (with violations)")
        print("  → Space may be approximately ultrametric")
    else:
        print("  → WEAK/NO p-adic structure")
        print("  → Viral space does NOT appear to be ultrametric for any tested prime")

    if best_correlation[3] < 0.05:
        print(f"\nBest conservation correlation: {best_correlation[0]}/{best_correlation[1]}-adic "
              f"(ρ = {best_correlation[2]:.4f}, p = {best_correlation[3]:.4e})")
    else:
        print("\nNo significant correlation with conservation for any prime")
        print("  → p-adic hierarchy does NOT predict conservation")

    # Save results
    output_dir = _script_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'n_sequences': len(sequences),
        'primes_tested': PRIMES_TO_TEST,
        'methods_tested': methods,
        'best_ultrametric': {
            'method': best_compliance[0],
            'prime': best_compliance[1],
            'compliance': best_compliance[2]
        },
        'best_correlation': {
            'method': best_correlation[0],
            'prime': best_correlation[1],
            'correlation': best_correlation[2],
            'pvalue': best_correlation[3]
        },
        'detailed_results': {}
    }

    for method in methods:
        output_data['detailed_results'][method] = {}
        for prime in PRIMES_TO_TEST:
            if results[method].get(prime):
                r = results[method][prime]
                output_data['detailed_results'][method][str(prime)] = {
                    'ultrametric_compliance': r.ultrametric_result.compliance_rate,
                    'n_violations': r.ultrametric_result.n_violations,
                    'n_triples': r.ultrametric_result.n_triples_tested,
                    'mean_violation': r.ultrametric_result.mean_violation_magnitude,
                    'conservation_correlation': r.hierarchy_correlation,
                    'conservation_pvalue': r.hierarchy_pvalue,
                    'distance_stats': r.distance_distribution
                }

    output_file = output_dir / "multi_prime_ultrametric_results.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return output_data


if __name__ == "__main__":
    main()
