#!/usr/bin/env python3
"""Multi-Prime P-adic Falsification Test.

Conjecture 2: The information evolution space is p-adic but NOT 3-adic.
Maybe 4-adic (nucleotides), 5-adic (nucleotides + stops), or 7-adic.

This test systematically evaluates different primes to find if ANY p-adic
structure correlates with thermodynamic stability.

Primes tested:
- p=2: Binary (purine/pyrimidine)
- p=3: Ternary (codon positions) - ALREADY FALSIFIED
- p=4: Quaternary (4 nucleotides directly) - NOT prime but test anyway
- p=5: Pentary (nucleotides + stop signal)
- p=7: Septenary (deeper periodicity?)
- p=11, p=13: Higher primes for completeness

Usage:
    python multiprime_falsification.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import re

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from src.biology.codons import AMINO_ACID_TO_CODONS
from src.encoders.codon_encoder import AA_PROPERTIES


# =============================================================================
# GENETIC CODE
# =============================================================================

GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


# =============================================================================
# MULTIPLE ENCODING SCHEMES
# =============================================================================


def get_nucleotide_encoding(scheme: str) -> Dict[str, int]:
    """Get nucleotide to integer mapping for different schemes."""

    if scheme == "standard":
        # Standard: T=0, C=1, A=2, G=3 (pyrimidines first, then purines)
        return {"T": 0, "C": 1, "A": 2, "G": 3, "U": 0}

    elif scheme == "chemical":
        # Chemical similarity: purines (A,G) vs pyrimidines (C,T)
        return {"A": 0, "G": 1, "C": 2, "T": 3, "U": 3}

    elif scheme == "hydrogen_bonds":
        # By hydrogen bonds: A-T (2 bonds), G-C (3 bonds)
        return {"A": 0, "T": 1, "G": 2, "C": 3, "U": 1}

    elif scheme == "ring_structure":
        # Single ring (pyrimidines) vs double ring (purines)
        return {"C": 0, "T": 1, "A": 2, "G": 3, "U": 1}

    elif scheme == "5adic":
        # 5-adic: 4 nucleotides + stop as 5th symbol
        # Stop codons (TAA, TAG, TGA) get special treatment
        return {"T": 0, "C": 1, "A": 2, "G": 3, "U": 0, "*": 4}

    else:
        return {"T": 0, "C": 1, "A": 2, "G": 3, "U": 0}


def codon_to_index(codon: str, scheme: str = "standard") -> int:
    """Convert codon to integer index using specified encoding scheme."""
    encoding = get_nucleotide_encoding(scheme)
    codon = codon.upper().replace("U", "T")

    # For 5-adic, treat stop codons specially
    if scheme == "5adic" and GENETIC_CODE.get(codon) == "*":
        # Encode stop codons in a way that uses the 5th symbol
        # Map to indices 64-66 to distinguish from regular codons
        return 64 + ["TAA", "TAG", "TGA"].index(codon)

    idx = 0
    base = 4 if scheme != "5adic" else 5
    for i, nuc in enumerate(codon):
        idx += encoding[nuc] * (base ** (2 - i))
    return idx


# =============================================================================
# GENERALIZED P-ADIC FUNCTIONS
# =============================================================================


def padic_valuation(n: int, p: int) -> int:
    """Compute p-adic valuation v_p(n)."""
    if n == 0:
        return 100  # Infinity
    n = abs(n)
    v = 0
    while n % p == 0:
        v += 1
        n //= p
    return v


def padic_norm(n: int, p: int) -> float:
    """Compute p-adic norm |n|_p = p^(-v_p(n))."""
    if n == 0:
        return 0.0
    v = padic_valuation(n, p)
    if v >= 100:
        return 0.0
    return float(p) ** (-v)


def padic_distance(a: int, b: int, p: int) -> float:
    """Compute p-adic distance d_p(a, b) = |a - b|_p."""
    if a == b:
        return 0.0
    return padic_norm(a - b, p)


def padic_distance_codons(codon1: str, codon2: str, p: int, scheme: str = "standard") -> float:
    """Compute p-adic distance between codons."""
    idx1 = codon_to_index(codon1, scheme)
    idx2 = codon_to_index(codon2, scheme)
    return padic_distance(idx1, idx2, p)


def padic_distance_aa(aa1: str, aa2: str, p: int, scheme: str = "standard") -> float:
    """Compute minimum p-adic distance between amino acids."""
    codons1 = AMINO_ACID_TO_CODONS.get(aa1, [])
    codons2 = AMINO_ACID_TO_CODONS.get(aa2, [])

    if not codons1 or not codons2:
        return 1.0

    min_dist = float('inf')
    for c1 in codons1:
        for c2 in codons2:
            d = padic_distance_codons(c1, c2, p, scheme)
            min_dist = min(min_dist, d)

    return min_dist if min_dist != float('inf') else 1.0


# =============================================================================
# TEST FUNCTIONS
# =============================================================================


@dataclass
class PrimeTestResult:
    """Result of testing one prime."""
    prime: int
    scheme: str
    description: str
    # DDG prediction
    loo_pearson: float
    loo_spearman: float
    loo_pvalue: float
    # Genetic code correlation
    physico_correlation: float
    physico_pvalue: float
    # Assessment
    supports_thermodynamics: bool


def test_prime_for_ddg(
    mutations: List[Tuple[str, str, float]],
    p: int,
    scheme: str = "standard",
) -> Tuple[float, float, float]:
    """Test if p-adic distance with prime p predicts DDG.

    Returns: (loo_pearson, loo_spearman, p_value)
    """
    X = []
    y = []

    for wt_aa, mut_aa, ddg in mutations:
        if wt_aa in AA_PROPERTIES and mut_aa in AA_PROPERTIES:
            dist = padic_distance_aa(wt_aa, mut_aa, p, scheme)
            X.append([dist])
            y.append(ddg)

    X = np.array(X)
    y = np.array(y)

    if len(y) < 10:
        return 0.0, 0.0, 1.0

    # LOO cross-validation
    loo = LeaveOneOut()
    model = Ridge(alpha=1.0)
    y_pred = cross_val_predict(model, X, y, cv=loo)

    r_pearson, p_pearson = pearsonr(y_pred, y)
    r_spearman, p_spearman = spearmanr(y_pred, y)

    return float(r_pearson), float(r_spearman), float(p_spearman)


def test_prime_genetic_code(p: int, scheme: str = "standard") -> Tuple[float, float]:
    """Test if p-adic distance correlates with physicochemical similarity.

    Returns: (spearman_r, p_value)
    """
    amino_acids = list(set(aa for aa in GENETIC_CODE.values() if aa != "*"))

    padic_dists = []
    physico_dists = []

    for i, aa1 in enumerate(amino_acids):
        for aa2 in amino_acids[i+1:]:
            pd = padic_distance_aa(aa1, aa2, p, scheme)

            # Physicochemical distance
            props1 = np.array(AA_PROPERTIES.get(aa1, (0, 0, 0, 0)))
            props2 = np.array(AA_PROPERTIES.get(aa2, (0, 0, 0, 0)))
            pc = float(np.linalg.norm(props1 - props2))

            padic_dists.append(pd)
            physico_dists.append(pc)

    r, p_val = spearmanr(padic_dists, physico_dists)
    return float(r), float(p_val)


def load_s669_mutations() -> List[Tuple[str, str, float]]:
    """Load S669 mutations."""
    data_path = PROJECT_ROOT / "deliverables/partners/jose_colbes/reproducibility/data/S669/S669.csv"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "deliverables/partners/jose_colbes/reproducibility/data/s669.csv"

    mutations = []

    if not data_path.exists():
        print(f"  WARNING: S669 data not found at {data_path}")
        return mutations

    with open(data_path) as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 4:
            try:
                mut_str = parts[2] if len(parts) > 2 else parts[0]
                match = re.match(r'([A-Z])(\d+)([A-Z])', mut_str)
                if match:
                    wt_aa = match.group(1)
                    mut_aa = match.group(3)
                    ddg_idx = 11 if len(parts) > 11 else 5
                    ddg = float(parts[ddg_idx])
                    mutations.append((wt_aa, mut_aa, ddg))
            except (ValueError, IndexError):
                continue

    return mutations


# =============================================================================
# MAIN FALSIFICATION
# =============================================================================


def main():
    print("=" * 70)
    print("MULTI-PRIME P-ADIC FALSIFICATION TEST")
    print("=" * 70)
    print("\nConjecture 2: Information evolution space is p-adic but NOT 3-adic")
    print("Testing primes: 2, 3, 4*, 5, 7, 11, 13")
    print("(*4 is not prime but tests quaternary nucleotide structure)")

    # Load mutations
    print("\nLoading S669 dataset...")
    mutations = load_s669_mutations()
    print(f"  Loaded {len(mutations)} mutations")

    if len(mutations) == 0:
        print("ERROR: No mutations loaded. Cannot proceed.")
        return 1

    # Define primes and schemes to test
    test_configs = [
        (2, "standard", "Binary (mod 2)"),
        (3, "standard", "Ternary (codon positions) - BASELINE"),
        (4, "standard", "Quaternary (4 nucleotides)"),
        (5, "standard", "Pentary (mod 5)"),
        (5, "5adic", "Pentary (nucleotides + stops)"),
        (7, "standard", "Septenary (mod 7)"),
        (7, "chemical", "Septenary (chemical ordering)"),
        (11, "standard", "Undecimal (mod 11)"),
        (13, "standard", "Tridecimal (mod 13)"),
    ]

    results: List[PrimeTestResult] = []

    print("\n" + "=" * 70)
    print("TESTING EACH PRIME")
    print("=" * 70)

    for p, scheme, description in test_configs:
        print(f"\n--- Testing p={p} ({description}) ---")

        # Test DDG prediction
        loo_pearson, loo_spearman, loo_pvalue = test_prime_for_ddg(mutations, p, scheme)
        print(f"  DDG: LOO Pearson={loo_pearson:+.4f}, Spearman={loo_spearman:+.4f}")

        # Test genetic code correlation
        physico_r, physico_p = test_prime_genetic_code(p, scheme)
        print(f"  Genetic code vs physico: r={physico_r:+.4f}")

        # Determine if it supports thermodynamics
        # Criteria: positive correlation with DDG AND with physicochemistry
        supports = (
            loo_pearson > 0.1 and
            loo_pvalue < 0.05 and
            physico_r > 0.1
        )

        results.append(PrimeTestResult(
            prime=p,
            scheme=scheme,
            description=description,
            loo_pearson=loo_pearson,
            loo_spearman=loo_spearman,
            loo_pvalue=loo_pvalue,
            physico_correlation=physico_r,
            physico_pvalue=physico_p,
            supports_thermodynamics=supports,
        ))

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-PRIME FALSIFICATION SUMMARY")
    print("=" * 70)

    print("\n| Prime | Scheme | DDG Pearson | DDG Spearman | Physico r | Supports? |")
    print("|-------|--------|-------------|--------------|-----------|-----------|")

    best_result = None
    best_pearson = -1

    for r in results:
        status = "YES" if r.supports_thermodynamics else "NO"
        print(f"| p={r.prime:2d} | {r.scheme:8s} | {r.loo_pearson:+.4f} | {r.loo_spearman:+.4f} | {r.physico_correlation:+.4f} | {status:9s} |")

        if r.loo_pearson > best_pearson:
            best_pearson = r.loo_pearson
            best_result = r

    # Count how many support thermodynamics
    n_support = sum(1 for r in results if r.supports_thermodynamics)

    print(f"\nPrimes supporting thermodynamics: {n_support}/{len(results)}")

    if best_result:
        print(f"\nBest performing prime: p={best_result.prime} ({best_result.scheme})")
        print(f"  LOO Pearson: {best_result.loo_pearson:+.4f}")
        print(f"  LOO Spearman: {best_result.loo_spearman:+.4f}")

    # Overall assessment
    print("\n" + "=" * 70)
    print("CONJECTURE 2 ASSESSMENT")
    print("=" * 70)

    if n_support == 0:
        print("\nVERDICT: CONJECTURE 2 IS FALSIFIED")
        print("\nInterpretation:")
        print("  NO prime tested shows positive correlation with thermodynamics.")
        print("  The p-adic structure (for any prime tested) does NOT encode")
        print("  thermodynamic stability information.")
        print("\nConclusion:")
        print("  P-adic geometry is fundamentally NOT the right framework")
        print("  for encoding thermodynamic properties of proteins.")
        print("  The information structure of biology may not be p-adic at all")
        print("  for thermodynamics (though it may be for other properties).")

    elif n_support <= 2:
        print("\nVERDICT: CONJECTURE 2 IS WEAKLY SUPPORTED")
        print(f"\nInterpretation:")
        print(f"  {n_support} prime(s) show weak positive correlation.")
        print("  This may be noise or a weak signal worth investigating.")
        print("\nRecommendation:")
        print(f"  Investigate p={best_result.prime} more deeply.")
        print("  Consider alternative encoding schemes.")

    else:
        print("\nVERDICT: CONJECTURE 2 IS SUPPORTED")
        print(f"\nInterpretation:")
        print(f"  {n_support} primes show positive thermodynamic correlation!")
        print("  The p-adic structure may encode thermodynamics under")
        print("  the right prime choice.")
        print(f"\nBest candidate: p={best_result.prime}")

    # Deeper analysis of patterns
    print("\n" + "=" * 70)
    print("PATTERN ANALYSIS")
    print("=" * 70)

    # Check if Spearman is consistently negative (anti-predictive)
    negative_spearman = sum(1 for r in results if r.loo_spearman < 0)
    print(f"\nPrimes with NEGATIVE Spearman (anti-predictive): {negative_spearman}/{len(results)}")

    if negative_spearman >= len(results) - 1:
        print("\nCRITICAL FINDING:")
        print("  Almost ALL primes show NEGATIVE Spearman correlation!")
        print("  This is not random - it's a systematic pattern.")
        print("\n  Possible interpretation:")
        print("  P-adic 'closeness' in the genetic code is ANTI-correlated")
        print("  with thermodynamic stability across ALL primes tested.")
        print("  This suggests the genetic code evolved for ERROR TOLERANCE")
        print("  (maximizing p-adic diversity) rather than thermodynamic optimization.")

    # Check genetic code correlations
    positive_physico = sum(1 for r in results if r.physico_correlation > 0.1)
    negative_physico = sum(1 for r in results if r.physico_correlation < -0.05)

    print(f"\nGenetic code vs physicochemistry:")
    print(f"  Positive correlation (>0.1): {positive_physico}/{len(results)}")
    print(f"  Negative correlation (<-0.05): {negative_physico}/{len(results)}")

    if negative_physico > positive_physico:
        print("\n  The genetic code is ANTI-optimized for p-adic similarity")
        print("  across multiple primes - this is a robust finding.")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
