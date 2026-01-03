#!/usr/bin/env python3
"""TEGB Conjecture Falsification Test.

Thermodynamical Effects Generalization Bridge (TEGB) Conjecture:
    Codon encoding → Mutational accessibility → Codon optimization
    → Translation kinetics → Thermodynamical effects

The conjecture states:
1. Evolution respects thermodynamics (not the reverse)
2. Individual codons don't encode thermodynamics directly
3. BUT full sequences that evolved successfully DO encode thermodynamic
   properties implicitly in their p-adic structure

Falsification Strategy:
- Test each link in the TEGB chain independently
- Use ONLY pure p-adic mathematics (no learned weights)
- No physicochemical features (to isolate p-adic contribution)
- Provide clear falsification criteria for each link

Usage:
    python tegb_falsification.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from src.core.padic_math import (
    padic_distance,
    padic_valuation,
    padic_digits,
    DEFAULT_P,
)
from src.biology.codons import (
    AMINO_ACID_TO_CODONS,
    CODON_TO_INDEX,
    INDEX_TO_CODON,
)
from src.encoders.codon_encoder import AA_PROPERTIES


# =============================================================================
# GENETIC CODE (Standard)
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

# Nucleotide to digit mapping (for pure 3-adic structure)
NUC_TO_DIGIT = {"T": 0, "C": 1, "A": 2, "G": 3}


# =============================================================================
# PURE P-ADIC FUNCTIONS (No learned weights)
# =============================================================================


def codon_to_3adic_index(codon: str) -> int:
    """Convert codon to index using 3-adic base encoding.

    Uses base-4 encoding of nucleotides, then maps to 0-63 range.
    This is the PURE mathematical mapping without any learned weights.
    """
    codon = codon.upper().replace("U", "T")
    idx = 0
    for i, nuc in enumerate(codon):
        idx += NUC_TO_DIGIT[nuc] * (4 ** (2 - i))
    return idx


def pure_padic_distance_codons(codon1: str, codon2: str, p: int = 3) -> float:
    """Compute pure p-adic distance between two codons.

    Based on codon index difference, no learned weights.
    """
    idx1 = codon_to_3adic_index(codon1)
    idx2 = codon_to_3adic_index(codon2)
    return padic_distance(idx1, idx2, p)


def pure_padic_distance_aa(aa1: str, aa2: str, p: int = 3) -> float:
    """Compute minimum pure p-adic distance between amino acids.

    Takes minimum over all synonymous codon pairs.
    """
    codons1 = AMINO_ACID_TO_CODONS.get(aa1, [])
    codons2 = AMINO_ACID_TO_CODONS.get(aa2, [])

    if not codons1 or not codons2:
        return 1.0

    min_dist = 1.0
    for c1 in codons1:
        for c2 in codons2:
            d = pure_padic_distance_codons(c1, c2, p)
            min_dist = min(min_dist, d)

    return min_dist


def hamming_distance_codons(codon1: str, codon2: str) -> int:
    """Count nucleotide differences between codons."""
    return sum(1 for a, b in zip(codon1.upper(), codon2.upper()) if a != b)


def physicochemical_distance_aa(aa1: str, aa2: str) -> float:
    """Compute Euclidean distance in physicochemical space."""
    props1 = np.array(AA_PROPERTIES.get(aa1, (0, 0, 0, 0)))
    props2 = np.array(AA_PROPERTIES.get(aa2, (0, 0, 0, 0)))
    return float(np.linalg.norm(props1 - props2))


# =============================================================================
# TEGB LINK TESTS
# =============================================================================


@dataclass
class LinkTestResult:
    """Result of testing one TEGB link."""
    link_name: str
    hypothesis: str
    test_statistic: float
    p_value: float
    effect_size: float
    interpretation: str
    falsified: bool


def test_link1_genetic_code_optimization() -> LinkTestResult:
    """Test Link 1: Genetic code p-adic structure → Physicochemical similarity.

    Hypothesis: Codons that are p-adically close encode amino acids with
    similar physicochemical properties.

    Falsification: No correlation between p-adic and physicochemical distances.
    """
    print("\n" + "=" * 70)
    print("LINK 1: Genetic Code Optimization")
    print("=" * 70)
    print("\nHypothesis: P-adically close codons encode similar amino acids")

    padic_dists = []
    physico_dists = []

    # Get all amino acid pairs
    amino_acids = list(set(aa for aa in GENETIC_CODE.values() if aa != "*"))

    for i, aa1 in enumerate(amino_acids):
        for aa2 in amino_acids[i+1:]:
            pd = pure_padic_distance_aa(aa1, aa2)
            pc = physicochemical_distance_aa(aa1, aa2)
            padic_dists.append(pd)
            physico_dists.append(pc)

    # Test correlation
    r, p = spearmanr(padic_dists, physico_dists)

    print(f"\n  N pairs: {len(padic_dists)}")
    print(f"  Spearman r: {r:.4f}")
    print(f"  P-value: {p:.2e}")

    # Effect size interpretation
    if abs(r) < 0.1:
        effect = "negligible"
    elif abs(r) < 0.3:
        effect = "small"
    elif abs(r) < 0.5:
        effect = "moderate"
    else:
        effect = "large"

    # Falsification criterion: r < 0.1 means no relationship
    falsified = abs(r) < 0.1 and p > 0.05

    if r < 0:
        interpretation = "COUNTER-INTUITIVE: Close codons encode DISSIMILAR amino acids!"
    elif falsified:
        interpretation = "FALSIFIED: No relationship between p-adic and physicochemical"
    else:
        interpretation = f"SUPPORTED: {effect} positive correlation"

    print(f"  Effect size: {effect}")
    print(f"  Interpretation: {interpretation}")

    return LinkTestResult(
        link_name="Genetic Code Optimization",
        hypothesis="P-adic close → Physico similar",
        test_statistic=r,
        p_value=p,
        effect_size=abs(r),
        interpretation=interpretation,
        falsified=falsified
    )


def test_link2_mutational_accessibility() -> LinkTestResult:
    """Test Link 2: P-adic distance → Mutational accessibility.

    Hypothesis: Amino acids that are p-adically close require fewer
    nucleotide changes (lower Hamming distance).

    Falsification: No correlation between p-adic distance and mutation steps.
    """
    print("\n" + "=" * 70)
    print("LINK 2: Mutational Accessibility")
    print("=" * 70)
    print("\nHypothesis: P-adically close AAs require fewer nucleotide changes")

    padic_dists = []
    min_hamming = []

    amino_acids = list(set(aa for aa in GENETIC_CODE.values() if aa != "*"))

    for i, aa1 in enumerate(amino_acids):
        codons1 = AMINO_ACID_TO_CODONS.get(aa1, [])
        for aa2 in amino_acids[i+1:]:
            codons2 = AMINO_ACID_TO_CODONS.get(aa2, [])

            # P-adic distance
            pd = pure_padic_distance_aa(aa1, aa2)
            padic_dists.append(pd)

            # Minimum Hamming distance (mutational accessibility)
            min_h = 3  # Maximum possible
            for c1 in codons1:
                for c2 in codons2:
                    h = hamming_distance_codons(c1, c2)
                    min_h = min(min_h, h)
            min_hamming.append(min_h)

    # Test correlation
    r, p = spearmanr(padic_dists, min_hamming)

    print(f"\n  N pairs: {len(padic_dists)}")
    print(f"  Spearman r: {r:.4f}")
    print(f"  P-value: {p:.2e}")

    # This SHOULD be positive: close in p-adic → few nucleotide changes
    # (by construction of the genetic code)

    if r < 0:
        interpretation = "UNEXPECTED: P-adic close requires MORE mutations!"
        falsified = True
    elif r > 0.3 and p < 0.05:
        interpretation = "STRONGLY SUPPORTED: P-adic encodes mutational accessibility"
        falsified = False
    elif r > 0.1 and p < 0.05:
        interpretation = "WEAKLY SUPPORTED: Partial relationship"
        falsified = False
    else:
        interpretation = "FALSIFIED: No clear relationship"
        falsified = True

    print(f"  Interpretation: {interpretation}")

    return LinkTestResult(
        link_name="Mutational Accessibility",
        hypothesis="P-adic close → Few nucleotide changes",
        test_statistic=r,
        p_value=p,
        effect_size=abs(r),
        interpretation=interpretation,
        falsified=falsified
    )


def test_link3_pure_padic_ddg() -> LinkTestResult:
    """Test Link 3: Pure p-adic → DDG prediction (without physicochemistry).

    Hypothesis: P-adic distance ALONE (no physicochemical features)
    can predict DDG.

    Falsification: Pure p-adic has zero or negative predictive power.
    """
    print("\n" + "=" * 70)
    print("LINK 3: Pure P-adic → DDG (No Physicochemistry)")
    print("=" * 70)
    print("\nHypothesis: Pure p-adic distance predicts DDG WITHOUT physicochemistry")

    # Load S669 data
    data_path = PROJECT_ROOT / "deliverables/partners/jose_colbes/reproducibility/data/S669/S669.csv"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "deliverables/partners/jose_colbes/reproducibility/data/s669.csv"

    if not data_path.exists():
        print("  ERROR: S669 data not found")
        return LinkTestResult(
            link_name="Pure P-adic DDG",
            hypothesis="P-adic alone → DDG",
            test_statistic=0,
            p_value=1,
            effect_size=0,
            interpretation="DATA NOT FOUND",
            falsified=True
        )

    # Parse mutations
    mutations = []
    with open(data_path) as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 4:
            try:
                # Parse mutation format (e.g., "S11A")
                import re
                mut_str = parts[2] if len(parts) > 2 else parts[0]
                match = re.match(r'([A-Z])(\d+)([A-Z])', mut_str)
                if match:
                    wt_aa = match.group(1)
                    mut_aa = match.group(3)
                    # DDG is typically in column 11 (Experimental_DDG_dir)
                    ddg_idx = 11 if len(parts) > 11 else 5
                    ddg = float(parts[ddg_idx])
                    mutations.append((wt_aa, mut_aa, ddg))
            except (ValueError, IndexError):
                continue

    print(f"  Loaded {len(mutations)} mutations")

    # Extract ONLY p-adic features (no physicochemistry!)
    X = []
    y = []

    for wt_aa, mut_aa, ddg in mutations:
        if wt_aa in AA_PROPERTIES and mut_aa in AA_PROPERTIES:
            # Pure p-adic distance only
            padic_dist = pure_padic_distance_aa(wt_aa, mut_aa)
            X.append([padic_dist])
            y.append(ddg)

    X = np.array(X)
    y = np.array(y)

    print(f"  Valid mutations: {len(y)}")
    print(f"  Features: PURE P-adic distance only (1 feature)")

    # Leave-one-out cross-validation
    loo = LeaveOneOut()
    model = Ridge(alpha=1.0)
    y_pred = cross_val_predict(model, X, y, cv=loo)

    r_pearson, p_pearson = pearsonr(y_pred, y)
    r_spearman, p_spearman = spearmanr(y_pred, y)

    print(f"\n  LOO Pearson r: {r_pearson:.4f} (p={p_pearson:.2e})")
    print(f"  LOO Spearman r: {r_spearman:.4f} (p={p_spearman:.2e})")

    # Falsification criterion: R² < 0 or not significant
    if r_pearson < 0:
        interpretation = "FALSIFIED: Pure p-adic is ANTI-predictive!"
        falsified = True
    elif r_pearson < 0.1 or p_pearson > 0.05:
        interpretation = "FALSIFIED: Pure p-adic has no significant predictive power"
        falsified = True
    elif r_pearson < 0.3:
        interpretation = "WEAK SIGNAL: Small predictive power detected"
        falsified = False
    else:
        interpretation = "SUPPORTED: P-adic alone has meaningful predictive power"
        falsified = False

    print(f"  Interpretation: {interpretation}")

    return LinkTestResult(
        link_name="Pure P-adic DDG",
        hypothesis="P-adic alone → DDG",
        test_statistic=r_spearman,
        p_value=p_spearman,
        effect_size=r_pearson ** 2,  # R² as effect size
        interpretation=interpretation,
        falsified=falsified
    )


def test_link4_padic_vs_physico_unique() -> LinkTestResult:
    """Test Link 4: Does p-adic add UNIQUE information beyond physicochemistry?

    Hypothesis: P-adic provides additional predictive signal not captured
    by physicochemical features alone.

    Falsification: P-adic contribution is fully redundant with physicochemistry.
    """
    print("\n" + "=" * 70)
    print("LINK 4: P-adic Unique Contribution (Beyond Physicochemistry)")
    print("=" * 70)
    print("\nHypothesis: P-adic adds information beyond physicochemical features")

    # Load S669 data
    data_path = PROJECT_ROOT / "deliverables/partners/jose_colbes/reproducibility/data/S669/S669.csv"
    if not data_path.exists():
        data_path = PROJECT_ROOT / "deliverables/partners/jose_colbes/reproducibility/data/s669.csv"

    if not data_path.exists():
        return LinkTestResult(
            link_name="P-adic Unique Contribution",
            hypothesis="P-adic + Physico > Physico alone",
            test_statistic=0,
            p_value=1,
            effect_size=0,
            interpretation="DATA NOT FOUND",
            falsified=True
        )

    # Parse mutations
    mutations = []
    with open(data_path) as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 4:
            try:
                import re
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

    print(f"  Loaded {len(mutations)} mutations")

    # Extract features
    X_padic = []
    X_physico = []
    X_combined = []
    y = []

    for wt_aa, mut_aa, ddg in mutations:
        if wt_aa in AA_PROPERTIES and mut_aa in AA_PROPERTIES:
            # P-adic features
            padic_dist = pure_padic_distance_aa(wt_aa, mut_aa)

            # Physicochemical features
            wt_props = np.array(AA_PROPERTIES[wt_aa])
            mut_props = np.array(AA_PROPERTIES[mut_aa])
            delta_hydro = mut_props[0] - wt_props[0]
            delta_charge = abs(mut_props[1] - wt_props[1])
            delta_size = mut_props[2] - wt_props[2]
            delta_polarity = mut_props[3] - wt_props[3]

            X_padic.append([padic_dist])
            X_physico.append([delta_hydro, delta_charge, delta_size, delta_polarity])
            X_combined.append([padic_dist, delta_hydro, delta_charge, delta_size, delta_polarity])
            y.append(ddg)

    X_padic = np.array(X_padic)
    X_physico = np.array(X_physico)
    X_combined = np.array(X_combined)
    y = np.array(y)

    # LOO CV for each model
    loo = LeaveOneOut()
    model = Ridge(alpha=1.0)

    # Model 1: Physicochemistry only
    y_pred_physico = cross_val_predict(model, X_physico, y, cv=loo)
    r2_physico = pearsonr(y_pred_physico, y)[0] ** 2

    # Model 2: P-adic + Physicochemistry
    y_pred_combined = cross_val_predict(model, X_combined, y, cv=loo)
    r2_combined = pearsonr(y_pred_combined, y)[0] ** 2

    # Model 3: P-adic only (for reference)
    y_pred_padic = cross_val_predict(model, X_padic, y, cv=loo)
    r2_padic = pearsonr(y_pred_padic, y)[0] ** 2

    delta_r2 = r2_combined - r2_physico

    print(f"\n  LOO R² (P-adic only):     {r2_padic:.4f}")
    print(f"  LOO R² (Physico only):    {r2_physico:.4f}")
    print(f"  LOO R² (Combined):        {r2_combined:.4f}")
    print(f"  Delta R² (P-adic adds):   {delta_r2:+.4f}")

    # Falsification criterion: P-adic adds < 1% improvement
    if delta_r2 < 0:
        interpretation = "FALSIFIED: P-adic HURTS prediction when combined!"
        falsified = True
    elif delta_r2 < 0.01:
        interpretation = "FALSIFIED: P-adic adds < 1% improvement (redundant)"
        falsified = True
    elif delta_r2 < 0.05:
        interpretation = "MARGINAL: P-adic adds 1-5% improvement"
        falsified = False
    else:
        interpretation = "SUPPORTED: P-adic adds > 5% unique information"
        falsified = False

    print(f"  Interpretation: {interpretation}")

    return LinkTestResult(
        link_name="P-adic Unique Contribution",
        hypothesis="P-adic + Physico > Physico alone",
        test_statistic=delta_r2,
        p_value=0,  # Not a statistical test per se
        effect_size=delta_r2,
        interpretation=interpretation,
        falsified=falsified
    )


# =============================================================================
# MAIN FALSIFICATION PROTOCOL
# =============================================================================


def main():
    print("=" * 70)
    print("TEGB CONJECTURE FALSIFICATION TEST")
    print("=" * 70)
    print("\nThermodynamical Effects Generalization Bridge (TEGB):")
    print("  Codon encoding → Mutational accessibility → Codon optimization")
    print("  → Translation kinetics → Thermodynamical effects")
    print("\nUsing ONLY pure p-adic mathematics (no learned weights)")

    results = []

    # Test each link
    results.append(test_link1_genetic_code_optimization())
    results.append(test_link2_mutational_accessibility())
    results.append(test_link3_pure_padic_ddg())
    results.append(test_link4_padic_vs_physico_unique())

    # Summary
    print("\n" + "=" * 70)
    print("TEGB FALSIFICATION SUMMARY")
    print("=" * 70)

    print("\n| Link | Hypothesis | r | Falsified? |")
    print("|------|------------|---|------------|")

    n_falsified = 0
    for r in results:
        status = "YES" if r.falsified else "NO"
        if r.falsified:
            n_falsified += 1
        print(f"| {r.link_name[:20]:20s} | {r.hypothesis[:25]:25s} | {r.test_statistic:+.3f} | {status} |")

    print(f"\nLinks falsified: {n_falsified}/{len(results)}")

    # Overall TEGB assessment
    print("\n" + "=" * 70)
    print("OVERALL TEGB ASSESSMENT")
    print("=" * 70)

    if n_falsified >= 3:
        print("\nVERDICT: TEGB CONJECTURE IS LARGELY FALSIFIED")
        print("\nInterpretation:")
        print("  The p-adic structure of the genetic code does NOT encode")
        print("  thermodynamic properties, even implicitly through evolution.")
        print("  The codon→thermodynamics bridge is broken at multiple links.")
        print("\nRecommendation:")
        print("  P-adic features are NOT useful for DDG prediction.")
        print("  Focus on physicochemical properties or structural features.")
    elif n_falsified >= 2:
        print("\nVERDICT: TEGB CONJECTURE IS PARTIALLY FALSIFIED")
        print("\nInterpretation:")
        print("  Some links in the TEGB chain hold, but critical ones fail.")
        print("  P-adic structure has limited relevance to thermodynamics.")
        print("\nRecommendation:")
        print("  P-adic may be useful for non-thermodynamic properties")
        print("  (e.g., evolutionary distance, mutational accessibility).")
    elif n_falsified == 1:
        print("\nVERDICT: TEGB CONJECTURE IS MOSTLY SUPPORTED")
        print("\nInterpretation:")
        print("  Most TEGB links hold, with one weak point.")
        print("  P-adic may encode thermodynamic information indirectly.")
        print("\nRecommendation:")
        print("  Investigate the failed link more deeply.")
        print("  Consider hybrid p-adic + structural approaches.")
    else:
        print("\nVERDICT: TEGB CONJECTURE IS SUPPORTED")
        print("\nInterpretation:")
        print("  All TEGB links hold! P-adic structure encodes")
        print("  thermodynamic information through evolutionary selection.")
        print("\nRecommendation:")
        print("  Develop full-sequence p-adic features for DDG prediction.")
        print("  The p-adic approach has scientific merit.")

    # Detailed recommendations based on specific failures
    print("\n" + "=" * 70)
    print("SPECIFIC RECOMMENDATIONS")
    print("=" * 70)

    for r in results:
        if r.falsified:
            print(f"\n{r.link_name}:")
            print(f"  Problem: {r.interpretation}")
            if "Genetic Code" in r.link_name:
                print("  Implication: Genetic code is NOT optimized for p-adic similarity")
            elif "Mutational" in r.link_name:
                print("  Implication: P-adic structure doesn't reflect mutation steps")
            elif "Pure P-adic" in r.link_name:
                print("  Implication: P-adic alone has no predictive power for DDG")
            elif "Unique" in r.link_name:
                print("  Implication: P-adic is redundant with physicochemistry")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
