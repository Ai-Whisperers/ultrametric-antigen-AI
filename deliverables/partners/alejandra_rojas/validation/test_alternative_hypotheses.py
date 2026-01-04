# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Alternative Hypotheses Falsification for DENV-4 Variability.

This script tests three alternative hypotheses for why DENV-4 shows
10-30x higher variability than other Dengue serotypes:

Hypothesis 1: SEROTYPE COMPETITION
    DENV-4 is outcompeted by DENV-1/2 in endemic regions, maintaining
    low prevalence despite global distribution, allowing neutral drift.

    FALSIFICATION: If DENV-4 shows similar proportions to other serotypes
    when co-circulating in the same country/year, competition is falsified.

Hypothesis 2: POLYMERASE FIDELITY
    DENV-4 NS5 (RNA polymerase) has lower replication fidelity, causing
    higher mutation rates independent of selection pressure.

    FALSIFICATION: If NS5 is equally conserved across all serotypes,
    the polymerase difference hypothesis is falsified.

Hypothesis 3: IMMUNE EVASION TRADE-OFF
    DENV-4 evolved toward immune evasion over transmission efficiency,
    resulting in higher E protein variability at epitope sites.

    FALSIFICATION: If E protein variability is similar across serotypes,
    immune evasion trade-off is falsified.

Usage:
    python validation/test_alternative_hypotheses.py
    python validation/test_alternative_hypotheses.py --use-cache
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import datetime

import numpy as np
from scipy.stats import spearmanr, chi2_contingency, mannwhitneyu

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deliverables"))

# Try to import BioPython
try:
    from Bio import Entrez, SeqIO
    from Bio.Seq import Seq
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("WARNING: BioPython not available")


@dataclass
class HypothesisResult:
    """Result of a hypothesis falsification test."""
    hypothesis: str
    prediction: str
    observation: str
    falsified: bool
    evidence: str
    statistics: dict


# =============================================================================
# GENE REGIONS IN DENGUE GENOME
# =============================================================================

# Approximate positions (bp) in Dengue genome for key genes
# Based on NC_001477 (DENV-1) reference
DENGUE_GENE_REGIONS = {
    "5UTR": (1, 94),
    "C": (95, 436),          # Capsid
    "prM": (437, 934),       # Pre-membrane
    "E": (935, 2419),        # Envelope (main target for antibodies)
    "NS1": (2420, 3475),     # Non-structural 1
    "NS2A": (3476, 4129),
    "NS2B": (4130, 4519),
    "NS3": (4520, 6376),     # Protease/helicase
    "NS4A": (6377, 6826),
    "NS4B": (6827, 7569),
    "NS5": (7570, 10269),    # RNA-dependent RNA polymerase (RdRp)
    "3UTR": (10270, 10735),
}

# Known epitope regions in E protein (relative to E protein start)
# Based on Dengue virus structural studies
E_PROTEIN_EPITOPES = {
    "domain_I": (1, 130),    # Domain I
    "domain_II": (131, 295), # Domain II (fusion loop, cross-reactive)
    "fusion_loop": (98, 111),
    "domain_III": (296, 400), # Domain III (serotype-specific, neutralizing)
    "lateral_ridge": (310, 340),
    "A_strand": (380, 395),
}


def compute_shannon_entropy(column: list[str]) -> float:
    """Compute Shannon entropy for a sequence column."""
    valid = [b for b in column if b in "ACGT"]
    if len(valid) == 0:
        return 2.0

    counts = Counter(valid)
    total = len(valid)
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def extract_gene_region(
    sequences: list[tuple[str, str]],
    gene_start: int,
    gene_end: int,
) -> list[str]:
    """Extract a gene region from multiple sequences."""
    regions = []
    for accession, seq in sequences:
        if len(seq) >= gene_end:
            regions.append(seq[gene_start:gene_end])
    return regions


def compute_region_entropy(regions: list[str]) -> tuple[float, list[float]]:
    """Compute mean and per-position entropy for a set of aligned regions."""
    if not regions:
        return 0.0, []

    min_len = min(len(r) for r in regions)
    entropies = []

    for i in range(min_len):
        column = [r[i] for r in regions if i < len(r)]
        entropy = compute_shannon_entropy(column)
        entropies.append(entropy)

    return np.mean(entropies) if entropies else 0.0, entropies


# =============================================================================
# HYPOTHESIS 1: SEROTYPE COMPETITION
# =============================================================================

def test_serotype_competition(metadata_cache: Path) -> HypothesisResult:
    """Test if DENV-4 is outcompeted when co-circulating with other serotypes.

    Prediction: In countries where multiple serotypes co-circulate,
    DENV-4 should have significantly lower proportion than expected (25%).

    Falsification: If DENV-4 proportion is ≥15% in co-endemic settings,
    competition alone cannot explain low prevalence.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1: SEROTYPE COMPETITION")
    print("=" * 70)

    # Load metadata
    if not metadata_cache.exists():
        return HypothesisResult(
            hypothesis="Serotype competition",
            prediction="DENV-4 <15% in co-endemic regions",
            observation="No data available",
            falsified=False,
            evidence="Cannot test - missing metadata",
            statistics={},
        )

    with open(metadata_cache) as f:
        metadata = json.load(f)

    # Count serotypes by country
    country_serotypes = defaultdict(lambda: defaultdict(int))

    for serotype, data in metadata.items():
        for country in data.get("countries", []):
            country_serotypes[country][serotype] += data.get("country_counts", {}).get(country, 1)

    # Find co-endemic countries (≥2 serotypes present)
    co_endemic = {}
    for country, serotypes in country_serotypes.items():
        if len(serotypes) >= 2:
            total = sum(serotypes.values())
            proportions = {s: c/total for s, c in serotypes.items()}
            co_endemic[country] = {
                "serotypes": dict(serotypes),
                "proportions": proportions,
                "total": total,
            }

    print(f"\nCo-endemic countries found: {len(co_endemic)}")

    if not co_endemic:
        # Use overall proportions as fallback
        totals = {s: d.get("total_genomes", 0) for s, d in metadata.items()}
        grand_total = sum(totals.values())
        if grand_total > 0:
            proportions = {s: t/grand_total for s, t in totals.items()}
            denv4_prop = proportions.get("DENV-4", 0)

            observation = f"DENV-4 is {denv4_prop:.1%} of all sequences (global)"
            falsified = denv4_prop >= 0.05  # 5% threshold for global

            if falsified:
                evidence = f"FALSIFIED: DENV-4 at {denv4_prop:.1%} is above 5% minimum"
            else:
                evidence = f"SUPPORTED: DENV-4 at {denv4_prop:.1%} shows competition effects"

            return HypothesisResult(
                hypothesis="Serotype competition",
                prediction="DENV-4 <5% globally if outcompeted",
                observation=observation,
                falsified=falsified,
                evidence=evidence,
                statistics={"global_proportions": proportions},
            )

    # Analyze DENV-4 proportions in co-endemic countries
    denv4_proportions = []
    for country, data in co_endemic.items():
        if "DENV-4" in data["proportions"]:
            denv4_proportions.append(data["proportions"]["DENV-4"])
            print(f"  {country}: DENV-4 = {data['proportions']['DENV-4']:.1%} "
                  f"(n={data['serotypes'].get('DENV-4', 0)}/{data['total']})")

    if denv4_proportions:
        mean_prop = np.mean(denv4_proportions)
        max_prop = np.max(denv4_proportions)

        observation = f"DENV-4 mean={mean_prop:.1%}, max={max_prop:.1%} in co-endemic regions"

        # Falsified if DENV-4 achieves ≥15% anywhere (can compete when present)
        falsified = max_prop >= 0.15

        if falsified:
            evidence = f"FALSIFIED: DENV-4 reaches {max_prop:.1%} in some regions (can compete)"
        else:
            evidence = f"SUPPORTED: DENV-4 never exceeds 15% (consistently outcompeted)"
    else:
        observation = "DENV-4 absent from all co-endemic countries in sample"
        falsified = False
        evidence = "SUPPORTED: Complete competitive exclusion observed"

    print(f"\n  Observation: {observation}")
    print(f"  Result: {evidence}")

    return HypothesisResult(
        hypothesis="Serotype competition",
        prediction="DENV-4 <15% in co-endemic regions",
        observation=observation,
        falsified=falsified,
        evidence=evidence,
        statistics={
            "co_endemic_countries": len(co_endemic),
            "denv4_proportions": denv4_proportions,
            "mean_proportion": float(np.mean(denv4_proportions)) if denv4_proportions else 0,
        },
    )


# =============================================================================
# HYPOTHESIS 2: POLYMERASE FIDELITY
# =============================================================================

def test_polymerase_fidelity(strain_cache: Path) -> HypothesisResult:
    """Test if DENV-4 NS5 (polymerase) shows different conservation pattern.

    Prediction: If DENV-4 has lower polymerase fidelity, the NS5 gene itself
    should show higher variability (more errors accumulated in polymerase).

    Falsification: If NS5 conservation is similar across all serotypes,
    polymerase fidelity cannot explain the differential mutation rates.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: POLYMERASE FIDELITY (NS5 CONSERVATION)")
    print("=" * 70)

    if not strain_cache.exists():
        return HypothesisResult(
            hypothesis="Polymerase fidelity",
            prediction="DENV-4 NS5 should be more variable",
            observation="No data available",
            falsified=False,
            evidence="Cannot test - missing strain data",
            statistics={},
        )

    with open(strain_cache) as f:
        all_sequences = json.load(f)

    ns5_start, ns5_end = DENGUE_GENE_REGIONS["NS5"]
    serotype_ns5_entropy = {}

    print(f"\nAnalyzing NS5 region ({ns5_start}-{ns5_end} bp)...")

    for serotype in ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]:
        sequences = all_sequences.get(serotype, [])

        # Convert from list of lists if needed
        if sequences and isinstance(sequences[0], list):
            sequences = [(s[0], s[1]) for s in sequences]

        if len(sequences) < 5:
            print(f"  {serotype}: Insufficient sequences ({len(sequences)})")
            continue

        # Extract NS5 regions
        ns5_regions = extract_gene_region(sequences, ns5_start, ns5_end)

        if len(ns5_regions) < 5:
            continue

        mean_entropy, position_entropies = compute_region_entropy(ns5_regions)
        serotype_ns5_entropy[serotype] = {
            "mean_entropy": mean_entropy,
            "max_entropy": max(position_entropies) if position_entropies else 0,
            "n_variable": sum(1 for e in position_entropies if e > 0.5),
            "n_positions": len(position_entropies),
            "n_sequences": len(ns5_regions),
        }

        print(f"  {serotype}: mean_entropy={mean_entropy:.4f}, "
              f"variable_positions={serotype_ns5_entropy[serotype]['n_variable']}/{len(position_entropies)}")

    if len(serotype_ns5_entropy) < 2:
        return HypothesisResult(
            hypothesis="Polymerase fidelity",
            prediction="DENV-4 NS5 should be more variable",
            observation="Insufficient data",
            falsified=False,
            evidence="Cannot test - need at least 2 serotypes",
            statistics={},
        )

    # Compare DENV-4 NS5 entropy to others
    denv4_entropy = serotype_ns5_entropy.get("DENV-4", {}).get("mean_entropy", 0)
    other_entropies = [
        data["mean_entropy"]
        for s, data in serotype_ns5_entropy.items()
        if s != "DENV-4"
    ]
    other_mean = np.mean(other_entropies) if other_entropies else 0

    # Calculate ratio
    if other_mean > 0:
        entropy_ratio = denv4_entropy / other_mean
    else:
        entropy_ratio = 1.0

    observation = f"DENV-4 NS5 entropy={denv4_entropy:.4f}, others={other_mean:.4f} (ratio={entropy_ratio:.2f}x)"

    # Falsified if DENV-4 NS5 is NOT significantly more variable (ratio < 2x)
    # If polymerase has lower fidelity, it should accumulate more errors in itself too
    falsified = entropy_ratio < 2.0

    if falsified:
        evidence = f"FALSIFIED: NS5 entropy ratio only {entropy_ratio:.2f}x (not >2x expected for fidelity difference)"
    else:
        evidence = f"SUPPORTED: NS5 shows {entropy_ratio:.2f}x higher entropy in DENV-4"

    print(f"\n  Observation: {observation}")
    print(f"  Result: {evidence}")

    return HypothesisResult(
        hypothesis="Polymerase fidelity",
        prediction="DENV-4 NS5 should be >2x more variable",
        observation=observation,
        falsified=falsified,
        evidence=evidence,
        statistics={
            "serotype_entropies": {k: v["mean_entropy"] for k, v in serotype_ns5_entropy.items()},
            "entropy_ratio": entropy_ratio,
            "denv4_variable_positions": serotype_ns5_entropy.get("DENV-4", {}).get("n_variable", 0),
        },
    )


# =============================================================================
# HYPOTHESIS 3: IMMUNE EVASION TRADE-OFF
# =============================================================================

def test_immune_evasion(strain_cache: Path) -> HypothesisResult:
    """Test if DENV-4 shows elevated E protein/epitope variability.

    Prediction: If DENV-4 evolved for immune evasion, the E protein
    (especially epitope regions) should show higher variability than NS5.

    Falsification: If E protein variability follows the same pattern as
    other genes (proportional to overall serotype variability), immune
    evasion-specific selection is falsified.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: IMMUNE EVASION TRADE-OFF (E PROTEIN)")
    print("=" * 70)

    if not strain_cache.exists():
        return HypothesisResult(
            hypothesis="Immune evasion trade-off",
            prediction="DENV-4 E protein should be disproportionately variable",
            observation="No data available",
            falsified=False,
            evidence="Cannot test - missing strain data",
            statistics={},
        )

    with open(strain_cache) as f:
        all_sequences = json.load(f)

    e_start, e_end = DENGUE_GENE_REGIONS["E"]
    ns5_start, ns5_end = DENGUE_GENE_REGIONS["NS5"]

    serotype_ratios = {}

    print(f"\nComparing E protein ({e_start}-{e_end} bp) vs NS5 ({ns5_start}-{ns5_end} bp)...")

    for serotype in ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]:
        sequences = all_sequences.get(serotype, [])

        # Convert from list of lists if needed
        if sequences and isinstance(sequences[0], list):
            sequences = [(s[0], s[1]) for s in sequences]

        if len(sequences) < 5:
            continue

        # Extract E and NS5 regions
        e_regions = extract_gene_region(sequences, e_start, e_end)
        ns5_regions = extract_gene_region(sequences, ns5_start, ns5_end)

        if len(e_regions) < 5 or len(ns5_regions) < 5:
            continue

        e_entropy, _ = compute_region_entropy(e_regions)
        ns5_entropy, _ = compute_region_entropy(ns5_regions)

        # E/NS5 ratio: if immune evasion, E should be disproportionately high
        if ns5_entropy > 0:
            e_ns5_ratio = e_entropy / ns5_entropy
        else:
            e_ns5_ratio = 1.0

        serotype_ratios[serotype] = {
            "e_entropy": e_entropy,
            "ns5_entropy": ns5_entropy,
            "e_ns5_ratio": e_ns5_ratio,
        }

        print(f"  {serotype}: E={e_entropy:.4f}, NS5={ns5_entropy:.4f}, ratio={e_ns5_ratio:.2f}")

    if len(serotype_ratios) < 2:
        return HypothesisResult(
            hypothesis="Immune evasion trade-off",
            prediction="DENV-4 E/NS5 ratio should be elevated",
            observation="Insufficient data",
            falsified=False,
            evidence="Cannot test - need at least 2 serotypes",
            statistics={},
        )

    # Compare DENV-4 E/NS5 ratio to others
    denv4_ratio = serotype_ratios.get("DENV-4", {}).get("e_ns5_ratio", 1.0)
    other_ratios = [
        data["e_ns5_ratio"]
        for s, data in serotype_ratios.items()
        if s != "DENV-4"
    ]
    other_mean_ratio = np.mean(other_ratios) if other_ratios else 1.0

    # Meta-ratio: is DENV-4's E/NS5 ratio elevated compared to other serotypes?
    if other_mean_ratio > 0:
        meta_ratio = denv4_ratio / other_mean_ratio
    else:
        meta_ratio = 1.0

    observation = (f"DENV-4 E/NS5 ratio={denv4_ratio:.2f}, others mean={other_mean_ratio:.2f} "
                  f"(meta-ratio={meta_ratio:.2f}x)")

    # Falsified if DENV-4 E/NS5 ratio is NOT significantly elevated (meta-ratio < 1.5x)
    falsified = meta_ratio < 1.5

    if falsified:
        evidence = f"FALSIFIED: E protein variability proportional to overall (meta-ratio={meta_ratio:.2f}x <1.5x)"
    else:
        evidence = f"SUPPORTED: E protein disproportionately variable in DENV-4 ({meta_ratio:.2f}x)"

    print(f"\n  Observation: {observation}")
    print(f"  Result: {evidence}")

    return HypothesisResult(
        hypothesis="Immune evasion trade-off",
        prediction="DENV-4 E/NS5 ratio should be >1.5x elevated",
        observation=observation,
        falsified=falsified,
        evidence=evidence,
        statistics={
            "serotype_ratios": {k: v["e_ns5_ratio"] for k, v in serotype_ratios.items()},
            "meta_ratio": meta_ratio,
            "denv4_e_entropy": serotype_ratios.get("DENV-4", {}).get("e_entropy", 0),
            "denv4_ns5_entropy": serotype_ratios.get("DENV-4", {}).get("ns5_entropy", 0),
        },
    )


# =============================================================================
# BONUS: dN/dS RATIO ANALYSIS
# =============================================================================

def compute_simple_dnds_proxy(regions: list[str]) -> dict:
    """Compute a simple proxy for dN/dS using codon position variability.

    In coding sequences:
    - 3rd codon positions are often synonymous (degenerate)
    - 1st and 2nd positions are usually non-synonymous

    If entropy(pos3) >> entropy(pos1,2), this suggests purifying selection
    (dN/dS < 1). If entropy is similar, suggests neutral evolution (dN/dS ~ 1).
    """
    if not regions or len(regions) < 5:
        return {"pos1_entropy": 0, "pos2_entropy": 0, "pos3_entropy": 0, "ratio": 1.0}

    min_len = min(len(r) for r in regions)
    # Ensure we work with complete codons
    codon_count = min_len // 3

    pos1_entropies = []
    pos2_entropies = []
    pos3_entropies = []

    for codon_idx in range(codon_count):
        base_idx = codon_idx * 3

        col1 = [r[base_idx] for r in regions if base_idx < len(r)]
        col2 = [r[base_idx + 1] for r in regions if base_idx + 1 < len(r)]
        col3 = [r[base_idx + 2] for r in regions if base_idx + 2 < len(r)]

        pos1_entropies.append(compute_shannon_entropy(col1))
        pos2_entropies.append(compute_shannon_entropy(col2))
        pos3_entropies.append(compute_shannon_entropy(col3))

    mean_pos1 = np.mean(pos1_entropies) if pos1_entropies else 0
    mean_pos2 = np.mean(pos2_entropies) if pos2_entropies else 0
    mean_pos3 = np.mean(pos3_entropies) if pos3_entropies else 0

    mean_pos12 = (mean_pos1 + mean_pos2) / 2

    # Proxy ratio: pos3/pos12
    # High ratio = more synonymous variation = purifying selection (dN/dS < 1)
    # Low ratio = similar variation = more neutral or positive selection
    if mean_pos12 > 0:
        ratio = mean_pos3 / mean_pos12
    else:
        ratio = 1.0

    return {
        "pos1_entropy": mean_pos1,
        "pos2_entropy": mean_pos2,
        "pos3_entropy": mean_pos3,
        "pos12_entropy": mean_pos12,
        "pos3_pos12_ratio": ratio,
    }


def test_neutral_evolution(strain_cache: Path) -> HypothesisResult:
    """Bonus test: Check if DENV-4 shows more neutral evolution (dN/dS ~ 1).

    If DENV-4 is under relaxed selection (due to competition, immune evasion,
    or polymerase errors), it should show more similar entropy at all codon
    positions (pos3/pos12 ratio closer to 1).
    """
    print("\n" + "=" * 70)
    print("BONUS: NEUTRAL EVOLUTION (dN/dS PROXY)")
    print("=" * 70)

    if not strain_cache.exists():
        return HypothesisResult(
            hypothesis="Neutral evolution",
            prediction="DENV-4 should have pos3/pos12 ratio closer to 1",
            observation="No data available",
            falsified=False,
            evidence="Cannot test - missing strain data",
            statistics={},
        )

    with open(strain_cache) as f:
        all_sequences = json.load(f)

    # Use E protein as representative coding region
    e_start, e_end = DENGUE_GENE_REGIONS["E"]
    serotype_dnds = {}

    print(f"\nAnalyzing codon position entropy in E protein...")

    for serotype in ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]:
        sequences = all_sequences.get(serotype, [])

        if sequences and isinstance(sequences[0], list):
            sequences = [(s[0], s[1]) for s in sequences]

        if len(sequences) < 5:
            continue

        e_regions = extract_gene_region(sequences, e_start, e_end)

        if len(e_regions) < 5:
            continue

        dnds = compute_simple_dnds_proxy(e_regions)
        serotype_dnds[serotype] = dnds

        print(f"  {serotype}: pos1={dnds['pos1_entropy']:.4f}, pos2={dnds['pos2_entropy']:.4f}, "
              f"pos3={dnds['pos3_entropy']:.4f}, ratio={dnds['pos3_pos12_ratio']:.2f}")

    if len(serotype_dnds) < 2:
        return HypothesisResult(
            hypothesis="Neutral evolution",
            prediction="DENV-4 pos3/pos12 ratio should approach 1",
            observation="Insufficient data",
            falsified=False,
            evidence="Cannot test",
            statistics={},
        )

    denv4_ratio = serotype_dnds.get("DENV-4", {}).get("pos3_pos12_ratio", 1.0)
    other_ratios = [
        data["pos3_pos12_ratio"]
        for s, data in serotype_dnds.items()
        if s != "DENV-4"
    ]
    other_mean = np.mean(other_ratios) if other_ratios else 1.0

    # Distance from 1.0 (neutral)
    denv4_neutrality = abs(denv4_ratio - 1.0)
    other_neutrality = abs(other_mean - 1.0)

    observation = (f"DENV-4 pos3/pos12={denv4_ratio:.2f} (dist from neutral: {denv4_neutrality:.2f}), "
                  f"others={other_mean:.2f} (dist: {other_neutrality:.2f})")

    # More neutral if closer to 1.0
    more_neutral = denv4_neutrality < other_neutrality

    if more_neutral:
        evidence = f"SUPPORTED: DENV-4 shows more neutral evolution (closer to dN/dS=1)"
    else:
        evidence = f"FALSIFIED: DENV-4 not more neutral than other serotypes"

    print(f"\n  Observation: {observation}")
    print(f"  Result: {evidence}")

    return HypothesisResult(
        hypothesis="Neutral evolution (dN/dS proxy)",
        prediction="DENV-4 codon position ratio should be closer to 1.0",
        observation=observation,
        falsified=not more_neutral,
        evidence=evidence,
        statistics={
            "serotype_ratios": {k: v["pos3_pos12_ratio"] for k, v in serotype_dnds.items()},
            "denv4_neutrality_distance": denv4_neutrality,
            "other_neutrality_distance": other_neutrality,
        },
    )


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_all_falsifications(cache_dir: Path) -> dict:
    """Run all three alternative hypothesis falsification tests."""

    print("=" * 70)
    print("ALTERNATIVE HYPOTHESES FALSIFICATION")
    print("=" * 70)
    print("\nTesting why DENV-4 shows 10-30x higher variability than other serotypes")
    print()

    metadata_cache = cache_dir / "dengue_metadata.json"
    strain_cache = cache_dir / "dengue_strains.json"

    results = []

    # Test 1: Serotype competition
    result1 = test_serotype_competition(metadata_cache)
    results.append(result1)

    # Test 2: Polymerase fidelity
    result2 = test_polymerase_fidelity(strain_cache)
    results.append(result2)

    # Test 3: Immune evasion
    result3 = test_immune_evasion(strain_cache)
    results.append(result3)

    # Bonus: Neutral evolution
    result4 = test_neutral_evolution(strain_cache)
    results.append(result4)

    # Summary
    print("\n" + "=" * 70)
    print("FALSIFICATION SUMMARY")
    print("=" * 70)
    print()

    falsified_count = sum(1 for r in results if r.falsified)
    supported_count = len(results) - falsified_count

    for i, r in enumerate(results, 1):
        status = "FALSIFIED" if r.falsified else "SUPPORTED"
        print(f"  {i}. {r.hypothesis}: {status}")

    print()
    print(f"  Falsified: {falsified_count}/{len(results)}")
    print(f"  Supported: {supported_count}/{len(results)}")
    print()

    # Interpretation
    if falsified_count == len(results):
        interpretation = "ALL HYPOTHESES FALSIFIED: Unknown mechanism"
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║  ALL HYPOTHESES FALSIFIED                                    ║")
        print("  ║  DENV-4 variability mechanism remains unknown                ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
    elif supported_count >= 2:
        supported = [r.hypothesis for r in results if not r.falsified]
        interpretation = f"MULTIPLE MECHANISMS: {', '.join(supported)}"
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║  MULTIPLE MECHANISMS SUPPORTED                               ║")
        print(f"  ║  {', '.join(supported)[:56]:56} ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
    elif supported_count == 1:
        supported = [r.hypothesis for r in results if not r.falsified][0]
        interpretation = f"SINGLE MECHANISM: {supported}"
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║  SINGLE MECHANISM IDENTIFIED                                 ║")
        print(f"  ║  {supported:56} ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
    else:
        interpretation = "INCONCLUSIVE"

    print()

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "hypotheses_tested": len(results),
        "falsified": falsified_count,
        "supported": supported_count,
        "interpretation": interpretation,
        "results": [asdict(r) for r in results],
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Alternative Hypotheses Falsification for DENV-4 Variability"
    )
    parser.add_argument("--use-cache", action="store_true", help="Use cached data")

    args = parser.parse_args()

    validation_dir = Path(__file__).parent
    cache_dir = validation_dir.parent / "data"

    results = run_all_falsifications(cache_dir)

    # Save results
    output_path = validation_dir / "alternative_hypotheses_results.json"

    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        return obj

    with open(output_path, "w") as f:
        json.dump(convert_types(results), f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
