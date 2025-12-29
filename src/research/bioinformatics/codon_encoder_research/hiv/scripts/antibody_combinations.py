"""
Antibody Combination Analysis for HIV bnAbs.

This module implements algorithms for selecting optimal broadly neutralizing
antibody (bnAb) combinations based on CATNAP neutralization data and
hyperbolic geometry analysis.

Key features:
1. Complementarity scoring based on non-overlapping epitopes
2. Breadth optimization using greedy/ILP algorithms
3. Synergy detection from neutralization curves
4. Resistance barrier estimation for combinations

Based on papers:
- Kong et al. 2015: bnAb combination strategies
- Wagh et al. 2016: Optimal antibody cocktails
- Caskey et al. 2017: Combination immunotherapy

Author: Research Team
Date: December 2025
"""

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd


# Epitope classification for bnAbs
BNAB_EPITOPES = {
    # CD4 binding site antibodies
    "VRC01": {"epitope": "CD4bs", "position_range": (124, 477), "glycan_dependent": False},
    "VRC03": {"epitope": "CD4bs", "position_range": (124, 477), "glycan_dependent": False},
    "VRC-PG04": {"epitope": "CD4bs", "position_range": (124, 477), "glycan_dependent": False},
    "NIH45-46": {"epitope": "CD4bs", "position_range": (124, 477), "glycan_dependent": False},
    "3BNC117": {"epitope": "CD4bs", "position_range": (124, 477), "glycan_dependent": False},
    "b12": {"epitope": "CD4bs", "position_range": (124, 477), "glycan_dependent": False},
    # V1/V2 apex antibodies
    "PG9": {"epitope": "V1V2", "position_range": (126, 196), "glycan_dependent": True},
    "PG16": {"epitope": "V1V2", "position_range": (126, 196), "glycan_dependent": True},
    "PGT145": {"epitope": "V1V2", "position_range": (126, 196), "glycan_dependent": True},
    "CH01-04": {"epitope": "V1V2", "position_range": (126, 196), "glycan_dependent": True},
    # V3 glycan antibodies
    "PGT121": {"epitope": "V3_glycan", "position_range": (295, 332), "glycan_dependent": True},
    "PGT128": {"epitope": "V3_glycan", "position_range": (295, 332), "glycan_dependent": True},
    "PGT135": {"epitope": "V3_glycan", "position_range": (295, 332), "glycan_dependent": True},
    "10-1074": {"epitope": "V3_glycan", "position_range": (295, 332), "glycan_dependent": True},
    # MPER antibodies
    "10E8": {"epitope": "MPER", "position_range": (656, 683), "glycan_dependent": False},
    "4E10": {"epitope": "MPER", "position_range": (656, 683), "glycan_dependent": False},
    "2F5": {"epitope": "MPER", "position_range": (656, 683), "glycan_dependent": False},
    # gp120/gp41 interface
    "35O22": {"epitope": "interface", "position_range": (84, 625), "glycan_dependent": True},
    "8ANC195": {"epitope": "interface", "position_range": (84, 625), "glycan_dependent": True},
    # Fusion peptide
    "VRC34": {"epitope": "fusion_peptide", "position_range": (512, 527), "glycan_dependent": False},
    "ACS202": {"epitope": "fusion_peptide", "position_range": (512, 527), "glycan_dependent": False},
}

# Epitope overlap matrix (1 = overlapping, 0 = non-overlapping)
EPITOPE_GROUPS = ["CD4bs", "V1V2", "V3_glycan", "MPER", "interface", "fusion_peptide"]


@dataclass
class AntibodyProfile:
    """Profile of a single antibody's neutralization characteristics."""

    name: str
    epitope: str
    breadth: float  # Fraction of viruses neutralized (IC50 < 50)
    median_ic50: float  # Median IC50 across sensitive viruses
    geometric_mean_ic50: float  # Geometric mean IC50
    resistant_fraction: float  # Fraction with IC50 > 50
    sensitive_viruses: set = field(default_factory=set)
    resistant_viruses: set = field(default_factory=set)


@dataclass
class CombinationResult:
    """Results of antibody combination analysis."""

    antibodies: list[str]
    combined_breadth: float
    unique_coverage: dict[str, float]  # Each Ab's unique contribution
    overlap_coverage: float  # Covered by multiple Abs
    synergy_score: float  # Estimated synergy
    resistance_barrier: float  # Estimated genetic barrier
    epitope_diversity: int  # Number of distinct epitopes targeted
    complementarity_score: float  # How well Abs complement each other


def calculate_epitope_overlap(ab1: str, ab2: str) -> float:
    """
    Calculate epitope overlap between two antibodies.

    Returns:
        Overlap score from 0 (no overlap) to 1 (complete overlap)
    """
    if ab1 not in BNAB_EPITOPES or ab2 not in BNAB_EPITOPES:
        return 0.5  # Unknown, assume moderate overlap

    info1 = BNAB_EPITOPES[ab1]
    info2 = BNAB_EPITOPES[ab2]

    # Same epitope class = high overlap
    if info1["epitope"] == info2["epitope"]:
        return 0.9

    # Different epitope classes = low overlap
    return 0.1


def calculate_antibody_profiles(
    catnap_data: pd.DataFrame, ic50_threshold: float = 50.0
) -> dict[str, AntibodyProfile]:
    """
    Calculate neutralization profiles for each antibody in CATNAP data.

    Args:
        catnap_data: DataFrame with columns ['antibody', 'virus', 'ic50']
        ic50_threshold: IC50 threshold for sensitivity (default 50 µg/mL)

    Returns:
        Dictionary mapping antibody names to AntibodyProfile objects
    """
    profiles = {}

    for ab_name, ab_data in catnap_data.groupby("antibody"):
        # Filter out missing values
        ab_data = ab_data.dropna(subset=["ic50"])

        if len(ab_data) == 0:
            continue

        # Parse IC50 values (handle censored data like ">50")
        ic50_values = []
        for val in ab_data["ic50"]:
            if isinstance(val, str):
                if val.startswith(">"):
                    ic50_values.append(float(val[1:]) * 2)  # Conservative estimate
                elif val.startswith("<"):
                    ic50_values.append(float(val[1:]) / 2)
                else:
                    try:
                        ic50_values.append(float(val))
                    except ValueError:
                        continue
            else:
                ic50_values.append(float(val))

        if len(ic50_values) == 0:
            continue

        ic50_array = np.array(ic50_values)
        viruses = ab_data["virus"].values

        # Calculate metrics
        sensitive_mask = ic50_array < ic50_threshold
        sensitive_viruses = set(viruses[sensitive_mask])
        resistant_viruses = set(viruses[~sensitive_mask])

        breadth = np.mean(sensitive_mask)
        sensitive_ic50s = ic50_array[sensitive_mask]

        if len(sensitive_ic50s) > 0:
            median_ic50 = np.median(sensitive_ic50s)
            geometric_mean_ic50 = np.exp(np.mean(np.log(sensitive_ic50s + 0.001)))
        else:
            median_ic50 = np.nan
            geometric_mean_ic50 = np.nan

        # Get epitope info
        epitope = BNAB_EPITOPES.get(ab_name, {}).get("epitope", "unknown")

        profiles[ab_name] = AntibodyProfile(
            name=ab_name,
            epitope=epitope,
            breadth=breadth,
            median_ic50=median_ic50,
            geometric_mean_ic50=geometric_mean_ic50,
            resistant_fraction=1 - breadth,
            sensitive_viruses=sensitive_viruses,
            resistant_viruses=resistant_viruses,
        )

    return profiles


def calculate_combination_breadth(
    antibodies: list[str], profiles: dict[str, AntibodyProfile]
) -> tuple[float, dict]:
    """
    Calculate combined breadth of antibody combination.

    Uses union of sensitive viruses across antibodies.

    Returns:
        Tuple of (combined breadth, coverage breakdown dict)
    """
    if not antibodies:
        return 0.0, {}

    # Collect all viruses and sensitive sets
    all_viruses = set()
    sensitive_by_ab = {}

    for ab in antibodies:
        if ab not in profiles:
            continue
        profile = profiles[ab]
        all_viruses.update(profile.sensitive_viruses)
        all_viruses.update(profile.resistant_viruses)
        sensitive_by_ab[ab] = profile.sensitive_viruses

    if len(all_viruses) == 0:
        return 0.0, {}

    # Calculate union coverage
    covered_viruses = set()
    unique_coverage = {}

    for ab in antibodies:
        if ab not in sensitive_by_ab:
            continue
        ab_sensitive = sensitive_by_ab[ab]
        # Unique coverage = viruses only covered by this Ab
        other_coverage = set()
        for other_ab, other_sens in sensitive_by_ab.items():
            if other_ab != ab:
                other_coverage.update(other_sens)

        unique_to_ab = ab_sensitive - other_coverage
        unique_coverage[ab] = len(unique_to_ab) / len(all_viruses)
        covered_viruses.update(ab_sensitive)

    combined_breadth = len(covered_viruses) / len(all_viruses)

    # Calculate overlap
    overlap_viruses = set()
    for v in covered_viruses:
        count = sum(1 for ab in antibodies if ab in sensitive_by_ab and v in sensitive_by_ab[ab])
        if count > 1:
            overlap_viruses.add(v)

    overlap_coverage = len(overlap_viruses) / len(all_viruses)

    coverage_breakdown = {
        "unique_coverage": unique_coverage,
        "overlap_coverage": overlap_coverage,
        "total_viruses": len(all_viruses),
        "covered_viruses": len(covered_viruses),
    }

    return combined_breadth, coverage_breakdown


def estimate_synergy(antibodies: list[str], profiles: dict[str, AntibodyProfile]) -> float:
    """
    Estimate synergy score for antibody combination.

    Synergy is estimated based on:
    1. Non-overlapping epitopes (different binding sites)
    2. Different glycan dependencies
    3. Complementary resistance profiles

    Returns:
        Synergy score from 0 (antagonistic) to 2 (highly synergistic)
        1.0 = additive (no synergy)
    """
    if len(antibodies) < 2:
        return 1.0

    synergy = 1.0
    n_pairs = 0

    for ab1, ab2 in combinations(antibodies, 2):
        if ab1 not in BNAB_EPITOPES or ab2 not in BNAB_EPITOPES:
            continue

        info1 = BNAB_EPITOPES[ab1]
        info2 = BNAB_EPITOPES[ab2]

        pair_synergy = 1.0

        # Different epitopes = potential synergy
        if info1["epitope"] != info2["epitope"]:
            pair_synergy += 0.2

        # Different glycan dependencies = synergy
        if info1["glycan_dependent"] != info2["glycan_dependent"]:
            pair_synergy += 0.1

        # Non-overlapping positions = synergy
        r1 = set(range(info1["position_range"][0], info1["position_range"][1] + 1))
        r2 = set(range(info2["position_range"][0], info2["position_range"][1] + 1))
        position_overlap = len(r1 & r2) / max(len(r1), len(r2))

        if position_overlap < 0.1:
            pair_synergy += 0.15

        synergy += (pair_synergy - 1.0)
        n_pairs += 1

    if n_pairs > 0:
        synergy = 1.0 + (synergy - 1.0) / n_pairs

    return min(2.0, max(0.5, synergy))


def estimate_resistance_barrier(
    antibodies: list[str], profiles: dict[str, AntibodyProfile]
) -> float:
    """
    Estimate genetic barrier to resistance for antibody combination.

    Higher barrier = more mutations needed to escape all antibodies.

    Returns:
        Estimated number of mutations needed to escape combination
    """
    if len(antibodies) == 0:
        return 0.0

    # Count distinct epitopes
    epitopes = set()
    for ab in antibodies:
        if ab in BNAB_EPITOPES:
            epitopes.add(BNAB_EPITOPES[ab]["epitope"])

    # Base barrier = number of distinct epitopes (need to escape each)
    barrier = len(epitopes)

    # Adjust for glycan-dependent Abs (harder to escape without fitness cost)
    glycan_count = sum(
        1 for ab in antibodies
        if ab in BNAB_EPITOPES and BNAB_EPITOPES[ab]["glycan_dependent"]
    )
    barrier += 0.5 * glycan_count

    # Adjust for combination size
    barrier *= (1 + 0.1 * (len(antibodies) - 1))

    return barrier


def analyze_combination(
    antibodies: list[str], profiles: dict[str, AntibodyProfile]
) -> CombinationResult:
    """
    Comprehensive analysis of an antibody combination.

    Args:
        antibodies: List of antibody names
        profiles: Dictionary of AntibodyProfile objects

    Returns:
        CombinationResult with all metrics
    """
    # Calculate breadth
    combined_breadth, coverage = calculate_combination_breadth(antibodies, profiles)

    # Calculate synergy
    synergy = estimate_synergy(antibodies, profiles)

    # Calculate resistance barrier
    barrier = estimate_resistance_barrier(antibodies, profiles)

    # Count epitope diversity
    epitopes = set()
    for ab in antibodies:
        if ab in BNAB_EPITOPES:
            epitopes.add(BNAB_EPITOPES[ab]["epitope"])
    epitope_diversity = len(epitopes)

    # Calculate complementarity score
    # High complementarity = high breadth with low overlap
    if combined_breadth > 0:
        complementarity = combined_breadth * (1 - coverage.get("overlap_coverage", 0))
    else:
        complementarity = 0.0

    return CombinationResult(
        antibodies=antibodies,
        combined_breadth=combined_breadth,
        unique_coverage=coverage.get("unique_coverage", {}),
        overlap_coverage=coverage.get("overlap_coverage", 0.0),
        synergy_score=synergy,
        resistance_barrier=barrier,
        epitope_diversity=epitope_diversity,
        complementarity_score=complementarity,
    )


def find_optimal_combinations(
    profiles: dict[str, AntibodyProfile],
    n_antibodies: int = 3,
    min_breadth: float = 0.9,
    top_k: int = 10,
) -> list[CombinationResult]:
    """
    Find optimal antibody combinations using greedy search.

    Args:
        profiles: Dictionary of AntibodyProfile objects
        n_antibodies: Number of antibodies in combination
        min_breadth: Minimum required breadth
        top_k: Number of top combinations to return

    Returns:
        List of top CombinationResult objects sorted by combined_breadth
    """
    all_antibodies = list(profiles.keys())

    if len(all_antibodies) < n_antibodies:
        return []

    results = []

    # Enumerate all combinations of size n_antibodies
    for combo in combinations(all_antibodies, n_antibodies):
        result = analyze_combination(list(combo), profiles)

        if result.combined_breadth >= min_breadth:
            results.append(result)

    # Sort by combined breadth (primary) and complementarity (secondary)
    results.sort(key=lambda x: (x.combined_breadth, x.complementarity_score), reverse=True)

    return results[:top_k]


def greedy_combination_selection(
    profiles: dict[str, AntibodyProfile],
    max_antibodies: int = 4,
    target_breadth: float = 0.98,
) -> CombinationResult:
    """
    Greedy algorithm to select minimal antibody combination for target breadth.

    Args:
        profiles: Dictionary of AntibodyProfile objects
        max_antibodies: Maximum number of antibodies to include
        target_breadth: Target breadth to achieve

    Returns:
        CombinationResult for selected combination
    """
    selected = []
    remaining = list(profiles.keys())

    while len(selected) < max_antibodies and remaining:
        best_ab = None
        best_marginal_gain = -1

        for ab in remaining:
            test_combo = selected + [ab]
            breadth, _ = calculate_combination_breadth(test_combo, profiles)

            # Calculate marginal gain
            if selected:
                current_breadth, _ = calculate_combination_breadth(selected, profiles)
                marginal_gain = breadth - current_breadth
            else:
                marginal_gain = breadth

            if marginal_gain > best_marginal_gain:
                best_marginal_gain = marginal_gain
                best_ab = ab

        if best_ab is None or best_marginal_gain <= 0:
            break

        selected.append(best_ab)
        remaining.remove(best_ab)

        # Check if target achieved
        current_breadth, _ = calculate_combination_breadth(selected, profiles)
        if current_breadth >= target_breadth:
            break

    return analyze_combination(selected, profiles)


def diversify_by_epitope(
    profiles: dict[str, AntibodyProfile],
    n_per_epitope: int = 1,
) -> CombinationResult:
    """
    Select antibodies to maximize epitope diversity.

    Selects the best antibody from each epitope class.

    Args:
        profiles: Dictionary of AntibodyProfile objects
        n_per_epitope: Number of antibodies per epitope class

    Returns:
        CombinationResult for diversified combination
    """
    # Group antibodies by epitope
    by_epitope: dict[str, list[tuple[str, float]]] = {}

    for ab_name, profile in profiles.items():
        epitope = profile.epitope
        if epitope not in by_epitope:
            by_epitope[epitope] = []
        by_epitope[epitope].append((ab_name, profile.breadth))

    # Select top N from each epitope
    selected = []
    for epitope, abs_list in by_epitope.items():
        # Sort by breadth
        abs_list.sort(key=lambda x: x[1], reverse=True)
        for i, (ab_name, _) in enumerate(abs_list):
            if i >= n_per_epitope:
                break
            selected.append(ab_name)

    return analyze_combination(selected, profiles)


def analyze_with_hyperbolic_features(
    catnap_data: pd.DataFrame,
    hyperbolic_embeddings: dict[str, np.ndarray],
    encode_codon_fn,
) -> pd.DataFrame:
    """
    Integrate CATNAP neutralization data with hyperbolic codon embeddings.

    Args:
        catnap_data: CATNAP neutralization data
        hyperbolic_embeddings: Precomputed hyperbolic embeddings for sequences
        encode_codon_fn: Function to encode codons to hyperbolic space

    Returns:
        DataFrame with hyperbolic features added
    """
    results = []

    for _, row in catnap_data.iterrows():
        virus_id = row.get("virus", "")
        antibody = row.get("antibody", "")
        ic50 = row.get("ic50", np.nan)

        # Get antibody epitope info
        epitope_info = BNAB_EPITOPES.get(antibody, {})
        epitope_type = epitope_info.get("epitope", "unknown")

        # Get hyperbolic features if available
        if virus_id in hyperbolic_embeddings:
            embedding = hyperbolic_embeddings[virus_id]
            radial_dist = np.linalg.norm(embedding)
            # Calculate centrality (inverse of radial distance in Poincaré disk)
            centrality = 1 / (1 + radial_dist)
        else:
            radial_dist = np.nan
            centrality = np.nan

        results.append({
            "virus": virus_id,
            "antibody": antibody,
            "epitope": epitope_type,
            "ic50": ic50,
            "hyperbolic_distance": radial_dist,
            "centrality": centrality,
        })

    return pd.DataFrame(results)


def generate_combination_report(result: CombinationResult) -> str:
    """
    Generate a human-readable report for an antibody combination.

    Args:
        result: CombinationResult object

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "ANTIBODY COMBINATION ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Antibodies: {', '.join(result.antibodies)}",
        f"Number of antibodies: {len(result.antibodies)}",
        "",
        "COVERAGE METRICS:",
        f"  Combined breadth: {result.combined_breadth:.1%}",
        f"  Overlap coverage: {result.overlap_coverage:.1%}",
        f"  Complementarity score: {result.complementarity_score:.3f}",
        "",
        "UNIQUE CONTRIBUTIONS:",
    ]

    for ab, coverage in result.unique_coverage.items():
        lines.append(f"  {ab}: {coverage:.1%} unique")

    lines.extend([
        "",
        "RESISTANCE PROFILE:",
        f"  Epitope diversity: {result.epitope_diversity} distinct epitopes",
        f"  Estimated resistance barrier: {result.resistance_barrier:.1f} mutations",
        f"  Synergy score: {result.synergy_score:.2f} (1.0 = additive)",
        "",
        "EPITOPES TARGETED:",
    ])

    epitopes_targeted = set()
    for ab in result.antibodies:
        if ab in BNAB_EPITOPES:
            epitopes_targeted.add(BNAB_EPITOPES[ab]["epitope"])

    for epitope in sorted(epitopes_targeted):
        abs_in_epitope = [ab for ab in result.antibodies
                         if ab in BNAB_EPITOPES and BNAB_EPITOPES[ab]["epitope"] == epitope]
        lines.append(f"  {epitope}: {', '.join(abs_in_epitope)}")

    lines.extend([
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic test data
    print("Testing Antibody Combination Analysis Module")
    print("=" * 50)

    # Generate synthetic CATNAP-like data
    np.random.seed(42)

    antibodies = ["VRC01", "PG9", "PGT121", "10E8", "3BNC117", "PGT128"]
    viruses = [f"virus_{i}" for i in range(100)]

    test_data = []
    for ab in antibodies:
        for virus in viruses:
            # Simulate IC50 based on antibody breadth
            if ab == "VRC01":
                ic50 = np.random.exponential(10) if np.random.random() < 0.9 else 100
            elif ab == "PG9":
                ic50 = np.random.exponential(15) if np.random.random() < 0.7 else 100
            elif ab == "PGT121":
                ic50 = np.random.exponential(5) if np.random.random() < 0.85 else 100
            elif ab == "10E8":
                ic50 = np.random.exponential(20) if np.random.random() < 0.95 else 100
            else:
                ic50 = np.random.exponential(12) if np.random.random() < 0.8 else 100

            test_data.append({"antibody": ab, "virus": virus, "ic50": ic50})

    df = pd.DataFrame(test_data)

    # Calculate profiles
    print("\nCalculating antibody profiles...")
    profiles = calculate_antibody_profiles(df)

    print("\nIndividual Antibody Profiles:")
    print("-" * 40)
    for name, profile in sorted(profiles.items(), key=lambda x: x[1].breadth, reverse=True):
        print(f"{name:12} | Epitope: {profile.epitope:12} | Breadth: {profile.breadth:.1%} | "
              f"Median IC50: {profile.median_ic50:.1f}")

    # Test combination analysis
    print("\n" + "=" * 50)
    print("Testing Combination Analysis")
    print("=" * 50)

    # Analyze a specific combination
    test_combo = ["VRC01", "PGT121", "10E8"]
    result = analyze_combination(test_combo, profiles)
    print(generate_combination_report(result))

    # Find optimal combinations
    print("\nFinding optimal 3-antibody combinations...")
    optimal = find_optimal_combinations(profiles, n_antibodies=3, min_breadth=0.8, top_k=5)

    print("\nTop 5 combinations with breadth >= 80%:")
    for i, combo in enumerate(optimal, 1):
        print(f"{i}. {', '.join(combo.antibodies):40} | Breadth: {combo.combined_breadth:.1%} | "
              f"Barrier: {combo.resistance_barrier:.1f}")

    # Greedy selection
    print("\nGreedy selection for 98% breadth...")
    greedy_result = greedy_combination_selection(profiles, max_antibodies=4, target_breadth=0.98)
    print(f"Selected: {', '.join(greedy_result.antibodies)}")
    print(f"Achieved breadth: {greedy_result.combined_breadth:.1%}")

    # Diversify by epitope
    print("\nEpitope-diversified selection...")
    diverse_result = diversify_by_epitope(profiles, n_per_epitope=1)
    print(f"Selected: {', '.join(diverse_result.antibodies)}")
    print(f"Epitope diversity: {diverse_result.epitope_diversity}")
    print(f"Combined breadth: {diverse_result.combined_breadth:.1%}")

    print("\n" + "=" * 50)
    print("Module testing complete!")
