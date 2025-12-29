# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""
HLA Population Coverage Calculator

Calculates vaccine epitope coverage across different populations based on
HLA allele frequencies.

Based on:
- Fischer W, et al. (2007) - Polyvalent vaccines for optimal coverage
- Korber B, et al. (2020) - T-cell vaccine strategies for HIV
- Bui HH, et al. (2006) - IEDB population coverage tool

Data sources:
- Allele Frequency Net Database (AFND)
- IEDB population coverage
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# ============================================================================
# HLA FREQUENCY DATA
# ============================================================================

# HLA-A frequencies by population (from AFND)
HLA_A_FREQUENCIES = {
    "A*01:01": {"European": 0.140, "African": 0.035, "Asian": 0.040, "Hispanic": 0.075},
    "A*02:01": {"European": 0.280, "African": 0.120, "Asian": 0.150, "Hispanic": 0.190},
    "A*02:02": {"European": 0.015, "African": 0.060, "Asian": 0.005, "Hispanic": 0.025},
    "A*02:05": {"European": 0.020, "African": 0.045, "Asian": 0.010, "Hispanic": 0.035},
    "A*03:01": {"European": 0.130, "African": 0.065, "Asian": 0.035, "Hispanic": 0.055},
    "A*11:01": {"European": 0.055, "African": 0.020, "Asian": 0.260, "Hispanic": 0.035},
    "A*23:01": {"European": 0.020, "African": 0.115, "Asian": 0.010, "Hispanic": 0.055},
    "A*24:02": {"European": 0.090, "African": 0.025, "Asian": 0.220, "Hispanic": 0.115},
    "A*26:01": {"European": 0.035, "African": 0.025, "Asian": 0.045, "Hispanic": 0.045},
    "A*29:02": {"European": 0.035, "African": 0.045, "Asian": 0.005, "Hispanic": 0.050},
    "A*30:01": {"European": 0.025, "African": 0.080, "Asian": 0.020, "Hispanic": 0.040},
    "A*30:02": {"European": 0.020, "African": 0.075, "Asian": 0.015, "Hispanic": 0.030},
    "A*31:01": {"European": 0.030, "African": 0.025, "Asian": 0.055, "Hispanic": 0.055},
    "A*32:01": {"European": 0.040, "African": 0.015, "Asian": 0.015, "Hispanic": 0.030},
    "A*33:01": {"European": 0.010, "African": 0.035, "Asian": 0.065, "Hispanic": 0.020},
    "A*33:03": {"European": 0.005, "African": 0.020, "Asian": 0.085, "Hispanic": 0.015},
    "A*68:01": {"European": 0.045, "African": 0.060, "Asian": 0.025, "Hispanic": 0.055},
    "A*68:02": {"European": 0.020, "African": 0.045, "Asian": 0.015, "Hispanic": 0.035},
}

# HLA-B frequencies by population (from AFND)
HLA_B_FREQUENCIES = {
    "B*07:02": {"European": 0.125, "African": 0.045, "Asian": 0.055, "Hispanic": 0.070},
    "B*08:01": {"European": 0.105, "African": 0.025, "Asian": 0.015, "Hispanic": 0.045},
    "B*13:02": {"European": 0.025, "African": 0.020, "Asian": 0.065, "Hispanic": 0.030},
    "B*14:02": {"European": 0.040, "African": 0.035, "Asian": 0.010, "Hispanic": 0.045},
    "B*15:01": {"European": 0.065, "African": 0.015, "Asian": 0.085, "Hispanic": 0.040},
    "B*18:01": {"European": 0.050, "African": 0.035, "Asian": 0.010, "Hispanic": 0.055},
    "B*27:02": {"European": 0.020, "African": 0.005, "Asian": 0.010, "Hispanic": 0.015},
    "B*27:05": {"European": 0.045, "African": 0.010, "Asian": 0.035, "Hispanic": 0.025},
    "B*35:01": {"European": 0.065, "African": 0.045, "Asian": 0.060, "Hispanic": 0.095},
    "B*35:03": {"European": 0.015, "African": 0.025, "Asian": 0.035, "Hispanic": 0.035},
    "B*38:01": {"European": 0.025, "African": 0.010, "Asian": 0.025, "Hispanic": 0.020},
    "B*39:01": {"European": 0.020, "African": 0.015, "Asian": 0.045, "Hispanic": 0.045},
    "B*40:01": {"European": 0.055, "African": 0.020, "Asian": 0.105, "Hispanic": 0.040},
    "B*40:02": {"European": 0.025, "African": 0.010, "Asian": 0.055, "Hispanic": 0.025},
    "B*44:02": {"European": 0.080, "African": 0.030, "Asian": 0.025, "Hispanic": 0.045},
    "B*44:03": {"European": 0.045, "African": 0.025, "Asian": 0.040, "Hispanic": 0.035},
    "B*45:01": {"European": 0.010, "African": 0.065, "Asian": 0.005, "Hispanic": 0.025},
    "B*51:01": {"European": 0.055, "African": 0.025, "Asian": 0.090, "Hispanic": 0.050},
    "B*52:01": {"European": 0.015, "African": 0.015, "Asian": 0.085, "Hispanic": 0.020},
    "B*53:01": {"European": 0.010, "African": 0.105, "Asian": 0.005, "Hispanic": 0.030},
    "B*55:01": {"European": 0.020, "African": 0.020, "Asian": 0.025, "Hispanic": 0.020},
    "B*57:01": {"European": 0.035, "African": 0.050, "Asian": 0.015, "Hispanic": 0.025},
    "B*57:03": {"European": 0.015, "African": 0.035, "Asian": 0.010, "Hispanic": 0.015},
    "B*58:01": {"European": 0.015, "African": 0.070, "Asian": 0.045, "Hispanic": 0.025},
    "B*58:02": {"European": 0.005, "African": 0.045, "Asian": 0.015, "Hispanic": 0.010},
}

# Combine all HLA frequencies
HLA_FREQUENCIES = {**HLA_A_FREQUENCIES, **HLA_B_FREQUENCIES}

# Population sizes (millions) for weighted global average
POPULATION_WEIGHTS = {
    "European": 0.12,   # ~12% world population
    "African": 0.17,    # ~17% world population
    "Asian": 0.60,      # ~60% world population
    "Hispanic": 0.11,   # ~11% world population
}


# ============================================================================
# COVERAGE CALCULATION
# ============================================================================

def get_hla_frequency(hla: str, population: str) -> float:
    """
    Get frequency of an HLA allele in a population.

    Args:
        hla: HLA allele (e.g., "A*02:01", "B*57:01")
        population: Population name

    Returns:
        float: Allele frequency (0-1)
    """
    # Normalize HLA format
    hla_normalized = normalize_hla(hla)

    if hla_normalized in HLA_FREQUENCIES:
        return HLA_FREQUENCIES[hla_normalized].get(population, 0.0)

    # Try supertype matching
    supertype = get_hla_supertype(hla_normalized)
    if supertype:
        # Sum frequencies of all alleles in supertype
        total = 0.0
        for allele, freqs in HLA_FREQUENCIES.items():
            if get_hla_supertype(allele) == supertype:
                total += freqs.get(population, 0.0)
        return min(total, 0.5)  # Cap at 50%

    return 0.0


def normalize_hla(hla: str) -> str:
    """
    Normalize HLA allele format.

    Examples:
        "A0201" -> "A*02:01"
        "B57" -> "B*57:01"
        "A*02:01" -> "A*02:01"
    """
    hla = hla.upper().strip()

    # Already in correct format
    if "*" in hla and ":" in hla:
        return hla

    # Handle formats like "A0201"
    if hla[0] in "AB" and hla[1].isdigit():
        if len(hla) == 4:
            return f"{hla[0]}*{hla[1:3]}:{hla[3]}1"
        elif len(hla) == 5:
            return f"{hla[0]}*{hla[1:3]}:{hla[3:5]}"

    # Handle formats like "A*0201"
    if "*" in hla and ":" not in hla:
        parts = hla.split("*")
        if len(parts) == 2 and len(parts[1]) >= 4:
            return f"{parts[0]}*{parts[1][:2]}:{parts[1][2:4]}"

    # Handle formats like "B57"
    if len(hla) <= 3 and hla[0] in "AB":
        return f"{hla[0]}*{hla[1:].zfill(2)}:01"

    return hla


def get_hla_supertype(hla: str) -> Optional[str]:
    """
    Get HLA supertype for an allele.

    HLA supertypes group alleles with similar binding specificities.
    """
    SUPERTYPES = {
        # A supertypes
        "A01": ["A*01:01", "A*26:01", "A*29:02", "A*30:02", "A*32:01"],
        "A02": ["A*02:01", "A*02:02", "A*02:03", "A*02:05", "A*02:06", "A*68:02"],
        "A03": ["A*03:01", "A*11:01", "A*31:01", "A*33:01", "A*68:01"],
        "A24": ["A*23:01", "A*24:02", "A*24:03"],

        # B supertypes
        "B07": ["B*07:02", "B*35:01", "B*35:03", "B*51:01", "B*53:01", "B*55:01"],
        "B08": ["B*08:01"],
        "B27": ["B*14:02", "B*27:02", "B*27:05", "B*38:01", "B*39:01"],
        "B44": ["B*18:01", "B*40:01", "B*40:02", "B*44:02", "B*44:03", "B*45:01"],
        "B58": ["B*15:01", "B*57:01", "B*57:03", "B*58:01", "B*58:02"],
        "B62": ["B*13:02", "B*40:06", "B*52:01"],
    }

    hla_normalized = normalize_hla(hla)

    for supertype, alleles in SUPERTYPES.items():
        if hla_normalized in alleles:
            return supertype

    return None


def calculate_epitope_coverage(
    hla_restrictions: list[str],
    population: str
) -> float:
    """
    Calculate population coverage for a single epitope.

    Uses the formula: coverage = 1 - Π(1 - 2*f_i + f_i^2)
    where f_i is the frequency of each restricting HLA.

    This accounts for homozygosity and heterozygosity.

    Args:
        hla_restrictions: List of HLA alleles that can present the epitope
        population: Population name

    Returns:
        float: Fraction of population covered (0-1)
    """
    if not hla_restrictions:
        return 0.0

    # Get frequencies for each HLA
    frequencies = []
    for hla in hla_restrictions:
        freq = get_hla_frequency(hla, population)
        if freq > 0:
            frequencies.append(freq)

    if not frequencies:
        return 0.0

    # Calculate coverage
    # P(not covered by any HLA) = Π(1 - 2f + f²) for each HLA
    # This accounts for diploid genotype
    prob_not_covered = 1.0
    for f in frequencies:
        # P(not having this allele) = (1-f)²
        prob_not_having = (1 - f) ** 2
        prob_not_covered *= prob_not_having

    coverage = 1.0 - prob_not_covered

    return min(coverage, 1.0)


def calculate_vaccine_coverage(
    epitopes: list[dict],
    populations: Optional[list[str]] = None
) -> dict:
    """
    Calculate population coverage for a set of vaccine epitopes.

    Args:
        epitopes: List of dicts with 'epitope' and 'hla_restrictions' keys
        populations: Populations to calculate coverage for

    Returns:
        dict: Coverage by population and combined
    """
    if populations is None:
        populations = ["European", "African", "Asian", "Hispanic"]

    results = {}

    for pop in populations:
        # Calculate coverage for each epitope
        epitope_coverages = []
        for ep in epitopes:
            hlas = ep.get("hla_restrictions", [])
            if isinstance(hlas, str):
                hlas = [h.strip() for h in hlas.split(",")]
            coverage = calculate_epitope_coverage(hlas, pop)
            epitope_coverages.append(coverage)

        # Combined coverage (at least one epitope recognized)
        # P(at least one) = 1 - Π(1 - coverage_i)
        prob_none = 1.0
        for cov in epitope_coverages:
            prob_none *= (1 - cov)
        combined_coverage = 1 - prob_none

        results[pop] = {
            "coverage": combined_coverage,
            "epitope_coverages": epitope_coverages,
            "n_epitopes": len(epitopes),
        }

    # Calculate global average
    global_coverage = sum(
        results[pop]["coverage"] * POPULATION_WEIGHTS.get(pop, 0.25)
        for pop in populations
    )
    results["global_average"] = global_coverage

    return results


# ============================================================================
# OPTIMIZATION
# ============================================================================

def optimize_epitope_selection(
    candidate_epitopes: pd.DataFrame,
    n_select: int = 10,
    min_coverage: float = 0.80,
    populations: Optional[list[str]] = None,
    constraints: Optional[dict] = None
) -> list[dict]:
    """
    Select optimal epitopes for maximum population coverage.

    Uses a greedy set cover algorithm with optional constraints.

    Args:
        candidate_epitopes: DataFrame with epitope info
        n_select: Maximum number of epitopes to select
        min_coverage: Minimum target coverage per population
        populations: Populations to optimize for
        constraints: Optional constraints (e.g., exclude resistance positions)

    Returns:
        list[dict]: Selected epitopes with coverage info
    """
    if populations is None:
        populations = ["European", "African", "Asian", "Hispanic"]

    # Initialize
    selected = []
    remaining = candidate_epitopes.copy()
    current_coverage = {pop: 0.0 for pop in populations}

    for _ in range(n_select):
        if len(remaining) == 0:
            break

        best_epitope = None
        best_marginal_gain = 0.0

        # Evaluate each candidate
        for idx, row in remaining.iterrows():
            hlas = row.get("hla_restrictions", row.get("HLA", ""))
            if isinstance(hlas, str):
                hlas = [h.strip() for h in hlas.split(",") if h.strip()]

            # Calculate marginal coverage gain
            marginal_gains = []
            for pop in populations:
                current = current_coverage[pop]
                # Calculate what coverage would be with this epitope
                test_epitopes = selected + [{"hla_restrictions": hlas}]
                new_coverage = calculate_vaccine_coverage(test_epitopes, [pop])
                new = new_coverage[pop]["coverage"]
                marginal_gains.append(new - current)

            # Average marginal gain across populations
            avg_gain = np.mean(marginal_gains)

            # Apply constraints if provided
            if constraints:
                if constraints.get("no_resistance_overlap"):
                    if row.get("resistance_overlap") == "Yes":
                        avg_gain *= 0.1  # Penalize

                if constraints.get("min_escape_velocity"):
                    if row.get("escape_velocity", 1.0) > constraints["min_escape_velocity"]:
                        avg_gain *= 0.5  # Penalize high escape velocity

            if avg_gain > best_marginal_gain:
                best_marginal_gain = avg_gain
                best_epitope = idx

        if best_epitope is None or best_marginal_gain < 0.001:
            break

        # Add best epitope
        best_row = remaining.loc[best_epitope]
        hlas = best_row.get("hla_restrictions", best_row.get("HLA", ""))
        if isinstance(hlas, str):
            hlas = [h.strip() for h in hlas.split(",") if h.strip()]

        selected.append({
            "epitope": best_row.get("epitope", best_row.get("Epitope", str(best_epitope))),
            "hla_restrictions": hlas,
            "protein": best_row.get("protein", best_row.get("Protein", "")),
            "marginal_gain": best_marginal_gain,
        })

        # Update coverage
        coverage_result = calculate_vaccine_coverage(selected, populations)
        for pop in populations:
            current_coverage[pop] = coverage_result[pop]["coverage"]

        # Remove selected epitope from candidates
        remaining = remaining.drop(best_epitope)

        # Check if minimum coverage reached for all populations
        if all(current_coverage[pop] >= min_coverage for pop in populations):
            break

    # Add final coverage info to each selected epitope
    final_coverage = calculate_vaccine_coverage(selected, populations)
    for ep in selected:
        ep["cumulative_coverage"] = final_coverage

    return selected


def calculate_hla_coverage_matrix(
    epitopes: pd.DataFrame,
    populations: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Calculate coverage matrix: epitopes x populations.

    Args:
        epitopes: DataFrame with epitope and HLA information
        populations: Populations to include

    Returns:
        DataFrame: Coverage matrix
    """
    if populations is None:
        populations = ["European", "African", "Asian", "Hispanic"]

    coverage_data = []

    for _, row in epitopes.iterrows():
        epitope = row.get("epitope", row.get("Epitope", ""))
        hlas = row.get("hla_restrictions", row.get("HLA", ""))
        if isinstance(hlas, str):
            hlas = [h.strip() for h in hlas.split(",") if h.strip()]

        coverage_row = {"epitope": epitope}
        for pop in populations:
            coverage_row[pop] = calculate_epitope_coverage(hlas, pop)

        coverage_data.append(coverage_row)

    df = pd.DataFrame(coverage_data)
    df["mean_coverage"] = df[populations].mean(axis=1)

    return df.sort_values("mean_coverage", ascending=False)


# ============================================================================
# REPORTING
# ============================================================================

def generate_coverage_report(
    selected_epitopes: list[dict],
    populations: Optional[list[str]] = None
) -> str:
    """
    Generate a human-readable coverage report.

    Args:
        selected_epitopes: Selected epitopes from optimization
        populations: Populations included

    Returns:
        str: Formatted report
    """
    if populations is None:
        populations = ["European", "African", "Asian", "Hispanic"]

    coverage = calculate_vaccine_coverage(selected_epitopes, populations)

    report = []
    report.append("=" * 60)
    report.append("VACCINE EPITOPE POPULATION COVERAGE REPORT")
    report.append("=" * 60)
    report.append("")

    report.append(f"Number of epitopes: {len(selected_epitopes)}")
    report.append("")

    report.append("POPULATION COVERAGE:")
    report.append("-" * 40)
    for pop in populations:
        cov = coverage[pop]["coverage"] * 100
        bar = "█" * int(cov / 5) + "░" * (20 - int(cov / 5))
        report.append(f"  {pop:12s}: {bar} {cov:5.1f}%")

    report.append("")
    report.append(f"  {'Global Average':12s}: {coverage['global_average']*100:5.1f}%")
    report.append("")

    report.append("SELECTED EPITOPES:")
    report.append("-" * 40)
    for i, ep in enumerate(selected_epitopes, 1):
        epitope = ep.get("epitope", "Unknown")
        protein = ep.get("protein", "")
        n_hla = len(ep.get("hla_restrictions", []))
        report.append(f"  {i:2d}. {epitope} ({protein}) - {n_hla} HLA restrictions")

    report.append("")
    report.append("=" * 60)

    return "\n".join(report)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("HLA Population Coverage Calculator Examples")
    print("=" * 60)

    # Test single epitope coverage
    print("\n1. Single Epitope Coverage:")
    hlas = ["A*02:01", "A*03:01", "B*57:01"]
    for pop in ["European", "African", "Asian", "Hispanic"]:
        cov = calculate_epitope_coverage(hlas, pop)
        print(f"  {pop}: {cov*100:.1f}%")

    # Test vaccine coverage
    print("\n2. Multi-Epitope Vaccine Coverage:")
    test_epitopes = [
        {"epitope": "SLYNTVATL", "hla_restrictions": ["A*02:01"]},
        {"epitope": "TPQDLNTML", "hla_restrictions": ["B*57:01", "B*58:01"]},
        {"epitope": "KRWIILGLNK", "hla_restrictions": ["A*03:01", "A*11:01"]},
    ]
    coverage = calculate_vaccine_coverage(test_epitopes)
    for pop, data in coverage.items():
        if pop != "global_average":
            print(f"  {pop}: {data['coverage']*100:.1f}%")
    print(f"  Global: {coverage['global_average']*100:.1f}%")

    # Test optimization
    print("\n3. Epitope Optimization:")
    # Create mock candidate epitopes
    candidates = pd.DataFrame([
        {"epitope": "EP1", "hla_restrictions": "A*02:01", "protein": "Gag"},
        {"epitope": "EP2", "hla_restrictions": "B*57:01, B*58:01", "protein": "Gag"},
        {"epitope": "EP3", "hla_restrictions": "A*03:01, A*11:01", "protein": "Pol"},
        {"epitope": "EP4", "hla_restrictions": "A*24:02", "protein": "Env"},
        {"epitope": "EP5", "hla_restrictions": "B*35:01, B*53:01", "protein": "Nef"},
    ])

    selected = optimize_epitope_selection(candidates, n_select=3)
    print(f"  Selected {len(selected)} epitopes:")
    for ep in selected:
        print(f"    - {ep['epitope']} (gain: {ep['marginal_gain']*100:.1f}%)")

    print("\n4. Coverage Report:")
    print(generate_coverage_report(selected))
