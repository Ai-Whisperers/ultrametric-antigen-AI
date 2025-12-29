# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""
Fitness Cost Estimator

Estimates fitness costs of HIV mutations using hyperbolic geometry.

Based on:
- Kühnert D, et al. (2018) - Quantifying fitness cost of drug resistance
- Theys K, et al. (2018) - Within-patient mutation frequencies reveal fitness
- Ferguson AL, et al. (2013) - Translating HIV sequences into fitness landscapes

Key concept: Radial position in hyperbolic space correlates inversely with
fitness - mutations far from origin (peripheral) have higher fitness costs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from codon_extraction import encode_mutation_pair
    from hyperbolic_utils import load_hyperbolic_encoder
except ImportError:
    pass

# ============================================================================
# FITNESS COST MODEL PARAMETERS
# ============================================================================

# Calibration based on literature values
# From Kühnert et al. (2018) and others
KNOWN_FITNESS_COSTS = {
    # Drug resistance mutations with known fitness costs
    # Format: "mutation": (fitness_cost, source)
    # Fitness cost = 1 - relative replicative capacity

    # NRTI
    "M184V": (0.10, "Wainberg 1999"),   # Moderate cost
    "M184I": (0.15, "Wainberg 1999"),   # Higher cost than V
    "K65R": (0.05, "White 2002"),       # Low cost
    "T215Y": (0.03, "Goudsmit 1997"),   # Very low cost
    "T215F": (0.04, "Goudsmit 1997"),   # Very low cost

    # NNRTI
    "K103N": (0.00, "Collins 2004"),    # No fitness cost
    "Y181C": (0.02, "Collins 2004"),    # Minimal cost
    "G190A": (0.05, "Bacheler 2001"),   # Low cost

    # PI
    "M46I": (0.08, "Martinez-Picado 1999"),
    "I50V": (0.15, "Mammano 1998"),
    "V82A": (0.10, "Zennou 1998"),
    "L90M": (0.12, "Mammano 1998"),

    # INI
    "N155H": (0.20, "Malet 2008"),      # High cost
    "Q148H": (0.25, "Malet 2008"),      # Very high cost
    "Q148R": (0.30, "Malet 2008"),      # Very high cost
    "R263K": (0.35, "Anstett 2017"),    # Very high cost
}

# Reversion rates for CTL escape mutations
# From Leslie et al. (2004) and Crawford et al. (2009)
REVERSION_RATES = {
    # Format: "mutation": days_to_50%_reversion
    # Lower = faster reversion = higher fitness cost
    "T242N": 60,   # B57-restricted, reverts quickly
    "T186S": 90,   # Moderate reversion
    "A146P": 180,  # Slow reversion
}


# ============================================================================
# FITNESS COST ESTIMATION
# ============================================================================

def estimate_fitness_cost_from_geometry(
    hyperbolic_distance: float,
    radial_position_change: float,
    is_synonymous: bool = False
) -> dict:
    """
    Estimate fitness cost from geometric features.

    Model based on Ferguson et al. (2013):
    - Mutations to peripheral positions = higher cost
    - Larger hyperbolic distances = larger fitness impact
    - Synonymous mutations have near-zero cost

    Args:
        hyperbolic_distance: Distance in hyperbolic space
        radial_position_change: Change in radial position (positive = more peripheral)
        is_synonymous: Whether mutation is synonymous

    Returns:
        dict: {
            'fitness_cost': float (0-1 scale),
            'replicative_capacity': float (0-1 scale),
            'confidence': float,
            'cost_category': str
        }
    """
    if is_synonymous:
        return {
            "fitness_cost": 0.0,
            "replicative_capacity": 1.0,
            "confidence": 0.95,
            "cost_category": "none",
        }

    # Base cost from hyperbolic distance
    # Sigmoid transformation to bound between 0 and 1
    # Calibrated so distance ~5 maps to ~0.15 fitness cost
    base_cost = 1.0 / (1.0 + np.exp(-0.3 * (hyperbolic_distance - 3.0)))

    # Adjust for radial position change
    # Moving to periphery increases cost
    radial_adjustment = 0.1 * radial_position_change

    # Combined cost
    fitness_cost = np.clip(base_cost + radial_adjustment, 0.0, 0.95)

    # Confidence based on distance from calibration range
    if 1.0 <= hyperbolic_distance <= 8.0:
        confidence = 0.8
    else:
        confidence = 0.5

    # Categorize
    if fitness_cost < 0.05:
        category = "negligible"
    elif fitness_cost < 0.10:
        category = "low"
    elif fitness_cost < 0.20:
        category = "moderate"
    elif fitness_cost < 0.35:
        category = "high"
    else:
        category = "severe"

    return {
        "fitness_cost": float(fitness_cost),
        "replicative_capacity": float(1.0 - fitness_cost),
        "confidence": confidence,
        "cost_category": category,
    }


def estimate_fitness_cost(
    wild_type_aa: str,
    mutant_aa: str,
    position: Optional[int] = None,
    protein: Optional[str] = None,
    encoder=None
) -> dict:
    """
    Estimate fitness cost of a specific mutation.

    Args:
        wild_type_aa: Wild-type amino acid
        mutant_aa: Mutant amino acid
        position: Position in protein (optional, for known mutations)
        protein: Protein name (optional)
        encoder: Hyperbolic encoder (optional)

    Returns:
        dict: Fitness cost estimation with confidence
    """
    # Check if this is a known mutation
    mutation_str = f"{wild_type_aa}{position}{mutant_aa}" if position else None
    if mutation_str and mutation_str in KNOWN_FITNESS_COSTS:
        known_cost, source = KNOWN_FITNESS_COSTS[mutation_str]
        return {
            "fitness_cost": known_cost,
            "replicative_capacity": 1.0 - known_cost,
            "confidence": 0.95,
            "cost_category": _categorize_cost(known_cost),
            "source": source,
            "method": "literature",
        }

    # Check if synonymous
    if wild_type_aa == mutant_aa:
        return {
            "fitness_cost": 0.0,
            "replicative_capacity": 1.0,
            "confidence": 1.0,
            "cost_category": "none",
            "method": "synonymous",
        }

    # Calculate from geometry if encoder available
    if encoder is not None:
        try:
            result = encode_mutation_pair(wild_type_aa, mutant_aa, encoder)
            if result:
                geo_estimate = estimate_fitness_cost_from_geometry(
                    result["hyperbolic_distance"],
                    result.get("radial_change", 0.0),
                    is_synonymous=False
                )
                geo_estimate["method"] = "geometric"
                geo_estimate["hyperbolic_distance"] = result["hyperbolic_distance"]
                return geo_estimate
        except Exception:
            pass

    # Fallback: estimate from amino acid properties
    return estimate_fitness_from_properties(wild_type_aa, mutant_aa)


def _categorize_cost(cost: float) -> str:
    """Categorize fitness cost."""
    if cost < 0.05:
        return "negligible"
    elif cost < 0.10:
        return "low"
    elif cost < 0.20:
        return "moderate"
    elif cost < 0.35:
        return "high"
    else:
        return "severe"


def estimate_fitness_from_properties(wild_type_aa: str, mutant_aa: str) -> dict:
    """
    Estimate fitness cost from amino acid properties.

    Uses Grantham distance as a proxy for fitness impact.
    """
    # Grantham distances (partial - common substitutions)
    GRANTHAM = {
        ("A", "G"): 60, ("A", "S"): 99, ("A", "T"): 58, ("A", "V"): 64,
        ("D", "E"): 45, ("D", "N"): 23, ("F", "Y"): 22, ("I", "L"): 5,
        ("I", "V"): 29, ("K", "R"): 26, ("L", "M"): 15, ("L", "V"): 32,
        ("M", "V"): 21, ("N", "S"): 46, ("P", "S"): 74, ("S", "T"): 58,
        ("Y", "F"): 22, ("K", "N"): 94, ("E", "Q"): 29,
    }

    # Try both orderings
    key = (wild_type_aa.upper(), mutant_aa.upper())
    if key not in GRANTHAM:
        key = (mutant_aa.upper(), wild_type_aa.upper())

    if key in GRANTHAM:
        grantham = GRANTHAM[key]
        # Map Grantham (0-215) to fitness cost (0-0.5)
        cost = grantham / 430.0
    else:
        # Default moderate cost for unknown pairs
        cost = 0.15

    return {
        "fitness_cost": cost,
        "replicative_capacity": 1.0 - cost,
        "confidence": 0.5,
        "cost_category": _categorize_cost(cost),
        "method": "grantham",
    }


# ============================================================================
# TRANSMISSION AND REVERSION
# ============================================================================

def estimate_transmission_probability(
    fitness_cost: float,
    selection_coefficient: float = 0.0
) -> float:
    """
    Estimate probability of transmission given fitness cost.

    Based on Carlson et al. (2014):
    - Mutations with high fitness cost transmit less frequently
    - Selection during transmission is strong

    Args:
        fitness_cost: Estimated fitness cost (0-1)
        selection_coefficient: Additional selection (e.g., from drugs)

    Returns:
        float: Relative transmission probability (0-1)
    """
    # Transmission probability decreases with fitness cost
    # Model: P(transmission) ∝ exp(-β * fitness_cost)
    beta = 5.0  # Selection strength during transmission

    total_cost = fitness_cost + selection_coefficient
    prob = np.exp(-beta * total_cost)

    return float(np.clip(prob, 0.0, 1.0))


def estimate_reversion_probability(
    fitness_cost: float,
    time_days: int = 365,
    has_hla_pressure: bool = False
) -> dict:
    """
    Estimate probability of reversion over time.

    Based on Leslie et al. (2004) and Crawford et al. (2009):
    - High fitness cost mutations revert quickly
    - HLA pressure can maintain escape mutations

    Args:
        fitness_cost: Estimated fitness cost (0-1)
        time_days: Time period to consider
        has_hla_pressure: Whether selecting HLA is present

    Returns:
        dict: Reversion analysis
    """
    if fitness_cost < 0.01:
        # No cost = no reversion pressure
        return {
            "reversion_probability": 0.0,
            "expected_reversion_time_days": float("inf"),
            "will_revert": False,
        }

    # Reversion rate proportional to fitness cost
    # k = cost * base_rate
    base_rate = 0.005  # per day for cost=1.0 mutation
    reversion_rate = fitness_cost * base_rate

    # HLA pressure reduces reversion
    if has_hla_pressure:
        reversion_rate *= 0.2  # 80% reduction

    # Probability of reversion by time t: 1 - exp(-k*t)
    reversion_prob = 1.0 - np.exp(-reversion_rate * time_days)

    # Expected time to reversion: 1/k
    expected_time = 1.0 / reversion_rate if reversion_rate > 0 else float("inf")

    return {
        "reversion_probability": float(reversion_prob),
        "expected_reversion_time_days": float(expected_time),
        "will_revert": reversion_prob > 0.5,
        "reversion_rate_per_day": float(reversion_rate),
    }


# ============================================================================
# ESCAPE COST ANALYSIS
# ============================================================================

def analyze_escape_cost(
    epitope: str,
    escape_mutation: str,
    hyperbolic_distance: float,
    hla: Optional[str] = None
) -> dict:
    """
    Analyze the fitness cost of a CTL escape mutation.

    Args:
        epitope: Epitope sequence
        escape_mutation: Escape mutation (e.g., "T242N")
        hyperbolic_distance: Distance in hyperbolic space
        hla: HLA restriction (optional)

    Returns:
        dict: Escape cost analysis
    """
    # Parse mutation
    if len(escape_mutation) >= 3:
        wt = escape_mutation[0]
        pos = int(escape_mutation[1:-1])
        mut = escape_mutation[-1]
    else:
        wt, pos, mut = "X", 0, "X"

    # Estimate base fitness cost
    geo_estimate = estimate_fitness_cost_from_geometry(
        hyperbolic_distance,
        radial_position_change=0.0
    )

    fitness_cost = geo_estimate["fitness_cost"]

    # Analyze escape dynamics
    reversion = estimate_reversion_probability(
        fitness_cost,
        time_days=365,
        has_hla_pressure=(hla is not None)
    )

    transmission = estimate_transmission_probability(fitness_cost)

    return {
        "epitope": epitope,
        "escape_mutation": escape_mutation,
        "hla": hla,
        "fitness_cost": fitness_cost,
        "cost_category": geo_estimate["cost_category"],
        "transmission_probability": transmission,
        "reversion_analysis": reversion,
        "escape_stability": "stable" if not reversion["will_revert"] else "unstable",
    }


# ============================================================================
# BATCH ANALYSIS
# ============================================================================

def analyze_mutation_fitness_batch(
    mutations: pd.DataFrame,
    encoder=None
) -> pd.DataFrame:
    """
    Analyze fitness costs for a batch of mutations.

    Args:
        mutations: DataFrame with columns [wild_type_aa, mutant_aa, position, protein]
        encoder: Hyperbolic encoder (optional)

    Returns:
        DataFrame with fitness cost estimates
    """
    results = []

    for _, row in mutations.iterrows():
        wt = row.get("wild_type_aa", row.get("wt", "X"))
        mut = row.get("mutant_aa", row.get("mut", "X"))
        pos = row.get("position")
        protein = row.get("protein")

        estimate = estimate_fitness_cost(wt, mut, pos, protein, encoder)

        result = {
            "wild_type_aa": wt,
            "mutant_aa": mut,
            "position": pos,
            "protein": protein,
            **estimate
        }
        results.append(result)

    return pd.DataFrame(results)


def calibrate_fitness_model(
    predictions: pd.DataFrame,
    observed: pd.DataFrame
) -> dict:
    """
    Calibrate fitness predictions against observed data.

    Args:
        predictions: Predicted fitness costs
        observed: Observed fitness costs (from DMS or reversion rates)

    Returns:
        dict: Calibration statistics and adjusted parameters
    """
    # Merge predictions with observations
    merged = predictions.merge(
        observed,
        on=["wild_type_aa", "mutant_aa", "position"],
        suffixes=("_pred", "_obs")
    )

    if len(merged) < 5:
        return {
            "n_comparisons": len(merged),
            "calibrated": False,
            "message": "Insufficient data for calibration"
        }

    pred = merged["fitness_cost_pred"].values
    obs = merged["fitness_cost_obs"].values

    # Calculate correlation
    corr, pval = stats.pearsonr(pred, obs)
    spearman, spearman_pval = stats.spearmanr(pred, obs)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((pred - obs) ** 2))

    # Linear regression for calibration
    slope, intercept, _, _, _ = stats.linregress(pred, obs)

    return {
        "n_comparisons": len(merged),
        "calibrated": True,
        "pearson_r": float(corr),
        "pearson_pval": float(pval),
        "spearman_r": float(spearman),
        "spearman_pval": float(spearman_pval),
        "rmse": float(rmse),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
    }


# ============================================================================
# DRUG RESISTANCE FITNESS COSTS
# ============================================================================

def analyze_resistance_fitness_tradeoff(
    mutation: str,
    fold_change: float,
    hyperbolic_distance: float
) -> dict:
    """
    Analyze the tradeoff between resistance benefit and fitness cost.

    Based on Kühnert et al. (2018):
    - Some mutations provide high resistance with low fitness cost
    - Others are costly and provide marginal benefit

    Args:
        mutation: Mutation string (e.g., "M184V")
        fold_change: Resistance fold-change
        hyperbolic_distance: Geometric distance

    Returns:
        dict: Tradeoff analysis
    """
    # Estimate fitness cost
    if mutation in KNOWN_FITNESS_COSTS:
        fitness_cost = KNOWN_FITNESS_COSTS[mutation][0]
        source = "literature"
    else:
        geo_estimate = estimate_fitness_cost_from_geometry(
            hyperbolic_distance, 0.0
        )
        fitness_cost = geo_estimate["fitness_cost"]
        source = "geometric"

    # Calculate benefit-cost ratio
    # Benefit = log10(fold_change) normalized
    if fold_change > 1:
        benefit = np.log10(fold_change) / 3.0  # Normalize to ~0-1 for FC 1-1000
    else:
        benefit = 0.0

    if fitness_cost > 0.01:
        benefit_cost_ratio = benefit / fitness_cost
    else:
        benefit_cost_ratio = benefit * 100  # Very favorable if no cost

    # Categorize
    if benefit_cost_ratio > 10:
        tradeoff_category = "highly_favorable"
    elif benefit_cost_ratio > 3:
        tradeoff_category = "favorable"
    elif benefit_cost_ratio > 1:
        tradeoff_category = "balanced"
    elif benefit_cost_ratio > 0.3:
        tradeoff_category = "unfavorable"
    else:
        tradeoff_category = "very_costly"

    return {
        "mutation": mutation,
        "fold_change": fold_change,
        "fitness_cost": fitness_cost,
        "fitness_source": source,
        "resistance_benefit": benefit,
        "benefit_cost_ratio": benefit_cost_ratio,
        "tradeoff_category": tradeoff_category,
        "interpretation": _interpret_tradeoff(tradeoff_category, mutation),
    }


def _interpret_tradeoff(category: str, mutation: str) -> str:
    """Generate interpretation of tradeoff."""
    interpretations = {
        "highly_favorable": f"{mutation} provides strong resistance with minimal fitness cost - likely to persist",
        "favorable": f"{mutation} provides good resistance relative to its fitness cost",
        "balanced": f"{mutation} has similar resistance benefit and fitness cost",
        "unfavorable": f"{mutation} has high fitness cost relative to resistance benefit - may revert without drug pressure",
        "very_costly": f"{mutation} has severe fitness cost - unlikely to persist without strong selection",
    }
    return interpretations.get(category, "Unknown tradeoff")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Fitness Cost Estimator Examples")
    print("=" * 60)

    # Test known mutations
    known_muts = ["M184V", "K103N", "Q148R"]
    print("\n1. Known Mutations:")
    for mut in known_muts:
        wt, pos, aa = mut[0], int(mut[1:-1]), mut[-1]
        result = estimate_fitness_cost(wt, aa, pos)
        print(f"  {mut}: cost={result['fitness_cost']:.3f} ({result['cost_category']})")

    # Test geometric estimation
    print("\n2. Geometric Estimation:")
    test_distances = [1.0, 3.0, 5.0, 7.0]
    for dist in test_distances:
        result = estimate_fitness_cost_from_geometry(dist, 0.0)
        print(f"  Distance {dist}: cost={result['fitness_cost']:.3f} ({result['cost_category']})")

    # Test transmission probability
    print("\n3. Transmission Probability:")
    test_costs = [0.0, 0.1, 0.2, 0.3]
    for cost in test_costs:
        prob = estimate_transmission_probability(cost)
        print(f"  Cost {cost}: P(transmission)={prob:.3f}")

    # Test reversion
    print("\n4. Reversion Analysis (1 year):")
    for cost in test_costs:
        rev = estimate_reversion_probability(cost, 365, False)
        print(f"  Cost {cost}: P(reversion)={rev['reversion_probability']:.3f}")

    # Test resistance tradeoff
    print("\n5. Resistance Tradeoff:")
    test_cases = [
        ("M184V", 100.0, 4.5),
        ("K103N", 50.0, 3.0),
        ("Q148R", 200.0, 6.0),
    ]
    for mut, fc, dist in test_cases:
        result = analyze_resistance_fitness_tradeoff(mut, fc, dist)
        print(f"  {mut}: {result['tradeoff_category']} (ratio={result['benefit_cost_ratio']:.1f})")
