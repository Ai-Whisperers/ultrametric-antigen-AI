# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""
Escape Probability Model

Models the probability of immune escape (CTL or antibody) over time.

Based on:
- Fryer HR, et al. (2010) - Modelling evolution and spread of escape mutants
- Ganusov VV, et al. (2011) - Fitness costs and CTL response diversity
- Hie B, et al. (2021) - Learning the language of viral evolution and escape

Key concepts:
- Escape rate depends on selection pressure (immune response strength)
- Escape is limited by fitness cost of mutations
- Geometric distance predicts both escape potential and fitness cost
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import odeint

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from fitness_cost_estimator import estimate_fitness_cost_from_geometry
except ImportError:
    def estimate_fitness_cost_from_geometry(dist, change):
        return {"fitness_cost": 0.1, "cost_category": "moderate"}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EscapeParameters:
    """Parameters for escape dynamics model."""
    selection_coefficient: float = 0.1  # s: selection advantage of escape
    fitness_cost: float = 0.1           # c: fitness cost of escape mutation
    mutation_rate: float = 3e-5         # μ: per nucleotide per replication
    population_size: float = 1e6        # N: effective viral population
    generation_time: float = 2.0        # days per generation

    @property
    def net_selection(self) -> float:
        """Net selection coefficient (s - c)."""
        return self.selection_coefficient - self.fitness_cost

    @property
    def fixation_probability(self) -> float:
        """Probability that escape mutation fixes (Kimura 1962)."""
        if abs(self.net_selection) < 1e-10:
            return 1.0 / self.population_size
        return (1 - np.exp(-2 * self.net_selection)) / \
               (1 - np.exp(-2 * self.net_selection * self.population_size))


# ============================================================================
# ESCAPE PROBABILITY CALCULATIONS
# ============================================================================

def calculate_escape_rate(
    selection_coefficient: float,
    fitness_cost: float,
    mutation_rate: float = 3e-5,
    n_escape_positions: int = 1
) -> float:
    """
    Calculate instantaneous escape rate.

    Based on Fryer et al. (2010):
    λ = μ * P_fix * n_positions
    where P_fix is the fixation probability

    Args:
        selection_coefficient: Selection advantage from escape
        fitness_cost: Fitness cost of escape mutation
        mutation_rate: Per-nucleotide mutation rate
        n_escape_positions: Number of positions that can escape

    Returns:
        float: Escape rate per generation
    """
    net_s = selection_coefficient - fitness_cost

    if net_s <= 0:
        # Deleterious mutation - very low escape rate
        return mutation_rate * n_escape_positions * 0.01

    # Fixation probability for beneficial mutation
    # P_fix ≈ 2s for s << 1 (Haldane approximation)
    p_fix = 2 * net_s if net_s < 0.1 else 1 - np.exp(-2 * net_s)

    return mutation_rate * p_fix * n_escape_positions


def model_escape_dynamics(
    params: EscapeParameters,
    time_days: np.ndarray
) -> dict:
    """
    Model escape dynamics over time using ODE.

    System:
    dx/dt = λ * (1 - x) - (d - s + c) * x

    where:
    - x = fraction of escape variants
    - λ = escape mutation rate
    - d = death rate (normalized to 1)
    - s = selection advantage
    - c = fitness cost

    Args:
        params: Escape parameters
        time_days: Time points to evaluate

    Returns:
        dict: Escape dynamics results
    """
    # Convert time to generations
    generations = time_days / params.generation_time

    def escape_ode(x, t, lam, net_s):
        """ODE for escape frequency dynamics."""
        # dx/dt = mutation supply - selection
        dxdt = lam * (1 - x) + net_s * x * (1 - x)
        return dxdt

    # Calculate escape rate
    escape_rate = calculate_escape_rate(
        params.selection_coefficient,
        params.fitness_cost
    )

    # Initial condition: very low escape frequency
    x0 = 1e-6

    # Solve ODE
    solution = odeint(
        escape_ode,
        x0,
        generations,
        args=(escape_rate, params.net_selection)
    )

    escape_freq = solution[:, 0]

    # Calculate escape probability (ever escaped)
    # P(escape by time t) = 1 - exp(-∫λdt)
    cumulative_rate = np.cumsum(np.full(len(generations), escape_rate))
    cumulative_rate *= (generations[1] - generations[0]) if len(generations) > 1 else 1
    escape_prob = 1 - np.exp(-cumulative_rate)

    return {
        "time_days": time_days,
        "escape_frequency": escape_freq,
        "escape_probability": escape_prob,
        "escape_rate": escape_rate,
        "net_selection": params.net_selection,
        "half_time_days": find_half_time(escape_freq, time_days),
    }


def find_half_time(frequency: np.ndarray, time: np.ndarray, threshold: float = 0.5) -> float:
    """Find time to reach threshold frequency."""
    above_threshold = np.where(frequency >= threshold)[0]
    if len(above_threshold) > 0:
        return time[above_threshold[0]]
    return float("inf")


def predict_escape_probability(
    epitope_data: dict,
    hyperbolic_distance: float,
    time_days: int = 365,
    hla_frequency: float = 0.1
) -> dict:
    """
    Predict escape probability for an epitope.

    Args:
        epitope_data: Epitope information
        hyperbolic_distance: Geometric distance to escape variants
        time_days: Prediction time horizon
        hla_frequency: Population frequency of restricting HLA

    Returns:
        dict: Escape prediction results
    """
    # Estimate fitness cost from geometry
    fitness_result = estimate_fitness_cost_from_geometry(
        hyperbolic_distance,
        radial_position_change=0.0
    )
    fitness_cost = fitness_result["fitness_cost"]

    # Selection coefficient depends on HLA frequency and immune pressure
    # Higher HLA frequency = stronger selection for escape
    base_selection = 0.3  # Maximum selection coefficient
    selection_coefficient = base_selection * hla_frequency * 10  # Scale by HLA freq

    # Create parameters
    params = EscapeParameters(
        selection_coefficient=min(selection_coefficient, 0.5),
        fitness_cost=fitness_cost
    )

    # Model dynamics
    time_points = np.linspace(0, time_days, 100)
    dynamics = model_escape_dynamics(params, time_points)

    # Get final probability
    final_prob = dynamics["escape_probability"][-1]
    final_freq = dynamics["escape_frequency"][-1]

    return {
        "epitope": epitope_data.get("epitope", "Unknown"),
        "hla": epitope_data.get("hla"),
        "escape_probability": float(final_prob),
        "final_escape_frequency": float(final_freq),
        "half_time_days": dynamics["half_time_days"],
        "fitness_cost": fitness_cost,
        "selection_coefficient": params.selection_coefficient,
        "net_selection": params.net_selection,
        "risk_category": categorize_escape_risk(final_prob, dynamics["half_time_days"]),
        "dynamics": dynamics,
    }


def categorize_escape_risk(probability: float, half_time: float) -> str:
    """Categorize escape risk."""
    if probability < 0.1:
        return "very_low"
    elif probability < 0.3:
        return "low"
    elif probability < 0.5:
        return "moderate"
    elif probability < 0.7:
        return "high"
    else:
        if half_time < 90:
            return "very_high_rapid"
        else:
            return "very_high"


# ============================================================================
# CTL ESCAPE SPECIFIC
# ============================================================================

def analyze_ctl_escape_landscape(
    epitopes: pd.DataFrame,
    encoder=None
) -> pd.DataFrame:
    """
    Analyze escape landscape for multiple CTL epitopes.

    Args:
        epitopes: DataFrame with epitope information
        encoder: Hyperbolic encoder (optional)

    Returns:
        DataFrame with escape predictions
    """
    results = []

    for _, row in epitopes.iterrows():
        epitope = row.get("Epitope", row.get("epitope", ""))
        hla = row.get("HLA", row.get("hla", ""))
        escape_velocity = row.get("escape_velocity", 0.3)

        # Use escape velocity as proxy for hyperbolic distance
        hyperbolic_distance = escape_velocity * 10  # Scale

        # Get HLA frequency (simplified)
        hla_freq = 0.1  # Default

        prediction = predict_escape_probability(
            {"epitope": epitope, "hla": hla},
            hyperbolic_distance,
            time_days=365,
            hla_frequency=hla_freq
        )

        results.append({
            "epitope": epitope,
            "hla": hla,
            "escape_velocity": escape_velocity,
            "escape_probability_1yr": prediction["escape_probability"],
            "half_time_days": prediction["half_time_days"],
            "fitness_cost": prediction["fitness_cost"],
            "net_selection": prediction["net_selection"],
            "risk_category": prediction["risk_category"],
        })

    return pd.DataFrame(results)


# ============================================================================
# ANTIBODY ESCAPE
# ============================================================================

def model_antibody_escape(
    antibody: str,
    epitope_positions: list[int],
    sequence_diversity: float,
    hyperbolic_features: Optional[dict] = None,
    time_days: int = 365
) -> dict:
    """
    Model escape from antibody neutralization.

    Based on Dingens et al. (2017) - comprehensive escape mapping.

    Args:
        antibody: Antibody name
        epitope_positions: Positions in antibody epitope
        sequence_diversity: Entropy at epitope positions
        hyperbolic_features: Geometric features (optional)
        time_days: Prediction time horizon

    Returns:
        dict: Antibody escape prediction
    """
    # Estimate escape potential from diversity
    # Higher diversity = more escape variants available
    n_escape_positions = len(epitope_positions)
    effective_escape_rate = sequence_diversity * n_escape_positions * 0.1

    # Fitness cost from hyperbolic features if available
    if hyperbolic_features:
        fitness_cost = hyperbolic_features.get("mean_radial_position", 0.5) * 0.2
    else:
        fitness_cost = 0.1

    # Antibody-specific selection pressure
    # Stronger neutralization = stronger selection for escape
    selection_coefficient = 0.2  # Base selection

    params = EscapeParameters(
        selection_coefficient=selection_coefficient,
        fitness_cost=fitness_cost,
        mutation_rate=effective_escape_rate / n_escape_positions
    )

    time_points = np.linspace(0, time_days, 100)
    dynamics = model_escape_dynamics(params, time_points)

    return {
        "antibody": antibody,
        "n_epitope_positions": n_escape_positions,
        "sequence_diversity": sequence_diversity,
        "escape_probability": float(dynamics["escape_probability"][-1]),
        "half_time_days": dynamics["half_time_days"],
        "fitness_cost": fitness_cost,
        "escape_resistance_score": 1.0 - dynamics["escape_probability"][-1],
    }


# ============================================================================
# COMBINATION THERAPY ESCAPE
# ============================================================================

def model_combination_escape(
    therapies: list[dict],
    time_days: int = 365
) -> dict:
    """
    Model escape from combination therapy (drugs or antibodies).

    Escape requires simultaneous mutations to all components.

    Args:
        therapies: List of therapy dicts with escape parameters
        time_days: Prediction time horizon

    Returns:
        dict: Combination escape analysis
    """
    # Individual escape probabilities
    individual_probs = []
    for therapy in therapies:
        params = EscapeParameters(
            selection_coefficient=therapy.get("selection", 0.2),
            fitness_cost=therapy.get("fitness_cost", 0.1)
        )
        time_points = np.linspace(0, time_days, 100)
        dynamics = model_escape_dynamics(params, time_points)
        individual_probs.append(dynamics["escape_probability"][-1])

    # For simultaneous escape, multiply probabilities
    # This assumes independence (conservative for cross-resistance)
    combined_escape_prob = np.prod(individual_probs)

    # Combined fitness cost is sum of individual costs
    combined_fitness = sum(t.get("fitness_cost", 0.1) for t in therapies)

    return {
        "n_components": len(therapies),
        "individual_escape_probs": individual_probs,
        "combined_escape_probability": float(combined_escape_prob),
        "combined_fitness_cost": combined_fitness,
        "protection_factor": 1.0 / max(combined_escape_prob, 1e-10),
        "recommendation": (
            "excellent" if combined_escape_prob < 0.001 else
            "good" if combined_escape_prob < 0.01 else
            "moderate" if combined_escape_prob < 0.1 else
            "poor"
        ),
    }


# ============================================================================
# VACCINE TARGET SCORING
# ============================================================================

def score_vaccine_target(
    epitope: str,
    hla_coverage: float,
    escape_probability: float,
    fitness_cost: float,
    conservation: float = 1.0,
    resistance_overlap: bool = False
) -> dict:
    """
    Score an epitope as a vaccine target.

    Combines multiple factors:
    - HLA coverage (higher = better)
    - Escape probability (lower = better)
    - Fitness cost of escape (higher = better)
    - Conservation (higher = better)
    - No resistance overlap (required)

    Args:
        epitope: Epitope sequence
        hla_coverage: Fraction of population covered
        escape_probability: Probability of escape
        fitness_cost: Cost of escape mutation
        conservation: Sequence conservation score
        resistance_overlap: Whether overlaps drug resistance

    Returns:
        dict: Vaccine target score
    """
    if resistance_overlap:
        return {
            "epitope": epitope,
            "score": 0.0,
            "rank_category": "excluded",
            "reason": "Overlaps drug resistance positions",
        }

    # Component scores (all 0-1, higher = better)
    coverage_score = hla_coverage
    escape_score = 1.0 - escape_probability
    fitness_score = fitness_cost  # Higher cost = better target
    conservation_score = conservation

    # Weighted combination
    weights = {
        "coverage": 0.25,
        "escape": 0.30,
        "fitness": 0.25,
        "conservation": 0.20,
    }

    total_score = (
        weights["coverage"] * coverage_score +
        weights["escape"] * escape_score +
        weights["fitness"] * fitness_score +
        weights["conservation"] * conservation_score
    )

    # Categorize
    if total_score >= 0.8:
        category = "excellent"
    elif total_score >= 0.6:
        category = "good"
    elif total_score >= 0.4:
        category = "moderate"
    else:
        category = "poor"

    return {
        "epitope": epitope,
        "score": float(total_score),
        "rank_category": category,
        "component_scores": {
            "coverage": coverage_score,
            "escape_resistance": escape_score,
            "fitness_cost": fitness_score,
            "conservation": conservation_score,
        },
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Escape Probability Model Examples")
    print("=" * 60)

    # Test escape rate calculation
    print("\n1. Escape Rate Calculation:")
    test_params = [
        (0.3, 0.1),  # Strong selection, low cost
        (0.2, 0.2),  # Balanced
        (0.1, 0.3),  # Net deleterious
    ]
    for s, c in test_params:
        rate = calculate_escape_rate(s, c)
        print(f"  s={s}, c={c}: rate={rate:.2e}")

    # Test escape dynamics
    print("\n2. Escape Dynamics (1 year):")
    params = EscapeParameters(selection_coefficient=0.2, fitness_cost=0.1)
    time = np.linspace(0, 365, 100)
    dynamics = model_escape_dynamics(params, time)
    print(f"  Net selection: {params.net_selection:.2f}")
    print(f"  Final escape frequency: {dynamics['escape_frequency'][-1]:.3f}")
    print(f"  Final escape probability: {dynamics['escape_probability'][-1]:.3f}")
    print(f"  Half-time: {dynamics['half_time_days']:.0f} days")

    # Test escape prediction
    print("\n3. Epitope Escape Prediction:")
    epitope_data = {"epitope": "SLYNTVATL", "hla": "A*02:01"}
    prediction = predict_escape_probability(epitope_data, hyperbolic_distance=5.0)
    print(f"  Epitope: {prediction['epitope']}")
    print(f"  Escape probability (1yr): {prediction['escape_probability']:.3f}")
    print(f"  Risk category: {prediction['risk_category']}")

    # Test combination escape
    print("\n4. Combination Therapy Escape:")
    therapies = [
        {"name": "Drug A", "selection": 0.2, "fitness_cost": 0.1},
        {"name": "Drug B", "selection": 0.2, "fitness_cost": 0.15},
    ]
    combo = model_combination_escape(therapies)
    print(f"  Individual probs: {[f'{p:.3f}' for p in combo['individual_escape_probs']]}")
    print(f"  Combined prob: {combo['combined_escape_probability']:.6f}")
    print(f"  Recommendation: {combo['recommendation']}")

    # Test vaccine target scoring
    print("\n5. Vaccine Target Scoring:")
    score = score_vaccine_target(
        epitope="SLYNTVATL",
        hla_coverage=0.35,
        escape_probability=0.2,
        fitness_cost=0.15,
        conservation=0.9,
        resistance_overlap=False
    )
    print(f"  Score: {score['score']:.3f}")
    print(f"  Category: {score['rank_category']}")
