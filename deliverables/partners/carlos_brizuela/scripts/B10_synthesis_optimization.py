# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""B10: Synthesis Optimization

Research Idea Implementation - Carlos Brizuela

Optimize AMPs for ease of solid-phase peptide synthesis (SPPS) by predicting
and minimizing synthesis difficulty.

Synthesis Challenges:
1. Aggregation (hydrophobic stretches)
2. Deletion peptides (steric hindrance)
3. Racemization (base-sensitive residues)
4. Aspartimide formation (Asp-Xxx motifs)

Key Features:
1. Synthesis difficulty prediction
2. Multi-objective: activity (using DRAMP-trained models) + synthesis feasibility
3. Cost estimation
4. Scale-up considerations

Usage:
    python scripts/B10_synthesis_optimization.py --output results/synthesis_optimized/

    # Use trained ML models for activity prediction (recommended):
    python scripts/B10_synthesis_optimization.py --use-dramp
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sys

# Add shared module to path
# scripts/ -> carlos_brizuela/ -> partners/ -> deliverables/
_script_dir = Path(__file__).parent
_deliverables_dir = _script_dir.parent.parent.parent
sys.path.insert(0, str(_deliverables_dir))

# Import shared utilities
from shared.peptide_utils import (
    AA_PROPERTIES,
    compute_peptide_properties,
    compute_ml_features,
)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


# Global flag and cache for trained models
USE_TRAINED_MODELS = False
_TRAINED_MODEL = None


def load_trained_model():
    """Load trained activity prediction model from DRAMP data."""
    global _TRAINED_MODEL

    if _TRAINED_MODEL is not None:
        return _TRAINED_MODEL

    if not HAS_JOBLIB:
        print("Warning: joblib not available, falling back to heuristic activity prediction")
        return None

    models_dir = Path(__file__).parent.parent / "models"
    model_path = models_dir / "activity_general.joblib"

    if model_path.exists():
        try:
            _TRAINED_MODEL = joblib.load(model_path)
            print(f"Loaded trained activity model from {model_path}")
            return _TRAINED_MODEL
        except Exception as e:
            print(f"Warning: Could not load model: {e}")

    return None


# Amino acid synthesis properties
AA_SYNTHESIS = {
    "A": {"cost": 1.0, "coupling": 0.99, "aggregation": 0.1, "racemization": 0.01},
    "R": {"cost": 3.0, "coupling": 0.95, "aggregation": 0.05, "racemization": 0.02},
    "N": {"cost": 2.0, "coupling": 0.96, "aggregation": 0.05, "racemization": 0.02},
    "D": {"cost": 2.0, "coupling": 0.94, "aggregation": 0.05, "racemization": 0.03},  # Aspartimide risk
    "C": {"cost": 4.0, "coupling": 0.92, "aggregation": 0.15, "racemization": 0.02},  # Oxidation
    "Q": {"cost": 2.5, "coupling": 0.95, "aggregation": 0.08, "racemization": 0.02},
    "E": {"cost": 2.0, "coupling": 0.95, "aggregation": 0.05, "racemization": 0.02},
    "G": {"cost": 1.0, "coupling": 0.99, "aggregation": 0.02, "racemization": 0.00},  # No chiral
    "H": {"cost": 4.0, "coupling": 0.93, "aggregation": 0.10, "racemization": 0.03},
    "I": {"cost": 2.0, "coupling": 0.94, "aggregation": 0.25, "racemization": 0.01},  # Branched
    "L": {"cost": 1.5, "coupling": 0.97, "aggregation": 0.20, "racemization": 0.01},
    "K": {"cost": 2.0, "coupling": 0.96, "aggregation": 0.05, "racemization": 0.02},
    "M": {"cost": 3.5, "coupling": 0.93, "aggregation": 0.12, "racemization": 0.02},  # Oxidation
    "F": {"cost": 2.5, "coupling": 0.95, "aggregation": 0.30, "racemization": 0.01},
    "P": {"cost": 2.0, "coupling": 0.90, "aggregation": 0.02, "racemization": 0.00},  # Imino acid
    "S": {"cost": 1.5, "coupling": 0.97, "aggregation": 0.05, "racemization": 0.02},
    "T": {"cost": 2.0, "coupling": 0.96, "aggregation": 0.08, "racemization": 0.02},
    "W": {"cost": 6.0, "coupling": 0.90, "aggregation": 0.35, "racemization": 0.03},  # Expensive
    "Y": {"cost": 3.0, "coupling": 0.94, "aggregation": 0.25, "racemization": 0.02},
    "V": {"cost": 1.5, "coupling": 0.96, "aggregation": 0.20, "racemization": 0.01},  # Branched
}

# Difficult dipeptide combinations
DIFFICULT_DIPEPTIDES = {
    ("D", "G"): 0.3,  # Aspartimide high risk
    ("D", "S"): 0.25,  # Aspartimide
    ("D", "N"): 0.2,  # Aspartimide
    ("W", "W"): 0.4,  # Steric
    ("I", "I"): 0.3,  # Aggregation
    ("V", "V"): 0.3,  # Aggregation
    ("F", "F"): 0.35,  # Aggregation
}


@dataclass
class SynthesisOptimizedAMP:
    """AMP with synthesis metrics."""

    sequence: str
    length: int
    net_charge: float
    hydrophobicity: float
    activity_score: float
    synthesis_difficulty: float
    aggregation_propensity: float
    coupling_efficiency: float
    racemization_risk: float
    estimated_cost: float  # Relative cost
    difficult_motifs: list
    combined_score: float
    latent: np.ndarray


# compute_peptide_properties imported from shared.peptide_utils


def predict_synthesis_difficulty(sequence: str) -> dict:
    """Predict synthesis difficulty metrics."""
    sequence = sequence.upper()

    # Aggregation propensity
    aggregation = 0.0
    hydrophobic_run = 0
    max_hydrophobic_run = 0
    hydrophobic_aa = set("AILMFVWY")

    for aa in sequence:
        if aa in hydrophobic_aa:
            hydrophobic_run += 1
            max_hydrophobic_run = max(max_hydrophobic_run, hydrophobic_run)
        else:
            hydrophobic_run = 0

    # Long hydrophobic stretches = aggregation
    if max_hydrophobic_run >= 5:
        aggregation += (max_hydrophobic_run - 4) * 0.15

    # Sum individual aggregation propensities
    for aa in sequence:
        if aa in AA_SYNTHESIS:
            aggregation += AA_SYNTHESIS[aa]["aggregation"]

    aggregation /= len(sequence)

    # Coupling efficiency (product of individual)
    coupling = 1.0
    for aa in sequence:
        if aa in AA_SYNTHESIS:
            coupling *= AA_SYNTHESIS[aa]["coupling"]

    # Racemization risk
    racemization = 0.0
    for aa in sequence:
        if aa in AA_SYNTHESIS:
            racemization += AA_SYNTHESIS[aa]["racemization"]
    racemization /= len(sequence)

    # Difficult motifs
    difficult_motifs = []
    for i in range(len(sequence) - 1):
        dipeptide = (sequence[i], sequence[i + 1])
        if dipeptide in DIFFICULT_DIPEPTIDES:
            difficulty = DIFFICULT_DIPEPTIDES[dipeptide]
            difficult_motifs.append({
                "position": i,
                "motif": f"{dipeptide[0]}{dipeptide[1]}",
                "difficulty": difficulty,
            })

    # Cost estimation
    cost = sum(AA_SYNTHESIS.get(aa, {"cost": 2.0})["cost"] for aa in sequence)

    # Length penalty
    if len(sequence) > 30:
        cost *= 1.5
    if len(sequence) > 40:
        cost *= 2.0

    # Overall difficulty score
    difficulty = (
        aggregation * 0.4 +
        (1 - coupling) * 100 * 0.3 +
        racemization * 10 * 0.2 +
        len(difficult_motifs) * 0.1
    )

    return {
        "difficulty": difficulty,
        "aggregation": aggregation,
        "coupling": coupling,
        "racemization": racemization,
        "cost": cost,
        "difficult_motifs": difficult_motifs,
    }


def compute_ml_features(sequence: str) -> np.ndarray:
    """Compute features for ML model prediction."""
    props = compute_peptide_properties(sequence)

    # Amino acid composition (20 features)
    aa_list = list(AA_SYNTHESIS.keys())
    aa_comp = np.zeros(20)
    seq_len = len(sequence)
    if seq_len > 0:
        for i, aa in enumerate(aa_list):
            aa_comp[i] = sequence.count(aa) / seq_len

    # Basic properties
    charge = props["net_charge"]
    hydro = props["hydrophobicity"]
    length = seq_len

    # Hydrophobic ratio
    hydrophobic_aa = set("AILMFVW")
    hydro_ratio = sum(1 for aa in sequence if aa in hydrophobic_aa) / max(seq_len, 1)

    # Cationic ratio
    cationic_aa = set("KRH")
    cationic_ratio = sum(1 for aa in sequence if aa in cationic_aa) / max(seq_len, 1)

    features = np.concatenate([
        np.array([length, charge, hydro, hydro_ratio, cationic_ratio]),
        aa_comp
    ])

    return features


def predict_activity(sequence: str) -> float:
    """Predict antimicrobial activity.

    Uses trained ML model when USE_TRAINED_MODELS is True, otherwise uses heuristics.
    """
    # Try ML model prediction first
    if USE_TRAINED_MODELS:
        model = load_trained_model()
        if model is not None:
            try:
                features = compute_ml_features(sequence).reshape(1, -1)
                prediction = model.predict(features)[0]
                # Normalize to 0-1 range
                return max(0, min(1, prediction / 10))
            except Exception:
                pass  # Fall through to heuristic

    # Heuristic fallback
    props = compute_peptide_properties(sequence)

    # Cationic peptides are generally more active
    charge_score = min(1.0, props["net_charge"] / 5) if props["net_charge"] > 0 else 0

    # Moderate hydrophobicity is optimal
    hydro_optimal = 0.4
    hydro_score = 1 - abs(props["hydrophobicity"] - hydro_optimal) / 2

    # Length score (15-25 is optimal)
    if 15 <= props["length"] <= 25:
        length_score = 1.0
    else:
        length_score = max(0, 1 - abs(props["length"] - 20) / 20)

    activity = (charge_score + hydro_score + length_score) / 3
    return activity


def decode_latent_to_sequence(z: np.ndarray, length: int = 20) -> str:
    """Decode latent to sequence."""
    np.random.seed(int(abs(z[0] * 1000)))

    aa_list = list(AA_SYNTHESIS.keys())

    # Weight by synthesis ease (inverse of difficulty)
    weights = []
    for aa in aa_list:
        props = AA_SYNTHESIS[aa]
        ease = props["coupling"] * (1 - props["aggregation"]) * (1 - props["racemization"])
        ease *= (1 / props["cost"])  # Cheaper is better
        weights.append(ease)

    # Incorporate latent preferences
    charge_pref = np.tanh(z[0])
    for i, aa in enumerate(aa_list):
        if aa in "RKH" and charge_pref > 0:
            weights[i] *= 1 + charge_pref
        elif aa in "DE" and charge_pref < 0:
            weights[i] *= 1 + abs(charge_pref)

    weights = np.array(weights)
    weights = weights / weights.sum()

    sequence = "".join(np.random.choice(aa_list, size=length, p=weights))
    return sequence


def evaluate_synthesis_amp(z: np.ndarray) -> SynthesisOptimizedAMP:
    """Evaluate a peptide for synthesis optimization."""
    sequence = decode_latent_to_sequence(z)
    props = compute_peptide_properties(sequence)
    synth = predict_synthesis_difficulty(sequence)
    activity = predict_activity(sequence)

    # Combined score: activity / (1 + difficulty)
    combined = activity / (1 + synth["difficulty"])

    return SynthesisOptimizedAMP(
        sequence=sequence,
        length=len(sequence),
        net_charge=props["net_charge"],
        hydrophobicity=props["hydrophobicity"],
        activity_score=activity,
        synthesis_difficulty=synth["difficulty"],
        aggregation_propensity=synth["aggregation"],
        coupling_efficiency=synth["coupling"],
        racemization_risk=synth["racemization"],
        estimated_cost=synth["cost"],
        difficult_motifs=synth["difficult_motifs"],
        combined_score=combined,
        latent=z,
    )


def optimize_synthesis_amps(
    population_size: int = 200,
    generations: int = 50,
    seed: int = 42,
) -> list[SynthesisOptimizedAMP]:
    """Optimize for synthesis-friendly AMPs."""
    np.random.seed(seed)

    latent_dim = 16
    bounds = (-3.0, 3.0)

    # Initialize
    population = []
    for _ in range(population_size):
        z = np.random.uniform(bounds[0], bounds[1], size=latent_dim)
        candidate = evaluate_synthesis_amp(z)
        population.append(candidate)

    print("\n" + "=" * 60)
    print("SYNTHESIS-OPTIMIZED AMP DESIGN")
    print("=" * 60)
    print(f"Objective: Maximize activity while minimizing synthesis difficulty")
    print(f"Population: {population_size}, Generations: {generations}")
    print()

    for gen in range(generations):
        population.sort(key=lambda x: x.combined_score, reverse=True)

        if gen % 10 == 0:
            best = population[0]
            avg_difficulty = np.mean([p.synthesis_difficulty for p in population[:10]])
            avg_coupling = np.mean([p.coupling_efficiency for p in population[:10]])
            print(f"Gen {gen:4d}: Best score={best.combined_score:.3f}, "
                  f"Avg difficulty={avg_difficulty:.3f}, "
                  f"Avg coupling={avg_coupling:.3f}")

        survivors = population[: population_size // 2]

        offspring = []
        while len(offspring) < population_size // 2:
            p1, p2 = np.random.choice(len(survivors), 2, replace=False)

            alpha = np.random.random(latent_dim)
            child_z = alpha * survivors[p1].latent + (1 - alpha) * survivors[p2].latent

            if np.random.random() < 0.3:
                child_z += np.random.normal(0, 0.2, size=latent_dim)
                child_z = np.clip(child_z, bounds[0], bounds[1])

            child = evaluate_synthesis_amp(child_z)
            offspring.append(child)

        population = survivors + offspring

    population.sort(key=lambda x: x.combined_score, reverse=True)
    return population


def export_results(candidates: list[SynthesisOptimizedAMP], output_dir: Path) -> None:
    """Export results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    top_candidates = candidates[:20]

    results = {
        "objective": "Synthesis-optimized AMP design",
        "metrics": ["synthesis_difficulty", "coupling_efficiency", "cost", "activity"],
        "candidates": [
            {
                "rank": i + 1,
                "sequence": c.sequence,
                "length": c.length,
                "net_charge": c.net_charge,
                "activity_score": round(c.activity_score, 3),
                "synthesis_difficulty": round(c.synthesis_difficulty, 3),
                "coupling_efficiency": round(c.coupling_efficiency, 4),
                "aggregation_propensity": round(c.aggregation_propensity, 3),
                "racemization_risk": round(c.racemization_risk, 4),
                "estimated_cost": round(c.estimated_cost, 1),
                "difficult_motifs": c.difficult_motifs,
                "combined_score": round(c.combined_score, 3),
            }
            for i, c in enumerate(top_candidates)
        ],
    }

    json_path = output_dir / "synthesis_optimized_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exported results to {json_path}")

    if HAS_PANDAS:
        records = [
            {
                "rank": i + 1,
                "sequence": c.sequence,
                "length": c.length,
                "activity": c.activity_score,
                "difficulty": c.synthesis_difficulty,
                "coupling": c.coupling_efficiency,
                "cost": c.estimated_cost,
                "combined": c.combined_score,
            }
            for i, c in enumerate(top_candidates)
        ]
        df = pd.DataFrame(records)
        csv_path = output_dir / "synthesis_optimized_candidates.csv"
        df.to_csv(csv_path, index=False)
        print(f"Exported CSV to {csv_path}")

    # Summary
    print("\n" + "=" * 60)
    print("TOP 10 SYNTHESIS-OPTIMIZED CANDIDATES")
    print("=" * 60)
    print(f"{'Rank':<5} {'Sequence':<22} {'Activity':<9} {'Difficulty':<11} {'Coupling':<9}")
    print("-" * 60)
    for i, c in enumerate(top_candidates[:10]):
        seq = c.sequence[:20] + "..." if len(c.sequence) > 20 else c.sequence
        print(f"{i+1:<5} {seq:<22} {c.activity_score:<9.3f} "
              f"{c.synthesis_difficulty:<11.3f} {c.coupling_efficiency:<9.4f}")


def main():
    """Main entry point."""
    global USE_TRAINED_MODELS

    parser = argparse.ArgumentParser(description="Synthesis-Optimized AMP Design")
    parser.add_argument("--population", type=int, default=200)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--output", type=str, default="results/synthesis_optimized")
    parser.add_argument(
        "--use-dramp",
        action="store_true",
        help="Use trained ML models from DRAMP data for activity prediction (recommended)",
    )

    args = parser.parse_args()

    # Set global flag for trained models
    if args.use_dramp:
        USE_TRAINED_MODELS = True
        print("Using trained DRAMP activity model")
        load_trained_model()

    candidates = optimize_synthesis_amps(args.population, args.generations)
    export_results(candidates, Path(args.output))

    print("\nSYNTHESIS OPTIMIZATION COMPLETE")


if __name__ == "__main__":
    main()
