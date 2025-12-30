# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""B1: Pathogen-Specific AMP Design

Research Idea Implementation - Carlos Brizuela

This module extends the NSGA-II optimizer to design antimicrobial peptides
targeting specific WHO priority pathogens. Each pathogen has distinct membrane
composition and optimal AMP characteristics.

Target Pathogens (WHO Priority):
1. Acinetobacter baumannii (Critical - Carbapenem-resistant)
2. Pseudomonas aeruginosa (Critical - MDR)
3. Enterobacteriaceae (Critical - Carbapenem-resistant)
4. Staphylococcus aureus (High - MRSA)
5. Helicobacter pylori (High - Clarithromycin-resistant)

Key Features:
1. Pathogen-specific activity prediction models (trained on DRAMP data)
2. Multi-objective optimization (activity, toxicity, stability)
3. Pareto front analysis per pathogen
4. Cross-pathogen activity assessment

Usage:
    python scripts/B1_pathogen_specific_design.py \
        --pathogen "A_baumannii" \
        --generations 100 \
        --output results/pathogen_specific/

    # Use trained ML models (recommended):
    python scripts/B1_pathogen_specific_design.py --use-dramp --pathogen "A_baumannii"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sys
from pathlib import Path

# Add shared module to path
_script_dir = Path(__file__).parent
_deliverables_dir = _script_dir.parent.parent
sys.path.insert(0, str(_deliverables_dir))

# Import shared utilities
from shared.peptide_utils import (
    AA_PROPERTIES,
    compute_peptide_properties,
    compute_ml_features,
    decode_latent_to_sequence,
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
_TRAINED_MODELS: dict = {}


def load_trained_models() -> dict:
    """Load trained activity prediction models from DRAMP data.

    Returns:
        Dictionary mapping pathogen keys to trained model objects.
    """
    global _TRAINED_MODELS

    if _TRAINED_MODELS:
        return _TRAINED_MODELS

    if not HAS_JOBLIB:
        print("Warning: joblib not available, falling back to heuristic models")
        return {}

    # Find models directory
    models_dir = Path(__file__).parent.parent / "models"
    if not models_dir.exists():
        print(f"Warning: Models directory not found at {models_dir}")
        return {}

    # Map pathogen keys to model files
    pathogen_model_map = {
        "A_baumannii": "activity_acinetobacter.joblib",
        "P_aeruginosa": "activity_pseudomonas.joblib",
        "Enterobacteriaceae": "activity_escherichia.joblib",
        "S_aureus": "activity_staphylococcus.joblib",
        "H_pylori": "activity_general.joblib",  # Use general for H. pylori
    }

    for pathogen_key, model_file in pathogen_model_map.items():
        model_path = models_dir / model_file
        if model_path.exists():
            try:
                _TRAINED_MODELS[pathogen_key] = joblib.load(model_path)
                print(f"Loaded trained model for {pathogen_key}")
            except Exception as e:
                print(f"Warning: Could not load model {model_file}: {e}")

    # Load general model as fallback
    general_path = models_dir / "activity_general.joblib"
    if general_path.exists():
        try:
            _TRAINED_MODELS["_general"] = joblib.load(general_path)
            print("Loaded general activity model as fallback")
        except Exception:
            pass

    return _TRAINED_MODELS

# Import from existing NSGA-II implementation
try:
    from latent_nsga2 import Individual, LatentNSGA2, OptimizationConfig
except ImportError:
    # Define locally if import fails
    @dataclass
    class Individual:
        latent: np.ndarray
        objectives: np.ndarray
        rank: int = 0
        crowding_distance: float = 0.0
        decoded_sequence: Optional[str] = None

    @dataclass
    class OptimizationConfig:
        latent_dim: int = 16
        population_size: int = 200
        generations: int = 100
        crossover_prob: float = 0.9
        mutation_prob: float = 0.1
        mutation_sigma: float = 0.1
        latent_bounds: tuple = (-3.0, 3.0)
        seed: int = 42


# WHO Priority Pathogens with membrane characteristics
WHO_PRIORITY_PATHOGENS = {
    "A_baumannii": {
        "full_name": "Acinetobacter baumannii",
        "gram": "negative",
        "priority": "critical",
        "resistance": "Carbapenem-resistant",
        "membrane_features": {
            "LPS_abundance": 0.85,
            "phosphatidylethanolamine": 0.70,
            "phosphatidylglycerol": 0.20,
            "cardiolipin": 0.05,
            "net_charge": -0.6,
        },
        "optimal_amp_features": {
            "net_charge": (4, 8),  # Cationic
            "hydrophobicity": (0.3, 0.5),
            "amphipathicity": (0.4, 0.7),
            "length": (15, 30),
        },
    },
    "P_aeruginosa": {
        "full_name": "Pseudomonas aeruginosa",
        "gram": "negative",
        "priority": "critical",
        "resistance": "MDR",
        "membrane_features": {
            "LPS_abundance": 0.90,
            "phosphatidylethanolamine": 0.65,
            "phosphatidylglycerol": 0.25,
            "cardiolipin": 0.08,
            "net_charge": -0.7,
        },
        "optimal_amp_features": {
            "net_charge": (5, 9),
            "hydrophobicity": (0.35, 0.55),
            "amphipathicity": (0.5, 0.8),
            "length": (18, 35),
        },
    },
    "Enterobacteriaceae": {
        "full_name": "Enterobacteriaceae (E. coli, Klebsiella)",
        "gram": "negative",
        "priority": "critical",
        "resistance": "Carbapenem-resistant",
        "membrane_features": {
            "LPS_abundance": 0.88,
            "phosphatidylethanolamine": 0.72,
            "phosphatidylglycerol": 0.18,
            "cardiolipin": 0.06,
            "net_charge": -0.55,
        },
        "optimal_amp_features": {
            "net_charge": (3, 7),
            "hydrophobicity": (0.25, 0.45),
            "amphipathicity": (0.35, 0.65),
            "length": (12, 25),
        },
    },
    "S_aureus": {
        "full_name": "Staphylococcus aureus (MRSA)",
        "gram": "positive",
        "priority": "high",
        "resistance": "Methicillin-resistant",
        "membrane_features": {
            "teichoic_acid": 0.40,
            "phosphatidylglycerol": 0.55,
            "lysyl_PG": 0.30,
            "cardiolipin": 0.10,
            "net_charge": -0.3,
        },
        "optimal_amp_features": {
            "net_charge": (2, 6),
            "hydrophobicity": (0.4, 0.6),
            "amphipathicity": (0.3, 0.6),
            "length": (10, 22),
        },
    },
    "H_pylori": {
        "full_name": "Helicobacter pylori",
        "gram": "negative",
        "priority": "high",
        "resistance": "Clarithromycin-resistant",
        "membrane_features": {
            "LPS_abundance": 0.75,
            "phosphatidylethanolamine": 0.60,
            "phosphatidylglycerol": 0.30,
            "cholesterol_glucosides": 0.15,
            "net_charge": -0.4,
        },
        "optimal_amp_features": {
            "net_charge": (2, 5),
            "hydrophobicity": (0.35, 0.50),
            "amphipathicity": (0.4, 0.65),
            "length": (12, 20),
        },
    },
}


# AA_PROPERTIES, decode_latent_to_sequence, compute_peptide_properties,
# and compute_ml_features are now imported from shared.peptide_utils


def create_pathogen_activity_predictor(pathogen: str) -> Callable[[np.ndarray], float]:
    """Create activity prediction function for specific pathogen.

    Uses trained ML models when USE_TRAINED_MODELS is True and models are available.
    Falls back to heuristic scoring based on optimal features for the target pathogen.
    """
    if pathogen not in WHO_PRIORITY_PATHOGENS:
        raise ValueError(f"Unknown pathogen: {pathogen}")

    pathogen_info = WHO_PRIORITY_PATHOGENS[pathogen]
    optimal = pathogen_info["optimal_amp_features"]
    membrane = pathogen_info["membrane_features"]

    # Try to get trained model
    trained_model = None
    if USE_TRAINED_MODELS:
        models = load_trained_models()
        trained_model = models.get(pathogen) or models.get("_general")
        if trained_model:
            print(f"Using trained ML model for {pathogen}")
        else:
            print(f"No trained model available for {pathogen}, using heuristics")

    def predict_activity(z: np.ndarray) -> float:
        """Predict activity against pathogen (lower = better, to minimize)."""
        # Decode latent to sequence
        sequence = decode_latent_to_sequence(z)

        # Use trained model if available
        if trained_model is not None:
            try:
                features = compute_ml_features(sequence).reshape(1, -1)
                # Model predicts MIC (lower MIC = more active)
                # We want to minimize, so lower prediction = better
                prediction = trained_model.predict(features)[0]
                # Return negative because lower MIC is better (we want high activity)
                # Scale to reasonable range for optimization
                return -prediction
            except Exception:
                pass  # Fall through to heuristic

        # Heuristic fallback
        props = compute_peptide_properties(sequence)

        # Score based on optimal features
        score = 0.0

        # Charge score (within optimal range = good)
        charge = props["net_charge"]
        charge_min, charge_max = optimal["net_charge"]
        if charge < charge_min:
            score += (charge_min - charge) ** 2
        elif charge > charge_max:
            score += (charge - charge_max) ** 2

        # Hydrophobicity score
        hydro = props["hydrophobicity"]
        hydro_min, hydro_max = optimal["hydrophobicity"]
        if hydro < hydro_min:
            score += (hydro_min - hydro) ** 2
        elif hydro > hydro_max:
            score += (hydro - hydro_max) ** 2

        # Length penalty
        length = props["length"]
        len_min, len_max = optimal["length"]
        if length < len_min:
            score += (len_min - length) * 0.1
        elif length > len_max:
            score += (length - len_max) * 0.1

        # Membrane interaction bonus for gram-negative (LPS binding)
        if pathogen_info["gram"] == "negative":
            # Cationic peptides bind LPS better
            if charge >= 4:
                score -= 0.2 * membrane["LPS_abundance"]

        return score

    return predict_activity


def create_toxicity_predictor() -> Callable[[np.ndarray], float]:
    """Create host toxicity prediction function."""

    def predict_toxicity(z: np.ndarray) -> float:
        """Predict host cell toxicity (lower = safer)."""
        sequence = decode_latent_to_sequence(z)
        props = compute_peptide_properties(sequence)

        toxicity = 0.0

        # High hydrophobicity increases toxicity
        toxicity += max(0, props["hydrophobicity"] - 0.5) * 2

        # Very high charge can be toxic
        toxicity += max(0, abs(props["net_charge"]) - 8) * 0.5

        # Long peptides may be more toxic
        toxicity += max(0, props["length"] - 30) * 0.1

        # Count problematic motifs
        sequence = sequence.upper()
        # Consecutive hydrophobic residues
        hydrophobic = set("AILMFVW")
        max_hydro_run = 0
        current_run = 0
        for aa in sequence:
            if aa in hydrophobic:
                current_run += 1
                max_hydro_run = max(max_hydro_run, current_run)
            else:
                current_run = 0

        if max_hydro_run > 5:
            toxicity += max_hydro_run - 5

        return toxicity

    return predict_toxicity


def create_stability_predictor() -> Callable[[np.ndarray], float]:
    """Create peptide stability/validity prediction function."""

    def predict_instability(z: np.ndarray) -> float:
        """Predict instability (lower = more stable)."""
        # Penalize extreme latent values
        latent_penalty = np.sum(z**2) / len(z)

        # Decode and check sequence quality
        sequence = decode_latent_to_sequence(z)

        instability = latent_penalty

        # Penalize rare amino acids
        rare_aa = set("CMW")
        rare_count = sum(1 for aa in sequence if aa in rare_aa)
        instability += rare_count * 0.1

        # Penalize proline in middle (helix breaker)
        if "P" in sequence[3:-3]:
            instability += 0.2

        return instability

    return predict_instability


def run_pathogen_optimization(
    pathogen: str,
    config: Optional[OptimizationConfig] = None,
    verbose: bool = True,
) -> tuple[list[Individual], dict]:
    """Run NSGA-II optimization for a specific pathogen.

    Args:
        pathogen: Target pathogen key
        config: Optimization configuration
        verbose: Print progress

    Returns:
        Tuple of (pareto_front, results_dict)
    """
    if config is None:
        config = OptimizationConfig(
            latent_dim=16,
            population_size=100,
            generations=50,
            seed=42,
        )

    # Create objective functions
    activity_fn = create_pathogen_activity_predictor(pathogen)
    toxicity_fn = create_toxicity_predictor()
    stability_fn = create_stability_predictor()

    objectives = [activity_fn, toxicity_fn, stability_fn]

    # Initialize optimizer
    optimizer = LatentNSGA2Simplified(config, objectives)

    if verbose:
        info = WHO_PRIORITY_PATHOGENS[pathogen]
        print(f"\n{'='*60}")
        print(f"PATHOGEN-SPECIFIC AMP DESIGN: {info['full_name']}")
        print(f"{'='*60}")
        print(f"Priority: {info['priority']}")
        print(f"Resistance: {info['resistance']}")
        print(f"Gram type: {info['gram']}")
        print(f"\nOptimization parameters:")
        print(f"  Population: {config.population_size}")
        print(f"  Generations: {config.generations}")
        print(f"  Objectives: activity, toxicity, stability")
        print()

    # Run optimization
    pareto_front = optimizer.run(verbose=verbose)

    # Decode sequences
    for ind in pareto_front:
        ind.decoded_sequence = decode_latent_to_sequence(ind.latent)

    # Compile results
    results = {
        "pathogen": pathogen,
        "pathogen_info": WHO_PRIORITY_PATHOGENS[pathogen],
        "config": {
            "population_size": config.population_size,
            "generations": config.generations,
            "latent_dim": config.latent_dim,
        },
        "pareto_size": len(pareto_front),
        "candidates": [
            {
                "rank": i + 1,
                "sequence": ind.decoded_sequence,
                "properties": compute_peptide_properties(ind.decoded_sequence),
                "objectives": {
                    "activity": float(ind.objectives[0]),
                    "toxicity": float(ind.objectives[1]),
                    "stability": float(ind.objectives[2]),
                },
                "latent": ind.latent.tolist(),
            }
            for i, ind in enumerate(pareto_front[:10])  # Top 10
        ],
    }

    return pareto_front, results


class LatentNSGA2Simplified:
    """Simplified NSGA-II for pathogen-specific optimization."""

    def __init__(
        self,
        config: OptimizationConfig,
        objectives: list[Callable[[np.ndarray], float]],
    ):
        self.config = config
        self.objectives = objectives
        self.n_objectives = len(objectives)
        np.random.seed(config.seed)

    def initialize_population(self) -> list[Individual]:
        population = []
        for _ in range(self.config.population_size):
            latent = np.random.uniform(
                self.config.latent_bounds[0],
                self.config.latent_bounds[1],
                size=self.config.latent_dim,
            )
            individual = Individual(
                latent=latent,
                objectives=np.zeros(self.n_objectives),
            )
            population.append(individual)
        return population

    def evaluate(self, population: list[Individual]) -> None:
        for ind in population:
            ind.objectives = np.array([obj(ind.latent) for obj in self.objectives])

    def dominates(self, p: Individual, q: Individual) -> bool:
        at_least_one_better = False
        for pi, qi in zip(p.objectives, q.objectives):
            if pi > qi:
                return False
            if pi < qi:
                at_least_one_better = True
        return at_least_one_better

    def fast_non_dominated_sort(self, population: list[Individual]) -> list[list[Individual]]:
        n = len(population)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.dominates(population[i], population[j]):
                    dominated_solutions[i].append(j)
                elif self.dominates(population[j], population[i]):
                    domination_count[i] += 1

            if domination_count[i] == 0:
                population[i].rank = 0
                fronts[0].append(population[i])

        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i, p in enumerate(population):
                if p.rank == current_front:
                    for j in dominated_solutions[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            population[j].rank = current_front + 1
                            next_front.append(population[j])
            current_front += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return [f for f in fronts if f]

    def crowding_distance(self, front: list[Individual]) -> None:
        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float("inf")
            return

        for ind in front:
            ind.crowding_distance = 0.0

        for m in range(self.n_objectives):
            front.sort(key=lambda x: x.objectives[m])
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            obj_range = front[-1].objectives[m] - front[0].objectives[m]
            if obj_range == 0:
                continue

            for i in range(1, len(front) - 1):
                front[i].crowding_distance += (
                    front[i + 1].objectives[m] - front[i - 1].objectives[m]
                ) / obj_range

    def tournament_select(self, population: list[Individual]) -> Individual:
        i, j = np.random.choice(len(population), 2, replace=False)
        a, b = population[i], population[j]
        if a.rank < b.rank:
            return a
        elif b.rank < a.rank:
            return b
        return a if a.crowding_distance > b.crowding_distance else b

    def crossover(self, p1: Individual, p2: Individual) -> Individual:
        if np.random.random() > self.config.crossover_prob:
            return Individual(latent=p1.latent.copy(), objectives=np.zeros(self.n_objectives))

        alpha = np.random.random(self.config.latent_dim)
        child_latent = alpha * p1.latent + (1 - alpha) * p2.latent

        return Individual(latent=child_latent, objectives=np.zeros(self.n_objectives))

    def mutate(self, ind: Individual) -> Individual:
        mutant = ind.latent.copy()
        for i in range(self.config.latent_dim):
            if np.random.random() < self.config.mutation_prob:
                mutant[i] += np.random.normal(0, self.config.mutation_sigma)
                mutant[i] = np.clip(
                    mutant[i], self.config.latent_bounds[0], self.config.latent_bounds[1]
                )
        return Individual(latent=mutant, objectives=np.zeros(self.n_objectives))

    def run(self, verbose: bool = True) -> list[Individual]:
        population = self.initialize_population()
        self.evaluate(population)

        for gen in range(self.config.generations):
            # Create offspring
            offspring = []
            while len(offspring) < self.config.population_size:
                p1 = self.tournament_select(population)
                p2 = self.tournament_select(population)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                offspring.append(child)

            self.evaluate(offspring)

            # Combine and select
            combined = population + offspring
            fronts = self.fast_non_dominated_sort(combined)

            population = []
            for front in fronts:
                self.crowding_distance(front)
                if len(population) + len(front) <= self.config.population_size:
                    population.extend(front)
                else:
                    front.sort(key=lambda x: x.crowding_distance, reverse=True)
                    remaining = self.config.population_size - len(population)
                    population.extend(front[:remaining])
                    break

            if verbose and gen % 10 == 0:
                pareto_size = len([p for p in population if p.rank == 0])
                best = np.min([p.objectives for p in population], axis=0)
                print(f"Gen {gen:4d}: Pareto={pareto_size}, Best={best}")

        return [p for p in population if p.rank == 0]


def export_results(results: dict, output_dir: Path) -> None:
    """Export optimization results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pathogen = results["pathogen"]

    # JSON export
    json_path = output_dir / f"{pathogen}_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exported results to {json_path}")

    # CSV export
    if HAS_PANDAS:
        candidates = results["candidates"]
        records = []
        for c in candidates:
            record = {
                "rank": c["rank"],
                "sequence": c["sequence"],
                "length": c["properties"]["length"],
                "net_charge": c["properties"]["net_charge"],
                "hydrophobicity": c["properties"]["hydrophobicity"],
                "activity_score": c["objectives"]["activity"],
                "toxicity_score": c["objectives"]["toxicity"],
                "stability_score": c["objectives"]["stability"],
            }
            records.append(record)

        df = pd.DataFrame(records)
        csv_path = output_dir / f"{pathogen}_candidates.csv"
        df.to_csv(csv_path, index=False)
        print(f"Exported candidates to {csv_path}")

    # FASTA export
    fasta_path = output_dir / f"{pathogen}_peptides.fasta"
    with open(fasta_path, "w") as f:
        for c in results["candidates"]:
            f.write(f">{pathogen}_rank{c['rank']:02d}\n")
            f.write(f"{c['sequence']}\n")
    print(f"Exported FASTA to {fasta_path}")


def main():
    """Main entry point."""
    global USE_TRAINED_MODELS

    parser = argparse.ArgumentParser(description="Pathogen-Specific AMP Design")
    parser.add_argument(
        "--pathogen",
        type=str,
        default="A_baumannii",
        choices=list(WHO_PRIORITY_PATHOGENS.keys()),
        help="Target pathogen",
    )
    parser.add_argument(
        "--all_pathogens",
        action="store_true",
        help="Run for all WHO priority pathogens",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=100,
        help="Population size",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/pathogen_specific",
        help="Output directory",
    )
    parser.add_argument(
        "--use-dramp",
        action="store_true",
        help="Use trained ML models from DRAMP data (recommended)",
    )

    args = parser.parse_args()

    # Set global flag for trained models
    if args.use_dramp:
        USE_TRAINED_MODELS = True
        print("Using trained DRAMP activity models")
        load_trained_models()  # Pre-load models
    output_dir = Path(args.output)

    config = OptimizationConfig(
        population_size=args.population,
        generations=args.generations,
    )

    if args.all_pathogens:
        pathogens = list(WHO_PRIORITY_PATHOGENS.keys())
    else:
        pathogens = [args.pathogen]

    all_results = {}
    for pathogen in pathogens:
        pareto_front, results = run_pathogen_optimization(pathogen, config)
        export_results(results, output_dir)
        all_results[pathogen] = results

    # Summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    for pathogen, results in all_results.items():
        print(f"\n{pathogen}:")
        print(f"  Pareto front size: {results['pareto_size']}")
        if results["candidates"]:
            top = results["candidates"][0]
            print(f"  Top candidate: {top['sequence'][:30]}...")
            print(f"  Properties: charge={top['properties']['net_charge']:.1f}, "
                  f"hydro={top['properties']['hydrophobicity']:.2f}")


if __name__ == "__main__":
    main()
