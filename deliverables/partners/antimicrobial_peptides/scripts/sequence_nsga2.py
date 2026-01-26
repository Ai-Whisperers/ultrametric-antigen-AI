#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""SequenceNSGA2: Multi-objective optimization in peptide sequence space.

This module implements NSGA-II optimization directly in sequence space,
evolving real peptide sequences through biologically-informed mutations.
Unlike latent-space optimization, every candidate is a valid peptide.

Architecture:
    Seed Sequences (from DRAMP database)
    -> Mutation Operators (substitution, insertion, deletion)
    -> PeptideVAE Scoring (MIC prediction, r=0.74)
    -> NSGA-II Selection (Pareto-optimal front)
    -> Output: Real peptide sequences with predicted properties

Objectives (Multi-Objective Optimization):
    1. Activity: Minimize predicted MIC (PeptideVAE model)
    2. Toxicity: Minimize predicted hemolytic activity (heuristic)
    3. Stability: Maximize synthesis feasibility (heuristic)

Mutation Operators:
    - Substitution: Replace AA with similar properties (conservative)
    - Insertion: Add AA at random position (constrained by length)
    - Deletion: Remove AA (constrained by minimum length)
    - Crossover: Recombine two parent sequences

Usage:
    # Command line
    python sequence_nsga2.py --generations 50 --population 100 --output results/

    # As module
    from scripts.sequence_nsga2 import SequenceNSGA2

    optimizer = SequenceNSGA2(
        seed_sequences=["KLAKLAKKLAKLAK", "KLWKKLKKALK"],
        population_size=100,
        generations=50,
    )
    pareto_front = optimizer.run()

Example Output:
    Generation 50/50 | Pareto size: 23 | Best MIC: 0.42
    Optimization complete: 23 Pareto-optimal peptides found
    Results saved to: results/pareto_front.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Add package root to path for local imports
SCRIPT_DIR = Path(__file__).parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent.parent
sys.path.insert(0, str(PACKAGE_DIR))  # For local src imports
sys.path.insert(0, str(PROJECT_ROOT))  # For ML model imports

# Import from local src (self-contained)
from src.constants import (
    AMINO_ACIDS,
    CHARGES,
    HYDROPHOBICITY,
    VOLUMES,
)

# Lazy imports
torch = None
deap = None
PeptideMICPredictor = None


def _lazy_import():
    """Lazily import heavy dependencies."""
    global torch, deap, PeptideMICPredictor

    if torch is None:
        import torch as _torch
        torch = _torch

    if deap is None:
        from deap import base, creator, tools, algorithms
        deap = type("deap", (), {
            "base": base,
            "creator": creator,
            "tools": tools,
            "algorithms": algorithms,
        })()

    if PeptideMICPredictor is None:
        from scripts.predict_mic import PeptideMICPredictor as _Predictor
        PeptideMICPredictor = _Predictor


# =============================================================================
# Constants
# =============================================================================

MIN_PEPTIDE_LENGTH = 8
MAX_PEPTIDE_LENGTH = 35
VALID_AAS = set(AMINO_ACIDS)

# Amino acid similarity groups for conservative mutations
AA_GROUPS = {
    "hydrophobic_aliphatic": list("AILV"),
    "hydrophobic_aromatic": list("FWY"),
    "positive": list("KRH"),
    "negative": list("DE"),
    "polar_uncharged": list("STNQ"),
    "special": list("CGP"),
    "sulfur": list("CM"),
}

# Reverse mapping: AA -> group
AA_TO_GROUP = {}
for group, aas in AA_GROUPS.items():
    for aa in aas:
        AA_TO_GROUP[aa] = group


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Peptide:
    """Candidate peptide with objectives and properties."""

    sequence: str
    mic_pred: float = float("inf")
    toxicity_pred: float = float("inf")
    stability_score: float = 0.0
    generation: int = 0
    parent_sequences: List[str] = field(default_factory=list)
    mutation_history: List[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def net_charge(self) -> float:
        return sum(CHARGES.get(aa, 0) for aa in self.sequence)

    @property
    def hydrophobicity(self) -> float:
        if not self.sequence:
            return 0.0
        return sum(HYDROPHOBICITY.get(aa, 0) for aa in self.sequence) / len(self.sequence)

    @property
    def cationic_ratio(self) -> float:
        if not self.sequence:
            return 0.0
        return sum(1 for aa in self.sequence if aa in "KRH") / len(self.sequence)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "sequence": self.sequence,
            "length": self.length,
            "mic_pred": round(self.mic_pred, 4),
            "mic_ug_ml": round(10 ** self.mic_pred, 4),
            "toxicity_pred": round(self.toxicity_pred, 4),
            "stability_score": round(self.stability_score, 4),
            "net_charge": round(self.net_charge, 2),
            "hydrophobicity": round(self.hydrophobicity, 3),
            "cationic_ratio": round(self.cationic_ratio, 3),
            "generation": self.generation,
            "parent_sequences": self.parent_sequences,
            "mutation_history": self.mutation_history,
        }


@dataclass
class OptimizationResult:
    """Result of NSGA-II optimization."""

    pareto_front: List[Peptide]
    all_evaluated: List[Peptide]
    generations_run: int
    total_evaluations: int
    convergence_history: List[Dict]

    def save_pareto(self, path: Path) -> None:
        """Save Pareto front to CSV."""
        with open(path, "w", newline="") as f:
            if not self.pareto_front:
                return
            writer = csv.DictWriter(f, fieldnames=list(self.pareto_front[0].to_dict().keys()))
            writer.writeheader()
            for peptide in self.pareto_front:
                row = peptide.to_dict()
                row["parent_sequences"] = "; ".join(row["parent_sequences"])
                row["mutation_history"] = "; ".join(row["mutation_history"])
                writer.writerow(row)

    def save_json(self, path: Path) -> None:
        """Save full results to JSON."""
        data = {
            "summary": {
                "pareto_size": len(self.pareto_front),
                "generations_run": self.generations_run,
                "total_evaluations": self.total_evaluations,
                "best_mic": min(p.mic_pred for p in self.pareto_front) if self.pareto_front else None,
                "best_stability": max(p.stability_score for p in self.pareto_front) if self.pareto_front else None,
            },
            "pareto_front": [p.to_dict() for p in self.pareto_front],
            "convergence": self.convergence_history,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Mutation Operators
# =============================================================================

class MutationOperators:
    """Biologically-informed mutation operators for peptide sequences."""

    def __init__(
        self,
        substitution_rate: float = 0.1,
        insertion_rate: float = 0.05,
        deletion_rate: float = 0.05,
        conservative_bias: float = 0.7,
    ):
        """Initialize mutation operators.

        Args:
            substitution_rate: Probability of substitution per position
            insertion_rate: Probability of insertion per sequence
            deletion_rate: Probability of deletion per sequence
            conservative_bias: Probability of conservative (same-group) substitution
        """
        self.substitution_rate = substitution_rate
        self.insertion_rate = insertion_rate
        self.deletion_rate = deletion_rate
        self.conservative_bias = conservative_bias

    def mutate(self, sequence: str) -> Tuple[str, List[str]]:
        """Apply random mutations to a sequence.

        Args:
            sequence: Input peptide sequence

        Returns:
            Tuple of (mutated_sequence, list_of_mutations_applied)
        """
        mutations = []
        seq_list = list(sequence)

        # Substitutions
        for i in range(len(seq_list)):
            if random.random() < self.substitution_rate:
                old_aa = seq_list[i]
                new_aa = self._substitute(old_aa)
                if new_aa != old_aa:
                    seq_list[i] = new_aa
                    mutations.append(f"{old_aa}{i+1}{new_aa}")

        # Insertion (respect max length)
        if len(seq_list) < MAX_PEPTIDE_LENGTH and random.random() < self.insertion_rate:
            pos = random.randint(0, len(seq_list))
            new_aa = self._select_insertion_aa(seq_list, pos)
            seq_list.insert(pos, new_aa)
            mutations.append(f"ins{pos+1}{new_aa}")

        # Deletion (respect min length)
        if len(seq_list) > MIN_PEPTIDE_LENGTH and random.random() < self.deletion_rate:
            pos = random.randint(0, len(seq_list) - 1)
            deleted_aa = seq_list.pop(pos)
            mutations.append(f"del{pos+1}{deleted_aa}")

        return "".join(seq_list), mutations

    def _substitute(self, aa: str) -> str:
        """Substitute an amino acid, with bias toward conservative changes."""
        if random.random() < self.conservative_bias:
            # Conservative: same group
            group = AA_TO_GROUP.get(aa)
            if group:
                candidates = [a for a in AA_GROUPS[group] if a != aa]
                if candidates:
                    return random.choice(candidates)
        # Random substitution
        candidates = [a for a in AMINO_ACIDS if a != aa]
        return random.choice(candidates)

    def _select_insertion_aa(self, seq_list: List[str], pos: int) -> str:
        """Select amino acid for insertion based on context."""
        # Bias toward cationic residues (important for AMP activity)
        if random.random() < 0.3:
            return random.choice(list("KR"))
        # Bias toward hydrophobic (membrane interaction)
        if random.random() < 0.3:
            return random.choice(list("AILV"))
        # Random
        return random.choice(list(AMINO_ACIDS))

    def crossover(self, seq1: str, seq2: str) -> Tuple[str, str]:
        """Single-point crossover between two sequences.

        Args:
            seq1: First parent sequence
            seq2: Second parent sequence

        Returns:
            Tuple of two offspring sequences
        """
        if len(seq1) < 4 or len(seq2) < 4:
            return seq1, seq2

        # Find crossover points
        point1 = random.randint(2, len(seq1) - 2)
        point2 = random.randint(2, len(seq2) - 2)

        # Create offspring
        offspring1 = seq1[:point1] + seq2[point2:]
        offspring2 = seq2[:point2] + seq1[point1:]

        # Validate lengths
        if len(offspring1) < MIN_PEPTIDE_LENGTH:
            offspring1 = seq1
        if len(offspring1) > MAX_PEPTIDE_LENGTH:
            offspring1 = offspring1[:MAX_PEPTIDE_LENGTH]
        if len(offspring2) < MIN_PEPTIDE_LENGTH:
            offspring2 = seq2
        if len(offspring2) > MAX_PEPTIDE_LENGTH:
            offspring2 = offspring2[:MAX_PEPTIDE_LENGTH]

        return offspring1, offspring2


# =============================================================================
# Objective Functions
# =============================================================================

class ObjectiveFunctions:
    """Objective functions for multi-objective optimization."""

    def __init__(
        self,
        predictor: Optional["PeptideMICPredictor"] = None,
        use_heuristics: bool = True,
    ):
        """Initialize objective functions.

        Args:
            predictor: PeptideMICPredictor for MIC prediction
            use_heuristics: Use heuristic objectives when predictor unavailable
        """
        self.predictor = predictor
        self.use_heuristics = use_heuristics

    def evaluate(self, sequence: str) -> Tuple[float, float, float]:
        """Evaluate all objectives for a sequence.

        Args:
            sequence: Peptide sequence

        Returns:
            Tuple of (mic_pred, toxicity_pred, stability_score)
            Note: MIC and toxicity are to be MINIMIZED, stability MAXIMIZED
        """
        # MIC prediction (primary objective)
        if self.predictor is not None:
            try:
                result = self.predictor.predict(sequence)
                mic_pred = result.log10_mic
            except Exception:
                mic_pred = self._heuristic_activity(sequence)
        else:
            mic_pred = self._heuristic_activity(sequence)

        # Toxicity prediction (secondary objective)
        toxicity_pred = self._predict_toxicity(sequence)

        # Stability/synthesis score (tertiary objective)
        stability_score = self._predict_stability(sequence)

        return mic_pred, toxicity_pred, stability_score

    def _heuristic_activity(self, sequence: str) -> float:
        """Heuristic activity prediction based on AMP properties.

        Lower values = better activity (lower MIC).
        """
        n = len(sequence)
        if n == 0:
            return 5.0

        # Optimal AMP properties
        charge = sum(CHARGES.get(aa, 0) for aa in sequence)
        hydro = sum(HYDROPHOBICITY.get(aa, 0) for aa in sequence) / n
        cationic = sum(1 for aa in sequence if aa in "KRH") / n

        # Penalize deviations from optimal
        score = 1.0  # Baseline log10(MIC)

        # Charge: optimal around +2 to +6
        if charge < 2:
            score += (2 - charge) * 0.3
        elif charge > 8:
            score += (charge - 8) * 0.2

        # Hydrophobicity: moderate is best
        if hydro < -1:
            score += (-1 - hydro) * 0.2
        elif hydro > 2:
            score += (hydro - 2) * 0.2

        # Cationic content: 20-40% optimal
        if cationic < 0.15:
            score += (0.15 - cationic) * 2
        elif cationic > 0.5:
            score += (cationic - 0.5) * 1.5

        # Length: 10-25 optimal
        if n < 10:
            score += (10 - n) * 0.1
        elif n > 25:
            score += (n - 25) * 0.05

        return max(0, min(score, 5.0))

    def _predict_toxicity(self, sequence: str) -> float:
        """Predict hemolytic toxicity (heuristic).

        Lower values = less toxic.
        """
        n = len(sequence)
        if n == 0:
            return 1.0

        toxicity = 0.0

        # High hydrophobicity increases toxicity
        hydro = sum(HYDROPHOBICITY.get(aa, 0) for aa in sequence) / n
        if hydro > 1.5:
            toxicity += (hydro - 1.5) * 0.3

        # Tryptophan content increases toxicity
        trp_ratio = sequence.count("W") / n
        toxicity += trp_ratio * 2.0

        # Very high charge can increase toxicity
        charge = sum(CHARGES.get(aa, 0) for aa in sequence)
        if charge > 8:
            toxicity += (charge - 8) * 0.1

        # Long sequences more toxic
        if n > 25:
            toxicity += (n - 25) * 0.02

        return max(0, min(toxicity, 1.0))

    def _predict_stability(self, sequence: str) -> float:
        """Predict synthesis stability/feasibility (heuristic).

        Higher values = more stable/easier to synthesize.
        """
        n = len(sequence)
        if n == 0:
            return 0.0

        stability = 1.0

        # Penalize problematic residues
        # Methionine: oxidation risk
        met_ratio = sequence.count("M") / n
        stability -= met_ratio * 0.5

        # Cysteine: disulfide issues
        cys_ratio = sequence.count("C") / n
        stability -= cys_ratio * 0.3

        # Asparagine-Glycine: deamidation risk
        stability -= sequence.count("NG") * 0.1 / n
        stability -= sequence.count("DG") * 0.1 / n

        # Proline at N-terminus: synthesis difficult
        if sequence and sequence[0] == "P":
            stability -= 0.2

        # Very long sequences harder to synthesize
        if n > 30:
            stability -= (n - 30) * 0.02

        # Consecutive repeats: aggregation risk
        for i in range(len(sequence) - 2):
            if sequence[i] == sequence[i+1] == sequence[i+2]:
                stability -= 0.1

        return max(0, min(stability, 1.0))


# =============================================================================
# NSGA-II Optimizer
# =============================================================================

class SequenceNSGA2:
    """NSGA-II optimizer for peptide sequences.

    Evolves peptide sequences using multi-objective optimization with:
    - Activity (MIC prediction via PeptideVAE)
    - Toxicity (hemolytic prediction)
    - Stability (synthesis feasibility)
    """

    @staticmethod
    def _validate_population_size(population_size: int) -> int:
        """Ensure population size is divisible by 4 for selTournamentDCD.

        DEAP's selTournamentDCD requires population size to be divisible by 4.
        If not, we round up to the next multiple of 4.

        Args:
            population_size: Requested population size

        Returns:
            Validated population size (divisible by 4)
        """
        if population_size % 4 != 0:
            adjusted = ((population_size // 4) + 1) * 4
            print(f"⚠️  DEAP Fix: Adjusted population from {population_size} to {adjusted} (must be divisible by 4)")
            return adjusted
        return population_size

    def __init__(
        self,
        seed_sequences: Optional[List[str]] = None,
        population_size: int = 100,
        generations: int = 50,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.3,
        checkpoint_path: Optional[Path] = None,
        verbose: bool = True,
        random_seed: Optional[int] = None,
    ):
        """Initialize the optimizer.

        Args:
            seed_sequences: Initial sequences to seed population. If None, uses defaults.
            population_size: Number of individuals per generation
            generations: Number of generations to evolve
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            checkpoint_path: Path to PeptideVAE checkpoint
            verbose: Print progress messages
            random_seed: Random seed for reproducibility
        """
        _lazy_import()

        # FIX: Validate population size for DEAP selTournamentDCD compatibility
        self.population_size = self._validate_population_size(population_size)
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.verbose = verbose

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Default seed sequences (known AMPs)
        self.seed_sequences = seed_sequences or [
            "KLAKLAKKLAKLAK",      # LL-37 derivative
            "KLWKKLKKALK",         # Magainin derivative
            "GIGKFLHSAKKFGKAFVGEIMNS",  # Magainin 2
            "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",  # Cecropin A
            "RLKKTFFKIV",          # Short cationic AMP
            "ILPWKWPWWPWRR",       # Indolicidin variant
            "KRFRIRVRV",           # BP100 derivative
            "FKCRRWQWRMKKLGAPSITCVRRAF",  # Protegrin-1 derivative
        ]

        # Initialize components
        self.mutation_ops = MutationOperators()

        # Try to load PeptideVAE predictor
        try:
            self.predictor = PeptideMICPredictor(
                checkpoint_path=checkpoint_path,
                verbose=False,
            )
            if self.verbose:
                print("Loaded PeptideVAE predictor (r=0.74)")
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load PeptideVAE: {e}")
                print("Using heuristic objectives only")
            self.predictor = None

        self.objectives = ObjectiveFunctions(
            predictor=self.predictor,
            use_heuristics=True,
        )

        # Setup DEAP
        self._setup_deap()

    def _setup_deap(self) -> None:
        """Configure DEAP for multi-objective optimization."""
        # Create fitness class (minimize MIC, minimize toxicity, maximize stability)
        # Note: weights = (-1, -1, 1) means minimize first two, maximize third
        if not hasattr(deap.creator, "FitnessMulti"):
            deap.creator.create("FitnessMulti", deap.base.Fitness, weights=(-1.0, -1.0, 1.0))
        if not hasattr(deap.creator, "Individual"):
            deap.creator.create("Individual", list, fitness=deap.creator.FitnessMulti)

        self.toolbox = deap.base.Toolbox()

        # Register operators
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", deap.tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", deap.tools.selNSGA2)

    def _create_individual(self) -> "deap.creator.Individual":
        """Create a random individual from seed sequences."""
        # Start with a seed or mutated seed
        base_seq = random.choice(self.seed_sequences)

        # Apply some initial mutations for diversity
        if random.random() < 0.5:
            mutated, _ = self.mutation_ops.mutate(base_seq)
            base_seq = mutated

        individual = deap.creator.Individual([base_seq])
        return individual

    def _evaluate_individual(self, individual: List[str]) -> Tuple[float, float, float]:
        """Evaluate fitness of an individual."""
        sequence = individual[0]
        return self.objectives.evaluate(sequence)

    def _crossover(
        self,
        ind1: List[str],
        ind2: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Crossover two individuals."""
        seq1, seq2 = self.mutation_ops.crossover(ind1[0], ind2[0])
        ind1[0] = seq1
        ind2[0] = seq2
        return ind1, ind2

    def _mutate(self, individual: List[str]) -> Tuple[List[str]]:
        """Mutate an individual."""
        mutated, _ = self.mutation_ops.mutate(individual[0])
        individual[0] = mutated
        return (individual,)

    def run(self) -> OptimizationResult:
        """Run the NSGA-II optimization.

        Returns:
            OptimizationResult with Pareto front and statistics
        """
        if self.verbose:
            print(f"Starting NSGA-II optimization")
            print(f"  Population: {self.population_size}")
            print(f"  Generations: {self.generations}")
            print(f"  Seed sequences: {len(self.seed_sequences)}")
            print()

        # Initialize population
        population = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        convergence_history = []
        all_evaluated = []
        total_evaluations = len(population)

        # Main evolution loop
        for gen in range(self.generations):
            # Assign crowding distance before tournament selection
            # (required for selTournamentDCD)
            fronts = deap.tools.sortNondominated(population, len(population))
            for front in fronts:
                deap.tools.emo.assignCrowdingDist(front)

            # Select next generation
            offspring = deap.tools.selTournamentDCD(population, len(population))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation
            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            total_evaluations += len(invalid_ind)

            # Select survivors
            population = self.toolbox.select(population + offspring, self.population_size)

            # Get current Pareto front
            pareto = deap.tools.sortNondominated(population, len(population), first_front_only=True)[0]

            # Record convergence
            best_mic = min(ind.fitness.values[0] for ind in pareto)
            best_stability = max(ind.fitness.values[2] for ind in pareto)
            convergence_history.append({
                "generation": gen + 1,
                "pareto_size": len(pareto),
                "best_mic": round(best_mic, 4),
                "best_stability": round(best_stability, 4),
            })

            if self.verbose and (gen + 1) % 10 == 0:
                print(f"Generation {gen+1}/{self.generations} | "
                      f"Pareto size: {len(pareto)} | "
                      f"Best MIC: {best_mic:.2f}")

        # Extract final Pareto front
        final_pareto = deap.tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # Convert to Peptide objects
        pareto_peptides = []
        for ind in final_pareto:
            peptide = Peptide(
                sequence=ind[0],
                mic_pred=ind.fitness.values[0],
                toxicity_pred=ind.fitness.values[1],
                stability_score=ind.fitness.values[2],
                generation=self.generations,
            )
            pareto_peptides.append(peptide)

        # Sort by MIC (best first)
        pareto_peptides.sort(key=lambda p: p.mic_pred)

        if self.verbose:
            print(f"\nOptimization complete: {len(pareto_peptides)} Pareto-optimal peptides found")

        return OptimizationResult(
            pareto_front=pareto_peptides,
            all_evaluated=[],  # Could track all, but memory intensive
            generations_run=self.generations,
            total_evaluations=total_evaluations,
            convergence_history=convergence_history,
        )


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for sequence optimization."""
    parser = argparse.ArgumentParser(
        description="NSGA-II peptide optimization in sequence space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default optimization
    python sequence_nsga2.py

    # Custom parameters
    python sequence_nsga2.py --generations 100 --population 200

    # With specific seeds
    python sequence_nsga2.py --seeds "KLAKLAK,KLWKKLK"

    # Save results
    python sequence_nsga2.py --output results/pareto_front.csv
        """,
    )

    parser.add_argument(
        "--generations", "-g",
        type=int,
        default=50,
        help="Number of generations (default: 50)",
    )
    parser.add_argument(
        "--population", "-p",
        type=int,
        default=100,
        help="Population size (default: 100)",
    )
    parser.add_argument(
        "--seeds", "-s",
        type=str,
        help="Comma-separated seed sequences",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for Pareto front (CSV or JSON)",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        help="Path to PeptideVAE checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    # Parse seeds
    seeds = None
    if args.seeds:
        seeds = [s.strip() for s in args.seeds.split(",")]

    # Run optimization
    optimizer = SequenceNSGA2(
        seed_sequences=seeds,
        population_size=args.population,
        generations=args.generations,
        checkpoint_path=args.checkpoint,
        verbose=not args.quiet,
        random_seed=args.seed,
    )

    result = optimizer.run()

    # Print summary
    print("\n" + "=" * 60)
    print("PARETO FRONT")
    print("=" * 60)
    print(f"{'Rank':<5} {'Sequence':<35} {'MIC':<8} {'Tox':<8} {'Stab':<8}")
    print("-" * 60)
    for i, peptide in enumerate(result.pareto_front[:10], 1):
        seq_display = peptide.sequence[:32] + "..." if len(peptide.sequence) > 35 else peptide.sequence
        print(f"{i:<5} {seq_display:<35} {peptide.mic_pred:<8.3f} "
              f"{peptide.toxicity_pred:<8.3f} {peptide.stability_score:<8.3f}")

    if len(result.pareto_front) > 10:
        print(f"... and {len(result.pareto_front) - 10} more")

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".json":
            result.save_json(args.output)
        else:
            result.save_pareto(args.output)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
