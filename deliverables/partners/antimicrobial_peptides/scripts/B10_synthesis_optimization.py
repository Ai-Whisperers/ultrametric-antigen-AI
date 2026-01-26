# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""B10: Synthesis Optimization - Sequence-Space NSGA-II

Research Idea Implementation - Carlos Brizuela

Multi-objective optimization for synthesis-friendly AMPs using sequence-space
mutations on real peptide sequences (10-35 AA).

Synthesis Challenges Addressed:
1. Aggregation (hydrophobic stretches)
2. Deletion peptides (steric hindrance)
3. Racemization (base-sensitive residues)
4. Aspartimide formation (Asp-Xxx motifs)
5. Cost estimation (expensive AAs)

Objectives:
1. MIC prediction (PeptideVAE)
2. Synthesis difficulty (minimize)
3. Coupling efficiency (maximize)
4. Cost (minimize)

Usage:
    python scripts/B10_synthesis_optimization.py --output results/synthesis_optimized/
    python scripts/B10_synthesis_optimization.py --population 200 --generations 100

    # Dry run without VAE model:
    python scripts/B10_synthesis_optimization.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import sys

import numpy as np

# Add paths - Standardized setup (Issue #3 fix)
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "deliverables"))
sys.path.insert(0, str(PACKAGE_DIR))

# Import base sequence optimizer
from scripts.sequence_nsga2 import (
    SequenceNSGA2,
    ObjectiveFunctions,
    MutationOperators,
    Peptide,
    OptimizationResult,
    AMINO_ACIDS,
)

# Import MIC predictor
from scripts.predict_mic import PeptideMICPredictor, PredictionResult

# Import from local src (self-contained)
from src.peptide_utils import AA_PROPERTIES, compute_peptide_properties


# =============================================================================
# SYNTHESIS-SPECIFIC DATA
# =============================================================================

# Amino acid synthesis properties (empirical SPPS data)
AA_SYNTHESIS = {
    "A": {"cost": 1.0, "coupling": 0.99, "aggregation": 0.10, "racemization": 0.01},
    "R": {"cost": 3.0, "coupling": 0.95, "aggregation": 0.05, "racemization": 0.02},
    "N": {"cost": 2.0, "coupling": 0.96, "aggregation": 0.05, "racemization": 0.02},
    "D": {"cost": 2.0, "coupling": 0.94, "aggregation": 0.05, "racemization": 0.03},  # Aspartimide risk
    "C": {"cost": 4.0, "coupling": 0.92, "aggregation": 0.15, "racemization": 0.02},  # Oxidation
    "Q": {"cost": 2.5, "coupling": 0.95, "aggregation": 0.08, "racemization": 0.02},
    "E": {"cost": 2.0, "coupling": 0.95, "aggregation": 0.05, "racemization": 0.02},
    "G": {"cost": 1.0, "coupling": 0.99, "aggregation": 0.02, "racemization": 0.00},  # No chiral center
    "H": {"cost": 4.0, "coupling": 0.93, "aggregation": 0.10, "racemization": 0.03},
    "I": {"cost": 2.0, "coupling": 0.94, "aggregation": 0.25, "racemization": 0.01},  # Beta-branched
    "L": {"cost": 1.5, "coupling": 0.97, "aggregation": 0.20, "racemization": 0.01},
    "K": {"cost": 2.0, "coupling": 0.96, "aggregation": 0.05, "racemization": 0.02},
    "M": {"cost": 3.5, "coupling": 0.93, "aggregation": 0.12, "racemization": 0.02},  # Oxidation
    "F": {"cost": 2.5, "coupling": 0.95, "aggregation": 0.30, "racemization": 0.01},
    "P": {"cost": 2.0, "coupling": 0.90, "aggregation": 0.02, "racemization": 0.00},  # Imino acid
    "S": {"cost": 1.5, "coupling": 0.97, "aggregation": 0.05, "racemization": 0.02},
    "T": {"cost": 2.0, "coupling": 0.96, "aggregation": 0.08, "racemization": 0.02},
    "W": {"cost": 6.0, "coupling": 0.90, "aggregation": 0.35, "racemization": 0.03},  # Expensive + difficult
    "Y": {"cost": 3.0, "coupling": 0.94, "aggregation": 0.25, "racemization": 0.02},
    "V": {"cost": 1.5, "coupling": 0.96, "aggregation": 0.20, "racemization": 0.01},  # Beta-branched
}

# Difficult dipeptide combinations in SPPS
DIFFICULT_DIPEPTIDES = {
    ("D", "G"): 0.30,   # High aspartimide risk
    ("D", "S"): 0.25,   # Aspartimide risk
    ("D", "N"): 0.20,   # Aspartimide risk
    ("D", "T"): 0.20,   # Aspartimide risk
    ("D", "H"): 0.15,   # Moderate aspartimide
    ("W", "W"): 0.40,   # Steric hindrance
    ("I", "I"): 0.30,   # Aggregation
    ("V", "V"): 0.30,   # Aggregation
    ("F", "F"): 0.35,   # Aggregation
    ("L", "L"): 0.25,   # Aggregation
    ("I", "V"): 0.25,   # Beta-branched combination
    ("V", "I"): 0.25,   # Beta-branched combination
}

# Synthesis-friendly AAs (prefer during mutations)
SYNTHESIS_FRIENDLY = ["A", "G", "K", "L", "S", "E", "Q"]

# Scale-up considerations by peptide length
LENGTH_SCALE_FACTORS = {
    (10, 15): 1.0,   # Easy
    (16, 20): 1.2,   # Standard
    (21, 25): 1.5,   # Moderate difficulty
    (26, 30): 2.0,   # Challenging
    (31, 35): 2.5,   # Difficult
    (36, 50): 3.5,   # Very difficult
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SynthesisCandidate:
    """Synthesis-optimized AMP candidate with full metrics."""

    sequence: str
    length: int

    # Activity metrics
    predicted_mic: float
    confidence: str
    confidence_score: float

    # Peptide properties
    net_charge: float
    hydrophobicity: float

    # Synthesis metrics
    synthesis_difficulty: float
    aggregation_propensity: float
    coupling_efficiency: float
    racemization_risk: float
    estimated_cost: float
    cost_per_mg: float
    difficult_motifs: list = field(default_factory=list)

    # Optimization metrics
    pareto_rank: int = 0
    crowding_distance: float = 0.0

    # Quality assessment
    synthesis_grade: str = ""
    scale_up_factor: float = 1.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "sequence": self.sequence,
            "length": self.length,
            "predicted_mic": round(self.predicted_mic, 4),
            "confidence": self.confidence,
            "confidence_score": round(self.confidence_score, 3),
            "net_charge": round(self.net_charge, 2),
            "hydrophobicity": round(self.hydrophobicity, 3),
            "synthesis_difficulty": round(self.synthesis_difficulty, 3),
            "aggregation_propensity": round(self.aggregation_propensity, 4),
            "coupling_efficiency": round(self.coupling_efficiency, 5),
            "racemization_risk": round(self.racemization_risk, 4),
            "estimated_cost": round(self.estimated_cost, 1),
            "cost_per_mg": round(self.cost_per_mg, 2),
            "difficult_motifs": self.difficult_motifs,
            "pareto_rank": self.pareto_rank,
            "crowding_distance": round(self.crowding_distance, 3),
            "synthesis_grade": self.synthesis_grade,
            "scale_up_factor": round(self.scale_up_factor, 2),
        }


# =============================================================================
# SYNTHESIS SCORING FUNCTIONS
# =============================================================================

def predict_synthesis_difficulty(sequence: str) -> dict:
    """Predict comprehensive synthesis difficulty metrics.

    Returns:
        Dictionary with all synthesis-related metrics
    """
    sequence = sequence.upper()
    seq_len = len(sequence)

    if seq_len == 0:
        return {
            "difficulty": 1.0,
            "aggregation": 0.0,
            "coupling": 0.0,
            "racemization": 0.0,
            "cost": 0.0,
            "cost_per_mg": 0.0,
            "difficult_motifs": [],
            "scale_factor": 1.0,
            "grade": "INVALID",
        }

    # 1. Aggregation propensity
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

    # Long hydrophobic stretches cause aggregation on resin
    if max_hydrophobic_run >= 5:
        aggregation += (max_hydrophobic_run - 4) * 0.15

    # Sum individual aggregation propensities
    for aa in sequence:
        if aa in AA_SYNTHESIS:
            aggregation += AA_SYNTHESIS[aa]["aggregation"]
    aggregation /= seq_len

    # 2. Coupling efficiency (product of individual efficiencies)
    coupling = 1.0
    for aa in sequence:
        if aa in AA_SYNTHESIS:
            coupling *= AA_SYNTHESIS[aa]["coupling"]

    # 3. Racemization risk (sum of individual risks)
    racemization = 0.0
    for aa in sequence:
        if aa in AA_SYNTHESIS:
            racemization += AA_SYNTHESIS[aa]["racemization"]
    racemization /= seq_len

    # 4. Difficult dipeptide motifs
    difficult_motifs = []
    motif_penalty = 0.0
    for i in range(seq_len - 1):
        dipeptide = (sequence[i], sequence[i + 1])
        if dipeptide in DIFFICULT_DIPEPTIDES:
            penalty = DIFFICULT_DIPEPTIDES[dipeptide]
            motif_penalty += penalty
            difficult_motifs.append({
                "position": i + 1,  # 1-indexed
                "motif": f"{dipeptide[0]}{dipeptide[1]}",
                "type": "aspartimide" if dipeptide[0] == "D" else "aggregation",
                "penalty": penalty,
            })

    # 5. Cost estimation
    base_cost = sum(AA_SYNTHESIS.get(aa, {"cost": 2.0})["cost"] for aa in sequence)

    # Scale factor by length
    scale_factor = 1.0
    for (min_len, max_len), factor in LENGTH_SCALE_FACTORS.items():
        if min_len <= seq_len <= max_len:
            scale_factor = factor
            break
    if seq_len > 50:
        scale_factor = 5.0  # Very difficult

    total_cost = base_cost * scale_factor
    cost_per_mg = total_cost / seq_len  # Normalized

    # 6. Overall difficulty score (0-1, lower is better for synthesis)
    difficulty = (
        aggregation * 0.30 +
        (1 - coupling) * 100 * 0.25 +  # Scale coupling loss
        racemization * 10 * 0.15 +
        motif_penalty * 0.20 +
        (scale_factor - 1) * 0.10  # Length penalty
    )
    difficulty = min(1.0, difficulty)  # Cap at 1.0

    # 7. Synthesis grade
    if difficulty < 0.2:
        grade = "EXCELLENT"
    elif difficulty < 0.35:
        grade = "GOOD"
    elif difficulty < 0.5:
        grade = "MODERATE"
    elif difficulty < 0.7:
        grade = "CHALLENGING"
    else:
        grade = "DIFFICULT"

    return {
        "difficulty": difficulty,
        "aggregation": aggregation,
        "coupling": coupling,
        "racemization": racemization,
        "cost": total_cost,
        "cost_per_mg": cost_per_mg,
        "difficult_motifs": difficult_motifs,
        "scale_factor": scale_factor,
        "grade": grade,
    }


# =============================================================================
# SYNTHESIS-SPECIFIC OBJECTIVES
# =============================================================================

class SynthesisObjectives(ObjectiveFunctions):
    """Synthesis-specific objective functions extending base objectives.

    4-objective optimization:
    1. MIC prediction (minimize = higher activity)
    2. Synthesis difficulty (minimize)
    3. Coupling efficiency (maximize -> minimize negative)
    4. Cost (minimize)
    """

    def __init__(self, predictor: Optional[PeptideMICPredictor] = None):
        """Initialize with MIC predictor."""
        super().__init__(predictor=predictor)

    def synthesis_difficulty(self, sequence: str) -> float:
        """Compute synthesis difficulty score (0-1, lower is better)."""
        synth = predict_synthesis_difficulty(sequence)
        return synth["difficulty"]

    def coupling_efficiency(self, sequence: str) -> float:
        """Compute coupling efficiency (0-1, higher is better).

        Returns negative for minimization (NSGA-II minimizes all objectives).
        """
        synth = predict_synthesis_difficulty(sequence)
        return -synth["coupling"]  # Negative because we maximize coupling

    def synthesis_cost(self, sequence: str) -> float:
        """Compute normalized synthesis cost."""
        synth = predict_synthesis_difficulty(sequence)
        # Normalize cost to 0-1 range (typical costs 20-200)
        normalized = min(1.0, synth["cost_per_mg"] / 10.0)
        return normalized

    def evaluate_all(self, sequence: str) -> tuple:
        """Evaluate all 4 synthesis objectives.

        Returns:
            Tuple of (mic, difficulty, neg_coupling, cost) - all to minimize
        """
        # Get MIC prediction (same logic as parent class)
        if self.predictor is not None:
            try:
                result = self.predictor.predict(sequence)
                mic = result.log10_mic
            except Exception:
                mic = self._heuristic_activity(sequence)
        else:
            mic = self._heuristic_activity(sequence)

        # Get synthesis metrics
        synth = predict_synthesis_difficulty(sequence)

        difficulty = synth["difficulty"]
        neg_coupling = -synth["coupling"]  # Negative for minimization
        cost = min(1.0, synth["cost_per_mg"] / 10.0)

        return (mic, difficulty, neg_coupling, cost)


# =============================================================================
# SYNTHESIS-AWARE MUTATION OPERATORS
# =============================================================================

class SynthesisMutationOperators(MutationOperators):
    """Mutation operators biased toward synthesis-friendly modifications."""

    def __init__(
        self,
        p_substitution: float = 0.5,
        p_insertion: float = 0.25,
        p_deletion: float = 0.25,
        min_length: int = 10,
        max_length: int = 35,
    ):
        # Fix parameter names to match parent class
        super().__init__(
            substitution_rate=p_substitution,
            insertion_rate=p_insertion,
            deletion_rate=p_deletion,
            conservative_bias=0.7,  # Use default synthesis-friendly bias
        )

        # Store length constraints for synthesis-aware mutations
        self.min_length = min_length
        self.max_length = max_length

        # Build weighted AA list favoring synthesis-friendly residues
        self.weighted_aa = []
        for aa in AMINO_ACIDS:
            if aa in SYNTHESIS_FRIENDLY:
                self.weighted_aa.extend([aa] * 3)  # 3x weight
            elif aa in ["W", "M", "C", "H"]:
                self.weighted_aa.extend([aa] * 1)  # 1x weight (difficult)
            else:
                self.weighted_aa.extend([aa] * 2)  # 2x weight (moderate)

    def _choose_synthesis_friendly_aa(self) -> str:
        """Choose AA biased toward synthesis-friendly residues."""
        return random.choice(self.weighted_aa)

    def substitution(self, sequence: str) -> str:
        """Substitute one AA, preferring synthesis-friendly replacements."""
        if not sequence:
            return sequence

        seq_list = list(sequence)
        pos = random.randint(0, len(seq_list) - 1)

        # Check if current position is in a difficult motif
        current_aa = seq_list[pos]

        # If it's a problematic AA, higher chance of replacing
        if current_aa in ["W", "C", "M", "D"]:
            new_aa = self._choose_synthesis_friendly_aa()
        else:
            # 50% chance: synthesis-friendly, 50% any AA
            if random.random() < 0.5:
                new_aa = self._choose_synthesis_friendly_aa()
            else:
                new_aa = random.choice(AMINO_ACIDS)

        # Avoid creating difficult dipeptides
        if pos > 0:
            prev_aa = seq_list[pos - 1]
            if (prev_aa, new_aa) in DIFFICULT_DIPEPTIDES:
                new_aa = random.choice(SYNTHESIS_FRIENDLY)

        if pos < len(seq_list) - 1:
            next_aa = seq_list[pos + 1]
            if (new_aa, next_aa) in DIFFICULT_DIPEPTIDES:
                new_aa = random.choice(SYNTHESIS_FRIENDLY)

        seq_list[pos] = new_aa
        return "".join(seq_list)

    def insertion(self, sequence: str) -> str:
        """Insert synthesis-friendly AA."""
        if len(sequence) >= self.max_length:
            return sequence

        pos = random.randint(0, len(sequence))
        new_aa = self._choose_synthesis_friendly_aa()

        return sequence[:pos] + new_aa + sequence[pos:]


# =============================================================================
# SYNTHESIS NSGA-II OPTIMIZER
# =============================================================================

class SynthesisNSGA2(SequenceNSGA2):
    """NSGA-II optimizer for synthesis-friendly AMP design.

    Optimizes 4 objectives:
    1. MIC prediction (activity)
    2. Synthesis difficulty
    3. Coupling efficiency
    4. Cost
    """

    def __init__(
        self,
        seed_sequences: Optional[list[str]] = None,
        population_size: int = 100,
        generations: int = 50,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.3,
        min_length: int = 12,
        max_length: int = 30,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        verbose: bool = True,
        random_seed: int = 42,
    ):
        """Initialize synthesis optimizer.

        Args:
            seed_sequences: Starting peptide sequences (defaults to known synthesis-friendly)
            population_size: NSGA-II population size
            generations: Number of generations
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            min_length: Minimum peptide length
            max_length: Maximum peptide length (shorter is easier to synthesize)
            checkpoint_path: Path to PeptideVAE checkpoint
            device: Torch device
            verbose: Print progress
            random_seed: Random seed for reproducibility
        """
        # Default seed sequences: known synthesis-friendly AMPs
        if seed_sequences is None:
            seed_sequences = [
                "KLAKLAKKLAKLAK",      # LL-37 derivative, easy synthesis
                "KWKLFKKIGAVLKVL",     # Indolicidin derivative
                "GIGKFLKKAKKFGKAFVK",  # Magainin derivative
                "ILPWKWPWWPWRR",       # Minimal hydrophobic
                "RLKRLKRLKRLKR",       # Poly-cationic
                "GKGKGKGKGKGKGK",      # Alternating, easy synthesis
                "LALKLALKALKAALK",     # Helical, synthesis-friendly
                "KSKSKSKSKSKSKSK",     # Poly-cationic alternating
            ]

        # Use synthesis-aware mutation operators
        mutation_ops = SynthesisMutationOperators(
            p_substitution=0.5,
            p_insertion=0.25,
            p_deletion=0.25,
            min_length=min_length,
            max_length=max_length,
        )

        # Fix: Only pass parameters that parent class accepts
        super().__init__(
            seed_sequences=seed_sequences,
            population_size=population_size,
            generations=generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            checkpoint_path=checkpoint_path,
            verbose=verbose,
            random_seed=random_seed,
        )

        # Store synthesis-specific parameters
        self.min_length = min_length
        self.max_length = max_length

        # Override mutation operators with synthesis-aware version
        self.mutation_ops = mutation_ops

        # Use synthesis-specific objectives
        self.objectives = SynthesisObjectives(predictor=self.predictor)

        if self.verbose:
            print("\n" + "=" * 70)
            print("SYNTHESIS-OPTIMIZED AMP DESIGN")
            print("=" * 70)
            print(f"Goal: Maximize activity while minimizing synthesis difficulty")
            print(f"Objectives: MIC, Difficulty, Coupling Efficiency, Cost")
            print(f"Population: {population_size}, Generations: {generations}")
            print(f"Length range: {min_length}-{max_length} AA")
            print("=" * 70)

    def _setup_deap(self) -> None:
        """Configure DEAP for 4-objective synthesis optimization."""
        import deap.base
        import deap.creator
        import deap.tools

        # Create fitness class for 4 objectives (minimize all: MIC, difficulty, neg_coupling, cost)
        if not hasattr(deap.creator, "FitnessMulti"):
            deap.creator.create("FitnessMulti", deap.base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
        if not hasattr(deap.creator, "Individual"):
            deap.creator.create("Individual", list, fitness=deap.creator.FitnessMulti)

        self.toolbox = deap.base.Toolbox()

        # Register operators
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", deap.tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", deap.tools.selNSGA2)
        self.toolbox.register("evaluate", self._evaluate_individual)

    def _evaluate_individual(self, individual: List[str]) -> tuple:
        """Evaluate all 4 synthesis objectives for an individual."""
        sequence = individual[0]  # Extract sequence from Individual
        return self.objectives.evaluate_all(sequence)

    def _create_candidate(
        self,
        sequence: str,
        fitness: tuple,
        rank: int = 0,
        crowding: float = 0.0,
    ) -> SynthesisCandidate:
        """Create a SynthesisCandidate from optimization results."""
        # Get peptide properties
        props = compute_peptide_properties(sequence)

        # Get synthesis metrics
        synth = predict_synthesis_difficulty(sequence)

        # Get confidence from predictor
        if self.predictor is not None:
            try:
                result = self.predictor.predict(sequence)
                confidence = result.confidence
                confidence_score = result.confidence_score
                predicted_mic = result.predicted_mic
            except Exception:
                confidence = "Unknown"
                confidence_score = 0.0
                predicted_mic = fitness[0]
        else:
            confidence = "N/A"
            confidence_score = 0.0
            predicted_mic = fitness[0]

        return SynthesisCandidate(
            sequence=sequence,
            length=len(sequence),
            predicted_mic=predicted_mic,
            confidence=confidence,
            confidence_score=confidence_score,
            net_charge=props["net_charge"],
            hydrophobicity=props["hydrophobicity"],
            synthesis_difficulty=synth["difficulty"],
            aggregation_propensity=synth["aggregation"],
            coupling_efficiency=synth["coupling"],
            racemization_risk=synth["racemization"],
            estimated_cost=synth["cost"],
            cost_per_mg=synth["cost_per_mg"],
            difficult_motifs=synth["difficult_motifs"],
            pareto_rank=rank,
            crowding_distance=crowding,
            synthesis_grade=synth["grade"],
            scale_up_factor=synth["scale_factor"],
        )

    def run(self) -> list[SynthesisCandidate]:
        """Run NSGA-II optimization and return synthesis candidates."""
        # Run base NSGA-II optimization but handle results manually
        # since we have 4 objectives instead of 3
        import deap.tools
        import random

        population = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Evolution loop
        for gen in range(self.generations):
            # Assign crowding distance for tournament selection
            fronts = deap.tools.sortNondominated(population, len(population))
            for front in fronts:
                deap.tools.emo.assignCrowdingDist(front)

            # Select and create offspring
            offspring = deap.tools.selTournamentDCD(population, len(population))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select next generation
            population = deap.tools.selNSGA2(population + offspring, self.population_size)

        # Get final Pareto front
        fronts = deap.tools.sortNondominated(population, len(population))
        final_pareto = fronts[0] if fronts else []

        if self.verbose:
            print(f"\nOptimization complete: {len(final_pareto)} Pareto-optimal peptides found")

        # Convert to SynthesisCandidate objects
        candidates = []
        for rank, ind in enumerate(final_pareto):
            # Extract 4 fitness values: (mic, difficulty, neg_coupling, cost)
            fitness_values = ind.fitness.values
            candidate = self._create_candidate(
                sequence=ind[0],
                fitness=fitness_values,
                rank=rank,
                crowding=getattr(ind.fitness, 'crowding_dist', 0.0),
            )
            candidates.append(candidate)

        # Sort by combined score: low MIC + low difficulty
        candidates.sort(
            key=lambda c: c.predicted_mic + c.synthesis_difficulty,
        )

        return candidates


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_results(
    candidates: list[SynthesisCandidate],
    output_dir: Path,
    top_n: int = 50,
) -> None:
    """Export synthesis optimization results.

    Args:
        candidates: List of SynthesisCandidate objects
        output_dir: Output directory
        top_n: Number of top candidates to export
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    top_candidates = candidates[:top_n]

    # 1. Export JSON results
    results = {
        "objective": "Synthesis-optimized AMP design (Sequence-Space NSGA-II)",
        "method": "Multi-objective optimization with synthesis-aware mutations",
        "objectives": [
            "MIC prediction (minimize)",
            "Synthesis difficulty (minimize)",
            "Coupling efficiency (maximize)",
            "Cost (minimize)",
        ],
        "metrics": {
            "total_candidates": len(candidates),
            "pareto_front_size": len(top_candidates),
            "avg_difficulty": round(np.mean([c.synthesis_difficulty for c in top_candidates]), 4),
            "avg_coupling": round(np.mean([c.coupling_efficiency for c in top_candidates]), 4),
            "avg_cost": round(np.mean([c.estimated_cost for c in top_candidates]), 2),
            "grade_distribution": {
                grade: sum(1 for c in top_candidates if c.synthesis_grade == grade)
                for grade in ["EXCELLENT", "GOOD", "MODERATE", "CHALLENGING", "DIFFICULT"]
            },
        },
        "candidates": [
            {"rank": i + 1, **c.to_dict()}
            for i, c in enumerate(top_candidates)
        ],
    }

    json_path = output_dir / "synthesis_optimized_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exported JSON: {json_path}")

    # 2. Export CSV
    try:
        import pandas as pd

        records = [
            {
                "rank": i + 1,
                "sequence": c.sequence,
                "length": c.length,
                "predicted_mic": c.predicted_mic,
                "confidence": c.confidence,
                "synthesis_grade": c.synthesis_grade,
                "difficulty": c.synthesis_difficulty,
                "coupling": c.coupling_efficiency,
                "aggregation": c.aggregation_propensity,
                "racemization": c.racemization_risk,
                "cost": c.estimated_cost,
                "cost_per_mg": c.cost_per_mg,
                "net_charge": c.net_charge,
                "hydrophobicity": c.hydrophobicity,
                "n_difficult_motifs": len(c.difficult_motifs),
                "scale_factor": c.scale_up_factor,
            }
            for i, c in enumerate(top_candidates)
        ]

        df = pd.DataFrame(records)
        csv_path = output_dir / "synthesis_optimized_candidates.csv"
        df.to_csv(csv_path, index=False)
        print(f"Exported CSV: {csv_path}")

    except ImportError:
        warnings.warn("pandas not available, skipping CSV export")

    # 3. Export FASTA
    fasta_path = output_dir / "synthesis_optimized.fasta"
    with open(fasta_path, "w") as f:
        for i, c in enumerate(top_candidates):
            header = (
                f">synthesis_opt_{i+1}|"
                f"MIC={c.predicted_mic:.4f}|"
                f"grade={c.synthesis_grade}|"
                f"difficulty={c.synthesis_difficulty:.3f}|"
                f"coupling={c.coupling_efficiency:.4f}|"
                f"cost=${c.estimated_cost:.1f}"
            )
            f.write(f"{header}\n{c.sequence}\n")
    print(f"Exported FASTA: {fasta_path}")

    # 4. Print summary
    print("\n" + "=" * 80)
    print("TOP 15 SYNTHESIS-OPTIMIZED CANDIDATES")
    print("=" * 80)
    print(f"{'Rank':<5} {'Sequence':<25} {'Grade':<12} {'MIC':<8} {'Diff':<7} {'Coupl':<7} {'Cost':<7}")
    print("-" * 80)

    for i, c in enumerate(top_candidates[:15]):
        seq_display = c.sequence[:22] + "..." if len(c.sequence) > 22 else c.sequence
        print(
            f"{i+1:<5} {seq_display:<25} {c.synthesis_grade:<12} "
            f"{c.predicted_mic:<8.4f} {c.synthesis_difficulty:<7.3f} "
            f"{c.coupling_efficiency:<7.4f} ${c.estimated_cost:<6.1f}"
        )

    # Grade distribution
    print("\n" + "-" * 40)
    print("SYNTHESIS GRADE DISTRIBUTION:")
    for grade, count in results["metrics"]["grade_distribution"].items():
        if count > 0:
            pct = count / len(top_candidates) * 100
            print(f"  {grade:<12}: {count:>3} ({pct:>5.1f}%)")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for synthesis optimization."""
    parser = argparse.ArgumentParser(
        description="Synthesis-Optimized AMP Design (Sequence-Space NSGA-II)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard run
    python B10_synthesis_optimization.py --output results/synthesis/

    # More thorough optimization
    python B10_synthesis_optimization.py --population 200 --generations 100

    # Shorter peptides (easier synthesis)
    python B10_synthesis_optimization.py --max-length 20

    # Dry run without VAE model
    python B10_synthesis_optimization.py --dry-run
        """,
    )

    parser.add_argument(
        "--population", type=int, default=100,
        help="Population size (default: 100)",
    )
    parser.add_argument(
        "--generations", type=int, default=50,
        help="Number of generations (default: 50)",
    )
    parser.add_argument(
        "--output", type=str, default="results/synthesis_optimized",
        help="Output directory (default: results/synthesis_optimized)",
    )
    parser.add_argument(
        "--min-length", type=int, default=12,
        help="Minimum peptide length (default: 12)",
    )
    parser.add_argument(
        "--max-length", type=int, default=30,
        help="Maximum peptide length (default: 30, shorter=easier synthesis)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to PeptideVAE checkpoint",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--top-n", type=int, default=50,
        help="Number of top candidates to export (default: 50)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without VAE model (uses mock predictor)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Handle dry run
    checkpoint_path = args.checkpoint
    if args.dry_run:
        checkpoint_path = None
        print("DRY RUN: Using mock MIC predictor")

    # Create optimizer
    optimizer = SynthesisNSGA2(
        population_size=args.population,
        generations=args.generations,
        min_length=args.min_length,
        max_length=args.max_length,
        checkpoint_path=checkpoint_path,
        verbose=not args.quiet,
        random_seed=args.seed,
    )

    # Run optimization
    candidates = optimizer.run()

    # Export results
    output_dir = Path(args.output)
    export_results(candidates, output_dir, top_n=args.top_n)

    print(f"\nSYNTHESIS OPTIMIZATION COMPLETE")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
