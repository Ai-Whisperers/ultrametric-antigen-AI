#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""B8: Microbiome-Safe AMP Design (Sequence-Space Optimization)

Research Idea Implementation - Carlos Brizuela

Design antimicrobial peptides that selectively kill pathogens while sparing
beneficial skin/gut microbiome members (commensals).

UPDATED 2026-01-05: Now uses validated sequence-space mutations instead of
latent-space decoding. All candidates are real peptide sequences (10-35 AA).

Target Selectivity:
- Kill: S. aureus (pathogen), MRSA, pathogenic P. acnes
- Spare: S. epidermidis (commensal), C. acnes, Lactobacillus, Corynebacterium

Key Features:
1. Selectivity index optimization (pathogen MIC / commensal MIC)
2. PeptideVAE activity prediction (Spearman r=0.74)
3. Multi-species activity modeling
4. Confidence scoring
5. Skin/gut microbiome context

Usage:
    python scripts/B8_microbiome_safe_amps.py --context skin
    python scripts/B8_microbiome_safe_amps.py --context gut --generations 100
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add paths
SCRIPT_DIR = Path(__file__).parent
PACKAGE_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "deliverables"))

# Import shared utilities
from shared.constants import AMINO_ACIDS, CHARGES, HYDROPHOBICITY

# Import from sequence optimizer
from scripts.sequence_nsga2 import (
    SequenceNSGA2,
    MutationOperators,
    ObjectiveFunctions,
    MIN_PEPTIDE_LENGTH,
    MAX_PEPTIDE_LENGTH,
)

# Try to import predictor
try:
    from scripts.predict_mic import PeptideMICPredictor
    HAS_PREDICTOR = True
except ImportError:
    HAS_PREDICTOR = False


# =============================================================================
# Microbiome Definitions
# =============================================================================

SKIN_MICROBIOME = {
    "pathogens": {
        "S_aureus": {
            "full_name": "Staphylococcus aureus",
            "gram": "positive",
            "target_mic": 4.0,
            "membrane_charge": -0.3,
        },
        "MRSA": {
            "full_name": "Methicillin-resistant S. aureus",
            "gram": "positive",
            "target_mic": 8.0,
            "membrane_charge": -0.25,
        },
        "P_acnes_pathogenic": {
            "full_name": "Cutibacterium acnes (pathogenic)",
            "gram": "positive",
            "target_mic": 8.0,
            "membrane_charge": -0.35,
        },
    },
    "commensals": {
        "S_epidermidis": {
            "full_name": "Staphylococcus epidermidis",
            "gram": "positive",
            "min_mic": 64.0,
            "membrane_charge": -0.15,
            "importance": "skin_barrier",
        },
        "C_acnes": {
            "full_name": "Cutibacterium acnes (commensal)",
            "gram": "positive",
            "min_mic": 64.0,
            "membrane_charge": -0.20,
            "importance": "lipid_metabolism",
        },
        "Corynebacterium": {
            "full_name": "Corynebacterium spp.",
            "gram": "positive",
            "min_mic": 32.0,
            "membrane_charge": -0.10,
            "importance": "moisture_regulation",
        },
    },
    "seed_sequences": [
        "KLAKLAKKLAKLAK",
        "KLWKKLKKALK",
        "FKCRRWQWRMKKLGAPS",
    ],
}

GUT_MICROBIOME = {
    "pathogens": {
        "C_difficile": {
            "full_name": "Clostridioides difficile",
            "gram": "positive",
            "target_mic": 4.0,
            "membrane_charge": -0.4,
        },
        "E_coli_pathogenic": {
            "full_name": "E. coli (pathogenic strains)",
            "gram": "negative",
            "target_mic": 8.0,
            "membrane_charge": -0.55,
        },
        "Salmonella": {
            "full_name": "Salmonella spp.",
            "gram": "negative",
            "target_mic": 8.0,
            "membrane_charge": -0.50,
        },
    },
    "commensals": {
        "Lactobacillus": {
            "full_name": "Lactobacillus spp.",
            "gram": "positive",
            "min_mic": 128.0,
            "membrane_charge": -0.05,
            "importance": "immune_modulation",
        },
        "Bifidobacterium": {
            "full_name": "Bifidobacterium spp.",
            "gram": "positive",
            "min_mic": 128.0,
            "membrane_charge": -0.08,
            "importance": "nutrient_absorption",
        },
        "Bacteroides": {
            "full_name": "Bacteroides spp.",
            "gram": "negative",
            "min_mic": 64.0,
            "membrane_charge": -0.35,
            "importance": "fiber_digestion",
        },
    },
    "seed_sequences": [
        "GIGKFLHSAKKFGKAFVGEIMNS",  # Magainin 2
        "KLAKLAKKLAKLAK",
        "RLKKTFFKIVKTVKW",
    ],
}

MICROBIOME_CONTEXTS = {
    "skin": SKIN_MICROBIOME,
    "gut": GUT_MICROBIOME,
}


# =============================================================================
# Selectivity Objectives
# =============================================================================

@dataclass
class SelectiveCandidate:
    """AMP candidate with selectivity metrics."""

    sequence: str
    mic_pred: float
    selectivity_index: float
    pathogen_activity: float
    commensal_sparing: float
    toxicity_pred: float
    confidence: str
    properties: Dict

    def to_dict(self) -> Dict:
        return {
            "sequence": self.sequence,
            "length": len(self.sequence),
            "mic_pred": round(self.mic_pred, 4),
            "mic_ug_ml": round(10 ** self.mic_pred, 4),
            "selectivity_index": round(self.selectivity_index, 4),
            "pathogen_activity": round(self.pathogen_activity, 4),
            "commensal_sparing": round(self.commensal_sparing, 4),
            "toxicity_pred": round(self.toxicity_pred, 4),
            "confidence": self.confidence,
            "net_charge": round(self.properties.get("net_charge", 0), 2),
            "hydrophobicity": round(self.properties.get("hydrophobicity", 0), 3),
        }


class SelectivityObjectives(ObjectiveFunctions):
    """Objectives for microbiome-safe optimization."""

    def __init__(
        self,
        context: str,
        predictor: Optional["PeptideMICPredictor"] = None,
    ):
        super().__init__(predictor=predictor, use_heuristics=True)
        self.context = context
        self.microbiome = MICROBIOME_CONTEXTS[context]
        self.pathogens = self.microbiome["pathogens"]
        self.commensals = self.microbiome["commensals"]

    def evaluate(self, sequence: str) -> Tuple[float, float, float, float]:
        """Evaluate selectivity objectives.

        Returns:
            Tuple of (mic_pred, selectivity, pathogen_activity, commensal_sparing)
            Note: We MAXIMIZE selectivity and commensal_sparing, MINIMIZE others
        """
        mic_pred, toxicity_pred, _ = super().evaluate(sequence)

        # Calculate pathogen and commensal activities
        pathogen_activity = self._pathogen_activity(sequence)
        commensal_sparing = self._commensal_sparing(sequence)

        # Selectivity index: commensal_sparing / pathogen_activity
        # Higher = better (want to kill pathogens, spare commensals)
        if pathogen_activity > 0:
            selectivity = commensal_sparing / pathogen_activity
        else:
            selectivity = 0.0

        # Return: minimize MIC, maximize selectivity, minimize pathogen_activity (lower=more active), maximize commensal_sparing
        # For DEAP fitness, we need to return values to minimize/maximize with correct signs
        return mic_pred, -selectivity, pathogen_activity, -commensal_sparing

    def _pathogen_activity(self, sequence: str) -> float:
        """Predict activity against pathogens (lower = more active)."""
        n = len(sequence)
        if n == 0:
            return 10.0

        score = 0.0
        charge = sum(CHARGES.get(aa, 0) for aa in sequence)
        hydro = sum(HYDROPHOBICITY.get(aa, 0) for aa in sequence) / n

        for name, pathogen in self.pathogens.items():
            # Activity based on charge-membrane interaction
            membrane_charge = pathogen["membrane_charge"]

            # Cationic peptides are more active against negative membranes
            charge_factor = max(0, charge * abs(membrane_charge))

            # Gram-specific factors
            if pathogen["gram"] == "negative":
                # Need higher charge and amphipathicity for Gram-
                if charge >= 4 and hydro > 0.3:
                    score += 0.2 * charge_factor
                else:
                    score -= 0.1
            else:
                # Gram+ more permeable
                if charge >= 2:
                    score += 0.3 * charge_factor

        # Lower score = more active
        return max(0.1, 1.0 - score / len(self.pathogens))

    def _commensal_sparing(self, sequence: str) -> float:
        """Predict sparing of commensals (higher = better)."""
        n = len(sequence)
        if n == 0:
            return 0.0

        score = 0.0
        charge = sum(CHARGES.get(aa, 0) for aa in sequence)
        hydro = sum(HYDROPHOBICITY.get(aa, 0) for aa in sequence) / n

        for name, commensal in self.commensals.items():
            membrane_charge = commensal["membrane_charge"]

            # Commensals with less negative membranes are naturally more resistant
            # Low membrane charge = harder for cationic AMPs to bind
            resistance = 1.0 - abs(membrane_charge)

            # Moderate charge is better for selectivity
            if 2 <= charge <= 5:
                score += 0.3 * resistance
            elif charge > 8:
                # Very high charge kills everything
                score -= 0.2

            # Moderate hydrophobicity is selective
            if 0.3 <= hydro <= 0.5:
                score += 0.2 * resistance

        return max(0, score / len(self.commensals))


# =============================================================================
# Microbiome-Safe Optimizer
# =============================================================================

class MicrobiomeNSGA2(SequenceNSGA2):
    """NSGA-II optimizer for microbiome-safe AMP design."""

    def __init__(
        self,
        context: str = "skin",
        population_size: int = 100,
        generations: int = 50,
        checkpoint_path: Optional[Path] = None,
        verbose: bool = True,
        random_seed: Optional[int] = None,
    ):
        if context not in MICROBIOME_CONTEXTS:
            raise ValueError(f"Unknown context: {context}. "
                           f"Available: {list(MICROBIOME_CONTEXTS.keys())}")

        self.context = context
        self.microbiome = MICROBIOME_CONTEXTS[context]

        seed_sequences = self.microbiome.get("seed_sequences", [
            "KLAKLAKKLAKLAK",
            "KLWKKLKKALK",
        ])

        super().__init__(
            seed_sequences=seed_sequences,
            population_size=population_size,
            generations=generations,
            checkpoint_path=checkpoint_path,
            verbose=verbose,
            random_seed=random_seed,
        )

        self.objectives = SelectivityObjectives(
            context=context,
            predictor=self.predictor,
        )

    def _setup_deap(self) -> None:
        """Configure DEAP for 4-objective optimization."""
        from scripts.sequence_nsga2 import deap, _lazy_import
        _lazy_import()

        # 4 objectives: minimize MIC, minimize -selectivity, minimize pathogen_activity, minimize -commensal_sparing
        if not hasattr(deap.creator, "FitnessMicrobiome"):
            deap.creator.create("FitnessMicrobiome", deap.base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
        if not hasattr(deap.creator, "IndividualMicrobiome"):
            deap.creator.create("IndividualMicrobiome", list, fitness=deap.creator.FitnessMicrobiome)

        self.toolbox = deap.base.Toolbox()
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", deap.tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", deap.tools.selNSGA2)

    def _create_individual(self):
        from scripts.sequence_nsga2 import deap
        base_seq = random.choice(self.seed_sequences)
        if random.random() < 0.5:
            mutated, _ = self.mutation_ops.mutate(base_seq)
            base_seq = mutated
        return deap.creator.IndividualMicrobiome([base_seq])

    def _evaluate_individual(self, individual) -> Tuple[float, float, float, float]:
        sequence = individual[0]
        return self.objectives.evaluate(sequence)

    def run(self) -> List[SelectiveCandidate]:
        """Run optimization and return selective candidates."""
        from scripts.sequence_nsga2 import deap

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"MICROBIOME-SAFE AMP DESIGN: {self.context.upper()} CONTEXT")
            print(f"{'='*60}")
            print(f"Pathogens to kill: {', '.join(self.microbiome['pathogens'].keys())}")
            print(f"Commensals to spare: {', '.join(self.microbiome['commensals'].keys())}")
            print(f"Population: {self.population_size}, Generations: {self.generations}")
            print()

        # Initialize population
        population = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Main evolution loop
        for gen in range(self.generations):
            offspring = deap.tools.selTournamentDCD(population, len(population))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            population = self.toolbox.select(population + offspring, self.population_size)

            if self.verbose and (gen + 1) % 10 == 0:
                pareto = deap.tools.sortNondominated(population, len(population), first_front_only=True)[0]
                best_selectivity = max(-ind.fitness.values[1] for ind in pareto)
                print(f"Generation {gen+1}/{self.generations} | "
                      f"Pareto size: {len(pareto)} | Best selectivity: {best_selectivity:.2f}")

        # Extract final Pareto front
        final_pareto = deap.tools.sortNondominated(population, len(population), first_front_only=True)[0]

        # Convert to SelectiveCandidate objects
        candidates = []
        for ind in final_pareto:
            seq = ind[0]
            n = len(seq)
            props = {
                "net_charge": sum(CHARGES.get(aa, 0) for aa in seq),
                "hydrophobicity": sum(HYDROPHOBICITY.get(aa, 0) for aa in seq) / n if n > 0 else 0,
            }

            # Determine confidence
            if self.predictor:
                try:
                    result = self.predictor.predict(seq)
                    confidence = result.confidence
                except Exception:
                    confidence = "Unknown"
            else:
                confidence = "Heuristic"

            candidates.append(SelectiveCandidate(
                sequence=seq,
                mic_pred=ind.fitness.values[0],
                selectivity_index=-ind.fitness.values[1],  # Negate back
                pathogen_activity=ind.fitness.values[2],
                commensal_sparing=-ind.fitness.values[3],  # Negate back
                toxicity_pred=0.0,  # Not tracked in this version
                confidence=confidence,
                properties=props,
            ))

        # Sort by selectivity (descending)
        candidates.sort(key=lambda c: c.selectivity_index, reverse=True)

        if self.verbose:
            print(f"\nOptimization complete: {len(candidates)} selective candidates")

        return candidates


# =============================================================================
# Export Functions
# =============================================================================

def export_results(
    context: str,
    candidates: List[SelectiveCandidate],
    output_dir: Path,
) -> None:
    """Export optimization results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    microbiome = MICROBIOME_CONTEXTS[context]

    # JSON export
    results = {
        "context": context,
        "pathogens": list(microbiome["pathogens"].keys()),
        "commensals": list(microbiome["commensals"].keys()),
        "n_candidates": len(candidates),
        "candidates": [c.to_dict() for c in candidates],
    }

    json_path = output_dir / f"microbiome_safe_{context}_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exported: {json_path}")

    # CSV export
    csv_path = output_dir / f"microbiome_safe_{context}_candidates.csv"
    with open(csv_path, "w", newline="") as f:
        if candidates:
            writer = csv.DictWriter(f, fieldnames=list(candidates[0].to_dict().keys()))
            writer.writeheader()
            for c in candidates:
                writer.writerow(c.to_dict())
    print(f"Exported: {csv_path}")

    # FASTA export
    fasta_path = output_dir / f"microbiome_safe_{context}_peptides.fasta"
    with open(fasta_path, "w") as f:
        for i, c in enumerate(candidates, 1):
            f.write(f">{context}_rank{i:02d}_SI{c.selectivity_index:.2f}_{c.confidence}\n")
            f.write(f"{c.sequence}\n")
    print(f"Exported: {fasta_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Microbiome-Safe AMP Design (Sequence-Space NSGA-II)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python B8_microbiome_safe_amps.py --context skin
    python B8_microbiome_safe_amps.py --context gut --generations 100
    python B8_microbiome_safe_amps.py --all-contexts --output results/
        """,
    )

    parser.add_argument(
        "--context",
        type=str,
        default="skin",
        choices=list(MICROBIOME_CONTEXTS.keys()),
        help="Microbiome context (default: skin)",
    )
    parser.add_argument(
        "--all-contexts",
        action="store_true",
        help="Run for all microbiome contexts",
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
        "--output", "-o",
        type=Path,
        default=PACKAGE_DIR / "results" / "microbiome_safe",
        help="Output directory",
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

    contexts = list(MICROBIOME_CONTEXTS.keys()) if args.all_contexts else [args.context]

    all_results = {}
    for context in contexts:
        optimizer = MicrobiomeNSGA2(
            context=context,
            population_size=args.population,
            generations=args.generations,
            checkpoint_path=args.checkpoint,
            verbose=not args.quiet,
            random_seed=args.seed,
        )

        candidates = optimizer.run()
        export_results(context, candidates, args.output)
        all_results[context] = candidates

    # Summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    for context, candidates in all_results.items():
        print(f"\n{context.upper()} Microbiome:")
        print(f"  Candidates: {len(candidates)}")
        if candidates:
            top = candidates[0]
            print(f"  Top: {top.sequence[:25]}{'...' if len(top.sequence) > 25 else ''}")
            print(f"  Selectivity Index: {top.selectivity_index:.2f} | Confidence: {top.confidence}")


if __name__ == "__main__":
    main()
