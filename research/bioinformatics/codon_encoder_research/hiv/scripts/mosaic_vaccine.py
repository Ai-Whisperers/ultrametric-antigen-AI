"""
Mosaic Vaccine Optimizer for HIV.

This module implements algorithms for designing mosaic vaccines that maximize:
1. Epitope coverage across diverse HIV sequences
2. HLA population coverage
3. Escape resistance (minimize escape probability)
4. Immunogenicity scores

Based on papers:
- Fischer et al. 2007: Polyvalent vaccines
- Barouch et al. 2010: Mosaic HIV vaccines
- Theiler & Korber 2018: Graph-based mosaic design

Requirements:
    pip install numpy pandas networkx

Author: Research Team
Date: December 2025
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Epitope:
    """Represents a T-cell epitope."""

    sequence: str
    position: int  # Start position in reference
    protein: str
    hla_restrictions: list[str]
    conservation: float  # 0-1, how conserved across sequences
    immunogenicity: float  # 0-1, predicted immunogenicity
    escape_probability: float  # 0-1, probability of escape
    fitness_cost: float  # Fitness cost if escaped


@dataclass
class MosaicSequence:
    """Represents a mosaic vaccine sequence."""

    sequence: str
    epitopes: list[Epitope]
    coverage_score: float
    hla_coverage: float
    escape_resistance: float
    immunogenicity_score: float


@dataclass
class MosaicVaccine:
    """Complete mosaic vaccine with multiple sequences."""

    sequences: list[MosaicSequence]
    n_sequences: int
    total_epitope_coverage: float
    population_coverage: float
    mean_escape_resistance: float
    diversity_score: float


def calculate_epitope_coverage(
    mosaic_epitopes: list[str],
    target_epitopes: list[str],
) -> float:
    """
    Calculate fraction of target epitopes covered by mosaic.

    Uses exact match - epitope must appear in mosaic sequence.

    Args:
        mosaic_epitopes: Epitopes in mosaic sequences
        target_epitopes: Target epitope set from natural sequences

    Returns:
        Coverage fraction (0-1)
    """
    if not target_epitopes:
        return 0.0

    covered = sum(1 for e in target_epitopes if e in mosaic_epitopes)
    return covered / len(target_epitopes)


def calculate_population_coverage(
    epitopes: list[Epitope],
    hla_frequencies: dict[str, dict[str, float]],
    population: str = "global",
) -> float:
    """
    Calculate population coverage of epitope set.

    Uses HLA allele frequencies to estimate fraction of population
    that would respond to at least one epitope.

    Args:
        epitopes: List of Epitope objects
        hla_frequencies: Dict mapping HLA alleles to population frequencies
        population: Target population

    Returns:
        Population coverage (0-1)
    """
    # Collect all HLA restrictions
    hla_covered = set()
    for epitope in epitopes:
        hla_covered.update(epitope.hla_restrictions)

    if not hla_covered:
        return 0.0

    # Calculate coverage using inclusion-exclusion approximation
    # P(at least one HLA) ≈ 1 - ∏(1 - freq_i)

    non_coverage = 1.0
    for hla in hla_covered:
        freq = hla_frequencies.get(hla, {}).get(population, 0.01)
        non_coverage *= (1 - freq)

    return 1 - non_coverage


def calculate_escape_resistance(epitopes: list[Epitope]) -> float:
    """
    Calculate overall escape resistance of epitope set.

    Higher score = harder to escape all epitopes.

    Args:
        epitopes: List of Epitope objects

    Returns:
        Escape resistance score (0-1)
    """
    if not epitopes:
        return 0.0

    # Weight by fitness cost - high fitness cost = good target
    total_weight = 0.0
    total_resistance = 0.0

    for epitope in epitopes:
        # Resistance = (1 - escape_prob) * fitness_cost
        resistance = (1 - epitope.escape_probability) * (1 + epitope.fitness_cost)
        weight = epitope.conservation * epitope.immunogenicity
        total_weight += weight
        total_resistance += weight * resistance

    if total_weight == 0:
        return 0.0

    return total_resistance / total_weight


def score_epitope(epitope: Epitope, weights: Optional[dict] = None) -> float:
    """
    Calculate composite score for single epitope.

    Args:
        epitope: Epitope object
        weights: Optional weight dictionary

    Returns:
        Composite score
    """
    weights = weights or {
        "conservation": 0.3,
        "immunogenicity": 0.25,
        "escape_resistance": 0.25,
        "hla_breadth": 0.2,
    }

    escape_resistance = (1 - epitope.escape_probability) * (1 + epitope.fitness_cost)
    hla_breadth = min(1.0, len(epitope.hla_restrictions) / 5)  # Normalize

    score = (
        weights["conservation"] * epitope.conservation +
        weights["immunogenicity"] * epitope.immunogenicity +
        weights["escape_resistance"] * escape_resistance +
        weights["hla_breadth"] * hla_breadth
    )

    return score


class MosaicOptimizer:
    """
    Optimizer for mosaic vaccine design.

    Uses greedy and genetic algorithms to find optimal epitope combinations.
    """

    def __init__(
        self,
        epitopes: list[Epitope],
        hla_frequencies: Optional[dict] = None,
        random_state: int = 42,
    ):
        """
        Initialize mosaic optimizer.

        Args:
            epitopes: Pool of candidate epitopes
            hla_frequencies: HLA frequency data
            random_state: Random seed
        """
        self.epitopes = epitopes
        self.hla_frequencies = hla_frequencies or {}
        self.random_state = random_state
        np.random.seed(random_state)

        # Pre-compute epitope scores
        self.epitope_scores = {e.sequence: score_epitope(e) for e in epitopes}

    def greedy_selection(
        self,
        n_epitopes: int = 10,
        min_hla_coverage: float = 0.8,
    ) -> list[Epitope]:
        """
        Greedy algorithm for epitope selection.

        Iteratively adds epitopes that maximize marginal gain.

        Args:
            n_epitopes: Number of epitopes to select
            min_hla_coverage: Minimum HLA coverage target

        Returns:
            List of selected epitopes
        """
        selected = []
        remaining = list(self.epitopes)

        while len(selected) < n_epitopes and remaining:
            best_epitope = None
            best_gain = -float("inf")

            for epitope in remaining:
                # Calculate marginal gain
                test_set = selected + [epitope]

                # Coverage gain
                coverage = calculate_population_coverage(
                    test_set, self.hla_frequencies
                )

                # Diversity gain (different positions/proteins)
                positions = len(set(e.position for e in test_set))
                proteins = len(set(e.protein for e in test_set))
                diversity = (positions + proteins) / (2 * len(test_set))

                # Escape resistance
                escape_res = calculate_escape_resistance(test_set)

                # Marginal gain
                gain = (
                    0.4 * coverage +
                    0.3 * escape_res +
                    0.2 * diversity +
                    0.1 * self.epitope_scores[epitope.sequence]
                )

                if gain > best_gain:
                    best_gain = gain
                    best_epitope = epitope

            if best_epitope is None:
                break

            selected.append(best_epitope)
            remaining.remove(best_epitope)

            # Check coverage target
            current_coverage = calculate_population_coverage(
                selected, self.hla_frequencies
            )
            if current_coverage >= min_hla_coverage and len(selected) >= 5:
                break

        return selected

    def genetic_optimization(
        self,
        n_epitopes: int = 10,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.1,
    ) -> list[Epitope]:
        """
        Genetic algorithm for epitope selection.

        Args:
            n_epitopes: Number of epitopes per solution
            population_size: Number of solutions in population
            generations: Number of generations
            mutation_rate: Mutation probability

        Returns:
            Best epitope set found
        """
        n_candidates = len(self.epitopes)

        if n_candidates < n_epitopes:
            return list(self.epitopes)

        # Initialize population (random subsets)
        population = []
        for _ in range(population_size):
            indices = np.random.choice(n_candidates, n_epitopes, replace=False)
            population.append(set(indices))

        def fitness(indices):
            """Calculate fitness of epitope set."""
            epitope_set = [self.epitopes[i] for i in indices]
            coverage = calculate_population_coverage(
                epitope_set, self.hla_frequencies
            )
            escape_res = calculate_escape_resistance(epitope_set)
            scores = sum(self.epitope_scores[e.sequence] for e in epitope_set)
            return 0.4 * coverage + 0.4 * escape_res + 0.2 * (scores / n_epitopes)

        for gen in range(generations):
            # Evaluate fitness
            fitnesses = [fitness(ind) for ind in population]

            # Selection (tournament)
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                t1, t2 = np.random.choice(population_size, 2, replace=False)
                winner = population[t1] if fitnesses[t1] > fitnesses[t2] else population[t2]
                new_population.append(set(winner))

            # Crossover
            for i in range(0, population_size - 1, 2):
                if np.random.random() < 0.7:  # Crossover probability
                    p1, p2 = list(new_population[i]), list(new_population[i + 1])
                    # Single-point crossover
                    point = np.random.randint(1, n_epitopes)
                    c1 = set(p1[:point] + p2[point:])
                    c2 = set(p2[:point] + p1[point:])
                    # Fix sizes
                    while len(c1) < n_epitopes:
                        c1.add(np.random.randint(n_candidates))
                    while len(c2) < n_epitopes:
                        c2.add(np.random.randint(n_candidates))
                    c1 = set(list(c1)[:n_epitopes])
                    c2 = set(list(c2)[:n_epitopes])
                    new_population[i] = c1
                    new_population[i + 1] = c2

            # Mutation
            for i in range(population_size):
                if np.random.random() < mutation_rate:
                    # Replace one random epitope
                    ind = list(new_population[i])
                    replace_idx = np.random.randint(n_epitopes)
                    new_epitope = np.random.randint(n_candidates)
                    ind[replace_idx] = new_epitope
                    new_population[i] = set(ind)

            population = new_population

        # Return best solution
        fitnesses = [fitness(ind) for ind in population]
        best_idx = np.argmax(fitnesses)
        best_indices = list(population[best_idx])

        return [self.epitopes[i] for i in best_indices]

    def create_mosaic_sequence(
        self,
        epitopes: list[Epitope],
        linker: str = "AAY",
    ) -> MosaicSequence:
        """
        Create a mosaic sequence from selected epitopes.

        Concatenates epitopes with linkers in optimal order.

        Args:
            epitopes: Selected epitopes
            linker: Linker sequence between epitopes

        Returns:
            MosaicSequence object
        """
        # Sort epitopes by position for natural ordering
        sorted_epitopes = sorted(epitopes, key=lambda e: (e.protein, e.position))

        # Concatenate with linkers
        sequence = linker.join(e.sequence for e in sorted_epitopes)

        # Calculate scores
        coverage = calculate_population_coverage(sorted_epitopes, self.hla_frequencies)
        escape_res = calculate_escape_resistance(sorted_epitopes)
        immunogenicity = np.mean([e.immunogenicity for e in sorted_epitopes])

        return MosaicSequence(
            sequence=sequence,
            epitopes=sorted_epitopes,
            coverage_score=len(sorted_epitopes) / len(self.epitopes),
            hla_coverage=coverage,
            escape_resistance=escape_res,
            immunogenicity_score=immunogenicity,
        )


class PolyvalentDesigner:
    """
    Design polyvalent (multi-sequence) mosaic vaccines.

    Creates multiple mosaic sequences that together provide maximum coverage.
    """

    def __init__(
        self,
        optimizer: MosaicOptimizer,
        n_sequences: int = 2,
    ):
        """
        Initialize polyvalent designer.

        Args:
            optimizer: MosaicOptimizer instance
            n_sequences: Number of sequences in vaccine
        """
        self.optimizer = optimizer
        self.n_sequences = n_sequences

    def design_polyvalent(
        self,
        epitopes_per_sequence: int = 10,
    ) -> MosaicVaccine:
        """
        Design polyvalent mosaic vaccine.

        Uses iterative greedy approach - each sequence targets epitopes
        not well covered by previous sequences.

        Args:
            epitopes_per_sequence: Epitopes per mosaic sequence

        Returns:
            MosaicVaccine object
        """
        sequences = []
        covered_hlas = set()
        used_epitopes = set()

        for seq_idx in range(self.n_sequences):
            # Filter to epitopes targeting uncovered HLAs
            remaining = [
                e for e in self.optimizer.epitopes
                if e.sequence not in used_epitopes
            ]

            if not remaining:
                break

            # Prioritize epitopes with uncovered HLAs
            prioritized = []
            for e in remaining:
                new_hlas = set(e.hla_restrictions) - covered_hlas
                priority = len(new_hlas) + 0.5 * score_epitope(e)
                prioritized.append((e, priority))

            prioritized.sort(key=lambda x: x[1], reverse=True)

            # Select top epitopes
            n_select = min(epitopes_per_sequence, len(prioritized))
            selected = [e for e, _ in prioritized[:n_select]]

            # Create mosaic sequence
            mosaic = self.optimizer.create_mosaic_sequence(selected)
            sequences.append(mosaic)

            # Update covered HLAs
            for e in selected:
                covered_hlas.update(e.hla_restrictions)
                used_epitopes.add(e.sequence)

        # Calculate overall metrics
        all_epitopes = []
        for seq in sequences:
            all_epitopes.extend(seq.epitopes)

        total_coverage = calculate_population_coverage(
            all_epitopes, self.optimizer.hla_frequencies
        )

        mean_escape_res = np.mean([s.escape_resistance for s in sequences])

        # Diversity = number of unique proteins/positions covered
        proteins = len(set(e.protein for e in all_epitopes))
        positions = len(set(e.position for e in all_epitopes))
        diversity = (proteins + positions) / (2 * len(all_epitopes))

        return MosaicVaccine(
            sequences=sequences,
            n_sequences=len(sequences),
            total_epitope_coverage=len(all_epitopes) / len(self.optimizer.epitopes),
            population_coverage=total_coverage,
            mean_escape_resistance=mean_escape_res,
            diversity_score=diversity,
        )


def analyze_mosaic_vaccine(vaccine: MosaicVaccine) -> dict:
    """
    Comprehensive analysis of mosaic vaccine design.

    Args:
        vaccine: MosaicVaccine object

    Returns:
        Dictionary with analysis results
    """
    # Protein coverage
    proteins = defaultdict(int)
    for seq in vaccine.sequences:
        for epitope in seq.epitopes:
            proteins[epitope.protein] += 1

    # HLA coverage
    hlas = set()
    for seq in vaccine.sequences:
        for epitope in seq.epitopes:
            hlas.update(epitope.hla_restrictions)

    # Conservation distribution
    conservations = [
        e.conservation
        for seq in vaccine.sequences
        for e in seq.epitopes
    ]

    return {
        "n_sequences": vaccine.n_sequences,
        "total_epitopes": sum(len(s.epitopes) for s in vaccine.sequences),
        "unique_epitopes": len(set(
            e.sequence for s in vaccine.sequences for e in s.epitopes
        )),
        "proteins_covered": dict(proteins),
        "hla_alleles_covered": len(hlas),
        "population_coverage": vaccine.population_coverage,
        "mean_escape_resistance": vaccine.mean_escape_resistance,
        "mean_conservation": np.mean(conservations),
        "min_conservation": np.min(conservations),
        "diversity_score": vaccine.diversity_score,
    }


def generate_vaccine_report(vaccine: MosaicVaccine) -> str:
    """
    Generate detailed report for mosaic vaccine.

    Args:
        vaccine: MosaicVaccine object

    Returns:
        Formatted report string
    """
    analysis = analyze_mosaic_vaccine(vaccine)

    lines = [
        "=" * 70,
        "MOSAIC VACCINE DESIGN REPORT",
        "=" * 70,
        "",
        "OVERVIEW:",
        f"  Number of sequences: {analysis['n_sequences']}",
        f"  Total epitopes: {analysis['total_epitopes']}",
        f"  Unique epitopes: {analysis['unique_epitopes']}",
        "",
        "COVERAGE METRICS:",
        f"  Population coverage: {analysis['population_coverage']:.1%}",
        f"  HLA alleles covered: {analysis['hla_alleles_covered']}",
        f"  Diversity score: {analysis['diversity_score']:.3f}",
        "",
        "QUALITY METRICS:",
        f"  Mean escape resistance: {analysis['mean_escape_resistance']:.3f}",
        f"  Mean conservation: {analysis['mean_conservation']:.3f}",
        f"  Min conservation: {analysis['min_conservation']:.3f}",
        "",
        "PROTEIN COVERAGE:",
    ]

    for protein, count in sorted(analysis['proteins_covered'].items()):
        lines.append(f"  {protein}: {count} epitopes")

    lines.extend([
        "",
        "SEQUENCES:",
        "-" * 50,
    ])

    for i, seq in enumerate(vaccine.sequences, 1):
        lines.extend([
            f"\nSequence {i}:",
            f"  Length: {len(seq.sequence)} aa",
            f"  Epitopes: {len(seq.epitopes)}",
            f"  HLA coverage: {seq.hla_coverage:.1%}",
            f"  Escape resistance: {seq.escape_resistance:.3f}",
            "",
            f"  Sequence: {seq.sequence[:50]}...",
        ])

    lines.extend([
        "",
        "=" * 70,
    ])

    return "\n".join(lines)


def create_test_epitopes(n_epitopes: int = 50) -> list[Epitope]:
    """
    Create synthetic epitope data for testing.

    Args:
        n_epitopes: Number of epitopes to generate

    Returns:
        List of Epitope objects
    """
    np.random.seed(42)

    proteins = ["Gag", "Pol", "Env", "Nef", "Rev"]
    hla_alleles = [
        "A*02:01", "A*03:01", "A*11:01", "A*24:02",
        "B*07:02", "B*08:01", "B*27:05", "B*35:01", "B*57:01",
    ]

    aa = "ACDEFGHIKLMNPQRSTVWY"

    epitopes = []
    for i in range(n_epitopes):
        # Random 9-mer epitope
        sequence = "".join(np.random.choice(list(aa), 9))

        # Random properties
        protein = np.random.choice(proteins)
        position = np.random.randint(1, 500)
        n_hlas = np.random.randint(1, 4)
        hla_restrictions = list(np.random.choice(hla_alleles, n_hlas, replace=False))

        epitope = Epitope(
            sequence=sequence,
            position=position,
            protein=protein,
            hla_restrictions=hla_restrictions,
            conservation=np.random.uniform(0.5, 1.0),
            immunogenicity=np.random.uniform(0.3, 0.9),
            escape_probability=np.random.uniform(0.1, 0.5),
            fitness_cost=np.random.uniform(0.0, 0.3),
        )
        epitopes.append(epitope)

    return epitopes


# Example usage
if __name__ == "__main__":
    print("Testing Mosaic Vaccine Optimizer Module")
    print("=" * 50)

    # Create test epitopes
    print("\nGenerating test epitope pool...")
    epitopes = create_test_epitopes(n_epitopes=50)
    print(f"  Created {len(epitopes)} test epitopes")

    # HLA frequencies (simplified)
    hla_frequencies = {
        "A*02:01": {"global": 0.25, "European": 0.28, "African": 0.12},
        "A*03:01": {"global": 0.12, "European": 0.14, "African": 0.10},
        "A*11:01": {"global": 0.10, "European": 0.06, "African": 0.05},
        "A*24:02": {"global": 0.15, "European": 0.10, "African": 0.08},
        "B*07:02": {"global": 0.10, "European": 0.12, "African": 0.08},
        "B*08:01": {"global": 0.08, "European": 0.10, "African": 0.04},
        "B*27:05": {"global": 0.04, "European": 0.05, "African": 0.02},
        "B*35:01": {"global": 0.08, "European": 0.09, "African": 0.10},
        "B*57:01": {"global": 0.04, "European": 0.05, "African": 0.06},
    }

    # Create optimizer
    print("\nInitializing optimizer...")
    optimizer = MosaicOptimizer(epitopes, hla_frequencies)

    # Greedy selection
    print("\nRunning greedy epitope selection...")
    greedy_epitopes = optimizer.greedy_selection(n_epitopes=10)
    print(f"  Selected {len(greedy_epitopes)} epitopes")

    coverage = calculate_population_coverage(greedy_epitopes, hla_frequencies)
    print(f"  Population coverage: {coverage:.1%}")

    # Genetic algorithm
    print("\nRunning genetic algorithm optimization...")
    ga_epitopes = optimizer.genetic_optimization(
        n_epitopes=10,
        population_size=50,
        generations=20,
    )
    print(f"  Selected {len(ga_epitopes)} epitopes")

    ga_coverage = calculate_population_coverage(ga_epitopes, hla_frequencies)
    print(f"  Population coverage: {ga_coverage:.1%}")

    # Create mosaic sequence
    print("\nCreating mosaic sequence...")
    mosaic = optimizer.create_mosaic_sequence(ga_epitopes)
    print(f"  Sequence length: {len(mosaic.sequence)}")
    print(f"  Escape resistance: {mosaic.escape_resistance:.3f}")

    # Design polyvalent vaccine
    print("\n" + "=" * 50)
    print("Designing bivalent mosaic vaccine...")

    designer = PolyvalentDesigner(optimizer, n_sequences=2)
    vaccine = designer.design_polyvalent(epitopes_per_sequence=8)

    print(generate_vaccine_report(vaccine))

    print("Module testing complete!")
