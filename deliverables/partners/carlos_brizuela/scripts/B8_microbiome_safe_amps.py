# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""B8: Microbiome-Safe AMP Design

Research Idea Implementation - Carlos Brizuela

Design antimicrobial peptides that selectively kill pathogens while sparing
beneficial skin/gut microbiome members (commensals).

Target Selectivity:
- Kill: S. aureus (pathogen), MRSA
- Spare: S. epidermidis (commensal), C. acnes, Lactobacillus

Key Features:
1. Selectivity index optimization (pathogen MIC / commensal MIC)
2. Multi-species activity prediction (using DRAMP-trained models)
3. Commensal-friendly property optimization
4. Skin/gut microbiome context

Usage:
    python scripts/B8_microbiome_safe_amps.py --output results/microbiome_safe/

    # Use trained ML models (recommended):
    python scripts/B8_microbiome_safe_amps.py --use-dramp
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
    """Load trained activity prediction models from DRAMP data."""
    global _TRAINED_MODELS

    if _TRAINED_MODELS:
        return _TRAINED_MODELS

    if not HAS_JOBLIB:
        print("Warning: joblib not available, falling back to heuristic models")
        return {}

    models_dir = Path(__file__).parent.parent / "models"
    if not models_dir.exists():
        print(f"Warning: Models directory not found at {models_dir}")
        return {}

    # Load available models
    model_files = {
        "staphylococcus": "activity_staphylococcus.joblib",
        "general": "activity_general.joblib",
    }

    for key, filename in model_files.items():
        model_path = models_dir / filename
        if model_path.exists():
            try:
                _TRAINED_MODELS[key] = joblib.load(model_path)
                print(f"Loaded trained model: {key}")
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")

    return _TRAINED_MODELS


def compute_ml_features(sequence: str) -> np.ndarray:
    """Compute features for ML model prediction."""
    props = compute_peptide_properties(sequence)

    # Amino acid composition (20 features)
    aa_list = list(AA_PROPERTIES.keys())
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


# Microbiome species definitions
SKIN_MICROBIOME = {
    "pathogens": {
        "S_aureus": {
            "full_name": "Staphylococcus aureus",
            "gram": "positive",
            "target_mic": 4.0,  # Target MIC in μg/mL
            "membrane_charge": -0.3,
            "role": "pathogen",
        },
        "MRSA": {
            "full_name": "Methicillin-resistant S. aureus",
            "gram": "positive",
            "target_mic": 8.0,
            "membrane_charge": -0.25,
            "role": "pathogen",
        },
        "P_acnes_pathogenic": {
            "full_name": "Cutibacterium acnes (pathogenic strain)",
            "gram": "positive",
            "target_mic": 8.0,
            "membrane_charge": -0.35,
            "role": "pathogen",
        },
    },
    "commensals": {
        "S_epidermidis": {
            "full_name": "Staphylococcus epidermidis",
            "gram": "positive",
            "min_mic": 64.0,  # Want high MIC (low activity)
            "membrane_charge": -0.15,  # Less negative = more resistant
            "role": "commensal",
            "importance": "skin_barrier",
        },
        "C_acnes": {
            "full_name": "Cutibacterium acnes (commensal)",
            "gram": "positive",
            "min_mic": 64.0,
            "membrane_charge": -0.20,
            "role": "commensal",
            "importance": "lipid_metabolism",
        },
        "Corynebacterium": {
            "full_name": "Corynebacterium spp.",
            "gram": "positive",
            "min_mic": 32.0,
            "membrane_charge": -0.10,
            "role": "commensal",
            "importance": "moisture_regulation",
        },
        "Malassezia": {
            "full_name": "Malassezia spp. (fungal)",
            "gram": "fungal",
            "min_mic": 64.0,
            "membrane_charge": -0.40,
            "role": "commensal",
            "importance": "lipid_metabolism",
        },
    },
}


# AA_PROPERTIES and compute_peptide_properties imported from shared.peptide_utils


@dataclass
class MicrobiomeSafeAMP:
    """AMP candidate with selectivity profile."""

    sequence: str
    length: int
    net_charge: float
    hydrophobicity: float
    pathogen_mics: dict  # {species: predicted MIC}
    commensal_mics: dict  # {species: predicted MIC}
    selectivity_index: float  # Higher = more selective
    toxicity_score: float
    combined_score: float
    latent: np.ndarray


def predict_mic(sequence: str, species_info: dict) -> float:
    """Predict MIC for a peptide against a species.

    Uses trained ML models when USE_TRAINED_MODELS is True.
    Falls back to heuristic prediction based on:
    1. Peptide charge vs membrane charge (electrostatic)
    2. Hydrophobicity (membrane insertion)
    3. Length (coverage)
    """
    # Try ML model prediction first
    if USE_TRAINED_MODELS:
        models = load_trained_models()
        model = None

        # Select appropriate model based on species
        species_name = species_info.get("full_name", "").lower()
        if "staphylococcus" in species_name or "aureus" in species_name:
            model = models.get("staphylococcus") or models.get("general")
        else:
            model = models.get("general")

        if model is not None:
            try:
                features = compute_ml_features(sequence).reshape(1, -1)
                # Model predicts activity score, convert to MIC-like scale
                prediction = model.predict(features)[0]
                # Transform to MIC (lower prediction = more active = lower MIC)
                mic = 16.0 * np.exp(-prediction * 0.5)
                return max(0.5, min(256, mic))
            except Exception:
                pass  # Fall through to heuristic

    # Heuristic fallback
    props = compute_peptide_properties(sequence)

    # Base MIC (moderate activity)
    base_mic = 16.0

    # Electrostatic interaction
    membrane_charge = species_info["membrane_charge"]
    charge_factor = props["net_charge"] * abs(membrane_charge)

    # Higher peptide charge + more negative membrane = lower MIC (better activity)
    if membrane_charge < 0:
        mic = base_mic * np.exp(-charge_factor * 0.3)
    else:
        mic = base_mic * np.exp(charge_factor * 0.3)

    # Hydrophobicity effect
    if species_info.get("gram") == "positive":
        # Moderate hydrophobicity good for gram-positive
        optimal_hydro = 0.4
    elif species_info.get("gram") == "fungal":
        # Higher hydrophobicity needed for fungal
        optimal_hydro = 0.6
    else:
        optimal_hydro = 0.35

    hydro_diff = abs(props["hydrophobicity"] - optimal_hydro)
    mic *= np.exp(hydro_diff * 0.5)

    # Length effect
    if 15 <= props["length"] <= 25:
        mic *= 0.8  # Optimal range
    elif props["length"] < 10 or props["length"] > 40:
        mic *= 2.0  # Suboptimal

    return max(0.5, min(256, mic))  # Clamp to reasonable range


def compute_selectivity_index(pathogen_mics: dict, commensal_mics: dict) -> float:
    """Compute selectivity index.

    SI = geometric_mean(commensal MICs) / geometric_mean(pathogen MICs)
    Higher SI = more selective (kills pathogens, spares commensals)
    """
    if not pathogen_mics or not commensal_mics:
        return 0.0

    pathogen_mic_values = list(pathogen_mics.values())
    commensal_mic_values = list(commensal_mics.values())

    # Geometric means
    pathogen_gm = np.exp(np.mean(np.log(pathogen_mic_values)))
    commensal_gm = np.exp(np.mean(np.log(commensal_mic_values)))

    return commensal_gm / pathogen_gm


def decode_latent_to_sequence(z: np.ndarray, length: int = 20) -> str:
    """Decode latent vector to peptide sequence."""
    np.random.seed(int(abs(z[0] * 1000)))

    charge_pref = np.tanh(z[0])
    hydro_pref = np.tanh(z[1])

    aa_list = list(AA_PROPERTIES.keys())
    probs = np.zeros(len(aa_list))

    for i, aa in enumerate(aa_list):
        props = AA_PROPERTIES[aa]
        charge_score = 1 - abs(props["charge"] - charge_pref)
        hydro_score = 1 - abs(props["hydrophobicity"] / 2 - hydro_pref)
        probs[i] = charge_score + hydro_score + 0.1

    # Ensure non-negative probabilities
    probs = np.clip(probs, 0.01, None)
    probs = probs / probs.sum()
    sequence = "".join(np.random.choice(aa_list, size=length, p=probs))

    return sequence


def evaluate_microbiome_safety(
    z: np.ndarray,
    microbiome: dict = SKIN_MICROBIOME,
) -> MicrobiomeSafeAMP:
    """Evaluate a peptide for microbiome safety."""
    sequence = decode_latent_to_sequence(z)
    props = compute_peptide_properties(sequence)

    # Predict MICs for pathogens
    pathogen_mics = {}
    for species, info in microbiome["pathogens"].items():
        pathogen_mics[species] = predict_mic(sequence, info)

    # Predict MICs for commensals
    commensal_mics = {}
    for species, info in microbiome["commensals"].items():
        commensal_mics[species] = predict_mic(sequence, info)

    # Compute selectivity
    selectivity = compute_selectivity_index(pathogen_mics, commensal_mics)

    # Toxicity (simplified)
    toxicity = max(0, props["hydrophobicity"] - 0.5) * 2
    toxicity += max(0, abs(props["net_charge"]) - 8) * 0.5

    # Combined score (maximize selectivity, minimize toxicity)
    combined = selectivity / (1 + toxicity)

    return MicrobiomeSafeAMP(
        sequence=sequence,
        length=len(sequence),
        net_charge=props["net_charge"],
        hydrophobicity=props["hydrophobicity"],
        pathogen_mics=pathogen_mics,
        commensal_mics=commensal_mics,
        selectivity_index=selectivity,
        toxicity_score=toxicity,
        combined_score=combined,
        latent=z,
    )


def optimize_microbiome_safe_amps(
    population_size: int = 200,
    generations: int = 50,
    seed: int = 42,
) -> list[MicrobiomeSafeAMP]:
    """Optimize for microbiome-safe AMPs using evolutionary strategy."""
    np.random.seed(seed)

    latent_dim = 16
    bounds = (-3.0, 3.0)

    # Initialize population
    population = []
    for _ in range(population_size):
        z = np.random.uniform(bounds[0], bounds[1], size=latent_dim)
        candidate = evaluate_microbiome_safety(z)
        population.append(candidate)

    print("\n" + "=" * 60)
    print("MICROBIOME-SAFE AMP OPTIMIZATION")
    print("=" * 60)
    print(f"Population: {population_size}")
    print(f"Generations: {generations}")
    print(f"Targets: Kill pathogens, spare commensals")
    print()

    for gen in range(generations):
        # Sort by combined score
        population.sort(key=lambda x: x.combined_score, reverse=True)

        # Print progress
        if gen % 10 == 0:
            best = population[0]
            avg_si = np.mean([p.selectivity_index for p in population[:10]])
            print(f"Gen {gen:4d}: Best SI={best.selectivity_index:.2f}, "
                  f"Avg SI (top 10)={avg_si:.2f}, "
                  f"Best score={best.combined_score:.3f}")

        # Selection: keep top 50%
        survivors = population[: population_size // 2]

        # Reproduction
        offspring = []
        while len(offspring) < population_size // 2:
            # Select parents
            p1, p2 = np.random.choice(len(survivors), 2, replace=False)
            parent1 = survivors[p1]
            parent2 = survivors[p2]

            # Crossover
            alpha = np.random.random(latent_dim)
            child_z = alpha * parent1.latent + (1 - alpha) * parent2.latent

            # Mutation
            if np.random.random() < 0.3:
                mutation = np.random.normal(0, 0.2, size=latent_dim)
                child_z += mutation
                child_z = np.clip(child_z, bounds[0], bounds[1])

            child = evaluate_microbiome_safety(child_z)
            offspring.append(child)

        population = survivors + offspring

    # Final sort
    population.sort(key=lambda x: x.combined_score, reverse=True)

    return population


def export_results(candidates: list[MicrobiomeSafeAMP], output_dir: Path) -> None:
    """Export optimization results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Top candidates
    top_candidates = candidates[:20]

    # JSON export
    results = {
        "objective": "Microbiome-safe AMP design",
        "selectivity_target": "Kill pathogens, spare commensals",
        "pathogens": list(SKIN_MICROBIOME["pathogens"].keys()),
        "commensals": list(SKIN_MICROBIOME["commensals"].keys()),
        "candidates": [
            {
                "rank": i + 1,
                "sequence": c.sequence,
                "length": c.length,
                "net_charge": c.net_charge,
                "hydrophobicity": round(c.hydrophobicity, 3),
                "selectivity_index": round(c.selectivity_index, 2),
                "toxicity_score": round(c.toxicity_score, 3),
                "combined_score": round(c.combined_score, 3),
                "pathogen_mics": {k: round(v, 1) for k, v in c.pathogen_mics.items()},
                "commensal_mics": {k: round(v, 1) for k, v in c.commensal_mics.items()},
            }
            for i, c in enumerate(top_candidates)
        ],
    }

    json_path = output_dir / "microbiome_safe_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exported results to {json_path}")

    # CSV export
    if HAS_PANDAS:
        records = []
        for i, c in enumerate(top_candidates):
            record = {
                "rank": i + 1,
                "sequence": c.sequence,
                "length": c.length,
                "net_charge": c.net_charge,
                "hydrophobicity": c.hydrophobicity,
                "selectivity_index": c.selectivity_index,
                "toxicity_score": c.toxicity_score,
            }
            # Add individual MICs
            for species, mic in c.pathogen_mics.items():
                record[f"MIC_{species}"] = mic
            for species, mic in c.commensal_mics.items():
                record[f"MIC_{species}"] = mic
            records.append(record)

        df = pd.DataFrame(records)
        csv_path = output_dir / "microbiome_safe_candidates.csv"
        df.to_csv(csv_path, index=False)
        print(f"Exported CSV to {csv_path}")

    # FASTA
    fasta_path = output_dir / "microbiome_safe_peptides.fasta"
    with open(fasta_path, "w") as f:
        for i, c in enumerate(top_candidates):
            f.write(f">microbiome_safe_rank{i+1:02d}_SI{c.selectivity_index:.1f}\n")
            f.write(f"{c.sequence}\n")
    print(f"Exported FASTA to {fasta_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TOP 10 MICROBIOME-SAFE CANDIDATES")
    print("=" * 60)
    print(f"{'Rank':<5} {'Sequence':<22} {'SI':<6} {'Charge':<7} {'Toxicity':<8}")
    print("-" * 55)
    for i, c in enumerate(top_candidates[:10]):
        seq_short = c.sequence[:20] + "..." if len(c.sequence) > 20 else c.sequence
        print(f"{i+1:<5} {seq_short:<22} {c.selectivity_index:<6.2f} "
              f"{c.net_charge:<7.1f} {c.toxicity_score:<8.3f}")


def main():
    """Main entry point."""
    global USE_TRAINED_MODELS

    parser = argparse.ArgumentParser(description="Microbiome-Safe AMP Design")
    parser.add_argument(
        "--population", type=int, default=200, help="Population size"
    )
    parser.add_argument(
        "--generations", type=int, default=50, help="Number of generations"
    )
    parser.add_argument(
        "--output", type=str, default="results/microbiome_safe", help="Output directory"
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
        load_trained_models()

    candidates = optimize_microbiome_safe_amps(
        population_size=args.population,
        generations=args.generations,
    )

    export_results(candidates, Path(args.output))

    print("\n" + "=" * 60)
    print("MICROBIOME-SAFE AMP DESIGN COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
