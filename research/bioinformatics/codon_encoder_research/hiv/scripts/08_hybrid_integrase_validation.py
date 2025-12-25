#!/usr/bin/env python3
"""Hybrid Integrase Validation: Clinical Data vs Geometric Predictions.

This experiment validates the hybrid AF3 pipeline by comparing
our geometric reveal scores against clinically validated:
1. INSTI (Integrase Strand Transfer Inhibitor) resistance mutations
2. CTL escape mutations
3. LEDGF interface disruption candidates

Key Question: Do our ternary VAE reveal scores correlate with
clinical outcomes (resistance level, fitness cost)?
"""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# Add paths for imports
# Script is at: research/bioinformatics/codon_encoder_research/hiv/scripts/
# Need 6 parents to reach project root
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
alphafold_path = project_root / "research" / "alphafold3"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(alphafold_path))

# Import hybrid predictor
try:
    from hybrid.structure_predictor import HybridStructurePredictor

    HAS_HYBRID = True
except ImportError as e:
    print(f"Warning: Could not import hybrid predictor: {e}")
    HAS_HYBRID = False

# Import local hyperbolic utils for fallback
try:
    from hyperbolic_utils import (CODON_TABLE, encode_codon,
                                  get_synonymous_codons, poincare_distance)

    HAS_LOCAL_ENCODER = True
except ImportError:
    HAS_LOCAL_ENCODER = False


# =============================================================================
# HIV-1 INTEGRASE SEQUENCE (HXB2 Reference)
# =============================================================================

HIV1_INTEGRASE_SEQUENCE = (
    "FLDGIDKAQEEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGK"
    "IILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSNFTSTTVKAACWWAGIKQEFGIP"
    "YNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDIIATDIQTKEL"
    "QKQITKIQNFRVYYRDSRDPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASG"
    "RQED"
)


# =============================================================================
# CLINICAL DATA: INSTI Resistance Mutations
# =============================================================================

INSTI_RESISTANCE_MUTATIONS = {
    "Y143R": {
        "position": 143,
        "wt_aa": "Y",
        "mut_aa": "R",
        "resistance_level": "high",
        "drugs": ["RAL"],
        "fitness_impact": "moderate_decrease",
        "clinical_padic_distance": 5.082528591156006,
        "mechanism": "Active site proximity",
    },
    "Q148H": {
        "position": 148,
        "wt_aa": "Q",
        "mut_aa": "H",
        "resistance_level": "high",
        "drugs": ["RAL", "EVG", "DTG"],
        "fitness_impact": "high_decrease",
        "clinical_padic_distance": 3.5381722450256348,
        "mechanism": "Catalytic triad adjacent",
    },
    "N155H": {
        "position": 155,
        "wt_aa": "N",
        "mut_aa": "H",
        "resistance_level": "high",
        "drugs": ["RAL", "EVG"],
        "fitness_impact": "moderate_decrease",
        "clinical_padic_distance": 4.187370777130127,
        "mechanism": "Metal coordination",
    },
    "R263K": {
        "position": 263,
        "wt_aa": "R",
        "mut_aa": "K",
        "resistance_level": "low",
        "drugs": ["DTG"],
        "fitness_impact": "high_decrease",
        "clinical_padic_distance": 4.396583557128906,
        "mechanism": "DNA binding region",
    },
}


# =============================================================================
# REVEAL MUTATION CANDIDATES (from 7 Conjectures)
# =============================================================================

REVEAL_CANDIDATES = {
    "E166K": {
        "position": 166,
        "wt_aa": "E",
        "mut_aa": "K",
        "predicted_reveal_score": 34.93,
        "mechanism": "Salt bridge reversal at LEDGF K364",
        "is_ledgf_interface": True,
    },
    "K175E": {
        "position": 175,
        "wt_aa": "K",
        "mut_aa": "E",
        "predicted_reveal_score": 34.93,
        "mechanism": "Charge reversal at helix",
        "is_ledgf_interface": True,
    },
    "W131A": {
        "position": 131,
        "wt_aa": "W",
        "mut_aa": "A",
        "predicted_reveal_score": 33.03,
        "mechanism": "Aromatic cap removal",
        "is_ledgf_interface": True,
    },
    "Q168E": {
        "position": 168,
        "wt_aa": "Q",
        "mut_aa": "E",
        "predicted_reveal_score": 28.5,
        "mechanism": "H-bond to charge",
        "is_ledgf_interface": True,
    },
    "I161G": {
        "position": 161,
        "wt_aa": "I",
        "mut_aa": "G",
        "predicted_reveal_score": 26.2,
        "mechanism": "Hydrophobic core disruption",
        "is_ledgf_interface": True,
    },
}


def resistance_level_to_numeric(level: str) -> float:
    """Convert resistance level to numeric score."""
    mapping = {"low": 1, "moderate": 2, "high": 3}
    return mapping.get(level.lower(), 0)


def fitness_impact_to_numeric(impact: str) -> float:
    """Convert fitness impact to numeric score (higher = worse fitness)."""
    mapping = {
        "minimal": 0.5,
        "low_decrease": 1,
        "moderate_decrease": 2,
        "high_decrease": 3,
    }
    return mapping.get(impact.lower(), 0)


def compute_fallback_reveal_score(wt_aa: str, mut_aa: str) -> float:
    """Compute reveal score without the full model."""
    # Representative codons for each amino acid
    REPRESENTATIVE_CODONS = {
        "A": "GCT",
        "R": "CGT",
        "N": "AAT",
        "D": "GAT",
        "C": "TGT",
        "Q": "CAA",
        "E": "GAA",
        "G": "GGT",
        "H": "CAT",
        "I": "ATT",
        "L": "CTT",
        "K": "AAA",
        "M": "ATG",
        "F": "TTT",
        "P": "CCT",
        "S": "TCT",
        "T": "ACT",
        "W": "TGG",
        "Y": "TAT",
        "V": "GTT",
    }

    wt_codon = REPRESENTATIVE_CODONS.get(wt_aa, "NNN")
    mut_codon = REPRESENTATIVE_CODONS.get(mut_aa, "NNN")

    # Simple Hamming-based distance with position weights
    weights = [3.0, 2.0, 1.0]  # First position most conserved
    distance = 0.0
    for i, (n1, n2) in enumerate(zip(wt_codon, mut_codon)):
        if n1 != n2:
            distance += weights[i]

    return distance


def run_validation_experiment(use_hybrid: bool = True) -> Dict:
    """Run the validation experiment.

    Compares:
    1. Clinical INSTI resistance mutations
    2. Our reveal mutation candidates
    3. Correlation between reveal scores and clinical outcomes
    """
    results = {
        "metadata": {
            "experiment": "Hybrid Integrase Validation",
            "hybrid_predictor_used": use_hybrid and HAS_HYBRID,
            "sequence_length": len(HIV1_INTEGRASE_SEQUENCE),
        },
        "insti_resistance_validation": [],
        "reveal_candidates_analysis": [],
        "correlation_analysis": {},
    }

    # Initialize predictor if available
    predictor = None
    if use_hybrid and HAS_HYBRID:
        try:
            predictor = HybridStructurePredictor()
            print("Using HybridStructurePredictor")
        except Exception as e:
            print(f"Could not initialize predictor: {e}")

    # ==========================================================================
    # PART 1: Validate INSTI Resistance Mutations
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PART 1: INSTI Resistance Mutations")
    print("=" * 60)

    for mut_name, data in INSTI_RESISTANCE_MUTATIONS.items():
        print(f"\nAnalyzing {mut_name}...")

        # Compute reveal score
        if predictor is not None:
            try:
                prediction = predictor.predict_reveal_effect(wt_sequence=HIV1_INTEGRASE_SEQUENCE, mutation=mut_name)
                reveal_score = prediction.get("reveal_score", 0)
                mechanism = prediction.get("primary_mechanism", "unknown")
            except Exception as e:
                print(f"  Predictor error: {e}, using fallback")
                reveal_score = compute_fallback_reveal_score(data["wt_aa"], data["mut_aa"])
                mechanism = data["mechanism"]
        else:
            reveal_score = compute_fallback_reveal_score(data["wt_aa"], data["mut_aa"])
            mechanism = data["mechanism"]

        result = {
            "mutation": mut_name,
            "position": data["position"],
            "reveal_score": reveal_score,
            "clinical_padic_distance": data["clinical_padic_distance"],
            "resistance_level": data["resistance_level"],
            "resistance_numeric": resistance_level_to_numeric(data["resistance_level"]),
            "fitness_impact": data["fitness_impact"],
            "fitness_numeric": fitness_impact_to_numeric(data["fitness_impact"]),
            "drugs": data["drugs"],
            "mechanism": mechanism,
        }
        results["insti_resistance_validation"].append(result)

        print(f"  Reveal Score: {reveal_score:.2f}")
        print(f"  Clinical p-adic: {data['clinical_padic_distance']:.2f}")
        print(f"  Resistance: {data['resistance_level']}, Fitness: {data['fitness_impact']}")

    # ==========================================================================
    # PART 2: Analyze Reveal Candidates
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PART 2: Reveal Mutation Candidates")
    print("=" * 60)

    for mut_name, data in REVEAL_CANDIDATES.items():
        print(f"\nAnalyzing {mut_name}...")

        if predictor is not None:
            try:
                prediction = predictor.predict_reveal_effect(wt_sequence=HIV1_INTEGRASE_SEQUENCE, mutation=mut_name)
                reveal_score = prediction.get("reveal_score", 0)
                is_ledgf = prediction.get("is_ledgf_interface", False)
                mechanism = prediction.get("primary_mechanism", "unknown")
            except Exception as e:
                print(f"  Predictor error: {e}, using fallback")
                reveal_score = compute_fallback_reveal_score(data["wt_aa"], data["mut_aa"])
                is_ledgf = data["is_ledgf_interface"]
                mechanism = data["mechanism"]
        else:
            reveal_score = compute_fallback_reveal_score(data["wt_aa"], data["mut_aa"])
            is_ledgf = data["is_ledgf_interface"]
            mechanism = data["mechanism"]

        result = {
            "mutation": mut_name,
            "position": data["position"],
            "computed_reveal_score": reveal_score,
            "predicted_reveal_score": data["predicted_reveal_score"],
            "is_ledgf_interface": is_ledgf,
            "mechanism": mechanism,
            "therapeutic_potential": ("high" if reveal_score > 4.0 else "moderate"),
        }
        results["reveal_candidates_analysis"].append(result)

        print(f"  Computed Reveal Score: {reveal_score:.2f}")
        print(f"  Predicted (from conjectures): {data['predicted_reveal_score']:.2f}")
        print(f"  LEDGF Interface: {is_ledgf}")

    # ==========================================================================
    # PART 3: Correlation Analysis
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PART 3: Correlation Analysis")
    print("=" * 60)

    # Extract arrays for correlation
    insti_results = results["insti_resistance_validation"]

    reveal_scores = np.array([r["reveal_score"] for r in insti_results])
    resistance_levels = np.array([r["resistance_numeric"] for r in insti_results])
    fitness_impacts = np.array([r["fitness_numeric"] for r in insti_results])
    clinical_distances = np.array([r["clinical_padic_distance"] for r in insti_results])

    # Pearson correlations
    if len(reveal_scores) > 2:
        # Reveal score vs resistance
        corr_resistance = np.corrcoef(reveal_scores, resistance_levels)[0, 1]

        # Reveal score vs fitness cost
        corr_fitness = np.corrcoef(reveal_scores, fitness_impacts)[0, 1]

        # Reveal score vs clinical p-adic distance
        corr_clinical = np.corrcoef(reveal_scores, clinical_distances)[0, 1]

        results["correlation_analysis"] = {
            "reveal_vs_resistance": float(corr_resistance),
            "reveal_vs_fitness_cost": float(corr_fitness),
            "reveal_vs_clinical_padic": float(corr_clinical),
            "interpretation": {
                "resistance": ("positive" if corr_resistance > 0 else "negative"),
                "fitness": "positive" if corr_fitness > 0 else "negative",
            },
        }

        print(f"\nReveal Score vs Resistance Level: r = {corr_resistance:.3f}")
        print(f"Reveal Score vs Fitness Cost: r = {corr_fitness:.3f}")
        print(f"Reveal Score vs Clinical p-adic: r = {corr_clinical:.3f}")

    # ==========================================================================
    # KEY FINDINGS
    # ==========================================================================
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Sort INSTI by reveal score
    sorted_insti = sorted(insti_results, key=lambda x: x["reveal_score"], reverse=True)
    print("\nINSTI Mutations by Reveal Score:")
    for r in sorted_insti:
        print(f"  {r['mutation']}: {r['reveal_score']:.2f} (Resistance: {r['resistance_level']})")

    # Sort reveal candidates by score
    sorted_reveal = sorted(
        results["reveal_candidates_analysis"],
        key=lambda x: x["computed_reveal_score"],
        reverse=True,
    )
    print("\nReveal Candidates by Score:")
    for r in sorted_reveal:
        print(f"  {r['mutation']}: {r['computed_reveal_score']:.2f} ({r['mechanism']})")

    # Therapeutic insight
    print("\n" + "=" * 60)
    print("THERAPEUTIC INSIGHT")
    print("=" * 60)

    # Compare INSTI resistance (we DON'T want) vs reveal candidates (we DO want)
    avg_insti_score = np.mean([r["reveal_score"] for r in insti_results])
    avg_reveal_score = np.mean([r["computed_reveal_score"] for r in results["reveal_candidates_analysis"]])

    print(f"\nAverage INSTI Resistance Score: {avg_insti_score:.2f}")
    print(f"Average Reveal Candidate Score: {avg_reveal_score:.2f}")

    if avg_reveal_score > avg_insti_score:
        print("\n✓ Reveal candidates show HIGHER geometric disruption than resistance mutations")
        print("  This supports the 'reveal' hypothesis: LEDGF interface mutations")
        print("  may expose integrase to immune recognition more than drug resistance sites.")
    else:
        print("\n! Reveal candidates show LOWER scores - may need refinement")

    results["summary"] = {
        "avg_insti_resistance_score": float(avg_insti_score),
        "avg_reveal_candidate_score": float(avg_reveal_score),
        "reveal_vs_resistance_ratio": (float(avg_reveal_score / avg_insti_score) if avg_insti_score > 0 else 0),
        "top_reveal_candidate": (sorted_reveal[0]["mutation"] if sorted_reveal else None),
        "highest_resistance_mutation": (sorted_insti[0]["mutation"] if sorted_insti else None),
    }

    return results


def main():
    """Run the experiment and save results."""
    print("=" * 60)
    print("HYBRID INTEGRASE VALIDATION EXPERIMENT")
    print("Validating Ternary VAE against Clinical HIV Data")
    print("=" * 60)

    # Run experiment
    results = run_validation_experiment(use_hybrid=True)

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "hybrid_integrase_validation.json"

    # Custom encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\n\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
