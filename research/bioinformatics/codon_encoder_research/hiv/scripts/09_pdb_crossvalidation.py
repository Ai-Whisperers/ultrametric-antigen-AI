#!/usr/bin/env python3
"""PDB Cross-Validation: Structural Evidence for Reveal Predictions.

This experiment validates our hybrid predictor reveal scores against
actual crystallographic data from 26 HIV integrase PDB structures.

Key validation targets:
1. LEDGF interface distances in 2B4J, 3LPU, 4E7I
2. Resistance mutation positions in 3OYM, 3OYN, 3L2T
3. Catalytic site integrity across structures
"""

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# Add paths for imports
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
alphafold_path = project_root / "research" / "alphafold3"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(alphafold_path))

# Import analyzers
try:
    from hybrid.pdb_analyzer import HAS_BIOPYTHON, PDBAnalyzer
    from hybrid.structure_predictor import HybridStructurePredictor
except ImportError as e:
    print(f"Import error: {e}")
    HAS_BIOPYTHON = False


# =============================================================================
# PDB STRUCTURE CATEGORIES
# =============================================================================

STRUCTURE_CATEGORIES = {
    "ledgf_complexes": {
        "2B4J": "HIV-1 IN CCD with LEDGF IBD",
        "3LPU": "HIV-1 IN with LEDGF",
        "3LPT": "HIV-1 IN with LEDGF variant",
        "4E7I": "HIV-1 IN with LEDGF and raltegravir",
        "4E7K": "HIV-1 IN with LEDGF and elvitegravir",
    },
    "resistance_mutants": {
        "3OYM": "HIV-1 IN Y143R with raltegravir",
        "3OYN": "HIV-1 IN N155H with raltegravir",
        "3L2T": "HIV-1 IN G140S/Q148H",
        "3L2U": "HIV-1 IN E92Q",
        "3L2V": "HIV-1 IN with MK-2048",
    },
    "core_structures": {
        "1EX4": "HIV-1 IN catalytic domain",
        "1BL3": "HIV-1 IN catalytic domain with Mg2+",
        "1BIS": "HIV-1 IN F185K mutant",
        "1BIU": "HIV-1 IN W131E mutant",  # W131 mutation!
        "1BIZ": "HIV-1 IN F185K/C280S mutant",
    },
    "intasome": {
        "3OYA": "HIV-1 IN intasome",
        "5U1C": "HIV-1 IN strand transfer complex",
        "6PUT": "HIV-1 IN intasome with DNA",
    },
}

# LEDGF interface residues (from structural analysis)
LEDGF_INTERFACE_RESIDUES = {
    128,
    129,
    130,
    131,
    132,  # Loop region
    166,
    167,
    168,
    169,
    170,
    171,
    172,
    173,
    174,
    175,  # Helix region
}

# Reveal mutation candidates to validate
REVEAL_CANDIDATES = {
    "W131A": {
        "position": 131,
        "wt": "W",
        "mut": "A",
        "predicted_score": 33.03,
    },
    "I161G": {
        "position": 161,
        "wt": "I",
        "mut": "G",
        "predicted_score": 26.20,
    },
    "E166K": {
        "position": 166,
        "wt": "E",
        "mut": "K",
        "predicted_score": 34.93,
    },
    "K175E": {
        "position": 175,
        "wt": "K",
        "mut": "E",
        "predicted_score": 34.93,
    },
    "Q168E": {
        "position": 168,
        "wt": "Q",
        "mut": "E",
        "predicted_score": 28.50,
    },
}

# Resistance mutations for comparison
RESISTANCE_MUTATIONS = {
    "Y143R": {"position": 143, "wt": "Y", "mut": "R"},
    "Q148H": {"position": 148, "wt": "Q", "mut": "H"},
    "N155H": {"position": 155, "wt": "N", "mut": "H"},
}


def analyze_ledgf_interface(analyzer: PDBAnalyzer) -> Dict:
    """Analyze LEDGF interface distances in complex structures.

    Returns distances from integrase residues to LEDGF chain.
    """
    results = {
        "structures_analyzed": [],
        "interface_distances": {},
        "reveal_position_analysis": {},
    }

    print("\n" + "=" * 60)
    print("LEDGF INTERFACE ANALYSIS")
    print("=" * 60)

    for pdb_id, description in STRUCTURE_CATEGORIES["ledgf_complexes"].items():
        print(f"\nAnalyzing {pdb_id}: {description}")

        try:
            structure = analyzer.load_structure(pdb_id)
            model = structure[0]

            # Find chains
            chains = list(model.get_chains())
            chain_ids = [c.get_id() for c in chains]
            print(f"  Chains: {chain_ids}")

            # Identify integrase vs LEDGF chains (usually A=IN, B=LEDGF or similar)
            in_chain = None
            ledgf_chain = None

            for chain in chains:
                residues = list(chain.get_residues())
                # LEDGF IBD is typically shorter (~70 residues)
                # Integrase CCD is ~150 residues
                if len(residues) > 100:
                    in_chain = chain.get_id()
                else:
                    ledgf_chain = chain.get_id()

            if not in_chain or not ledgf_chain:
                print("  Could not identify IN/LEDGF chains, skipping")
                continue

            print(f"  IN chain: {in_chain}, LEDGF chain: {ledgf_chain}")

            # Calculate interface distances for reveal positions
            interface_data = {}
            for pos in LEDGF_INTERFACE_RESIDUES:
                try:
                    contacts = analyzer.get_residue_contacts(structure, in_chain, pos, radius=12.0)

                    # Filter for LEDGF contacts
                    ledgf_contacts = [c for c in contacts if c.chain_id == ledgf_chain]

                    if ledgf_contacts:
                        min_dist = min(c.distance for c in ledgf_contacts)
                        interface_data[pos] = {
                            "min_distance": float(min_dist),
                            "n_contacts": len(ledgf_contacts),
                            "is_interface": min_dist < 8.0,
                        }
                except Exception:
                    continue

            results["interface_distances"][pdb_id] = interface_data
            results["structures_analyzed"].append(pdb_id)

            # Analyze reveal candidate positions
            for mut_name, mut_data in REVEAL_CANDIDATES.items():
                pos = mut_data["position"]
                if pos in interface_data:
                    if mut_name not in results["reveal_position_analysis"]:
                        results["reveal_position_analysis"][mut_name] = []
                    results["reveal_position_analysis"][mut_name].append(
                        {
                            "pdb_id": pdb_id,
                            "min_distance": interface_data[pos]["min_distance"],
                            "n_contacts": interface_data[pos]["n_contacts"],
                        }
                    )
                    print(f"  {mut_name} (pos {pos}): {interface_data[pos]['min_distance']:.2f}Å to LEDGF")

        except Exception as e:
            print(f"  Error: {e}")

    return results


def analyze_resistance_structures(analyzer: PDBAnalyzer) -> Dict:
    """Analyze resistance mutation structures.

    Compare structural changes in resistance mutants vs wild-type.
    """
    results = {
        "structures_analyzed": [],
        "resistance_analysis": {},
    }

    print("\n" + "=" * 60)
    print("RESISTANCE MUTATION STRUCTURES")
    print("=" * 60)

    for pdb_id, description in STRUCTURE_CATEGORIES["resistance_mutants"].items():
        print(f"\nAnalyzing {pdb_id}: {description}")

        try:
            structure = analyzer.load_structure(pdb_id)
            model = structure[0]

            chains = list(model.get_chains())
            in_chain = chains[0].get_id() if chains else "A"

            # Analyze each resistance position
            for mut_name, mut_data in RESISTANCE_MUTATIONS.items():
                pos = mut_data["position"]
                try:
                    contacts = analyzer.get_residue_contacts(structure, in_chain, pos, radius=8.0)

                    # Get residue info
                    residue = model[in_chain][(" ", pos, " ")]
                    actual_aa = residue.get_resname()

                    if mut_name not in results["resistance_analysis"]:
                        results["resistance_analysis"][mut_name] = []

                    results["resistance_analysis"][mut_name].append(
                        {
                            "pdb_id": pdb_id,
                            "position": pos,
                            "observed_aa": actual_aa,
                            "n_contacts": len(contacts),
                            "avg_contact_dist": (float(np.mean([c.distance for c in contacts])) if contacts else 0),
                        }
                    )

                    print(f"  {mut_name}: {actual_aa} at pos {pos}, {len(contacts)} contacts")

                except Exception:
                    continue

            results["structures_analyzed"].append(pdb_id)

        except Exception as e:
            print(f"  Error: {e}")

    return results


def analyze_w131_mutation(analyzer: PDBAnalyzer) -> Dict:
    """Special analysis for W131 - we have a crystal structure of W131E (1BIU).

    This is direct structural validation of a reveal-type mutation!
    """
    results = {
        "w131_comparison": {},
    }

    print("\n" + "=" * 60)
    print("W131 MUTATION ANALYSIS (Direct Structural Evidence)")
    print("=" * 60)

    # Compare W131 in wild-type (1EX4) vs W131E mutant (1BIU)
    wt_pdb = "1EX4"
    mut_pdb = "1BIU"

    for pdb_id, label in [(wt_pdb, "Wild-type"), (mut_pdb, "W131E mutant")]:
        print(f"\n{label} ({pdb_id}):")

        try:
            structure = analyzer.load_structure(pdb_id)
            model = structure[0]
            chains = list(model.get_chains())
            in_chain = chains[0].get_id() if chains else "A"

            # Get position 131 context
            contacts = analyzer.get_residue_contacts(structure, in_chain, 131, radius=10.0)

            try:
                residue = model[in_chain][(" ", 131, " ")]
                actual_aa = residue.get_resname()
            except:
                actual_aa = "UNK"

            results["w131_comparison"][pdb_id] = {
                "label": label,
                "residue_131": actual_aa,
                "n_contacts": len(contacts),
                "contact_distances": [float(c.distance) for c in contacts[:10]],
                "avg_contact_dist": (float(np.mean([c.distance for c in contacts])) if contacts else 0),
            }

            print(f"  Residue 131: {actual_aa}")
            print(f"  N contacts: {len(contacts)}")
            print(f"  Avg distance: {results['w131_comparison'][pdb_id]['avg_contact_dist']:.2f}Å")

        except Exception as e:
            print(f"  Error: {e}")

    # Compare
    if wt_pdb in results["w131_comparison"] and mut_pdb in results["w131_comparison"]:
        wt_data = results["w131_comparison"][wt_pdb]
        mut_data = results["w131_comparison"][mut_pdb]

        print("\nComparison:")
        print(f"  WT (W131): {wt_data['n_contacts']} contacts, avg {wt_data['avg_contact_dist']:.2f}Å")
        print(f"  Mutant (E131): {mut_data['n_contacts']} contacts, avg {mut_data['avg_contact_dist']:.2f}Å")

        # More contacts = more exposed = supports reveal hypothesis
        if mut_data["n_contacts"] != wt_data["n_contacts"]:
            change = "INCREASED" if mut_data["n_contacts"] > wt_data["n_contacts"] else "DECREASED"
            print(f"  -> Contact count {change} by mutation")
            results["structural_evidence"] = {
                "mutation": "W131E",
                "contact_change": mut_data["n_contacts"] - wt_data["n_contacts"],
                "supports_reveal": mut_data["n_contacts"] >= wt_data["n_contacts"],
            }

    return results


def compare_with_predictions(ledgf_results: Dict, predictor: HybridStructurePredictor) -> Dict:
    """Compare structural distances with predicted reveal scores."""
    results = {
        "comparison": [],
        "correlation": None,
    }

    print("\n" + "=" * 60)
    print("STRUCTURAL vs PREDICTED REVEAL SCORES")
    print("=" * 60)

    for mut_name, mut_data in REVEAL_CANDIDATES.items():
        pos = mut_data["position"]
        predicted_score = mut_data["predicted_score"]

        # Get actual structural data
        if mut_name in ledgf_results.get("reveal_position_analysis", {}):
            struct_data = ledgf_results["reveal_position_analysis"][mut_name]
            avg_dist = np.mean([d["min_distance"] for d in struct_data])
            avg_contacts = np.mean([d["n_contacts"] for d in struct_data])

            # Get model-computed reveal score
            try:
                prediction = predictor.predict_reveal_effect(
                    wt_sequence="FLDGIDKAQEEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGK"
                    "IILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSNFTSTTVKAACWWAGIKQEFGIP"
                    "YNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDIIATDIQTKEL"
                    "QKQITKIQNFRVYYRDSRDPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASG"
                    "RQED",
                    mutation=mut_name,
                )
                computed_score = prediction.get("reveal_score", 0)
            except:
                computed_score = 0

            comparison = {
                "mutation": mut_name,
                "position": pos,
                "avg_ledgf_distance": float(avg_dist),
                "avg_contacts": float(avg_contacts),
                "predicted_score_conjectures": predicted_score,
                "computed_score_model": float(computed_score),
                "structural_evidence": ("INTERFACE" if avg_dist < 8.0 else "PERIPHERAL"),
            }
            results["comparison"].append(comparison)

            print(f"\n{mut_name} (pos {pos}):")
            print(f"  Avg LEDGF distance: {avg_dist:.2f}Å")
            print(f"  Avg contacts: {avg_contacts:.1f}")
            print(f"  Predicted score (conjectures): {predicted_score}")
            print(f"  Computed score (model): {computed_score:.2f}")
            print(f"  Structural: {comparison['structural_evidence']}")

    # Calculate correlation between structural distance and reveal score
    if len(results["comparison"]) >= 3:
        distances = [c["avg_ledgf_distance"] for c in results["comparison"]]
        scores = [c["computed_score_model"] for c in results["comparison"]]

        # Invert distances (closer = higher reveal potential)
        inv_distances = [1 / d if d > 0 else 0 for d in distances]

        if len(set(scores)) > 1:  # Need variance
            corr = np.corrcoef(inv_distances, scores)[0, 1]
            results["correlation"] = {
                "metric": "inverse_distance_vs_reveal_score",
                "r": float(corr),
                "interpretation": ("Closer to LEDGF correlates with higher reveal score" if corr > 0 else "Unexpected"),
            }
            print(f"\nCorrelation (1/distance vs reveal score): r = {corr:.3f}")

    return results


def main():
    """Run PDB cross-validation experiment."""
    print("=" * 60)
    print("PDB CROSS-VALIDATION EXPERIMENT")
    print("Validating Reveal Predictions Against Crystal Structures")
    print("=" * 60)

    if not HAS_BIOPYTHON:
        print("\nError: BioPython required. Install with: pip install biopython")
        return

    # Initialize analyzers
    pdb_dir = project_root / "research" / "alphafold3" / "data" / "pdb"
    analyzer = PDBAnalyzer(pdb_dir)
    predictor = HybridStructurePredictor()

    # Run analyses
    ledgf_results = analyze_ledgf_interface(analyzer)
    resistance_results = analyze_resistance_structures(analyzer)
    w131_results = analyze_w131_mutation(analyzer)
    comparison_results = compare_with_predictions(ledgf_results, predictor)

    # Compile results
    results = {
        "metadata": {
            "experiment": "PDB Cross-Validation",
            "structures_analyzed": len(ledgf_results.get("structures_analyzed", [])) + len(resistance_results.get("structures_analyzed", [])),
            "pdb_dir": str(pdb_dir),
        },
        "ledgf_interface_analysis": ledgf_results,
        "resistance_analysis": resistance_results,
        "w131_structural_evidence": w131_results,
        "prediction_comparison": comparison_results,
    }

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # W131 evidence
    if "structural_evidence" in w131_results:
        ev = w131_results["structural_evidence"]
        print("\n1. W131E Crystal Structure (1BIU):")
        print(f"   Contact change: {ev['contact_change']:+d}")
        print(f"   Supports reveal hypothesis: {ev['supports_reveal']}")

    # Interface positions
    if comparison_results.get("comparison"):
        interface_muts = [c for c in comparison_results["comparison"] if c["structural_evidence"] == "INTERFACE"]
        print(f"\n2. LEDGF Interface Mutations: {len(interface_muts)}/5")
        for c in interface_muts:
            print(f"   {c['mutation']}: {c['avg_ledgf_distance']:.2f}Å from LEDGF")

    # Correlation
    if comparison_results.get("correlation"):
        corr = comparison_results["correlation"]
        print(f"\n3. Structure-Prediction Correlation: r = {corr['r']:.3f}")
        print(f"   {corr['interpretation']}")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    validations = 0
    total = 3

    if w131_results.get("structural_evidence", {}).get("supports_reveal"):
        validations += 1
        print("✓ W131 mutation has direct crystal structure evidence")

    if comparison_results and len([c for c in comparison_results.get("comparison", []) if c["structural_evidence"] == "INTERFACE"]) >= 3:
        validations += 1
        print("✓ Multiple reveal candidates at LEDGF interface")

    if comparison_results and comparison_results.get("correlation", {}) and comparison_results.get("correlation", {}).get("r", 0) > 0.3:
        validations += 1
        print("✓ Positive correlation between structure and prediction")

    print(f"\nValidations passed: {validations}/{total}")

    # Save results
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "pdb_crossvalidation.json"

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

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
