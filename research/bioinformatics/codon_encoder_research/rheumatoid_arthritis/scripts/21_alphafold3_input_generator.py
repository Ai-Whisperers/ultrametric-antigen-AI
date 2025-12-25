#!/usr/bin/env python3
"""
AlphaFold 3 Input JSON Generator for Citrullination Studies

Generate AlphaFold 3 compatible JSON input files for:
1. Native peptides (with arginine)
2. Citrullinated peptides (R -> Cit modification)
3. Peptide-HLA complexes (for antigen presentation modeling)

AlphaFold 3 JSON format reference:
https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md

Output directory: results/alphafold3/21_input_jsons/

Version: 1.0
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# ============================================================================
# CONFIGURATION
# ============================================================================

# CCD code for citrulline (modified arginine)
CITRULLINE_CCD = "CIR"

# Common HLA alleles for RA (for peptide-HLA complexes)
# These are risk alleles associated with RA
HLA_ALLELES = {
    "HLA-DRB1*04:01": "GDTRPRFLWQLKFECHFFNGTERVRLLERCIYNQEESVRFDSDVGEYRAVTELGRPDAEYWNSQKDLLEQRRAAVDTYCRHNYGVGESFTVQRR",
    "HLA-DRB1*01:01": "GDTRPRFLWQLKFECHFFNGTERVRLLERCIYNQEESVRFDSDVGEYRAVTELGRPDAEYWNSQKDLLEQKRAAVDTYCRHNYGVGESFTVQRR",
}

# Output configuration
SCRIPT_NUM = "21"
OUTPUT_SUBDIR = f"{SCRIPT_NUM}_alphafold3_inputs"


# ============================================================================
# DIRECTORY SETUP
# ============================================================================


def get_output_dir() -> Path:
    """Get output directory for this script."""
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / "alphafold3" / OUTPUT_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# ============================================================================
# EPITOPE DATA
# ============================================================================


def load_epitope_data() -> List[Dict]:
    """Load epitope data with entropy change values."""
    script_dir = Path(__file__).parent
    results_path = script_dir.parent / "results" / "hyperbolic" / "goldilocks_validation" / "goldilocks_validation_results.json"

    if not results_path.exists():
        print(f"  Warning: Goldilocks results not found at {results_path}")
        return get_default_epitopes()

    with open(results_path, "r") as f:
        data = json.load(f)

    return data.get("epitope_details", [])


def get_default_epitopes() -> List[Dict]:
    """Default key epitopes if results not found."""
    return [
        {
            "epitope_id": "VIM_R71",
            "sequence": "RLRSSVPGVR",
            "immunodominant": True,
            "mean_entropy_change": 0.049,
        },
        {
            "epitope_id": "FGA_R38",
            "sequence": "GPRVVERHQS",
            "immunodominant": True,
            "mean_entropy_change": 0.041,
        },
        {
            "epitope_id": "ENO1_CEP1",
            "sequence": "KIREEIFDSRGNP",
            "immunodominant": True,
            "mean_entropy_change": 0.026,
        },
        {
            "epitope_id": "FLG_CCP",
            "sequence": "SHQESTRGRS",
            "immunodominant": True,
            "mean_entropy_change": 0.028,
        },
        {
            "epitope_id": "FGB_R406",
            "sequence": "SARGHRPLDKK",
            "immunodominant": True,
            "mean_entropy_change": 0.038,
        },
    ]


# ============================================================================
# JSON GENERATION
# ============================================================================


def generate_native_peptide_json(epitope: Dict, seed: int = 1) -> Dict:
    """
    Generate AlphaFold 3 JSON for native (unmodified) peptide.

    Format: alphafold3 dialect version 1
    """
    epitope_id = epitope["epitope_id"]
    sequence = epitope["sequence"]

    return {
        "name": f"{epitope_id}_native",
        "sequences": [{"protein": {"id": "A", "sequence": sequence}}],
        "modelSeeds": [seed],
        "dialect": "alphafold3",
        "version": 1,
    }


def generate_citrullinated_peptide_json(epitope: Dict, seed: int = 1) -> Dict:
    """
    Generate AlphaFold 3 JSON for citrullinated peptide.

    Replaces arginine (R) with citrulline using CCD code CIR.
    Uses modifications array to specify citrulline positions.
    """
    epitope_id = epitope["epitope_id"]
    sequence = epitope["sequence"]

    # Find all arginine positions (1-indexed for AlphaFold)
    r_positions = [i + 1 for i, aa in enumerate(sequence) if aa == "R"]

    # Replace R with placeholder X in sequence (will be defined as CIR)
    # Actually, AlphaFold 3 uses modifications array

    # Build modifications list for each R position
    modifications = []
    for pos in r_positions:
        modifications.append({"ptmType": CITRULLINE_CCD, "ptmPosition": pos})

    return {
        "name": f"{epitope_id}_citrullinated",
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": sequence,
                    "modifications": modifications,
                }
            }
        ],
        "modelSeeds": [seed],
        "dialect": "alphafold3",
        "version": 1,
    }


def generate_peptide_hla_complex_json(
    epitope: Dict,
    hla_allele: str,
    hla_sequence: str,
    citrullinated: bool = False,
    seed: int = 1,
) -> Dict:
    """
    Generate AlphaFold 3 JSON for peptide-HLA complex.

    Models how the peptide binds to HLA for antigen presentation.
    """
    epitope_id = epitope["epitope_id"]
    sequence = epitope["sequence"]
    state = "cit" if citrullinated else "native"

    # Peptide chain
    peptide_chain = {"protein": {"id": "A", "sequence": sequence}}

    # Add citrullination modifications if needed
    if citrullinated:
        r_positions = [i + 1 for i, aa in enumerate(sequence) if aa == "R"]
        modifications = [{"ptmType": CITRULLINE_CCD, "ptmPosition": pos} for pos in r_positions]
        peptide_chain["protein"]["modifications"] = modifications

    # HLA chain
    hla_chain = {"protein": {"id": "B", "sequence": hla_sequence}}

    return {
        "name": f"{epitope_id}_{state}_{hla_allele.replace('*', '_').replace(':', '_')}",
        "sequences": [peptide_chain, hla_chain],
        "modelSeeds": [seed],
        "dialect": "alphafold3",
        "version": 1,
    }


def generate_comparison_pair_json(epitope: Dict, seed: int = 1) -> Dict:
    """
    Generate a single JSON with both native and citrullinated versions
    as separate chains for direct structural comparison.
    """
    epitope_id = epitope["epitope_id"]
    sequence = epitope["sequence"]

    # Find R positions for citrullination
    r_positions = [i + 1 for i, aa in enumerate(sequence) if aa == "R"]
    modifications = [{"ptmType": CITRULLINE_CCD, "ptmPosition": pos} for pos in r_positions]

    return {
        "name": f"{epitope_id}_comparison",
        "sequences": [
            {"protein": {"id": "A", "sequence": sequence}},  # Native
            {
                "protein": {
                    "id": "B",  # Citrullinated
                    "sequence": sequence,
                    "modifications": modifications,
                }
            },
        ],
        "modelSeeds": [seed],
        "dialect": "alphafold3",
        "version": 1,
    }


# ============================================================================
# BATCH GENERATION
# ============================================================================


def generate_all_inputs(epitopes: List[Dict], output_dir: Path) -> Dict:
    """
    Generate all AlphaFold 3 input JSON files.

    Creates:
    1. Individual native peptide JSONs
    2. Individual citrullinated peptide JSONs
    3. Comparison pair JSONs
    4. Peptide-HLA complex JSONs
    5. Batch file listing all inputs
    """
    generated_files = {
        "native": [],
        "citrullinated": [],
        "comparison": [],
        "hla_complexes": [],
    }

    # Create subdirectories
    (output_dir / "native").mkdir(exist_ok=True)
    (output_dir / "citrullinated").mkdir(exist_ok=True)
    (output_dir / "comparison").mkdir(exist_ok=True)
    (output_dir / "hla_complexes").mkdir(exist_ok=True)

    # Sort epitopes by entropy change (most immunodominant first)
    sorted_epitopes = sorted(epitopes, key=lambda x: x.get("mean_entropy_change", 0), reverse=True)

    # Take top 10 most immunodominant for processing
    top_epitopes = sorted_epitopes[:10]

    print(f"\n  Generating JSONs for top {len(top_epitopes)} epitopes...")

    for epitope in top_epitopes:
        epitope_id = epitope["epitope_id"]
        print(f"    {epitope_id}: {epitope['sequence']}")

        # 1. Native peptide
        native_json = generate_native_peptide_json(epitope)
        native_path = output_dir / "native" / f"{epitope_id}_native.json"
        with open(native_path, "w") as f:
            json.dump(native_json, f, indent=2)
        generated_files["native"].append(str(native_path))

        # 2. Citrullinated peptide
        cit_json = generate_citrullinated_peptide_json(epitope)
        cit_path = output_dir / "citrullinated" / f"{epitope_id}_citrullinated.json"
        with open(cit_path, "w") as f:
            json.dump(cit_json, f, indent=2)
        generated_files["citrullinated"].append(str(cit_path))

        # 3. Comparison pair
        comp_json = generate_comparison_pair_json(epitope)
        comp_path = output_dir / "comparison" / f"{epitope_id}_comparison.json"
        with open(comp_path, "w") as f:
            json.dump(comp_json, f, indent=2)
        generated_files["comparison"].append(str(comp_path))

        # 4. HLA complexes (for top 5 only to limit compute)
        if epitope in top_epitopes[:5]:
            for hla_name, hla_seq in HLA_ALLELES.items():
                # Native + HLA
                native_hla = generate_peptide_hla_complex_json(epitope, hla_name, hla_seq, citrullinated=False)
                native_hla_path = output_dir / "hla_complexes" / f"{epitope_id}_native_{hla_name.replace('*', '_').replace(':', '_')}.json"
                with open(native_hla_path, "w") as f:
                    json.dump(native_hla, f, indent=2)
                generated_files["hla_complexes"].append(str(native_hla_path))

                # Citrullinated + HLA
                cit_hla = generate_peptide_hla_complex_json(epitope, hla_name, hla_seq, citrullinated=True)
                cit_hla_path = output_dir / "hla_complexes" / f"{epitope_id}_cit_{hla_name.replace('*', '_').replace(':', '_')}.json"
                with open(cit_hla_path, "w") as f:
                    json.dump(cit_hla, f, indent=2)
                generated_files["hla_complexes"].append(str(cit_hla_path))

    return generated_files


def generate_batch_manifest(epitopes: List[Dict], generated_files: Dict, output_dir: Path):
    """
    Generate manifest file describing all inputs for batch processing.
    """
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "description": "AlphaFold 3 input files for citrullination study",
        "research_context": {
            "hypothesis": "Citrullination causes structural changes that correlate with immunogenicity",
            "goldilocks_zone": "Entropy change in [-0.12, +0.05] distinguishes immunodominant sites",
            "key_metric": "Compare native vs citrullinated structures",
        },
        "epitopes_included": [
            {
                "id": e["epitope_id"],
                "sequence": e["sequence"],
                "immunodominant": e.get("immunodominant", True),
                "entropy_change": e.get("mean_entropy_change", 0),
                "n_arginines": e["sequence"].count("R"),
            }
            for e in sorted(
                epitopes,
                key=lambda x: x.get("mean_entropy_change", 0),
                reverse=True,
            )[:10]
        ],
        "files": generated_files,
        "total_files": sum(len(v) for v in generated_files.values()),
        "usage_instructions": {
            "alphafold3_command": "python run_alphafold.py --json_path <input.json> --output_dir <output>",
            "recommended_order": [
                "1. Run comparison JSONs first (native + cit in same prediction)",
                "2. Run individual native/citrullinated for higher accuracy",
                "3. Run HLA complexes for antigen presentation modeling",
            ],
        },
        "expected_analyses": [
            "RMSD between native and citrullinated structures",
            "Contact map differences",
            "Surface accessibility changes at R/Cit positions",
            "HLA binding groove occupancy differences",
        ],
    }

    manifest_path = output_dir / "batch_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 80)
    print("ALPHAFOLD 3 INPUT GENERATOR")
    print("Generating JSON inputs for citrullination structure prediction")
    print("=" * 80)

    output_dir = get_output_dir()
    print(f"\nOutput directory: {output_dir}")

    # Load epitope data
    print("\n[1] Loading epitope data...")
    epitopes = load_epitope_data()
    print(f"  Loaded {len(epitopes)} epitopes")

    # Generate all inputs
    print("\n[2] Generating AlphaFold 3 input JSONs...")
    generated_files = generate_all_inputs(epitopes, output_dir)

    # Generate manifest
    print("\n[3] Generating batch manifest...")
    manifest_path = generate_batch_manifest(epitopes, generated_files, output_dir)
    print(f"  Saved: {manifest_path}")

    # Summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)

    total = sum(len(v) for v in generated_files.values())
    print(f"\n  Total files generated: {total}")
    print(f"    Native peptides: {len(generated_files['native'])}")
    print(f"    Citrullinated peptides: {len(generated_files['citrullinated'])}")
    print(f"    Comparison pairs: {len(generated_files['comparison'])}")
    print(f"    HLA complexes: {len(generated_files['hla_complexes'])}")

    print(f"\n  Output directory: {output_dir}")

    print("\n  Next steps:")
    print("    1. Upload JSONs to AlphaFold 3 server or run locally")
    print("    2. Compare native vs citrullinated structures")
    print("    3. Analyze structural changes at citrullination sites")
    print("    4. Correlate with entropy change predictions")

    # List files
    print("\n  Generated files:")
    for category, files in generated_files.items():
        print(f"\n    {category}/")
        for f in files[:3]:
            print(f"      {Path(f).name}")
        if len(files) > 3:
            print(f"      ... and {len(files) - 3} more")

    return generated_files


if __name__ == "__main__":
    main()
