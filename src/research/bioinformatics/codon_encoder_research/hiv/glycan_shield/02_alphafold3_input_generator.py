#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
AlphaFold 3 Input Generator for HIV Glycan Sentinel Analysis

Generate AlphaFold 3 compatible JSON input files for:
1. Wild-type BG505 gp120 (full glycan shield)
2. Deglycosylation mutants (N→Q) at top sentinel candidates
3. Multi-site deglycosylation for cluster analysis

The goal is to validate that sentinel glycans identified by p-adic analysis
show epitope exposure changes upon removal.

AlphaFold 3 JSON format reference:
https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md

Version: 1.0
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# =============================================================================
# CONFIGURATION
# =============================================================================

# BG505 SOSIP.664 gp120 sequence
BG505_GP120 = """
AENLWVTVYYGVPVWKDAETTLFCASDAKAYETEVHNVWATHACVPTDPNPQEIHLENVTEEFNMWKNNMVEQMHT
DIISLWDQSLKPCVKLTPLCVTLQCTNVTNNITDDMRGELKNCSFNMTTELRDKKQKVYSLFYRLDVVQINENQGN
RSNNSNKEYRLINCNTSAITQACPKVSFEPIPIHYCAPAGFAILKCKDKKFNGTGPCPSVSTVQCTHGIKPVVSTQ
LLLNGSLAEEEVMIRSENITNNAKNILVQFNTPVQINCTRPNNNTRKSIRIGPGQAFYATGDIIGDIRQAHCNISR
AKWNNTLQQVAKKLREHFPNKTIKFANSSGGDLEITTHSFNCGGEFFYCNTSGLFNSTWISNTSVQGSNSTGSNDS
ITLPCRIKQIINMWQRIGQAMYAPPIQGVIRCVSNITGLILTRDGGSTNSTTETFRPGGGDMRDNWRSELYKYKVV
KIEPLGVAPTRCKRRVVGRR
""".replace(
    "\n", ""
).replace(
    " ", ""
)

# Output configuration
SCRIPT_NUM = "02"
OUTPUT_SUBDIR = "alphafold3_inputs"


def get_output_dir() -> Path:
    """Get output directory for this script."""
    script_dir = Path(__file__).parent
    results_dir = script_dir / OUTPUT_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def load_glycan_results() -> Dict:
    """Load glycan analysis results from hiv/results folder."""
    script_dir = Path(__file__).parent
    results_path = script_dir.parent / "results" / "glycan_analysis_results.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Run 01_glycan_sentinel_analysis.py first. Expected: {results_path}")

    with open(results_path, "r") as f:
        return json.load(f)


def mutate_sequence(sequence: str, positions: List[int], from_aa: str = "N", to_aa: str = "Q") -> str:
    """Mutate specific positions in sequence."""
    seq_list = list(sequence)
    for pos in positions:
        if 0 <= pos < len(seq_list) and seq_list[pos] == from_aa:
            seq_list[pos] = to_aa
    return "".join(seq_list)


# =============================================================================
# JSON GENERATION
# =============================================================================


def find_glycan_sequons(sequence: str) -> List[Dict]:
    """Find all N-X-S/T sequons and return as glycan specifications."""
    glycans = []
    for i in range(len(sequence) - 2):
        if sequence[i] == "N" and sequence[i + 1] != "P":
            if sequence[i + 2] in ["S", "T"]:
                # AlphaFold server uses 1-indexed positions
                glycans.append(
                    {
                        "residues": "NAG",  # N-Acetylglucosamine (common N-glycan core)
                        "position": i + 1,  # 1-indexed
                    }
                )
    return glycans


def generate_wt_env_json(include_glycans: bool = True) -> Dict:
    """Generate AlphaFold Server JSON for wild-type BG505 gp120."""
    protein_chain = {"sequence": BG505_GP120, "count": 1}

    # Optionally include glycan annotations
    if include_glycans:
        glycans = find_glycan_sequons(BG505_GP120)
        if glycans:
            protein_chain["glycans"] = glycans

    return {
        "name": "BG505_gp120_WT",
        "sequences": [{"proteinChain": protein_chain}],
        "modelSeeds": [],  # Let server assign seeds
        "dialect": "alphafoldserver",
        "version": 1,
    }


def generate_single_deglyc_json(glycan_name: str, position: int, all_glycan_positions: List[int]) -> Dict:
    """
    Generate AlphaFold Server JSON for single deglycosylation mutant.

    Strategy: Keep all glycans EXCEPT the one we're removing.
    This models the effect of removing one specific glycan.
    """
    # Get all glycan positions except the one being removed
    remaining_glycans = [{"residues": "NAG", "position": pos + 1} for pos in all_glycan_positions if pos != position]  # 1-indexed

    # Mutate the sequence at the target position (N→Q removes glycan attachment)
    mut_sequence = mutate_sequence(BG505_GP120, [position])

    protein_chain = {"sequence": mut_sequence, "count": 1}

    # Add remaining glycans
    if remaining_glycans:
        protein_chain["glycans"] = remaining_glycans

    return {
        "name": f"BG505_deglyc_{glycan_name}",
        "sequences": [{"proteinChain": protein_chain}],
        "modelSeeds": [],
        "dialect": "alphafoldserver",
        "version": 1,
    }


def generate_multi_deglyc_json(
    glycan_names: List[str],
    positions: List[int],
    name_suffix: str,
    all_glycan_positions: List[int],
) -> Dict:
    """Generate AlphaFold Server JSON for multi-site deglycosylation."""
    # Remove specified glycans
    mut_sequence = mutate_sequence(BG505_GP120, positions)

    # Keep glycans at positions not being removed
    positions_set = set(positions)
    remaining_glycans = [{"residues": "NAG", "position": pos + 1} for pos in all_glycan_positions if pos not in positions_set]

    protein_chain = {"sequence": mut_sequence, "count": 1}

    if remaining_glycans:
        protein_chain["glycans"] = remaining_glycans

    return {
        "name": f"BG505_deglyc_{name_suffix}",
        "sequences": [{"proteinChain": protein_chain}],
        "modelSeeds": [],
        "dialect": "alphafoldserver",
        "version": 1,
    }


def generate_comparison_json(glycan_name: str, position: int, all_glycan_positions: List[int]) -> Dict:
    """
    Generate comparison JSON with both WT and mutant as separate chains.

    Uses fragment approach to reduce compute while showing local structural changes.
    """
    # For gp120, use a truncated version - focus on ±50 residues around glycan
    window = 50
    start = max(0, position - window)
    end = min(len(BG505_GP120), position + window)

    wt_fragment = BG505_GP120[start:end]
    mut_fragment = mutate_sequence(BG505_GP120, [position])[start:end]

    # Find glycans within this fragment
    fragment_glycans_wt = []
    fragment_glycans_mut = []
    for gp in all_glycan_positions:
        if start <= gp < end:
            # Position relative to fragment (1-indexed)
            rel_pos = gp - start + 1
            fragment_glycans_wt.append({"residues": "NAG", "position": rel_pos})
            # For mutant, exclude the removed glycan
            if gp != position:
                fragment_glycans_mut.append({"residues": "NAG", "position": rel_pos})

    wt_chain = {"sequence": wt_fragment, "count": 1}
    mut_chain = {"sequence": mut_fragment, "count": 1}

    if fragment_glycans_wt:
        wt_chain["glycans"] = fragment_glycans_wt
    if fragment_glycans_mut:
        mut_chain["glycans"] = fragment_glycans_mut

    return {
        "name": f"BG505_cmp_{glycan_name}",
        "sequences": [{"proteinChain": wt_chain}, {"proteinChain": mut_chain}],
        "modelSeeds": [],
        "dialect": "alphafoldserver",
        "version": 1,
    }


# =============================================================================
# BATCH GENERATION
# =============================================================================


def generate_all_inputs(glycan_results: Dict, output_dir: Path, max_candidates: int = 10) -> Dict:
    """Generate all AlphaFold Server input JSON files."""

    generated_files = {
        "wild_type": [],
        "single_deglyc": [],
        "multi_deglyc": [],
        "comparison": [],
    }

    # Create subdirectories
    for subdir in generated_files.keys():
        (output_dir / subdir).mkdir(exist_ok=True)

    # Get all glycan positions for proper glycan handling
    results = glycan_results["results"]
    all_glycan_positions = [r["position"] for r in results]

    goldilocks_sites = [r for r in results if r["goldilocks_zone"] == "goldilocks"]
    top_candidates = results[:max_candidates]

    print(f"\n  Generating JSONs for {len(top_candidates)} candidates...")
    print(f"  Total glycan sites in sequence: {len(all_glycan_positions)}")

    # 1. Wild-type structure (with all glycans)
    print("\n  [1] Wild-type BG505 gp120 (with glycans)...")
    wt_json = generate_wt_env_json(include_glycans=True)
    wt_path = output_dir / "wild_type" / "BG505_gp120_WT.json"
    with open(wt_path, "w") as f:
        json.dump(wt_json, f, indent=2)
    generated_files["wild_type"].append(str(wt_path))
    print(f"      Saved: {wt_path.name} ({len(all_glycan_positions)} glycans)")

    # 2. Single deglycosylation mutants
    print("\n  [2] Single deglycosylation mutants...")
    for r in top_candidates:
        glycan_name = r["name"]
        position = r["position"]

        single_json = generate_single_deglyc_json(glycan_name, position, all_glycan_positions)
        single_path = output_dir / "single_deglyc" / f"BG505_deglyc_{glycan_name}.json"
        with open(single_path, "w") as f:
            json.dump(single_json, f, indent=2)
        generated_files["single_deglyc"].append(str(single_path))

        zone = "GOLDILOCKS" if r["goldilocks_zone"] == "goldilocks" else r["goldilocks_zone"]
        print(f"      {glycan_name}: {r['centroid_shift']*100:.1f}% shift ({zone})")

    # 3. Multi-site deglycosylation (Goldilocks sites only)
    print("\n  [3] Multi-site deglycosylation (Goldilocks cluster)...")
    if goldilocks_sites:
        goldilocks_positions = [r["position"] for r in goldilocks_sites]
        goldilocks_names = [r["name"] for r in goldilocks_sites]

        multi_json = generate_multi_deglyc_json(
            goldilocks_names,
            goldilocks_positions,
            "all_goldilocks",
            all_glycan_positions,
        )
        multi_path = output_dir / "multi_deglyc" / "BG505_deglyc_all_goldilocks.json"
        with open(multi_path, "w") as f:
            json.dump(multi_json, f, indent=2)
        generated_files["multi_deglyc"].append(str(multi_path))
        print(f"      All Goldilocks ({len(goldilocks_sites)} sites): {', '.join(goldilocks_names)}")

    # 4. Regional multi-deglycosylation
    print("\n  [4] Regional deglycosylation clusters...")
    regions = {}
    for r in top_candidates:
        region = r["region"]
        if region not in regions:
            regions[region] = []
        regions[region].append(r)

    for region, sites in regions.items():
        if len(sites) >= 2:
            positions = [s["position"] for s in sites]
            names = [s["name"] for s in sites]

            region_json = generate_multi_deglyc_json(names, positions, f"region_{region}", all_glycan_positions)
            region_path = output_dir / "multi_deglyc" / f"BG505_deglyc_region_{region}.json"
            with open(region_path, "w") as f:
                json.dump(region_json, f, indent=2)
            generated_files["multi_deglyc"].append(str(region_path))
            print(f"      {region} ({len(sites)} sites): {', '.join(names)}")

    # 5. Comparison JSONs (top 5 only to save compute)
    print("\n  [5] Comparison pairs (fragment-based)...")
    for r in top_candidates[:5]:
        comp_json = generate_comparison_json(r["name"], r["position"], all_glycan_positions)
        comp_path = output_dir / "comparison" / f"BG505_cmp_{r['name']}.json"
        with open(comp_path, "w") as f:
            json.dump(comp_json, f, indent=2)
        generated_files["comparison"].append(str(comp_path))
        print(f"      {r['name']}: ±50 residue window")

    return generated_files


def generate_manifest(glycan_results: Dict, generated_files: Dict, output_dir: Path) -> Path:
    """Generate manifest file for batch processing."""

    goldilocks_sites = [r for r in glycan_results["results"] if r["goldilocks_zone"] == "goldilocks"]

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "description": "AlphaFold 3 inputs for HIV glycan sentinel validation",
        "research_context": {
            "hypothesis": "Deglycosylation at sentinel glycans exposes bnAb epitopes",
            "inverse_goldilocks": "PTM removal (not addition) shifts into immunogenic zone",
            "goldilocks_zone": "15-30% centroid shift predicts optimal epitope exposure",
            "validation_goal": "Structural confirmation of epitope accessibility changes",
        },
        "sequence": {
            "name": "BG505 SOSIP.664 gp120",
            "length": len(BG505_GP120),
            "total_glycans": glycan_results["metadata"]["total_sites"],
            "goldilocks_glycans": len(goldilocks_sites),
        },
        "top_sentinel_candidates": [
            {
                "name": r["name"],
                "position": r["position"] + 1,  # 1-indexed for clarity
                "region": r["region"],
                "centroid_shift": f"{r['centroid_shift']*100:.1f}%",
                "goldilocks_zone": r["goldilocks_zone"],
                "score": r["goldilocks_score"],
            }
            for r in glycan_results["results"][:10]
        ],
        "files": generated_files,
        "total_files": sum(len(v) for v in generated_files.values()),
        "recommended_jobs": {
            "priority_1": "wild_type/BG505_gp120_WT.json",
            "priority_2": [
                f
                for f in generated_files["single_deglyc"]
                if "goldilocks" in str(f).lower() or any(gs["name"] in str(f) for gs in goldilocks_sites[:3])
            ],
            "priority_3": "multi_deglyc/BG505_deglyc_all_goldilocks.json",
        },
        "expected_analyses": [
            "Compare WT vs single deglyc RMSD at glycan site",
            "Measure solvent-accessible surface area changes",
            "Identify newly exposed epitope regions",
            "Map to known bnAb footprints (VRC01, PGT121, etc.)",
            "Validate Goldilocks prediction accuracy",
        ],
    }

    manifest_path = output_dir / "batch_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("ALPHAFOLD 3 INPUT GENERATOR")
    print("HIV Glycan Sentinel Structural Validation")
    print("=" * 70)

    output_dir = get_output_dir()
    print(f"\nOutput directory: {output_dir}")

    # Load glycan analysis results
    print("\n[1] Loading glycan analysis results...")
    try:
        glycan_results = load_glycan_results()
        print(f"  Loaded {glycan_results['metadata']['total_sites']} glycan sites")
        print(f"  Goldilocks sites: {glycan_results['metadata']['goldilocks_count']}")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return

    # Generate all inputs
    print("\n[2] Generating AlphaFold 3 input JSONs...")
    generated_files = generate_all_inputs(glycan_results, output_dir, max_candidates=10)

    # Generate manifest
    print("\n[3] Generating batch manifest...")
    manifest_path = generate_manifest(glycan_results, generated_files, output_dir)
    print(f"  Saved: {manifest_path}")

    # Summary
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    total = sum(len(v) for v in generated_files.values())
    print(f"\n  Total files generated: {total}")
    print(f"    Wild-type: {len(generated_files['wild_type'])}")
    print(f"    Single deglycosylation: {len(generated_files['single_deglyc'])}")
    print(f"    Multi-site deglycosylation: {len(generated_files['multi_deglyc'])}")
    print(f"    Comparison pairs: {len(generated_files['comparison'])}")

    print(f"\n  Output directory: {output_dir}")

    # Recommended jobs
    goldilocks_sites = [r for r in glycan_results["results"] if r["goldilocks_zone"] == "goldilocks"]
    print("\n  RECOMMENDED ALPHAFOLD3 JOBS (within 20 job limit):")
    print("    1. BG505_gp120_WT.json (baseline)")

    job_count = 1
    for site in goldilocks_sites[:5]:
        job_count += 1
        print(f"    {job_count}. BG505_deglyc_{site['name']}.json (Goldilocks sentinel)")

    if goldilocks_sites:
        job_count += 1
        print(f"    {job_count}. BG505_deglyc_all_goldilocks.json (all sentinels)")

    for site in goldilocks_sites[:3]:
        job_count += 1
        print(f"    {job_count}. BG505_comparison_{site['name']}.json (structural comparison)")

    print(f"\n  Total recommended jobs: {job_count}")

    return generated_files


if __name__ == "__main__":
    main()
