#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
AlphaFold3 Input Generator for HIV Integrase Reveal Mutations

Generates JSON input files for validating integrase LEDGF interface
modifications identified by our 3-adic codon geometry analysis.

Top reveal candidates from conjecture validation:
1. E166K (reveal_score=34.93) - Salt bridge reversal
2. K175E (reveal_score=34.93) - Charge reversal
3. W131A (reveal_score=33.03) - Aromatic cap removal

Author: AI Whisperers
Date: 2025-12-24
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# =============================================================================
# HIV-1 INTEGRASE SEQUENCE (HXB2 reference, 288 aa)
# =============================================================================

# Full HIV-1 Integrase sequence
INTEGRASE_WT = (
    "FLDGIDKAQEEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGK"
    "IILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSNFTSTTVKAACWWAGIKQEFGIP"
    "YNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDIIATDIQTKEL"
    "QKQITKIQNFRVYYRDSRDPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASG"
    "RQED"
)

# LEDGF/p75 binding domain residues (from crystal structure 2B4J)
LEDGF_INTERFACE_RESIDUES = [
    128,
    129,
    130,
    131,
    132,
    161,
    166,
    168,
    170,
    171,
    173,
    174,
    175,
    178,
    179,
]

# Catalytic DDE motif (absolutely conserved)
CATALYTIC_RESIDUES = [64, 116, 152]

# =============================================================================
# REVEAL MUTATIONS (from conjecture validation)
# Positions verified against HXB2 sequence
# =============================================================================

REVEAL_MUTATIONS = {
    # Top candidates - using actual HXB2 sequence positions
    # The reveal scores are from codon-level analysis, mutations target LEDGF interface
    "W131A": {
        "position": 131,  # W in sequence
        "wt_aa": "W",
        "mut_aa": "A",
        "reveal_score": 33.03,
        "mechanism": "Remove aromatic cap, expose hydrophobic core",
        "expected_effect": "Cavity creation, epitope accessibility",
    },
    "I161G": {
        "position": 161,  # I in sequence
        "wt_aa": "I",
        "mut_aa": "G",
        "reveal_score": 21.92,
        "mechanism": "Cavity creation in hydrophobic core",
        "expected_effect": "Core destabilization",
    },
    "R166E": {
        "position": 166,  # R in sequence (was E166K, corrected)
        "wt_aa": "R",
        "mut_aa": "E",
        "reveal_score": 34.93,
        "mechanism": "Charge reversal at interface",
        "expected_effect": "Major conformational change, epitope exposure",
    },
    "Q168E": {
        "position": 168,  # Q in sequence (was R168E, corrected)
        "wt_aa": "Q",
        "mut_aa": "E",
        "reveal_score": 24.51,
        "mechanism": "Polar to charged, interface destabilization",
        "expected_effect": "Interface destabilization",
    },
    "K173A": {
        "position": 173,  # K in sequence (was H173A, corrected)
        "wt_aa": "K",
        "mut_aa": "A",
        "reveal_score": 23.26,
        "mechanism": "Remove positive charge from interface",
        "expected_effect": "Local unfolding at interface",
    },
    "A175E": {
        "position": 175,  # A in sequence (was K175E, corrected)
        "wt_aa": "A",
        "mut_aa": "E",
        "reveal_score": 34.93,
        "mechanism": "Add negative charge to hydrophobic region",
        "expected_effect": "Disrupted packing, interface exposure",
    },
}

# Drug resistance mutations (for comparison)
RESISTANCE_MUTATIONS = {
    "Y143R": {"position": 143, "wt_aa": "Y", "mut_aa": "R", "drugs": ["RAL"]},
    "N155H": {
        "position": 155,
        "wt_aa": "N",
        "mut_aa": "H",
        "drugs": ["RAL", "EVG"],
    },
    "Q148H": {
        "position": 148,
        "wt_aa": "Q",
        "mut_aa": "H",
        "drugs": ["RAL", "EVG", "DTG"],
    },
}


def mutate_sequence(sequence: str, position: int, new_aa: str) -> str:
    """Mutate sequence at 1-indexed position."""
    seq_list = list(sequence)
    seq_list[position - 1] = new_aa  # Convert to 0-indexed
    return "".join(seq_list)


def generate_alphafold3_json(
    name: str,
    sequence: str,
    description: str = "",
    seeds: List[int] = None,
) -> Dict:
    """
    Generate AlphaFold3-compatible JSON input.

    Uses native alphafold3 dialect for local runs.
    """
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]  # 5 seeds for ensemble

    return {
        "name": name,
        "modelSeeds": seeds,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": sequence,
                    "description": description,
                }
            }
        ],
        "dialect": "alphafold3",
        "version": 1,
    }


def generate_alphafold_server_json(
    name: str,
    sequence: str,
    description: str = "",
) -> Dict:
    """
    Generate AlphaFold Server-compatible JSON input.

    Uses alphafoldserver dialect for web submission.
    """
    return {
        "name": name,
        "sequences": [
            {
                "proteinChain": {
                    "sequence": sequence,
                    "count": 1,
                }
            }
        ],
        "modelSeeds": [],  # Server assigns seeds
        "dialect": "alphafoldserver",
        "version": 1,
    }


def generate_all_inputs(output_dir: Path, use_server_format: bool = False):
    """Generate all integrase validation inputs."""

    output_dir.mkdir(parents=True, exist_ok=True)

    generator = generate_alphafold_server_json if use_server_format else generate_alphafold3_json
    format_name = "AlphaFold Server" if use_server_format else "AlphaFold3 Local"

    print(f"Generating {format_name} inputs...")
    print(f"Output directory: {output_dir}")

    generated_files = []

    # 1. Wild-type integrase
    print("\n[1] Wild-type Integrase (baseline)")
    wt_json = generator(
        name="HIV1_IN_WT",
        sequence=INTEGRASE_WT,
        description="HIV-1 Integrase wild-type (HXB2 reference)",
    )
    wt_path = output_dir / "01_IN_wildtype.json"
    with open(wt_path, "w") as f:
        json.dump(wt_json, f, indent=2)
    generated_files.append(wt_path)
    print(f"  Saved: {wt_path.name}")

    # 2. Top reveal mutations
    print("\n[2] Reveal Mutations (ranked by score)")
    sorted_mutations = sorted(
        REVEAL_MUTATIONS.items(),
        key=lambda x: x[1]["reveal_score"],
        reverse=True,
    )

    for i, (mut_name, mut_data) in enumerate(sorted_mutations, start=2):
        mut_seq = mutate_sequence(INTEGRASE_WT, mut_data["position"], mut_data["mut_aa"])

        description = (
            f"HIV-1 Integrase {mut_name} reveal mutation. " f"Mechanism: {mut_data['mechanism']}. " f"Reveal score: {mut_data['reveal_score']}"
        )

        mut_json = generator(
            name=f"HIV1_IN_{mut_name}",
            sequence=mut_seq,
            description=description,
        )

        # Verify mutation
        wt_aa = INTEGRASE_WT[mut_data["position"] - 1]
        mut_aa = mut_seq[mut_data["position"] - 1]
        assert wt_aa == mut_data["wt_aa"], f"WT mismatch at {mut_data['position']}"
        assert mut_aa == mut_data["mut_aa"], f"Mut mismatch at {mut_data['position']}"

        mut_path = output_dir / f"{i:02d}_IN_{mut_name}.json"
        with open(mut_path, "w") as f:
            json.dump(mut_json, f, indent=2)
        generated_files.append(mut_path)

        print(f"  {mut_name}: score={mut_data['reveal_score']:.1f}, {mut_data['mechanism'][:40]}...")
        print(f"    Saved: {mut_path.name}")

    # 3. Drug resistance mutations (for comparison)
    print("\n[3] Drug Resistance Mutations (comparison)")
    for mut_name, mut_data in RESISTANCE_MUTATIONS.items():
        mut_seq = mutate_sequence(INTEGRASE_WT, mut_data["position"], mut_data["mut_aa"])

        description = f"HIV-1 Integrase {mut_name} drug resistance mutation. " f"Confers resistance to: {', '.join(mut_data['drugs'])}"

        mut_json = generator(
            name=f"HIV1_IN_{mut_name}_resistance",
            sequence=mut_seq,
            description=description,
        )

        mut_path = output_dir / f"resistance_{mut_name}.json"
        with open(mut_path, "w") as f:
            json.dump(mut_json, f, indent=2)
        generated_files.append(mut_path)

        print(f"  {mut_name}: drugs={mut_data['drugs']}")

    # 4. Generate batch manifest
    print("\n[4] Generating batch manifest...")
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "format": format_name,
        "description": "HIV-1 Integrase reveal mutation validation",
        "hypothesis": "LEDGF interface modifications expose epitopes for immune recognition",
        "sequence_info": {
            "reference": "HIV-1 HXB2",
            "length": len(INTEGRASE_WT),
            "ledgf_interface_residues": LEDGF_INTERFACE_RESIDUES,
            "catalytic_residues": CATALYTIC_RESIDUES,
        },
        "jobs": [
            {
                "file": f.name,
                "type": ("wildtype" if "wildtype" in f.name else ("reveal" if "resistance" not in f.name else "resistance")),
                "priority": i + 1,
            }
            for i, f in enumerate(generated_files)
        ],
        "analysis_plan": [
            "1. Compare WT vs mutant backbone RMSD",
            "2. Calculate pLDDT changes at LEDGF interface",
            "3. Measure solvent accessible surface area (SASA) changes",
            "4. Identify newly exposed epitope regions",
            "5. Compare reveal mutations vs resistance mutations",
        ],
        "total_jobs": len(generated_files),
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Saved: {manifest_path.name}")

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"\n  Total files generated: {len(generated_files) + 1}")
    print(f"  Output directory: {output_dir}")

    print("\n  Priority order for validation:")
    print("    1. HIV1_IN_WT (baseline)")
    print("    2. HIV1_IN_E166K (top reveal candidate)")
    print("    3. HIV1_IN_K175E (second reveal candidate)")
    print("    4. HIV1_IN_W131A (third reveal candidate)")

    return generated_files


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent

    # Generate both formats
    print("\n" + "=" * 60)
    print("ALPHAFOLD INTEGRASE INPUT GENERATOR")
    print("=" * 60)

    # Local AlphaFold3 format
    local_dir = script_dir.parent / "inputs" / "integrase" / "local"
    generate_all_inputs(local_dir, use_server_format=False)

    print("\n")

    # AlphaFold Server format
    server_dir = script_dir.parent / "inputs" / "integrase" / "server"
    generate_all_inputs(server_dir, use_server_format=True)

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(
        """
  For AlphaFold Server (immediate):
    1. Go to https://alphafoldserver.com
    2. Upload files from inputs/integrase/server/
    3. Submit jobs (20/day limit)
    4. Download results to outputs/

  For Local AlphaFold3 (requires setup):
    1. Request model weights from Google
    2. Follow README.md setup instructions
    3. Use files from inputs/integrase/local/
"""
    )


if __name__ == "__main__":
    main()
