#!/usr/bin/env python3
"""
RA AlphaFold Validation Job Generator

Generates AlphaFold Server jobs for structural validation of high-priority
citrullination targets identified by handshake analysis.

Focus on HLA-DRB1-peptide (pMHC) complexes to validate:
1. Citrullinated peptide binding geometry
2. Shared epitope interactions
3. Known ACPA sites vs predicted targets

Part of Phase 1: RA Extensions (PRIORITY)
See: research/genetic_code/PTM_EXTENSION_PLAN.md

Input: research/bioinformatics/rheumatoid_arthritis/data/ra_high_priority_targets.json
Output: research/bioinformatics/rheumatoid_arthritis/alphafold_jobs/ra_validation_batch.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# =============================================================================
# HLA-DRB1*04:01 SEQUENCE (HIGH-RISK RA ALLELE)
# =============================================================================

# DRB1*04:01 extracellular domain (positions 1-227)
# Source: UniProt P04229, IMGT/HLA
HLA_DRB1_0401_SEQUENCE = (
    "GDTRPRFLWQLKFECHFFNGTERVRLLERCIYNQEESVRFDSDVGEYRAVTELGRPDAEYWNSQKDLLE"
    "QRRAAVDTYCRHNYGVGESFTVQRRVHPKVTVYPSKTQPLQHHNLLVCSVSGFYPGSIEVRWFRNGQEE"
    "KAGVVSTGLIQNGDWTFQTLVMLETVPRSGEVYTCQVEHPSVTSPLTVEWRARSESAQSKMLSGVGGFV"
    "LGLLFLGAGLFIIYFRNQK"
)

# DRB1*04:01 alpha chain (DRA*01:01) - constant for all DR molecules
HLA_DRA_SEQUENCE = (
    "IKEEHVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKAN"
    "LEIMTKRSNYTPITNVPPEVTVLTNSPVELREPNVLICFIDKFTPPVVNVTWLRNGKPVTTGVSETVFL"
    "PREDHLFRKFHYLPFLPSTEDVYDCRVEHWGLDEPLLKHWEFDAPSPLPETTEN"
)


# =============================================================================
# PEPTIDE EXTRACTION
# =============================================================================


def extract_peptide_context(sequence: str, position: int, length: int = 15) -> str:
    """
    Extract peptide centered on modification site.
    HLA class II typically binds 13-25 aa peptides, core is 9 aa.
    """
    half_len = length // 2
    start = max(0, position - 1 - half_len)  # position is 1-indexed
    end = min(len(sequence), position - 1 + half_len + 1)

    peptide = sequence[start:end]

    # Pad if at boundaries
    if len(peptide) < length:
        if start == 0:
            peptide = peptide.ljust(length, "G")
        else:
            peptide = peptide.rjust(length, "G")

    return peptide


def apply_citrullination(sequence: str, position: int) -> str:
    """Apply R->Q citrullination at specified position (1-indexed)."""
    seq_list = list(sequence)
    if 0 < position <= len(seq_list) and seq_list[position - 1] == "R":
        seq_list[position - 1] = "Q"
    return "".join(seq_list)


# =============================================================================
# JOB GENERATION
# =============================================================================


def create_pmhc_job(name: str, peptide: str, description: str = "") -> dict:
    """
    Create AlphaFold Server job for pMHC complex.

    Structure: DRA (alpha) + DRB1*04:01 (beta) + peptide
    """
    return {
        "name": name,
        "modelSeeds": [],
        "sequences": [
            {"proteinChain": {"sequence": HLA_DRA_SEQUENCE, "count": 1}},
            {"proteinChain": {"sequence": HLA_DRB1_0401_SEQUENCE, "count": 1}},
            {"proteinChain": {"sequence": peptide, "count": 1}},
        ],
        "dialect": "alphafoldserver",
        "version": 1,
        "_metadata": {
            "description": description,
            "hla_allele": "DRB1*04:01",
            "peptide_length": len(peptide),
        },
    }


def generate_ra_validation_jobs(proteins_data: dict, high_priority_targets: list) -> list:
    """Generate AlphaFold jobs for RA validation."""

    jobs = []

    # Get protein sequences
    protein_seqs = {p["name"]: p["sequence"] for p in proteins_data["proteins"]}

    # Focus on known ACPA sites first
    known_acpa = [t for t in high_priority_targets if t["is_known_acpa"]]

    print(f"\n  Found {len(known_acpa)} known ACPA sites with score=3")

    # Generate jobs for known ACPA sites (priority)
    for target in known_acpa[:3]:  # Top 3 known sites
        protein = target["protein"]
        position = target["position"]
        seq = protein_seqs.get(protein, "")

        if not seq:
            continue

        # Extract peptide context
        wt_peptide = extract_peptide_context(seq, position, length=15)
        cit_peptide = apply_citrullination(wt_peptide, 8)  # Center position

        # Generate WT job
        job_name_wt = f"ra_{protein.lower().replace(' ', '_')}_R{position}_WT_DRB10401"
        jobs.append(
            create_pmhc_job(
                name=job_name_wt[:50],  # AlphaFold name limit
                peptide=wt_peptide,
                description=f"WT {protein} R{position} with HLA-DRB1*04:01",
            )
        )

        # Generate citrullinated job
        job_name_cit = f"ra_{protein.lower().replace(' ', '_')}_R{position}Q_DRB10401"
        jobs.append(
            create_pmhc_job(
                name=job_name_cit[:50],
                peptide=cit_peptide,
                description=f"Citrullinated {protein} R{position}Q with HLA-DRB1*04:01 [KNOWN ACPA]",
            )
        )

    # Add 1-2 predicted high-priority sites (not known ACPA)
    predicted = [t for t in high_priority_targets if not t["is_known_acpa"] and t["ptm_type"] == "R->Q"]

    for target in predicted[:2]:
        protein = target["protein"]
        position = target["position"]
        seq = protein_seqs.get(protein, "")

        if not seq:
            continue

        wt_peptide = extract_peptide_context(seq, position, length=15)
        cit_peptide = apply_citrullination(wt_peptide, 8)

        job_name = f"ra_{protein.lower().replace(' ', '_')}_R{position}Q_predicted"
        jobs.append(
            create_pmhc_job(
                name=job_name[:50],
                peptide=cit_peptide,
                description=f"PREDICTED {protein} R{position}Q with HLA-DRB1*04:01 (score=3)",
            )
        )

    return jobs


def main():
    print("=" * 70)
    print("RA ALPHAFOLD VALIDATION JOB GENERATOR")
    print("Phase 1: RA Extensions (PRIORITY)")
    print("=" * 70)

    # Load data
    data_dir = SCRIPT_DIR.parent / "data"

    proteins_path = data_dir / "acpa_proteins.json"
    if not proteins_path.exists():
        print(f"ERROR: ACPA proteins not found at {proteins_path}")
        return 1

    targets_path = data_dir / "ra_high_priority_targets.json"
    if not targets_path.exists():
        print(f"ERROR: High-priority targets not found at {targets_path}")
        return 1

    print(f"\nLoading ACPA proteins from: {proteins_path}")
    with open(proteins_path) as f:
        proteins_data = json.load(f)

    print(f"\nLoading high-priority targets from: {targets_path}")
    with open(targets_path) as f:
        high_priority_targets = json.load(f)

    print(f"  Total high-priority targets: {len(high_priority_targets)}")

    # Generate jobs
    print("\nGenerating AlphaFold validation jobs...")
    jobs = generate_ra_validation_jobs(proteins_data, high_priority_targets)

    # Summary
    print("\n" + "=" * 70)
    print("JOB GENERATION SUMMARY")
    print("=" * 70)
    print(f"\n  Total jobs generated: {len(jobs)}")
    print("\n  Jobs:")
    for i, job in enumerate(jobs, 1):
        meta = job.get("_metadata", {})
        desc = meta.get("description", job["name"])
        print(f"    {i}. {desc}")

    # Remove metadata before saving (not part of AlphaFold spec)
    jobs_clean = []
    for job in jobs:
        job_copy = {k: v for k, v in job.items() if k != "_metadata"}
        jobs_clean.append(job_copy)

    # Save jobs
    output_dir = SCRIPT_DIR.parent / "alphafold_jobs"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "ra_validation_batch.json"
    with open(output_path, "w") as f:
        json.dump(jobs_clean, f, indent=2)
    print(f"\n  Saved: {output_path}")

    # Save metadata separately
    metadata = {
        "generation_date": datetime.now().isoformat(),
        "hla_allele": "DRB1*04:01",
        "job_count": len(jobs),
        "jobs": [
            {
                "name": job["name"],
                "description": job.get("_metadata", {}).get("description", ""),
                "peptide_length": job.get("_metadata", {}).get("peptide_length", 0),
            }
            for job in jobs
        ],
    }

    metadata_path = output_dir / "ra_validation_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {metadata_path}")

    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("  1. Upload ra_validation_batch.json to AlphaFold Server")
    print("  2. Compare WT vs citrullinated pMHC structures")
    print("  3. Validate shared epitope pocket interactions")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
