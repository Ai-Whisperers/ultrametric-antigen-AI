#!/usr/bin/env python3
"""
AlphaFold Server Validation Jobs for Tau Phosphorylation Analysis

Generates AlphaFold Server JSON inputs for structural validation of:
1. Tau MTBR region with/without phosphomimics
2. Tau-Tubulin interface with key phosphorylation sites
3. Pathological epitope combinations (AT8, PHF-1, KXGS)

AlphaFold Server: https://alphafoldserver.com/
Format reference: https://github.com/google-deepmind/alphafold/blob/main/server/README.md

Based on combinatorial analysis showing:
- All combinations are ADDITIVE (no synergistic tipping points)
- KXGS phosphorylation causes maximal MTBR dysfunction
- Braak V/VI shows highest combined shift (7.2%)
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from data.tau_phospho_database import (
    TAU_2N4R_SEQUENCE,
    TAU_DOMAINS,
    TAU_PHOSPHO_SITES,
    KXGS_MOTIFS,
)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "alphafold3_jobs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# SEQUENCE DATA
# ============================================================================

# Human alpha-tubulin (TUBA1B) - partial binding domain
# Residues that interact with tau MTBR (from cryo-EM: PDB 6CVN)
# Using residues 408-451 from alpha-tubulin (tau binding surface)
ALPHA_TUBULIN_BINDING_REGION = (
    "SCLLYNYQDQTKQSHLQTVQEDLKNVMDDCDPRHGKYMACCL"
)

# Human beta-tubulin (TUBB) - partial binding domain
# Residues 387-430 from beta-tubulin
BETA_TUBULIN_BINDING_REGION = (
    "ATNGCDLAKEIMEGVDLQPVLSNASTTAVCDIPRGLKMSATF"
)

# Full MTBR region from tau (residues 244-368)
MTBR_START = 244 - 1  # 0-indexed
MTBR_END = 368
TAU_MTBR_SEQUENCE = TAU_2N4R_SEQUENCE[MTBR_START:MTBR_END]

# Extended MTBR with flanking regions for better folding context
# Include proline-rich region and C-terminal (residues 200-400)
EXTENDED_START = 200 - 1
EXTENDED_END = 400
TAU_EXTENDED_SEQUENCE = TAU_2N4R_SEQUENCE[EXTENDED_START:EXTENDED_END]


def apply_phosphomimic(sequence: str, positions: list, offset: int = 0) -> str:
    """
    Apply S/T→D phosphomimetic mutations at specified positions.

    Args:
        sequence: Protein sequence
        positions: List of positions (1-indexed in full tau)
        offset: Offset to convert full tau positions to sequence positions

    Returns:
        Modified sequence with S/T replaced by D
    """
    seq_list = list(sequence)

    for pos in positions:
        seq_pos = pos - 1 - offset  # Convert to 0-indexed in this sequence

        if 0 <= seq_pos < len(seq_list):
            aa = seq_list[seq_pos]
            if aa in ['S', 'T']:
                seq_list[seq_pos] = 'D'
            elif aa == 'Y':
                seq_list[seq_pos] = 'D'  # Y→D for tyrosine phosphomimetic

    return ''.join(seq_list)


def create_job(name: str, sequences: list) -> dict:
    """
    Create an AlphaFold Server compatible job structure.

    Args:
        name: Job name
        sequences: List of sequence strings (will be wrapped in proteinChain)

    Returns:
        AlphaFold Server job dictionary
    """
    sequence_entries = []
    for seq in sequences:
        sequence_entries.append({
            "proteinChain": {
                "sequence": seq,
                "count": 1
            }
        })

    return {
        "name": name,
        "modelSeeds": [],
        "sequences": sequence_entries,
        "dialect": "alphafoldserver",
        "version": 1
    }


def create_tau_mtbr_jobs():
    """
    Job Set 1: Tau MTBR region alone with phosphomimics.

    Tests structural impact of KXGS motif phosphorylation.
    """
    jobs = []

    # 1a. Wild-type MTBR (reference)
    jobs.append(create_job("tau_mtbr_wildtype", [TAU_MTBR_SEQUENCE]))

    # 1b. Single KXGS phosphomimics
    kxgs_sites = [262, 293, 324, 356]  # S262, S293, S324, S356

    for site in kxgs_sites:
        mtbr_modified = apply_phosphomimic(TAU_MTBR_SEQUENCE, [site], offset=MTBR_START)
        jobs.append(create_job(f"tau_mtbr_S{site}D", [mtbr_modified]))

    # 1c. Full KXGS quadruple phosphomimic (maximum MTBR disruption)
    mtbr_all_kxgs = apply_phosphomimic(TAU_MTBR_SEQUENCE, kxgs_sites, offset=MTBR_START)
    jobs.append(create_job("tau_mtbr_all_KXGS_phospho", [mtbr_all_kxgs]))

    return jobs


def create_tau_tubulin_jobs():
    """
    Job Set 2: Tau-Tubulin interface analysis.

    Tests whether phosphomimics disrupt tau-tubulin binding.
    """
    jobs = []

    # Use extended tau sequence for better structural context
    offset = EXTENDED_START

    # 2a. Wild-type tau (extended) + tubulin fragments
    jobs.append(create_job(
        "tau_tubulin_wildtype",
        [TAU_EXTENDED_SEQUENCE, ALPHA_TUBULIN_BINDING_REGION, BETA_TUBULIN_BINDING_REGION]
    ))

    # 2b. S262D (major MT detachment site) + tubulin
    tau_s262d = apply_phosphomimic(TAU_EXTENDED_SEQUENCE, [262], offset=offset)
    jobs.append(create_job(
        "tau_tubulin_S262D",
        [tau_s262d, ALPHA_TUBULIN_BINDING_REGION, BETA_TUBULIN_BINDING_REGION]
    ))

    # 2c. All KXGS phospho + tubulin (maximum disruption)
    kxgs_sites = [262, 293, 324, 356]
    tau_all_kxgs = apply_phosphomimic(TAU_EXTENDED_SEQUENCE, kxgs_sites, offset=offset)
    jobs.append(create_job(
        "tau_tubulin_all_KXGS",
        [tau_all_kxgs, ALPHA_TUBULIN_BINDING_REGION, BETA_TUBULIN_BINDING_REGION]
    ))

    return jobs


def create_pathological_epitope_jobs():
    """
    Job Set 3: Pathological epitope combinations.

    Tests AT8, PHF-1, and AT180 epitopes.
    """
    jobs = []

    offset = EXTENDED_START

    # 3a. AT8 (S202 + T205) - early pathology marker
    at8_sites = [202, 205]
    tau_at8 = apply_phosphomimic(TAU_EXTENDED_SEQUENCE, at8_sites, offset=offset)
    jobs.append(create_job("tau_AT8_epitope", [tau_at8]))

    # 3b. PHF-1 (S396 + S404) - late pathology marker
    # Need to use different sequence range for C-terminal
    c_term_start = 350 - 1
    c_term_end = 441
    tau_cterm = TAU_2N4R_SEQUENCE[c_term_start:c_term_end]
    tau_phf1 = apply_phosphomimic(tau_cterm, [396, 404], offset=c_term_start)
    jobs.append(create_job("tau_PHF1_epitope", [tau_phf1]))

    # 3c. AT180 (T231 + S235) - conformational epitope
    at180_sites = [231, 235]
    tau_at180 = apply_phosphomimic(TAU_EXTENDED_SEQUENCE, at180_sites, offset=offset)
    jobs.append(create_job("tau_AT180_epitope", [tau_at180]))

    # 3d. Braak V/VI simulation (maximum phosphorylation)
    # Sites from our combinatorial analysis that reached 7.2% shift
    braak_vi_sites = [181, 231, 202, 205, 262, 293, 324, 356, 396]
    # Use full tau for this comprehensive test
    tau_braak = apply_phosphomimic(TAU_2N4R_SEQUENCE, braak_vi_sites, offset=0)
    jobs.append(create_job("tau_Braak_VI_simulation", [tau_braak]))

    return jobs


def main():
    print("=" * 70)
    print("AlphaFold Server Validation Job Generator")
    print("Tau Phosphorylation - Alzheimer's Disease Analysis")
    print("=" * 70)

    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Verify sequences
    print(f"\nSequence lengths:")
    print(f"  Full tau (2N4R): {len(TAU_2N4R_SEQUENCE)} aa")
    print(f"  MTBR region: {len(TAU_MTBR_SEQUENCE)} aa (residues {MTBR_START+1}-{MTBR_END})")
    print(f"  Extended region: {len(TAU_EXTENDED_SEQUENCE)} aa (residues {EXTENDED_START+1}-{EXTENDED_END})")
    print(f"  Alpha-tubulin fragment: {len(ALPHA_TUBULIN_BINDING_REGION)} aa")
    print(f"  Beta-tubulin fragment: {len(BETA_TUBULIN_BINDING_REGION)} aa")

    # Collect all jobs
    all_jobs = []

    # ========================================================================
    # Job Set 1: MTBR Phosphomimics
    # ========================================================================
    print("\n" + "-" * 70)
    print("JOB SET 1: Tau MTBR with KXGS Phosphomimics")
    print("-" * 70)

    job1_list = create_tau_mtbr_jobs()
    all_jobs.extend(job1_list)
    print(f"  Generated {len(job1_list)} MTBR jobs")

    # ========================================================================
    # Job Set 2: Tau-Tubulin Interface
    # ========================================================================
    print("\n" + "-" * 70)
    print("JOB SET 2: Tau-Tubulin Interface Analysis")
    print("-" * 70)

    job2_list = create_tau_tubulin_jobs()
    all_jobs.extend(job2_list)
    print(f"  Generated {len(job2_list)} tau-tubulin jobs")

    # ========================================================================
    # Job Set 3: Pathological Epitopes
    # ========================================================================
    print("\n" + "-" * 70)
    print("JOB SET 3: Pathological Epitope Combinations")
    print("-" * 70)

    job3_list = create_pathological_epitope_jobs()
    all_jobs.extend(job3_list)
    print(f"  Generated {len(job3_list)} pathological epitope jobs")

    # ========================================================================
    # Write single batch JSON file
    # ========================================================================
    print("\n" + "=" * 70)
    print("BATCH OUTPUT")
    print("=" * 70)

    batch_file = OUTPUT_DIR / "tau_phospho_batch.json"
    with open(batch_file, 'w') as f:
        json.dump(all_jobs, f, indent=2)
    print(f"\nBatch file saved: {batch_file}")
    print(f"Total jobs in batch: {len(all_jobs)}")

    # List all jobs
    print("\nJobs in batch:")
    for i, job in enumerate(all_jobs, 1):
        n_chains = len(job['sequences'])
        print(f"  {i:2d}. {job['name']} ({n_chains} chain{'s' if n_chains > 1 else ''})")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
Total jobs: {len(all_jobs)}

FORMAT: AlphaFold Server (dialect: alphafoldserver, version: 1)

JOB SET 1 - MTBR KXGS Phosphomimics ({len(job1_list)} jobs):
  Purpose: Test structural impact of MARK kinase phosphorylation sites
  Key sites: S262 (12E8), S293, S324, S356
  Prediction: Phosphomimics should increase disorder/flexibility

JOB SET 2 - Tau-Tubulin Interface ({len(job2_list)} jobs):
  Purpose: Validate phosphomimic disruption of MT binding
  Expected: S262D should reduce tau-tubulin iPTM
  Full KXGS phospho should show maximum interface disruption

JOB SET 3 - Pathological Epitopes ({len(job3_list)} jobs):
  Purpose: Test clinically-relevant phosphorylation patterns
  AT8: Early AD marker (S202+T205)
  PHF-1: Late AD marker (S396+S404)
  AT180: Conformational epitope (T231+S235)
  Braak VI: Maximum hyperphosphorylation simulation

INSTRUCTIONS:
  1. Go to https://alphafoldserver.com/
  2. Sign in with Google account
  3. Upload {batch_file.name}
  4. Submit batch and wait for results
  5. Download results for analysis

KEY METRICS TO COMPARE:
  - pTM: Overall confidence (should decrease with phosphomimics)
  - iPTM: Interface PTM for tau-tubulin complexes
  - PAE: Predicted aligned error (should increase with phosphomimics)
  - pLDDT: Per-residue confidence
  - Fraction disordered: Should INCREASE with phosphomimics
""")

    # Create metadata file
    metadata = {
        "generated": datetime.now().isoformat(),
        "analysis": "Tau Phosphorylation - Alzheimer's Disease",
        "format": {
            "dialect": "alphafoldserver",
            "version": 1,
            "reference": "https://github.com/google-deepmind/alphafold/blob/main/server/README.md"
        },
        "batch_file": "tau_phospho_batch.json",
        "total_jobs": len(all_jobs),
        "job_sets": {
            "set1_mtbr": len(job1_list),
            "set2_tau_tubulin": len(job2_list),
            "set3_epitopes": len(job3_list)
        },
        "job_names": [job['name'] for job in all_jobs],
        "hypotheses": {
            "primary": "Phosphomimics increase tau disorder and reduce tubulin binding",
            "mtbr": "KXGS phosphorylation disrupts MTBR structure",
            "interface": "S262D should show reduced iPTM with tubulin",
            "disease": "Braak VI pattern shows maximum structural disruption"
        },
        "reference": {
            "framework": "3-adic codon encoder v5.11.3",
            "combinatorial_result": "All combinations ADDITIVE, max 7.2% shift at Braak V/VI",
            "p_adic_prediction": "Linear accumulation, no synergistic tipping points"
        }
    }

    with open(OUTPUT_DIR / "batch_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {OUTPUT_DIR / 'batch_metadata.json'}")


if __name__ == "__main__":
    main()
