#!/usr/bin/env python3
"""
AlphaFold3 Validation Jobs for SARS-CoV-2 Handshake Analysis

Generates AlphaFold3 Server JSON inputs for structural validation of:
1. RBD-ACE2 complex with phospho-mimic modifications
2. Therapeutic peptide candidates binding to ACE2

AlphaFold3 Server: https://alphafoldserver.com/
"""

import json
from pathlib import Path
from datetime import datetime

# Output directory
OUTPUT_DIR = Path(__file__).parent / "alphafold3_jobs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# SEQUENCE DATA
# ============================================================================

# SARS-CoV-2 Spike RBD (residues 319-541) - 223 amino acids
RBD_SEQUENCE = (
    "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFK"
    "CYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNS"
    "NNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQ"
    "PTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNF"
)

# Human ACE2 ectodomain (residues 19-615) - 597 amino acids
ACE2_SEQUENCE = (
    "STIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQST"
    "LAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNP"
    "QECLLLEPGLNEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYED"
    "YGDYWRGDYEVNGVDGYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISP"
    "IGCLPAHLLGDMWGRFWTNLYSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSV"
    "GLPNMTQGFWENSMLTDPGNVQKAVCHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGH"
    "IQYDMAYAAQPFLLRNGANEGFHEAVGEIMSLSAATPKHLKSIGLLSPDFQEDNETEINFLL"
    "KQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQWMKKWWEMKREIVGVVEPVPHDETYC"
    "DPASLFHVSNDYSFIRYYTRTLYQFQFQEALCQAAKHEGPLHKCDISNSTEAGQKLFNML"
    "RLGKSEPWTLALENVVGAKNMNVRPLLNYFEPLFTWLKDQNKNSFVGWSTDWSPYA"
)

# Therapeutic peptide candidates from handshake analysis
# Priority 1: N439/N440 region with S→D modifications
PEPTIDE_PRIORITY_1_WT = "VIAWNSNNLDSKVGG"  # Wild-type
PEPTIDE_PRIORITY_1_MOD = "VIAWNDNNLDDKVGG"  # S→D at positions 6 and 11

# Priority 2: Y449 region with Y→D modification
PEPTIDE_PRIORITY_2_WT = "DSKVGGNYNYLYRLF"  # Wild-type
PEPTIDE_PRIORITY_2_MOD = "DSKVGGNDNYLYRLF"  # Y→D at position 8

# Combined peptide targeting both hotspots
PEPTIDE_COMBINED = "WNDNNLDGGGGGNDNYL"  # Hybrid with GGG linker


def create_alphafold3_job(
    name: str,
    sequences: list,
    description: str = ""
) -> dict:
    """
    Create AlphaFold3 Server compatible JSON structure.

    Args:
        name: Job name
        sequences: List of dicts with 'id', 'sequence', and 'type' keys
        description: Job description

    Returns:
        AlphaFold3 job JSON structure
    """
    # Build sequence entries
    sequence_entries = []
    for seq in sequences:
        entry = {
            "proteinChain": {
                "sequence": seq["sequence"],
                "count": seq.get("count", 1)
            }
        }
        sequence_entries.append(entry)

    job = {
        "name": name,
        "modelSeeds": [],  # Let server choose
        "sequences": sequence_entries,
        "dialect": "alphafold3",
        "version": 1
    }

    return job


def create_job1_rbd_ace2_complexes():
    """
    Job 1: RBD-ACE2 complexes with and without modifications.

    Generates 3 predictions:
    1a. Wild-type RBD + ACE2 (reference)
    1b. RBD with S439D mutation + ACE2
    1c. RBD with S440D mutation + ACE2
    """
    jobs = []

    # 1a. Wild-type reference
    job_wt = {
        "name": "sarscov2_rbd_ace2_wildtype",
        "modelSeeds": [],
        "sequences": [
            {"proteinChain": {"sequence": RBD_SEQUENCE, "count": 1}},
            {"proteinChain": {"sequence": ACE2_SEQUENCE, "count": 1}}
        ],
        "dialect": "alphafold3",
        "version": 1
    }
    jobs.append(("job1a_rbd_ace2_wildtype", job_wt))

    # 1b. RBD with S→D at position 121 (corresponds to S439 in full spike, position 121 in RBD starting at 319)
    # Position 439 in full spike = position 439-319+1 = 121 in our RBD sequence
    # Actually let me recalculate: RBD starts at 319, so position 439 = 439-319 = 120 (0-indexed) = 121st residue
    # The sequence around 439 is: ...CVIAWNSNNLDS...
    # Looking at our RBD: position 120 (0-indexed) should be around the VIAWNSNNLDS region

    # Let's find the exact position of VIAWNSNNLDS in our RBD sequence
    motif = "VIAWNSNNLDS"
    motif_start = RBD_SEQUENCE.find(motif)
    if motif_start >= 0:
        # S at position 6 of motif = motif_start + 5 (0-indexed)
        s439_pos = motif_start + 5  # This is the S in VIAWNSNNLDS
        s440_pos = motif_start + 10  # This is the S in VIAWNSNNLDS (second S after NNLD)

        # Create mutant with S→D at first S position
        rbd_s439d = list(RBD_SEQUENCE)
        rbd_s439d[s439_pos] = 'D'
        rbd_s439d = ''.join(rbd_s439d)

        job_s439d = {
            "name": "sarscov2_rbd_S439D_ace2",
            "modelSeeds": [],
            "sequences": [
                {"proteinChain": {"sequence": rbd_s439d, "count": 1}},
                {"proteinChain": {"sequence": ACE2_SEQUENCE, "count": 1}}
            ],
            "dialect": "alphafold3",
            "version": 1
        }
        jobs.append(("job1b_rbd_S439D_ace2", job_s439d))

        # Create mutant with S→D at second S position
        rbd_s440d = list(RBD_SEQUENCE)
        rbd_s440d[s440_pos] = 'D'
        rbd_s440d = ''.join(rbd_s440d)

        job_s440d = {
            "name": "sarscov2_rbd_S440D_ace2",
            "modelSeeds": [],
            "sequences": [
                {"proteinChain": {"sequence": rbd_s440d, "count": 1}},
                {"proteinChain": {"sequence": ACE2_SEQUENCE, "count": 1}}
            ],
            "dialect": "alphafold3",
            "version": 1
        }
        jobs.append(("job1c_rbd_S440D_ace2", job_s440d))

        # Double mutant
        rbd_double = list(RBD_SEQUENCE)
        rbd_double[s439_pos] = 'D'
        rbd_double[s440_pos] = 'D'
        rbd_double = ''.join(rbd_double)

        job_double = {
            "name": "sarscov2_rbd_S439D_S440D_ace2",
            "modelSeeds": [],
            "sequences": [
                {"proteinChain": {"sequence": rbd_double, "count": 1}},
                {"proteinChain": {"sequence": ACE2_SEQUENCE, "count": 1}}
            ],
            "dialect": "alphafold3",
            "version": 1
        }
        jobs.append(("job1d_rbd_double_mutant_ace2", job_double))

    return jobs


def create_job2_peptide_inhibitors():
    """
    Job 2: Therapeutic peptide candidates binding to ACE2.

    Generates predictions for:
    2a. ACE2 + Wild-type peptide (from N439/N440 region)
    2b. ACE2 + Modified peptide (S→D)
    2c. ACE2 + Priority 2 peptide (Y449 region)
    2d. ACE2 + Combined hybrid peptide
    """
    jobs = []

    # 2a. Wild-type peptide
    job_pep_wt = {
        "name": "ace2_peptide_wt_n439",
        "modelSeeds": [],
        "sequences": [
            {"proteinChain": {"sequence": ACE2_SEQUENCE, "count": 1}},
            {"proteinChain": {"sequence": PEPTIDE_PRIORITY_1_WT, "count": 1}}
        ],
        "dialect": "alphafold3",
        "version": 1
    }
    jobs.append(("job2a_ace2_peptide_wt", job_pep_wt))

    # 2b. Modified peptide (S→D)
    job_pep_mod = {
        "name": "ace2_peptide_phosphomimic_n439",
        "modelSeeds": [],
        "sequences": [
            {"proteinChain": {"sequence": ACE2_SEQUENCE, "count": 1}},
            {"proteinChain": {"sequence": PEPTIDE_PRIORITY_1_MOD, "count": 1}}
        ],
        "dialect": "alphafold3",
        "version": 1
    }
    jobs.append(("job2b_ace2_peptide_phosphomimic", job_pep_mod))

    # 2c. Y449 region peptide (modified)
    job_pep_y449 = {
        "name": "ace2_peptide_y449_mod",
        "modelSeeds": [],
        "sequences": [
            {"proteinChain": {"sequence": ACE2_SEQUENCE, "count": 1}},
            {"proteinChain": {"sequence": PEPTIDE_PRIORITY_2_MOD, "count": 1}}
        ],
        "dialect": "alphafold3",
        "version": 1
    }
    jobs.append(("job2c_ace2_peptide_y449", job_pep_y449))

    # 2d. Combined hybrid peptide
    job_pep_hybrid = {
        "name": "ace2_peptide_hybrid",
        "modelSeeds": [],
        "sequences": [
            {"proteinChain": {"sequence": ACE2_SEQUENCE, "count": 1}},
            {"proteinChain": {"sequence": PEPTIDE_COMBINED, "count": 1}}
        ],
        "dialect": "alphafold3",
        "version": 1
    }
    jobs.append(("job2d_ace2_peptide_hybrid", job_pep_hybrid))

    return jobs


def main():
    print("=" * 70)
    print("AlphaFold3 Validation Job Generator")
    print("SARS-CoV-2 Handshake Analysis")
    print("=" * 70)

    print(f"\nOutput directory: {OUTPUT_DIR}")

    # ========================================================================
    # Job 1: RBD-ACE2 Complexes
    # ========================================================================
    print("\n" + "-" * 70)
    print("JOB 1: RBD-ACE2 Complex Predictions")
    print("-" * 70)

    job1_list = create_job1_rbd_ace2_complexes()

    print(f"\nGenerated {len(job1_list)} RBD-ACE2 complex jobs:")
    for name, job in job1_list:
        filepath = OUTPUT_DIR / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(job, f, indent=2)
        print(f"  - {name}.json")
        print(f"    Sequences: {len(job['sequences'])} chains")

    # ========================================================================
    # Job 2: Peptide Inhibitor Predictions
    # ========================================================================
    print("\n" + "-" * 70)
    print("JOB 2: Peptide Inhibitor Predictions")
    print("-" * 70)

    job2_list = create_job2_peptide_inhibitors()

    print(f"\nGenerated {len(job2_list)} peptide binding jobs:")
    for name, job in job2_list:
        filepath = OUTPUT_DIR / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(job, f, indent=2)
        print(f"  - {name}.json")
        print(f"    Peptide: {job['sequences'][1]['proteinChain']['sequence']}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
Total jobs generated: {len(job1_list) + len(job2_list)}

JOB 1 - RBD-ACE2 Complex Analysis:
  Purpose: Validate whether S→D mutations disrupt RBD-ACE2 interface
  Files:
    - job1a_rbd_ace2_wildtype.json (reference)
    - job1b_rbd_S439D_ace2.json (single mutant)
    - job1c_rbd_S440D_ace2.json (single mutant)
    - job1d_rbd_double_mutant_ace2.json (double mutant)

  Expected outcome:
    - Wild-type: High iPTM, tight interface
    - Mutants: Lower iPTM, disrupted interface contacts

JOB 2 - Peptide Inhibitor Binding:
  Purpose: Validate whether therapeutic peptides bind ACE2 at K353 interface
  Files:
    - job2a_ace2_peptide_wt.json (wild-type peptide)
    - job2b_ace2_peptide_phosphomimic.json (S→D modified)
    - job2c_ace2_peptide_y449.json (Y449 region)
    - job2d_ace2_peptide_hybrid.json (combined)

  Expected outcome:
    - Modified peptides should show binding to ACE2
    - Compare iPTM between wild-type and modified peptides

INSTRUCTIONS:
  1. Go to https://alphafoldserver.com/
  2. Sign in with Google account
  3. Click "New Job"
  4. Upload each JSON file or paste contents
  5. Submit and wait for results (~minutes to hours)
  6. Download results for analysis
""")

    # Create a batch submission script description
    batch_info = {
        "generated": datetime.now().isoformat(),
        "total_jobs": len(job1_list) + len(job2_list),
        "job1_count": len(job1_list),
        "job2_count": len(job2_list),
        "files": [name + ".json" for name, _ in job1_list + job2_list],
        "expected_outcomes": {
            "job1": "S→D mutations should reduce iPTM scores, indicating interface disruption",
            "job2": "Modified peptides should show binding affinity to ACE2 K353 region"
        }
    }

    with open(OUTPUT_DIR / "batch_info.json", 'w') as f:
        json.dump(batch_info, f, indent=2)

    print(f"\nBatch info saved to: {OUTPUT_DIR / 'batch_info.json'}")


if __name__ == "__main__":
    main()
