#!/usr/bin/env python3
"""
Combine AlphaFold Server jobs into a single batch JSON file.

AlphaFold Server format: List of job dictionaries
Reference: https://github.com/google-deepmind/alphafold/blob/main/server/README.md
"""

import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent / "alphafold3_inputs"
OUTPUT_FILE = BASE_DIR / "batch_all_jobs.json"

# Job order (priority-based)
JOB_ORDER = [
    # Priority 1: Wild type baseline
    "wild_type/BG505_gp120_WT.json",

    # Priority 2: Top Goldilocks sentinel sites (single deglycosylation)
    "single_deglyc/BG505_deglyc_N58.json",   # V1, 22.4% shift
    "single_deglyc/BG505_deglyc_N429.json",  # C5, 22.6% shift
    "single_deglyc/BG505_deglyc_N103.json",  # V2, 23.7% shift
    "single_deglyc/BG505_deglyc_N204.json",  # V3, 25.1% shift

    # Priority 3: Above-Goldilocks sites
    "single_deglyc/BG505_deglyc_N246.json",  # C3, 30.0% shift
    "single_deglyc/BG505_deglyc_N152.json",  # C2, 30.6% shift
    "single_deglyc/BG505_deglyc_N155.json",  # C2, 31.2% shift
    "single_deglyc/BG505_deglyc_N232.json",  # C3, 31.6% shift
    "single_deglyc/BG505_deglyc_N324.json",  # C4, 32.1% shift
    "single_deglyc/BG505_deglyc_N360.json",  # C4, 33.2% shift

    # Priority 4: Multi-deglycosylation combinations
    "multi_deglyc/BG505_deglyc_all_goldilocks.json",  # All 7 Goldilocks sites
    "multi_deglyc/BG505_deglyc_region_C2.json",       # CD4bs proximal
    "multi_deglyc/BG505_deglyc_region_C3.json",       # Core glycans
    "multi_deglyc/BG505_deglyc_region_C4.json",       # gp41 interface

    # Priority 5: Direct comparisons (WT fragment vs deglycosylated)
    "comparison/BG505_cmp_N58.json",
    "comparison/BG505_cmp_N429.json",
    "comparison/BG505_cmp_N103.json",
    "comparison/BG505_cmp_N204.json",
    "comparison/BG505_cmp_N246.json",
]


def main():
    batch_jobs = []
    missing_files = []

    for job_path in JOB_ORDER:
        full_path = BASE_DIR / job_path
        if full_path.exists():
            with open(full_path, 'r') as f:
                job = json.load(f)
                batch_jobs.append(job)
                print(f"  Added: {job['name']}")
        else:
            missing_files.append(job_path)
            print(f"  MISSING: {job_path}")

    # Write combined batch file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(batch_jobs, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Batch file created: {OUTPUT_FILE}")
    print(f"Total jobs: {len(batch_jobs)}")

    if missing_files:
        print(f"\nWARNING: {len(missing_files)} files missing:")
        for mf in missing_files:
            print(f"  - {mf}")

    # Validate format
    print(f"\nFormat validation:")
    print(f"  - Is list: {isinstance(batch_jobs, list)}")
    print(f"  - All have 'dialect': {all('dialect' in j for j in batch_jobs)}")
    print(f"  - All alphafoldserver: {all(j.get('dialect') == 'alphafoldserver' for j in batch_jobs)}")
    print(f"  - All have 'sequences': {all('sequences' in j for j in batch_jobs)}")
    print(f"  - All have 'modelSeeds': {all('modelSeeds' in j for j in batch_jobs)}")


if __name__ == "__main__":
    main()
