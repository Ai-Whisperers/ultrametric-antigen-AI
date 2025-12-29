#!/usr/bin/env python3
"""
Extract Arginine Contexts from Human Proteome

For each arginine (R) in the proteome:
- Extract a window of ±4 residues (9-mer centered on R)
- Record position, flanking sequence, protein metadata

Output directory: results/proteome_wide/13_arginine_contexts/

Version: 1.0
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

# Window size: ±4 residues around R = 9-mer total
WINDOW_HALF = 4
WINDOW_SIZE = 2 * WINDOW_HALF + 1  # 9

# Output configuration
SCRIPT_NUM = "13"
OUTPUT_SUBDIR = f"{SCRIPT_NUM}_arginine_contexts"
INPUT_SUBDIR = "12_human_proteome"


# ============================================================================
# DIRECTORY SETUP
# ============================================================================


def get_output_dir() -> Path:
    """Get output directory for this script."""
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / "proteome_wide" / OUTPUT_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_input_dir() -> Path:
    """Get input directory (from previous script)."""
    script_dir = Path(__file__).parent
    return script_dir.parent / "results" / "proteome_wide" / INPUT_SUBDIR


# ============================================================================
# ARGININE EXTRACTION
# ============================================================================


def extract_arginine_sites(proteins: Dict) -> List[Dict]:
    """
    Extract all arginine sites with context windows.

    Args:
        proteins: Dict mapping accession -> protein data (with 'sequence')

    Returns:
        List of arginine site records
    """
    print("\n[1] Extracting arginine sites...")

    sites = []
    proteins_processed = 0
    proteins_with_r = 0

    for acc, prot in proteins.items():
        seq = prot.get("sequence", "")
        if not seq:
            continue

        proteins_processed += 1
        protein_arginines = []

        # Find all R positions
        for pos, aa in enumerate(seq):
            if aa == "R":
                # Extract window
                start = max(0, pos - WINDOW_HALF)
                end = min(len(seq), pos + WINDOW_HALF + 1)
                window = seq[start:end]

                # Position of R within window
                r_pos_in_window = pos - start

                # Pad if at edges
                left_pad = WINDOW_HALF - (pos - start)
                right_pad = WINDOW_HALF - (end - pos - 1)

                if left_pad > 0:
                    window = "X" * left_pad + window
                    r_pos_in_window += left_pad
                if right_pad > 0:
                    window = window + "X" * right_pad

                site = {
                    "protein_id": acc,
                    "gene_name": (prot.get("gene_names", "").split()[0] if prot.get("gene_names") else acc),
                    "protein_name": prot.get("protein_name", "")[:100],  # Truncate long names
                    "protein_length": len(seq),
                    "r_position": pos,  # 0-indexed position in protein
                    "r_position_normalized": pos / len(seq),  # Relative position
                    "window_sequence": window,
                    "r_pos_in_window": r_pos_in_window,
                    "n_terminal_distance": pos,
                    "c_terminal_distance": len(seq) - pos - 1,
                    "total_arginines_in_protein": prot.get("n_arginines", seq.count("R")),
                    "is_n_terminal": pos < 20,
                    "is_c_terminal": pos > len(seq) - 20,
                    # Metadata for enrichment
                    "go_cc": prot.get("go_cellular_component", ""),
                    "go_mf": prot.get("go_molecular_function", ""),
                    "go_bp": prot.get("go_biological_process", ""),
                    "subcellular_location": prot.get("subcellular_location", ""),
                    "has_disease_annotation": bool(prot.get("disease", "")),
                    "has_structure": bool(prot.get("pdb_structures", "")),
                }

                # Count other amino acids in window
                site["n_arginines_in_window"] = window.count("R")
                site["n_lysines_in_window"] = window.count("K")
                site["n_charged_in_window"] = sum(window.count(aa) for aa in "RKDEH")
                site["n_hydrophobic_in_window"] = sum(window.count(aa) for aa in "AVILMFYW")

                sites.append(site)
                protein_arginines.append(pos)

        if protein_arginines:
            proteins_with_r += 1

        # Progress
        if proteins_processed % 5000 == 0:
            print(f"  Processed {proteins_processed:,} proteins, found {len(sites):,} R sites...")

    print(f"\n  Total proteins: {proteins_processed:,}")
    print(f"  Proteins with R: {proteins_with_r:,}")
    print(f"  Total R sites: {len(sites):,}")
    print(f"  Mean R per protein: {len(sites)/proteins_processed:.1f}")

    return sites


# ============================================================================
# STATISTICS AND FILTERING
# ============================================================================


def compute_site_statistics(sites: List[Dict], output_dir: Path) -> Dict:
    """Compute statistics about arginine sites."""
    print("\n[2] Computing site statistics...")

    stats = {
        "total_sites": len(sites),
        "unique_proteins": len(set(s["protein_id"] for s in sites)),
        "window_size": WINDOW_SIZE,
    }

    # Position distribution
    positions = [s["r_position_normalized"] for s in sites]
    stats["position_mean"] = sum(positions) / len(positions)
    stats["n_terminal_sites"] = sum(1 for s in sites if s["is_n_terminal"])
    stats["c_terminal_sites"] = sum(1 for s in sites if s["is_c_terminal"])

    # Window composition
    stats["mean_arginines_in_window"] = sum(s["n_arginines_in_window"] for s in sites) / len(sites)
    stats["mean_charged_in_window"] = sum(s["n_charged_in_window"] for s in sites) / len(sites)

    # Annotations
    stats["sites_with_go"] = sum(1 for s in sites if s["go_bp"])
    stats["sites_with_disease"] = sum(1 for s in sites if s["has_disease_annotation"])
    stats["sites_with_structure"] = sum(1 for s in sites if s["has_structure"])

    # Most common window sequences
    window_counts = Counter(s["window_sequence"] for s in sites)
    stats["unique_windows"] = len(window_counts)
    stats["most_common_windows"] = window_counts.most_common(20)

    # Save
    stats_path = output_dir / "arginine_site_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Saved: {stats_path}")

    # Print summary
    print(f"\n  Total sites: {stats['total_sites']:,}")
    print(f"  Unique proteins: {stats['unique_proteins']:,}")
    print(f"  Unique windows: {stats['unique_windows']:,}")
    print(f"  N-terminal sites (<20 AA): {stats['n_terminal_sites']:,}")
    print(f"  C-terminal sites: {stats['c_terminal_sites']:,}")
    print(f"  Sites with GO: {stats['sites_with_go']:,}")

    return stats


def filter_valid_sites(sites: List[Dict]) -> List[Dict]:
    """
    Filter to sites that can be analyzed.

    Removes:
    - Sites with 'X' in critical positions (edge effects)
    - Sites with non-standard amino acids
    """
    print("\n[3] Filtering valid sites...")

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    valid_sites = []

    for site in sites:
        window = site["window_sequence"]

        # Check for padding (X) in critical region
        # Allow X only at very edges
        core = window[1:-1]  # Central 7 residues
        if "X" in core:
            continue

        # Check for non-standard AA
        if not all(aa in valid_aa or aa == "X" for aa in window):
            continue

        valid_sites.append(site)

    print(f"  Input sites: {len(sites):,}")
    print(f"  Valid sites: {len(valid_sites):,}")
    print(f"  Filtered out: {len(sites) - len(valid_sites):,}")

    return valid_sites


# ============================================================================
# SAVE OUTPUTS
# ============================================================================


def save_sites(sites: List[Dict], output_dir: Path):
    """Save arginine sites in multiple formats."""
    print("\n[4] Saving arginine sites...")

    # Save as JSON (full data)
    json_path = output_dir / "arginine_sites.json"
    with open(json_path, "w") as f:
        json.dump(sites, f)
    print(f"  Saved: {json_path} ({json_path.stat().st_size:,} bytes)")

    # Save as Parquet (efficient for large data)
    try:
        df = pd.DataFrame(sites)
        parquet_path = output_dir / "arginine_sites.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"  Saved: {parquet_path} ({parquet_path.stat().st_size:,} bytes)")
    except Exception as e:
        print(f"  Warning: Could not save Parquet ({e})")

    # Save as CSV (human-readable subset)
    csv_columns = [
        "protein_id",
        "gene_name",
        "r_position",
        "window_sequence",
        "protein_length",
        "total_arginines_in_protein",
        "n_arginines_in_window",
        "is_n_terminal",
        "is_c_terminal",
    ]
    csv_path = output_dir / "arginine_sites_summary.csv"
    df_csv = pd.DataFrame([{k: s.get(k) for k in csv_columns} for s in sites])
    df_csv.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({csv_path.stat().st_size:,} bytes)")

    # Save unique windows for quick reference
    unique_windows = list(set(s["window_sequence"] for s in sites))
    windows_path = output_dir / "unique_windows.txt"
    with open(windows_path, "w") as f:
        f.write("\n".join(sorted(unique_windows)))
    print(f"  Saved: {windows_path} ({len(unique_windows):,} unique windows)")


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 80)
    print("EXTRACT ARGININE CONTEXTS")
    print("Extracting all R sites with 9-mer windows from human proteome")
    print("=" * 80)

    # Setup directories
    input_dir = get_input_dir()
    output_dir = get_output_dir()
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Load proteome data
    print("\n[0] Loading proteome data...")
    proteome_path = input_dir / "human_proteome_full.json"

    if not proteome_path.exists():
        print(f"  ERROR: Proteome file not found: {proteome_path}")
        print("  Please run script 12_download_human_proteome.py first")
        return

    with open(proteome_path, "r") as f:
        proteins = json.load(f)
    print(f"  Loaded {len(proteins):,} proteins")

    # Extract arginine sites
    sites = extract_arginine_sites(proteins)

    # Filter valid sites
    valid_sites = filter_valid_sites(sites)

    # Compute statistics
    stats = compute_site_statistics(valid_sites, output_dir)

    # Save outputs
    save_sites(valid_sites, output_dir)

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print(f"Output: {output_dir}")
    print("=" * 80)

    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")

    print(f"\nReady for Script 14: {len(valid_sites):,} arginine sites to process")

    return valid_sites


if __name__ == "__main__":
    main()
