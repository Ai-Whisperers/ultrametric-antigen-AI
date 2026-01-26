#!/usr/bin/env python3
"""Regenerate FASTA files from existing primer CSV files.

The original A2_pan_arbovirus_primers.py only exports 'specific' primers to FASTA,
but all primers in the CSV have is_specific=False (biological reality for DENV-4
cryptic diversity), resulting in empty FASTA files.

This script reads the CSV files and generates FASTA files with ALL primers,
marking non-specific ones in the header.

Usage:
    python scripts/regenerate_fasta_from_csv.py
    python scripts/regenerate_fasta_from_csv.py --top 5  # Only top 5 per virus
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def regenerate_fastas(
    results_dir: Path,
    top_n: int | None = None,
    only_specific: bool = False,
) -> dict[str, int]:
    """Regenerate FASTA files from CSV primer data.

    Args:
        results_dir: Directory containing *_primers.csv files
        top_n: Only include top N primers per virus (None = all)
        only_specific: Only include specific primers (original behavior)

    Returns:
        Dict of virus -> count of primers exported
    """
    stats = {}

    # Find all primer CSV files
    csv_files = list(results_dir.glob("*_primers.csv"))

    if not csv_files:
        print(f"No primer CSV files found in {results_dir}")
        return stats

    print(f"Found {len(csv_files)} primer CSV files\n")

    for csv_path in sorted(csv_files):
        virus = csv_path.stem.replace("_primers", "")
        fasta_path = csv_path.with_suffix(".fasta")

        # Read CSV
        primers = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                primers.append(row)

        # Filter if only_specific
        if only_specific:
            primers = [p for p in primers if p.get("is_specific", "").lower() == "true"]

        # Limit to top_n
        if top_n is not None:
            primers = primers[:top_n]

        # Write FASTA
        with open(fasta_path, "w") as f:
            for i, p in enumerate(primers, 1):
                rank = int(p.get("rank", i))
                position = p.get("position", "?")
                sequence = p.get("sequence", "")
                is_specific = p.get("is_specific", "").lower() == "true"
                score = p.get("combined_score", "?")

                # Build header with metadata
                specificity_tag = "specific" if is_specific else "cross-reactive"
                header = f">{virus}_primer_{rank:02d}_pos{position}_{specificity_tag}_score{score}"
                f.write(f"{header}\n{sequence}\n")

        stats[virus] = len(primers)
        print(f"  {virus}: {len(primers)} primers -> {fasta_path.name}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate FASTA files from primer CSVs"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent.parent / "results" / "pan_arbovirus_primers",
        help="Directory containing primer CSV files",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Only export top N primers per virus",
    )
    parser.add_argument(
        "--only-specific",
        action="store_true",
        help="Only export specific primers (original behavior)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("REGENERATING FASTA FILES FROM CSV DATA")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Top N: {args.top or 'all'}")
    print(f"Only specific: {args.only_specific}")
    print()

    stats = regenerate_fastas(
        args.results_dir,
        top_n=args.top,
        only_specific=args.only_specific,
    )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = sum(stats.values())
    print(f"Total primers exported: {total}")
    print(f"Viruses processed: {len(stats)}")

    if not args.only_specific:
        print("\nNote: All primers exported (including cross-reactive).")
        print("Headers indicate specificity status.")


if __name__ == "__main__":
    main()
