#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Download minimal PDB subset for HIV integrase validation.

This script downloads only the structures needed for validating reveal mutations
in HIV integrase (~100MB instead of the full 50GB PDB archive).

Usage:
    python download_integrase_structures.py [--output-dir PATH]
"""

import argparse
import gzip
import sys
from pathlib import Path
from typing import NamedTuple

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)


class PDBEntry(NamedTuple):
    """PDB entry with metadata."""

    pdb_id: str
    description: str
    category: str


# Curated list of HIV integrase-related structures
INTEGRASE_STRUCTURES = [
    # HIV-1 Integrase catalytic domain structures
    PDBEntry("1EX4", "HIV-1 IN catalytic domain", "core"),
    PDBEntry("1BL3", "HIV-1 IN catalytic domain with Mg2+", "core"),
    PDBEntry("1BIS", "HIV-1 IN F185K mutant", "core"),
    PDBEntry("1BIU", "HIV-1 IN W131E mutant", "core"),
    PDBEntry("1BIZ", "HIV-1 IN F185K/C280S mutant", "core"),
    # HIV-1 Integrase-LEDGF complexes (critical for reveal mutations)
    PDBEntry("2B4J", "HIV-1 IN CCD with LEDGF IBD", "ledgf"),
    PDBEntry("3LPU", "HIV-1 IN with LEDGF", "ledgf"),
    PDBEntry("3LPT", "HIV-1 IN with LEDGF variant", "ledgf"),
    PDBEntry("4E7I", "HIV-1 IN with LEDGF and raltegravir", "ledgf"),
    PDBEntry("4E7K", "HIV-1 IN with LEDGF and elvitegravir", "ledgf"),
    # HIV-1 Intasome structures (DNA-bound)
    PDBEntry("3OYA", "HIV-1 IN intasome", "intasome"),
    PDBEntry("5U1C", "HIV-1 IN strand transfer complex", "intasome"),
    PDBEntry("6PUT", "HIV-1 IN intasome with DNA", "intasome"),
    PDBEntry("6PUW", "HIV-1 IN intasome variant", "intasome"),
    PDBEntry("6PUY", "HIV-1 IN intasome with target DNA", "intasome"),
    # Drug resistance structures
    PDBEntry("3OYM", "HIV-1 IN Y143R with raltegravir", "resistance"),
    PDBEntry("3OYN", "HIV-1 IN N155H with raltegravir", "resistance"),
    PDBEntry("3L2T", "HIV-1 IN G140S/Q148H", "resistance"),
    PDBEntry("3L2U", "HIV-1 IN E92Q", "resistance"),
    PDBEntry("3L2V", "HIV-1 IN with MK-2048", "resistance"),
    # LEDGF/p75 alone (host factor)
    PDBEntry("2B4L", "LEDGF IBD domain", "host"),
    PDBEntry("3HPH", "LEDGF PWWP domain", "host"),
    # Related retroviral integrases (evolutionary context)
    PDBEntry("1C6V", "ASV integrase", "related"),
    PDBEntry("3OS0", "PFV intasome", "related"),
    PDBEntry("3OS1", "PFV intasome with strand transfer", "related"),
    PDBEntry("4E7L", "MVV integrase", "related"),
]


def download_structure(pdb_id: str, output_dir: Path, format: str = "cif") -> bool:
    """Download a single PDB structure.

    Args:
        pdb_id: PDB identifier (e.g., "2B4J")
        output_dir: Directory to save the file
        format: File format ("cif" or "pdb")

    Returns:
        True if successful, False otherwise
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.{format}.gz"
    output_path = output_dir / f"{pdb_id}.{format}.gz"

    if output_path.exists():
        print(f"  [SKIP] {pdb_id} already exists")
        return True

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            # Verify it's valid gzip
            try:
                with gzip.open(output_path, "rb") as f:
                    f.read(100)  # Read first 100 bytes to verify
                print(f"  [OK] {pdb_id}")
                return True
            except gzip.BadGzipFile:
                output_path.unlink()
                print(f"  [FAIL] {pdb_id} - Invalid gzip")
                return False
        else:
            print(f"  [FAIL] {pdb_id} - HTTP {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"  [FAIL] {pdb_id} - {e}")
        return False


def download_all(output_dir: Path, format: str = "cif") -> dict:
    """Download all integrase structures.

    Args:
        output_dir: Directory to save files
        format: File format ("cif" or "pdb")

    Returns:
        Dictionary with download statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"success": 0, "failed": 0, "skipped": 0}
    categories = {}

    print(f"\nDownloading {len(INTEGRASE_STRUCTURES)} structures to {output_dir}\n")

    for entry in INTEGRASE_STRUCTURES:
        if entry.category not in categories:
            categories[entry.category] = []
            print(f"\n=== {entry.category.upper()} ===")

        categories[entry.category].append(entry.pdb_id)

        output_path = output_dir / f"{entry.pdb_id}.{format}.gz"
        if output_path.exists():
            stats["skipped"] += 1
            print(f"  [SKIP] {entry.pdb_id} - {entry.description}")
        elif download_structure(entry.pdb_id, output_dir, format):
            stats["success"] += 1
        else:
            stats["failed"] += 1

    return stats


def create_manifest(output_dir: Path) -> None:
    """Create a manifest file listing all downloaded structures."""
    manifest_path = output_dir / "MANIFEST.txt"

    with open(manifest_path, "w") as f:
        f.write("# HIV Integrase PDB Structure Manifest\n")
        f.write("# Generated by download_integrase_structures.py\n")
        f.write("#\n")
        f.write("# Format: PDB_ID | Category | Description\n")
        f.write("#\n\n")

        current_category = None
        for entry in INTEGRASE_STRUCTURES:
            if entry.category != current_category:
                current_category = entry.category
                f.write(f"\n## {current_category.upper()}\n")
            f.write(f"{entry.pdb_id} | {entry.category} | {entry.description}\n")

    print(f"\nManifest written to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download minimal PDB subset for HIV integrase validation")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "pdb",
        help="Output directory for PDB files",
    )
    parser.add_argument(
        "--format",
        choices=["cif", "pdb"],
        default="cif",
        help="Structure file format (default: cif)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List structures without downloading",
    )

    args = parser.parse_args()

    if args.dry_run:
        print("\n=== HIV Integrase Structure Set ===\n")
        for entry in INTEGRASE_STRUCTURES:
            print(f"{entry.pdb_id}: {entry.description} [{entry.category}]")
        print(f"\nTotal: {len(INTEGRASE_STRUCTURES)} structures")
        print("Estimated size: ~100MB compressed")
        return

    stats = download_all(args.output_dir, args.format)
    create_manifest(args.output_dir)

    print("\n=== Download Summary ===")
    print(f"  Success: {stats['success']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed:  {stats['failed']}")
    print(f"  Total:   {len(INTEGRASE_STRUCTURES)}")


if __name__ == "__main__":
    main()
