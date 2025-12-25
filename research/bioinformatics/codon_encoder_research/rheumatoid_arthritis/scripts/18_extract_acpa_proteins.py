#!/usr/bin/env python3
"""
Extract ACPA Target Proteins for Comprehensive PTM Analysis

This script fetches all 10 major ACPA (Anti-Citrullinated Protein Antibody) target
proteins from UniProt and prepares them for comprehensive PTM sweep analysis.

Part of Phase 1: RA Extensions (PRIORITY)
See: research/genetic_code/PTM_EXTENSION_PLAN.md

Output: research/bioinformatics/rheumatoid_arthritis/data/acpa_proteins.json
"""

import json
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# =============================================================================
# ACPA TARGET PROTEINS
# =============================================================================

ACPA_TARGETS = [
    {
        "name": "Fibrinogen alpha",
        "uniprot": "P02671",
        "gene": "FGA",
        "known_cit_sites": [
            271,
            573,
        ],  # Literature-validated citrullination sites
        "acpa_association": "Strong",
        "notes": "Major ACPA target, forms fibrin clots",
    },
    {
        "name": "Fibrinogen beta",
        "uniprot": "P02675",
        "gene": "FGB",
        "known_cit_sites": [72],
        "acpa_association": "Strong",
        "notes": "Part of fibrinogen complex",
    },
    {
        "name": "Fibrinogen gamma",
        "uniprot": "P02679",
        "gene": "FGG",
        "known_cit_sites": [],  # Multiple sites, need literature review
        "acpa_association": "Strong",
        "notes": "Part of fibrinogen complex",
    },
    {
        "name": "Vimentin",
        "uniprot": "P08670",
        "gene": "VIM",
        "known_cit_sites": [71, 38],
        "acpa_association": "Strong",
        "notes": "Intermediate filament, Sa antigen",
    },
    {
        "name": "Alpha-enolase",
        "uniprot": "P06733",
        "gene": "ENO1",
        "known_cit_sites": [9, 15],
        "acpa_association": "Strong",
        "notes": "CEP-1 epitope, glycolytic enzyme",
    },
    {
        "name": "Collagen type II alpha 1",
        "uniprot": "P02458",
        "gene": "COL2A1",
        "known_cit_sites": [],  # Multiple sites in triple helix
        "acpa_association": "Strong",
        "notes": "Cartilage component, joint destruction",
    },
    {
        "name": "Histone H2B type 1-K",
        "uniprot": "O60814",
        "gene": "HIST1H2BK",
        "known_cit_sites": [29],
        "acpa_association": "Moderate",
        "notes": "NETs-associated, nuclear antigen",
    },
    {
        "name": "Histone H4",
        "uniprot": "P62805",
        "gene": "HIST1H4A",
        "known_cit_sites": [3],
        "acpa_association": "Moderate",
        "notes": "NETs-associated, highly conserved",
    },
    {
        "name": "Filaggrin",
        "uniprot": "P20930",
        "gene": "FLG",
        "known_cit_sites": [],  # Many sites, original CCP antigen
        "acpa_association": "Strong",
        "notes": "Original CCP test antigen, skin protein",
    },
    {
        "name": "hnRNP A2/B1",
        "uniprot": "P22626",
        "gene": "HNRNPA2B1",
        "known_cit_sites": [],
        "acpa_association": "Moderate",
        "notes": "RA33 antigen, RNA-binding protein",
    },
]

# Modifiable residues for PTM analysis
MODIFIABLE_RESIDUES = {
    "R": "Citrullination (R->Q)",
    "S": "Phosphoserine (S->D)",
    "T": "Phosphothreonine (T->D)",
    "Y": "Phosphotyrosine (Y->D)",
    "N": "Deglycosylation (N->Q)",
    "K": "Acetylation (K->Q)",
    "M": "Oxidation (M->Q)",
}


# =============================================================================
# UNIPROT FETCHING
# =============================================================================


def fetch_uniprot_sequence(uniprot_id: str, max_retries: int = 3) -> dict:
    """Fetch protein sequence and metadata from UniProt."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Python/research"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

                # Extract sequence
                sequence = data.get("sequence", {}).get("value", "")
                length = data.get("sequence", {}).get("length", 0)

                # Extract protein name
                protein_name = data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")

                # Extract organism
                organism = data.get("organism", {}).get("scientificName", "")

                return {
                    "uniprot_id": uniprot_id,
                    "sequence": sequence,
                    "length": length,
                    "protein_name": protein_name,
                    "organism": organism,
                    "success": True,
                }

        except urllib.error.HTTPError as e:
            print(f"  HTTP Error {e.code} for {uniprot_id}, attempt {attempt + 1}/{max_retries}")
            time.sleep(1)
        except urllib.error.URLError as e:
            print(f"  URL Error for {uniprot_id}: {e.reason}, attempt {attempt + 1}/{max_retries}")
            time.sleep(1)
        except Exception as e:
            print(f"  Error for {uniprot_id}: {e}, attempt {attempt + 1}/{max_retries}")
            time.sleep(1)

    return {
        "uniprot_id": uniprot_id,
        "success": False,
        "error": "Max retries exceeded",
    }


def extract_context(sequence: str, position: int, window: int = 5) -> str:
    """Extract context window around a position (0-indexed)."""
    start = max(0, position - window)
    end = min(len(sequence), position + window + 1)

    # Pad with '-' if at sequence boundaries
    left_pad = "-" * (window - (position - start))
    right_pad = "-" * (window - (end - position - 1))

    context = left_pad + sequence[start:end] + right_pad
    return context


def identify_modifiable_sites(sequence: str) -> dict:
    """Identify all modifiable residues in a protein sequence."""
    sites = defaultdict(list)

    for pos, residue in enumerate(sequence):
        if residue in MODIFIABLE_RESIDUES:
            sites[residue].append(
                {
                    "position": pos + 1,  # 1-indexed for biology convention
                    "position_0idx": pos,
                    "residue": residue,
                    "context_11mer": extract_context(sequence, pos, window=5),
                    "context_15mer": extract_context(sequence, pos, window=7),
                    "ptm_type": MODIFIABLE_RESIDUES[residue],
                }
            )

    return dict(sites)


def count_modifiable_sites(sites: dict) -> dict:
    """Count modifiable sites by residue type."""
    return {residue: len(positions) for residue, positions in sites.items()}


# =============================================================================
# MAIN EXTRACTION
# =============================================================================


def extract_all_acpa_proteins():
    """Extract all ACPA target proteins and their modifiable sites."""
    print("=" * 70)
    print("ACPA TARGET PROTEIN EXTRACTION")
    print("Phase 1: RA Extensions (PRIORITY)")
    print("=" * 70)

    results = {
        "metadata": {
            "extraction_date": datetime.now().isoformat(),
            "source": "UniProt",
            "purpose": "Comprehensive PTM analysis for RA ground truth",
            "modifiable_residues": MODIFIABLE_RESIDUES,
        },
        "proteins": [],
        "summary": {
            "total_proteins": 0,
            "total_residues": 0,
            "total_modifiable_sites": 0,
            "sites_by_residue": defaultdict(int),
            "known_cit_sites": 0,
        },
    }

    for target in ACPA_TARGETS:
        print(f"\nFetching {target['name']} ({target['uniprot']})...")

        # Fetch from UniProt
        uniprot_data = fetch_uniprot_sequence(target["uniprot"])

        if not uniprot_data.get("success"):
            print(f"  ERROR: Failed to fetch {target['uniprot']}")
            continue

        sequence = uniprot_data["sequence"]
        length = len(sequence)

        print(f"  Length: {length} aa")

        # Identify modifiable sites
        modifiable_sites = identify_modifiable_sites(sequence)
        site_counts = count_modifiable_sites(modifiable_sites)
        total_sites = sum(site_counts.values())

        print(f"  Modifiable sites: {total_sites}")
        for res, count in sorted(site_counts.items()):
            print(f"    {res}: {count}")

        # Mark known citrullination sites
        known_cit_sites = target.get("known_cit_sites", [])
        for site in modifiable_sites.get("R", []):
            site["is_known_acpa"] = site["position"] in known_cit_sites

        # Build protein entry
        protein_entry = {
            "name": target["name"],
            "uniprot": target["uniprot"],
            "gene": target["gene"],
            "sequence": sequence,
            "length": length,
            "organism": uniprot_data.get("organism", "Homo sapiens"),
            "acpa_association": target["acpa_association"],
            "notes": target["notes"],
            "known_cit_sites": known_cit_sites,
            "modifiable_sites": modifiable_sites,
            "site_counts": site_counts,
            "total_modifiable_sites": total_sites,
        }

        results["proteins"].append(protein_entry)

        # Update summary
        results["summary"]["total_proteins"] += 1
        results["summary"]["total_residues"] += length
        results["summary"]["total_modifiable_sites"] += total_sites
        results["summary"]["known_cit_sites"] += len(known_cit_sites)
        for res, count in site_counts.items():
            results["summary"]["sites_by_residue"][res] += count

    # Convert defaultdict to dict for JSON serialization
    results["summary"]["sites_by_residue"] = dict(results["summary"]["sites_by_residue"])

    return results


def main():
    # Extract proteins
    results = extract_all_acpa_proteins()

    # Summary
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    summary = results["summary"]
    print(f"  Proteins extracted: {summary['total_proteins']}/{len(ACPA_TARGETS)}")
    print(f"  Total residues: {summary['total_residues']}")
    print(f"  Total modifiable sites: {summary['total_modifiable_sites']}")
    print(f"  Known citrullination sites: {summary['known_cit_sites']}")
    print("\n  Sites by residue type:")
    for res, count in sorted(summary["sites_by_residue"].items()):
        print(f"    {res} ({MODIFIABLE_RESIDUES[res]}): {count}")

    # Estimate PTM sweep size
    total_ptm_samples = summary["total_modifiable_sites"]  # Each site gets 1 PTM type
    print(f"\n  Estimated PTM samples: {total_ptm_samples}")

    # Save results
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "acpa_proteins.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {output_path}")

    # Also save a simplified version for quick reference
    quick_ref = {
        "proteins": [
            {
                "name": p["name"],
                "uniprot": p["uniprot"],
                "length": p["length"],
                "total_sites": p["total_modifiable_sites"],
                "known_cit": p["known_cit_sites"],
            }
            for p in results["proteins"]
        ],
        "total_sites": summary["total_modifiable_sites"],
    }

    quick_path = output_dir / "acpa_proteins_summary.json"
    with open(quick_path, "w") as f:
        json.dump(quick_ref, f, indent=2)
    print(f"  Saved: {quick_path}")

    print("\n" + "=" * 70)
    print("NEXT STEP: Run 19_comprehensive_ra_ptm_sweep.py")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
