#!/usr/bin/env python3
"""
Download Human Proteome from UniProt

Downloads the complete reviewed human proteome (Swiss-Prot) including:
- Protein sequences (FASTA)
- Metadata (gene names, GO terms, subcellular location)

Output directory: results/proteome_wide/12_human_proteome/

Version: 1.0
"""

import json
from pathlib import Path
from typing import Dict

import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

# UniProt REST API endpoints
UNIPROT_API = "https://rest.uniprot.org/uniprotkb"

# Query for reviewed human proteins (Swiss-Prot)
HUMAN_PROTEOME_QUERY = "(organism_id:9606) AND (reviewed:true)"

# Fields to download
METADATA_FIELDS = [
    "accession",
    "id",  # Entry name
    "gene_names",
    "protein_name",
    "length",
    "sequence",
    "go_c",  # Cellular component
    "go_f",  # Molecular function
    "go_p",  # Biological process
    "cc_subcellular_location",
    "cc_tissue_specificity",
    "cc_disease",
    "ft_domain",
    "ft_region",
    "xref_pdb",  # Structure availability
]

# Output configuration
SCRIPT_NUM = "12"
OUTPUT_SUBDIR = f"{SCRIPT_NUM}_human_proteome"


# ============================================================================
# OUTPUT DIRECTORY SETUP
# ============================================================================


def get_output_dir() -> Path:
    """Get output directory for this script."""
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / "proteome_wide" / OUTPUT_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


# ============================================================================
# UNIPROT DOWNLOAD FUNCTIONS
# ============================================================================


def download_proteome_fasta(output_dir: Path) -> Path:
    """
    Download human proteome in FASTA format.

    Returns path to downloaded file.
    """
    print("\n[1] Downloading FASTA sequences...")

    url = f"{UNIPROT_API}/stream"
    params = {
        "query": HUMAN_PROTEOME_QUERY,
        "format": "fasta",
        "compressed": "true",
    }

    output_path = output_dir / "human_proteome.fasta.gz"

    # Check if already downloaded
    if output_path.exists():
        print(f"  Found existing file: {output_path}")
        print("  Skipping download (delete file to re-download)")
        return output_path

    print(f"  URL: {url}")
    print(f"  Query: {HUMAN_PROTEOME_QUERY}")

    response = requests.get(url, params=params, stream=True)
    response.raise_for_status()

    # Get total size if available
    total_size = int(response.headers.get("content-length", 0))

    # Download with progress
    downloaded = 0
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = 100 * downloaded / total_size
                print(
                    f"\r  Progress: {downloaded:,} / {total_size:,} bytes ({pct:.1f}%)",
                    end="",
                )

    print(f"\n  Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size:,} bytes")

    return output_path


def download_proteome_metadata(output_dir: Path) -> Path:
    """
    Download human proteome metadata in TSV format.

    Returns path to downloaded file.
    """
    print("\n[2] Downloading metadata...")

    url = f"{UNIPROT_API}/stream"
    params = {
        "query": HUMAN_PROTEOME_QUERY,
        "format": "tsv",
        "fields": ",".join(METADATA_FIELDS),
    }

    output_path = output_dir / "human_proteome_metadata.tsv"

    # Check if already downloaded
    if output_path.exists():
        print(f"  Found existing file: {output_path}")
        print("  Skipping download (delete file to re-download)")
        return output_path

    print(f"  Fields: {len(METADATA_FIELDS)}")

    response = requests.get(url, params=params, stream=True)
    response.raise_for_status()

    # Download
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"  Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size:,} bytes")

    return output_path


# ============================================================================
# PARSING FUNCTIONS
# ============================================================================


def parse_fasta(fasta_path: Path) -> Dict[str, Dict]:
    """
    Parse FASTA file into dictionary.

    Returns:
        Dict mapping accession -> {sequence, description, length}
    """
    print("\n[3] Parsing FASTA file...")

    proteins = {}
    current_acc = None
    current_seq = []
    current_desc = ""

    # Handle gzipped or plain
    if str(fasta_path).endswith(".gz"):
        import gzip

        opener = lambda p: gzip.open(p, "rt")
    else:
        opener = lambda p: open(p, "r")

    with opener(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # Save previous protein
                if current_acc:
                    seq = "".join(current_seq)
                    proteins[current_acc] = {
                        "sequence": seq,
                        "length": len(seq),
                        "description": current_desc,
                        "n_arginines": seq.count("R"),
                    }

                # Parse new header
                # Format: >sp|P12345|NAME_HUMAN Description OS=...
                parts = line[1:].split("|")
                if len(parts) >= 2:
                    current_acc = parts[1]
                    current_desc = parts[2] if len(parts) > 2 else ""
                else:
                    current_acc = parts[0].split()[0]
                    current_desc = line[1:]

                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget last protein
        if current_acc:
            seq = "".join(current_seq)
            proteins[current_acc] = {
                "sequence": seq,
                "length": len(seq),
                "description": current_desc,
                "n_arginines": seq.count("R"),
            }

    print(f"  Parsed {len(proteins):,} proteins")

    # Statistics
    total_length = sum(p["length"] for p in proteins.values())
    total_arginines = sum(p["n_arginines"] for p in proteins.values())

    print(f"  Total residues: {total_length:,}")
    print(f"  Total arginines: {total_arginines:,}")
    print(f"  Mean protein length: {total_length/len(proteins):.1f}")
    print(f"  Mean arginines per protein: {total_arginines/len(proteins):.1f}")

    return proteins


def parse_metadata(tsv_path: Path) -> Dict[str, Dict]:
    """
    Parse TSV metadata file.

    Returns:
        Dict mapping accession -> metadata dict
    """
    print("\n[4] Parsing metadata...")

    metadata = {}

    with open(tsv_path, "r", encoding="utf-8") as f:
        # Read header
        header = f.readline().strip().split("\t")

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            row = dict(zip(header, parts))
            acc = row.get("Entry", row.get("accession", ""))

            if acc:
                metadata[acc] = {
                    "accession": acc,
                    "entry_name": row.get("Entry Name", ""),
                    "gene_names": row.get("Gene Names", ""),
                    "protein_name": row.get("Protein names", ""),
                    "length": (int(row.get("Length", 0)) if row.get("Length", "").isdigit() else 0),
                    "go_cellular_component": row.get("Gene Ontology (cellular component)", ""),
                    "go_molecular_function": row.get("Gene Ontology (molecular function)", ""),
                    "go_biological_process": row.get("Gene Ontology (biological process)", ""),
                    "subcellular_location": row.get("Subcellular location [CC]", ""),
                    "tissue_specificity": row.get("Tissue specificity", ""),
                    "disease": row.get("Involvement in disease", ""),
                    "domains": row.get("Domain [FT]", ""),
                    "regions": row.get("Region", ""),
                    "pdb_structures": row.get("PDB", ""),
                }

    print(f"  Parsed metadata for {len(metadata):,} proteins")

    return metadata


def merge_data(proteins: Dict, metadata: Dict) -> Dict:
    """Merge sequence data with metadata."""
    print("\n[5] Merging sequence and metadata...")

    merged = {}
    matched = 0

    for acc, prot in proteins.items():
        merged[acc] = prot.copy()

        if acc in metadata:
            merged[acc].update(metadata[acc])
            matched += 1
        else:
            # Add empty metadata fields
            merged[acc].update(
                {
                    "entry_name": "",
                    "gene_names": "",
                    "protein_name": prot.get("description", ""),
                    "go_cellular_component": "",
                    "go_molecular_function": "",
                    "go_biological_process": "",
                    "subcellular_location": "",
                    "tissue_specificity": "",
                    "disease": "",
                    "domains": "",
                    "regions": "",
                    "pdb_structures": "",
                }
            )

    print(f"  Matched {matched:,} / {len(proteins):,} proteins with metadata")

    return merged


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================


def compute_statistics(data: Dict, output_dir: Path):
    """Compute and save summary statistics."""
    print("\n[6] Computing statistics...")

    stats = {
        "total_proteins": len(data),
        "total_residues": sum(p["length"] for p in data.values()),
        "total_arginines": sum(p["n_arginines"] for p in data.values()),
        "proteins_with_arginine": sum(1 for p in data.values() if p["n_arginines"] > 0),
        "proteins_with_go": sum(1 for p in data.values() if p.get("go_biological_process")),
        "proteins_with_disease": sum(1 for p in data.values() if p.get("disease")),
        "proteins_with_structure": sum(1 for p in data.values() if p.get("pdb_structures")),
    }

    # Length distribution
    lengths = [p["length"] for p in data.values()]
    stats["length_min"] = min(lengths)
    stats["length_max"] = max(lengths)
    stats["length_mean"] = sum(lengths) / len(lengths)
    stats["length_median"] = sorted(lengths)[len(lengths) // 2]

    # Arginine distribution
    arg_counts = [p["n_arginines"] for p in data.values()]
    stats["arginines_min"] = min(arg_counts)
    stats["arginines_max"] = max(arg_counts)
    stats["arginines_mean"] = sum(arg_counts) / len(arg_counts)

    # Save statistics
    stats_path = output_dir / "proteome_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Saved: {stats_path}")

    # Print summary
    print("\n  Summary:")
    print(f"    Proteins: {stats['total_proteins']:,}")
    print(f"    Total residues: {stats['total_residues']:,}")
    print(f"    Total arginines: {stats['total_arginines']:,}")
    print(f"    Proteins with R: {stats['proteins_with_arginine']:,}")
    print(f"    Mean length: {stats['length_mean']:.1f}")
    print(f"    Mean arginines: {stats['arginines_mean']:.1f}")

    return stats


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 80)
    print("DOWNLOAD HUMAN PROTEOME")
    print("UniProt reviewed human proteins (Swiss-Prot)")
    print("=" * 80)

    # Setup output directory
    output_dir = get_output_dir()
    print(f"\nOutput directory: {output_dir}")

    # Download FASTA
    fasta_path = download_proteome_fasta(output_dir)

    # Download metadata
    metadata_path = download_proteome_metadata(output_dir)

    # Parse FASTA
    proteins = parse_fasta(fasta_path)

    # Parse metadata
    metadata = parse_metadata(metadata_path)

    # Merge data
    data = merge_data(proteins, metadata)

    # Save merged data (without sequences to save space)
    print("\n[7] Saving processed data...")

    # Save full data with sequences
    full_path = output_dir / "human_proteome_full.json"
    with open(full_path, "w") as f:
        json.dump(data, f)
    print(f"  Saved: {full_path} ({full_path.stat().st_size:,} bytes)")

    # Save index (without sequences) for quick loading
    index = {acc: {k: v for k, v in prot.items() if k != "sequence"} for acc, prot in data.items()}
    index_path = output_dir / "human_proteome_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Saved: {index_path} ({index_path.stat().st_size:,} bytes)")

    # Compute statistics
    stats = compute_statistics(data, output_dir)

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE")
    print(f"Output: {output_dir}")
    print("=" * 80)

    print("\nFiles created:")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")

    return data


if __name__ == "__main__":
    main()
