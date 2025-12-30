# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""NCBI Arbovirus Sequence Loader.

Downloads and caches arbovirus sequences from NCBI for primer design.
Supports Dengue (1-4), Zika, Chikungunya, and Mayaro viruses.

Usage:
    python ncbi_arbovirus_loader.py --download
    python ncbi_arbovirus_loader.py --list
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import time
from typing import Optional
from dataclasses import dataclass, field, asdict

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "deliverables"))

from shared.config import get_config
from shared.constants import ARBOVIRUS_TAXIDS

# Try to import BioPython
try:
    from Bio import Entrez, SeqIO
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("BioPython not available - will use demo sequences")


@dataclass
class VirusSequence:
    """Container for a viral sequence."""
    accession: str
    virus: str
    description: str
    sequence: str
    length: int
    country: Optional[str] = None
    collection_date: Optional[str] = None
    strain: Optional[str] = None

    def to_fasta(self) -> str:
        """Convert to FASTA format."""
        return f">{self.accession} {self.description}\n{self.sequence}\n"


@dataclass
class ArbovirusDatabase:
    """Database of arbovirus sequences."""
    sequences: dict[str, list[VirusSequence]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def add_sequence(self, virus: str, seq: VirusSequence):
        """Add a sequence for a virus."""
        if virus not in self.sequences:
            self.sequences[virus] = []
        self.sequences[virus].append(seq)

    def get_sequences(self, virus: str) -> list[VirusSequence]:
        """Get all sequences for a virus."""
        return self.sequences.get(virus, [])

    def count(self, virus: str = None) -> int:
        """Count sequences."""
        if virus:
            return len(self.sequences.get(virus, []))
        return sum(len(seqs) for seqs in self.sequences.values())

    def save(self, path: Path):
        """Save database to JSON."""
        data = {
            "metadata": self.metadata,
            "sequences": {
                virus: [asdict(s) for s in seqs]
                for virus, seqs in self.sequences.items()
            }
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ArbovirusDatabase":
        """Load database from JSON."""
        with open(path) as f:
            data = json.load(f)

        db = cls(metadata=data.get("metadata", {}))
        for virus, seqs in data.get("sequences", {}).items():
            for seq_data in seqs:
                db.add_sequence(virus, VirusSequence(**seq_data))
        return db


class NCBIArbovirusLoader:
    """Load arbovirus sequences from NCBI."""

    def __init__(self, email: str = None, api_key: str = None):
        self.config = get_config()
        self.email = email or self.config.ncbi_email
        self.api_key = api_key or self.config.ncbi_api_key
        self.cache_dir = self.config.get_partner_dir("alejandra") / "data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if BIOPYTHON_AVAILABLE:
            Entrez.email = self.email
            if self.api_key:
                Entrez.api_key = self.api_key

    def download_virus(
        self,
        virus: str,
        max_sequences: int = 100,
        region_filter: str = None,
    ) -> list[VirusSequence]:
        """Download sequences for a specific virus from NCBI.

        Args:
            virus: Virus name (e.g., "DENV-1", "ZIKV")
            max_sequences: Maximum sequences to download
            region_filter: Optional geographic filter (e.g., "Paraguay")

        Returns:
            List of VirusSequence objects
        """
        if not BIOPYTHON_AVAILABLE:
            return self._generate_demo_sequences(virus, max_sequences)

        taxid = ARBOVIRUS_TAXIDS.get(virus)
        if taxid is None:
            print(f"Unknown virus: {virus}")
            return []

        # Build search query
        query = f"txid{taxid}[Organism] AND complete genome[Title]"
        if region_filter:
            query += f" AND {region_filter}[All Fields]"

        try:
            # Search NCBI
            print(f"Searching NCBI for {virus}...")
            handle = Entrez.esearch(
                db="nucleotide",
                term=query,
                retmax=max_sequences,
                usehistory="y"
            )
            record = Entrez.read(handle)
            handle.close()

            count = int(record["Count"])
            if count == 0:
                print(f"No sequences found for {virus}")
                return self._generate_demo_sequences(virus, min(10, max_sequences))

            print(f"Found {count} sequences for {virus}")

            # Fetch sequences
            sequences = []
            webenv = record["WebEnv"]
            query_key = record["QueryKey"]

            batch_size = 50
            for start in range(0, min(count, max_sequences), batch_size):
                print(f"  Downloading {start + 1} to {min(start + batch_size, count)}...")

                fetch_handle = Entrez.efetch(
                    db="nucleotide",
                    rettype="gb",
                    retmode="text",
                    retstart=start,
                    retmax=batch_size,
                    webenv=webenv,
                    query_key=query_key
                )

                for record in SeqIO.parse(fetch_handle, "genbank"):
                    seq = VirusSequence(
                        accession=record.id,
                        virus=virus,
                        description=record.description,
                        sequence=str(record.seq),
                        length=len(record.seq),
                        country=self._extract_feature(record, "country"),
                        collection_date=self._extract_feature(record, "collection_date"),
                        strain=self._extract_feature(record, "strain"),
                    )
                    sequences.append(seq)

                fetch_handle.close()
                time.sleep(0.5)  # Be nice to NCBI

            return sequences

        except Exception as e:
            print(f"Error downloading {virus}: {e}")
            return self._generate_demo_sequences(virus, min(10, max_sequences))

    def _extract_feature(self, record, feature_name: str) -> Optional[str]:
        """Extract a feature from a GenBank record."""
        for feature in record.features:
            if feature.type == "source":
                qualifiers = feature.qualifiers
                if feature_name in qualifiers:
                    return qualifiers[feature_name][0]
        return None

    def _generate_demo_sequences(self, virus: str, n: int) -> list[VirusSequence]:
        """Generate demo sequences for testing."""
        import random
        random.seed(hash(virus) % (2**31))

        # Approximate genome sizes
        genome_sizes = {
            "DENV-1": 10700, "DENV-2": 10700, "DENV-3": 10700, "DENV-4": 10700,
            "ZIKV": 10800, "CHIKV": 11800, "MAYV": 11400,
        }
        size = genome_sizes.get(virus, 11000)

        sequences = []
        for i in range(n):
            # Generate random sequence
            seq = "".join(random.choices("ACGT", k=size))

            # Add conserved regions for primer targeting
            conserved_regions = [
                (100, "ATGAACAACCAACGGAAAAAGACGG"),  # 5' UTR region
                (500, "GGACTAGAGGTTAGAGGAGACC"),  # NS1 region
                (5000, "TGGATGACACGGAAAGACATG"),  # NS3 region
                (9000, "AGACCCATGGATTTCCTTAC"),  # NS5 region
            ]

            seq_list = list(seq)
            for pos, conserved in conserved_regions:
                if pos + len(conserved) < len(seq_list):
                    for j, nt in enumerate(conserved):
                        seq_list[pos + j] = nt
            seq = "".join(seq_list)

            sequences.append(VirusSequence(
                accession=f"{virus}_DEMO_{i + 1:03d}",
                virus=virus,
                description=f"Demo {virus} complete genome isolate {i + 1}",
                sequence=seq,
                length=len(seq),
                country="Paraguay" if random.random() > 0.3 else "Brazil",
                collection_date=f"202{random.randint(0, 4)}",
                strain=f"Demo-{i + 1}",
            ))

        return sequences

    def download_all(self, max_per_virus: int = 50) -> ArbovirusDatabase:
        """Download sequences for all arboviruses.

        Args:
            max_per_virus: Maximum sequences per virus

        Returns:
            ArbovirusDatabase with all sequences
        """
        db = ArbovirusDatabase(metadata={
            "download_date": time.strftime("%Y-%m-%d"),
            "source": "NCBI" if BIOPYTHON_AVAILABLE else "Demo",
            "max_per_virus": max_per_virus,
        })

        for virus in ARBOVIRUS_TAXIDS.keys():
            sequences = self.download_virus(virus, max_per_virus)
            for seq in sequences:
                db.add_sequence(virus, seq)
            print(f"  {virus}: {len(sequences)} sequences")

        return db

    def load_or_download(
        self,
        cache_name: str = "arbovirus_db.json",
        max_per_virus: int = 50,
        force_download: bool = False,
    ) -> ArbovirusDatabase:
        """Load from cache or download if needed.

        Args:
            cache_name: Name of cache file
            max_per_virus: Maximum sequences per virus (if downloading)
            force_download: Force re-download even if cache exists

        Returns:
            ArbovirusDatabase
        """
        cache_path = self.cache_dir / cache_name

        if cache_path.exists() and not force_download:
            print(f"Loading cached database from {cache_path}")
            return ArbovirusDatabase.load(cache_path)

        print("Downloading arbovirus sequences...")
        db = self.download_all(max_per_virus)

        print(f"Saving to cache: {cache_path}")
        db.save(cache_path)

        return db

    def export_fasta(self, db: ArbovirusDatabase, output_dir: Path = None):
        """Export database to FASTA files (one per virus).

        Args:
            db: ArbovirusDatabase to export
            output_dir: Output directory (default: cache_dir)
        """
        output_dir = output_dir or self.cache_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        for virus, sequences in db.sequences.items():
            fasta_path = output_dir / f"{virus}_genomes.fasta"
            with open(fasta_path, "w") as f:
                for seq in sequences:
                    f.write(seq.to_fasta())
            print(f"Exported {len(sequences)} sequences to {fasta_path}")


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Download arbovirus sequences from NCBI")
    parser.add_argument("--download", action="store_true", help="Download sequences")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--max-per-virus", type=int, default=50, help="Max sequences per virus")
    parser.add_argument("--list", action="store_true", help="List cached sequences")
    parser.add_argument("--export", action="store_true", help="Export to FASTA")
    args = parser.parse_args()

    loader = NCBIArbovirusLoader()

    if args.download or args.force:
        db = loader.load_or_download(
            max_per_virus=args.max_per_virus,
            force_download=args.force
        )
        print(f"\nTotal sequences: {db.count()}")
        for virus in ARBOVIRUS_TAXIDS.keys():
            print(f"  {virus}: {db.count(virus)}")

        if args.export:
            loader.export_fasta(db)

    elif args.list:
        cache_path = loader.cache_dir / "arbovirus_db.json"
        if cache_path.exists():
            db = ArbovirusDatabase.load(cache_path)
            print(f"Cached database: {db.count()} total sequences")
            for virus in ARBOVIRUS_TAXIDS.keys():
                print(f"  {virus}: {db.count(virus)}")
        else:
            print("No cached database found. Run with --download first.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
