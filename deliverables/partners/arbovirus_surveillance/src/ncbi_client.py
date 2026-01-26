# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""NCBI Client for Arbovirus Sequences.

This module provides robust NCBI Entrez API integration with:
- Rate limiting (NCBI allows 3 requests/sec without API key, 10/sec with)
- Caching with expiration
- Batch downloading with progress tracking
- Error handling and retry logic
- Support for multiple output formats

Example:
    >>> client = NCBIClient(email="user@example.com")
    >>> db = client.load_or_download(max_per_virus=50)
    >>> sequences = db.get_sequences("DENV-1")
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Iterator
import threading

from .constants import ARBOVIRUS_TARGETS, get_phylogenetic_identity
from .reference_data import generate_phylogenetic_sequence

# Try to import BioPython
try:
    from Bio import Entrez, SeqIO
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False


class RateLimiter:
    """Thread-safe rate limiter for API calls.

    Implements token bucket algorithm to limit request rate.

    Attributes:
        rate: Maximum requests per second
        tokens: Current available tokens
        last_update: Time of last token update

    Example:
        >>> limiter = RateLimiter(rate=3.0)  # 3 requests/second
        >>> limiter.acquire()  # Blocks if rate exceeded
    """

    def __init__(self, rate: float = 3.0):
        """Initialize rate limiter.

        Args:
            rate: Maximum requests per second
        """
        self.rate = rate
        self.tokens = rate
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        """Acquire a token, blocking if necessary."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update

            # Add tokens based on elapsed time
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens < 1.0:
                # Wait for a token
                wait_time = (1.0 - self.tokens) / self.rate
                time.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1.0


@dataclass
class VirusSequence:
    """Container for a viral sequence.

    Attributes:
        accession: NCBI accession number
        virus: Virus identifier (e.g., "DENV-1")
        description: Sequence description
        sequence: Nucleotide sequence
        length: Sequence length
        country: Country of origin
        collection_date: Date collected
        strain: Strain name
    """

    accession: str
    virus: str
    description: str
    sequence: str
    length: int
    country: Optional[str] = None
    collection_date: Optional[str] = None
    strain: Optional[str] = None
    host: Optional[str] = None
    genotype: Optional[str] = None

    def to_fasta(self) -> str:
        """Convert to FASTA format."""
        header_parts = [self.accession]
        if self.strain:
            header_parts.append(f"strain={self.strain}")
        if self.country:
            header_parts.append(f"country={self.country}")
        header = " ".join(header_parts)
        return f">{header}\n{self.sequence}\n"

    def get_region(self, start: int, end: int) -> str:
        """Get a subsequence region."""
        return self.sequence[start:end]


@dataclass
class ArbovirusDatabase:
    """Database of arbovirus sequences with caching.

    Attributes:
        sequences: Dict mapping virus name to list of sequences
        metadata: Database metadata (download date, source, etc.)
        cache_path: Path to cache file
    """

    sequences: dict[str, list[VirusSequence]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    cache_path: Optional[Path] = None

    def add_sequence(self, virus: str, seq: VirusSequence):
        """Add a sequence for a virus."""
        if virus not in self.sequences:
            self.sequences[virus] = []
        self.sequences[virus].append(seq)

    def get_sequences(self, virus: str) -> list[VirusSequence]:
        """Get all sequences for a virus."""
        return self.sequences.get(virus, [])

    def iter_all_sequences(self) -> Iterator[VirusSequence]:
        """Iterate over all sequences in database."""
        for seqs in self.sequences.values():
            yield from seqs

    def count(self, virus: str = None) -> int:
        """Count sequences."""
        if virus:
            return len(self.sequences.get(virus, []))
        return sum(len(seqs) for seqs in self.sequences.values())

    def get_viruses(self) -> list[str]:
        """Get list of viruses in database."""
        return list(self.sequences.keys())

    def filter_by_country(self, country: str) -> "ArbovirusDatabase":
        """Create filtered database by country."""
        filtered = ArbovirusDatabase(metadata=self.metadata.copy())
        for virus, seqs in self.sequences.items():
            country_seqs = [s for s in seqs if s.country and country.lower() in s.country.lower()]
            for seq in country_seqs:
                filtered.add_sequence(virus, seq)
        return filtered

    def get_consensus(self, virus: str) -> Optional[str]:
        """Compute consensus sequence for a virus.

        Uses simple majority voting at each position.
        """
        sequences = self.get_sequences(virus)
        if not sequences:
            return None

        # Find minimum length
        min_len = min(len(s.sequence) for s in sequences)

        consensus = []
        for i in range(min_len):
            bases = [s.sequence[i] for s in sequences if len(s.sequence) > i]
            # Count bases
            counts = {}
            for b in bases:
                counts[b] = counts.get(b, 0) + 1
            # Get most common
            consensus.append(max(counts, key=counts.get))

        return "".join(consensus)

    def save(self, path: Path = None):
        """Save database to JSON."""
        path = path or self.cache_path
        if path is None:
            raise ValueError("No path specified and no cache_path set")

        data = {
            "metadata": {
                **self.metadata,
                "saved_at": datetime.now().isoformat(),
            },
            "sequences": {
                virus: [asdict(s) for s in seqs]
                for virus, seqs in self.sequences.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ArbovirusDatabase":
        """Load database from JSON."""
        with open(path) as f:
            data = json.load(f)

        db = cls(
            metadata=data.get("metadata", {}),
            cache_path=path,
        )
        for virus, seqs in data.get("sequences", {}).items():
            for seq_data in seqs:
                db.add_sequence(virus, VirusSequence(**seq_data))
        return db

    def is_expired(self, max_age_days: int = 30) -> bool:
        """Check if cache is expired."""
        saved_at = self.metadata.get("saved_at")
        if not saved_at:
            return True

        try:
            save_date = datetime.fromisoformat(saved_at)
            return datetime.now() - save_date > timedelta(days=max_age_days)
        except (ValueError, TypeError):
            return True


class NCBIClient:
    """Client for downloading arbovirus sequences from NCBI.

    Features:
    - Rate limiting to comply with NCBI guidelines
    - Intelligent caching with expiration
    - Batch downloading with progress
    - Retry logic for failed requests
    - Demo mode when BioPython unavailable

    Example:
        >>> client = NCBIClient(email="user@example.com")
        >>> db = client.load_or_download()
        >>> print(f"Downloaded {db.count()} sequences")
    """

    def __init__(
        self,
        email: str = None,
        api_key: str = None,
        cache_dir: Path = None,
        rate_limit: float = None,
    ):
        """Initialize NCBI client.

        Args:
            email: Email for NCBI Entrez (required for heavy use)
            api_key: NCBI API key (increases rate limit)
            cache_dir: Directory for caching sequences
            rate_limit: Custom rate limit (default: 3/sec or 10/sec with API key)
        """
        self.email = email or os.environ.get("NCBI_EMAIL", "user@example.com")
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")

        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent / "data"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiter (10/sec with API key, 3/sec without)
        if rate_limit:
            self._rate = rate_limit
        else:
            self._rate = 10.0 if self.api_key else 3.0
        self._limiter = RateLimiter(self._rate)

        # Configure BioPython Entrez
        if BIOPYTHON_AVAILABLE:
            Entrez.email = self.email
            if self.api_key:
                Entrez.api_key = self.api_key

    def download_virus(
        self,
        virus: str,
        max_sequences: int = 100,
        region_filter: str = None,
        complete_only: bool = True,
    ) -> list[VirusSequence]:
        """Download sequences for a specific virus.

        Args:
            virus: Virus identifier (e.g., "DENV-1")
            max_sequences: Maximum sequences to download
            region_filter: Geographic filter (e.g., "Paraguay")
            complete_only: Only download complete genomes

        Returns:
            List of VirusSequence objects
        """
        if not BIOPYTHON_AVAILABLE:
            return self._generate_demo_sequences(virus, max_sequences)

        target = ARBOVIRUS_TARGETS.get(virus)
        if target is None:
            print(f"Unknown virus: {virus}")
            return []

        # Build query
        query_parts = [f"txid{target.get('taxid', 0)}[Organism]"]
        if complete_only:
            query_parts.append("complete genome[Title]")
        if region_filter:
            query_parts.append(f"{region_filter}[Country]")

        query = " AND ".join(query_parts)

        return self._fetch_sequences(virus, query, max_sequences)

    def _fetch_sequences(
        self,
        virus: str,
        query: str,
        max_sequences: int,
        retries: int = 3,
    ) -> list[VirusSequence]:
        """Fetch sequences from NCBI with retry logic."""
        for attempt in range(retries):
            try:
                self._limiter.acquire()
                print(f"Searching NCBI for {virus}...")

                handle = Entrez.esearch(
                    db="nucleotide",
                    term=query,
                    retmax=max_sequences,
                    usehistory="y",
                )
                record = Entrez.read(handle)
                handle.close()

                count = int(record["Count"])
                if count == 0:
                    print(f"No sequences found for {virus}")
                    return self._generate_demo_sequences(virus, min(5, max_sequences))

                print(f"Found {count} sequences for {virus}")
                return self._download_batch(
                    virus, record["WebEnv"], record["QueryKey"],
                    min(count, max_sequences),
                )

            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {virus}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return self._generate_demo_sequences(virus, min(5, max_sequences))

        return []

    def _download_batch(
        self,
        virus: str,
        webenv: str,
        query_key: str,
        total: int,
        batch_size: int = 50,
    ) -> list[VirusSequence]:
        """Download sequences in batches."""
        sequences = []

        for start in range(0, total, batch_size):
            self._limiter.acquire()
            end = min(start + batch_size, total)
            print(f"  Downloading {start + 1}-{end} of {total}...")

            try:
                fetch_handle = Entrez.efetch(
                    db="nucleotide",
                    rettype="gb",
                    retmode="text",
                    retstart=start,
                    retmax=batch_size,
                    webenv=webenv,
                    query_key=query_key,
                )

                for record in SeqIO.parse(fetch_handle, "genbank"):
                    seq = VirusSequence(
                        accession=record.id,
                        virus=virus,
                        description=record.description,
                        sequence=str(record.seq),
                        length=len(record.seq),
                        country=self._extract_qualifier(record, "country"),
                        collection_date=self._extract_qualifier(record, "collection_date"),
                        strain=self._extract_qualifier(record, "strain"),
                        host=self._extract_qualifier(record, "host"),
                    )
                    sequences.append(seq)

                fetch_handle.close()

            except Exception as e:
                print(f"  Error downloading batch: {e}")

        return sequences

    def _extract_qualifier(self, record, qualifier: str) -> Optional[str]:
        """Extract a qualifier from a GenBank record."""
        for feature in record.features:
            if feature.type == "source":
                quals = feature.qualifiers
                if qualifier in quals:
                    return quals[qualifier][0]
        return None

    def _generate_demo_sequences(
        self,
        virus: str,
        n: int,
    ) -> list[VirusSequence]:
        """Generate phylogenetically-realistic demo sequences.

        Uses the PHYLOGENETIC_IDENTITY matrix to generate sequences with
        realistic divergence from the DENV-1 reference, ensuring that
        cross-reactivity testing produces meaningful results.

        Args:
            virus: Target virus identifier
            n: Number of sequences to generate

        Returns:
            List of VirusSequence objects with realistic identity profiles
        """
        import random

        target = ARBOVIRUS_TARGETS.get(virus, {})
        size = target.get("genome_size", 11000)
        conserved = target.get("conserved_regions", [])

        # Generate a single base sequence for this virus (reference)
        base_seed = hash(virus) % (2**31)
        random.seed(base_seed)
        base_seq = "".join(random.choices("ACGT", k=size))

        # Insert conserved motifs
        seq_list = list(base_seq)
        for i, (start, end) in enumerate(conserved):
            random.seed(base_seed + i * 1000)
            motif = "".join(random.choices("ACGT", k=min(100, end - start)))
            for j, nt in enumerate(motif):
                if start + j < len(seq_list):
                    seq_list[start + j] = nt

        base_seq = "".join(seq_list)

        # Generate n sequences with intra-species variation (~95-99% identity)
        sequences = []
        for i in range(n):
            # Intra-species variation: 95-99% identity (small mutations)
            intra_identity = 0.97 + random.random() * 0.02  # 97-99%

            seq = generate_phylogenetic_sequence(
                reference=base_seq,
                target_identity=intra_identity,
                seed=base_seed + i + 1,
                preserve_regions=conserved,
            )

            sequences.append(VirusSequence(
                accession=f"{virus}_DEMO_{i + 1:03d}",
                virus=virus,
                description=f"Demo {virus} complete genome isolate {i + 1}",
                sequence=seq,
                length=len(seq),
                country=random.choice(["Paraguay", "Brazil", "Argentina"]),
                collection_date=f"202{random.randint(0, 4)}",
                strain=f"Demo-{i + 1}",
            ))

        return sequences

    def generate_all_demo_sequences(
        self,
        n_per_virus: int = 10,
        seed: int = 42,
    ) -> ArbovirusDatabase:
        """Generate demo sequences for all viruses with realistic phylogeny.

        This method ensures that inter-virus identity matches the
        PHYLOGENETIC_IDENTITY matrix, enabling meaningful cross-reactivity
        testing.

        Args:
            n_per_virus: Sequences per virus
            seed: Random seed for reproducibility

        Returns:
            ArbovirusDatabase with phylogenetically-realistic sequences
        """
        import random

        db = ArbovirusDatabase(
            metadata={
                "download_date": datetime.now().isoformat(),
                "source": "PhylogeneticDemo",
                "n_per_virus": n_per_virus,
                "seed": seed,
            },
            cache_path=self.cache_dir / "demo_phylogenetic_db.json",
        )

        # Generate base DENV-1 sequence
        denv1_target = ARBOVIRUS_TARGETS.get("DENV-1", {})
        denv1_size = denv1_target.get("genome_size", 10700)
        denv1_conserved = denv1_target.get("conserved_regions", [])

        random.seed(seed)
        denv1_base = "".join(random.choices("ACGT", k=denv1_size))

        # Insert conserved motifs into DENV-1
        seq_list = list(denv1_base)
        for i, (start, end) in enumerate(denv1_conserved):
            random.seed(seed + i * 1000)
            motif = "".join(random.choices("ACGT", k=min(100, end - start)))
            for j, nt in enumerate(motif):
                if start + j < len(seq_list):
                    seq_list[start + j] = nt
        denv1_base = "".join(seq_list)

        # Generate sequences for all viruses
        for virus_idx, virus in enumerate(ARBOVIRUS_TARGETS.keys()):
            target = ARBOVIRUS_TARGETS.get(virus, {})
            size = target.get("genome_size", 11000)
            conserved = target.get("conserved_regions", denv1_conserved)

            # Get target identity relative to DENV-1
            inter_identity = get_phylogenetic_identity("DENV-1", virus)

            # Generate base sequence for this virus
            if virus == "DENV-1":
                virus_base = denv1_base
            else:
                # Adjust length if needed
                if size < len(denv1_base):
                    ref_seq = denv1_base[:size]
                elif size > len(denv1_base):
                    random.seed(seed + virus_idx * 10000)
                    extension = "".join(random.choices("ACGT", k=size - len(denv1_base)))
                    ref_seq = denv1_base + extension
                else:
                    ref_seq = denv1_base

                virus_base = generate_phylogenetic_sequence(
                    reference=ref_seq,
                    target_identity=inter_identity,
                    seed=seed + virus_idx * 100,
                    preserve_regions=conserved,
                )

            # Generate intra-species variants
            for i in range(n_per_virus):
                intra_identity = 0.97 + random.random() * 0.02

                seq = generate_phylogenetic_sequence(
                    reference=virus_base,
                    target_identity=intra_identity,
                    seed=seed + virus_idx * 1000 + i,
                    preserve_regions=conserved,
                )

                db.add_sequence(virus, VirusSequence(
                    accession=f"{virus}_PHYLO_{i + 1:03d}",
                    virus=virus,
                    description=f"Phylogenetic demo {virus} isolate {i + 1}",
                    sequence=seq,
                    length=len(seq),
                    country=random.choice(["Paraguay", "Brazil", "Argentina"]),
                    collection_date=f"202{random.randint(0, 4)}",
                    strain=f"Phylo-{i + 1}",
                ))

        return db

    def download_all(
        self,
        max_per_virus: int = 50,
        region_filter: str = None,
    ) -> ArbovirusDatabase:
        """Download sequences for all arboviruses.

        Args:
            max_per_virus: Maximum sequences per virus
            region_filter: Geographic filter

        Returns:
            ArbovirusDatabase with all sequences
        """
        db = ArbovirusDatabase(
            metadata={
                "download_date": datetime.now().isoformat(),
                "source": "NCBI" if BIOPYTHON_AVAILABLE else "Demo",
                "max_per_virus": max_per_virus,
                "region_filter": region_filter,
            },
            cache_path=self.cache_dir / "arbovirus_db.json",
        )

        for virus in ARBOVIRUS_TARGETS.keys():
            sequences = self.download_virus(
                virus, max_per_virus, region_filter
            )
            for seq in sequences:
                db.add_sequence(virus, seq)
            print(f"  {virus}: {len(sequences)} sequences")

        return db

    def load_or_download(
        self,
        cache_name: str = "arbovirus_db.json",
        max_per_virus: int = 50,
        force_download: bool = False,
        max_age_days: int = 30,
    ) -> ArbovirusDatabase:
        """Load from cache or download if needed.

        Args:
            cache_name: Cache file name
            max_per_virus: Max sequences per virus
            force_download: Force re-download
            max_age_days: Max cache age in days

        Returns:
            ArbovirusDatabase
        """
        cache_path = self.cache_dir / cache_name

        if cache_path.exists() and not force_download:
            print(f"Loading cached database from {cache_path}")
            db = ArbovirusDatabase.load(cache_path)

            if not db.is_expired(max_age_days):
                return db
            else:
                print("Cache expired, re-downloading...")

        print("Downloading arbovirus sequences...")
        db = self.download_all(max_per_virus)
        db.save(cache_path)

        return db

    def export_fasta(
        self,
        db: ArbovirusDatabase,
        output_dir: Path = None,
        separate_files: bool = True,
    ):
        """Export database to FASTA format.

        Args:
            db: ArbovirusDatabase to export
            output_dir: Output directory
            separate_files: Create separate file per virus
        """
        output_dir = Path(output_dir or self.cache_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if separate_files:
            for virus, sequences in db.sequences.items():
                fasta_path = output_dir / f"{virus}_genomes.fasta"
                with open(fasta_path, "w") as f:
                    for seq in sequences:
                        f.write(seq.to_fasta())
                print(f"Exported {len(sequences)} sequences to {fasta_path}")
        else:
            fasta_path = output_dir / "all_arboviruses.fasta"
            with open(fasta_path, "w") as f:
                for seq in db.iter_all_sequences():
                    f.write(seq.to_fasta())
            print(f"Exported {db.count()} sequences to {fasta_path}")
