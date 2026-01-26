# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Dengue Strain Variation Analysis at Primer Binding Sites.

This script analyzes sequence conservation at CDC primer binding sites across
multiple Dengue strains to understand why some primers failed to match RefSeq.

Scientific Questions:
1. How conserved are the CDC primer binding sites across strains?
2. Do DENV-2 and DENV-3 have higher variation at primer sites than DENV-1/4?
3. Which positions within primers are most variable?
4. Can we identify better conserved regions for primer design?

Methodology:
1. Download multiple complete genomes per Dengue serotype from NCBI
2. Extract sequences at CDC primer binding positions (±50bp context)
3. Compute per-position conservation (Shannon entropy)
4. Compare conservation between serotypes
5. Identify variable positions that explain primer failures

Usage:
    python validation/test_dengue_strain_variation.py
    python validation/test_dengue_strain_variation.py --max-strains 50
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Add package root to path for local imports
_package_root = Path(__file__).resolve().parents[1]
if str(_package_root) not in sys.path:
    sys.path.insert(0, str(_package_root))

from src.constants import (
    ARBOVIRUS_TAXIDS,
    CDC_PRIMERS,
)

# Try to import BioPython
try:
    from Bio import Entrez, SeqIO
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("WARNING: BioPython not available")


@dataclass
class StrainInfo:
    """Information about a viral strain."""
    accession: str
    serotype: str
    strain: Optional[str]
    country: Optional[str]
    year: Optional[str]
    length: int


@dataclass
class ConservationResult:
    """Conservation analysis result for a primer binding site."""
    primer_name: str
    serotype: str
    n_strains: int
    binding_site_start: int
    binding_site_end: int
    mean_entropy: float
    max_entropy: float
    min_entropy: float
    variable_positions: list[int]  # Positions with entropy > 0.5
    conservation_profile: list[float]  # Entropy per position
    consensus_sequence: str
    consensus_match_to_cdc: float


@dataclass
class VariationReport:
    """Complete strain variation analysis report."""
    timestamp: str
    serotypes_analyzed: list[str]
    strains_per_serotype: dict[str, int]
    primer_conservation: list[dict]
    key_findings: list[str]


def download_dengue_strains(
    serotype: str,
    max_strains: int = 30,
    email: str = "user@example.com",
) -> list[tuple[str, str]]:
    """Download multiple strains of a Dengue serotype.

    Args:
        serotype: DENV-1, DENV-2, DENV-3, or DENV-4
        max_strains: Maximum strains to download
        email: Email for NCBI Entrez

    Returns:
        List of (accession, sequence) tuples
    """
    if not BIOPYTHON_AVAILABLE:
        print(f"  BioPython not available for {serotype}")
        return []

    taxid = ARBOVIRUS_TAXIDS.get(serotype)
    if not taxid:
        print(f"  Unknown serotype: {serotype}")
        return []

    Entrez.email = email

    try:
        # Search for complete genomes
        query = f"txid{taxid}[Organism] AND complete genome[Title]"
        print(f"  Searching NCBI for {serotype}...")

        handle = Entrez.esearch(
            db="nucleotide",
            term=query,
            retmax=max_strains,
            usehistory="y"
        )
        record = Entrez.read(handle)
        handle.close()

        count = int(record["Count"])
        print(f"  Found {count} complete genomes, downloading up to {max_strains}...")

        if count == 0:
            return []

        # Fetch sequences
        sequences = []
        webenv = record["WebEnv"]
        query_key = record["QueryKey"]

        batch_size = 10
        for start in range(0, min(count, max_strains), batch_size):
            print(f"    Downloading {start + 1} to {min(start + batch_size, max_strains)}...")

            fetch_handle = Entrez.efetch(
                db="nucleotide",
                rettype="fasta",
                retmode="text",
                retstart=start,
                retmax=batch_size,
                webenv=webenv,
                query_key=query_key
            )

            for record in SeqIO.parse(fetch_handle, "fasta"):
                sequences.append((record.id, str(record.seq)))

            fetch_handle.close()
            time.sleep(0.5)  # Rate limiting

        print(f"  Downloaded {len(sequences)} strains for {serotype}")
        return sequences

    except Exception as e:
        print(f"  Error downloading {serotype}: {e}")
        return []


def compute_shannon_entropy(column: list[str]) -> float:
    """Compute Shannon entropy for a sequence column.

    Args:
        column: List of nucleotides at a position

    Returns:
        Entropy value (0 = fully conserved, 2 = random)
    """
    # Filter out gaps and Ns
    valid = [b for b in column if b in "ACGT"]
    if len(valid) == 0:
        return 2.0  # Maximum entropy for unknown

    counts = Counter(valid)
    total = len(valid)

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


def extract_binding_region(
    sequences: list[tuple[str, str]],
    position: int,
    primer_length: int,
    context: int = 50,
) -> list[str]:
    """Extract primer binding region from multiple sequences.

    Args:
        sequences: List of (accession, sequence) tuples
        position: Start position of primer binding site
        primer_length: Length of primer
        context: Extra bases on each side

    Returns:
        List of extracted regions (aligned by position)
    """
    regions = []
    start = max(0, position - context)
    end = position + primer_length + context

    for accession, seq in sequences:
        if len(seq) >= end:
            regions.append(seq[start:end])

    return regions


def compute_conservation_profile(
    regions: list[str],
) -> tuple[list[float], str]:
    """Compute per-position conservation and consensus.

    Args:
        regions: List of aligned sequence regions

    Returns:
        Tuple of (entropy_profile, consensus_sequence)
    """
    if not regions:
        return [], ""

    length = min(len(r) for r in regions)
    entropy_profile = []
    consensus = []

    for i in range(length):
        column = [r[i] for r in regions if i < len(r)]
        entropy = compute_shannon_entropy(column)
        entropy_profile.append(entropy)

        # Get consensus base
        valid = [b for b in column if b in "ACGT"]
        if valid:
            counts = Counter(valid)
            consensus.append(counts.most_common(1)[0][0])
        else:
            consensus.append("N")

    return entropy_profile, "".join(consensus)


def analyze_primer_conservation(
    serotype: str,
    sequences: list[tuple[str, str]],
    primer_name: str,
    primer_seq: str,
    binding_position: int,
) -> Optional[ConservationResult]:
    """Analyze conservation at a primer binding site.

    Args:
        serotype: Dengue serotype
        sequences: List of strain sequences
        primer_name: Name of CDC primer
        primer_seq: Primer sequence
        binding_position: Position where primer binds

    Returns:
        ConservationResult or None if insufficient data
    """
    if len(sequences) < 5:
        return None

    primer_len = len(primer_seq)
    context = 30

    # Extract regions
    regions = extract_binding_region(sequences, binding_position, primer_len, context)

    if len(regions) < 5:
        return None

    # Compute conservation
    entropy_profile, consensus = compute_conservation_profile(regions)

    if not entropy_profile:
        return None

    # Extract primer region from profile (skip context)
    primer_start = context
    primer_end = context + primer_len
    primer_entropy = entropy_profile[primer_start:primer_end]

    # Find variable positions (entropy > 0.5 bits)
    variable_positions = [
        i for i, e in enumerate(primer_entropy) if e > 0.5
    ]

    # Compute match between consensus and CDC primer
    consensus_primer = consensus[primer_start:primer_end]
    matches = sum(1 for a, b in zip(primer_seq.upper(), consensus_primer.upper()) if a == b)
    match_rate = matches / len(primer_seq) if primer_seq else 0

    return ConservationResult(
        primer_name=primer_name,
        serotype=serotype,
        n_strains=len(regions),
        binding_site_start=binding_position,
        binding_site_end=binding_position + primer_len,
        mean_entropy=sum(primer_entropy) / len(primer_entropy) if primer_entropy else 0,
        max_entropy=max(primer_entropy) if primer_entropy else 0,
        min_entropy=min(primer_entropy) if primer_entropy else 0,
        variable_positions=variable_positions,
        conservation_profile=primer_entropy,
        consensus_sequence=consensus_primer,
        consensus_match_to_cdc=match_rate,
    )


def run_analysis(
    cache_dir: Path,
    max_strains: int = 30,
    use_cache: bool = True,
) -> VariationReport:
    """Run complete strain variation analysis.

    Args:
        cache_dir: Directory for caching
        max_strains: Maximum strains per serotype
        use_cache: Whether to use cached strain data

    Returns:
        VariationReport with results
    """
    import datetime

    print("=" * 70)
    print("DENGUE STRAIN VARIATION ANALYSIS")
    print("=" * 70)
    print()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "dengue_strains.json"

    # Primer binding positions (from CDC recovery validation)
    primer_positions = {
        "CDC_DENV1": {"forward": 8972, "reverse": 9059},
        "CDC_DENV2": {"forward": 141, "reverse": 833},
        "CDC_DENV3": {"forward": 9192, "reverse": 1129},
        "CDC_DENV4": {"forward": 903, "reverse": 972},
    }

    serotypes = ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]
    all_sequences = {}

    # Load or download strains
    if cache_file.exists() and use_cache:
        print("Loading cached strain data...")
        with open(cache_file) as f:
            all_sequences = json.load(f)
        print(f"  Loaded {sum(len(v) for v in all_sequences.values())} strains")
    else:
        print("Downloading Dengue strains from NCBI...")
        for serotype in serotypes:
            sequences = download_dengue_strains(serotype, max_strains)
            all_sequences[serotype] = sequences
            time.sleep(1)  # Rate limiting between serotypes

        # Cache results
        with open(cache_file, "w") as f:
            json.dump(all_sequences, f)
        print(f"  Cached to {cache_file}")

    print()
    print("-" * 70)
    print("CONSERVATION ANALYSIS")
    print("-" * 70)
    print()

    results = []
    strains_per_serotype = {}

    for serotype in serotypes:
        sequences = all_sequences.get(serotype, [])
        # Convert from list of lists if needed
        if sequences and isinstance(sequences[0], list):
            sequences = [(s[0], s[1]) for s in sequences]

        strains_per_serotype[serotype] = len(sequences)

        if len(sequences) < 5:
            print(f"{serotype}: Insufficient strains ({len(sequences)})")
            continue

        print(f"{serotype}: {len(sequences)} strains")

        # Find the matching CDC primer
        primer = None
        for p in CDC_PRIMERS:
            if p.target_virus == serotype:
                primer = p
                break

        if not primer:
            continue

        primer_key = f"CDC_{serotype.replace('-', '')}"
        positions = primer_positions.get(primer_key, {})

        # Analyze forward primer
        if "forward" in positions:
            result = analyze_primer_conservation(
                serotype=serotype,
                sequences=sequences,
                primer_name=f"{primer.name}_forward",
                primer_seq=primer.forward,
                binding_position=positions["forward"],
            )
            if result:
                results.append(result)
                print(f"  Forward: entropy={result.mean_entropy:.3f}, "
                      f"variable_pos={len(result.variable_positions)}, "
                      f"consensus_match={result.consensus_match_to_cdc:.1%}")

        # Analyze reverse primer
        if "reverse" in positions:
            result = analyze_primer_conservation(
                serotype=serotype,
                sequences=sequences,
                primer_name=f"{primer.name}_reverse",
                primer_seq=primer.reverse,
                binding_position=positions["reverse"],
            )
            if result:
                results.append(result)
                print(f"  Reverse: entropy={result.mean_entropy:.3f}, "
                      f"variable_pos={len(result.variable_positions)}, "
                      f"consensus_match={result.consensus_match_to_cdc:.1%}")

    print()
    print("-" * 70)
    print("KEY FINDINGS")
    print("-" * 70)
    print()

    findings = []

    # Analyze results
    if results:
        # Compare entropy between serotypes
        serotype_entropy = {}
        for r in results:
            st = r.serotype
            if st not in serotype_entropy:
                serotype_entropy[st] = []
            serotype_entropy[st].append(r.mean_entropy)

        for st, entropies in serotype_entropy.items():
            avg = sum(entropies) / len(entropies)
            finding = f"{st}: Mean entropy = {avg:.3f}"
            findings.append(finding)
            print(f"  {finding}")

        # Find most variable primers
        print()
        most_variable = sorted(results, key=lambda x: x.mean_entropy, reverse=True)
        if most_variable:
            mv = most_variable[0]
            finding = f"Most variable: {mv.primer_name} (entropy={mv.mean_entropy:.3f})"
            findings.append(finding)
            print(f"  {finding}")

        # Find best conserved primers
        most_conserved = sorted(results, key=lambda x: x.mean_entropy)
        if most_conserved:
            mc = most_conserved[0]
            finding = f"Most conserved: {mc.primer_name} (entropy={mc.mean_entropy:.3f})"
            findings.append(finding)
            print(f"  {finding}")

        # Check consensus match rates
        print()
        for r in results:
            if r.consensus_match_to_cdc < 0.9:
                finding = f"{r.primer_name}: consensus only {r.consensus_match_to_cdc:.1%} match to CDC"
                findings.append(finding)
                print(f"  WARNING: {finding}")

    print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    report = VariationReport(
        timestamp=datetime.datetime.now().isoformat(),
        serotypes_analyzed=serotypes,
        strains_per_serotype=strains_per_serotype,
        primer_conservation=[asdict(r) for r in results],
        key_findings=findings,
    )

    return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dengue Strain Variation Analysis"
    )
    parser.add_argument(
        "--max-strains",
        type=int,
        default=30,
        help="Maximum strains per serotype to download",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached strain data if available",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of strains",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for JSON report",
    )

    args = parser.parse_args()

    # Set up paths
    validation_dir = Path(__file__).parent
    cache_dir = validation_dir.parent / "data"

    # Run analysis
    report = run_analysis(
        cache_dir=cache_dir,
        max_strains=args.max_strains,
        use_cache=not args.force_download,
    )

    # Save report
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = validation_dir / "dengue_strain_variation_report.json"

    with open(output_path, "w") as f:
        json.dump(asdict(report), f, indent=2)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
