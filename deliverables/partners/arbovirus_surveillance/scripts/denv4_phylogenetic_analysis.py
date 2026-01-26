# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Comprehensive DENV-4 Phylogenetic Analysis.

This script performs extensive phylogenetic analysis of DENV-4 to:
1. Download ALL complete DENV-4 genomes from NCBI (~270)
2. Compute pairwise sequence identities
3. Perform hierarchical clustering to identify clades
4. Map geographic distribution per clade
5. Compute per-clade entropy for conserved region identification
6. Select representative sequences per clade for primer design
7. Generate comprehensive reports with visualizations

The goal is to understand DENV-4's cryptic diversity structure and identify
genotype-specific conserved regions suitable for multiplexed primer design.

Usage:
    python scripts/denv4_phylogenetic_analysis.py

Requirements:
    - Biopython (for sequence handling)
    - scipy (for clustering)
    - numpy, matplotlib (for analysis and visualization)
    - Optional: MUSCLE/MAFFT (for alignment), IQ-TREE (for phylogeny)

Author: AI Whisperers
Date: 2026-01-04
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
from scipy import cluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr

try:
    from Bio import SeqIO, Entrez, Phylo
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.Align import MultipleSeqAlignment
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False
    print("WARNING: Biopython not installed. Install with: pip install biopython")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: Matplotlib not installed. Visualizations will be skipped.")

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "denv4"
RESULTS_DIR = PROJECT_ROOT / "results" / "phylogenetic"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# NCBI Configuration
Entrez.email = "aiwhisperers@example.com"  # Required by NCBI
DENV4_TAXON_ID = 11070
MAX_SEQUENCES = 500  # Safety limit


@dataclass
class SequenceMetadata:
    """Metadata for a DENV-4 sequence."""
    accession: str
    organism: str = "Dengue virus 4"
    strain: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    collection_date: Optional[str] = None
    year: Optional[int] = None
    genotype: Optional[str] = None  # If annotated
    host: Optional[str] = None
    length: int = 0

    # Computed fields
    clade: Optional[str] = None  # Assigned by clustering
    is_representative: bool = False


@dataclass
class CladeInfo:
    """Information about a phylogenetic clade."""
    name: str
    size: int = 0
    countries: list = field(default_factory=list)
    year_range: tuple = (None, None)
    mean_identity: float = 0.0
    representative_accession: Optional[str] = None
    conserved_regions: list = field(default_factory=list)
    color: str = "#000000"


class DENV4PhylogeneticAnalyzer:
    """Comprehensive phylogenetic analyzer for DENV-4."""

    # Gene regions in DENV-4 genome (approximate positions)
    GENE_REGIONS = {
        "5UTR": (1, 101),
        "C": (102, 476),        # Capsid
        "prM": (477, 976),      # Pre-membrane
        "E": (977, 2471),       # Envelope
        "NS1": (2472, 3527),    # Non-structural 1
        "NS2A": (3528, 4180),   # NS2A
        "NS2B": (4181, 4571),   # NS2B
        "NS3": (4572, 6428),    # NS3 (protease/helicase)
        "NS4A": (6429, 6809),   # NS4A
        "NS4B": (6810, 7558),   # NS4B
        "NS5": (7559, 10271),   # NS5 (RdRp/MTase)
        "3UTR": (10272, 10723),
    }

    # Known DENV-4 genotypes (based on literature)
    KNOWN_GENOTYPES = {
        "I": {"regions": ["Southeast Asia", "China", "Philippines"], "years": (1956, 2000)},
        "II": {"regions": ["Americas", "Caribbean", "Haiti"], "years": (1980, 2025)},
        "III": {"regions": ["Thailand", "Malaysia"], "years": (1970, 2010)},
        "IV": {"regions": ["Malaysia sylvatic"], "years": (1970, 1990)},
    }

    def __init__(self, cache_enabled: bool = True):
        """Initialize the analyzer.

        Args:
            cache_enabled: Whether to use cached data
        """
        self.cache_enabled = cache_enabled
        self.sequences: dict[str, str] = {}
        self.metadata: dict[str, SequenceMetadata] = {}
        self.distance_matrix: Optional[np.ndarray] = None
        self.clades: dict[str, CladeInfo] = {}
        self.alignment: Optional[list] = None

        # Ensure directories exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def run_full_analysis(self) -> dict:
        """Run the complete phylogenetic analysis pipeline.

        Returns:
            Dictionary with all analysis results
        """
        print("=" * 80)
        print("DENV-4 COMPREHENSIVE PHYLOGENETIC ANALYSIS")
        print("=" * 80)
        print()

        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "pipeline_steps": [],
        }

        # Step 1: Download sequences
        print("-" * 80)
        print("STEP 1: Downloading DENV-4 sequences from NCBI")
        print("-" * 80)
        step1 = self.download_denv4_sequences()
        results["pipeline_steps"].append({"step": "download", "result": step1})
        print(f"Downloaded {len(self.sequences)} sequences")
        print()

        # Step 2: Compute pairwise distances
        print("-" * 80)
        print("STEP 2: Computing pairwise sequence distances")
        print("-" * 80)
        step2 = self.compute_pairwise_distances()
        results["pipeline_steps"].append({"step": "distances", "result": step2})
        print()

        # Step 3: Hierarchical clustering
        print("-" * 80)
        print("STEP 3: Hierarchical clustering to identify clades")
        print("-" * 80)
        step3 = self.perform_clustering()
        results["pipeline_steps"].append({"step": "clustering", "result": step3})
        print()

        # Step 4: Geographic mapping
        print("-" * 80)
        print("STEP 4: Geographic distribution analysis")
        print("-" * 80)
        step4 = self.analyze_geographic_distribution()
        results["pipeline_steps"].append({"step": "geography", "result": step4})
        print()

        # Step 5: Temporal analysis
        print("-" * 80)
        print("STEP 5: Temporal distribution analysis")
        print("-" * 80)
        step5 = self.analyze_temporal_distribution()
        results["pipeline_steps"].append({"step": "temporal", "result": step5})
        print()

        # Step 6: Per-clade conservation
        print("-" * 80)
        print("STEP 6: Per-clade conservation analysis")
        print("-" * 80)
        step6 = self.analyze_per_clade_conservation()
        results["pipeline_steps"].append({"step": "conservation", "result": step6})
        print()

        # Step 7: Select representatives
        print("-" * 80)
        print("STEP 7: Selecting clade representatives")
        print("-" * 80)
        step7 = self.select_clade_representatives()
        results["pipeline_steps"].append({"step": "representatives", "result": step7})
        print()

        # Step 8: Generate visualizations
        if HAS_MATPLOTLIB:
            print("-" * 80)
            print("STEP 8: Generating visualizations")
            print("-" * 80)
            step8 = self.generate_visualizations()
            results["pipeline_steps"].append({"step": "visualizations", "result": step8})
            print()

        # Step 9: Generate report
        print("-" * 80)
        print("STEP 9: Generating comprehensive report")
        print("-" * 80)
        step9 = self.generate_report(results)
        results["pipeline_steps"].append({"step": "report", "result": step9})
        print()

        # Save results
        results_path = RESULTS_DIR / "phylogenetic_analysis_results.json"
        self._save_json(results, results_path)
        print(f"\nResults saved to: {results_path}")

        return results

    def download_denv4_sequences(self) -> dict:
        """Download all DENV-4 complete genomes from NCBI.

        Returns:
            Dictionary with download statistics
        """
        cache_path = CACHE_DIR / "denv4_sequences.json"
        metadata_cache = CACHE_DIR / "denv4_metadata.json"

        # Check cache
        if self.cache_enabled and cache_path.exists():
            print(f"Loading cached sequences from {cache_path}")
            with open(cache_path) as f:
                self.sequences = json.load(f)
            if metadata_cache.exists():
                with open(metadata_cache) as f:
                    meta_dict = json.load(f)
                    self.metadata = {
                        k: SequenceMetadata(**v) for k, v in meta_dict.items()
                    }
            return {"source": "cache", "count": len(self.sequences)}

        if not HAS_BIOPYTHON:
            print("Biopython required for NCBI download. Using existing cache if available.")
            return {"source": "error", "count": 0}

        print("Searching NCBI for DENV-4 complete genomes...")

        try:
            # Search for complete genomes
            search_term = (
                "txid11070[Organism] AND "
                "complete genome[Title] AND "
                "10000:12000[Sequence Length]"
            )

            handle = Entrez.esearch(
                db="nucleotide",
                term=search_term,
                retmax=MAX_SEQUENCES,
                usehistory="y",
            )
            search_results = Entrez.read(handle)
            handle.close()

            id_list = search_results["IdList"]
            print(f"Found {len(id_list)} complete DENV-4 genomes")

            if not id_list:
                print("No sequences found. Using fallback data.")
                return self._use_fallback_data()

            # Download in batches
            batch_size = 50
            all_records = []

            for start in range(0, len(id_list), batch_size):
                end = min(start + batch_size, len(id_list))
                print(f"Downloading batch {start//batch_size + 1}/{(len(id_list)-1)//batch_size + 1}...")

                try:
                    fetch_handle = Entrez.efetch(
                        db="nucleotide",
                        id=id_list[start:end],
                        rettype="gb",
                        retmode="text",
                    )
                    records = list(SeqIO.parse(fetch_handle, "genbank"))
                    fetch_handle.close()
                    all_records.extend(records)

                    # Be nice to NCBI
                    time.sleep(0.5)

                except Exception as e:
                    print(f"Error downloading batch: {e}")
                    continue

            # Process records
            for record in all_records:
                accession = record.id.split(".")[0]
                self.sequences[accession] = str(record.seq)

                # Extract metadata
                meta = self._extract_metadata(record)
                self.metadata[accession] = meta

            # Cache results
            with open(cache_path, "w") as f:
                json.dump(self.sequences, f)

            with open(metadata_cache, "w") as f:
                meta_dict = {k: asdict(v) for k, v in self.metadata.items()}
                json.dump(meta_dict, f, indent=2, default=str)

            return {"source": "ncbi", "count": len(self.sequences)}

        except Exception as e:
            print(f"NCBI download error: {e}")
            return self._use_fallback_data()

    def _extract_metadata(self, record) -> SequenceMetadata:
        """Extract metadata from a GenBank record."""
        accession = record.id.split(".")[0]

        # Get basic info
        strain = None
        country = None
        region = None
        collection_date = None
        year = None
        host = None
        genotype = None

        # Parse features
        for feature in record.features:
            if feature.type == "source":
                qualifiers = feature.qualifiers

                strain = qualifiers.get("strain", [None])[0]
                if not strain:
                    strain = qualifiers.get("isolate", [None])[0]

                country_raw = qualifiers.get("country", [None])[0]
                if country_raw:
                    parts = country_raw.split(":")
                    country = parts[0].strip()
                    region = parts[1].strip() if len(parts) > 1 else None

                collection_date = qualifiers.get("collection_date", [None])[0]
                if collection_date:
                    # Extract year
                    year_match = re.search(r"(\d{4})", collection_date)
                    if year_match:
                        year = int(year_match.group(1))

                host = qualifiers.get("host", [None])[0]

                # Check for genotype annotation
                note = qualifiers.get("note", [""])[0]
                gt_match = re.search(r"genotype[:\s]*(\w+)", note, re.I)
                if gt_match:
                    genotype = gt_match.group(1)

        return SequenceMetadata(
            accession=accession,
            strain=strain,
            country=country,
            region=region,
            collection_date=collection_date,
            year=year,
            host=host,
            genotype=genotype,
            length=len(record.seq),
        )

    def _use_fallback_data(self) -> dict:
        """Use existing cached strain data as fallback."""
        strain_cache = PROJECT_ROOT / "data" / "dengue_strains.json"

        if strain_cache.exists():
            print(f"Using fallback data from {strain_cache}")
            with open(strain_cache) as f:
                all_strains = json.load(f)

            denv4_strains = all_strains.get("DENV-4", [])

            for item in denv4_strains:
                if isinstance(item, list) and len(item) >= 2:
                    accession, sequence = item[0], item[1]
                    self.sequences[accession] = sequence
                    self.metadata[accession] = SequenceMetadata(
                        accession=accession,
                        length=len(sequence),
                    )

            return {"source": "fallback", "count": len(self.sequences)}

        return {"source": "none", "count": 0}

    def compute_pairwise_distances(self) -> dict:
        """Compute pairwise sequence distances.

        Uses k-mer based distance for speed when alignment is not available.

        Returns:
            Dictionary with distance statistics
        """
        accessions = list(self.sequences.keys())
        n = len(accessions)

        if n < 2:
            print("Not enough sequences for distance computation")
            return {"error": "insufficient_sequences"}

        print(f"Computing pairwise distances for {n} sequences...")

        # Use k-mer based distance (faster than alignment)
        k = 6  # k-mer size

        # Compute k-mer profiles
        print("Computing k-mer profiles...")
        kmer_profiles = {}
        for acc in accessions:
            seq = self.sequences[acc].upper()
            kmers = Counter()
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if all(c in "ACGT" for c in kmer):
                    kmers[kmer] += 1
            kmer_profiles[acc] = kmers

        # Compute Jaccard distances
        print("Computing Jaccard distances...")
        self.distance_matrix = np.zeros((n, n))

        total_pairs = n * (n - 1) // 2
        pair_count = 0

        for i in range(n):
            for j in range(i + 1, n):
                profile_i = set(kmer_profiles[accessions[i]].keys())
                profile_j = set(kmer_profiles[accessions[j]].keys())

                intersection = len(profile_i & profile_j)
                union = len(profile_i | profile_j)

                jaccard_sim = intersection / union if union > 0 else 0
                jaccard_dist = 1 - jaccard_sim

                self.distance_matrix[i, j] = jaccard_dist
                self.distance_matrix[j, i] = jaccard_dist

                pair_count += 1
                if pair_count % 1000 == 0:
                    print(f"  Processed {pair_count}/{total_pairs} pairs...")

        # Convert to identity
        identity_matrix = 1 - self.distance_matrix

        # Statistics
        upper_tri = self.distance_matrix[np.triu_indices(n, k=1)]

        stats = {
            "n_sequences": n,
            "mean_distance": float(np.mean(upper_tri)),
            "std_distance": float(np.std(upper_tri)),
            "min_distance": float(np.min(upper_tri)),
            "max_distance": float(np.max(upper_tri)),
            "mean_identity": float(1 - np.mean(upper_tri)),
            "min_identity": float(1 - np.max(upper_tri)),
            "max_identity": float(1 - np.min(upper_tri)),
        }

        print(f"  Mean pairwise identity: {stats['mean_identity']:.1%}")
        print(f"  Identity range: {stats['min_identity']:.1%} - {stats['max_identity']:.1%}")

        # Save distance matrix
        np.save(DATA_DIR / "distance_matrix.npy", self.distance_matrix)

        # Save accession order
        with open(DATA_DIR / "accession_order.json", "w") as f:
            json.dump(accessions, f)

        return stats

    def perform_clustering(self, n_clades: int = 5) -> dict:
        """Perform hierarchical clustering to identify clades.

        Args:
            n_clades: Target number of clades to identify

        Returns:
            Dictionary with clustering results
        """
        if self.distance_matrix is None:
            print("Distance matrix not computed. Running distance computation first.")
            self.compute_pairwise_distances()

        accessions = list(self.sequences.keys())
        n = len(accessions)

        print(f"Performing hierarchical clustering into {n_clades} clades...")

        # Convert to condensed form for scipy
        condensed = squareform(self.distance_matrix)

        # Hierarchical clustering
        linkage = cluster.hierarchy.linkage(condensed, method='average')

        # Cut tree to get desired number of clusters
        cluster_labels = cluster.hierarchy.fcluster(
            linkage, t=n_clades, criterion='maxclust'
        )

        # Assign clades to sequences
        clade_names = ["Clade_" + chr(ord('A') + i) for i in range(n_clades)]
        colors = plt.cm.Set1(np.linspace(0, 1, n_clades)) if HAS_MATPLOTLIB else [(0,0,0,1)] * n_clades

        for i, acc in enumerate(accessions):
            clade_idx = cluster_labels[i] - 1
            clade_name = clade_names[clade_idx]
            self.metadata[acc].clade = clade_name

        # Compute clade statistics
        for i, clade_name in enumerate(clade_names):
            members = [acc for acc, meta in self.metadata.items() if meta.clade == clade_name]

            if not members:
                continue

            # Countries in this clade
            countries = [
                self.metadata[acc].country for acc in members
                if self.metadata[acc].country
            ]

            # Years in this clade
            years = [
                self.metadata[acc].year for acc in members
                if self.metadata[acc].year
            ]

            # Within-clade identity
            member_indices = [accessions.index(acc) for acc in members]
            if len(member_indices) > 1:
                within_dists = []
                for ii, idx_i in enumerate(member_indices):
                    for idx_j in member_indices[ii+1:]:
                        within_dists.append(self.distance_matrix[idx_i, idx_j])
                mean_identity = 1 - np.mean(within_dists) if within_dists else 1.0
            else:
                mean_identity = 1.0

            self.clades[clade_name] = CladeInfo(
                name=clade_name,
                size=len(members),
                countries=list(Counter(countries).most_common()),
                year_range=(min(years) if years else None, max(years) if years else None),
                mean_identity=float(mean_identity),
                color=f"#{int(colors[i][0]*255):02x}{int(colors[i][1]*255):02x}{int(colors[i][2]*255):02x}",
            )

        # Print summary
        print("\nClade Summary:")
        for clade_name, info in sorted(self.clades.items(), key=lambda x: -x[1].size):
            top_countries = ", ".join([f"{c}({n})" for c, n in info.countries[:3]])
            print(f"  {clade_name}: {info.size} sequences, identity={info.mean_identity:.1%}")
            print(f"    Countries: {top_countries}")
            print(f"    Years: {info.year_range}")

        # Save linkage for tree visualization
        np.save(DATA_DIR / "linkage.npy", linkage)

        return {
            "n_clades": n_clades,
            "clade_sizes": {name: info.size for name, info in self.clades.items()},
            "clade_identities": {name: info.mean_identity for name, info in self.clades.items()},
        }

    def analyze_geographic_distribution(self) -> dict:
        """Analyze geographic distribution of clades.

        Returns:
            Dictionary with geographic analysis results
        """
        print("Analyzing geographic distribution...")

        # Overall country distribution
        all_countries = Counter()
        clade_countries = defaultdict(Counter)

        for acc, meta in self.metadata.items():
            if meta.country:
                all_countries[meta.country] += 1
                if meta.clade:
                    clade_countries[meta.clade][meta.country] += 1

        print(f"\nOverall distribution ({len(all_countries)} countries):")
        for country, count in all_countries.most_common(10):
            print(f"  {country}: {count}")

        # Per-clade geographic analysis
        geographic_signal = {}

        for clade_name, country_counts in clade_countries.items():
            total = sum(country_counts.values())
            top_country, top_count = country_counts.most_common(1)[0] if country_counts else (None, 0)

            # Geographic concentration (how focused is this clade?)
            concentration = top_count / total if total > 0 else 0

            # Geographic entropy
            entropy = 0
            for count in country_counts.values():
                p = count / total
                entropy -= p * math.log2(p) if p > 0 else 0

            geographic_signal[clade_name] = {
                "n_countries": len(country_counts),
                "top_country": top_country,
                "concentration": concentration,
                "entropy": entropy,
                "distribution": dict(country_counts.most_common(5)),
            }

            print(f"\n{clade_name} geographic profile:")
            print(f"  Countries: {len(country_counts)}")
            print(f"  Top: {top_country} ({concentration:.1%})")
            print(f"  Entropy: {entropy:.2f} bits")

        # Save geographic data
        geo_data = {
            "overall_distribution": dict(all_countries),
            "clade_distribution": {
                k: dict(v) for k, v in clade_countries.items()
            },
            "geographic_signal": geographic_signal,
        }

        self._save_json(geo_data, RESULTS_DIR / "geographic_distribution.json")

        return geographic_signal

    def analyze_temporal_distribution(self) -> dict:
        """Analyze temporal distribution of clades.

        Returns:
            Dictionary with temporal analysis results
        """
        print("Analyzing temporal distribution...")

        # Overall year distribution
        all_years = Counter()
        clade_years = defaultdict(Counter)

        for acc, meta in self.metadata.items():
            if meta.year:
                all_years[meta.year] += 1
                if meta.clade:
                    clade_years[meta.clade][meta.year] += 1

        if not all_years:
            print("No temporal data available")
            return {"error": "no_temporal_data"}

        print(f"\nOverall temporal range: {min(all_years)} - {max(all_years)}")

        # Per-clade temporal analysis
        temporal_signal = {}

        for clade_name, year_counts in clade_years.items():
            if not year_counts:
                continue

            years_list = []
            for year, count in year_counts.items():
                years_list.extend([year] * count)

            temporal_signal[clade_name] = {
                "year_range": (min(year_counts), max(year_counts)),
                "mean_year": float(np.mean(years_list)),
                "median_year": float(np.median(years_list)),
                "n_years": len(year_counts),
                "distribution": dict(sorted(year_counts.items())),
            }

            print(f"\n{clade_name} temporal profile:")
            print(f"  Range: {min(year_counts)} - {max(year_counts)}")
            print(f"  Mean year: {temporal_signal[clade_name]['mean_year']:.1f}")

        # Check for temporal stratification
        clade_means = {
            k: v["mean_year"] for k, v in temporal_signal.items()
        }

        if len(clade_means) >= 2:
            mean_years = list(clade_means.values())
            year_spread = max(mean_years) - min(mean_years)
            print(f"\nTemporal spread between clades: {year_spread:.1f} years")

        # Save temporal data
        temporal_data = {
            "overall_distribution": dict(all_years),
            "clade_distribution": {
                k: dict(sorted(v.items())) for k, v in clade_years.items()
            },
            "temporal_signal": temporal_signal,
        }

        self._save_json(temporal_data, RESULTS_DIR / "temporal_distribution.json")

        return temporal_signal

    def analyze_per_clade_conservation(self) -> dict:
        """Analyze sequence conservation within each clade.

        Identifies conserved regions suitable for clade-specific primer design.

        Returns:
            Dictionary with conservation analysis results
        """
        print("Analyzing per-clade conservation...")

        conservation_results = {}

        for clade_name, clade_info in self.clades.items():
            # Get sequences in this clade
            members = [
                acc for acc, meta in self.metadata.items()
                if meta.clade == clade_name
            ]

            if len(members) < 2:
                print(f"  {clade_name}: Too few sequences for conservation analysis")
                continue

            print(f"\n  Analyzing {clade_name} ({len(members)} sequences)...")

            # Get sequences
            seqs = [self.sequences[acc] for acc in members]

            # Find minimum length for alignment
            min_len = min(len(s) for s in seqs)
            print(f"    Minimum sequence length: {min_len}")

            # Compute per-position entropy
            entropy_profile = self._compute_entropy_profile(seqs, min_len)

            # Find conserved windows
            window_size = 25  # Primer length target
            conserved_windows = self._find_conserved_windows(
                entropy_profile,
                window_size=window_size,
                max_entropy=0.3,
            )

            # Map to gene regions
            annotated_windows = []
            for start, end, mean_entropy in conserved_windows:
                gene = self._get_gene_at_position(start)
                annotated_windows.append({
                    "start": start,
                    "end": end,
                    "mean_entropy": mean_entropy,
                    "gene": gene,
                })

            conservation_results[clade_name] = {
                "n_sequences": len(members),
                "min_length": min_len,
                "mean_entropy": float(np.mean(entropy_profile)),
                "median_entropy": float(np.median(entropy_profile)),
                "conserved_windows": annotated_windows[:20],  # Top 20
                "entropy_percentiles": {
                    "p10": float(np.percentile(entropy_profile, 10)),
                    "p50": float(np.percentile(entropy_profile, 50)),
                    "p90": float(np.percentile(entropy_profile, 90)),
                },
            }

            # Update clade info
            self.clades[clade_name].conserved_regions = annotated_windows[:10]

            print(f"    Mean entropy: {np.mean(entropy_profile):.3f}")
            print(f"    Found {len(conserved_windows)} conserved windows (entropy < 0.3)")
            if annotated_windows:
                best = annotated_windows[0]
                print(f"    Best window: {best['gene']} {best['start']}-{best['end']} (entropy={best['mean_entropy']:.3f})")

        # Save conservation data
        self._save_json(conservation_results, RESULTS_DIR / "per_clade_conservation.json")

        return conservation_results

    def _compute_entropy_profile(self, sequences: list[str], length: int) -> np.ndarray:
        """Compute Shannon entropy at each position."""
        entropy = np.zeros(length)

        for pos in range(length):
            # Count nucleotides at this position
            counts = Counter()
            valid = 0
            for seq in sequences:
                if pos < len(seq):
                    nt = seq[pos].upper()
                    if nt in "ACGT":
                        counts[nt] += 1
                        valid += 1

            # Compute entropy
            if valid > 0:
                for nt_count in counts.values():
                    p = nt_count / valid
                    if p > 0:
                        entropy[pos] -= p * math.log2(p)

        return entropy

    def _find_conserved_windows(
        self,
        entropy: np.ndarray,
        window_size: int,
        max_entropy: float,
    ) -> list[tuple[int, int, float]]:
        """Find windows with low average entropy."""
        windows = []

        for start in range(len(entropy) - window_size + 1):
            end = start + window_size
            window_entropy = np.mean(entropy[start:end])

            if window_entropy < max_entropy:
                windows.append((start, end, float(window_entropy)))

        # Sort by entropy (best first)
        windows.sort(key=lambda x: x[2])

        return windows

    def _get_gene_at_position(self, position: int) -> str:
        """Get the gene name at a given position."""
        for gene, (start, end) in self.GENE_REGIONS.items():
            if start <= position <= end:
                return gene
        return "intergenic"

    def select_clade_representatives(self) -> dict:
        """Select representative sequences for each clade.

        Selects the sequence with minimum average distance to all others in clade.

        Returns:
            Dictionary with representative information
        """
        print("Selecting clade representatives...")

        accessions = list(self.sequences.keys())
        representatives = {}

        for clade_name, clade_info in self.clades.items():
            # Get members of this clade
            members = [
                acc for acc, meta in self.metadata.items()
                if meta.clade == clade_name
            ]

            if not members:
                continue

            if len(members) == 1:
                rep = members[0]
            else:
                # Find member with minimum average distance to others
                member_indices = [accessions.index(acc) for acc in members]

                min_avg_dist = float('inf')
                rep = members[0]

                for acc in members:
                    idx = accessions.index(acc)
                    avg_dist = np.mean([
                        self.distance_matrix[idx, other_idx]
                        for other_idx in member_indices
                        if other_idx != idx
                    ])

                    if avg_dist < min_avg_dist:
                        min_avg_dist = avg_dist
                        rep = acc

            # Mark as representative
            self.metadata[rep].is_representative = True
            self.clades[clade_name].representative_accession = rep

            representatives[clade_name] = {
                "accession": rep,
                "country": self.metadata[rep].country,
                "year": self.metadata[rep].year,
                "length": self.metadata[rep].length,
            }

            print(f"  {clade_name}: {rep} ({self.metadata[rep].country}, {self.metadata[rep].year})")

        # Save representative sequences to FASTA
        rep_seqs = []
        for clade_name in sorted(representatives.keys()):
            acc = representatives[clade_name]["accession"]
            rep_seqs.append(SeqRecord(
                Seq(self.sequences[acc]),
                id=f"{acc}|{clade_name}",
                description=f"DENV-4 {clade_name} representative",
            ))

        if HAS_BIOPYTHON and rep_seqs:
            fasta_path = RESULTS_DIR / "clade_representatives.fasta"
            SeqIO.write(rep_seqs, fasta_path, "fasta")
            print(f"\nRepresentatives saved to: {fasta_path}")

        # Save representative info
        self._save_json(representatives, RESULTS_DIR / "clade_representatives.json")

        return representatives

    def generate_visualizations(self) -> dict:
        """Generate visualization figures.

        Returns:
            Dictionary with figure paths
        """
        if not HAS_MATPLOTLIB:
            return {"error": "matplotlib_not_available"}

        figures = {}

        # 1. Distance matrix heatmap
        print("  Generating distance matrix heatmap...")
        fig, ax = plt.subplots(figsize=(12, 10))

        # Sort by clade
        accessions = list(self.sequences.keys())
        sorted_indices = sorted(
            range(len(accessions)),
            key=lambda i: self.metadata[accessions[i]].clade or "ZZZ"
        )

        sorted_matrix = self.distance_matrix[np.ix_(sorted_indices, sorted_indices)]

        im = ax.imshow(1 - sorted_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title("DENV-4 Pairwise Sequence Identity (Sorted by Clade)")
        plt.colorbar(im, ax=ax, label="Sequence Identity")

        # Add clade boundaries
        current_clade = None
        for i, idx in enumerate(sorted_indices):
            clade = self.metadata[accessions[idx]].clade
            if clade != current_clade:
                if current_clade is not None:
                    ax.axhline(i - 0.5, color='black', linewidth=1)
                    ax.axvline(i - 0.5, color='black', linewidth=1)
                current_clade = clade

        fig.tight_layout()
        path = RESULTS_DIR / "distance_matrix_heatmap.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures["distance_matrix"] = str(path)

        # 2. Dendrogram
        print("  Generating dendrogram...")
        linkage_path = DATA_DIR / "linkage.npy"
        if linkage_path.exists():
            linkage = np.load(linkage_path)

            fig, ax = plt.subplots(figsize=(16, 8))

            # Create color function based on clades
            clade_colors = {
                name: info.color for name, info in self.clades.items()
            }

            cluster.hierarchy.dendrogram(
                linkage,
                ax=ax,
                leaf_rotation=90,
                leaf_font_size=6,
                no_labels=len(accessions) > 50,
            )

            ax.set_title("DENV-4 Phylogenetic Clustering")
            ax.set_xlabel("Sequence")
            ax.set_ylabel("Distance")

            # Add legend
            legend_elements = [
                Patch(facecolor=info.color, label=f"{name} (n={info.size})")
                for name, info in sorted(self.clades.items())
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            fig.tight_layout()
            path = RESULTS_DIR / "dendrogram.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            figures["dendrogram"] = str(path)

        # 3. Geographic distribution
        print("  Generating geographic distribution...")
        fig, ax = plt.subplots(figsize=(12, 6))

        clade_country_data = defaultdict(lambda: defaultdict(int))
        for acc, meta in self.metadata.items():
            if meta.clade and meta.country:
                clade_country_data[meta.clade][meta.country] += 1

        # Get all countries and clades
        all_countries = sorted(set(
            meta.country for meta in self.metadata.values() if meta.country
        ))[:15]  # Top 15
        clade_names = sorted(self.clades.keys())

        # Create stacked bar chart
        x = np.arange(len(all_countries))
        width = 0.8
        bottom = np.zeros(len(all_countries))

        for clade_name in clade_names:
            counts = [clade_country_data[clade_name].get(c, 0) for c in all_countries]
            ax.bar(
                x, counts, width,
                label=clade_name,
                bottom=bottom,
                color=self.clades[clade_name].color,
            )
            bottom += counts

        ax.set_xlabel("Country")
        ax.set_ylabel("Number of Sequences")
        ax.set_title("DENV-4 Geographic Distribution by Clade")
        ax.set_xticks(x)
        ax.set_xticklabels(all_countries, rotation=45, ha='right')
        ax.legend()

        fig.tight_layout()
        path = RESULTS_DIR / "geographic_distribution.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures["geographic"] = str(path)

        # 4. Temporal distribution
        print("  Generating temporal distribution...")
        fig, ax = plt.subplots(figsize=(12, 6))

        clade_year_data = defaultdict(lambda: defaultdict(int))
        for acc, meta in self.metadata.items():
            if meta.clade and meta.year:
                clade_year_data[meta.clade][meta.year] += 1

        # Get year range
        all_years = sorted(set(
            meta.year for meta in self.metadata.values() if meta.year
        ))

        if all_years:
            for clade_name in clade_names:
                years = sorted(clade_year_data[clade_name].keys())
                counts = [clade_year_data[clade_name][y] for y in years]
                ax.plot(years, counts, 'o-', label=clade_name, color=self.clades[clade_name].color)

            ax.set_xlabel("Year")
            ax.set_ylabel("Number of Sequences")
            ax.set_title("DENV-4 Temporal Distribution by Clade")
            ax.legend()

            fig.tight_layout()
            path = RESULTS_DIR / "temporal_distribution.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            figures["temporal"] = str(path)

        # 5. Clade composition pie chart
        print("  Generating clade composition...")
        fig, ax = plt.subplots(figsize=(8, 8))

        sizes = [info.size for info in self.clades.values()]
        labels = [f"{name}\n(n={info.size})" for name, info in self.clades.items()]
        colors = [info.color for info in self.clades.values()]

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title("DENV-4 Clade Composition")

        path = RESULTS_DIR / "clade_composition.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures["composition"] = str(path)

        print(f"  Generated {len(figures)} figures")

        return figures

    def generate_report(self, results: dict) -> dict:
        """Generate comprehensive markdown report.

        Args:
            results: Analysis results dictionary

        Returns:
            Dictionary with report path
        """
        report_path = RESULTS_DIR / "DENV4_PHYLOGENETIC_REPORT.md"

        # Build report content
        lines = [
            "# DENV-4 Comprehensive Phylogenetic Analysis Report",
            "",
            f"**Doc-Type:** Phylogenetic Analysis Report · Version 1.0 · {datetime.datetime.now().strftime('%Y-%m-%d')} · AI Whisperers",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"This report presents a comprehensive phylogenetic analysis of **{len(self.sequences)} DENV-4 complete genomes** "
            f"from NCBI. The analysis identifies **{len(self.clades)} major clades** with distinct geographic and temporal patterns.",
            "",
            "### Key Findings",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Sequences | {len(self.sequences)} |",
            f"| Number of Clades | {len(self.clades)} |",
        ]

        # Add distance statistics if available
        for step in results.get("pipeline_steps", []):
            if step["step"] == "distances" and "mean_identity" in step.get("result", {}):
                stats = step["result"]
                lines.extend([
                    f"| Mean Pairwise Identity | {stats['mean_identity']:.1%} |",
                    f"| Identity Range | {stats['min_identity']:.1%} - {stats['max_identity']:.1%} |",
                ])

        lines.extend([
            "",
            "---",
            "",
            "## Clade Overview",
            "",
            "| Clade | Size | Mean Identity | Top Country | Year Range |",
            "|-------|------|---------------|-------------|------------|",
        ])

        for name, info in sorted(self.clades.items(), key=lambda x: -x[1].size):
            top_country = info.countries[0][0] if info.countries else "Unknown"
            year_range = f"{info.year_range[0]}-{info.year_range[1]}" if info.year_range[0] else "Unknown"
            lines.append(
                f"| {name} | {info.size} | {info.mean_identity:.1%} | {top_country} | {year_range} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## Geographic Distribution",
            "",
        ])

        # Geographic details per clade
        for name, info in sorted(self.clades.items()):
            lines.append(f"### {name}")
            lines.append("")
            lines.append(f"**Size:** {info.size} sequences")
            lines.append("")
            if info.countries:
                lines.append("**Countries:**")
                for country, count in info.countries[:5]:
                    lines.append(f"- {country}: {count}")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## Conserved Regions for Primer Design",
            "",
            "Top conserved windows per clade (suitable for genotype-specific primers):",
            "",
        ])

        for name, info in sorted(self.clades.items()):
            lines.append(f"### {name}")
            lines.append("")
            if info.conserved_regions:
                lines.append("| Gene | Start | End | Entropy |")
                lines.append("|------|-------|-----|---------|")
                for region in info.conserved_regions[:5]:
                    lines.append(
                        f"| {region['gene']} | {region['start']} | {region['end']} | {region['mean_entropy']:.3f} |"
                    )
            else:
                lines.append("*No conserved regions identified*")
            lines.append("")

        lines.extend([
            "---",
            "",
            "## Clade Representatives",
            "",
            "Selected representative sequences for each clade (medoid selection):",
            "",
            "| Clade | Accession | Country | Year | Length |",
            "|-------|-----------|---------|------|--------|",
        ])

        for name, info in sorted(self.clades.items()):
            if info.representative_accession:
                meta = self.metadata[info.representative_accession]
                lines.append(
                    f"| {name} | {info.representative_accession} | "
                    f"{meta.country or 'Unknown'} | {meta.year or 'Unknown'} | {meta.length} bp |"
                )

        lines.extend([
            "",
            "---",
            "",
            "## Implications for Primer Design",
            "",
            "### Challenge",
            "",
            "DENV-4 shows extensive cryptic diversity requiring **multiplexed detection**:",
            "",
            "1. Within-clade identity is high enough for consensus primers",
            "2. Between-clade identity requires separate primer pairs",
            "3. Each clade has distinct conserved regions",
            "",
            "### Recommended Strategy",
            "",
            "1. **Design clade-specific primers** for each identified clade",
            "2. **Use conserved windows** identified above as primer binding sites",
            "3. **Multiplex all primers** in single reaction with staggered amplicons",
            "4. **Monitor quarterly** for new clade emergence",
            "",
            "---",
            "",
            "## Files Generated",
            "",
            "| File | Description |",
            "|------|-------------|",
            f"| `{RESULTS_DIR.name}/phylogenetic_analysis_results.json` | Complete analysis results |",
            f"| `{RESULTS_DIR.name}/clade_representatives.fasta` | Representative sequences |",
            f"| `{RESULTS_DIR.name}/geographic_distribution.json` | Geographic data |",
            f"| `{RESULTS_DIR.name}/temporal_distribution.json` | Temporal data |",
            f"| `{RESULTS_DIR.name}/per_clade_conservation.json` | Conservation data |",
            f"| `{DATA_DIR.name}/distance_matrix.npy` | Pairwise distance matrix |",
            f"| `{DATA_DIR.name}/linkage.npy` | Hierarchical clustering linkage |",
            "",
            "---",
            "",
            f"*Analysis completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "*IICS-UNA Arbovirus Surveillance Program*",
            "",
        ])

        # Write report
        report_path.write_text("\n".join(lines))
        print(f"Report saved to: {report_path}")

        return {"report_path": str(report_path)}

    def _save_json(self, data: dict, path: Path) -> None:
        """Save dictionary to JSON with type conversion."""
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif hasattr(obj, '__dict__'):
                return convert(obj.__dict__)
            return obj

        with open(path, "w") as f:
            json.dump(convert(data), f, indent=2, default=str)


def main():
    """Main entry point."""
    print()
    print("=" * 80)
    print("DENV-4 PHYLOGENETIC ANALYSIS PIPELINE")
    print("=" * 80)
    print()
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print()

    # Check dependencies
    if not HAS_BIOPYTHON:
        print("ERROR: Biopython is required. Install with: pip install biopython")
        return

    # Run analysis
    analyzer = DENV4PhylogeneticAnalyzer(cache_enabled=True)
    results = analyzer.run_full_analysis()

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(f"Results saved to: {RESULTS_DIR}")
    print()


if __name__ == "__main__":
    main()
