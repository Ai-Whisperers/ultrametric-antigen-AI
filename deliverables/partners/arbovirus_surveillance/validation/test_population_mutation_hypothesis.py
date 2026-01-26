# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Population-Driven Mutation Rate Hypothesis - Falsification Test.

CONJECTURE:
Viral strains with lower prevalence (DENV-4, HIV-2) show higher mutation rates
due to human population-related variables:
- Population bottlenecks in transmission chains
- Founder effects from geographic restriction
- Different selective pressures in smaller host populations

FALSIFICATION CRITERIA:
1. If DENV-4 is NOT geographically restricted → rejects bottleneck hypothesis
2. If DENV-4 strains are NOT temporally clustered → rejects recent divergence
3. If entropy does NOT correlate with prevalence → rejects population-size effect
4. If HIV-2 shows DIFFERENT pattern than DENV-4 → rejects universal mechanism

METHODOLOGY:
1. Download metadata (country, year) for all Dengue serotypes
2. Compute geographic diversity (number of countries, entropy of distribution)
3. Analyze temporal patterns (collection year distribution)
4. Correlate prevalence with sequence variability
5. Compare with HIV-1/HIV-2 patterns if data available

Usage:
    python validation/test_population_mutation_hypothesis.py
"""

from __future__ import annotations

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

from src.constants import ARBOVIRUS_TAXIDS

# Try to import BioPython
try:
    from Bio import Entrez
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False


@dataclass
class SerotypeMetadata:
    """Metadata for a Dengue serotype."""
    serotype: str
    total_genomes: int
    countries: list[str]
    country_counts: dict[str, int]
    years: list[int]
    year_counts: dict[int, int]
    geographic_entropy: float  # Higher = more widespread
    temporal_span: int  # Years between oldest and newest
    mean_year: float


@dataclass
class HypothesisResult:
    """Result of hypothesis test."""
    hypothesis: str
    prediction: str
    observation: str
    falsified: bool
    evidence: str


def fetch_serotype_metadata(
    serotype: str,
    max_records: int = 500,
    email: str = "user@example.com",
) -> Optional[SerotypeMetadata]:
    """Fetch metadata for a Dengue serotype from NCBI.

    Args:
        serotype: DENV-1, DENV-2, DENV-3, or DENV-4
        max_records: Maximum records to fetch
        email: Email for NCBI Entrez

    Returns:
        SerotypeMetadata or None
    """
    if not BIOPYTHON_AVAILABLE:
        return None

    taxid = ARBOVIRUS_TAXIDS.get(serotype)
    if not taxid:
        return None

    Entrez.email = email

    try:
        # Search for complete genomes
        query = f"txid{taxid}[Organism] AND complete genome[Title]"

        handle = Entrez.esearch(
            db="nucleotide",
            term=query,
            retmax=max_records,
            usehistory="y"
        )
        search_results = Entrez.read(handle)
        handle.close()

        total = int(search_results["Count"])
        ids = search_results["IdList"]

        if not ids:
            return None

        print(f"  {serotype}: {total} total genomes, fetching metadata for {len(ids)}...")

        # Fetch summaries in batches
        countries = []
        years = []

        batch_size = 100
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]

            handle = Entrez.esummary(
                db="nucleotide",
                id=",".join(batch_ids)
            )
            summaries = Entrez.read(handle)
            handle.close()

            for summary in summaries:
                # Extract country from title or extra field
                title = summary.get("Title", "")
                extra = summary.get("Extra", "")

                # Try to extract country
                country = None
                for field in [title, extra]:
                    field_lower = field.lower()
                    # Common country patterns
                    country_keywords = [
                        "brazil", "india", "thailand", "vietnam", "philippines",
                        "indonesia", "singapore", "malaysia", "china", "taiwan",
                        "mexico", "colombia", "peru", "venezuela", "puerto rico",
                        "usa", "united states", "australia", "japan", "korea",
                        "sri lanka", "bangladesh", "pakistan", "myanmar", "cambodia",
                        "laos", "nicaragua", "honduras", "guatemala", "el salvador",
                        "costa rica", "panama", "cuba", "dominican republic", "haiti",
                        "jamaica", "trinidad", "paraguay", "argentina", "bolivia",
                        "ecuador", "french guiana", "guyana", "suriname", "uruguay"
                    ]
                    for kw in country_keywords:
                        if kw in field_lower:
                            country = kw.title()
                            break
                    if country:
                        break

                if country:
                    countries.append(country)

                # Try to extract year
                create_date = summary.get("CreateDate", "")
                if create_date and "/" in create_date:
                    try:
                        year = int(create_date.split("/")[0])
                        if 1990 <= year <= 2030:
                            years.append(year)
                    except:
                        pass

            time.sleep(0.3)  # Rate limiting

        # Compute statistics
        country_counts = dict(Counter(countries))
        year_counts = dict(Counter(years))

        # Geographic entropy
        if countries:
            total_countries = len(countries)
            geo_entropy = 0.0
            for count in country_counts.values():
                p = count / total_countries
                if p > 0:
                    geo_entropy -= p * math.log2(p)
        else:
            geo_entropy = 0.0

        # Temporal statistics
        if years:
            temporal_span = max(years) - min(years)
            mean_year = sum(years) / len(years)
        else:
            temporal_span = 0
            mean_year = 0

        return SerotypeMetadata(
            serotype=serotype,
            total_genomes=total,
            countries=list(set(countries)),
            country_counts=country_counts,
            years=sorted(set(years)),
            year_counts=year_counts,
            geographic_entropy=geo_entropy,
            temporal_span=temporal_span,
            mean_year=mean_year,
        )

    except Exception as e:
        print(f"  Error fetching {serotype}: {e}")
        return None


def test_geographic_restriction(metadata: dict[str, SerotypeMetadata]) -> HypothesisResult:
    """Test if DENV-4 is geographically restricted (bottleneck hypothesis)."""

    denv4 = metadata.get("DENV-4")
    others = [m for k, m in metadata.items() if k != "DENV-4" and m]

    if not denv4 or not others:
        return HypothesisResult(
            hypothesis="DENV-4 is geographically restricted",
            prediction="DENV-4 should have lower geographic entropy",
            observation="Insufficient data",
            falsified=False,
            evidence="Could not compare due to missing data"
        )

    avg_other_entropy = sum(m.geographic_entropy for m in others) / len(others)
    avg_other_countries = sum(len(m.countries) for m in others) / len(others)

    # DENV-4 should have LOWER entropy if bottleneck hypothesis is true
    entropy_ratio = denv4.geographic_entropy / avg_other_entropy if avg_other_entropy > 0 else 0
    country_ratio = len(denv4.countries) / avg_other_countries if avg_other_countries > 0 else 0

    # Falsified if DENV-4 has similar or higher geographic diversity
    falsified = entropy_ratio >= 0.8 and country_ratio >= 0.8

    return HypothesisResult(
        hypothesis="DENV-4 geographic restriction (bottleneck)",
        prediction="DENV-4 should have <80% of other serotypes' geographic entropy",
        observation=f"DENV-4 entropy={denv4.geographic_entropy:.2f} vs avg={avg_other_entropy:.2f} (ratio={entropy_ratio:.2f}), "
                   f"countries={len(denv4.countries)} vs avg={avg_other_countries:.1f} (ratio={country_ratio:.2f})",
        falsified=falsified,
        evidence="SUPPORTED" if not falsified else "FALSIFIED: DENV-4 is NOT geographically restricted"
    )


def test_recent_divergence(metadata: dict[str, SerotypeMetadata]) -> HypothesisResult:
    """Test if DENV-4 diverged more recently (temporal clustering)."""

    denv4 = metadata.get("DENV-4")
    others = [m for k, m in metadata.items() if k != "DENV-4" and m]

    if not denv4 or not others or not denv4.years:
        return HypothesisResult(
            hypothesis="DENV-4 is more recently diverged",
            prediction="DENV-4 should have more recent mean collection year",
            observation="Insufficient data",
            falsified=False,
            evidence="Could not compare due to missing data"
        )

    avg_other_mean_year = sum(m.mean_year for m in others if m.mean_year > 0) / len([m for m in others if m.mean_year > 0])

    # DENV-4 should have MORE RECENT mean year if recent divergence is true
    year_diff = denv4.mean_year - avg_other_mean_year

    # Falsified if DENV-4 is not significantly more recent (within 2 years)
    falsified = year_diff < 2.0

    return HypothesisResult(
        hypothesis="DENV-4 recent divergence",
        prediction="DENV-4 mean collection year should be >2 years more recent",
        observation=f"DENV-4 mean year={denv4.mean_year:.1f} vs avg={avg_other_mean_year:.1f} (diff={year_diff:+.1f} years)",
        falsified=falsified,
        evidence="SUPPORTED" if not falsified else "FALSIFIED: DENV-4 is NOT more recently sampled"
    )


def test_prevalence_variability_correlation(
    metadata: dict[str, SerotypeMetadata],
    entropy_data: dict[str, float],
) -> HypothesisResult:
    """Test if lower prevalence correlates with higher variability."""

    # Need at least 3 points for meaningful correlation
    valid_serotypes = [s for s in metadata if s in entropy_data and metadata[s]]

    if len(valid_serotypes) < 3:
        return HypothesisResult(
            hypothesis="Lower prevalence correlates with higher mutation rate",
            prediction="Negative correlation between genome count and entropy",
            observation="Insufficient data points",
            falsified=False,
            evidence="Could not compute correlation"
        )

    # Get prevalence (genome count) and entropy for each serotype
    prevalences = [metadata[s].total_genomes for s in valid_serotypes]
    entropies = [entropy_data[s] for s in valid_serotypes]

    # Compute Spearman correlation
    from scipy.stats import spearmanr

    try:
        rho, pvalue = spearmanr(prevalences, entropies)
    except:
        # Manual Spearman if scipy not available
        n = len(prevalences)
        rank_prev = [sorted(prevalences).index(x) + 1 for x in prevalences]
        rank_ent = [sorted(entropies).index(x) + 1 for x in entropies]
        d_squared = sum((rp - re)**2 for rp, re in zip(rank_prev, rank_ent))
        rho = 1 - (6 * d_squared) / (n * (n**2 - 1))
        pvalue = 0.5  # Placeholder

    # Hypothesis predicts NEGATIVE correlation (lower prevalence = higher entropy)
    # Falsified if correlation is positive or weakly negative
    falsified = rho > -0.5

    data_points = ", ".join(f"{s}({metadata[s].total_genomes}, {entropy_data[s]:.3f})"
                           for s in valid_serotypes)

    return HypothesisResult(
        hypothesis="Prevalence-variability inverse correlation",
        prediction="Spearman ρ < -0.5 (lower prevalence = higher entropy)",
        observation=f"ρ = {rho:.3f}, p = {pvalue:.3f}. Data: {data_points}",
        falsified=falsified,
        evidence="SUPPORTED" if not falsified else f"FALSIFIED: Correlation is {rho:.3f}, not strongly negative"
    )


def run_falsification(cache_dir: Path, use_cache: bool = True) -> dict:
    """Run complete falsification test."""

    import datetime

    print("=" * 70)
    print("POPULATION-DRIVEN MUTATION HYPOTHESIS - FALSIFICATION TEST")
    print("=" * 70)
    print()
    print("CONJECTURE: Strains with lower prevalence (DENV-4) show higher")
    print("mutation rates due to population bottlenecks or founder effects.")
    print()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "dengue_metadata.json"

    serotypes = ["DENV-1", "DENV-2", "DENV-3", "DENV-4"]
    metadata = {}

    # Load or fetch metadata
    if cache_file.exists() and use_cache:
        print("Loading cached metadata...")
        with open(cache_file) as f:
            cached = json.load(f)
        for serotype in serotypes:
            if serotype in cached:
                data = cached[serotype]
                metadata[serotype] = SerotypeMetadata(**data)
    else:
        print("Fetching metadata from NCBI...")
        for serotype in serotypes:
            meta = fetch_serotype_metadata(serotype, max_records=300)
            if meta:
                metadata[serotype] = meta
            time.sleep(1)

        # Cache results
        with open(cache_file, "w") as f:
            json.dump({k: asdict(v) for k, v in metadata.items()}, f, indent=2)

    print()
    print("-" * 70)
    print("SEROTYPE METADATA")
    print("-" * 70)
    print()

    for serotype in serotypes:
        if serotype in metadata:
            m = metadata[serotype]
            print(f"{serotype}:")
            print(f"  Total genomes: {m.total_genomes}")
            print(f"  Countries: {len(m.countries)} (entropy={m.geographic_entropy:.2f})")
            print(f"  Years: {min(m.years) if m.years else 'N/A'}-{max(m.years) if m.years else 'N/A'} (mean={m.mean_year:.1f})")
            print()

    # Load entropy data from previous analysis
    entropy_file = cache_dir.parent / "validation" / "dengue_strain_variation_report.json"
    entropy_data = {}

    if entropy_file.exists():
        with open(entropy_file) as f:
            variation_report = json.load(f)

        # Average entropy per serotype
        for primer in variation_report.get("primer_conservation", []):
            serotype = primer.get("serotype")
            entropy = primer.get("mean_entropy", 0)
            if serotype not in entropy_data:
                entropy_data[serotype] = []
            entropy_data[serotype].append(entropy)

        entropy_data = {k: sum(v)/len(v) for k, v in entropy_data.items()}

    print("-" * 70)
    print("HYPOTHESIS TESTS")
    print("-" * 70)
    print()

    results = []

    # Test 1: Geographic restriction
    result1 = test_geographic_restriction(metadata)
    results.append(result1)
    print(f"TEST 1: {result1.hypothesis}")
    print(f"  Prediction: {result1.prediction}")
    print(f"  Observation: {result1.observation}")
    print(f"  Result: {result1.evidence}")
    print()

    # Test 2: Recent divergence
    result2 = test_recent_divergence(metadata)
    results.append(result2)
    print(f"TEST 2: {result2.hypothesis}")
    print(f"  Prediction: {result2.prediction}")
    print(f"  Observation: {result2.observation}")
    print(f"  Result: {result2.evidence}")
    print()

    # Test 3: Prevalence-variability correlation
    if entropy_data:
        result3 = test_prevalence_variability_correlation(metadata, entropy_data)
        results.append(result3)
        print(f"TEST 3: {result3.hypothesis}")
        print(f"  Prediction: {result3.prediction}")
        print(f"  Observation: {result3.observation}")
        print(f"  Result: {result3.evidence}")
        print()

    print("=" * 70)
    print("FALSIFICATION SUMMARY")
    print("=" * 70)
    print()

    falsified_count = sum(1 for r in results if r.falsified)
    total_tests = len(results)

    for i, r in enumerate(results, 1):
        status = "FALSIFIED" if r.falsified else "SUPPORTED"
        print(f"  Test {i}: {status}")

    print()

    if falsified_count == 0:
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║  HYPOTHESIS SUPPORTED - No tests falsified                   ║")
        print("  ║  Population bottleneck mechanism remains plausible           ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
        overall = "SUPPORTED"
    elif falsified_count == total_tests:
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║  HYPOTHESIS FALSIFIED - All tests rejected                   ║")
        print("  ║  DENV-4 variability NOT explained by population dynamics     ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
        overall = "FALSIFIED"
    else:
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print(f"  ║  HYPOTHESIS PARTIALLY FALSIFIED - {falsified_count}/{total_tests} tests rejected         ║")
        print("  ║  Alternative mechanisms may contribute to DENV-4 variability║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
        overall = "PARTIALLY_FALSIFIED"

    print()

    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "hypothesis": "Population-driven mutation rate",
        "overall_result": overall,
        "tests_falsified": falsified_count,
        "total_tests": total_tests,
        "results": [asdict(r) for r in results],
        "metadata": {k: asdict(v) for k, v in metadata.items()},
        "entropy_data": entropy_data,
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Population-Driven Mutation Hypothesis Falsification"
    )
    parser.add_argument("--use-cache", action="store_true", help="Use cached data")
    parser.add_argument("--force", action="store_true", help="Force re-fetch")

    args = parser.parse_args()

    validation_dir = Path(__file__).parent
    cache_dir = validation_dir.parent / "data"

    results = run_falsification(cache_dir, use_cache=not args.force)

    # Save results - convert numpy types to Python types for JSON
    def convert_types(obj):
        """Recursively convert numpy types to Python types."""
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif isinstance(obj, bool):
            return bool(obj)
        return obj

    output_path = validation_dir / "population_hypothesis_results.json"
    with open(output_path, "w") as f:
        json.dump(convert_types(results), f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
