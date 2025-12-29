#!/usr/bin/env python3
"""
AlphaFold 3 Prediction Analysis

Analyzes AF3 predictions to compare:
1. Native vs citrullinated structures
2. HLA binding affinity differences
3. Correlation with hyperbolic encoder predictions

Author: Research Team
Date: December 2025
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import statistics

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
VALIDATION_DIR = SCRIPT_DIR.parent
RESEARCH_DIR = VALIDATION_DIR.parent

DISEASE_DIRS = {
    "rheumatoid_arthritis": RESEARCH_DIR / "rheumatoid_arthritis" / "results" / "alphafold3",
    "hiv": RESEARCH_DIR / "hiv" / "results" / "alphafold3",
    "alzheimers": RESEARCH_DIR / "neurodegeneration" / "alzheimers" / "results" / "alphafold3",
}


@dataclass
class PredictionResult:
    """Parsed AlphaFold 3 prediction result."""
    job_id: str
    prediction_dir: Path
    iptm: float = 0.0
    ptm: float = 0.0
    ranking_score: float = 0.0
    has_clash: bool = False
    chain_pair_iptm: dict = field(default_factory=dict)
    chain_pair_pae: dict = field(default_factory=dict)
    num_models: int = 0


@dataclass
class ComparisonResult:
    """Native vs modified comparison."""
    epitope: str
    protein: str
    native_iptm: float
    modified_iptm: float
    iptm_change: float
    iptm_change_pct: float
    hla_allele: Optional[str] = None
    binding_improved: bool = False
    native_ranking: float = 0.0
    modified_ranking: float = 0.0


# ============================================================================
# PARSING
# ============================================================================


def parse_prediction(pred_dir: Path) -> Optional[PredictionResult]:
    """Parse AF3 prediction output directory."""
    if not pred_dir.exists() or not pred_dir.is_dir():
        return None

    result = PredictionResult(
        job_id=pred_dir.name,
        prediction_dir=pred_dir,
    )

    # Find confidence files
    conf_files = sorted(pred_dir.glob("*_summary_confidences_*.json"))

    if not conf_files:
        return None

    # Parse all models, use best one
    best_ranking = -1
    for conf_file in conf_files:
        try:
            with open(conf_file) as f:
                data = json.load(f)

            ranking = data.get("ranking_score", 0)
            if ranking > best_ranking:
                best_ranking = ranking
                result.iptm = data.get("iptm", 0)
                result.ptm = data.get("ptm", 0)
                result.ranking_score = ranking
                result.has_clash = data.get("has_clash", False)

                # Chain pair metrics
                result.chain_pair_iptm = data.get("chain_pair_iptm", {})
                result.chain_pair_pae = data.get("chain_pair_pae_min", {})

        except (json.JSONDecodeError, IOError):
            continue

    result.num_models = len(conf_files)

    return result if result.num_models > 0 else None


def find_all_predictions(disease: str) -> list[PredictionResult]:
    """Find and parse all predictions for a disease."""
    base_dir = DISEASE_DIRS.get(disease)
    if not base_dir:
        return []

    predictions_dir = base_dir / "predictions"
    if not predictions_dir.exists():
        return []

    results = []

    # Check direct subdirectories
    for subdir in predictions_dir.iterdir():
        if subdir.is_dir() and subdir.name != "folds_*":
            if subdir.name.startswith("folds_"):
                # Check inside folds directories
                for inner in subdir.iterdir():
                    if inner.is_dir():
                        parsed = parse_prediction(inner)
                        if parsed:
                            results.append(parsed)
            else:
                parsed = parse_prediction(subdir)
                if parsed:
                    results.append(parsed)

    return results


# ============================================================================
# COMPARISON
# ============================================================================


def extract_epitope_info(job_id: str) -> tuple[str, str, str, Optional[str]]:
    """Extract protein, epitope, modification type, and HLA from job ID.

    Returns: (protein, epitope, mod_type, hla_allele)
    """
    job_lower = job_id.lower()

    # HLA pattern
    hla_match = re.search(r'(drb1_\d+_\d+|drb1\*\d+:\d+)', job_lower)
    hla_allele = hla_match.group(1) if hla_match else None

    # Modification type
    if "_cit" in job_lower:
        mod_type = "citrullinated"
    elif "_native" in job_lower:
        mod_type = "native"
    elif "_mutant" in job_lower:
        mod_type = "mutant"
    else:
        mod_type = "unknown"

    # Protein and epitope
    parts = job_id.split("_")
    protein = parts[0].upper() if parts else "UNKNOWN"
    epitope = "_".join(parts[:2]) if len(parts) >= 2 else job_id

    return protein, epitope, mod_type, hla_allele


def find_pairs(predictions: list[PredictionResult]) -> list[tuple[PredictionResult, PredictionResult]]:
    """Find native/modified prediction pairs."""
    # Group by base epitope and HLA
    by_base = {}

    for pred in predictions:
        protein, epitope, mod_type, hla = extract_epitope_info(pred.job_id)

        # Create base key (epitope + HLA)
        base_key = epitope.lower().replace("_native", "").replace("_cit", "").replace("_citrullinated", "")
        if hla:
            base_key += f"_{hla}"

        if base_key not in by_base:
            by_base[base_key] = {"native": None, "modified": None}

        if mod_type == "native":
            by_base[base_key]["native"] = pred
        else:
            by_base[base_key]["modified"] = pred

    # Build pairs
    pairs = []
    for base_key, group in by_base.items():
        if group["native"] and group["modified"]:
            pairs.append((group["native"], group["modified"]))

    return pairs


def compare_predictions(native: PredictionResult, modified: PredictionResult) -> ComparisonResult:
    """Compare native vs modified prediction."""
    protein, epitope, _, hla = extract_epitope_info(native.job_id)

    iptm_change = modified.iptm - native.iptm
    iptm_change_pct = (iptm_change / native.iptm * 100) if native.iptm > 0 else 0

    return ComparisonResult(
        epitope=epitope,
        protein=protein,
        native_iptm=native.iptm,
        modified_iptm=modified.iptm,
        iptm_change=iptm_change,
        iptm_change_pct=iptm_change_pct,
        hla_allele=hla,
        binding_improved=modified.iptm > native.iptm,
        native_ranking=native.ranking_score,
        modified_ranking=modified.ranking_score,
    )


# ============================================================================
# ANALYSIS
# ============================================================================


def analyze_comparisons(comparisons: list[ComparisonResult]) -> dict:
    """Generate statistical analysis of comparisons."""
    if not comparisons:
        return {"error": "No comparisons available"}

    iptm_changes = [c.iptm_change for c in comparisons]
    iptm_change_pcts = [c.iptm_change_pct for c in comparisons]

    improved = [c for c in comparisons if c.binding_improved]

    # Group by protein
    by_protein = {}
    for c in comparisons:
        if c.protein not in by_protein:
            by_protein[c.protein] = []
        by_protein[c.protein].append(c)

    # Group by HLA
    by_hla = {}
    for c in comparisons:
        hla = c.hla_allele or "no_hla"
        if hla not in by_hla:
            by_hla[hla] = []
        by_hla[hla].append(c)

    analysis = {
        "total_comparisons": len(comparisons),
        "binding_improved": len(improved),
        "binding_decreased": len(comparisons) - len(improved),
        "improvement_rate": len(improved) / len(comparisons) * 100,
        "iptm_change": {
            "mean": statistics.mean(iptm_changes),
            "std": statistics.stdev(iptm_changes) if len(iptm_changes) > 1 else 0,
            "min": min(iptm_changes),
            "max": max(iptm_changes),
        },
        "iptm_change_pct": {
            "mean": statistics.mean(iptm_change_pcts),
            "std": statistics.stdev(iptm_change_pcts) if len(iptm_change_pcts) > 1 else 0,
        },
        "by_protein": {},
        "by_hla": {},
    }

    # Protein breakdown
    for protein, comps in by_protein.items():
        changes = [c.iptm_change for c in comps]
        analysis["by_protein"][protein] = {
            "count": len(comps),
            "mean_change": statistics.mean(changes),
            "improved": sum(1 for c in comps if c.binding_improved),
        }

    # HLA breakdown
    for hla, comps in by_hla.items():
        changes = [c.iptm_change for c in comps]
        analysis["by_hla"][hla] = {
            "count": len(comps),
            "mean_change": statistics.mean(changes),
            "improved": sum(1 for c in comps if c.binding_improved),
        }

    return analysis


def correlate_with_encoder(comparisons: list[ComparisonResult], encoder_scores: dict) -> dict:
    """Correlate AF3 binding changes with hyperbolic encoder predictions."""
    # This would load encoder predictions and compute correlation
    # Placeholder for integration
    return {
        "status": "pending",
        "note": "Encoder correlation requires loading codon encoder predictions",
    }


# ============================================================================
# REPORTING
# ============================================================================


def generate_report(disease: str) -> dict:
    """Generate comprehensive validation report."""
    print(f"Analyzing predictions for {disease}...")

    predictions = find_all_predictions(disease)
    print(f"  Found {len(predictions)} predictions")

    if not predictions:
        return {"disease": disease, "error": "No predictions found"}

    pairs = find_pairs(predictions)
    print(f"  Found {len(pairs)} native/modified pairs")

    comparisons = [compare_predictions(n, m) for n, m in pairs]

    analysis = analyze_comparisons(comparisons)

    report = {
        "timestamp": datetime.now().isoformat(),
        "disease": disease,
        "total_predictions": len(predictions),
        "paired_comparisons": len(comparisons),
        "analysis": analysis,
        "comparisons": [
            {
                "epitope": c.epitope,
                "protein": c.protein,
                "hla": c.hla_allele,
                "native_iptm": c.native_iptm,
                "modified_iptm": c.modified_iptm,
                "iptm_change": c.iptm_change,
                "iptm_change_pct": c.iptm_change_pct,
                "binding_improved": c.binding_improved,
            }
            for c in sorted(comparisons, key=lambda x: x.iptm_change, reverse=True)
        ],
    }

    return report


def print_report(report: dict):
    """Print human-readable report."""
    print(f"\n{'='*60}")
    print(f"AlphaFold 3 Validation Report: {report['disease'].upper()}")
    print(f"{'='*60}")
    print(f"Generated: {report['timestamp']}")
    print(f"Total predictions: {report['total_predictions']}")
    print(f"Paired comparisons: {report['paired_comparisons']}")

    if "analysis" in report and "error" not in report["analysis"]:
        analysis = report["analysis"]
        print(f"\n--- Summary ---")
        print(f"Binding improved: {analysis['binding_improved']}/{analysis['total_comparisons']} ({analysis['improvement_rate']:.1f}%)")
        print(f"Mean iPTM change: {analysis['iptm_change']['mean']:+.4f} ({analysis['iptm_change_pct']['mean']:+.1f}%)")

        if analysis.get("by_protein"):
            print(f"\n--- By Protein ---")
            for protein, data in analysis["by_protein"].items():
                print(f"  {protein}: {data['improved']}/{data['count']} improved, mean change: {data['mean_change']:+.4f}")

        if analysis.get("by_hla") and len(analysis["by_hla"]) > 1:
            print(f"\n--- By HLA ---")
            for hla, data in analysis["by_hla"].items():
                print(f"  {hla}: {data['improved']}/{data['count']} improved, mean change: {data['mean_change']:+.4f}")

    if report.get("comparisons"):
        print(f"\n--- Top Changes ---")
        for comp in report["comparisons"][:10]:
            direction = "↑" if comp["binding_improved"] else "↓"
            print(f"  {comp['epitope']}: {comp['iptm_change']:+.4f} {direction} ({comp['iptm_change_pct']:+.1f}%)")


# ============================================================================
# MAIN
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze AlphaFold 3 Predictions")
    parser.add_argument("--disease", choices=["rheumatoid_arthritis", "hiv", "alzheimers", "all"],
                        default="all", help="Disease to analyze")
    parser.add_argument("--output", type=Path, help="Output file for report")
    parser.add_argument("--json", action="store_true", help="Output as JSON only")

    args = parser.parse_args()

    diseases = list(DISEASE_DIRS.keys()) if args.disease == "all" else [args.disease]

    all_reports = {}
    for disease in diseases:
        report = generate_report(disease)
        all_reports[disease] = report

        if not args.json:
            print_report(report)

    # Save report
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_reports, f, indent=2)
        print(f"\nReport saved to: {args.output}")
    elif args.json:
        print(json.dumps(all_reports, indent=2))


if __name__ == "__main__":
    main()
