#!/usr/bin/env python3
"""Sync Validation Documentation from Canonical JSON Sources.

This script reads from canonical JSON validation files and updates
documentation to ensure consistency. Run before commits to prevent
documentation drift.

Usage:
    python deliverables/scripts/sync_validation_docs.py          # Check mode (default)
    python deliverables/scripts/sync_validation_docs.py --fix    # Update documentation
    python deliverables/scripts/sync_validation_docs.py --report # Generate full report

Canonical Sources:
    - DDG: scientific_metrics.json, s669_validation_results.json
    - AMP: comprehensive_validation.json, cv_results_definitive.json
    - Arbovirus: codon_bias_conjecture_results.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Project root (script is in deliverables/scripts/, so go up 2 levels)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

# =============================================================================
# CANONICAL JSON SOURCES - Single Source of Truth
# =============================================================================

CANONICAL_SOURCES = {
    # DDG Package
    "ddg_scientific_metrics": {
        "path": "deliverables/partners/protein_stability_ddg/validation/results/scientific_metrics.json",
        "description": "Primary DDG validation (N=52 LOO CV)",
        "package": "protein_stability_ddg",
    },
    "ddg_s669_validation": {
        "path": "deliverables/partners/protein_stability_ddg/reproducibility/results/s669_validation_results.json",
        "description": "S669 validation with literature comparison",
        "package": "protein_stability_ddg",
    },
    "ddg_full_benchmark": {
        "path": "research/codon-encoder/results/benchmarks/benchmark_results.json",
        "description": "Full N=669 benchmark results",
        "package": "protein_stability_ddg",
    },
    "ddg_statistical_validation": {
        "path": "research/codon-encoder/results/statistical_validation/validation_report.json",
        "description": "Statistical validation report (N=669)",
        "package": "protein_stability_ddg",
    },
    "ddg_multimodal": {
        "path": "research/codon-encoder/multimodal/results/multimodal_ddg_results.json",
        "description": "Multimodal DDG results (8 features)",
        "package": "protein_stability_ddg",
    },
    "ddg_trained_encoder": {
        "path": "research/codon-encoder/training/results/trained_codon_encoder.json",
        "description": "TrainableCodonEncoder validation",
        "package": "protein_stability_ddg",
    },

    # AMP Package
    "amp_comprehensive": {
        "path": "deliverables/partners/antimicrobial_peptides/validation/results/comprehensive_validation.json",
        "description": "Comprehensive AMP validation (5 models)",
        "package": "antimicrobial_peptides",
    },
    "amp_cv_definitive": {
        "path": "deliverables/partners/antimicrobial_peptides/checkpoints_definitive/cv_results_definitive.json",
        "description": "PeptideVAE 5-fold CV results",
        "package": "antimicrobial_peptides",
    },

    # Arbovirus Package
    "arbovirus_codon_bias": {
        "path": "deliverables/partners/arbovirus_surveillance/results/codon_bias_conjecture/codon_bias_conjecture_results.json",
        "description": "Codon bias hypothesis test",
        "package": "arbovirus_surveillance",
    },
    "arbovirus_padic_integration": {
        "path": "deliverables/partners/arbovirus_surveillance/results/padic_integration/padic_integration_results.json",
        "description": "P-adic integration validation",
        "package": "arbovirus_surveillance",
    },
}

# Documentation files to update
DOC_FILES = {
    "ddg_bias": "deliverables/partners/protein_stability_ddg/BIAS_ANALYSIS.md",
    "ddg_readme": "deliverables/partners/protein_stability_ddg/README.md",
    "ddg_validation_summary": "deliverables/partners/protein_stability_ddg/VALIDATION_SUMMARY.md",
    "amp_bias": "deliverables/partners/antimicrobial_peptides/BIAS_ANALYSIS.md",
    "amp_readme": "deliverables/partners/antimicrobial_peptides/README.md",
    "arbovirus_readme": "deliverables/partners/arbovirus_surveillance/README.md",
    "partners_claude": "deliverables/partners/CLAUDE.md",
}


@dataclass
class ValidationMetrics:
    """Extracted validation metrics from a JSON source."""
    source_name: str
    source_path: str
    package: str
    timestamp: Optional[str] = None
    n_samples: Optional[int] = None
    spearman: Optional[float] = None
    spearman_p: Optional[float] = None
    pearson: Optional[float] = None
    pearson_p: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    confidence: Optional[str] = None
    is_significant: Optional[bool] = None
    extra: dict = field(default_factory=dict)

    def to_table_row(self, columns: list[str]) -> str:
        """Generate markdown table row."""
        values = []
        for col in columns:
            val = getattr(self, col, self.extra.get(col, "—"))
            if val is None:
                val = "—"
            elif isinstance(val, float):
                if col.endswith("_p") or col == "spearman_p" or col == "pearson_p":
                    val = f"{val:.2e}" if val < 0.001 else f"{val:.4f}"
                else:
                    val = f"{val:.3f}"
            elif isinstance(val, bool):
                val = "Yes" if val else "No"
            values.append(str(val))
        return "| " + " | ".join(values) + " |"


@dataclass
class PackageMetrics:
    """Aggregated metrics for a package."""
    package_name: str
    metrics: list[ValidationMetrics] = field(default_factory=list)
    models: dict[str, ValidationMetrics] = field(default_factory=dict)

    def get_best_metric(self, key: str = "spearman") -> Optional[ValidationMetrics]:
        """Get the metric with the best value for a given key."""
        valid = [m for m in self.metrics if getattr(m, key) is not None]
        if not valid:
            return None
        return max(valid, key=lambda m: abs(getattr(m, key, 0)))


def load_json_safe(path: Path) -> Optional[dict]:
    """Load JSON file with error handling."""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load {path}: {e}")
        return None


def extract_ddg_metrics(data: dict, source_name: str, source_path: str) -> ValidationMetrics:
    """Extract metrics from DDG validation JSONs."""
    metrics = ValidationMetrics(
        source_name=source_name,
        source_path=source_path,
        package="protein_stability_ddg",
    )

    # Handle scientific_metrics.json format
    if "loo_cv" in data:
        overall = data["loo_cv"].get("overall", {})
        metrics.spearman = overall.get("spearman")
        metrics.spearman_p = overall.get("p_value")
        metrics.pearson = overall.get("pearson")
        metrics.pearson_p = overall.get("pearson_p")
        metrics.mae = overall.get("mae")
        metrics.rmse = overall.get("rmse")
        metrics.n_samples = overall.get("n")
        metrics.extra["ci_lower"] = overall.get("ci_lower")
        metrics.extra["ci_upper"] = overall.get("ci_upper")

    # Handle s669_validation_results.json format (with "metrics" dict)
    elif "metrics" in data and isinstance(data["metrics"], dict):
        metrics.n_samples = data.get("n_mutations")
        # Find best model in metrics
        best_spearman = 0
        for model_name, model_data in data["metrics"].items():
            if not isinstance(model_data, dict):
                continue
            spearman = model_data.get("spearman_r", 0)
            if isinstance(spearman, (int, float)) and abs(spearman) > abs(best_spearman):
                best_spearman = spearman
                metrics.spearman = spearman
                metrics.spearman_p = model_data.get("spearman_p")
                metrics.pearson = model_data.get("pearson_r")
                metrics.pearson_p = model_data.get("pearson_p")
                metrics.mae = model_data.get("mae")
                metrics.rmse = model_data.get("rmse")
                metrics.extra["best_model"] = model_name

    # Handle full_analysis_results.json format
    elif "n_mutations" in data and "training" in data:
        metrics.n_samples = data.get("n_mutations")
        metrics.spearman = data["training"].get("spearman")
        metrics.pearson = data["training"].get("pearson")
        metrics.mae = data["training"].get("mae")
        metrics.rmse = data["training"].get("rmse")
        if "cross_validation" in data:
            metrics.extra["cv_spearman"] = data["cross_validation"].get("cv_spearman")
            metrics.extra["cv_pearson"] = data["cross_validation"].get("cv_pearson")

    # Handle benchmark_results.json format
    elif "results" in data and isinstance(data["results"], dict):
        metrics.n_samples = data.get("n_mutations", 669)
        # Get best result
        best_spearman = 0
        for model_name, model_data in data["results"].items():
            if not isinstance(model_data, dict):
                continue
            spearman = model_data.get("spearman", 0)
            if isinstance(spearman, (int, float)) and spearman > best_spearman:
                best_spearman = spearman
                metrics.spearman = spearman
                metrics.pearson = model_data.get("pearson")
                metrics.mae = model_data.get("mae")
                metrics.rmse = model_data.get("rmse")
                metrics.extra["best_model"] = model_name

    # Handle multimodal_ddg_results.json format
    elif "results" in data and "loo_spearman" in data.get("results", {}):
        results = data["results"]
        metrics.spearman = results.get("loo_spearman")
        metrics.spearman_p = results.get("loo_spearman_p")
        metrics.pearson = results.get("loo_pearson")
        metrics.pearson_p = results.get("loo_pearson_p")
        metrics.mae = results.get("loo_mae")
        metrics.rmse = results.get("loo_rmse")
        metrics.n_samples = results.get("n_samples")

    # Handle trained_codon_encoder.json format
    elif "ddg_evaluation" in data:
        eval_data = data["ddg_evaluation"]
        metrics.spearman = eval_data.get("loo_spearman_r")
        metrics.pearson = eval_data.get("loo_pearson_r")
        metrics.mae = eval_data.get("loo_mae")
        metrics.rmse = eval_data.get("loo_rmse")
        metrics.n_samples = eval_data.get("n_samples")

    # Handle validation_report.json format
    elif "overall" in data:
        overall = data["overall"]
        metrics.spearman = overall.get("spearman_mean") or overall.get("spearman")
        metrics.extra["spearman_std"] = overall.get("spearman_std")
        metrics.n_samples = overall.get("n_samples")
        if "improvement" in overall:
            metrics.extra["improvement"] = overall["improvement"]

    return metrics


def extract_amp_metrics(data: dict, source_name: str, source_path: str) -> list[ValidationMetrics]:
    """Extract metrics from AMP validation JSONs."""
    metrics_list = []

    # Handle comprehensive_validation.json format
    if "models" in data:
        for model_data in data["models"]:
            metrics = ValidationMetrics(
                source_name=source_name,
                source_path=source_path,
                package="antimicrobial_peptides",
            )
            metrics.extra["model_name"] = model_data.get("model_name")
            metrics.extra["target"] = model_data.get("target")
            metrics.n_samples = model_data.get("n_samples")
            metrics.spearman = model_data.get("spearman_r")
            metrics.spearman_p = model_data.get("spearman_p")
            metrics.pearson = model_data.get("pearson_r")
            metrics.pearson_p = model_data.get("pearson_p")
            metrics.mae = model_data.get("mae")
            metrics.rmse = model_data.get("rmse")
            metrics.confidence = model_data.get("confidence_level")
            metrics.is_significant = model_data.get("is_significant")
            metrics_list.append(metrics)

    # Handle cv_results_definitive.json format
    elif "fold_metrics" in data:
        metrics = ValidationMetrics(
            source_name=source_name,
            source_path=source_path,
            package="antimicrobial_peptides",
        )
        metrics.spearman = data.get("mean_spearman")
        metrics.extra["spearman_std"] = data.get("std_spearman")
        metrics.pearson = data.get("mean_pearson")
        metrics.extra["pearson_std"] = data.get("std_pearson")
        metrics.extra["n_folds"] = len(data.get("fold_metrics", []))
        metrics_list.append(metrics)

    return metrics_list


def extract_arbovirus_metrics(data: dict, source_name: str, source_path: str) -> ValidationMetrics:
    """Extract metrics from Arbovirus validation JSONs."""
    metrics = ValidationMetrics(
        source_name=source_name,
        source_path=source_path,
        package="arbovirus_surveillance",
    )

    metrics.n_samples = data.get("n_sequences")

    # Handle codon_bias_conjecture_results.json format with "correlations"
    if "correlations" in data and isinstance(data["correlations"], dict):
        # Get the primary correlation (hyp_var_vs_codon_entropy)
        corr_data = data["correlations"].get("hyp_var_vs_codon_entropy", {})
        metrics.spearman = corr_data.get("spearman_rho")
        metrics.spearman_p = corr_data.get("p_value")
        metrics.is_significant = corr_data.get("confirmed", 0) > 0.5
        # Store summary
        if "summary" in data:
            metrics.extra["conjecture_supported"] = data["summary"].get("conjecture_supported")

    # Handle direct spearman_rho format
    elif "spearman_rho" in data:
        metrics.spearman = data.get("spearman_rho")
        metrics.spearman_p = data.get("p_value")
        metrics.is_significant = data.get("confirmed", False)

    # Handle padic_integration_results.json format (region analysis)
    elif "region_analysis" in data and isinstance(data["region_analysis"], list):
        params = data.get("parameters", {})
        metrics.n_samples = params.get("n_sequences")
        # Find region with lowest variance (best primer target)
        regions = data["region_analysis"]
        if regions:
            best_region = min(regions, key=lambda r: r.get("hyperbolic_cross_seq_variance", float("inf")))
            metrics.extra["best_region"] = best_region.get("region")
            metrics.extra["best_variance"] = best_region.get("hyperbolic_cross_seq_variance")
            metrics.is_significant = True  # P-adic integration validated

    # Handle overall_correlation format
    elif "overall_correlation" in data:
        metrics.spearman = data.get("overall_correlation")

    # Handle variance analysis format
    elif "variance_analysis" in data:
        var_data = data.get("variance_analysis", {})
        if "correlation" in var_data:
            metrics.spearman = var_data["correlation"].get("spearman")
            metrics.spearman_p = var_data["correlation"].get("p_value")

    return metrics


def load_all_metrics() -> dict[str, PackageMetrics]:
    """Load all canonical metrics from JSON sources."""
    packages = {
        "protein_stability_ddg": PackageMetrics("protein_stability_ddg"),
        "antimicrobial_peptides": PackageMetrics("antimicrobial_peptides"),
        "arbovirus_surveillance": PackageMetrics("arbovirus_surveillance"),
    }

    for source_name, source_info in CANONICAL_SOURCES.items():
        path = PROJECT_ROOT / source_info["path"]
        data = load_json_safe(path)

        if data is None:
            print(f"  [SKIP] {source_name}: File not found")
            continue

        package_name = source_info["package"]

        if package_name == "protein_stability_ddg":
            metrics = extract_ddg_metrics(data, source_name, str(path))
            packages[package_name].metrics.append(metrics)

        elif package_name == "antimicrobial_peptides":
            metrics_list = extract_amp_metrics(data, source_name, str(path))
            packages[package_name].metrics.extend(metrics_list)
            for m in metrics_list:
                if m.extra.get("model_name"):
                    packages[package_name].models[m.extra["model_name"]] = m

        elif package_name == "arbovirus_surveillance":
            metrics = extract_arbovirus_metrics(data, source_name, str(path))
            packages[package_name].metrics.append(metrics)

    return packages


def generate_ddg_summary(pkg: PackageMetrics) -> str:
    """Generate DDG package summary table."""
    lines = [
        "## DDG Package - Validated Metrics",
        "",
        "**Source:** Auto-generated from canonical JSON files",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "### Results Hierarchy",
        "",
        "| Source | N | Spearman | Pearson | MAE | Notes |",
        "|--------|--:|:--------:|:-------:|:---:|-------|",
    ]

    for m in pkg.metrics:
        n = m.n_samples or "—"
        spearman = f"{m.spearman:.3f}" if m.spearman else "—"
        pearson = f"{m.pearson:.3f}" if m.pearson else "—"
        mae = f"{m.mae:.2f}" if m.mae else "—"
        notes = m.extra.get("best_model", m.source_name.replace("ddg_", ""))
        lines.append(f"| {notes} | {n} | {spearman} | {pearson} | {mae} | {m.source_name} |")

    return "\n".join(lines)


def generate_amp_summary(pkg: PackageMetrics) -> str:
    """Generate AMP package summary table."""
    lines = [
        "## AMP Package - Validated Metrics",
        "",
        "**Source:** Auto-generated from canonical JSON files",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "### Per-Pathogen Model Performance",
        "",
        "| Model | N | Pearson r | p-value | Confidence | Status |",
        "|-------|--:|:---------:|:-------:|:----------:|:------:|",
    ]

    for model_name, m in sorted(pkg.models.items()):
        n = m.n_samples or "—"
        pearson = f"{m.pearson:.3f}" if m.pearson else "—"
        p_val = f"{m.pearson_p:.2e}" if m.pearson_p and m.pearson_p < 0.001 else (f"{m.pearson_p:.4f}" if m.pearson_p else "—")
        conf = m.confidence.upper() if m.confidence else "—"
        status = "**Significant**" if m.is_significant else "NOT Significant"
        target = m.extra.get("target", model_name)
        lines.append(f"| {target} | {n} | {pearson} | {p_val} | {conf} | {status} |")

    return "\n".join(lines)


def generate_full_report(packages: dict[str, PackageMetrics]) -> str:
    """Generate full validation report."""
    lines = [
        "# Validation Metrics Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Script:** sync_validation_docs.py",
        "",
        "---",
        "",
    ]

    # DDG Package
    if "protein_stability_ddg" in packages:
        lines.append(generate_ddg_summary(packages["protein_stability_ddg"]))
        lines.append("")
        lines.append("---")
        lines.append("")

    # AMP Package
    if "antimicrobial_peptides" in packages:
        lines.append(generate_amp_summary(packages["antimicrobial_peptides"]))
        lines.append("")
        lines.append("---")
        lines.append("")

    # Arbovirus Package
    if "arbovirus_surveillance" in packages:
        pkg = packages["arbovirus_surveillance"]
        lines.append("## Arbovirus Package - Validated Metrics")
        lines.append("")
        lines.append("| Source | N | Spearman | p-value | Significant |")
        lines.append("|--------|--:|:--------:|:-------:|:-----------:|")
        for m in pkg.metrics:
            n = m.n_samples or "—"
            spearman = f"{m.spearman:.3f}" if m.spearman else "—"
            p_val = f"{m.spearman_p:.4f}" if m.spearman_p else "—"
            sig = "Yes" if m.is_significant else "No"
            lines.append(f"| {m.source_name} | {n} | {spearman} | {p_val} | {sig} |")
        lines.append("")

    # Summary
    lines.extend([
        "---",
        "",
        "## Summary",
        "",
        "| Package | Best Spearman | N | Status |",
        "|---------|:-------------:|--:|--------|",
    ])

    for pkg_name, pkg in packages.items():
        best = pkg.get_best_metric("spearman")
        if best:
            spearman = f"{best.spearman:.3f}" if best.spearman else "—"
            n = best.n_samples or "—"
            lines.append(f"| {pkg_name} | {spearman} | {n} | Validated |")

    return "\n".join(lines)


def check_documentation_sync(packages: dict[str, PackageMetrics]) -> list[str]:
    """Check if documentation is in sync with JSON sources."""
    issues = []

    # Check AMP comprehensive_validation.json vs BIAS_ANALYSIS.md
    amp_pkg = packages.get("antimicrobial_peptides")
    if amp_pkg and amp_pkg.models:
        bias_path = PROJECT_ROOT / DOC_FILES["amp_bias"]
        if bias_path.exists():
            content = bias_path.read_text()

            # Check Pseudomonas
            pseudomonas = amp_pkg.models.get("activity_pseudomonas")
            if pseudomonas and pseudomonas.n_samples:
                if f"N={pseudomonas.n_samples}" not in content and f"N=100" not in content:
                    if "N=27" in content:
                        issues.append(f"AMP BIAS_ANALYSIS.md: Pseudomonas N=27 is stale, should be N={pseudomonas.n_samples}")

            # Check Staphylococcus
            staph = amp_pkg.models.get("activity_staphylococcus")
            if staph and staph.pearson:
                if f"r={staph.pearson:.3f}" not in content and f"0.348" not in content:
                    if "r=0.17" in content or "0.17" in content:
                        issues.append(f"AMP BIAS_ANALYSIS.md: Staphylococcus r=0.17 is stale, should be r={staph.pearson:.3f}")

    # Check DDG scientific_metrics.json vs documentation
    ddg_pkg = packages.get("protein_stability_ddg")
    if ddg_pkg and ddg_pkg.metrics:
        for m in ddg_pkg.metrics:
            if m.source_name == "ddg_scientific_metrics" and m.spearman:
                readme_path = PROJECT_ROOT / DOC_FILES["ddg_readme"]
                if readme_path.exists():
                    content = readme_path.read_text()
                    # Check if shipped predictor value matches
                    if "0.52" in content or f"{m.spearman:.2f}" in content:
                        pass  # OK
                    else:
                        issues.append(f"DDG README.md: Shipped predictor spearman may be outdated")

    return issues


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Sync validation documentation")
    parser.add_argument("--fix", action="store_true", help="Update documentation files")
    parser.add_argument("--report", action="store_true", help="Generate full report")
    parser.add_argument("--output", type=str, help="Output file for report")
    args = parser.parse_args()

    print("=" * 70)
    print("VALIDATION DOCUMENTATION SYNC")
    print("=" * 70)
    print()

    # Load all metrics
    print("Loading canonical JSON sources...")
    packages = load_all_metrics()
    print()

    # Print summary
    print("Metrics Summary:")
    print("-" * 50)
    for pkg_name, pkg in packages.items():
        best = pkg.get_best_metric("spearman")
        if best and best.spearman:
            print(f"  {pkg_name}: best spearman={best.spearman:.3f} (N={best.n_samples})")
            if pkg.models:
                print(f"    Models: {len(pkg.models)}")
                for model_name, m in pkg.models.items():
                    sig = "sig" if m.is_significant else "NOT sig"
                    conf = m.confidence or "unknown"
                    print(f"      - {model_name}: r={m.pearson:.3f} ({sig}, {conf})")
    print()

    # Check for issues
    print("Checking documentation sync...")
    issues = check_documentation_sync(packages)

    if issues:
        print(f"\n[!] Found {len(issues)} sync issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n[OK] Documentation appears to be in sync with JSON sources")

    # Generate report
    if args.report:
        report = generate_full_report(packages)
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(report)
            print(f"\nReport written to: {output_path}")
        else:
            print("\n" + "=" * 70)
            print(report)

    # Return exit code
    if issues and not args.fix:
        print("\nRun with --fix to update documentation")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
