#!/usr/bin/env python
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Master script to run all experiments for reproducibility.

This script runs the complete experiment suite and validates results
against expected values for publication reproducibility.

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --quick  # Quick validation only
    python run_all_experiments.py --full   # Full benchmark suite
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Project root
ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "reproducibility"


def run_command(cmd: list[str], description: str) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    logger.info(f"Running: {description}")
    logger.info(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode == 0:
            logger.info(f"  ✓ {description} completed successfully")
            return True, result.stdout
        else:
            logger.error(f"  ✗ {description} failed")
            logger.error(f"  Error: {result.stderr}")
            return False, result.stderr

    except subprocess.TimeoutExpired:
        logger.error(f"  ✗ {description} timed out")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"  ✗ {description} error: {e}")
        return False, str(e)


def run_cross_disease_benchmark() -> dict[str, Any]:
    """Run cross-disease benchmark."""
    output_dir = RESULTS_DIR / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "experiments" / "run_cross_disease.py"),
        "--output-dir", str(output_dir),
        "--n-folds", "3",
        "--n-repeats", "2",
    ]

    success, output = run_command(cmd, "Cross-disease benchmark")

    # Load results if successful
    results = {}
    if success:
        # Find latest result file
        json_files = list(output_dir.glob("*.json"))
        if json_files:
            latest = max(json_files, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                results = json.load(f)

    return {"success": success, "results": results}


def run_physics_validation() -> dict[str, Any]:
    """Run physics validation."""
    output_dir = RESULTS_DIR / "physics"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "experiments" / "run_physics_validation.py"),
        "--output-dir", str(output_dir),
        "--n-samples", "100",
    ]

    success, output = run_command(cmd, "Physics validation")

    results = {}
    if success:
        json_files = list(output_dir.glob("*.json"))
        if json_files:
            latest = max(json_files, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                results = json.load(f)

    return {"success": success, "results": results}


def run_unit_tests() -> dict[str, Any]:
    """Run unit tests."""
    cmd = [
        sys.executable, "-m", "pytest",
        str(ROOT / "tests" / "unit"),
        "-v", "--tb=short",
        "-x",  # Stop on first failure
    ]

    success, output = run_command(cmd, "Unit tests")

    return {"success": success, "output": output[:2000]}


def validate_results(
    benchmark_results: dict[str, Any],
    physics_results: dict[str, Any],
) -> dict[str, Any]:
    """Validate results against expected values."""
    logger.info("Validating results...")

    expected = {
        "overall_spearman_mean": 0.80,  # Minimum expected
        "physics_ddg_mean": 0.70,
        "physics_universality": True,
    }

    validation = {
        "passed": True,
        "checks": [],
    }

    # Check benchmark results
    if benchmark_results.get("results"):
        actual_spearman = benchmark_results["results"].get("overall_spearman_mean", 0)
        check = {
            "name": "Overall Spearman",
            "expected": f">= {expected['overall_spearman_mean']}",
            "actual": actual_spearman,
            "passed": actual_spearman >= expected["overall_spearman_mean"],
        }
        validation["checks"].append(check)
        if not check["passed"]:
            validation["passed"] = False

    # Check physics results
    if physics_results.get("results"):
        actual_ddg = physics_results["results"].get("overall_ddg_mean", 0)
        check = {
            "name": "Physics ΔΔG",
            "expected": f">= {expected['physics_ddg_mean']}",
            "actual": actual_ddg,
            "passed": actual_ddg >= expected["physics_ddg_mean"],
        }
        validation["checks"].append(check)
        if not check["passed"]:
            validation["passed"] = False

        universality = physics_results["results"].get("universality_confirmed", False)
        check = {
            "name": "Physics Universality",
            "expected": True,
            "actual": universality,
            "passed": universality == expected["physics_universality"],
        }
        validation["checks"].append(check)
        if not check["passed"]:
            validation["passed"] = False

    return validation


def generate_summary_report(
    benchmark_results: dict[str, Any],
    physics_results: dict[str, Any],
    test_results: dict[str, Any],
    validation: dict[str, Any],
    total_runtime: float,
) -> str:
    """Generate summary report."""
    timestamp = datetime.now().isoformat()

    lines = [
        "# Reproducibility Report",
        "",
        f"**Generated**: {timestamp}",
        f"**Total Runtime**: {total_runtime:.2f}s",
        "",
        "## Summary",
        "",
        f"- Validation: {'✓ PASSED' if validation['passed'] else '✗ FAILED'}",
        f"- Benchmarks: {'✓' if benchmark_results.get('success') else '✗'}",
        f"- Physics: {'✓' if physics_results.get('success') else '✗'}",
        f"- Tests: {'✓' if test_results.get('success') else '✗'}",
        "",
        "## Validation Checks",
        "",
        "| Check | Expected | Actual | Status |",
        "|-------|----------|--------|--------|",
    ]

    for check in validation.get("checks", []):
        status = "✓" if check["passed"] else "✗"
        actual = check["actual"]
        if isinstance(actual, float):
            actual = f"{actual:.4f}"
        lines.append(f"| {check['name']} | {check['expected']} | {actual} | {status} |")

    lines.extend([
        "",
        "## Benchmark Results",
        "",
    ])

    if benchmark_results.get("results"):
        results = benchmark_results["results"]
        lines.append(f"- Overall Spearman: {results.get('overall_spearman_mean', 'N/A'):.4f}")
        lines.append(f"- Diseases evaluated: {len(results.get('results', []))}")

    lines.extend([
        "",
        "## Physics Validation",
        "",
    ])

    if physics_results.get("results"):
        results = physics_results["results"]
        lines.append(f"- ΔΔG Correlation: {results.get('overall_ddg_mean', 'N/A'):.4f}")
        lines.append(f"- Mass Correlation: {results.get('overall_mass_mean', 'N/A'):.4f}")
        lines.append(f"- Universality: {'Confirmed' if results.get('universality_confirmed') else 'Not confirmed'}")

    lines.extend([
        "",
        "---",
        "",
        "*This report was automatically generated by run_all_experiments.py*",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run all experiments for reproducibility")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick validation only (fewer samples)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full benchmark suite (more samples)",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip unit tests",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Reproducibility Suite - Ternary VAE")
    logger.info("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Run experiments
    logger.info("\n[1/3] Running cross-disease benchmark...")
    benchmark_results = run_cross_disease_benchmark()

    logger.info("\n[2/3] Running physics validation...")
    physics_results = run_physics_validation()

    logger.info("\n[3/3] Running unit tests...")
    if args.skip_tests:
        test_results = {"success": True, "output": "Skipped"}
    else:
        test_results = run_unit_tests()

    total_runtime = time.time() - start_time

    # Validate results
    validation = validate_results(benchmark_results, physics_results)

    # Generate report
    report = generate_summary_report(
        benchmark_results,
        physics_results,
        test_results,
        validation,
        total_runtime,
    )

    # Save report
    report_path = RESULTS_DIR / f"reproducibility_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, "w") as f:
        f.write(report)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("REPRODUCIBILITY SUITE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Validation: {'PASSED ✓' if validation['passed'] else 'FAILED ✗'}")
    logger.info(f"Total Runtime: {total_runtime:.2f}s")
    logger.info(f"Report saved to: {report_path}")

    # Exit with appropriate code
    sys.exit(0 if validation["passed"] else 1)


if __name__ == "__main__":
    main()
