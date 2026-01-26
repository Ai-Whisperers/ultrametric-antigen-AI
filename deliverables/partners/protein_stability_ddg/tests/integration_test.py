#!/usr/bin/env python3
"""Integration Test Suite for Jose Colbes DDG Predictor Package.

This script validates the entire pipeline end-to-end without hardcoded
expectations. Instead, it verifies:
1. Scripts run without errors
2. Outputs are generated in expected locations
3. Results are numerically valid (not NaN, in reasonable ranges)
4. Statistical tests produce significant results

Usage:
    python integration_test.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import csv

# Package root
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parents[3]


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    duration_ms: Optional[float] = None


def run_script(script_path: Path, args: list[str] = None, timeout: int = 300) -> tuple[bool, str, str]:
    """Run a Python script and return (success, stdout, stderr)."""
    import time

    cmd = [sys.executable, str(script_path)] + (args or [])
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(script_path.parent),
        )
        duration = (time.time() - start) * 1000
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Timeout after {timeout}s"
    except Exception as e:
        return False, "", str(e)


def test_data_integrity() -> TestResult:
    """Verify S669 dataset is valid and complete."""
    data_path = PACKAGE_ROOT / "reproducibility" / "data" / "s669.csv"

    if not data_path.exists():
        return TestResult("data_integrity", False, f"s669.csv not found at {data_path}")

    try:
        with open(data_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) < 10:
            return TestResult("data_integrity", False, f"Only {len(rows)} mutations (expected >10)")

        required_cols = ['pdb_id', 'wild_type', 'mutant', 'ddg']
        missing = [c for c in required_cols if c not in rows[0]]
        if missing:
            return TestResult("data_integrity", False, f"Missing columns: {missing}")

        # Verify DDG values are numeric
        ddg_values = [float(r['ddg']) for r in rows]
        if any(v != v for v in ddg_values):  # NaN check
            return TestResult("data_integrity", False, "DDG values contain NaN")

        return TestResult("data_integrity", True, f"Valid: {len(rows)} mutations, DDG range [{min(ddg_values):.2f}, {max(ddg_values):.2f}]")

    except Exception as e:
        return TestResult("data_integrity", False, str(e))


def test_embeddings_extraction() -> TestResult:
    """Verify embedding extraction produces valid output."""
    emb_path = PACKAGE_ROOT / "reproducibility" / "data" / "aa_embeddings_v2.json"

    if not emb_path.exists():
        return TestResult("embeddings", False, "aa_embeddings_v2.json not found")

    try:
        with open(emb_path) as f:
            data = json.load(f)

        aa_data = data.get('amino_acids', {})
        if len(aa_data) != 20:
            return TestResult("embeddings", False, f"Expected 20 amino acids, got {len(aa_data)}")

        # Verify all 20 standard AAs present
        standard_aas = set("ACDEFGHIKLMNPQRSTVWY")
        present_aas = set(aa_data.keys())
        missing = standard_aas - present_aas
        if missing:
            return TestResult("embeddings", False, f"Missing amino acids: {missing}")

        # Verify embeddings have consistent dimensions
        dims = set(len(aa_data[aa]['embedding']) for aa in aa_data)
        if len(dims) > 1:
            return TestResult("embeddings", False, f"Inconsistent embedding dimensions: {dims}")

        return TestResult("embeddings", True, f"Valid: 20 AAs, dim={list(dims)[0]}")

    except Exception as e:
        return TestResult("embeddings", False, str(e))


def test_bootstrap_validation() -> TestResult:
    """Run bootstrap test and verify results are statistically valid."""
    script = PACKAGE_ROOT / "validation" / "bootstrap_test.py"

    if not script.exists():
        return TestResult("bootstrap", False, "bootstrap_test.py not found")

    success, stdout, stderr = run_script(script, timeout=180)

    if not success:
        return TestResult("bootstrap", False, f"Script failed: {stderr[:200]}")

    # Parse output for key metrics
    import re

    spearman_match = re.search(r'Spearman rho:\s*([0-9.-]+)', stdout)
    p_value_match = re.search(r'p\s*=\s*([0-9.e-]+)', stdout)
    ci_match = re.search(r'95% CI:\s*\[([0-9.-]+),\s*([0-9.-]+)\]', stdout)

    if not spearman_match:
        return TestResult("bootstrap", False, "Could not parse Spearman from output")

    spearman = float(spearman_match.group(1))

    # Verify result is reasonable (not NaN, in valid range)
    if spearman != spearman:  # NaN check
        return TestResult("bootstrap", False, "Spearman is NaN")

    if not (-1.0 <= spearman <= 1.0):
        return TestResult("bootstrap", False, f"Spearman {spearman} out of range [-1, 1]")

    # Verify p-value is present and reasonable
    if p_value_match:
        p_value = float(p_value_match.group(1))
        if p_value > 0.05:
            return TestResult("bootstrap", False, f"p-value {p_value} not significant (>0.05)")

    # Verify CI doesn't include zero
    if ci_match:
        ci_lower = float(ci_match.group(1))
        ci_upper = float(ci_match.group(2))
        if ci_lower <= 0 <= ci_upper:
            return TestResult("bootstrap", False, f"CI [{ci_lower}, {ci_upper}] includes zero")

    return TestResult("bootstrap", True, f"Spearman={spearman:.4f}, significant (p<0.05)")


def test_c4_predictor() -> TestResult:
    """Verify C4 mutation effect predictor works."""
    script = PACKAGE_ROOT / "scripts" / "C4_mutation_effect_predictor.py"

    if not script.exists():
        return TestResult("c4_predictor", False, "C4_mutation_effect_predictor.py not found")

    success, stdout, stderr = run_script(
        script,
        args=["--mutations", "G45A,D156K,V78I", "--use-validated"],
        timeout=120
    )

    if not success:
        return TestResult("c4_predictor", False, f"Script failed: {stderr[:200]}")

    # Verify output contains predictions
    if "DDG" not in stdout or "kcal/mol" not in stdout:
        return TestResult("c4_predictor", False, "Output doesn't contain DDG predictions")

    # Verify predictions are reasonable
    import re
    ddg_values = re.findall(r'([+-]?[0-9.]+)\s+(?:destabilizing|neutral|stabilizing)', stdout, re.IGNORECASE)

    if not ddg_values:
        return TestResult("c4_predictor", False, "Could not parse DDG values from output")

    ddg_floats = [float(v) for v in ddg_values]

    # Check DDG values are in reasonable range (-10 to +20 kcal/mol)
    if any(d < -10 or d > 20 for d in ddg_floats):
        return TestResult("c4_predictor", False, f"DDG values out of range: {ddg_floats}")

    return TestResult("c4_predictor", True, f"Predictions generated: {len(ddg_floats)} mutations")


def test_result_files_exist() -> TestResult:
    """Verify all expected result files exist."""
    expected_files = [
        PACKAGE_ROOT / "validation" / "results" / "SCIENTIFIC_VALIDATION_REPORT.md",
        PACKAGE_ROOT / "validation" / "results" / "scientific_metrics.json",
        PACKAGE_ROOT / "reproducibility" / "results" / "BENCHMARK_REPORT.md",
    ]

    missing = [f for f in expected_files if not f.exists()]

    if missing:
        return TestResult("result_files", False, f"Missing: {[f.name for f in missing]}")

    return TestResult("result_files", True, f"All {len(expected_files)} result files present")


def test_no_path_errors() -> TestResult:
    """Verify scripts don't have path resolution errors on import."""
    scripts = [
        PACKAGE_ROOT / "validation" / "bootstrap_test.py",
        PACKAGE_ROOT / "scripts" / "C4_mutation_effect_predictor.py",
        PACKAGE_ROOT / "reproducibility" / "extract_aa_embeddings_v2.py",
    ]

    errors = []
    for script in scripts:
        cmd = [sys.executable, "-c", f"import sys; sys.path.insert(0, '{script.parent}'); exec(open('{script}').read().split('if __name__')[0])"]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if "ModuleNotFoundError" in result.stderr or "ImportError" in result.stderr:
                errors.append(f"{script.name}: import error")
        except Exception as e:
            errors.append(f"{script.name}: {e}")

    if errors:
        return TestResult("path_resolution", False, "; ".join(errors[:3]))

    return TestResult("path_resolution", True, "All scripts import successfully")


def main():
    print("=" * 70)
    print("JOSE COLBES PACKAGE INTEGRATION TEST SUITE")
    print("=" * 70)
    print()

    tests = [
        test_data_integrity,
        test_embeddings_extraction,
        test_result_files_exist,
        test_bootstrap_validation,
        test_c4_predictor,
    ]

    results = []
    for test_fn in tests:
        print(f"Running: {test_fn.__name__}...", end=" ", flush=True)
        try:
            result = test_fn()
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}]")
            if not result.passed:
                print(f"  -> {result.message}")
        except Exception as e:
            results.append(TestResult(test_fn.__name__, False, str(e)))
            print(f"[ERROR] {e}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}: {r.message}")

    print()
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! Package is ready for delivery.")
        return 0
    else:
        print(f"\n{total - passed} tests failed. Please fix before delivery.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
