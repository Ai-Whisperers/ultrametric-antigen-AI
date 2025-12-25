#!/usr/bin/env python
"""
Master visualization generator for RA research.
Generates all pitch and scientific visualizations.
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent


def run_generator(script_path: Path, name: str) -> bool:
    """Run a visualization generator script."""
    print(f"\n{'='*60}")
    print(f"Generating: {name}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print(f"SUCCESS: {name}")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"ERROR: {name}")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"EXCEPTION: {name} - {e}")
        return False


def main():
    print("=" * 60)
    print("RA VISUALIZATION GENERATOR")
    print("=" * 60)

    generators = [
        # Pitch visualizations
        (
            BASE_DIR / "pitch/01_pathophysiology_funnel/generate.py",
            "Pathophysiology Funnel",
        ),
        (BASE_DIR / "pitch/02_hla_risk_charts/generate.py", "HLA Risk Charts"),
        (
            BASE_DIR / "pitch/03_intervention_pathways/generate.py",
            "Intervention Pathways",
        ),
        (
            BASE_DIR / "pitch/04_safety_comparisons/generate.py",
            "Safety Comparisons",
        ),
        (
            BASE_DIR / "pitch/05_goldilocks_radar/generate.py",
            "Goldilocks Zone",
        ),
        # Scientific visualizations
        (
            BASE_DIR / "scientific/01_hla_pca_projections/generate.py",
            "HLA PCA Projections",
        ),
        (
            BASE_DIR / "scientific/02_cluster_boundary_3d/generate.py",
            "3D Cluster Boundaries",
        ),
        (
            BASE_DIR / "scientific/04_calabi_yau_manifolds/generate.py",
            "Calabi-Yau Manifolds",
        ),
        (
            BASE_DIR / "scientific/05_distance_heatmaps/generate.py",
            "Distance Heatmaps",
        ),
    ]

    results = []
    for script_path, name in generators:
        if script_path.exists():
            success = run_generator(script_path, name)
            results.append((name, success))
        else:
            print(f"SKIP: {name} - script not found at {script_path}")
            results.append((name, None))

    # Summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)

    successes = sum(1 for _, r in results if r is True)
    failures = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)

    for name, result in results:
        status = "OK" if result is True else "FAIL" if result is False else "SKIP"
        print(f"  [{status:4}] {name}")

    print(f"\nTotal: {successes} success, {failures} failed, {skipped} skipped")

    # List generated files
    print("\n" + "=" * 60)
    print("GENERATED FILES")
    print("=" * 60)

    for subdir in ["pitch", "scientific"]:
        dir_path = BASE_DIR / subdir
        if dir_path.exists():
            print(f"\n{subdir.upper()}/")
            for child in sorted(dir_path.rglob("*")):
                if child.is_file() and child.suffix in [
                    ".png",
                    ".svg",
                    ".html",
                ]:
                    rel_path = child.relative_to(BASE_DIR)
                    size_kb = child.stat().st_size / 1024
                    print(f"  {rel_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
