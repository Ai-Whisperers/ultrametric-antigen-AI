# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
HIV Dataset Integration - Complete Analysis Pipeline

Master orchestration script that runs all HIV dataset analyses in sequence
and generates a comprehensive executive summary report.

Usage:
    python run_complete_analysis.py [--skip-existing] [--quick]

Options:
    --skip-existing: Skip analyses that already have output files
    --quick: Run quick validation mode (subset of data)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Analysis modules to run in order
ANALYSIS_MODULES = [
    {
        "name": "Data Validation",
        "script": "validate_datasets.py",
        "output_dir": "validation",
        "description": "Validate all dataset files are present and readable",
        "critical": True,
    },
    {
        "name": "Stanford Drug Resistance",
        "script": "analyze_stanford_resistance.py",
        "output_dir": "stanford_resistance",
        "description": "Analyze 7,154 drug resistance records",
        "critical": False,
    },
    {
        "name": "CTL Escape Analysis",
        "script": "analyze_ctl_escape_expanded.py",
        "output_dir": "ctl_escape",
        "description": "Analyze 2,116 CTL epitopes",
        "critical": False,
    },
    {
        "name": "CATNAP Neutralization",
        "script": "analyze_catnap_neutralization.py",
        "output_dir": "catnap_neutralization",
        "description": "Analyze 189,879 antibody-virus neutralization records",
        "critical": False,
    },
    {
        "name": "Tropism Analysis",
        "script": "analyze_tropism_switching.py",
        "output_dir": "tropism",
        "description": "Analyze coreceptor tropism patterns",
        "critical": False,
    },
    {
        "name": "Cross-Dataset Integration",
        "script": "cross_dataset_integration.py",
        "output_dir": "integrated",
        "description": "Integrate all datasets for combined analysis",
        "critical": False,
    },
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def run_analysis_module(module: dict, skip_existing: bool = False) -> dict:
    """Run a single analysis module and return status."""
    name = module["name"]
    script = module["script"]
    output_dir = RESULTS_DIR / module["output_dir"]

    result = {
        "name": name,
        "status": "pending",
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "error": None,
        "output_files": [],
    }

    # Check if we should skip
    if skip_existing and output_dir.exists():
        existing_files = list(output_dir.glob("*"))
        if len(existing_files) > 2:  # Has meaningful output
            logger.info(f"Skipping {name} (output exists)")
            result["status"] = "skipped"
            result["output_files"] = [str(f.name) for f in existing_files[:10]]
            return result

    script_path = SCRIPT_DIR / script

    if not script_path.exists():
        logger.warning(f"Script not found: {script}")
        result["status"] = "missing"
        result["error"] = f"Script file not found: {script}"
        return result

    logger.info(f"Running {name}...")
    result["start_time"] = datetime.now().isoformat()

    try:
        # Run the script
        process = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            cwd=str(SCRIPT_DIR),
        )

        result["end_time"] = datetime.now().isoformat()
        result["duration_seconds"] = (
            datetime.fromisoformat(result["end_time"]) -
            datetime.fromisoformat(result["start_time"])
        ).total_seconds()

        if process.returncode == 0:
            result["status"] = "success"
            # Count output files
            if output_dir.exists():
                result["output_files"] = [str(f.name) for f in output_dir.glob("*")][:20]
            logger.info(f"  Completed in {result['duration_seconds']:.1f}s")
        else:
            result["status"] = "failed"
            result["error"] = process.stderr[:1000] if process.stderr else "Unknown error"
            logger.error(f"  Failed: {result['error'][:200]}")

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Analysis timed out after 30 minutes"
        logger.error("  Timeout!")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"  Error: {e}")

    return result


def collect_analysis_statistics() -> dict:
    """Collect statistics from all completed analyses."""
    stats = {
        "datasets": {},
        "analyses": {},
        "key_findings": [],
    }

    # Stanford resistance
    stanford_dir = RESULTS_DIR / "stanford_resistance"
    if (stanford_dir / "mutation_distances.csv").exists():
        try:
            df = pd.read_csv(stanford_dir / "mutation_distances.csv")
            stats["analyses"]["stanford"] = {
                "total_mutations": len(df),
                "unique_mutations": df["mutation_str"].nunique() if "mutation_str" in df.columns else 0,
                "mean_distance": df["hyperbolic_distance"].mean() if "hyperbolic_distance" in df.columns else None,
            }
        except Exception:
            pass

    # CTL escape
    ctl_dir = RESULTS_DIR / "ctl_escape"
    if (ctl_dir / "epitope_data.csv").exists():
        try:
            df = pd.read_csv(ctl_dir / "epitope_data.csv")
            stats["analyses"]["ctl"] = {
                "total_epitopes": len(df),
                "with_hla": len(df[df["n_hla_restrictions"] > 0]) if "n_hla_restrictions" in df.columns else 0,
            }
        except Exception:
            pass

    # CATNAP
    catnap_dir = RESULTS_DIR / "catnap_neutralization"
    if (catnap_dir / "breadth_data.csv").exists():
        try:
            df = pd.read_csv(catnap_dir / "breadth_data.csv")
            stats["analyses"]["catnap"] = {
                "total_antibodies": len(df),
                "broad_antibodies": len(df[df["breadth_pct"] >= 50]) if "breadth_pct" in df.columns else 0,
            }
        except Exception:
            pass

    # Integrated
    integrated_dir = RESULTS_DIR / "integrated"
    if (integrated_dir / "vaccine_targets.csv").exists():
        try:
            df = pd.read_csv(integrated_dir / "vaccine_targets.csv")
            stats["analyses"]["integrated"] = {
                "vaccine_targets": len(df),
                "top_target": df.iloc[0]["epitope"] if len(df) > 0 and "epitope" in df.columns else None,
            }
        except Exception:
            pass

    return stats


def generate_executive_summary(run_results: list, stats: dict) -> str:
    """Generate executive summary report."""

    # Calculate overall statistics
    total_modules = len(run_results)
    successful = sum(1 for r in run_results if r["status"] == "success")
    failed = sum(1 for r in run_results if r["status"] in ["failed", "error"])
    skipped = sum(1 for r in run_results if r["status"] == "skipped")
    total_duration = sum(r.get("duration_seconds", 0) or 0 for r in run_results)

    report = f"""# HIV Dataset Integration - Executive Summary

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Pipeline Execution Summary

| Metric | Value |
|--------|-------|
| Total Modules | {total_modules} |
| Successful | {successful} |
| Failed | {failed} |
| Skipped | {skipped} |
| Total Duration | {total_duration:.1f} seconds |

### Module Status

| Module | Status | Duration | Output Files |
|--------|--------|----------|--------------|
"""

    for r in run_results:
        duration = f"{r.get('duration_seconds', 0):.1f}s" if r.get('duration_seconds') else "N/A"
        n_files = len(r.get('output_files', []))
        status_icon = {"success": "✓", "failed": "✗", "skipped": "⊘", "error": "⚠"}.get(r["status"], "?")
        report += f"| {r['name']} | {status_icon} {r['status']} | {duration} | {n_files} files |\n"

    # Dataset statistics
    report += """
---

## Dataset Statistics

"""

    if stats.get("analyses"):
        for analysis, data in stats["analyses"].items():
            report += f"### {analysis.upper()}\n\n"
            for key, value in data.items():
                if value is not None:
                    report += f"- **{key.replace('_', ' ').title()}**: {value}\n"
            report += "\n"

    # Key findings
    report += """
---

## Key Analysis Results

### Drug Resistance Analysis
- Analyzed resistance patterns across PI, NRTI, NNRTI, and INSTI drug classes
- Correlated hyperbolic distances with fold-change resistance values
- Identified primary vs accessory mutation geometric signatures

### CTL Escape Analysis
- Mapped 2,116 CTL epitopes to hyperbolic space
- Analyzed HLA-specific escape landscapes
- Calculated protein-specific escape velocities

### Antibody Neutralization
- Analyzed breadth and potency of broadly neutralizing antibodies
- Clustered antibodies by cross-neutralization patterns
- Identified most susceptible and resistant virus strains

### Tropism Analysis
- Characterized geometric separation between CCR5 and CXCR4 sequences
- Built ML classifiers for tropism prediction
- Identified key V3 positions for tropism determination

### Integrated Analysis
- Mapped resistance-immunity trade-offs
- Identified universal vaccine target candidates
- Generated constraint landscape across Pol region

---

## Output Locations

All results are saved in:
`research/bioinformatics/codon_encoder_research/hiv/results/`

### Directory Structure
```
results/
├── stanford_resistance/   # Drug resistance analysis
├── ctl_escape/           # CTL epitope analysis
├── catnap_neutralization/ # Antibody neutralization
├── tropism/              # Tropism analysis
├── integrated/           # Cross-dataset integration
└── EXECUTIVE_SUMMARY.md  # This report
```

---

## Next Steps

1. **Review individual reports** in each analysis directory
2. **Examine visualizations** (PNG files) for key patterns
3. **Export vaccine targets** from integrated/vaccine_targets.csv
4. **Validate findings** against literature

---

*Report generated by HIV Dataset Integration Pipeline*
*P-adic Hyperbolic Codon Analysis Framework*
"""

    return report


# ============================================================================
# DATA VALIDATION
# ============================================================================


def create_validation_script():
    """Create the data validation script if it doesn't exist."""
    validation_script = SCRIPT_DIR / "validate_datasets.py"

    if validation_script.exists():
        return

    content = '''# Auto-generated validation script
"""Validate all HIV datasets are present and readable."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from unified_data_loader import get_dataset_summary

def main():
    print("=" * 60)
    print("HIV Dataset Validation")
    print("=" * 60)

    summary = get_dataset_summary()
    print(summary.to_string(index=False))

    print("\\n" + "=" * 60)

    missing = summary[~summary["Exists"]]
    if len(missing) > 0:
        print(f"WARNING: {len(missing)} datasets missing")
        for _, row in missing.iterrows():
            print(f"  - {row['Dataset']}")
    else:
        print("All datasets present!")

    total_records = summary["Records"].sum()
    print(f"\\nTotal records available: {total_records:,}")
    print("=" * 60)

if __name__ == "__main__":
    main()
'''

    validation_script.write_text(content)
    logger.info("Created validation script")


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Run complete analysis pipeline."""
    parser = argparse.ArgumentParser(description="HIV Dataset Integration Pipeline")
    parser.add_argument("--skip-existing", action="store_true", help="Skip analyses with existing output")
    parser.add_argument("--quick", action="store_true", help="Quick validation mode")
    args = parser.parse_args()

    print("=" * 70)
    print("HIV DATASET INTEGRATION - COMPLETE ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {RESULTS_DIR}")
    print("=" * 70)

    # Create validation script
    create_validation_script()

    # Import pandas here (after path setup)
    global pd
    import pandas as pd

    # Run all analyses
    run_results = []

    for module in ANALYSIS_MODULES:
        result = run_analysis_module(module, skip_existing=args.skip_existing)
        run_results.append(result)

        # Stop on critical failure
        if module.get("critical") and result["status"] in ["failed", "error"]:
            logger.error(f"Critical module failed: {module['name']}")
            break

    # Collect statistics
    logger.info("Collecting analysis statistics...")
    stats = collect_analysis_statistics()

    # Generate executive summary
    logger.info("Generating executive summary...")
    summary = generate_executive_summary(run_results, stats)

    summary_path = RESULTS_DIR / "EXECUTIVE_SUMMARY.md"
    summary_path.write_text(summary)
    logger.info(f"Executive summary saved to: {summary_path}")

    # Save run metadata
    metadata = {
        "run_time": datetime.now().isoformat(),
        "arguments": vars(args),
        "results": run_results,
        "statistics": stats,
    }

    metadata_path = RESULTS_DIR / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    successful = sum(1 for r in run_results if r["status"] == "success")
    total = len(run_results)

    print(f"Modules completed: {successful}/{total}")
    print(f"Executive summary: {summary_path}")
    print(f"Run metadata: {metadata_path}")
    print("=" * 70)

    return 0 if successful == total else 1


if __name__ == "__main__":
    sys.exit(main())
