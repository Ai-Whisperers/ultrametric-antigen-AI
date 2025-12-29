#!/usr/bin/env python3
"""
AlphaFold 3 Automated Validation Pipeline

Unified pipeline for running AF3 predictions across all disease research:
- Rheumatoid Arthritis (citrullination)
- HIV (drug resistance mutations)
- Neurodegeneration (aggregation-prone mutations)

Supports:
1. Local AF3 installation
2. AlphaFold Server API (alphafoldserver.com)
3. ColabFold alternative

Author: Research Team
Date: December 2025
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import shutil

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
VALIDATION_DIR = SCRIPT_DIR.parent
RESEARCH_DIR = VALIDATION_DIR.parent
QUEUE_DIR = VALIDATION_DIR / "queue"
RESULTS_DIR = VALIDATION_DIR / "results"

# AF3 paths
AF3_REPO = RESEARCH_DIR.parent.parent / "alphafold3" / "repo"
AF3_SCRIPT = AF3_REPO / "run_alphafold.py"

# Disease-specific input directories
DISEASE_DIRS = {
    "rheumatoid_arthritis": RESEARCH_DIR / "rheumatoid_arthritis" / "results" / "alphafold3",
    "hiv": RESEARCH_DIR / "hiv" / "results" / "alphafold3",
    "alzheimers": RESEARCH_DIR / "neurodegeneration" / "alzheimers" / "results" / "alphafold3",
}


@dataclass
class AF3Job:
    """Represents an AlphaFold 3 prediction job."""
    job_id: str
    input_path: Path
    output_dir: Path
    disease: str
    job_type: str  # native, citrullinated, comparison, hla_complex
    status: str = "pending"  # pending, running, completed, failed
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


# ============================================================================
# JOB DISCOVERY
# ============================================================================


def discover_pending_jobs() -> list[AF3Job]:
    """Scan all disease directories for pending AF3 jobs."""
    jobs = []

    for disease, base_dir in DISEASE_DIRS.items():
        if not base_dir.exists():
            continue

        inputs_dir = base_dir / "21_alphafold3_inputs"
        predictions_dir = base_dir / "predictions"

        if not inputs_dir.exists():
            continue

        # Scan subdirectories: native, citrullinated, comparison, hla_complexes
        for subdir in ["native", "citrullinated", "comparison", "hla_complexes"]:
            input_subdir = inputs_dir / subdir
            if not input_subdir.exists():
                continue

            for json_file in input_subdir.glob("*.json"):
                job_id = json_file.stem
                output_dir = predictions_dir / job_id

                # Check if already completed
                if output_dir.exists() and any(output_dir.glob("*_summary_confidences_*.json")):
                    continue  # Already done

                jobs.append(AF3Job(
                    job_id=job_id,
                    input_path=json_file,
                    output_dir=output_dir,
                    disease=disease,
                    job_type=subdir.rstrip("s"),  # Remove plural
                ))

    return jobs


def normalize_job_name(name: str) -> str:
    """Normalize job name for matching."""
    return name.lower().replace("-", "_").replace("hla_drb1", "drb1").replace("hla-drb1", "drb1")


def find_existing_prediction(job: AF3Job) -> Optional[Path]:
    """Find existing prediction for a job, checking various naming conventions."""
    predictions_dir = job.output_dir.parent

    if not predictions_dir.exists():
        return None

    job_normalized = normalize_job_name(job.job_id)

    # Check direct match
    if job.output_dir.exists():
        return job.output_dir

    # Check folds directories
    for fold_dir in predictions_dir.glob("folds_*"):
        if not fold_dir.is_dir():
            continue
        for subdir in fold_dir.iterdir():
            if not subdir.is_dir():
                continue
            subdir_normalized = normalize_job_name(subdir.name)
            if job_normalized in subdir_normalized or subdir_normalized in job_normalized:
                return subdir

    return None


def get_job_status(job: AF3Job) -> str:
    """Check the current status of a job."""
    # First check for existing predictions with any naming convention
    existing = find_existing_prediction(job)

    if existing:
        # Check for completion markers
        confidence_files = list(existing.glob("*_summary_confidences_*.json"))
        if confidence_files:
            return "completed"

        # Check for error markers
        error_files = list(existing.glob("*.error"))
        if error_files:
            return "failed"

        # Check if running (has partial outputs)
        if any(existing.iterdir()):
            return "running"

    if not job.output_dir.exists():
        return "pending"

    # Check for completion markers
    confidence_files = list(job.output_dir.glob("*_summary_confidences_*.json"))
    if confidence_files:
        return "completed"

    # Check for error markers
    error_files = list(job.output_dir.glob("*.error"))
    if error_files:
        return "failed"

    # Check if running (has partial outputs)
    if any(job.output_dir.iterdir()):
        return "running"

    return "pending"


# ============================================================================
# LOCAL AF3 EXECUTION
# ============================================================================


def run_local_af3(job: AF3Job, timeout: int = 7200) -> bool:
    """Run AF3 locally using the installed repository."""
    if not AF3_SCRIPT.exists():
        print(f"  ERROR: AF3 not found at {AF3_SCRIPT}")
        return False

    job.output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", str(AF3_SCRIPT),
        "--json_path", str(job.input_path),
        "--output_dir", str(job.output_dir),
    ]

    print(f"  Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return True
        else:
            job.error = result.stderr[:500]
            return False

    except subprocess.TimeoutExpired:
        job.error = f"Timeout after {timeout}s"
        return False
    except Exception as e:
        job.error = str(e)
        return False


# ============================================================================
# ALPHAFOLD SERVER API
# ============================================================================


def submit_to_af_server(job: AF3Job) -> Optional[str]:
    """Submit job to AlphaFold Server (alphafoldserver.com).

    Note: Requires API access. Currently limited to 20 jobs/day for academic use.
    Returns job tracking ID if successful.
    """
    # Read input JSON
    with open(job.input_path) as f:
        input_data = json.load(f)

    # AlphaFold Server uses a specific format
    # This is a placeholder - actual implementation requires their API
    print(f"  Would submit to AlphaFold Server: {job.job_id}")
    print(f"  Note: Manual submission required at alphafoldserver.com")
    print(f"  Input file: {job.input_path}")

    return None  # Return tracking ID when implemented


def check_af_server_status(tracking_id: str) -> dict:
    """Check status of job on AlphaFold Server."""
    # Placeholder for API implementation
    return {"status": "unknown", "tracking_id": tracking_id}


# ============================================================================
# RESULT PARSING
# ============================================================================


def parse_af3_results(output_dir: Path) -> dict:
    """Parse AF3 prediction results."""
    results = {
        "job_id": output_dir.name,
        "completed": False,
        "confidences": [],
        "structures": [],
    }

    # Find confidence files
    conf_files = sorted(output_dir.glob("*_summary_confidences_*.json"))
    for conf_file in conf_files:
        with open(conf_file) as f:
            conf_data = json.load(f)
        results["confidences"].append({
            "file": conf_file.name,
            "iptm": conf_data.get("iptm", 0),
            "ptm": conf_data.get("ptm", 0),
            "ranking_score": conf_data.get("ranking_score", 0),
        })

    # Find structure files
    cif_files = list(output_dir.glob("*.cif"))
    results["structures"] = [f.name for f in cif_files]

    results["completed"] = len(results["confidences"]) > 0

    # Best model
    if results["confidences"]:
        best = max(results["confidences"], key=lambda x: x.get("ranking_score", 0))
        results["best_model"] = best

    return results


def compare_native_vs_modified(native_results: dict, modified_results: dict) -> dict:
    """Compare native vs modified (citrullinated/mutant) predictions."""
    comparison = {
        "native_iptm": 0,
        "modified_iptm": 0,
        "iptm_change": 0,
        "iptm_change_pct": 0,
        "binding_improved": False,
    }

    if native_results.get("best_model"):
        comparison["native_iptm"] = native_results["best_model"].get("iptm", 0)

    if modified_results.get("best_model"):
        comparison["modified_iptm"] = modified_results["best_model"].get("iptm", 0)

    if comparison["native_iptm"] > 0:
        comparison["iptm_change"] = comparison["modified_iptm"] - comparison["native_iptm"]
        comparison["iptm_change_pct"] = (comparison["iptm_change"] / comparison["native_iptm"]) * 100
        comparison["binding_improved"] = comparison["modified_iptm"] > comparison["native_iptm"]

    return comparison


# ============================================================================
# BATCH PROCESSING
# ============================================================================


def process_batch(jobs: list[AF3Job], method: str = "local", max_jobs: int = 5) -> dict:
    """Process a batch of AF3 jobs."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "method": method,
        "total_jobs": len(jobs),
        "completed": 0,
        "failed": 0,
        "pending": 0,
        "jobs": [],
    }

    for i, job in enumerate(jobs[:max_jobs]):
        print(f"\nJob {i+1}/{min(len(jobs), max_jobs)}: {job.job_id}")
        print(f"  Disease: {job.disease}")
        print(f"  Type: {job.job_type}")

        job.submitted_at = datetime.now().isoformat()

        if method == "local":
            success = run_local_af3(job)
        elif method == "server":
            tracking_id = submit_to_af_server(job)
            success = tracking_id is not None
        else:
            print(f"  Unknown method: {method}")
            success = False

        if success:
            job.status = "completed"
            job.completed_at = datetime.now().isoformat()
            results["completed"] += 1
        else:
            job.status = "failed"
            results["failed"] += 1

        results["jobs"].append({
            "job_id": job.job_id,
            "disease": job.disease,
            "type": job.job_type,
            "status": job.status,
            "error": job.error,
        })

    results["pending"] = len(jobs) - min(len(jobs), max_jobs)

    return results


# ============================================================================
# REPORT GENERATION
# ============================================================================


def generate_validation_report(disease: str = None) -> dict:
    """Generate validation report for completed predictions."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "disease": disease or "all",
        "comparisons": [],
        "summary": {},
    }

    for dis, base_dir in DISEASE_DIRS.items():
        if disease and dis != disease:
            continue

        predictions_dir = base_dir / "predictions"
        if not predictions_dir.exists():
            continue

        # Find native/modified pairs
        for pred_dir in predictions_dir.iterdir():
            if not pred_dir.is_dir():
                continue

            name = pred_dir.name.lower()

            # Look for native version
            if "_cit" in name or "_mutant" in name or "_modified" in name:
                native_name = name.replace("_cit", "_native").replace("_mutant", "_native").replace("_modified", "_native")
                native_dir = predictions_dir / native_name

                if native_dir.exists():
                    native_results = parse_af3_results(native_dir)
                    modified_results = parse_af3_results(pred_dir)

                    if native_results["completed"] and modified_results["completed"]:
                        comparison = compare_native_vs_modified(native_results, modified_results)
                        comparison["pair"] = f"{native_name} vs {name}"
                        comparison["disease"] = dis
                        report["comparisons"].append(comparison)

    # Summary statistics
    if report["comparisons"]:
        improvements = [c for c in report["comparisons"] if c["binding_improved"]]
        report["summary"] = {
            "total_pairs": len(report["comparisons"]),
            "binding_improved": len(improvements),
            "improvement_rate": len(improvements) / len(report["comparisons"]) * 100,
            "mean_iptm_change_pct": sum(c["iptm_change_pct"] for c in report["comparisons"]) / len(report["comparisons"]),
        }

    return report


# ============================================================================
# BATCH EXPORT FOR ALPHAFOLD SERVER
# ============================================================================


def export_for_server(jobs: list[AF3Job], output_dir: Path, batch_size: int = 20) -> dict:
    """Export jobs in batches for AlphaFold Server submission.

    AlphaFold Server limits: 20 jobs/day for academic use.
    Creates organized batches with priority ranking.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Priority order: HLA complexes > comparison > citrullinated > native
    priority_order = {"hla_complex": 0, "hla_complexe": 0, "comparison": 1, "citrullinated": 2, "native": 3}
    sorted_jobs = sorted(jobs, key=lambda j: priority_order.get(j.job_type, 99))

    batches = []
    batch_num = 1

    for i in range(0, len(sorted_jobs), batch_size):
        batch = sorted_jobs[i:i + batch_size]
        batch_dir = output_dir / f"batch_{batch_num:02d}"
        batch_dir.mkdir(exist_ok=True)

        batch_info = {
            "batch": batch_num,
            "jobs": [],
            "directory": str(batch_dir),
        }

        for job in batch:
            # Copy input file to batch directory
            dest = batch_dir / job.input_path.name
            shutil.copy2(job.input_path, dest)

            batch_info["jobs"].append({
                "job_id": job.job_id,
                "disease": job.disease,
                "type": job.job_type,
                "file": job.input_path.name,
            })

        batches.append(batch_info)
        batch_num += 1

    # Write batch manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "total_jobs": len(jobs),
        "batch_size": batch_size,
        "num_batches": len(batches),
        "estimated_days": len(batches),  # One batch per day
        "batches": batches,
    }

    manifest_file = output_dir / "batch_manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    # Write instructions
    instructions = output_dir / "SUBMISSION_INSTRUCTIONS.md"
    with open(instructions, "w") as f:
        f.write("# AlphaFold Server Batch Submission\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"## Summary\n")
        f.write(f"- Total jobs: {len(jobs)}\n")
        f.write(f"- Batches: {len(batches)}\n")
        f.write(f"- Estimated days: {len(batches)} (20 jobs/day limit)\n\n")
        f.write("## Submission Process\n\n")
        f.write("1. Go to https://alphafoldserver.com\n")
        f.write("2. Sign in with Google account\n")
        f.write("3. For each batch:\n")
        f.write("   - Click 'New Job'\n")
        f.write("   - Upload JSON files from batch directory\n")
        f.write("   - Submit and wait for completion email\n")
        f.write("4. Download results and place in predictions/ directory\n\n")
        f.write("## Batch Contents\n\n")
        for batch in batches:
            f.write(f"### Batch {batch['batch']}\n")
            f.write(f"Directory: `{batch['directory']}`\n")
            f.write(f"Jobs: {len(batch['jobs'])}\n\n")
            for job in batch['jobs']:
                f.write(f"- {job['job_id']} ({job['type']})\n")
            f.write("\n")

    return manifest


def import_server_results(results_zip: Path, predictions_dir: Path) -> dict:
    """Import results downloaded from AlphaFold Server."""
    import zipfile

    if not results_zip.exists():
        return {"error": f"File not found: {results_zip}"}

    # Create timestamped folder
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    output_dir = predictions_dir / f"folds_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract zip
    with zipfile.ZipFile(results_zip, 'r') as zf:
        zf.extractall(output_dir)

    # Count extracted predictions
    subdirs = [d for d in output_dir.iterdir() if d.is_dir()]

    return {
        "status": "success",
        "output_dir": str(output_dir),
        "predictions_extracted": len(subdirs),
        "timestamp": timestamp,
    }


# ============================================================================
# MAIN CLI
# ============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AlphaFold 3 Validation Pipeline")
    parser.add_argument("command", choices=["discover", "run", "status", "report", "queue", "export", "import"],
                        help="Command to execute")
    parser.add_argument("--disease", choices=["rheumatoid_arthritis", "hiv", "alzheimers"],
                        help="Filter by disease")
    parser.add_argument("--method", choices=["local", "server"], default="local",
                        help="Execution method")
    parser.add_argument("--max-jobs", type=int, default=5,
                        help="Maximum jobs to process")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Jobs per batch for server export (default: 20)")
    parser.add_argument("--zip", type=Path, help="ZIP file to import from server")
    parser.add_argument("--output", type=Path, help="Output file for results")

    args = parser.parse_args()

    if args.command == "discover":
        print("Discovering pending AF3 jobs...\n")
        jobs = discover_pending_jobs()

        if args.disease:
            jobs = [j for j in jobs if j.disease == args.disease]

        print(f"Found {len(jobs)} pending jobs:\n")

        by_disease = {}
        for job in jobs:
            by_disease.setdefault(job.disease, []).append(job)

        for disease, disease_jobs in by_disease.items():
            print(f"\n{disease.upper()}:")
            for job in disease_jobs[:10]:
                print(f"  - {job.job_id} ({job.job_type})")
            if len(disease_jobs) > 10:
                print(f"  ... and {len(disease_jobs) - 10} more")

        # Save queue
        queue_file = QUEUE_DIR / "pending_jobs.json"
        queue_data = [{"job_id": j.job_id, "disease": j.disease, "type": j.job_type,
                       "input": str(j.input_path)} for j in jobs]
        with open(queue_file, "w") as f:
            json.dump(queue_data, f, indent=2)
        print(f"\nQueue saved to: {queue_file}")

    elif args.command == "run":
        print("Running AF3 predictions...\n")
        jobs = discover_pending_jobs()

        if args.disease:
            jobs = [j for j in jobs if j.disease == args.disease]

        if not jobs:
            print("No pending jobs found.")
            return

        print(f"Processing {min(len(jobs), args.max_jobs)} of {len(jobs)} jobs...\n")

        results = process_batch(jobs, method=args.method, max_jobs=args.max_jobs)

        # Save results
        output_file = args.output or (RESULTS_DIR / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    elif args.command == "status":
        print("Checking job status...\n")
        jobs = discover_pending_jobs()

        pending = [j for j in jobs if get_job_status(j) == "pending"]
        running = [j for j in jobs if get_job_status(j) == "running"]
        completed = [j for j in jobs if get_job_status(j) == "completed"]
        failed = [j for j in jobs if get_job_status(j) == "failed"]

        print(f"Pending:   {len(pending)}")
        print(f"Running:   {len(running)}")
        print(f"Completed: {len(completed)}")
        print(f"Failed:    {len(failed)}")

    elif args.command == "report":
        print("Generating validation report...\n")
        report = generate_validation_report(args.disease)

        if report["comparisons"]:
            print(f"Comparisons: {report['summary']['total_pairs']}")
            print(f"Binding improved: {report['summary']['binding_improved']} ({report['summary']['improvement_rate']:.1f}%)")
            print(f"Mean iPTM change: {report['summary']['mean_iptm_change_pct']:+.1f}%")

            print("\nTop improvements:")
            sorted_comps = sorted(report["comparisons"], key=lambda x: x["iptm_change_pct"], reverse=True)
            for comp in sorted_comps[:5]:
                print(f"  {comp['pair']}: {comp['iptm_change_pct']:+.1f}%")
        else:
            print("No completed comparisons found.")

        # Save report
        output_file = args.output or (RESULTS_DIR / f"validation_report_{datetime.now().strftime('%Y%m%d')}.json")
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_file}")

    elif args.command == "queue":
        print("Current queue:\n")
        queue_file = QUEUE_DIR / "pending_jobs.json"
        if queue_file.exists():
            with open(queue_file) as f:
                queue = json.load(f)
            print(f"Total jobs in queue: {len(queue)}")
            for job in queue[:20]:
                print(f"  - [{job['disease']}] {job['job_id']}")
        else:
            print("No queue file found. Run 'discover' first.")

    elif args.command == "export":
        print("Exporting jobs for AlphaFold Server submission...\n")
        jobs = discover_pending_jobs()

        if args.disease:
            jobs = [j for j in jobs if j.disease == args.disease]

        # Filter to only pending jobs
        jobs = [j for j in jobs if get_job_status(j) == "pending"]

        if not jobs:
            print("No pending jobs to export.")
            return

        export_dir = args.output or (VALIDATION_DIR / "server_export" / datetime.now().strftime("%Y%m%d_%H%M%S"))
        manifest = export_for_server(jobs, export_dir, batch_size=args.batch_size)

        print(f"Exported {manifest['total_jobs']} jobs in {manifest['num_batches']} batches")
        print(f"Estimated completion: {manifest['estimated_days']} days (20 jobs/day limit)")
        print(f"\nExport directory: {export_dir}")
        print(f"Instructions: {export_dir / 'SUBMISSION_INSTRUCTIONS.md'}")

    elif args.command == "import":
        if not args.zip:
            print("ERROR: --zip argument required for import command")
            return

        print(f"Importing results from: {args.zip}\n")

        # Determine predictions directory
        if args.disease:
            predictions_dir = DISEASE_DIRS[args.disease] / "predictions"
        else:
            print("ERROR: --disease argument required for import command")
            return

        result = import_server_results(args.zip, predictions_dir)

        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"Successfully imported {result['predictions_extracted']} predictions")
            print(f"Output directory: {result['output_dir']}")


if __name__ == "__main__":
    main()
