#!/usr/bin/env python3
"""P1 Conjecture 5 Test: Pathogen specificity requires NON-AMP information.

Hypothesis:
    MIC specificity requires signals not present in peptide-only data.
    Pathogen metadata (Gram type, membrane composition) may provide this.

CRITICAL - R2 CONSTRAINT (Hold-Out Generalization):
    Train on SUBSET of pathogens, evaluate on HELD-OUT pathogens.
    If separation COLLAPSES on held-out = non-generalizable lookup.
    This is an INFORMATION AVAILABILITY test, not a modeling exercise.

Test Design:
    1. Split: Train on 3 pathogens, hold out 2
    2. Train model using pathogen metadata as features
    3. Evaluate on held-out pathogens (with their metadata)
    4. If performance degrades significantly → lookup behavior → FALSIFIED

Falsifies if:
    Metadata adds no predictive power, OR
    Performance collapses on held-out pathogens (R2 violation)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PACKAGE_DIR / "results" / "validation_batch"

# Pathogen metadata (from pathogens.json - available at inference time)
PATHOGEN_METADATA = {
    "A_baumannii": {
        "gram": "negative",
        "LPS_abundance": 0.85,
        "net_charge": -0.6,
        "priority_critical": 1,
    },
    "P_aeruginosa": {
        "gram": "negative",
        "LPS_abundance": 0.90,
        "net_charge": -0.7,
        "priority_critical": 1,
    },
    "Enterobacteriaceae": {
        "gram": "negative",
        "LPS_abundance": 0.88,
        "net_charge": -0.55,
        "priority_critical": 1,
    },
    "S_aureus": {
        "gram": "positive",
        "LPS_abundance": 0.0,  # Gram+, no LPS
        "net_charge": -0.3,
        "priority_critical": 0,
    },
    "H_pylori": {
        "gram": "negative",
        "LPS_abundance": 0.75,
        "net_charge": -0.4,
        "priority_critical": 0,
    },
}

# R2: Hold-out splits - define BEFORE any analysis
# We'll test multiple splits to ensure robustness
HOLDOUT_SPLITS = [
    # Split 1: Hold out one Gram+ and one Gram-
    {"train": ["A_baumannii", "P_aeruginosa", "Enterobacteriaceae"], "test": ["S_aureus", "H_pylori"]},
    # Split 2: Hold out two Gram-
    {"train": ["S_aureus", "H_pylori", "A_baumannii"], "test": ["P_aeruginosa", "Enterobacteriaceae"]},
    # Split 3: Hold out critical priority pathogens
    {"train": ["S_aureus", "H_pylori", "Enterobacteriaceae"], "test": ["A_baumannii", "P_aeruginosa"]},
]


def load_all_candidates(results_dir: Path) -> List[dict]:
    """Load candidates from all pathogen result files."""
    import csv

    pathogens = list(PATHOGEN_METADATA.keys())
    all_candidates = []

    for pathogen in pathogens:
        csv_path = results_dir / f"{pathogen}_candidates.csv"
        if not csv_path.exists():
            continue

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_candidates.append({
                    "sequence": row["sequence"],
                    "mic_pred": float(row["mic_pred"]),
                    "net_charge": float(row["net_charge"]),
                    "hydrophobicity": float(row["hydrophobicity"]),
                    "length": len(row["sequence"]),
                    "pathogen": pathogen,
                })

    return all_candidates


def extract_peptide_features(c: dict) -> np.ndarray:
    """Extract peptide-only features (available for any peptide)."""
    return np.array([
        c["length"],
        c["net_charge"],
        c["hydrophobicity"],
    ])


def extract_metadata_features(pathogen: str) -> np.ndarray:
    """Extract pathogen metadata features (available at inference if target known)."""
    meta = PATHOGEN_METADATA.get(pathogen, {})
    return np.array([
        1.0 if meta.get("gram") == "negative" else 0.0,
        meta.get("LPS_abundance", 0.0),
        meta.get("net_charge", 0.0),
        meta.get("priority_critical", 0.0),
    ])


def run_holdout_test(
    candidates: List[dict],
    train_pathogens: List[str],
    test_pathogens: List[str],
) -> Dict:
    """Run R2 hold-out test: train on subset, evaluate on held-out.

    Compares:
    - Peptide-only model
    - Peptide + Metadata model

    Returns metrics for both on train and test sets.
    """
    # Split data
    train_data = [c for c in candidates if c["pathogen"] in train_pathogens]
    test_data = [c for c in candidates if c["pathogen"] in test_pathogens]

    if len(train_data) < 10 or len(test_data) < 10:
        return {"error": "Insufficient data", "train_n": len(train_data), "test_n": len(test_data)}

    # Extract features and targets
    X_train_pep = np.array([extract_peptide_features(c) for c in train_data])
    X_test_pep = np.array([extract_peptide_features(c) for c in test_data])

    X_train_meta = np.array([extract_metadata_features(c["pathogen"]) for c in train_data])
    X_test_meta = np.array([extract_metadata_features(c["pathogen"]) for c in test_data])

    X_train_combined = np.hstack([X_train_pep, X_train_meta])
    X_test_combined = np.hstack([X_test_pep, X_test_meta])

    y_train = np.array([c["mic_pred"] for c in train_data])
    y_test = np.array([c["mic_pred"] for c in test_data])

    # Scale features
    scaler_pep = StandardScaler()
    scaler_comb = StandardScaler()

    X_train_pep_s = scaler_pep.fit_transform(X_train_pep)
    X_test_pep_s = scaler_pep.transform(X_test_pep)

    X_train_comb_s = scaler_comb.fit_transform(X_train_combined)
    X_test_comb_s = scaler_comb.transform(X_test_combined)

    # Model 1: Peptide-only
    model_pep = Ridge(alpha=1.0)
    model_pep.fit(X_train_pep_s, y_train)
    y_pred_pep_train = model_pep.predict(X_train_pep_s)
    y_pred_pep_test = model_pep.predict(X_test_pep_s)

    # Model 2: Peptide + Metadata
    model_comb = Ridge(alpha=1.0)
    model_comb.fit(X_train_comb_s, y_train)
    y_pred_comb_train = model_comb.predict(X_train_comb_s)
    y_pred_comb_test = model_comb.predict(X_test_comb_s)

    # Metrics
    def compute_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        corr, p = stats.spearmanr(y_true, y_pred)
        return {"mse": mse, "rmse": np.sqrt(mse), "spearman_r": corr, "p_value": p}

    results = {
        "train_pathogens": train_pathogens,
        "test_pathogens": test_pathogens,
        "train_n": len(train_data),
        "test_n": len(test_data),
        "peptide_only": {
            "train": compute_metrics(y_train, y_pred_pep_train),
            "test": compute_metrics(y_test, y_pred_pep_test),
        },
        "peptide_plus_metadata": {
            "train": compute_metrics(y_train, y_pred_comb_train),
            "test": compute_metrics(y_test, y_pred_comb_test),
        },
    }

    # Generalization check
    pep_train_r = results["peptide_only"]["train"]["spearman_r"]
    pep_test_r = results["peptide_only"]["test"]["spearman_r"]
    comb_train_r = results["peptide_plus_metadata"]["train"]["spearman_r"]
    comb_test_r = results["peptide_plus_metadata"]["test"]["spearman_r"]

    # Does metadata improve test performance?
    metadata_helps_test = comb_test_r > pep_test_r + 0.05  # Meaningful improvement

    # Does metadata generalize or is it lookup?
    # If train >> test for combined model, it's memorizing
    comb_generalizes = comb_test_r > 0.5 * comb_train_r if comb_train_r > 0 else False

    results["analysis"] = {
        "metadata_improves_test": metadata_helps_test,
        "metadata_improvement": comb_test_r - pep_test_r,
        "combined_generalizes": comb_generalizes,
        "generalization_ratio": comb_test_r / comb_train_r if comb_train_r > 0 else 0,
    }

    return results


def main():
    print("=" * 70)
    print("P1 CONJECTURE 5 TEST: Non-AMP Information (Pathogen Metadata)")
    print("=" * 70)
    print()
    print("R2 CONSTRAINT: Hold-out generalization required.")
    print("If metadata helps on training but not held-out pathogens → LOOKUP behavior")
    print()

    # Load data
    print("Loading candidates...")
    candidates = load_all_candidates(RESULTS_DIR)
    if not candidates:
        print("ERROR: No candidates found")
        sys.exit(1)
    print(f"Loaded {len(candidates)} candidates")
    print()

    # Report pathogen distribution
    by_pathogen = defaultdict(int)
    for c in candidates:
        by_pathogen[c["pathogen"]] += 1
    print("Pathogen distribution:")
    for p, n in sorted(by_pathogen.items()):
        print(f"  {p}: {n}")
    print()

    # Run all hold-out splits
    print("-" * 70)
    print("R2 HOLD-OUT TESTS")
    print("-" * 70)

    all_results = []
    for i, split in enumerate(HOLDOUT_SPLITS):
        print(f"\nSplit {i+1}: Train on {split['train']}, Hold out {split['test']}")
        print("-" * 50)

        result = run_holdout_test(candidates, split["train"], split["test"])

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        all_results.append(result)

        pep = result["peptide_only"]
        comb = result["peptide_plus_metadata"]
        analysis = result["analysis"]

        print(f"  Train N: {result['train_n']}, Test N: {result['test_n']}")
        print()
        print("  Peptide-only model:")
        print(f"    Train: r={pep['train']['spearman_r']:.3f}")
        print(f"    Test:  r={pep['test']['spearman_r']:.3f}")
        print()
        print("  Peptide + Metadata model:")
        print(f"    Train: r={comb['train']['spearman_r']:.3f}")
        print(f"    Test:  r={comb['test']['spearman_r']:.3f}")
        print()
        print(f"  Metadata improvement on test: {analysis['metadata_improvement']:.3f}")
        print(f"  Generalization ratio: {analysis['generalization_ratio']:.2f}")
        print(f"  --> {'GENERALIZES' if analysis['combined_generalizes'] else 'LOOKUP BEHAVIOR'}")

    if not all_results:
        print("ERROR: All splits failed")
        sys.exit(1)

    # Aggregate across splits
    print()
    print("=" * 70)
    print("CONJECTURE 5 VERDICT")
    print("=" * 70)

    # Criteria:
    # 1. Metadata must improve test performance on MOST splits
    # 2. Combined model must generalize (test/train ratio > 0.5) on MOST splits

    n_metadata_helps = sum(1 for r in all_results if r["analysis"]["metadata_improves_test"])
    n_generalizes = sum(1 for r in all_results if r["analysis"]["combined_generalizes"])
    n_splits = len(all_results)

    avg_improvement = np.mean([r["analysis"]["metadata_improvement"] for r in all_results])
    avg_gen_ratio = np.mean([r["analysis"]["generalization_ratio"] for r in all_results])

    print(f"Metadata improves test: {n_metadata_helps}/{n_splits} splits")
    print(f"Combined generalizes: {n_generalizes}/{n_splits} splits")
    print(f"Average improvement: {avg_improvement:.3f}")
    print(f"Average generalization ratio: {avg_gen_ratio:.2f}")
    print()

    # Verdict
    if n_metadata_helps >= n_splits / 2 and n_generalizes >= n_splits / 2:
        verdict = "CONFIRMED"
        interpretation = (
            "Pathogen metadata improves prediction AND generalizes to held-out pathogens. "
            "Non-AMP information provides deployable signal."
        )
    elif n_metadata_helps >= n_splits / 2:
        verdict = "LOOKUP_BEHAVIOR"
        interpretation = (
            "Metadata improves training but does NOT generalize to held-out pathogens. "
            "This is LOOKUP behavior, not learnable signal. R2 FAILS."
        )
    elif avg_improvement > 0:
        verdict = "WEAK_SIGNAL"
        interpretation = (
            "Metadata shows weak improvement but inconsistent across splits. "
            "Treat with caution."
        )
    else:
        verdict = "FALSIFIED"
        interpretation = (
            "Pathogen metadata provides NO predictive improvement. "
            "Non-AMP information does not help."
        )

    print(f"VERDICT: {verdict}")
    print(f"INTERPRETATION: {interpretation}")
    print()

    # R3 Classification
    print("-" * 70)
    print("R3 CLASSIFICATION (Inference-Time Availability)")
    print("-" * 70)

    if verdict == "CONFIRMED":
        r3_class = "Deployable (metadata available when target pathogen known)"
    elif verdict == "LOOKUP_BEHAVIOR":
        r3_class = "Research-Only (R2 violation - does not generalize)"
    else:
        r3_class = "Research-Only (no useful signal)"

    print(f"Classification: {r3_class}")

    # Save results
    output_file = RESULTS_DIR / "P1_C5_results.json"
    output = {
        "conjecture": "C5: Pathogen specificity requires NON-AMP information",
        "n_splits": n_splits,
        "n_metadata_helps": n_metadata_helps,
        "n_generalizes": n_generalizes,
        "avg_improvement": avg_improvement,
        "avg_generalization_ratio": avg_gen_ratio,
        "verdict": verdict,
        "interpretation": interpretation,
        "r3_classification": r3_class,
        "split_results": all_results,
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print()
    print(f"Results saved to: {output_file}")

    return verdict


if __name__ == "__main__":
    verdict = main()
    sys.exit(0 if verdict in ["CONFIRMED", "FALSIFIED", "LOOKUP_BEHAVIOR", "WEAK_SIGNAL"] else 1)
