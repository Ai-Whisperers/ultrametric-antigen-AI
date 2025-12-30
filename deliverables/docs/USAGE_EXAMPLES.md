# Usage Examples & Complete Workflows

**Version**: 1.0.0
**Updated**: 2025-12-30
**Authors**: AI Whisperers

---

## Overview

This document provides complete, end-to-end examples for common bioinformatics workflows using the shared utilities. Each example is self-contained and can be run directly.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Peptide Analysis Workflow](#2-peptide-analysis-workflow)
3. [Training an Activity Predictor](#3-training-an-activity-predictor)
4. [Screening Peptide Candidates](#4-screening-peptide-candidates)
5. [Primer Design for Cloning](#5-primer-design-for-cloning)
6. [Protein Stability Analysis](#6-protein-stability-analysis)
7. [Full Drug Development Pipeline](#7-full-drug-development-pipeline)
8. [Logging and Reproducibility](#8-logging-and-reproducibility)

---

## 1. Quick Start

### Basic Peptide Analysis

```python
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    compute_peptide_properties,
    validate_sequence,
    HemolysisPredictor,
)

# Analyze a peptide
peptide = "GIGKFLHSAKKFGKAFVGEIMNS"  # Magainin 2

# Validate sequence first
is_valid, error = validate_sequence(peptide)
if not is_valid:
    print(f"Invalid sequence: {error}")
    exit(1)

# Compute properties
props = compute_peptide_properties(peptide)
print(f"Peptide: {peptide}")
print(f"  Length: {props['length']} amino acids")
print(f"  Net charge: {props['net_charge']:+.1f}")
print(f"  Hydrophobicity: {props['hydrophobicity']:.2f}")
print(f"  Hydrophobic ratio: {props['hydrophobic_ratio']:.1%}")
print(f"  Cationic ratio: {props['cationic_ratio']:.1%}")

# Predict hemolysis
predictor = HemolysisPredictor()
hemo_result = predictor.predict(peptide)
print(f"  HC50: {hemo_result['hc50_predicted']:.1f} uM")
print(f"  Risk: {hemo_result['risk_category']}")
```

**Output:**
```
Peptide: GIGKFLHSAKKFGKAFVGEIMNS
  Length: 23 amino acids
  Net charge: +4.0
  Hydrophobicity: 0.12
  Hydrophobic ratio: 47.8%
  Cationic ratio: 17.4%
  HC50: 105.4 uM
  Risk: Moderate
```

---

## 2. Peptide Analysis Workflow

### Complete Property Analysis

```python
from shared import (
    compute_peptide_properties,
    compute_physicochemical_descriptors,
    compute_amino_acid_composition,
    compute_ml_features,
    validate_sequence,
)
import numpy as np

def analyze_peptide_complete(sequence: str) -> dict:
    """Complete analysis of a peptide sequence."""

    # Validate
    is_valid, error = validate_sequence(sequence)
    if not is_valid:
        return {"error": error}

    # Basic properties
    props = compute_peptide_properties(sequence)

    # Extended properties
    ext_props = compute_physicochemical_descriptors(sequence)

    # Amino acid composition
    aa_comp = compute_amino_acid_composition(sequence)

    # ML features
    features = compute_ml_features(sequence)

    return {
        "sequence": sequence,
        "length": props["length"],
        "net_charge": props["net_charge"],
        "hydrophobicity": props["hydrophobicity"],
        "hydrophobic_ratio": props["hydrophobic_ratio"],
        "cationic_ratio": props["cationic_ratio"],
        "aromaticity": ext_props["aromaticity"],
        "aliphatic_index": ext_props["aliphatic_index"],
        "polar_ratio": ext_props["polar_ratio"],
        "aa_composition": aa_comp,
        "ml_features": features,
    }


# Example usage
peptides = [
    ("Magainin 2", "GIGKFLHSAKKFGKAFVGEIMNS"),
    ("Melittin", "GIGAVLKVLTTGLPALISWIKRKRQQ"),
    ("Indolicidin", "ILPWKWPWWPWRR"),
]

print(f"{'Peptide':<15} {'Length':<8} {'Charge':<8} {'Hydro':<8} {'Aromatic':<10}")
print("-" * 55)

for name, seq in peptides:
    result = analyze_peptide_complete(seq)
    print(f"{name:<15} {result['length']:<8} {result['net_charge']:<+8.1f} "
          f"{result['hydrophobicity']:<8.2f} {result['aromaticity']:<10.1%}")
```

---

## 3. Training an Activity Predictor

### Complete Training Pipeline with Uncertainty

```python
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

# Import from project
from carlos_brizuela.scripts.dramp_activity_loader import DRAMPLoader
from shared import (
    bootstrap_prediction_interval,
    compute_prediction_metrics_with_uncertainty,
    get_logger,
    setup_logging,
)

# Setup logging
setup_logging(level="INFO")
logger = get_logger("training")

# Load curated data
logger.info("Loading curated AMP database...")
loader = DRAMPLoader()
db = loader.generate_curated_database()
logger.info(f"Loaded {len(db.records)} entries")

# Get training data for E. coli
target = "Escherichia coli"
X, y = db.get_training_data(target=target)
logger.info(f"Training samples for {target}: {len(y)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
logger.info("Training Gradient Boosting model...")
model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    min_samples_leaf=3,
    random_state=42,
)
model.fit(X_train_scaled, y_train)

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2")
logger.info(f"CV R2: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

# Test set predictions with uncertainty
logger.info("Computing predictions with uncertainty...")
mean_pred, lower, upper = bootstrap_prediction_interval(
    model, X_train_scaled, y_train, X_test_scaled,
    n_bootstrap=50, confidence=0.90
)

# Compute metrics
metrics = compute_prediction_metrics_with_uncertainty(
    y_test, mean_pred, lower, upper
)

logger.model_metrics("activity_predictor", {
    "rmse": metrics["rmse"],
    "mae": metrics["mae"],
    "r": metrics["pearson_r"],
    "coverage": metrics["coverage"],
    "mean_width": metrics["mean_interval_width"],
})

print("\n=== Model Performance ===")
print(f"RMSE: {metrics['rmse']:.3f}")
print(f"MAE: {metrics['mae']:.3f}")
print(f"Pearson r: {metrics['pearson_r']:.3f}")
print(f"Spearman rho: {metrics['spearman_rho']:.3f}")
print(f"Coverage (90% CI): {metrics['coverage']:.1%}")
print(f"Mean interval width: {metrics['mean_interval_width']:.3f}")
```

---

## 4. Screening Peptide Candidates

### Multi-Criteria Peptide Screening

```python
from shared import (
    compute_peptide_properties,
    HemolysisPredictor,
    validate_sequence,
)
import numpy as np

class PeptideScreener:
    """Screen peptide candidates using multiple criteria."""

    def __init__(self, trained_model=None, scaler=None):
        """Initialize screener.

        Args:
            trained_model: Trained sklearn regressor for activity
            scaler: Fitted StandardScaler for features
        """
        self.model = trained_model
        self.scaler = scaler
        self.hemo_predictor = HemolysisPredictor()

    def screen(self, sequence: str, target_pathogen: str = "E. coli") -> dict:
        """Screen a peptide candidate.

        Returns:
            Dictionary with all screening metrics
        """
        # Validate
        is_valid, error = validate_sequence(sequence)
        if not is_valid:
            return {"valid": False, "error": error}

        # Properties
        props = compute_peptide_properties(sequence)

        # Hemolysis prediction
        hemo = self.hemo_predictor.predict(sequence)

        # Activity prediction (if model available)
        if self.model and self.scaler:
            from shared import compute_ml_features
            features = compute_ml_features(sequence).reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            predicted_activity = self.model.predict(features_scaled)[0]
        else:
            predicted_activity = None

        # Compute therapeutic index if activity available
        if predicted_activity is not None:
            # Convert log2(MIC) to uM (assuming MIC in ug/mL, MW ~2500)
            mic_ug_ml = 2 ** predicted_activity
            mic_um = mic_ug_ml / 2.5  # Approximate conversion
            ti = hemo["hc50_predicted"] / max(mic_um, 0.1)
        else:
            ti = None

        # Compute score
        score = self._compute_score(props, hemo, predicted_activity, ti)

        return {
            "valid": True,
            "sequence": sequence,
            "length": props["length"],
            "charge": props["net_charge"],
            "hydrophobicity": props["hydrophobicity"],
            "hc50": hemo["hc50_predicted"],
            "hemolytic_risk": hemo["risk_category"],
            "predicted_activity": predicted_activity,
            "therapeutic_index": ti,
            "overall_score": score,
            "recommendation": self._get_recommendation(score),
        }

    def _compute_score(self, props, hemo, activity, ti) -> float:
        """Compute overall score (0-100, higher = better)."""
        score = 50.0  # Base score

        # Length penalty (ideal 15-25)
        if 15 <= props["length"] <= 25:
            score += 10
        elif props["length"] < 10 or props["length"] > 35:
            score -= 10

        # Charge bonus (ideal +2 to +6)
        if 2 <= props["net_charge"] <= 6:
            score += 10
        elif props["net_charge"] < 0:
            score -= 15

        # Hemolysis (lower = better)
        if hemo["hc50_predicted"] > 200:
            score += 15
        elif hemo["hc50_predicted"] < 50:
            score -= 20

        # Therapeutic index (if available)
        if ti is not None:
            if ti > 10:
                score += 15
            elif ti > 5:
                score += 10
            elif ti < 2:
                score -= 10

        return max(0, min(100, score))

    def _get_recommendation(self, score: float) -> str:
        if score >= 80:
            return "Excellent candidate - Proceed to synthesis"
        elif score >= 60:
            return "Good candidate - Consider with modifications"
        elif score >= 40:
            return "Moderate - Significant optimization needed"
        else:
            return "Poor candidate - Not recommended"

    def screen_batch(self, sequences: list) -> list:
        """Screen multiple peptides."""
        return [self.screen(seq) for seq in sequences]


# Example usage
screener = PeptideScreener()

candidates = [
    "GIGKFLHSAKKFGKAFVGEIMNS",      # Magainin 2
    "GIGAVLKVLTTGLPALISWIKRKRQQ",   # Melittin
    "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",  # Cecropin A
    "ILPWKWPWWPWRR",                 # Indolicidin
]

print(f"{'Peptide':<25} {'HC50':<10} {'Risk':<10} {'Score':<8} {'Recommendation':<30}")
print("-" * 85)

for seq in candidates:
    result = screener.screen(seq)
    name = seq[:22] + "..." if len(seq) > 25 else seq
    print(f"{name:<25} {result['hc50']:<10.1f} {result['hemolytic_risk']:<10} "
          f"{result['overall_score']:<8.1f} {result['recommendation']:<30}")
```

---

## 5. Primer Design for Cloning

### Design Primers for Expression Construct

```python
from shared import PrimerDesigner, calculate_gc, calculate_tm

def design_expression_primers(
    peptide: str,
    name: str,
    vector_system: str = "pet28a",
    organism: str = "ecoli",
) -> dict:
    """Design primers for expression in a vector system.

    Args:
        peptide: Amino acid sequence
        name: Peptide name
        vector_system: Expression vector (pet28a, pgex, etc.)
        organism: Codon optimization target

    Returns:
        Dictionary with primer information
    """
    designer = PrimerDesigner()

    # Vector-specific restriction sites and overhangs
    vector_sites = {
        "pet28a": {
            "forward_site": "CATATG",  # NdeI (includes ATG)
            "reverse_site": "CTCGAG",  # XhoI
            "overhang": "AAAA",
        },
        "pgex": {
            "forward_site": "GAATTC",  # EcoRI
            "reverse_site": "CTCGAG",  # XhoI
            "overhang": "AAAA",
        },
    }

    sites = vector_sites.get(vector_system, vector_sites["pet28a"])

    # Get codon-optimized DNA
    dna = designer.peptide_to_dna(peptide, codon_optimization=organism)

    # Design basic primers
    basic_primers = designer.design_for_peptide(
        peptide,
        codon_optimization=organism,
        add_start_codon=True,
        add_stop_codon=True,
    )

    # Add restriction sites
    forward_with_site = sites["overhang"] + sites["forward_site"] + basic_primers.forward[3:]  # Skip ATG
    reverse_with_site = sites["overhang"] + sites["reverse_site"] + basic_primers.reverse

    # Full construct sequence
    full_construct = f"ATG{dna}TAA"

    return {
        "name": name,
        "peptide_sequence": peptide,
        "peptide_length": len(peptide),
        "dna_sequence": full_construct,
        "dna_length": len(full_construct),
        "vector": vector_system,
        "codon_optimization": organism,
        "forward_primer": forward_with_site,
        "forward_tm": calculate_tm(forward_with_site),
        "forward_gc": calculate_gc(forward_with_site),
        "reverse_primer": reverse_with_site,
        "reverse_tm": calculate_tm(reverse_with_site),
        "reverse_gc": calculate_gc(reverse_with_site),
        "restriction_sites": {
            "5_prime": sites["forward_site"],
            "3_prime": sites["reverse_site"],
        },
    }


# Example: Design primers for multiple peptides
peptides = [
    ("Magainin_2", "GIGKFLHSAKKFGKAFVGEIMNS"),
    ("Melittin", "GIGAVLKVLTTGLPALISWIKRKRQQ"),
]

for name, seq in peptides:
    result = design_expression_primers(seq, name, vector_system="pet28a")

    print(f"\n=== {result['name']} Expression Primers ===")
    print(f"Vector: pET28a")
    print(f"Peptide: {result['peptide_sequence']} ({result['peptide_length']} aa)")
    print(f"Gene: {result['dna_length']} bp")
    print()
    print(f"Forward (NdeI): 5'-{result['forward_primer']}-3'")
    print(f"  Tm: {result['forward_tm']:.1f}C, GC: {result['forward_gc']:.1f}%")
    print()
    print(f"Reverse (XhoI): 5'-{result['reverse_primer']}-3'")
    print(f"  Tm: {result['reverse_tm']:.1f}C, GC: {result['reverse_gc']:.1f}%")
```

---

## 6. Protein Stability Analysis

### Analyze Mutation Effects on Stability

```python
from jose_colbes.scripts.protherm_ddg_loader import ProThermLoader
from shared import get_logger
import numpy as np

# Setup
logger = get_logger("stability")

# Load curated DDG data
loader = ProThermLoader()
db = loader.generate_curated_database()
logger.info(f"Loaded {len(db.records)} mutations")

# Analyze by protein
protein_stats = {}
for record in db.records:
    pdb = record["pdb_id"]
    if pdb not in protein_stats:
        protein_stats[pdb] = {"ddg_values": [], "mutations": []}
    protein_stats[pdb]["ddg_values"].append(record["ddg"])
    protein_stats[pdb]["mutations"].append(
        f"{record['wild_type']}{record['position']}{record['mutant']}"
    )

print(f"{'Protein':<15} {'Mutations':<10} {'Mean DDG':<10} {'Std DDG':<10} {'Range':<15}")
print("-" * 65)

for pdb in sorted(protein_stats.keys()):
    ddg = np.array(protein_stats[pdb]["ddg_values"])
    print(f"{pdb:<15} {len(ddg):<10} {ddg.mean():<10.2f} {ddg.std():<10.2f} "
          f"[{ddg.min():.1f}, {ddg.max():.1f}]")

# Analyze by secondary structure
ss_stats = {"H": [], "E": [], "C": []}
for record in db.records:
    ss = record.get("secondary_structure", "C")
    ss_stats[ss].append(record["ddg"])

print("\n=== DDG by Secondary Structure ===")
print(f"{'Structure':<15} {'Count':<10} {'Mean DDG':<10}")
for ss, values in ss_stats.items():
    name = {"H": "Helix", "E": "Sheet", "C": "Coil"}[ss]
    print(f"{name:<15} {len(values):<10} {np.mean(values):.2f}")
```

---

## 7. Full Drug Development Pipeline

### End-to-End Peptide Discovery Workflow

```python
"""
Complete peptide drug development pipeline.

Steps:
1. Generate/import candidate sequences
2. Compute biophysical properties
3. Predict activity
4. Predict toxicity (hemolysis)
5. Compute therapeutic index
6. Rank candidates
7. Design primers for synthesis
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    compute_peptide_properties,
    compute_ml_features,
    validate_sequence,
    HemolysisPredictor,
    PrimerDesigner,
    UncertaintyPredictor,
    get_logger,
    setup_logging,
)

# Setup
setup_logging(level="INFO", log_file="pipeline.log")
logger = get_logger("pipeline")


class PeptideDevelopmentPipeline:
    """End-to-end peptide drug development pipeline."""

    def __init__(self, activity_model=None, scaler=None):
        """Initialize pipeline.

        Args:
            activity_model: Trained activity predictor
            scaler: Feature scaler for activity model
        """
        self.activity_model = activity_model
        self.scaler = scaler
        self.hemo_predictor = HemolysisPredictor()
        self.primer_designer = PrimerDesigner()
        logger.info("Pipeline initialized")

    def run(self, candidates: list, target_pathogen: str = "E. coli") -> list:
        """Run full pipeline on candidates.

        Args:
            candidates: List of (name, sequence) tuples
            target_pathogen: Target pathogen for activity

        Returns:
            Ranked list of results
        """
        results = []

        for name, sequence in candidates:
            logger.info(f"Processing: {name}")

            # Step 1: Validate
            is_valid, error = validate_sequence(sequence)
            if not is_valid:
                logger.warning(f"Invalid sequence: {name} - {error}")
                continue

            # Step 2: Properties
            props = compute_peptide_properties(sequence)

            # Step 3: Hemolysis
            hemo = self.hemo_predictor.predict(sequence)

            # Step 4: Activity (if model available)
            if self.activity_model and self.scaler:
                features = compute_ml_features(sequence).reshape(1, -1)
                features_scaled = self.scaler.transform(features)
                activity = self.activity_model.predict(features_scaled)[0]
            else:
                activity = None

            # Step 5: Therapeutic index
            if activity is not None:
                mic_um = (2 ** activity) / 2.5  # Approximate conversion
                ti = hemo["hc50_predicted"] / max(mic_um, 0.1)
            else:
                ti = None

            # Step 6: Primers
            primers = self.primer_designer.design_for_peptide(
                sequence, codon_optimization="ecoli"
            )

            # Step 7: Compute score
            score = self._compute_development_score(props, hemo, ti)

            results.append({
                "name": name,
                "sequence": sequence,
                "length": props["length"],
                "charge": props["net_charge"],
                "hydrophobicity": props["hydrophobicity"],
                "hc50": hemo["hc50_predicted"],
                "hemolytic_risk": hemo["risk_category"],
                "predicted_activity": activity,
                "therapeutic_index": ti,
                "forward_primer": primers.forward,
                "reverse_primer": primers.reverse,
                "development_score": score,
            })

        # Rank by score
        results.sort(key=lambda x: x["development_score"], reverse=True)
        logger.info(f"Pipeline complete: {len(results)} candidates processed")

        return results

    def _compute_development_score(self, props, hemo, ti) -> float:
        """Compute development priority score (0-100)."""
        score = 0.0

        # Length (ideal 15-30)
        if 15 <= props["length"] <= 30:
            score += 20
        elif 10 <= props["length"] <= 35:
            score += 10

        # Charge (ideal +2 to +6)
        if 2 <= props["net_charge"] <= 6:
            score += 20
        elif 1 <= props["net_charge"] <= 8:
            score += 10

        # Hemolysis (lower risk = better)
        if hemo["risk_category"] == "Low":
            score += 30
        elif hemo["risk_category"] == "Moderate":
            score += 15

        # Therapeutic index
        if ti is not None:
            if ti > 10:
                score += 30
            elif ti > 5:
                score += 20
            elif ti > 2:
                score += 10

        return min(100, score)

    def generate_report(self, results: list) -> str:
        """Generate development report."""
        report = []
        report.append("=" * 60)
        report.append("PEPTIDE DEVELOPMENT PIPELINE REPORT")
        report.append("=" * 60)
        report.append("")

        for i, r in enumerate(results[:10], 1):  # Top 10
            report.append(f"Rank {i}: {r['name']}")
            report.append(f"  Sequence: {r['sequence']}")
            report.append(f"  Length: {r['length']} aa")
            report.append(f"  Charge: {r['charge']:+.1f}")
            report.append(f"  HC50: {r['hc50']:.1f} uM ({r['hemolytic_risk']} risk)")
            if r['therapeutic_index']:
                report.append(f"  TI: {r['therapeutic_index']:.1f}")
            report.append(f"  Score: {r['development_score']:.0f}/100")
            report.append("")

        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # Define candidates
    candidates = [
        ("Magainin_2", "GIGKFLHSAKKFGKAFVGEIMNS"),
        ("Melittin", "GIGAVLKVLTTGLPALISWIKRKRQQ"),
        ("Cecropin_A", "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK"),
        ("Indolicidin", "ILPWKWPWWPWRR"),
        ("Pexiganan", "GIGKFLKKAKKFGKAFVKILKK"),
        ("Buforin_II", "TRSSRAGLQFPVGRVHRLLRK"),
        ("LL-37", "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES"),
    ]

    # Run pipeline
    pipeline = PeptideDevelopmentPipeline()
    results = pipeline.run(candidates)

    # Generate report
    report = pipeline.generate_report(results)
    print(report)

    # Save report
    with open("development_report.txt", "w") as f:
        f.write(report)
    logger.info("Report saved to development_report.txt")
```

---

## 8. Logging and Reproducibility

### Setting Up Comprehensive Logging

```python
from shared import (
    get_logger,
    setup_logging,
    LogContext,
    log_function_call,
)
import time

# Setup logging with file output
setup_logging(
    level="DEBUG",
    log_file="analysis.log",
    use_colors=True,
)

logger = get_logger("analysis")

# Basic logging
logger.info("Starting analysis pipeline")
logger.debug("Debug information here")
logger.warning("This is a warning")

# Log predictions with confidence
logger.prediction(
    "MIC",
    value=4.5,
    confidence=0.92,
    peptide="Magainin 2"
)

# Log model metrics
logger.model_metrics("activity_predictor", {
    "rmse": 0.35,
    "r": 0.85,
    "coverage": 0.90,
})

# Use context for grouped logging
with LogContext(logger, "Data Loading"):
    logger.info("Loading AMP database...")
    time.sleep(0.1)
    logger.info("Loaded 224 entries")

# Log function calls automatically
@log_function_call
def process_peptide(sequence: str) -> dict:
    """Process a peptide sequence."""
    from shared import compute_peptide_properties
    return compute_peptide_properties(sequence)

result = process_peptide("GIGKFLHSAKKFGKAFVGEIMNS")
logger.analysis(f"Processed peptide with charge {result['net_charge']}")
```

### Reproducible Analysis with Logging

```python
from shared import get_logger, setup_logging
import json
from datetime import datetime
import hashlib

class ReproducibleAnalysis:
    """Analysis framework with full reproducibility tracking."""

    def __init__(self, name: str, log_file: str = None):
        self.name = name
        self.start_time = datetime.now()
        self.log_file = log_file or f"{name}_{self.start_time:%Y%m%d_%H%M%S}.log"

        setup_logging(level="INFO", log_file=self.log_file)
        self.logger = get_logger(name)

        self.metadata = {
            "analysis_name": name,
            "start_time": self.start_time.isoformat(),
            "parameters": {},
            "inputs": [],
            "outputs": [],
        }

        self.logger.info(f"Analysis initialized: {name}")

    def set_parameter(self, name: str, value):
        """Record a parameter."""
        self.metadata["parameters"][name] = value
        self.logger.info(f"Parameter set: {name} = {value}")

    def add_input(self, name: str, data):
        """Record input data."""
        if isinstance(data, str):
            checksum = hashlib.md5(data.encode()).hexdigest()[:8]
        else:
            checksum = "complex"
        self.metadata["inputs"].append({"name": name, "checksum": checksum})
        self.logger.info(f"Input added: {name} (checksum: {checksum})")

    def add_output(self, name: str, value):
        """Record output."""
        self.metadata["outputs"].append({"name": name, "value": str(value)})
        self.logger.info(f"Output: {name} = {value}")

    def complete(self):
        """Finalize analysis and save metadata."""
        end_time = datetime.now()
        self.metadata["end_time"] = end_time.isoformat()
        self.metadata["duration_seconds"] = (end_time - self.start_time).total_seconds()

        # Save metadata
        metadata_file = self.log_file.replace(".log", "_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

        self.logger.info(f"Analysis complete. Duration: {self.metadata['duration_seconds']:.1f}s")
        self.logger.info(f"Metadata saved to: {metadata_file}")


# Example usage
analysis = ReproducibleAnalysis("peptide_screening")
analysis.set_parameter("target_pathogen", "E. coli")
analysis.set_parameter("confidence_level", 0.90)
analysis.add_input("peptide_sequence", "GIGKFLHSAKKFGKAFVGEIMNS")
analysis.add_output("predicted_hc50", 105.4)
analysis.add_output("risk_category", "Moderate")
analysis.complete()
```

---

## Summary

This documentation provides complete examples for:

| Workflow | Key Modules Used |
|----------|------------------|
| Peptide Analysis | `compute_peptide_properties`, `compute_ml_features` |
| Activity Prediction | `DRAMPLoader`, `bootstrap_prediction_interval` |
| Candidate Screening | `HemolysisPredictor`, `compute_peptide_properties` |
| Primer Design | `PrimerDesigner`, `peptide_to_dna` |
| Stability Analysis | `ProThermLoader` |
| Full Pipeline | All modules integrated |
| Logging | `get_logger`, `setup_logging`, `LogContext` |

For detailed API documentation, see:
- [Shared Module README](../shared/README.md)
- [Curated Databases](./CURATED_DATABASES.md)
- [Uncertainty Quantification](./UNCERTAINTY_QUANTIFICATION.md)
- [Hemolysis Predictor](./HEMOLYSIS_PREDICTOR.md)
- [Primer Design](./PRIMER_DESIGN.md)
