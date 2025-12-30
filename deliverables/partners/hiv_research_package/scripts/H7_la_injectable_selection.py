# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""H7: Long-Acting Injectable Selection Tool

Research Idea Implementation - HIV Research Package

Predict which patients will maintain viral suppression on long-acting
injectables (CAB-LA/RPV-LA) versus those at risk of failure.

FDA-approved LA regimens:
- Cabenuva: Cabotegravir (CAB-LA) + Rilpivirine (RPV-LA)
- Apretude: Cabotegravir for PrEP

Risk Factors for LA Failure:
1. Baseline resistance (especially RPV mutations)
2. BMI (affects PK)
3. Adherence history
4. Prior treatment experience

Usage:
    python scripts/H7_la_injectable_selection.py \
        --sequence patient_sequence.fasta \
        --clinical patient_data.json \
        --output results/la_selection/

    # Use Stanford HIVdb for resistance analysis (recommended):
    python scripts/H7_la_injectable_selection.py --use-stanford --demo
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# Global flag for Stanford HIVdb client
USE_STANFORD_CLIENT = False
_STANFORD_CLIENT = None


def get_stanford_client():
    """Get or create Stanford HIVdb client instance."""
    global _STANFORD_CLIENT

    if _STANFORD_CLIENT is not None:
        return _STANFORD_CLIENT

    try:
        from stanford_hivdb_client import StanfordHIVdbClient
        _STANFORD_CLIENT = StanfordHIVdbClient()
        print("Stanford HIVdb client initialized")
        return _STANFORD_CLIENT
    except ImportError:
        print("Warning: stanford_hivdb_client not found, using local analysis")
        return None
    except Exception as e:
        print(f"Warning: Could not initialize Stanford client: {e}")
        return None


# LA Injectable drugs and resistance mutations
LA_DRUGS = {
    "CAB": {
        "name": "Cabotegravir",
        "class": "INSTI",
        "half_life_days": 25,
        "administration": "IM gluteal",
        "mutations": {
            "Q148H": {"fold_change": 5.0, "level": "high"},
            "Q148R": {"fold_change": 3.0, "level": "moderate"},
            "N155H": {"fold_change": 2.5, "level": "moderate"},
            "G140S": {"fold_change": 2.0, "level": "low"},
            "E138K": {"fold_change": 1.5, "level": "low"},
            "S147G": {"fold_change": 1.8, "level": "low"},
        },
    },
    "RPV": {
        "name": "Rilpivirine",
        "class": "NNRTI",
        "half_life_days": 45,
        "administration": "IM gluteal",
        "mutations": {
            "E138K": {"fold_change": 3.0, "level": "high"},
            "E138A": {"fold_change": 2.5, "level": "moderate"},
            "K101E": {"fold_change": 2.0, "level": "moderate"},
            "K101P": {"fold_change": 4.0, "level": "high"},
            "Y181C": {"fold_change": 5.0, "level": "high"},
            "Y181I": {"fold_change": 4.0, "level": "high"},
            "H221Y": {"fold_change": 2.5, "level": "moderate"},
            "F227C": {"fold_change": 3.0, "level": "high"},
        },
    },
}

# BMI impact on PK
BMI_CATEGORIES = {
    "underweight": {"range": (0, 18.5), "pk_adjustment": 1.15},
    "normal": {"range": (18.5, 25), "pk_adjustment": 1.0},
    "overweight": {"range": (25, 30), "pk_adjustment": 0.95},
    "obese_1": {"range": (30, 35), "pk_adjustment": 0.85},
    "obese_2": {"range": (35, 40), "pk_adjustment": 0.75},
    "obese_3": {"range": (40, 100), "pk_adjustment": 0.60},  # Higher risk
}


@dataclass
class PatientData:
    """Patient clinical data for LA selection."""

    patient_id: str
    age: int
    sex: str  # M/F
    bmi: float
    viral_load: float  # copies/mL (should be <50 for switch)
    cd4_count: int
    prior_regimens: list[str]
    adherence_history: str  # "excellent", "good", "moderate", "poor"
    injection_site_concerns: bool
    psychiatric_history: bool


@dataclass
class LASelectionResult:
    """Result of LA injectable eligibility assessment."""

    patient_id: str
    eligible: bool
    success_probability: float
    cab_resistance_risk: float
    rpv_resistance_risk: float
    pk_adequacy_score: float
    adherence_score: float
    detected_mutations: list[dict]
    recommendation: str
    risk_factors: list[str]
    monitoring_plan: list[str]


def detect_la_mutations_stanford(sequence: str) -> list[dict]:
    """Detect LA-relevant mutations using Stanford HIVdb."""
    client = get_stanford_client()
    if client is None:
        return detect_la_mutations_local(sequence)

    try:
        report = client.analyze_sequence(sequence, "LA_PATIENT")

        detected = []
        for mut in report.mutations:
            # Check if mutation is relevant to CAB or RPV
            for drug, info in LA_DRUGS.items():
                if mut.notation in info["mutations"]:
                    details = info["mutations"][mut.notation]
                    detected.append({
                        "mutation": mut.notation,
                        "drug": drug,
                        "fold_change": details["fold_change"],
                        "level": details["level"],
                    })

        return detected

    except Exception as e:
        print(f"Stanford analysis failed: {e}, falling back to local")
        return detect_la_mutations_local(sequence)


def detect_la_mutations_local(sequence: str) -> list[dict]:
    """Detect mutations relevant to LA injectables (local/demo)."""
    detected = []

    # Simulate mutation detection (demo)
    np.random.seed(hash(sequence[:20]) % 2**32)

    for drug, info in LA_DRUGS.items():
        for mut, details in info["mutations"].items():
            # Low probability of detection (these are rare)
            if np.random.random() < 0.05:
                detected.append({
                    "mutation": mut,
                    "drug": drug,
                    "fold_change": details["fold_change"],
                    "level": details["level"],
                })

    return detected


def detect_la_mutations(sequence: str) -> list[dict]:
    """Detect mutations relevant to LA injectables.

    Uses Stanford HIVdb when USE_STANFORD_CLIENT is True.
    """
    if USE_STANFORD_CLIENT:
        return detect_la_mutations_stanford(sequence)
    return detect_la_mutations_local(sequence)


def compute_resistance_risk(
    mutations: list[dict],
    drug: str,
) -> float:
    """Compute resistance risk for a specific LA drug."""
    relevant = [m for m in mutations if m["drug"] == drug]

    if not relevant:
        return 0.0

    # Accumulate risk
    risk = 0.0
    for mut in relevant:
        if mut["level"] == "high":
            risk += 0.4
        elif mut["level"] == "moderate":
            risk += 0.2
        else:
            risk += 0.1

    return min(1.0, risk)


def compute_pk_adequacy(patient: PatientData) -> float:
    """Compute PK adequacy based on BMI and other factors."""
    # BMI adjustment
    pk_score = 1.0
    for category, info in BMI_CATEGORIES.items():
        if info["range"][0] <= patient.bmi < info["range"][1]:
            pk_score = info["pk_adjustment"]
            break

    # Age adjustment (elderly may have altered PK)
    if patient.age > 65:
        pk_score *= 0.95
    elif patient.age > 75:
        pk_score *= 0.90

    # Sex adjustment (minor)
    if patient.sex == "F":
        pk_score *= 1.05  # Slightly higher levels in women

    return min(1.0, pk_score)


def compute_adherence_score(patient: PatientData) -> float:
    """Compute predicted injection adherence."""
    base_scores = {
        "excellent": 0.95,
        "good": 0.85,
        "moderate": 0.70,
        "poor": 0.50,
    }

    score = base_scores.get(patient.adherence_history, 0.70)

    # Injection site concerns reduce score
    if patient.injection_site_concerns:
        score *= 0.85

    # Monthly vs q2month (we assume monthly for now)
    # Monthly has better adherence than q2month

    return score


def assess_eligibility(
    patient: PatientData,
    sequence: Optional[str] = None,
) -> LASelectionResult:
    """Assess patient eligibility for LA injectables."""
    # Detect mutations
    if sequence:
        mutations = detect_la_mutations(sequence)
    else:
        mutations = []

    # Compute component risks
    cab_risk = compute_resistance_risk(mutations, "CAB")
    rpv_risk = compute_resistance_risk(mutations, "RPV")
    pk_score = compute_pk_adequacy(patient)
    adherence_score = compute_adherence_score(patient)

    # Risk factors
    risk_factors = []

    # Viral suppression check
    if patient.viral_load >= 50:
        risk_factors.append("Not virally suppressed (VL >= 50)")

    # BMI check
    if patient.bmi >= 35:
        risk_factors.append(f"High BMI ({patient.bmi:.1f}) may affect drug levels")

    # Resistance check
    if cab_risk > 0.2:
        risk_factors.append("CAB resistance mutations detected")
    if rpv_risk > 0.2:
        risk_factors.append("RPV resistance mutations detected")

    # Psychiatric history (RPV caution)
    if patient.psychiatric_history:
        risk_factors.append("Psychiatric history (monitor for mood changes)")

    # Prior NNRTI failure
    if any("NNRTI" in reg.upper() or "EFV" in reg.upper() or "NVP" in reg.upper()
           for reg in patient.prior_regimens):
        risk_factors.append("Prior NNRTI exposure (check for archived resistance)")

    # Compute success probability
    # Base probability for well-selected patients is ~95%
    success_prob = 0.95

    # Resistance penalties
    success_prob -= cab_risk * 0.3
    success_prob -= rpv_risk * 0.4  # RPV resistance is more impactful

    # PK penalty
    if pk_score < 0.8:
        success_prob -= (0.8 - pk_score) * 0.2

    # Adherence bonus/penalty
    if adherence_score < 0.8:
        success_prob -= (0.8 - adherence_score) * 0.3
    elif adherence_score > 0.9:
        success_prob += 0.02

    # Viral load requirement
    if patient.viral_load >= 50:
        success_prob -= 0.15

    success_prob = max(0.1, min(0.99, success_prob))

    # Eligibility decision
    eligible = (
        patient.viral_load < 50 and
        cab_risk < 0.3 and
        rpv_risk < 0.4 and
        pk_score >= 0.6 and
        len(risk_factors) <= 2
    )

    # Recommendation
    if eligible and success_prob >= 0.80:
        recommendation = "ELIGIBLE - Recommend LA injectable switch"
    elif eligible and success_prob >= 0.60:
        recommendation = "ELIGIBLE WITH CAUTION - Close monitoring required"
    elif success_prob >= 0.50:
        recommendation = "CONSIDER ALTERNATIVES - Risk factors present"
    else:
        recommendation = "NOT RECOMMENDED - Continue oral therapy"

    # Monitoring plan
    monitoring = []
    if eligible:
        monitoring.append("HIV RNA at 1, 3, and 6 months post-switch")
        monitoring.append("CD4 count at 6 months")
        if patient.bmi >= 30:
            monitoring.append("Consider drug level monitoring (Ctrough)")
        if patient.psychiatric_history:
            monitoring.append("Psychiatric assessment at each visit")
        if mutations:
            monitoring.append("Resistance testing if virologic failure")

    return LASelectionResult(
        patient_id=patient.patient_id,
        eligible=eligible,
        success_probability=success_prob,
        cab_resistance_risk=cab_risk,
        rpv_resistance_risk=rpv_risk,
        pk_adequacy_score=pk_score,
        adherence_score=adherence_score,
        detected_mutations=mutations,
        recommendation=recommendation,
        risk_factors=risk_factors,
        monitoring_plan=monitoring,
    )


def generate_demo_patients(n: int = 5) -> list[PatientData]:
    """Generate demo patient data."""
    np.random.seed(42)

    patients = []
    for i in range(n):
        patient = PatientData(
            patient_id=f"LA_PATIENT_{i+1:03d}",
            age=np.random.randint(25, 65),
            sex=np.random.choice(["M", "F"]),
            bmi=np.random.normal(27, 5),
            viral_load=np.random.choice([0, 0, 0, 0, 20, 50, 100, 500]),  # Most suppressed
            cd4_count=np.random.randint(400, 900),
            prior_regimens=[["TDF/FTC/DTG"], ["TDF/3TC/EFV"], ["ABC/3TC/DTG"], ["TDF/FTC/EFV", "ABC/3TC/DTG"]][np.random.randint(0, 4)],
            adherence_history=np.random.choice(
                ["excellent", "excellent", "good", "good", "moderate", "poor"],
            ),
            injection_site_concerns=np.random.choice([False, False, False, True]),
            psychiatric_history=np.random.choice([False, False, False, False, True]),
        )
        patients.append(patient)

    return patients


def generate_demo_sequence() -> str:
    """Generate demo sequence."""
    np.random.seed(42)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(np.random.choice(list(aa), size=500))


def export_results(results: list[LASelectionResult], output_dir: Path) -> None:
    """Export LA selection results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    eligible_count = sum(1 for r in results if r.eligible)

    output = {
        "summary": {
            "total_patients": len(results),
            "eligible": eligible_count,
            "eligible_rate": f"{eligible_count/len(results)*100:.1f}%",
            "mean_success_probability": f"{np.mean([r.success_probability for r in results])*100:.1f}%",
        },
        "patients": [
            {
                "patient_id": r.patient_id,
                "eligible": bool(r.eligible),
                "success_probability": f"{r.success_probability*100:.1f}%",
                "recommendation": r.recommendation,
                "cab_resistance_risk": f"{r.cab_resistance_risk*100:.1f}%",
                "rpv_resistance_risk": f"{r.rpv_resistance_risk*100:.1f}%",
                "pk_adequacy": f"{r.pk_adequacy_score*100:.1f}%",
                "adherence_score": f"{r.adherence_score*100:.1f}%",
                "risk_factors": r.risk_factors,
                "monitoring_plan": r.monitoring_plan,
            }
            for r in results
        ],
    }

    json_path = output_dir / "la_selection_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Exported results to {json_path}")

    # Print clinical summary
    print("\n" + "=" * 70)
    print("LONG-ACTING INJECTABLE SELECTION REPORT")
    print("=" * 70)
    print(f"\nPatients assessed: {len(results)}")
    print(f"Eligible for LA: {eligible_count} ({eligible_count/len(results)*100:.1f}%)")

    for r in results:
        print(f"\n--- {r.patient_id} ---")
        print(f"Eligible: {'YES' if r.eligible else 'NO'}")
        print(f"Success Probability: {r.success_probability*100:.1f}%")
        print(f"Recommendation: {r.recommendation}")

        if r.risk_factors:
            print("Risk Factors:")
            for rf in r.risk_factors:
                print(f"  - {rf}")

        if r.monitoring_plan:
            print("Monitoring Plan:")
            for mp in r.monitoring_plan:
                print(f"  - {mp}")


def main():
    """Main entry point."""
    global USE_STANFORD_CLIENT

    parser = argparse.ArgumentParser(description="LA Injectable Selection Tool")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with demo patients",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/la_selection",
        help="Output directory",
    )
    parser.add_argument(
        "--use-stanford",
        action="store_true",
        help="Use Stanford HIVdb API for resistance analysis (recommended)",
    )

    args = parser.parse_args()

    # Set global flag for Stanford client
    if args.use_stanford:
        USE_STANFORD_CLIENT = True
        print("Using Stanford HIVdb for resistance analysis")
        get_stanford_client()

    print("Long-Acting Injectable Selection Tool")
    print("=" * 40)

    # Generate demo patients
    patients = generate_demo_patients(5)
    sequence = generate_demo_sequence()

    results = []
    for patient in patients:
        result = assess_eligibility(patient, sequence)
        results.append(result)

    export_results(results, Path(args.output))

    print("\nLA Selection Assessment Complete!")


if __name__ == "__main__":
    main()
