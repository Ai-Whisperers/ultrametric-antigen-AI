# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Long-Acting Injectable Selection Tool.

This module predicts which patients will maintain viral suppression on
long-acting injectables (CAB-LA/RPV-LA) versus those at risk of failure.

FDA-approved LA regimens:
- Cabenuva: Cabotegravir (CAB-LA) + Rilpivirine (RPV-LA)
- Apretude: Cabotegravir for PrEP

Example:
    >>> selector = LASelector()
    >>> result = selector.assess_eligibility(patient, sequence)
    >>> print(f"Eligible: {result.eligible}")
    >>> print(f"Success probability: {result.success_probability:.1%}")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np

from .models import PatientData, LASelectionResult
from .constants import LA_DRUGS, BMI_CATEGORIES

if TYPE_CHECKING:
    from .stanford_client import StanfordHIVdbClient


class LASelector:
    """Long-acting injectable eligibility selector.

    Assesses patient eligibility for switch to long-acting injectable
    antiretroviral therapy (Cabenuva/Apretude).

    Attributes:
        use_stanford: Whether to use Stanford HIVdb API
        stanford_client: Stanford HIVdb client instance

    Example:
        >>> selector = LASelector(use_stanford=True)
        >>> patient = PatientData(
        ...     patient_id="P001",
        ...     age=35, sex="M", bmi=24.5,
        ...     viral_load=0, cd4_count=650,
        ...     prior_regimens=["TDF/FTC/DTG"],
        ...     adherence_history="excellent",
        ... )
        >>> result = selector.assess_eligibility(patient, sequence)
    """

    def __init__(
        self,
        use_stanford: bool = False,
        stanford_client: Optional[StanfordHIVdbClient] = None,
    ):
        """Initialize LA selector.

        Args:
            use_stanford: Whether to use Stanford HIVdb API
            stanford_client: Pre-configured Stanford client instance
        """
        self.use_stanford = use_stanford
        self._stanford_client = stanford_client

    @property
    def stanford_client(self) -> Optional[StanfordHIVdbClient]:
        """Get or create Stanford HIVdb client."""
        if self._stanford_client is not None:
            return self._stanford_client

        if not self.use_stanford:
            return None

        try:
            from .stanford_client import StanfordHIVdbClient

            self._stanford_client = StanfordHIVdbClient()
            return self._stanford_client
        except Exception as e:
            print(f"Warning: Could not initialize Stanford client: {e}")
            return None

    def detect_la_mutations(self, sequence: str) -> list[dict]:
        """Detect mutations relevant to LA injectables.

        Args:
            sequence: HIV sequence (pol gene)

        Returns:
            List of detected mutation dictionaries
        """
        if self.use_stanford and self.stanford_client:
            return self._detect_mutations_stanford(sequence)
        return self._detect_mutations_local(sequence)

    def _detect_mutations_stanford(self, sequence: str) -> list[dict]:
        """Detect LA mutations using Stanford HIVdb."""
        client = self.stanford_client
        if client is None:
            return self._detect_mutations_local(sequence)

        try:
            report = client.analyze_sequence(sequence, "LA_PATIENT")
            if report is None:
                return self._detect_mutations_local(sequence)

            detected = []
            for mut in report.mutations:
                # Check if mutation is relevant to CAB or RPV
                for drug, info in LA_DRUGS.items():
                    if mut.notation in info["mutations"]:
                        details = info["mutations"][mut.notation]
                        detected.append(
                            {
                                "mutation": mut.notation,
                                "drug": drug,
                                "fold_change": details["fold_change"],
                                "level": details["level"],
                            }
                        )

            return detected

        except Exception as e:
            print(f"Stanford analysis failed: {e}, falling back to local")
            return self._detect_mutations_local(sequence)

    def _detect_mutations_local(self, sequence: str) -> list[dict]:
        """Detect LA mutations using local database (demo)."""
        detected = []

        # Seed based on sequence for reproducible results
        np.random.seed(hash(sequence[:20]) % 2**32)

        for drug, info in LA_DRUGS.items():
            for mut, details in info["mutations"].items():
                # Low probability of detection (these are rare)
                if np.random.random() < 0.05:
                    detected.append(
                        {
                            "mutation": mut,
                            "drug": drug,
                            "fold_change": details["fold_change"],
                            "level": details["level"],
                        }
                    )

        return detected

    def compute_resistance_risk(
        self,
        mutations: list[dict],
        drug: str,
    ) -> float:
        """Compute resistance risk for a specific LA drug.

        Args:
            mutations: List of detected mutations
            drug: Drug abbreviation ("CAB" or "RPV")

        Returns:
            Risk score (0-1)
        """
        relevant = [m for m in mutations if m["drug"] == drug]

        if not relevant:
            return 0.0

        # Accumulate risk
        risk = 0.0
        for mut in relevant:
            level = mut["level"]
            if level == "high":
                risk += 0.4
            elif level == "moderate":
                risk += 0.2
            else:
                risk += 0.1

        return min(1.0, risk)

    def compute_pk_adequacy(self, patient: PatientData) -> float:
        """Compute pharmacokinetic adequacy based on patient factors.

        Args:
            patient: Patient clinical data

        Returns:
            PK adequacy score (0-1)
        """
        pk_score = 1.0

        # BMI adjustment
        for category, info in BMI_CATEGORIES.items():
            if info["range"][0] <= patient.bmi < info["range"][1]:
                pk_score = info["pk_adjustment"]
                break

        # Age adjustment
        if patient.age > 75:
            pk_score *= 0.90
        elif patient.age > 65:
            pk_score *= 0.95

        # Sex adjustment (minor)
        if patient.sex == "F":
            pk_score *= 1.05  # Slightly higher levels in women

        return min(1.0, pk_score)

    def compute_adherence_score(self, patient: PatientData) -> float:
        """Compute predicted injection adherence.

        Args:
            patient: Patient clinical data

        Returns:
            Adherence score (0-1)
        """
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

        return score

    def assess_eligibility(
        self,
        patient: PatientData,
        sequence: Optional[str] = None,
    ) -> LASelectionResult:
        """Assess patient eligibility for LA injectables.

        Args:
            patient: Patient clinical data
            sequence: HIV sequence for resistance analysis (optional)

        Returns:
            LASelectionResult with eligibility assessment
        """
        # Detect mutations if sequence provided
        mutations = self.detect_la_mutations(sequence) if sequence else []

        # Compute component risks
        cab_risk = self.compute_resistance_risk(mutations, "CAB")
        rpv_risk = self.compute_resistance_risk(mutations, "RPV")
        pk_score = self.compute_pk_adequacy(patient)
        adherence_score = self.compute_adherence_score(patient)

        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            patient, mutations, cab_risk, rpv_risk
        )

        # Compute success probability
        success_prob = self._compute_success_probability(
            patient, cab_risk, rpv_risk, pk_score, adherence_score
        )

        # Eligibility decision
        eligible = self._determine_eligibility(
            patient, cab_risk, rpv_risk, pk_score, risk_factors
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(eligible, success_prob)

        # Generate monitoring plan
        monitoring = self._generate_monitoring_plan(
            patient, mutations, eligible
        )

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

    def _identify_risk_factors(
        self,
        patient: PatientData,
        mutations: list[dict],
        cab_risk: float,
        rpv_risk: float,
    ) -> list[str]:
        """Identify clinical risk factors."""
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

        # Psychiatric history
        if patient.psychiatric_history:
            risk_factors.append("Psychiatric history (monitor for mood changes)")

        # Prior NNRTI failure
        nnrti_keywords = ["NNRTI", "EFV", "NVP", "EFAVIRENZ", "NEVIRAPINE"]
        if any(
            any(kw in reg.upper() for kw in nnrti_keywords)
            for reg in patient.prior_regimens
        ):
            risk_factors.append("Prior NNRTI exposure (check for archived resistance)")

        return risk_factors

    def _compute_success_probability(
        self,
        patient: PatientData,
        cab_risk: float,
        rpv_risk: float,
        pk_score: float,
        adherence_score: float,
    ) -> float:
        """Compute predicted success probability."""
        # Base probability for well-selected patients is ~95%
        success_prob = 0.95

        # Resistance penalties
        success_prob -= cab_risk * 0.3
        success_prob -= rpv_risk * 0.4  # RPV resistance more impactful

        # PK penalty
        if pk_score < 0.8:
            success_prob -= (0.8 - pk_score) * 0.2

        # Adherence adjustment
        if adherence_score < 0.8:
            success_prob -= (0.8 - adherence_score) * 0.3
        elif adherence_score > 0.9:
            success_prob += 0.02

        # Viral load requirement
        if patient.viral_load >= 50:
            success_prob -= 0.15

        return max(0.1, min(0.99, success_prob))

    def _determine_eligibility(
        self,
        patient: PatientData,
        cab_risk: float,
        rpv_risk: float,
        pk_score: float,
        risk_factors: list[str],
    ) -> bool:
        """Determine if patient is eligible for LA switch."""
        return (
            patient.viral_load < 50
            and cab_risk < 0.3
            and rpv_risk < 0.4
            and pk_score >= 0.6
            and len(risk_factors) <= 2
        )

    def _generate_recommendation(
        self,
        eligible: bool,
        success_prob: float,
    ) -> str:
        """Generate clinical recommendation text."""
        if eligible and success_prob >= 0.80:
            return "ELIGIBLE - Recommend LA injectable switch"
        elif eligible and success_prob >= 0.60:
            return "ELIGIBLE WITH CAUTION - Close monitoring required"
        elif success_prob >= 0.50:
            return "CONSIDER ALTERNATIVES - Risk factors present"
        else:
            return "NOT RECOMMENDED - Continue oral therapy"

    def _generate_monitoring_plan(
        self,
        patient: PatientData,
        mutations: list[dict],
        eligible: bool,
    ) -> list[str]:
        """Generate monitoring plan if eligible."""
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

        return monitoring

    def assess_batch(
        self,
        patients: list[tuple[PatientData, Optional[str]]],
    ) -> list[LASelectionResult]:
        """Assess multiple patients.

        Args:
            patients: List of (PatientData, sequence) tuples

        Returns:
            List of LASelectionResult objects
        """
        results = []
        for patient, sequence in patients:
            result = self.assess_eligibility(patient, sequence)
            results.append(result)
        return results

    def export_results(
        self,
        results: list[LASelectionResult],
        output_dir: Path,
    ) -> Path:
        """Export LA selection results to JSON.

        Args:
            results: List of LASelectionResult objects
            output_dir: Output directory path

        Returns:
            Path to exported JSON file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        eligible_count = sum(1 for r in results if r.eligible)

        output = {
            "summary": {
                "total_patients": len(results),
                "eligible": eligible_count,
                "eligible_rate": f"{eligible_count / len(results) * 100:.1f}%",
                "mean_success_probability": f"{np.mean([r.success_probability for r in results]) * 100:.1f}%",
            },
            "patients": [r.to_dict() for r in results],
        }

        json_path = output_dir / "la_selection_results.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)

        return json_path


def generate_demo_patients(n: int = 5) -> list[PatientData]:
    """Generate demo patient data for testing.

    Args:
        n: Number of patients to generate

    Returns:
        List of PatientData objects
    """
    np.random.seed(42)

    patients = []
    regimen_options = [
        ["TDF/FTC/DTG"],
        ["TDF/3TC/EFV"],
        ["ABC/3TC/DTG"],
        ["TDF/FTC/EFV", "ABC/3TC/DTG"],
    ]

    for i in range(n):
        patient = PatientData(
            patient_id=f"LA_PATIENT_{i + 1:03d}",
            age=np.random.randint(25, 65),
            sex=np.random.choice(["M", "F"]),
            bmi=np.random.normal(27, 5),
            viral_load=np.random.choice([0, 0, 0, 0, 20, 50, 100, 500]),
            cd4_count=np.random.randint(400, 900),
            prior_regimens=regimen_options[np.random.randint(0, 4)],
            adherence_history=np.random.choice(
                ["excellent", "excellent", "good", "good", "moderate", "poor"],
            ),
            injection_site_concerns=np.random.choice([False, False, False, True]),
            psychiatric_history=np.random.choice([False, False, False, False, True]),
        )
        patients.append(patient)

    return patients
