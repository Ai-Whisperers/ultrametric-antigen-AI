# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Data models for HIV clinical decision support.

This module defines the core data classes used throughout the HIV
research package for representing patient data, resistance results,
and clinical recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class ResistanceLevel(Enum):
    """Drug resistance level classification.

    Based on Stanford HIVdb scoring system:
    - SUSCEPTIBLE: Score < 10
    - POTENTIAL_LOW: Score 10-14
    - LOW: Score 15-29
    - INTERMEDIATE: Score 30-59
    - HIGH: Score >= 60
    """

    SUSCEPTIBLE = 1
    POTENTIAL_LOW = 2
    LOW = 3
    INTERMEDIATE = 4
    HIGH = 5

    @classmethod
    def from_score(cls, score: int) -> ResistanceLevel:
        """Convert Stanford resistance score to level.

        Args:
            score: Stanford resistance score (0-100)

        Returns:
            ResistanceLevel enum value
        """
        if score < 10:
            return cls.SUSCEPTIBLE
        elif score < 15:
            return cls.POTENTIAL_LOW
        elif score < 30:
            return cls.LOW
        elif score < 60:
            return cls.INTERMEDIATE
        else:
            return cls.HIGH

    @classmethod
    def from_text(cls, text: str) -> ResistanceLevel:
        """Convert Stanford text description to level.

        Args:
            text: Stanford resistance text (e.g., "High-Level Resistance")

        Returns:
            ResistanceLevel enum value
        """
        text = text.lower()
        if "susceptible" in text:
            return cls.SUSCEPTIBLE
        elif "potential" in text:
            return cls.POTENTIAL_LOW
        elif "low" in text:
            return cls.LOW
        elif "intermediate" in text:
            return cls.INTERMEDIATE
        elif "high" in text:
            return cls.HIGH
        return cls.SUSCEPTIBLE

    def is_resistant(self) -> bool:
        """Check if level indicates clinical resistance."""
        return self.value >= ResistanceLevel.LOW.value


@dataclass
class MutationInfo:
    """Information about a detected resistance mutation.

    Attributes:
        gene: Gene name (RT, PR, IN)
        position: Amino acid position
        reference: Reference amino acid
        mutation: Mutant amino acid
        text: Standard notation (e.g., "K103N")
        is_sdrm: Whether this is a WHO surveillance drug resistance mutation
        is_major: Whether this is a major resistance mutation
        drug_class: Drug class affected (NRTI, NNRTI, PI, INSTI)
    """

    gene: str
    position: int
    reference: str
    mutation: str
    text: str
    is_sdrm: bool = False
    is_major: bool = False
    drug_class: Optional[str] = None

    @property
    def notation(self) -> str:
        """Standard mutation notation (e.g., K103N)."""
        return self.text

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class DrugScore:
    """Resistance score for a single antiretroviral drug.

    Attributes:
        drug_name: Full drug name
        drug_abbr: Drug abbreviation (e.g., "EFV")
        drug_class: Drug class (NRTI, NNRTI, PI, INSTI)
        score: Stanford resistance score (0-100)
        level: Resistance level classification
        text: Human-readable resistance description
        mutations: List of mutations contributing to resistance
    """

    drug_name: str
    drug_abbr: str
    drug_class: str
    score: int
    level: ResistanceLevel
    text: str
    mutations: list[str] = field(default_factory=list)

    def is_resistant(self) -> bool:
        """Check if drug shows clinically significant resistance."""
        return self.level.is_resistant()

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "drug_name": self.drug_name,
            "drug_abbr": self.drug_abbr,
            "drug_class": self.drug_class,
            "score": self.score,
            "level": self.level.name,
            "text": self.text,
            "mutations": self.mutations,
        }


@dataclass
class TDRResult:
    """Transmitted drug resistance (TDR) screening result.

    Attributes:
        patient_id: Patient identifier
        sequence_id: Sequence identifier (if from FASTA)
        detected_mutations: List of detected TDR mutations
        drug_susceptibility: Drug -> susceptibility status mapping
        tdr_positive: Whether any TDR mutations were detected
        recommended_regimen: Primary recommended first-line regimen
        alternative_regimens: Alternative regimen options
        resistance_summary: Human-readable resistance summary
        confidence: Analysis confidence score (0-1)
    """

    patient_id: str
    sequence_id: Optional[str]
    detected_mutations: list[dict]
    drug_susceptibility: dict[str, dict]
    tdr_positive: bool
    recommended_regimen: str
    alternative_regimens: list[str]
    resistance_summary: str
    confidence: float

    def get_resistant_drugs(self) -> list[str]:
        """Get list of drugs with detected resistance."""
        return [
            drug
            for drug, info in self.drug_susceptibility.items()
            if info.get("status") == "resistant"
        ]

    def get_susceptible_drugs(self) -> list[str]:
        """Get list of drugs showing full susceptibility."""
        return [
            drug
            for drug, info in self.drug_susceptibility.items()
            if info.get("status") == "susceptible"
        ]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "patient_id": self.patient_id,
            "sequence_id": self.sequence_id,
            "tdr_positive": self.tdr_positive,
            "mutations_detected": len(self.detected_mutations),
            "mutations": self.detected_mutations,
            "resistance_summary": self.resistance_summary,
            "recommended_regimen": self.recommended_regimen,
            "alternative_regimens": self.alternative_regimens,
            "drug_susceptibility": {
                drug: info["status"] for drug, info in self.drug_susceptibility.items()
            },
            "resistant_drugs": self.get_resistant_drugs(),
            "confidence": self.confidence,
        }


@dataclass
class PatientData:
    """Patient clinical data for LA injectable selection.

    Attributes:
        patient_id: Patient identifier
        age: Patient age in years
        sex: Biological sex (M/F)
        bmi: Body mass index
        viral_load: HIV RNA copies/mL (should be <50 for LA switch)
        cd4_count: CD4+ T-cell count (cells/mm3)
        prior_regimens: List of prior ART regimens
        adherence_history: Self-reported adherence level
        injection_site_concerns: Whether patient has injection concerns
        psychiatric_history: History of psychiatric conditions
    """

    patient_id: str
    age: int
    sex: str
    bmi: float
    viral_load: float
    cd4_count: int
    prior_regimens: list[str]
    adherence_history: str  # "excellent", "good", "moderate", "poor"
    injection_site_concerns: bool = False
    psychiatric_history: bool = False

    def is_virally_suppressed(self) -> bool:
        """Check if patient is virally suppressed (<50 copies/mL)."""
        return self.viral_load < 50

    def get_bmi_category(self) -> str:
        """Get BMI category classification."""
        if self.bmi < 18.5:
            return "underweight"
        elif self.bmi < 25:
            return "normal"
        elif self.bmi < 30:
            return "overweight"
        elif self.bmi < 35:
            return "obese_1"
        elif self.bmi < 40:
            return "obese_2"
        else:
            return "obese_3"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class LASelectionResult:
    """Long-acting injectable eligibility assessment result.

    Attributes:
        patient_id: Patient identifier
        eligible: Whether patient is eligible for LA switch
        success_probability: Predicted probability of maintaining suppression
        cab_resistance_risk: Risk of cabotegravir resistance (0-1)
        rpv_resistance_risk: Risk of rilpivirine resistance (0-1)
        pk_adequacy_score: Pharmacokinetic adequacy score (0-1)
        adherence_score: Predicted injection adherence score (0-1)
        detected_mutations: List of LA-relevant mutations
        recommendation: Clinical recommendation text
        risk_factors: List of identified risk factors
        monitoring_plan: Recommended monitoring schedule
    """

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

    def get_risk_category(self) -> str:
        """Get overall risk category."""
        if self.success_probability >= 0.90:
            return "low_risk"
        elif self.success_probability >= 0.75:
            return "moderate_risk"
        else:
            return "high_risk"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "patient_id": self.patient_id,
            "eligible": self.eligible,
            "success_probability": f"{self.success_probability * 100:.1f}%",
            "recommendation": self.recommendation,
            "risk_category": self.get_risk_category(),
            "cab_resistance_risk": f"{self.cab_resistance_risk * 100:.1f}%",
            "rpv_resistance_risk": f"{self.rpv_resistance_risk * 100:.1f}%",
            "pk_adequacy": f"{self.pk_adequacy_score * 100:.1f}%",
            "adherence_score": f"{self.adherence_score * 100:.1f}%",
            "detected_mutations": self.detected_mutations,
            "risk_factors": self.risk_factors,
            "monitoring_plan": self.monitoring_plan,
        }


@dataclass
class ResistanceReport:
    """Complete Stanford HIVdb resistance analysis report.

    Attributes:
        patient_id: Patient identifier
        subtype: HIV-1 subtype
        mutations: List of all detected mutations
        drug_scores: List of drug resistance scores
        quality_issues: List of sequence quality warnings
    """

    patient_id: str
    subtype: str
    mutations: list[MutationInfo]
    drug_scores: list[DrugScore]
    quality_issues: list[str] = field(default_factory=list)

    def get_resistant_drugs(self) -> list[str]:
        """Get list of drugs with resistance."""
        return [d.drug_abbr for d in self.drug_scores if d.is_resistant()]

    def get_sdrm_mutations(self) -> list[MutationInfo]:
        """Get WHO surveillance drug resistance mutations only."""
        return [m for m in self.mutations if m.is_sdrm]

    def has_tdr(self) -> bool:
        """Check if any transmitted drug resistance is present."""
        return len(self.get_sdrm_mutations()) > 0

    def get_recommended_regimens(self) -> list[str]:
        """Get recommended first-line regimens based on resistance profile."""
        resistant = set(self.get_resistant_drugs())

        regimens = [
            ("TDF/3TC/DTG", {"TDF", "3TC", "DTG"}),
            ("TDF/FTC/DTG", {"TDF", "FTC", "DTG"}),
            ("TAF/FTC/DTG", {"TAF", "FTC", "DTG"}),
            ("TDF/3TC/EFV", {"TDF", "3TC", "EFV"}),
            ("ABC/3TC/DTG", {"ABC", "3TC", "DTG"}),
        ]

        recommended = []
        for name, drugs in regimens:
            if not drugs & resistant:
                recommended.append(name)

        return recommended if recommended else ["Specialist referral needed"]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "patient_id": self.patient_id,
            "subtype": self.subtype,
            "tdr_positive": self.has_tdr(),
            "mutations": [m.to_dict() for m in self.mutations],
            "sdrm_mutations": [m.notation for m in self.get_sdrm_mutations()],
            "drug_scores": [d.to_dict() for d in self.drug_scores],
            "resistant_drugs": self.get_resistant_drugs(),
            "recommended_regimens": self.get_recommended_regimens(),
            "quality_issues": self.quality_issues,
        }
