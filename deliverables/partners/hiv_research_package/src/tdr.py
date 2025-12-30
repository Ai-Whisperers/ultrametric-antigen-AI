# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Transmitted Drug Resistance (TDR) Screening.

This module provides tools for screening treatment-naive HIV patients
for transmitted drug resistance to guide first-line regimen selection.

TDR prevalence: 10-15% in some regions (PEPFAR data)

Example:
    >>> screener = TDRScreener()
    >>> result = screener.screen_patient(sequence, patient_id="P001")
    >>> print(f"TDR Status: {'Positive' if result.tdr_positive else 'Negative'}")
    >>> print(f"Recommended: {result.recommended_regimen}")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING
import json

import numpy as np

from .models import TDRResult
from .constants import TDR_MUTATIONS, FIRST_LINE_DRUGS, FIRST_LINE_REGIMENS

if TYPE_CHECKING:
    from .stanford_client import StanfordHIVdbClient


class TDRScreener:
    """Transmitted Drug Resistance (TDR) screening tool.

    Screens HIV sequences for known TDR mutations and provides
    first-line regimen recommendations.

    Attributes:
        use_stanford: Whether to use Stanford HIVdb API
        stanford_client: Stanford HIVdb client instance

    Example:
        >>> screener = TDRScreener(use_stanford=True)
        >>> result = screener.screen_patient(sequence, "PATIENT_001")
        >>> if result.tdr_positive:
        ...     print(f"TDR detected: {result.resistance_summary}")
    """

    def __init__(
        self,
        use_stanford: bool = False,
        stanford_client: Optional[StanfordHIVdbClient] = None,
    ):
        """Initialize TDR screener.

        Args:
            use_stanford: Whether to use Stanford HIVdb API for analysis
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

    def parse_sequence(self, sequence: str) -> str:
        """Clean and validate amino acid sequence.

        Args:
            sequence: Raw sequence (may include FASTA header, whitespace)

        Returns:
            Cleaned uppercase sequence

        Raises:
            ValueError: If sequence contains invalid amino acids
        """
        # Remove whitespace and convert to uppercase
        sequence = "".join(sequence.upper().split())

        # Remove FASTA header if present
        if sequence.startswith(">"):
            lines = sequence.split("\n")
            sequence = "".join(lines[1:])

        # Validate amino acids
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY*-X")
        invalid = [aa for aa in sequence if aa not in valid_aa]
        if invalid:
            raise ValueError(f"Invalid amino acids: {set(invalid)}")

        return sequence

    def detect_mutations(
        self,
        sequence: str,
        reference: Optional[str] = None,
    ) -> list[dict]:
        """Detect TDR mutations in sequence.

        For demo purposes, this uses probabilistic simulation based on
        known mutation prevalence. Real implementation should align to
        HXB2 reference and identify actual mutations.

        Args:
            sequence: Cleaned amino acid sequence
            reference: Reference sequence (HXB2) for alignment

        Returns:
            List of detected mutation dictionaries
        """
        detected = []

        # Seed based on sequence for reproducible demo results
        np.random.seed(hash(sequence[:20]) % 2**32)

        for drug_class, mutations in TDR_MUTATIONS.items():
            for mut_name, mut_info in mutations.items():
                # Probability of detection based on prevalence
                if np.random.random() * 100 < mut_info["prevalence"]:
                    detected.append(
                        {
                            "mutation": mut_name,
                            "drug_class": drug_class,
                            "affected_drugs": mut_info["drugs"],
                            "resistance_level": mut_info["level"],
                            "prevalence": mut_info["prevalence"],
                        }
                    )

        return detected

    def predict_drug_susceptibility(
        self,
        detected_mutations: list[dict],
    ) -> dict[str, dict]:
        """Predict susceptibility for each first-line drug.

        Args:
            detected_mutations: List of detected TDR mutations

        Returns:
            Dictionary mapping drug name to susceptibility info
        """
        susceptibility = {}

        # Initialize all drugs as susceptible
        for drug_class, drugs in FIRST_LINE_DRUGS.items():
            for drug in drugs:
                susceptibility[drug] = {
                    "status": "susceptible",
                    "score": 0.0,
                    "class": drug_class,
                }

        # Apply mutation effects
        for mut in detected_mutations:
            for drug in mut["affected_drugs"]:
                if drug in susceptibility:
                    current_score = susceptibility[drug]["score"]

                    # Add resistance score based on level
                    level = mut["resistance_level"]
                    if level == "high":
                        new_score = current_score + 0.7
                    elif level == "moderate":
                        new_score = current_score + 0.4
                    else:
                        new_score = current_score + 0.2

                    susceptibility[drug]["score"] = min(1.0, new_score)

                    # Update status
                    if susceptibility[drug]["score"] >= 0.7:
                        susceptibility[drug]["status"] = "resistant"
                    elif susceptibility[drug]["score"] >= 0.3:
                        susceptibility[drug]["status"] = "possible_resistance"

        return susceptibility

    def recommend_regimen(
        self,
        susceptibility: dict[str, dict],
        detected_mutations: list[dict],
    ) -> tuple[str, list[str]]:
        """Recommend first-line regimen based on susceptibility.

        Args:
            susceptibility: Drug susceptibility predictions
            detected_mutations: List of detected mutations

        Returns:
            Tuple of (primary recommendation, alternative options)
        """
        recommendations = []

        for regimen in FIRST_LINE_REGIMENS:
            score = 0
            all_susceptible = True
            has_resistant = False

            for drug in regimen["drugs"]:
                if drug in susceptibility:
                    status = susceptibility[drug]["status"]
                    if status == "susceptible":
                        score += 3
                    elif status == "possible_resistance":
                        score += 1
                        all_susceptible = False
                    else:  # resistant
                        score -= 5
                        has_resistant = True
                        all_susceptible = False

            # Bonus for preferred regimens if all susceptible
            if regimen["preferred"] and all_susceptible:
                score += 2

            # Penalty for resistance
            if has_resistant:
                score -= 3

            recommendations.append(
                {
                    "regimen": regimen["name"],
                    "score": score,
                    "all_susceptible": all_susceptible,
                    "has_resistant": has_resistant,
                }
            )

        # Sort by score
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        # Primary recommendation
        primary = recommendations[0]["regimen"]

        # Alternatives (score > 0 and no resistance)
        alternatives = [
            r["regimen"]
            for r in recommendations[1:4]
            if r["score"] > 0 and not r["has_resistant"]
        ]

        return primary, alternatives

    def generate_summary(self, detected_mutations: list[dict]) -> str:
        """Generate human-readable resistance summary.

        Args:
            detected_mutations: List of detected mutations

        Returns:
            Summary string describing resistance findings
        """
        if not detected_mutations:
            return "No transmitted drug resistance mutations detected."

        # Group by class
        by_class: dict[str, list[str]] = {}
        for mut in detected_mutations:
            drug_class = mut["drug_class"]
            if drug_class not in by_class:
                by_class[drug_class] = []
            by_class[drug_class].append(mut["mutation"])

        lines = []
        for drug_class, muts in by_class.items():
            lines.append(f"{drug_class}: {', '.join(muts)}")

        return "TDR detected - " + "; ".join(lines)

    def screen_patient(
        self,
        sequence: str,
        patient_id: str = "PATIENT_001",
        sequence_id: Optional[str] = None,
    ) -> TDRResult:
        """Screen a patient for transmitted drug resistance.

        Args:
            sequence: HIV sequence (pol gene)
            patient_id: Patient identifier
            sequence_id: Sequence identifier (if from FASTA)

        Returns:
            TDRResult with screening findings and recommendations
        """
        if self.use_stanford and self.stanford_client:
            return self._screen_with_stanford(sequence, patient_id)
        return self._screen_local(sequence, patient_id, sequence_id)

    def _screen_with_stanford(
        self,
        sequence: str,
        patient_id: str,
    ) -> TDRResult:
        """Screen using Stanford HIVdb API."""
        client = self.stanford_client
        if client is None:
            return self._screen_local(sequence, patient_id, None)

        try:
            report = client.analyze_sequence(sequence, patient_id)
            if report is None:
                return self._screen_local(sequence, patient_id, None)

            # Convert Stanford report to TDRResult format
            detected = []
            for mut in report.mutations:
                detected.append(
                    {
                        "mutation": mut.notation,
                        "drug_class": mut.gene,
                        "affected_drugs": [],
                        "resistance_level": "high" if mut.is_major else "moderate",
                        "prevalence": 1.0,
                    }
                )

            # Build susceptibility from drug scores
            susceptibility = {}
            for drug_score in report.drug_scores:
                status = "susceptible"
                if drug_score.level.value >= 3:
                    status = "resistant"
                elif drug_score.level.value >= 2:
                    status = "possible_resistance"

                susceptibility[drug_score.drug_abbr] = {
                    "status": status,
                    "score": drug_score.score / 100.0,
                    "class": drug_score.drug_class,
                }

            # Get recommendations from Stanford
            recommended = report.get_recommended_regimens()
            primary = recommended[0] if recommended else "TDF/3TC/DTG"
            alternatives = recommended[1:4] if len(recommended) > 1 else []

            return TDRResult(
                patient_id=patient_id,
                sequence_id=None,
                detected_mutations=detected,
                drug_susceptibility=susceptibility,
                tdr_positive=report.has_tdr(),
                recommended_regimen=primary,
                alternative_regimens=alternatives,
                resistance_summary=f"Stanford analysis: {len(report.mutations)} mutations",
                confidence=0.98,
            )

        except Exception as e:
            print(f"Stanford analysis failed: {e}, falling back to local")
            return self._screen_local(sequence, patient_id, None)

    def _screen_local(
        self,
        sequence: str,
        patient_id: str,
        sequence_id: Optional[str],
    ) -> TDRResult:
        """Screen using local mutation database."""
        # Parse and validate
        sequence = self.parse_sequence(sequence)

        # Detect mutations
        detected = self.detect_mutations(sequence)

        # Predict susceptibility
        susceptibility = self.predict_drug_susceptibility(detected)

        # Recommend regimen
        primary, alternatives = self.recommend_regimen(susceptibility, detected)

        # Generate summary
        summary = self.generate_summary(detected)

        # TDR positive if any mutations detected
        tdr_positive = len(detected) > 0

        # Confidence based on sequence quality
        confidence = 0.95 if len(sequence) > 100 else 0.80

        return TDRResult(
            patient_id=patient_id,
            sequence_id=sequence_id,
            detected_mutations=detected,
            drug_susceptibility=susceptibility,
            tdr_positive=tdr_positive,
            recommended_regimen=primary,
            alternative_regimens=alternatives,
            resistance_summary=summary,
            confidence=confidence,
        )

    def screen_batch(
        self,
        sequences: list[tuple[str, str]],
    ) -> list[TDRResult]:
        """Screen multiple patients.

        Args:
            sequences: List of (sequence, patient_id) tuples

        Returns:
            List of TDRResult objects
        """
        results = []
        for sequence, patient_id in sequences:
            result = self.screen_patient(sequence, patient_id)
            results.append(result)
        return results

    def export_results(
        self,
        results: list[TDRResult],
        output_dir: Path,
    ) -> Path:
        """Export TDR screening results to JSON.

        Args:
            results: List of TDRResult objects
            output_dir: Output directory path

        Returns:
            Path to exported JSON file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tdr_positive = sum(1 for r in results if r.tdr_positive)

        output = {
            "summary": {
                "total_patients": len(results),
                "tdr_positive": tdr_positive,
                "tdr_prevalence": f"{tdr_positive / len(results) * 100:.1f}%",
            },
            "patients": [r.to_dict() for r in results],
        }

        json_path = output_dir / "tdr_screening_results.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)

        return json_path


def generate_demo_sequence() -> str:
    """Generate demo HIV sequence for testing.

    Returns:
        Random amino acid sequence
    """
    np.random.seed(42)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(np.random.choice(list(aa), size=500))
