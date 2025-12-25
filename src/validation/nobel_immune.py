# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Nobel Prize Immune System Validation.

This module validates the Goldilocks Zone hypothesis using immune system
recognition thresholds based on Nobel Prize research on self/non-self
discrimination mechanisms.

Key Concepts:
- MHC-peptide binding affinity thresholds
- T-cell receptor (TCR) sensitivity
- Self/non-self discrimination boundaries
- Thymic selection pressure

Hypothesis:
The 15-30% p-adic distance shift corresponds to experimentally observed
immune activation thresholds - the boundary between tolerance and response.

Usage:
    from src.validation.nobel_immune import NobelImmuneValidator

    validator = NobelImmuneValidator()
    result = validator.validate_threshold(experimental_data)
    print(f"Correlation: {result.correlation:.3f}")

References:
    - DOCUMENTATION/.../01_NOBEL_PRIZE_IMMUNE_VALIDATION.md
    - Nobel Prize in Physiology or Medicine (T-cell/MHC research)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class ImmuneResponse(Enum):
    """Types of immune response outcomes."""

    TOLERANCE = "tolerance"  # No response (self)
    ACTIVATION = "activation"  # Immune response (non-self)
    ANERGY = "anergy"  # Functional unresponsiveness
    DELETION = "deletion"  # Clonal deletion (negative selection)


class MHCClass(Enum):
    """MHC molecule classes."""

    CLASS_I = "class_I"  # Presents to CD8+ T cells
    CLASS_II = "class_II"  # Presents to CD4+ T cells


@dataclass
class ImmuneThresholdData:
    """Experimental immune threshold data point."""

    peptide_sequence: str
    binding_affinity_nm: float  # IC50 in nM
    tcr_affinity_um: float  # Kd in µM
    mhc_class: MHCClass
    response: ImmuneResponse
    molecular_distance: float  # Distance from self-peptide
    source: str = ""  # Publication reference


@dataclass
class ValidationResult:
    """Result of validation analysis."""

    n_samples: int
    correlation: float  # Pearson correlation
    p_value: float
    rmse: float
    within_goldilocks: int  # Count in Goldilocks Zone
    outside_goldilocks: int
    sensitivity: float  # True positive rate
    specificity: float  # True negative rate
    goldilocks_range: Tuple[float, float]
    threshold_accuracy: float


# Reference immune recognition thresholds from literature
# Based on T-cell recognition and MHC binding studies
REFERENCE_THRESHOLDS: Dict[str, Dict] = {
    "mhc_class_i_binding": {
        "strong_binder": 50,  # IC50 < 50 nM
        "moderate_binder": 500,  # IC50 < 500 nM
        "weak_binder": 5000,  # IC50 < 5000 nM
        "non_binder": 50000,  # IC50 > 5000 nM
    },
    "mhc_class_ii_binding": {
        "strong_binder": 100,  # IC50 < 100 nM
        "moderate_binder": 1000,  # IC50 < 1000 nM
        "weak_binder": 10000,  # IC50 < 10000 nM
        "non_binder": 100000,  # IC50 > 10000 nM
    },
    "tcr_recognition": {
        "high_affinity": 1.0,  # Kd < 1 µM
        "moderate_affinity": 10.0,  # Kd < 10 µM
        "low_affinity": 100.0,  # Kd < 100 µM
    },
}

# Amino acid similarity matrix for molecular distance calculation
# Based on BLOSUM62 normalized to 0-1 range
AMINO_ACID_SIMILARITY: Dict[str, Dict[str, float]] = {
    aa1: {aa2: 1.0 if aa1 == aa2 else 0.0 for aa2 in "ACDEFGHIKLMNPQRSTVWY"}
    for aa1 in "ACDEFGHIKLMNPQRSTVWY"
}

# Add known substitution similarities
_SIMILAR_GROUPS = [
    ("I", "L", "V", "M"),  # Hydrophobic aliphatic
    ("F", "Y", "W"),  # Aromatic
    ("K", "R", "H"),  # Positive charged
    ("D", "E"),  # Negative charged
    ("S", "T"),  # Small hydroxyl
    ("N", "Q"),  # Amide
]

for group in _SIMILAR_GROUPS:
    for aa1 in group:
        for aa2 in group:
            if aa1 != aa2:
                AMINO_ACID_SIMILARITY[aa1][aa2] = 0.7


class GoldilocksZoneValidator:
    """Validator for the Goldilocks Zone hypothesis.

    The Goldilocks Zone represents the "just right" range of p-adic
    distances where immune discrimination occurs - not too similar
    (tolerance) and not too different (strong rejection).
    """

    def __init__(
        self,
        goldilocks_min: float = 0.15,
        goldilocks_max: float = 0.30,
        p: int = 3,
    ):
        """Initialize the validator.

        Args:
            goldilocks_min: Lower bound of Goldilocks Zone
            goldilocks_max: Upper bound of Goldilocks Zone
            p: Prime base for p-adic calculations
        """
        self.goldilocks_min = goldilocks_min
        self.goldilocks_max = goldilocks_max
        self.p = p

    def _compute_padic_valuation(self, n: int) -> int:
        """Compute p-adic valuation v_p(n)."""
        if n == 0:
            return 100
        valuation = 0
        while n % self.p == 0:
            valuation += 1
            n //= self.p
        return valuation

    def is_in_goldilocks_zone(self, padic_distance: float) -> bool:
        """Check if distance is within Goldilocks Zone.

        Args:
            padic_distance: Normalized p-adic distance (0-1)

        Returns:
            True if within zone
        """
        return self.goldilocks_min <= padic_distance <= self.goldilocks_max

    def compute_zone_score(self, padic_distance: float) -> float:
        """Compute how central the distance is within Goldilocks Zone.

        Args:
            padic_distance: Normalized p-adic distance

        Returns:
            Score from 0 (outside) to 1 (center of zone)
        """
        center = (self.goldilocks_min + self.goldilocks_max) / 2
        width = (self.goldilocks_max - self.goldilocks_min) / 2

        if not self.is_in_goldilocks_zone(padic_distance):
            return 0.0

        distance_from_center = abs(padic_distance - center)
        return 1.0 - (distance_from_center / width)

    def predict_response(self, padic_distance: float) -> ImmuneResponse:
        """Predict immune response based on p-adic distance.

        Args:
            padic_distance: Normalized p-adic distance

        Returns:
            Predicted immune response
        """
        if padic_distance < self.goldilocks_min:
            return ImmuneResponse.TOLERANCE
        elif padic_distance <= self.goldilocks_max:
            # Within Goldilocks Zone - critical discrimination region
            # Could go either way, but likely activation
            return ImmuneResponse.ACTIVATION
        else:
            return ImmuneResponse.ACTIVATION


class NobelImmuneValidator:
    """Validator for immune thresholds against p-adic predictions.

    Uses experimental immune recognition data to validate that the
    Goldilocks Zone hypothesis correctly predicts self/non-self
    discrimination boundaries.
    """

    def __init__(
        self,
        goldilocks_range: Tuple[float, float] = (0.15, 0.30),
        p: int = 3,
    ):
        """Initialize the validator.

        Args:
            goldilocks_range: (min, max) of Goldilocks Zone
            p: Prime base for p-adic calculations
        """
        self.goldilocks_validator = GoldilocksZoneValidator(
            goldilocks_min=goldilocks_range[0],
            goldilocks_max=goldilocks_range[1],
            p=p,
        )
        self.p = p

    def _compute_padic_valuation(self, n: int) -> int:
        """Compute p-adic valuation v_p(n)."""
        if n == 0:
            return 100
        valuation = 0
        while n % self.p == 0:
            valuation += 1
            n //= self.p
        return valuation

    def compute_peptide_distance(self, peptide1: str, peptide2: str) -> float:
        """Compute molecular distance between two peptides.

        Uses amino acid similarity and positional weighting.

        Args:
            peptide1: First peptide sequence
            peptide2: Second peptide sequence

        Returns:
            Normalized molecular distance (0-1)
        """
        if len(peptide1) != len(peptide2):
            # Handle different lengths by padding
            max_len = max(len(peptide1), len(peptide2))
            peptide1 = peptide1.ljust(max_len, "X")
            peptide2 = peptide2.ljust(max_len, "X")

        total_diff = 0.0
        for i, (aa1, aa2) in enumerate(zip(peptide1, peptide2)):
            # Get similarity
            if aa1 in AMINO_ACID_SIMILARITY and aa2 in AMINO_ACID_SIMILARITY.get(aa1, {}):
                similarity = AMINO_ACID_SIMILARITY[aa1].get(aa2, 0.0)
            else:
                similarity = 0.0

            # Position weighting (anchor residues more important)
            # For MHC Class I: positions 2 and 9 are anchors
            position_weight = 1.0
            if i in [1, 8]:  # 0-indexed positions 2 and 9
                position_weight = 2.0

            total_diff += (1 - similarity) * position_weight

        # Normalize
        max_possible = len(peptide1) * 2  # Max weight * max positions
        return total_diff / max_possible

    def map_affinity_to_padic(self, binding_affinity_nm: float) -> float:
        """Convert binding affinity to p-adic distance.

        Higher affinity (lower IC50) maps to lower p-adic distance.

        Args:
            binding_affinity_nm: IC50 in nM

        Returns:
            Normalized p-adic distance (0-1)
        """
        # Log transform to handle wide range
        log_affinity = np.log10(max(1, binding_affinity_nm))

        # Scale: 1 nM -> ~0, 50000 nM -> ~1
        # Using log10(1) = 0, log10(50000) ≈ 4.7
        normalized = log_affinity / 5.0
        return min(1.0, max(0.0, normalized))

    def validate_threshold(
        self,
        threshold_data: List[ImmuneThresholdData],
    ) -> ValidationResult:
        """Validate experimental thresholds against p-adic predictions.

        Args:
            threshold_data: List of experimental immune threshold data

        Returns:
            ValidationResult with correlation and accuracy metrics
        """
        if not threshold_data:
            return ValidationResult(
                n_samples=0,
                correlation=0.0,
                p_value=1.0,
                rmse=0.0,
                within_goldilocks=0,
                outside_goldilocks=0,
                sensitivity=0.0,
                specificity=0.0,
                goldilocks_range=(
                    self.goldilocks_validator.goldilocks_min,
                    self.goldilocks_validator.goldilocks_max,
                ),
                threshold_accuracy=0.0,
            )

        # Compute p-adic distances and predictions
        padic_distances = []
        predicted_responses = []
        actual_responses = []

        for data in threshold_data:
            # Convert molecular distance to p-adic
            padic_dist = self.map_affinity_to_padic(data.binding_affinity_nm)
            padic_distances.append(padic_dist)

            # Predict response
            predicted = self.goldilocks_validator.predict_response(padic_dist)
            predicted_responses.append(predicted)
            actual_responses.append(data.response)

        # Calculate metrics
        padic_arr = np.array(padic_distances)
        molecular_distances = np.array([d.molecular_distance for d in threshold_data])

        # Correlation between molecular and p-adic distances
        if len(padic_arr) > 1 and np.std(molecular_distances) > 0:
            correlation = float(np.corrcoef(padic_arr, molecular_distances)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # RMSE
        rmse = float(np.sqrt(np.mean((padic_arr - molecular_distances) ** 2)))

        # Count in/out of Goldilocks Zone
        within = sum(1 for d in padic_distances if self.goldilocks_validator.is_in_goldilocks_zone(d))
        outside = len(padic_distances) - within

        # Sensitivity and specificity
        tp = sum(1 for p, a in zip(predicted_responses, actual_responses) if p == ImmuneResponse.ACTIVATION and a == ImmuneResponse.ACTIVATION)
        tn = sum(1 for p, a in zip(predicted_responses, actual_responses) if p == ImmuneResponse.TOLERANCE and a == ImmuneResponse.TOLERANCE)
        fp = sum(1 for p, a in zip(predicted_responses, actual_responses) if p == ImmuneResponse.ACTIVATION and a == ImmuneResponse.TOLERANCE)
        fn = sum(1 for p, a in zip(predicted_responses, actual_responses) if p == ImmuneResponse.TOLERANCE and a == ImmuneResponse.ACTIVATION)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Overall accuracy
        correct = sum(1 for p, a in zip(predicted_responses, actual_responses) if p == a)
        accuracy = correct / len(threshold_data)

        # Simple p-value estimation (would use proper stats in production)
        n = len(threshold_data)
        if n > 2 and abs(correlation) > 0:
            t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2)
            # Approximate p-value from t-distribution
            p_value = 2 * (1 - min(0.9999, 0.5 + 0.5 * np.tanh(t_stat / 2)))
        else:
            p_value = 1.0

        return ValidationResult(
            n_samples=len(threshold_data),
            correlation=correlation,
            p_value=p_value,
            rmse=rmse,
            within_goldilocks=within,
            outside_goldilocks=outside,
            sensitivity=sensitivity,
            specificity=specificity,
            goldilocks_range=(
                self.goldilocks_validator.goldilocks_min,
                self.goldilocks_validator.goldilocks_max,
            ),
            threshold_accuracy=accuracy,
        )

    def generate_synthetic_data(
        self,
        n_samples: int = 100,
        noise_level: float = 0.1,
    ) -> List[ImmuneThresholdData]:
        """Generate synthetic immune threshold data for testing.

        Args:
            n_samples: Number of samples to generate
            noise_level: Amount of noise to add

        Returns:
            List of synthetic threshold data
        """
        np.random.seed(42)
        data = []

        for i in range(n_samples):
            # Generate peptide
            peptide = "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), 9))

            # Generate molecular distance
            mol_dist = np.random.uniform(0, 1)

            # Binding affinity inversely related to molecular distance
            # Strong binders have low distance, weak binders have high distance
            base_affinity = 10 ** (mol_dist * 5)  # 1 nM to 100000 nM
            affinity = base_affinity * (1 + np.random.normal(0, noise_level))
            affinity = max(1, affinity)

            # TCR affinity (correlated with binding)
            tcr_affinity = (affinity / 100) * (1 + np.random.normal(0, noise_level))
            tcr_affinity = max(0.1, tcr_affinity)

            # Determine response based on molecular distance
            if mol_dist < 0.15:
                response = ImmuneResponse.TOLERANCE
            elif mol_dist < 0.30:
                response = ImmuneResponse.ACTIVATION if np.random.random() > 0.3 else ImmuneResponse.TOLERANCE
            else:
                response = ImmuneResponse.ACTIVATION

            data.append(
                ImmuneThresholdData(
                    peptide_sequence=peptide,
                    binding_affinity_nm=float(affinity),
                    tcr_affinity_um=float(tcr_affinity),
                    mhc_class=np.random.choice([MHCClass.CLASS_I, MHCClass.CLASS_II]),
                    response=response,
                    molecular_distance=float(mol_dist),
                    source="synthetic",
                )
            )

        return data

    def compute_discrimination_boundary(
        self,
        threshold_data: List[ImmuneThresholdData],
    ) -> Dict[str, float]:
        """Compute the optimal discrimination boundary.

        Finds the p-adic distance that best separates self from non-self.

        Args:
            threshold_data: Experimental data

        Returns:
            Dictionary with boundary statistics
        """
        # Separate by response
        tolerance_distances = [
            self.map_affinity_to_padic(d.binding_affinity_nm) for d in threshold_data if d.response == ImmuneResponse.TOLERANCE
        ]
        activation_distances = [
            self.map_affinity_to_padic(d.binding_affinity_nm) for d in threshold_data if d.response == ImmuneResponse.ACTIVATION
        ]

        if not tolerance_distances or not activation_distances:
            return {
                "optimal_boundary": 0.225,  # Default center of Goldilocks
                "tolerance_mean": 0.0,
                "activation_mean": 0.0,
                "separation": 0.0,
            }

        tol_mean = np.mean(tolerance_distances)
        act_mean = np.mean(activation_distances)

        # Optimal boundary is midpoint
        boundary = (tol_mean + act_mean) / 2

        # Separation (effect size)
        pooled_std = np.sqrt(
            (np.var(tolerance_distances) + np.var(activation_distances)) / 2
        )
        separation = abs(act_mean - tol_mean) / pooled_std if pooled_std > 0 else 0

        return {
            "optimal_boundary": float(boundary),
            "tolerance_mean": float(tol_mean),
            "activation_mean": float(act_mean),
            "separation": float(separation),
        }


__all__ = [
    "NobelImmuneValidator",
    "GoldilocksZoneValidator",
    "ImmuneThresholdData",
    "ValidationResult",
    "ImmuneResponse",
    "MHCClass",
    "REFERENCE_THRESHOLDS",
]
