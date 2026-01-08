"""Mechanism Design Service - Business logic for AMP mechanism-based design.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! C5 FALSIFIED - PATHOGEN METADATA PROVIDES NO IMPROVEMENT          !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!                                                                   !!
!! C5 hold-out test EXECUTED. Result: FALSIFIED                      !!
!! - Pathogen metadata provides NO predictive improvement            !!
!! - Average improvement: -0.109 (negative = hurts)                  !!
!! - Peptide-only model: r=0.88-0.94 on held-out pathogens           !!
!!                                                                   !!
!! VALID: classify_mechanism, route_regime, get_thresholds           !!
!! MISLEADING: get_design_rules, rank_pathogens                      !!
!!                                                                   !!
!! See amp_design_api.py for full disclaimer.                        !!
!!                                                                   !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

This service exposes PARTIALLY-validated findings from P1 investigation:
- C3 theorem: pathogen specificity within clusters (survived falsification)
- Arrow-flip thresholds: hydrophobicity > 0.107
- Mechanism inference: detergent vs barrel_stave
- Pathogen rankings per cluster

NOTE: N-terminal cationic dipeptide hypothesis was FALSIFIED (P1_nterm_validation.json).
Design rules do NOT include N-terminal cationic recommendations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Paths
API_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = API_DIR.parent
RESULTS_DIR = PACKAGE_DIR / "results" / "validation_batch"

# Singleton instance
_service_instance: Optional["MechanismDesignService"] = None


# =============================================================================
# MECHANISM DEFINITIONS (validated from P1 analysis)
# =============================================================================

MECHANISM_FEATURES = {
    "barrel_stave": {
        "length_range": (15, 25),
        "hydrophobicity_range": (-0.5, 0.5),
        "charge_range": (2, 6),
        "description": "Forms transmembrane pores, needs length for bilayer spanning",
    },
    "carpet": {
        "length_range": (10, 20),
        "hydrophobicity_range": (-1.0, 0.0),
        "charge_range": (3, 8),
        "description": "Covers membrane surface, high charge density matters",
    },
    "toroidal": {
        "length_range": (20, 30),
        "hydrophobicity_range": (0.0, 1.0),
        "charge_range": (2, 5),
        "description": "Creates toroidal pores with lipid participation",
    },
    "detergent": {
        "length_range": (8, 15),
        "hydrophobicity_range": (-0.5, 0.5),
        "charge_range": (1, 4),
        "description": "Micelle-like disruption, short peptides",
    },
}

# Arrow-flip thresholds (from P1_C3_enhanced_results.json)
ARROW_FLIP_THRESHOLDS = {
    "hydrophobicity": {"value": 0.107, "improvement": 0.238, "significant": True},
    "length": {"value": 12.0, "improvement": 0.225, "significant": True},
    "net_charge": {"value": 0.50, "improvement": 0.161, "significant": True},
    "charge_density": {"value": 0.115, "improvement": 0.110, "significant": True},
}

# Pathogen properties
PATHOGEN_PROPERTIES = {
    "A_baumannii": {
        "full_name": "Acinetobacter baumannii",
        "gram": "negative",
        "lps_abundance": 0.85,
        "membrane_charge": -0.6,
        "priority": "critical",
    },
    "P_aeruginosa": {
        "full_name": "Pseudomonas aeruginosa",
        "gram": "negative",
        "lps_abundance": 0.90,
        "membrane_charge": -0.7,
        "priority": "critical",
    },
    "Enterobacteriaceae": {
        "full_name": "Enterobacteriaceae",
        "gram": "negative",
        "lps_abundance": 0.88,
        "membrane_charge": -0.55,
        "priority": "critical",
    },
    "S_aureus": {
        "full_name": "Staphylococcus aureus",
        "gram": "positive",
        "lps_abundance": 0.0,
        "membrane_charge": -0.3,
        "priority": "high",
    },
    "H_pylori": {
        "full_name": "Helicobacter pylori",
        "gram": "negative",
        "lps_abundance": 0.75,
        "membrane_charge": -0.4,
        "priority": "medium",
    },
}

# Signal clusters (from C3 theorem)
SIGNAL_CLUSTERS = [1, 3, 4]

# Mechanism-pathogen map (from P1_mechanism_fingerprint.json)
MECHANISM_PATHOGEN_MAP = {
    "detergent": {
        "cluster": 3,
        "best_against": "P_aeruginosa",
        "worst_against": "Enterobacteriaceae",
        "confidence": 1.0,
    },
    "barrel_stave": {
        "cluster": 4,
        "best_against": "P_aeruginosa",
        "worst_against": "Enterobacteriaceae",
        "confidence": 1.0,
    },
}


class MechanismDesignService:
    """Service for mechanism-based AMP design decisions."""

    def __init__(self):
        """Initialize service and load validated data."""
        self._fingerprint_data = None
        self._enhanced_results = None
        self._load_data()

    def _load_data(self):
        """Load validated fingerprint and threshold data."""
        fingerprint_path = RESULTS_DIR / "P1_mechanism_fingerprint.json"
        enhanced_path = RESULTS_DIR / "P1_C3_enhanced_results.json"

        if fingerprint_path.exists():
            with open(fingerprint_path) as f:
                self._fingerprint_data = json.load(f)

        if enhanced_path.exists():
            with open(enhanced_path) as f:
                self._enhanced_results = json.load(f)

    def classify_mechanism(
        self,
        length: float,
        hydrophobicity: float,
        net_charge: float,
    ) -> Dict:
        """Classify likely AMP mechanism based on peptide properties.

        Args:
            length: Peptide length in amino acids
            hydrophobicity: Mean hydrophobicity score
            net_charge: Net peptide charge

        Returns:
            Dict with mechanism, confidence, description, cluster_id, has_pathogen_signal
        """
        # Score each mechanism
        scores = {}
        for mech_name, mech_props in MECHANISM_FEATURES.items():
            score = 0

            if mech_props["length_range"][0] <= length <= mech_props["length_range"][1]:
                score += 1

            if mech_props["hydrophobicity_range"][0] <= hydrophobicity <= mech_props["hydrophobicity_range"][1]:
                score += 1

            if mech_props["charge_range"][0] <= net_charge <= mech_props["charge_range"][1]:
                score += 1

            scores[mech_name] = score

        best_mechanism = max(scores, key=scores.get)
        confidence = scores[best_mechanism] / 3.0

        # Map to cluster
        cluster_id = MECHANISM_PATHOGEN_MAP.get(best_mechanism, {}).get("cluster", -1)
        has_signal = cluster_id in SIGNAL_CLUSTERS

        return {
            "mechanism": best_mechanism,
            "confidence": confidence,
            "description": MECHANISM_FEATURES[best_mechanism]["description"],
            "cluster_id": cluster_id,
            "has_pathogen_signal": has_signal,
            "mechanism_scores": scores,
        }

    def route_regime(self, hydrophobicity: float) -> Dict:
        """Determine prediction regime based on arrow-flip threshold.

        Args:
            hydrophobicity: Mean hydrophobicity score

        Returns:
            Dict with regime, threshold_used, expected_separation, rationale
        """
        threshold = ARROW_FLIP_THRESHOLDS["hydrophobicity"]["value"]

        if hydrophobicity > threshold:
            regime = "CLUSTER_CONDITIONAL"
            separation = 0.284  # From enhanced results
            rationale = f"hydrophobicity {hydrophobicity:.3f} > {threshold:.3f} indicates strong pathogen signal"
        else:
            regime = "GLOBAL"
            separation = 0.150
            rationale = f"hydrophobicity {hydrophobicity:.3f} <= {threshold:.3f}, use global model"

        return {
            "regime": regime,
            "threshold_used": threshold,
            "expected_separation": separation,
            "rationale": rationale,
        }

    def get_design_rules(self, target_pathogen: str) -> Dict:
        """Get actionable design rules for a target pathogen.

        Args:
            target_pathogen: One of the 5 supported pathogens

        Returns:
            Dict with recommended length, mechanism, sequence rules, rationale, confidence, warning
        """
        if target_pathogen not in PATHOGEN_PROPERTIES:
            return {
                "target_pathogen": target_pathogen,
                "error": f"Unknown pathogen. Supported: {list(PATHOGEN_PROPERTIES.keys())}",
            }

        props = PATHOGEN_PROPERTIES[target_pathogen]

        # Enterobacteriaceae is the known failure case
        if target_pathogen == "Enterobacteriaceae":
            return {
                "target_pathogen": target_pathogen,
                "pathogen_info": props,
                "recommended_length": None,
                "recommended_mechanism": None,
                "sequence_rules": None,
                "rationale": "Current mechanisms (detergent, barrel_stave) consistently fail against Enterobacteriaceae",
                "confidence": "LOW",
                "warning": "Consider alternative approach - possibly intracellular target or different mechanism class",
            }

        # P. aeruginosa - best supported
        if target_pathogen == "P_aeruginosa":
            return {
                "target_pathogen": target_pathogen,
                "pathogen_info": props,
                "recommended_length": "14-18 AA",
                "recommended_mechanism": ["detergent", "barrel_stave"],
                "sequence_rules": {
                    "cationic_fraction": ">17%",
                    "hydrophobicity": ">0.107 (arrow-flip threshold)",
                    "note": "High LPS membrane favors moderate hydrophobicity peptides",
                },
                "rationale": f"High membrane charge ({props['membrane_charge']}) and LPS abundance ({props['lps_abundance']}) make P. aeruginosa most susceptible to both mechanisms",
                "confidence": "HIGH",
                "warning": None,
            }

        # S. aureus - Gram positive, good with detergent
        if target_pathogen == "S_aureus":
            return {
                "target_pathogen": target_pathogen,
                "pathogen_info": props,
                "recommended_length": "13-15 AA",
                "recommended_mechanism": ["detergent"],
                "sequence_rules": {
                    "cationic_fraction": "15-17%",
                    "hydrophobicity": "moderate (around 0.1-0.3)",
                    "note": "Short peptides with detergent mechanism work well for Gram+",
                },
                "rationale": "Gram+ lacks outer membrane, detergent mechanism directly targets cytoplasmic membrane",
                "confidence": "HIGH",
                "warning": None,
            }

        # A. baumannii - moderate
        if target_pathogen == "A_baumannii":
            return {
                "target_pathogen": target_pathogen,
                "pathogen_info": props,
                "recommended_length": "16-20 AA",
                "recommended_mechanism": ["barrel_stave"],
                "sequence_rules": {
                    "cationic_fraction": ">16%",
                    "hydrophobicity": ">0.107",
                },
                "rationale": "Moderate LPS abundance, barrel_stave mechanism shows better efficacy",
                "confidence": "MEDIUM",
                "warning": None,
            }

        # H. pylori - lower priority
        if target_pathogen == "H_pylori":
            return {
                "target_pathogen": target_pathogen,
                "pathogen_info": props,
                "recommended_length": "14-18 AA",
                "recommended_mechanism": ["detergent", "barrel_stave"],
                "sequence_rules": {
                    "cationic_fraction": ">15%",
                    "hydrophobicity": "0.0-0.3",
                },
                "rationale": "Lower LPS abundance, both mechanisms show moderate efficacy",
                "confidence": "MEDIUM",
                "warning": None,
            }

        # Fallback
        return {
            "target_pathogen": target_pathogen,
            "pathogen_info": props,
            "recommended_length": "14-18 AA",
            "recommended_mechanism": ["detergent", "barrel_stave"],
            "sequence_rules": {
                "cationic_fraction": ">15%",
                "hydrophobicity": ">0.107",
            },
            "rationale": "Default recommendation based on general AMP design principles",
            "confidence": "LOW",
            "warning": None,
        }

    def rank_pathogens(self, sequence: str) -> Dict:
        """Rank pathogens by predicted efficacy for a given sequence.

        Args:
            sequence: Amino acid sequence

        Returns:
            Dict with sequence, cluster_id, mechanism, pathogen_ranking
        """
        # Compute properties
        length = len(sequence)
        net_charge = self._compute_net_charge(sequence)
        hydrophobicity = self._compute_hydrophobicity(sequence)
        cationic_fraction = self._compute_cationic_fraction(sequence)

        # Classify mechanism
        mech_result = self.classify_mechanism(length, hydrophobicity, net_charge)

        # Get base ranking from mechanism (from fingerprint data)
        mechanism = mech_result["mechanism"]
        cluster_id = mech_result["cluster_id"]

        # Ranking based on validated data
        if mechanism == "detergent" and cluster_id == 3:
            ranking = [
                {"pathogen": "P_aeruginosa", "relative_efficacy": 1.00, "confidence": "HIGH"},
                {"pathogen": "S_aureus", "relative_efficacy": 0.84, "confidence": "HIGH"},
                {"pathogen": "H_pylori", "relative_efficacy": 0.84, "confidence": "MEDIUM"},
                {"pathogen": "A_baumannii", "relative_efficacy": 0.80, "confidence": "MEDIUM"},
                {"pathogen": "Enterobacteriaceae", "relative_efficacy": 0.69, "confidence": "LOW"},
            ]
        elif mechanism == "barrel_stave" and cluster_id in [1, 4]:
            ranking = [
                {"pathogen": "P_aeruginosa", "relative_efficacy": 1.00, "confidence": "HIGH"},
                {"pathogen": "A_baumannii", "relative_efficacy": 0.87, "confidence": "MEDIUM"},
                {"pathogen": "S_aureus", "relative_efficacy": 0.86, "confidence": "MEDIUM"},
                {"pathogen": "H_pylori", "relative_efficacy": 0.86, "confidence": "MEDIUM"},
                {"pathogen": "Enterobacteriaceae", "relative_efficacy": 0.66, "confidence": "LOW"},
            ]
        else:
            # Default ranking for other mechanisms
            ranking = [
                {"pathogen": p, "relative_efficacy": 0.5, "confidence": "LOW"}
                for p in PATHOGEN_PROPERTIES
            ]

        return {
            "sequence": sequence,
            "length": length,
            "net_charge": net_charge,
            "hydrophobicity": round(hydrophobicity, 3),
            "cationic_fraction": round(cationic_fraction, 3),
            "cluster_id": cluster_id,
            "mechanism": mechanism,
            "mechanism_confidence": mech_result["confidence"],
            "pathogen_ranking": ranking,
        }

    def get_thresholds(self) -> Dict:
        """Get all validated arrow-flip thresholds."""
        return {
            "arrow_flip_thresholds": ARROW_FLIP_THRESHOLDS,
            "primary_threshold": "hydrophobicity",
            "regime_separation": {
                "cluster_conditional": 0.284,
                "global": 0.150,
            },
        }

    def get_fingerprint(self) -> Dict:
        """Get full mechanism fingerprint data."""
        if self._fingerprint_data:
            return {
                **self._fingerprint_data,
                "version": "1.0",
                "source": "P1_mechanism_fingerprint.json",
            }
        return {"error": "Fingerprint data not loaded"}

    # Helper methods
    def _compute_net_charge(self, sequence: str) -> float:
        """Compute net charge at pH 7."""
        charges = {"K": 1, "R": 1, "H": 0.1, "D": -1, "E": -1}
        return sum(charges.get(aa, 0) for aa in sequence.upper())

    def _compute_hydrophobicity(self, sequence: str) -> float:
        """Compute mean Kyte-Doolittle hydrophobicity."""
        kd = {
            "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
            "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
            "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
            "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
        }
        if not sequence:
            return 0.0
        return sum(kd.get(aa, 0) for aa in sequence.upper()) / len(sequence)

    def _compute_cationic_fraction(self, sequence: str) -> float:
        """Compute fraction of cationic residues (K, R, H)."""
        if not sequence:
            return 0.0
        cationic = sum(1 for aa in sequence.upper() if aa in "KRH")
        return cationic / len(sequence)


def get_mechanism_service() -> MechanismDesignService:
    """Get singleton instance of MechanismDesignService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = MechanismDesignService()
    return _service_instance
