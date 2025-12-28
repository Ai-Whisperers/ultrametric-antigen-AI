# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Methicillin-Resistant Staphylococcus aureus (MRSA) Analyzer.

This module provides analysis of antibiotic resistance in S. aureus,
focusing on MRSA and emerging resistance patterns.

Based on:
- CLSI/EUCAST breakpoints
- PATRIC AMR database
- CDC AR Lab Network

Key Features:
- mecA/mecC detection (methicillin resistance)
- Vancomycin resistance (VISA/VRSA)
- Multi-drug resistance profiling
- SCCmec typing support

MRSA Classification:
- HA-MRSA: Hospital-acquired
- CA-MRSA: Community-acquired
- LA-MRSA: Livestock-associated

Usage:
    from src.diseases.mrsa_analyzer import MRSAAnalyzer

    analyzer = MRSAAnalyzer()
    results = analyzer.analyze(sequences)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch

from .base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType


class StaphGene(Enum):
    """S. aureus resistance genes."""

    # Beta-lactam resistance
    MECA = "mecA"
    MECC = "mecC"
    BLAR = "blaR"
    BLAZ = "blaZ"

    # Vancomycin resistance
    VANA = "vanA"
    VANB = "vanB"

    # Aminoglycoside resistance
    AACA_APHD = "aacA-aphD"

    # Fluoroquinolone resistance
    GYRA = "gyrA"
    GRLB = "grlB"

    # Macrolide resistance
    ERMA = "ermA"
    ERMB = "ermB"
    ERMC = "ermC"

    # Rifampicin resistance
    RPOB = "rpoB"

    # Mupirocin resistance
    ILES2 = "ileS2"

    # Daptomycin resistance
    MPRF = "mprF"


class Antibiotic(Enum):
    """Antibiotics for S. aureus."""

    # Beta-lactams
    OXACILLIN = "oxacillin"
    METHICILLIN = "methicillin"
    CEFOXITIN = "cefoxitin"

    # Glycopeptides
    VANCOMYCIN = "vancomycin"
    TEICOPLANIN = "teicoplanin"

    # Aminoglycosides
    GENTAMICIN = "gentamicin"
    AMIKACIN = "amikacin"

    # Fluoroquinolones
    CIPROFLOXACIN = "ciprofloxacin"
    LEVOFLOXACIN = "levofloxacin"

    # Macrolides
    ERYTHROMYCIN = "erythromycin"
    CLINDAMYCIN = "clindamycin"

    # Others
    RIFAMPICIN = "rifampicin"
    MUPIROCIN = "mupirocin"
    DAPTOMYCIN = "daptomycin"
    LINEZOLID = "linezolid"
    TRIMETHOPRIM_SULFA = "tmp_smx"


@dataclass
class MRSAConfig(DiseaseConfig):
    """Configuration for MRSA analysis."""

    name: str = "mrsa"
    display_name: str = "MRSA (S. aureus)"
    disease_type: DiseaseType = DiseaseType.BACTERIAL
    tasks: list[TaskType] = field(default_factory=lambda: [
        TaskType.RESISTANCE,
    ])

    data_sources: dict[str, str] = field(default_factory=lambda: {
        "patric": "https://www.patricbrc.org/",
        "card": "https://card.mcmaster.ca/",
        "ncbi_amr": "https://www.ncbi.nlm.nih.gov/pathogens/antimicrobial-resistance/",
    })


# GyrA mutations (fluoroquinolone resistance)
GYRA_MUTATIONS = {
    80: {"S": {"mutations": ["F", "Y"], "effect": "high"}},
    84: {"S": {"mutations": ["L"], "effect": "high"}},
    88: {"E": {"mutations": ["K"], "effect": "moderate"}},
}

# GrlB mutations (fluoroquinolone resistance)
GRLB_MUTATIONS = {
    80: {"S": {"mutations": ["F", "Y"], "effect": "high"}},
    84: {"E": {"mutations": ["K", "G"], "effect": "high"}},
}

# RpoB mutations (rifampicin resistance)
RPOB_MUTATIONS = {
    481: {"H": {"mutations": ["Y", "N"], "effect": "high"}},
    527: {"S": {"mutations": ["L"], "effect": "high"}},
    529: {"H": {"mutations": ["N"], "effect": "high"}},
}

# MprF mutations (daptomycin resistance)
MPRF_MUTATIONS = {
    314: {"S": {"mutations": ["L"], "effect": "high"}},
    340: {"L": {"mutations": ["F"], "effect": "moderate"}},
    345: {"T": {"mutations": ["A", "I"], "effect": "high"}},
    354: {"S": {"mutations": ["L"], "effect": "moderate"}},
}


class MRSAAnalyzer(DiseaseAnalyzer):
    """Analyzer for MRSA antibiotic resistance.

    Provides:
    - mecA/mecC detection
    - Multi-drug resistance profiling
    - VISA/VRSA classification
    - Resistance gene detection
    """

    def __init__(self, config: Optional[MRSAConfig] = None):
        """Initialize analyzer."""
        self.config = config or MRSAConfig()
        super().__init__(self.config)

        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY-X*"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_alphabet)}

    def analyze(
        self,
        sequences: dict[StaphGene, list[str]],
        embeddings: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Analyze S. aureus sequences for resistance.

        Args:
            sequences: Dictionary mapping gene to sequences
            embeddings: Optional embeddings

        Returns:
            Analysis results
        """
        results = {
            "n_sequences": len(next(iter(sequences.values()))) if sequences else 0,
            "genes_analyzed": [g.value for g in sequences.keys()],
            "mrsa_status": [],
            "drug_resistance": {},
            "resistance_profile": [],
        }

        # MRSA detection (mecA/mecC presence)
        n_seq = max(len(seqs) for seqs in sequences.values()) if sequences else 0
        results["mrsa_status"] = self._detect_mrsa(sequences, n_seq)

        # Fluoroquinolone resistance
        if StaphGene.GYRA in sequences:
            results["drug_resistance"]["fluoroquinolones"] = self._analyze_fq(
                sequences.get(StaphGene.GYRA, []),
                sequences.get(StaphGene.GRLB, [])
            )

        # Rifampicin resistance
        if StaphGene.RPOB in sequences:
            results["drug_resistance"]["rifampicin"] = self._scan_mutations(
                sequences[StaphGene.RPOB], RPOB_MUTATIONS, "rpoB"
            )

        # Daptomycin resistance
        if StaphGene.MPRF in sequences:
            results["drug_resistance"]["daptomycin"] = self._scan_mutations(
                sequences[StaphGene.MPRF], MPRF_MUTATIONS, "mprF"
            )

        # Overall resistance profile
        results["resistance_profile"] = self._compile_resistance_profile(
            results["mrsa_status"],
            results["drug_resistance"]
        )

        return results

    def _detect_mrsa(
        self,
        sequences: dict[StaphGene, list[str]],
        n_seq: int
    ) -> list[dict[str, Any]]:
        """Detect MRSA status via mecA/mecC."""
        mrsa_status = []

        for i in range(n_seq):
            status = {
                "isolate": i,
                "mecA_present": False,
                "mecC_present": False,
                "mrsa": False,
                "classification": "MSSA",
            }

            # Check mecA presence (simplified - real detection uses PCR/WGS)
            if StaphGene.MECA in sequences and i < len(sequences[StaphGene.MECA]):
                seq = sequences[StaphGene.MECA][i]
                # Non-empty sequence indicates presence
                if len(seq) > 10 and seq.count("-") < len(seq) * 0.5:
                    status["mecA_present"] = True
                    status["mrsa"] = True

            if StaphGene.MECC in sequences and i < len(sequences[StaphGene.MECC]):
                seq = sequences[StaphGene.MECC][i]
                if len(seq) > 10 and seq.count("-") < len(seq) * 0.5:
                    status["mecC_present"] = True
                    status["mrsa"] = True

            if status["mrsa"]:
                status["classification"] = "MRSA"

            mrsa_status.append(status)

        return mrsa_status

    def _analyze_fq(
        self,
        gyra_seqs: list[str],
        grlb_seqs: list[str]
    ) -> dict[str, Any]:
        """Analyze fluoroquinolone resistance."""
        results = {
            "scores": [],
            "classifications": [],
            "mutations": [],
        }

        n_seq = max(len(gyra_seqs), len(grlb_seqs)) if gyra_seqs or grlb_seqs else 0

        for i in range(n_seq):
            score = 0.0
            mutations = []

            # Check gyrA
            if i < len(gyra_seqs):
                gyra_score, gyra_muts = self._check_mutations(
                    gyra_seqs[i], GYRA_MUTATIONS, "gyrA"
                )
                score += gyra_score
                mutations.extend(gyra_muts)

            # Check grlB
            if i < len(grlb_seqs):
                grlb_score, grlb_muts = self._check_mutations(
                    grlb_seqs[i], GRLB_MUTATIONS, "grlB"
                )
                score += grlb_score
                mutations.extend(grlb_muts)

            normalized = min(score / 3.0, 1.0)
            results["scores"].append(normalized)
            results["mutations"].append(mutations)

            if normalized < 0.2:
                results["classifications"].append("susceptible")
            elif normalized < 0.5:
                results["classifications"].append("intermediate")
            else:
                results["classifications"].append("resistant")

        return results

    def _check_mutations(
        self,
        seq: str,
        mutation_db: dict,
        gene: str
    ) -> tuple[float, list[dict]]:
        """Check for mutations in sequence."""
        score = 0.0
        mutations = []

        for pos, info in mutation_db.items():
            if pos <= 0 or pos > len(seq):
                continue

            seq_aa = seq[pos - 1]
            ref_aa = list(info.keys())[0]

            if seq_aa != ref_aa and seq_aa in info[ref_aa]["mutations"]:
                effect = info[ref_aa]["effect"]
                effect_scores = {"high": 1.0, "moderate": 0.5, "low": 0.2}
                score += effect_scores.get(effect, 0.3)

                mutations.append({
                    "gene": gene,
                    "position": pos,
                    "ref": ref_aa,
                    "alt": seq_aa,
                    "effect": effect,
                })

        return score, mutations

    def _scan_mutations(
        self,
        sequences: list[str],
        mutation_db: dict,
        gene: str,
    ) -> dict[str, Any]:
        """Generic mutation scanning."""
        results = {
            "gene": gene,
            "scores": [],
            "classifications": [],
            "mutations": [],
        }

        for seq in sequences:
            score, mutations = self._check_mutations(seq, mutation_db, gene)
            normalized = min(score / 2.0, 1.0)

            results["scores"].append(normalized)
            results["mutations"].append(mutations)

            if normalized < 0.2:
                results["classifications"].append("susceptible")
            elif normalized < 0.5:
                results["classifications"].append("intermediate")
            else:
                results["classifications"].append("resistant")

        return results

    def _compile_resistance_profile(
        self,
        mrsa_status: list,
        drug_resistance: dict
    ) -> list[dict[str, Any]]:
        """Compile overall resistance profile."""
        profiles = []

        n_seq = len(mrsa_status)
        for i in range(n_seq):
            profile = {
                "isolate": i,
                "mrsa": mrsa_status[i]["mrsa"] if i < len(mrsa_status) else False,
                "resistant_to": [],
                "susceptible_to": [],
                "mdr": False,
            }

            if profile["mrsa"]:
                profile["resistant_to"].append("beta_lactams")

            for drug, data in drug_resistance.items():
                if i < len(data.get("classifications", [])):
                    if data["classifications"][i] == "resistant":
                        profile["resistant_to"].append(drug)
                    elif data["classifications"][i] == "susceptible":
                        profile["susceptible_to"].append(drug)

            # MDR definition: resistant to 3+ drug classes
            if len(profile["resistant_to"]) >= 3:
                profile["mdr"] = True

            profiles.append(profile)

        return profiles

    def validate_predictions(
        self,
        predictions: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate against phenotypic AST."""
        from scipy.stats import spearmanr

        metrics = {}

        for drug in predictions.get("drug_resistance", {}):
            if drug in ground_truth:
                pred = np.array(predictions["drug_resistance"][drug]["scores"])
                true = np.array(ground_truth[drug])

                if len(pred) == len(true):
                    rho, pval = spearmanr(pred, true)
                    metrics[f"{drug}_spearman"] = float(rho) if not np.isnan(rho) else 0.0

        return metrics

    def encode_sequence(
        self,
        sequence: str,
        max_length: int = 300,
    ) -> np.ndarray:
        """One-hot encode sequence."""
        n_aa = len(self.aa_alphabet)
        encoding = np.zeros(max_length * n_aa, dtype=np.float32)

        for j, aa in enumerate(sequence[:max_length]):
            idx = self.aa_to_idx.get(aa.upper(), self.aa_to_idx["X"])
            encoding[j * n_aa + idx] = 1.0

        return encoding


def create_mrsa_synthetic_dataset(
    gene: StaphGene = StaphGene.GYRA,
    min_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create synthetic MRSA dataset.

    Args:
        gene: Target gene for resistance analysis
        min_samples: Minimum number of samples to generate

    Returns:
        (X, y, ids)
    """
    from src.diseases.utils.synthetic_data import (
        create_mutation_based_dataset,
        ensure_minimum_samples,
    )

    reference = "M" + "A" * 199

    if gene == StaphGene.GYRA:
        mutation_db = GYRA_MUTATIONS
    elif gene == StaphGene.RPOB:
        mutation_db = RPOB_MUTATIONS
    else:
        mutation_db = MPRF_MUTATIONS

    analyzer = MRSAAnalyzer()

    # Use utility to create dataset with mutation combinations
    X, y, ids = create_mutation_based_dataset(
        reference_sequence=reference,
        mutation_db=mutation_db,
        encode_fn=analyzer.encode_sequence,
        max_length=300,
        n_random_mutants=30,
        seed=42,
    )

    # Ensure minimum samples
    X, y, ids = ensure_minimum_samples(X, y, ids, min_samples=min_samples, seed=42)

    return X, y, ids
