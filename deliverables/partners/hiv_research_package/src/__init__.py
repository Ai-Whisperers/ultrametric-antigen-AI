# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""HIV Research Package - Core library.

This module provides clinical decision support tools for HIV treatment,
including transmitted drug resistance (TDR) screening and long-acting
injectable (LA) selection.

Example:
    >>> from hiv_research_package.src import TDRScreener, LASelector
    >>> screener = TDRScreener()
    >>> result = screener.screen_patient(sequence, patient_id="P001")
    >>> print(result.recommendation)
"""

from __future__ import annotations

from .models import (
    TDRResult,
    PatientData,
    LASelectionResult,
    ResistanceReport,
    DrugScore,
    MutationInfo,
    ResistanceLevel,
)
from .constants import (
    TDR_MUTATIONS,
    FIRST_LINE_DRUGS,
    FIRST_LINE_REGIMENS,
    LA_DRUGS,
    BMI_CATEGORIES,
    WHO_SDRM_NRTI,
    WHO_SDRM_NNRTI,
    WHO_SDRM_INSTI,
)
from .tdr import TDRScreener
from .la_injectable import LASelector
from .stanford_client import StanfordHIVdbClient
from .report import ClinicalReportGenerator
from .alignment import HIVSequenceAligner, AlignmentResult, MutationResult

__all__ = [
    # Models
    "TDRResult",
    "PatientData",
    "LASelectionResult",
    "ResistanceReport",
    "DrugScore",
    "MutationInfo",
    "ResistanceLevel",
    # Constants
    "TDR_MUTATIONS",
    "FIRST_LINE_DRUGS",
    "FIRST_LINE_REGIMENS",
    "LA_DRUGS",
    "BMI_CATEGORIES",
    "WHO_SDRM_NRTI",
    "WHO_SDRM_NNRTI",
    "WHO_SDRM_INSTI",
    # Main classes
    "TDRScreener",
    "LASelector",
    "StanfordHIVdbClient",
    "ClinicalReportGenerator",
    # Alignment
    "HIVSequenceAligner",
    "AlignmentResult",
    "MutationResult",
]

__version__ = "1.0.0"
