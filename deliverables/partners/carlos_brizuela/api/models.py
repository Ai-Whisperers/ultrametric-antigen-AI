"""Pydantic models for AMP Design API request/response validation.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! C5 FALSIFIED - PATHOGEN METADATA PROVIDES NO IMPROVEMENT          !!
!! Models for /design/rules and /predict/pathogen-rank are MISLEADING!!
!! See amp_design_api.py for full disclaimer.                        !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# =============================================================================
# REQUEST MODELS
# =============================================================================

class MechanismClassifyRequest(BaseModel):
    """Request for mechanism classification."""
    length: float = Field(..., ge=5, le=50, description="Peptide length in amino acids")
    hydrophobicity: float = Field(..., ge=-5, le=5, description="Mean hydrophobicity score")
    net_charge: float = Field(..., ge=-10, le=15, description="Net peptide charge")


class MechanismClassifyBatchRequest(BaseModel):
    """Batch request for mechanism classification."""
    peptides: List[MechanismClassifyRequest] = Field(..., max_length=100)


class RegimeRouteRequest(BaseModel):
    """Request for regime routing decision."""
    hydrophobicity: float = Field(..., ge=-5, le=5, description="Mean hydrophobicity score")


class DesignRulesRequest(BaseModel):
    """Request for design rules."""
    target_pathogen: str = Field(..., description="Target pathogen identifier")


class PathogenRankRequest(BaseModel):
    """Request for pathogen ranking."""
    sequence: str = Field(..., min_length=5, max_length=50, description="Amino acid sequence")


class PathogenRankBatchRequest(BaseModel):
    """Batch request for pathogen ranking."""
    sequences: List[str] = Field(..., max_length=100)


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class MechanismClassifyResponse(BaseModel):
    """Response for mechanism classification."""
    mechanism: str
    confidence: float
    description: str
    cluster_id: int
    has_pathogen_signal: bool
    mechanism_scores: Optional[Dict[str, int]] = None


class MechanismClassifyBatchResponse(BaseModel):
    """Batch response for mechanism classification."""
    results: List[MechanismClassifyResponse]
    processed: int


class RegimeRouteResponse(BaseModel):
    """Response for regime routing decision."""
    regime: str
    threshold_used: float
    expected_separation: float
    rationale: str


class SequenceRules(BaseModel):
    """Sequence design rules."""
    cationic_fraction: Optional[str] = None
    hydrophobicity: Optional[str] = None
    note: Optional[str] = None


class PathogenInfo(BaseModel):
    """Pathogen information."""
    full_name: str
    gram: str
    lps_abundance: float
    membrane_charge: float
    priority: str


class DesignRulesResponse(BaseModel):
    """Response for design rules."""
    target_pathogen: str
    pathogen_info: Optional[PathogenInfo] = None
    recommended_length: Optional[str] = None
    recommended_mechanism: Optional[List[str]] = None
    sequence_rules: Optional[SequenceRules] = None
    rationale: str
    confidence: str
    warning: Optional[str] = None
    error: Optional[str] = None


class PathogenRanking(BaseModel):
    """Single pathogen in ranking."""
    pathogen: str
    relative_efficacy: float
    confidence: str


class PathogenRankResponse(BaseModel):
    """Response for pathogen ranking."""
    sequence: str
    length: int
    net_charge: float
    hydrophobicity: float
    cationic_fraction: float
    cluster_id: int
    mechanism: str
    mechanism_confidence: float
    pathogen_ranking: List[PathogenRanking]


class PathogenRankBatchResponse(BaseModel):
    """Batch response for pathogen ranking."""
    results: List[PathogenRankResponse]
    processed: int


class ThresholdInfo(BaseModel):
    """Arrow-flip threshold information."""
    value: float
    improvement: float
    significant: bool


class ThresholdsResponse(BaseModel):
    """Response for thresholds endpoint."""
    arrow_flip_thresholds: Dict[str, ThresholdInfo]
    primary_threshold: str
    regime_separation: Dict[str, float]


class FingerprintResponse(BaseModel):
    """Response for fingerprint endpoint."""
    profiles: Optional[Dict[str, Any]] = None
    gram_correlation: Optional[Dict[str, Any]] = None
    mechanism_inference: Optional[Dict[str, Any]] = None
    design_implications: Optional[Dict[str, Any]] = None
    version: str
    source: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str
