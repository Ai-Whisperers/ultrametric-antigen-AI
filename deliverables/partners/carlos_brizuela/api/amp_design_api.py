"""AMP Mechanism-Based Design API - FastAPI REST endpoints.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! DISCLAIMER: PREMATURE MOCK - NOT PRODUCTION READY                 !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!                                                                   !!
!! This API was created BEFORE completing critical validation steps: !!
!!                                                                   !!
!! MISSING VALIDATION:                                               !!
!! - C5 hold-out generalization test NOT RUN                         !!
!! - Signal may not generalize to unseen pathogens                   !!
!! - R2 constraint (hold-out testing) not satisfied                  !!
!!                                                                   !!
!! CURRENT STATUS:                                                   !!
!! - C3 signal survived seed-artifact falsification                  !!
!! - C5 script exists but was NEVER EXECUTED                         !!
!! - Findings are PARTIALLY VALIDATED, not fully validated           !!
!!                                                                   !!
!! DO NOT USE FOR:                                                   !!
!! - Production deployment                                           !!
!! - Clinical decision support                                       !!
!! - Publication claims                                              !!
!!                                                                   !!
!! NEXT STEPS REQUIRED:                                              !!
!! 1. Run C5 hold-out generalization test                            !!
!! 2. Train PeptideVAE checkpoint (blocking item)                    !!
!! 3. Connect to Foundation Encoder pipeline                         !!
!! 4. Remove this disclaimer only after full validation              !!
!!                                                                   !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Exposes PARTIALLY-validated findings from P1 investigation:
- Mechanism classification (detergent vs barrel_stave)
- Regime routing based on arrow-flip thresholds
- Design rules per target pathogen
- Pathogen ranking by predicted efficacy

NOTE: N-terminal cationic dipeptide hypothesis was FALSIFIED.
Design rules do NOT include N-terminal cationic recommendations.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .mechanism_service import get_mechanism_service
from .models import (
    DesignRulesRequest,
    DesignRulesResponse,
    FingerprintResponse,
    HealthResponse,
    MechanismClassifyBatchRequest,
    MechanismClassifyBatchResponse,
    MechanismClassifyRequest,
    MechanismClassifyResponse,
    PathogenInfo,
    PathogenRanking,
    PathogenRankBatchRequest,
    PathogenRankBatchResponse,
    PathogenRankRequest,
    PathogenRankResponse,
    RegimeRouteRequest,
    RegimeRouteResponse,
    SequenceRules,
    ThresholdInfo,
    ThresholdsResponse,
)

# API metadata
API_VERSION = "1.0.0"
API_TITLE = "AMP Mechanism-Based Design API"
API_DESCRIPTION = """
## !! PREMATURE MOCK - NOT PRODUCTION READY !!

**WARNING**: This API was created BEFORE completing C5 hold-out generalization testing.
Findings are PARTIALLY VALIDATED. Do not use for production, clinical decisions, or publication claims.

---

Classical REST API exposing PARTIALLY-validated mechanism-based AMP design findings.

## Key Features

- **Mechanism Classification**: Infer killing mechanism from peptide properties
- **Regime Routing**: Determine if cluster-conditional or global prediction applies
- **Design Rules**: Get actionable design rules for target pathogens
- **Pathogen Ranking**: Rank pathogens by predicted efficacy for a sequence

## Validated Thresholds

- **Arrow-flip**: hydrophobicity > 0.107 triggers cluster-conditional regime
- **Signal clusters**: 1, 3, 4 show pathogen-specific patterns

## Known Limitations

- Enterobacteriaceae consistently fails with current mechanisms
- N-terminal cationic hypothesis was FALSIFIED (not included in rules)
"""

# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HEALTH CHECK
# =============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service=API_TITLE,
        version=API_VERSION,
    )


# =============================================================================
# MECHANISM CLASSIFICATION
# =============================================================================


@app.post(
    "/classify/mechanism",
    response_model=MechanismClassifyResponse,
    tags=["Classification"],
)
async def classify_mechanism(request: MechanismClassifyRequest) -> MechanismClassifyResponse:
    """Classify likely AMP killing mechanism based on peptide properties.

    Mechanisms include:
    - **barrel_stave**: Forms transmembrane pores (15-25 AA)
    - **carpet**: Covers membrane surface (10-20 AA)
    - **toroidal**: Creates toroidal pores (20-30 AA)
    - **detergent**: Micelle-like disruption (8-15 AA)
    """
    service = get_mechanism_service()
    result = service.classify_mechanism(
        length=request.length,
        hydrophobicity=request.hydrophobicity,
        net_charge=request.net_charge,
    )

    return MechanismClassifyResponse(
        mechanism=result["mechanism"],
        confidence=result["confidence"],
        description=result["description"],
        cluster_id=result["cluster_id"],
        has_pathogen_signal=result["has_pathogen_signal"],
        mechanism_scores=result.get("mechanism_scores"),
    )


@app.post(
    "/classify/mechanism/batch",
    response_model=MechanismClassifyBatchResponse,
    tags=["Classification"],
)
async def classify_mechanism_batch(
    request: MechanismClassifyBatchRequest,
) -> MechanismClassifyBatchResponse:
    """Batch mechanism classification for multiple peptides."""
    service = get_mechanism_service()
    results = []

    for peptide in request.peptides:
        result = service.classify_mechanism(
            length=peptide.length,
            hydrophobicity=peptide.hydrophobicity,
            net_charge=peptide.net_charge,
        )
        results.append(
            MechanismClassifyResponse(
                mechanism=result["mechanism"],
                confidence=result["confidence"],
                description=result["description"],
                cluster_id=result["cluster_id"],
                has_pathogen_signal=result["has_pathogen_signal"],
                mechanism_scores=result.get("mechanism_scores"),
            )
        )

    return MechanismClassifyBatchResponse(
        results=results,
        processed=len(results),
    )


# =============================================================================
# REGIME ROUTING
# =============================================================================


@app.post(
    "/route/regime",
    response_model=RegimeRouteResponse,
    tags=["Routing"],
)
async def route_regime(request: RegimeRouteRequest) -> RegimeRouteResponse:
    """Determine prediction regime based on arrow-flip threshold.

    - **CLUSTER_CONDITIONAL**: Use cluster-specific model (hydrophobicity > 0.107)
    - **GLOBAL**: Use global model (hydrophobicity <= 0.107)
    """
    service = get_mechanism_service()
    result = service.route_regime(hydrophobicity=request.hydrophobicity)

    return RegimeRouteResponse(
        regime=result["regime"],
        threshold_used=result["threshold_used"],
        expected_separation=result["expected_separation"],
        rationale=result["rationale"],
    )


# =============================================================================
# DESIGN RULES
# =============================================================================


@app.post(
    "/design/rules",
    response_model=DesignRulesResponse,
    tags=["Design"],
)
async def get_design_rules(request: DesignRulesRequest) -> DesignRulesResponse:
    """Get actionable design rules for a target pathogen.

    Supported pathogens:
    - A_baumannii (critical priority)
    - P_aeruginosa (critical priority, best supported)
    - Enterobacteriaceae (critical priority, known failure case)
    - S_aureus (high priority)
    - H_pylori (medium priority)
    """
    service = get_mechanism_service()
    result = service.get_design_rules(target_pathogen=request.target_pathogen)

    # Handle error case
    if "error" in result:
        return DesignRulesResponse(
            target_pathogen=result["target_pathogen"],
            error=result["error"],
            rationale="",
            confidence="",
        )

    # Build pathogen info
    pathogen_info = None
    if result.get("pathogen_info"):
        pi = result["pathogen_info"]
        pathogen_info = PathogenInfo(
            full_name=pi["full_name"],
            gram=pi["gram"],
            lps_abundance=pi["lps_abundance"],
            membrane_charge=pi["membrane_charge"],
            priority=pi["priority"],
        )

    # Build sequence rules
    sequence_rules = None
    if result.get("sequence_rules"):
        sr = result["sequence_rules"]
        sequence_rules = SequenceRules(
            cationic_fraction=sr.get("cationic_fraction"),
            hydrophobicity=sr.get("hydrophobicity"),
            note=sr.get("note"),
        )

    return DesignRulesResponse(
        target_pathogen=result["target_pathogen"],
        pathogen_info=pathogen_info,
        recommended_length=result.get("recommended_length"),
        recommended_mechanism=result.get("recommended_mechanism"),
        sequence_rules=sequence_rules,
        rationale=result["rationale"],
        confidence=result["confidence"],
        warning=result.get("warning"),
    )


# =============================================================================
# PATHOGEN RANKING
# =============================================================================


@app.post(
    "/predict/pathogen-rank",
    response_model=PathogenRankResponse,
    tags=["Prediction"],
)
async def predict_pathogen_rank(request: PathogenRankRequest) -> PathogenRankResponse:
    """Rank pathogens by predicted efficacy for a given sequence.

    Returns mechanism classification and pathogen ranking based on
    validated cluster-pathogen relationships.
    """
    service = get_mechanism_service()
    result = service.rank_pathogens(sequence=request.sequence)

    rankings = [
        PathogenRanking(
            pathogen=r["pathogen"],
            relative_efficacy=r["relative_efficacy"],
            confidence=r["confidence"],
        )
        for r in result["pathogen_ranking"]
    ]

    return PathogenRankResponse(
        sequence=result["sequence"],
        length=result["length"],
        net_charge=result["net_charge"],
        hydrophobicity=result["hydrophobicity"],
        cationic_fraction=result["cationic_fraction"],
        cluster_id=result["cluster_id"],
        mechanism=result["mechanism"],
        mechanism_confidence=result["mechanism_confidence"],
        pathogen_ranking=rankings,
    )


@app.post(
    "/predict/pathogen-rank/batch",
    response_model=PathogenRankBatchResponse,
    tags=["Prediction"],
)
async def predict_pathogen_rank_batch(
    request: PathogenRankBatchRequest,
) -> PathogenRankBatchResponse:
    """Batch pathogen ranking for multiple sequences."""
    service = get_mechanism_service()
    results = []

    for sequence in request.sequences:
        result = service.rank_pathogens(sequence=sequence)
        rankings = [
            PathogenRanking(
                pathogen=r["pathogen"],
                relative_efficacy=r["relative_efficacy"],
                confidence=r["confidence"],
            )
            for r in result["pathogen_ranking"]
        ]
        results.append(
            PathogenRankResponse(
                sequence=result["sequence"],
                length=result["length"],
                net_charge=result["net_charge"],
                hydrophobicity=result["hydrophobicity"],
                cationic_fraction=result["cationic_fraction"],
                cluster_id=result["cluster_id"],
                mechanism=result["mechanism"],
                mechanism_confidence=result["mechanism_confidence"],
                pathogen_ranking=rankings,
            )
        )

    return PathogenRankBatchResponse(
        results=results,
        processed=len(results),
    )


# =============================================================================
# METRICS & THRESHOLDS
# =============================================================================


@app.get(
    "/thresholds",
    response_model=ThresholdsResponse,
    tags=["Metrics"],
)
async def get_thresholds() -> ThresholdsResponse:
    """Get all validated arrow-flip thresholds.

    These thresholds determine when cluster-conditional prediction
    provides better separation than global prediction.
    """
    service = get_mechanism_service()
    result = service.get_thresholds()

    thresholds = {
        name: ThresholdInfo(
            value=info["value"],
            improvement=info["improvement"],
            significant=info["significant"],
        )
        for name, info in result["arrow_flip_thresholds"].items()
    }

    return ThresholdsResponse(
        arrow_flip_thresholds=thresholds,
        primary_threshold=result["primary_threshold"],
        regime_separation=result["regime_separation"],
    )


@app.get(
    "/metrics/fingerprint",
    response_model=FingerprintResponse,
    tags=["Metrics"],
)
async def get_fingerprint() -> FingerprintResponse:
    """Get full mechanism fingerprint data.

    Returns validated cluster profiles, Gram correlations,
    mechanism inference, and design implications.
    """
    service = get_mechanism_service()
    result = service.get_fingerprint()

    if "error" in result:
        return FingerprintResponse(
            version="1.0",
            error=result["error"],
        )

    return FingerprintResponse(
        profiles=result.get("profiles"),
        gram_correlation=result.get("gram_correlation"),
        mechanism_inference=result.get("mechanism_inference"),
        design_implications=result.get("design_implications"),
        version=result.get("version", "1.0"),
        source=result.get("source"),
    )


# =============================================================================
# ENTRYPOINT
# =============================================================================


def create_app() -> FastAPI:
    """Factory function for creating the FastAPI app."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("amp_design_api:app", host="0.0.0.0", port=8080, reload=True)
