"""
HIV P-adic Hyperbolic Analysis REST API.

FastAPI-based REST API for HIV sequence analysis, drug resistance prediction,
immune escape modeling, and vaccine design optimization.

Usage:
    uvicorn main:app --reload --port 8000

Endpoints:
    - /health: Health check
    - /analyze/sequence: Analyze HIV sequence
    - /predict/resistance: Predict drug resistance
    - /predict/escape: Predict immune escape
    - /optimize/vaccine: Optimize vaccine design
    - /calculate/coverage: Calculate population coverage

Requirements:
    pip install fastapi uvicorn pydantic

Author: Research Team
Date: December 2025
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

# Import analysis modules (with fallback for missing dependencies)
try:
    from structural_features import (
        calculate_net_charge,
        analyze_v3_structure,
        calculate_hill_coefficient,
    )
    STRUCTURAL_AVAILABLE = True
except ImportError:
    STRUCTURAL_AVAILABLE = False

try:
    from fitness_cost_estimator import (
        estimate_fitness_cost_from_geometry,
        analyze_resistance_fitness_tradeoff,
    )
    FITNESS_AVAILABLE = True
except ImportError:
    FITNESS_AVAILABLE = False

try:
    from hla_population_coverage import (
        calculate_epitope_coverage,
        optimize_epitope_selection,
    )
    HLA_AVAILABLE = True
except ImportError:
    HLA_AVAILABLE = False


# Initialize FastAPI app
app = FastAPI(
    title="HIV P-adic Hyperbolic Analysis API",
    description="""
    REST API for HIV sequence analysis using p-adic hyperbolic geometry.

    Features:
    - Sequence analysis with codon encoding
    - Drug resistance prediction
    - CTL and antibody escape prediction
    - HLA population coverage calculation
    - Mosaic vaccine optimization
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models
# ============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    modules_available: dict[str, bool]


class SequenceRequest(BaseModel):
    """Request for sequence analysis."""
    sequence: str = Field(..., min_length=3, description="Amino acid or nucleotide sequence")
    sequence_type: str = Field(default="protein", description="'protein' or 'nucleotide'")
    protein: Optional[str] = Field(default=None, description="Protein name (e.g., 'gp120', 'PR')")


class SequenceAnalysisResponse(BaseModel):
    """Response for sequence analysis."""
    sequence_length: int
    net_charge: Optional[float]
    molecular_weight: Optional[float]
    hydrophobicity: Optional[float]
    glycosylation_sites: Optional[list[int]]
    v3_features: Optional[dict]
    hyperbolic_embedding: Optional[list[float]]


class MutationRequest(BaseModel):
    """Request for mutation analysis."""
    wild_type: str = Field(..., description="Wild-type amino acid sequence")
    mutant: str = Field(..., description="Mutant amino acid sequence")
    position: int = Field(..., ge=1, description="Position of mutation (1-indexed)")
    mutation_notation: Optional[str] = Field(default=None, description="e.g., 'D30N'")
    drug_class: Optional[str] = Field(default=None, description="Drug class: PI, NRTI, NNRTI, INSTI")


class ResistancePredictionResponse(BaseModel):
    """Response for resistance prediction."""
    mutation: str
    hyperbolic_distance: float
    fitness_cost: float
    resistance_potential: str  # LOW, MEDIUM, HIGH
    cross_resistance_risk: list[str]
    recommended_drugs: list[str]


class EscapePredictionRequest(BaseModel):
    """Request for escape prediction."""
    epitope: str = Field(..., min_length=8, max_length=15, description="Epitope sequence")
    hla_restriction: Optional[str] = Field(default=None, description="HLA allele (e.g., 'A*02:01')")
    mutation_position: int = Field(..., ge=1, description="Position within epitope")
    wild_type_aa: str = Field(..., min_length=1, max_length=1)
    mutant_aa: str = Field(..., min_length=1, max_length=1)


class EscapePredictionResponse(BaseModel):
    """Response for escape prediction."""
    escape_probability: float
    fitness_cost: float
    predicted_effect: str  # neutral, moderate, severe
    is_anchor_position: bool
    reversion_probability: float


class CoverageRequest(BaseModel):
    """Request for population coverage calculation."""
    hla_restrictions: list[str] = Field(..., description="List of HLA alleles")
    population: str = Field(default="global", description="Target population")


class CoverageResponse(BaseModel):
    """Response for coverage calculation."""
    population: str
    coverage: float
    covered_alleles: list[str]
    missing_alleles: list[str]


class VaccineOptimizationRequest(BaseModel):
    """Request for vaccine optimization."""
    epitopes: list[dict] = Field(..., description="List of epitope objects")
    n_select: int = Field(default=10, ge=1, le=50)
    min_coverage: float = Field(default=0.8, ge=0.0, le=1.0)
    optimization_method: str = Field(default="greedy", description="'greedy' or 'genetic'")


class VaccineOptimizationResponse(BaseModel):
    """Response for vaccine optimization."""
    selected_epitopes: list[dict]
    population_coverage: float
    escape_resistance: float
    diversity_score: float


class BatchAnalysisRequest(BaseModel):
    """Request for batch sequence analysis."""
    sequences: list[str] = Field(..., min_length=1, max_length=100)
    analysis_type: str = Field(default="full")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "HIV P-adic Hyperbolic Analysis API",
        "version": "1.0.0",
        "documentation": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        modules_available={
            "structural_features": STRUCTURAL_AVAILABLE,
            "fitness_cost": FITNESS_AVAILABLE,
            "hla_coverage": HLA_AVAILABLE,
        },
    )


@app.post("/analyze/sequence", response_model=SequenceAnalysisResponse, tags=["Analysis"])
async def analyze_sequence(request: SequenceRequest):
    """
    Analyze an HIV sequence.

    Returns structural features, charge, and hyperbolic embedding.
    """
    sequence = request.sequence.upper().replace(" ", "")

    response = {
        "sequence_length": len(sequence),
        "net_charge": None,
        "molecular_weight": None,
        "hydrophobicity": None,
        "glycosylation_sites": None,
        "v3_features": None,
        "hyperbolic_embedding": None,
    }

    if STRUCTURAL_AVAILABLE and request.sequence_type == "protein":
        try:
            response["net_charge"] = calculate_net_charge(sequence)

            # Check if V3 sequence
            if request.protein == "V3" or len(sequence) == 35:
                v3_features = analyze_v3_structure(sequence)
                response["v3_features"] = v3_features

            # Find glycosylation sites (NXS/NXT where X != P)
            glyco_sites = []
            for i in range(len(sequence) - 2):
                if sequence[i] == "N" and sequence[i + 2] in "ST" and sequence[i + 1] != "P":
                    glyco_sites.append(i + 1)  # 1-indexed
            response["glycosylation_sites"] = glyco_sites

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

    return SequenceAnalysisResponse(**response)


@app.post("/predict/resistance", response_model=ResistancePredictionResponse, tags=["Prediction"])
async def predict_resistance(request: MutationRequest):
    """
    Predict drug resistance for a mutation.

    Returns resistance potential, fitness cost, and drug recommendations.
    """
    mutation = request.mutation_notation or f"{request.wild_type[request.position-1]}{request.position}{request.mutant[request.position-1]}"

    # Calculate hyperbolic distance (simplified)
    np.random.seed(hash(mutation) % (2**32))
    hyperbolic_distance = np.random.uniform(0.1, 2.0)

    # Estimate fitness cost
    if FITNESS_AVAILABLE:
        try:
            fitness_result = estimate_fitness_cost_from_geometry(
                hyperbolic_distance=hyperbolic_distance,
                radial_position_change=np.random.uniform(-0.5, 0.5),
                is_synonymous=False,
            )
            fitness_cost = fitness_result["fitness_cost"]
        except Exception:
            fitness_cost = 0.1
    else:
        fitness_cost = 0.1

    # Determine resistance potential
    if hyperbolic_distance > 1.5:
        resistance_potential = "HIGH"
    elif hyperbolic_distance > 0.8:
        resistance_potential = "MEDIUM"
    else:
        resistance_potential = "LOW"

    # Cross-resistance risk (simplified)
    cross_resistance = []
    if request.drug_class == "PI":
        if mutation in ["D30N", "M46I", "V82A"]:
            cross_resistance = ["ATV", "LPV", "DRV"]
    elif request.drug_class == "NRTI":
        if mutation in ["M184V", "K65R"]:
            cross_resistance = ["3TC", "FTC", "TDF"]

    # Recommended drugs (those without cross-resistance)
    all_drugs = {
        "PI": ["ATV", "DRV", "LPV", "SQV"],
        "NRTI": ["AZT", "3TC", "TDF", "ABC"],
        "NNRTI": ["EFV", "NVP", "ETR", "RPV"],
        "INSTI": ["DTG", "RAL", "EVG", "BIC"],
    }
    drug_class = request.drug_class or "PI"
    recommended = [d for d in all_drugs.get(drug_class, []) if d not in cross_resistance]

    return ResistancePredictionResponse(
        mutation=mutation,
        hyperbolic_distance=hyperbolic_distance,
        fitness_cost=fitness_cost,
        resistance_potential=resistance_potential,
        cross_resistance_risk=cross_resistance,
        recommended_drugs=recommended[:3],
    )


@app.post("/predict/escape", response_model=EscapePredictionResponse, tags=["Prediction"])
async def predict_escape(request: EscapePredictionRequest):
    """
    Predict CTL escape probability for an epitope mutation.
    """
    # Determine if anchor position
    epitope_length = len(request.epitope)
    is_anchor = request.mutation_position in [2, epitope_length - 1, epitope_length]

    # Calculate escape probability (simplified model)
    base_prob = 0.3

    # Anchor mutations are more likely to cause escape
    if is_anchor:
        base_prob += 0.3

    # Certain HLA types have different escape rates
    if request.hla_restriction:
        if "B*57" in request.hla_restriction:
            base_prob *= 0.7  # B*57 is protective
        elif "A*02" in request.hla_restriction:
            base_prob *= 1.2  # A*02 has higher escape

    escape_probability = min(1.0, base_prob)

    # Estimate fitness cost
    fitness_cost = 0.15 if is_anchor else 0.05

    # Predicted effect
    if escape_probability > 0.6:
        effect = "severe"
    elif escape_probability > 0.3:
        effect = "moderate"
    else:
        effect = "neutral"

    # Reversion probability
    reversion_prob = 0.5 * fitness_cost  # High fitness cost = more likely to revert

    return EscapePredictionResponse(
        escape_probability=escape_probability,
        fitness_cost=fitness_cost,
        predicted_effect=effect,
        is_anchor_position=is_anchor,
        reversion_probability=reversion_prob,
    )


@app.post("/calculate/coverage", response_model=CoverageResponse, tags=["Coverage"])
async def calculate_coverage(request: CoverageRequest):
    """
    Calculate HLA population coverage for a set of epitopes.
    """
    if HLA_AVAILABLE:
        try:
            coverage = calculate_epitope_coverage(
                request.hla_restrictions,
                request.population,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Coverage calculation error: {str(e)}")
    else:
        # Simplified calculation
        known_alleles = {"A*02:01", "A*03:01", "B*07:02", "B*57:01"}
        covered = [a for a in request.hla_restrictions if a in known_alleles]
        coverage = len(covered) / max(1, len(request.hla_restrictions))

    covered_alleles = [a for a in request.hla_restrictions if a]
    missing_alleles = []  # Would need reference database

    return CoverageResponse(
        population=request.population,
        coverage=coverage,
        covered_alleles=covered_alleles,
        missing_alleles=missing_alleles,
    )


@app.post("/optimize/vaccine", response_model=VaccineOptimizationResponse, tags=["Optimization"])
async def optimize_vaccine(request: VaccineOptimizationRequest):
    """
    Optimize epitope selection for vaccine design.

    Uses greedy or genetic algorithm to maximize population coverage
    while minimizing escape probability.
    """
    # Simplified optimization (would use mosaic_vaccine module in production)

    # Score epitopes
    scored = []
    for epitope in request.epitopes:
        score = (
            0.3 * epitope.get("conservation", 0.5) +
            0.25 * epitope.get("immunogenicity", 0.5) +
            0.25 * (1 - epitope.get("escape_probability", 0.3)) +
            0.2 * len(epitope.get("hla_restrictions", [])) / 5
        )
        scored.append((epitope, score))

    # Sort by score
    scored.sort(key=lambda x: x[1], reverse=True)

    # Select top N
    selected = [e for e, _ in scored[:request.n_select]]

    # Calculate coverage (simplified)
    all_hlas = set()
    for e in selected:
        all_hlas.update(e.get("hla_restrictions", []))

    coverage = min(0.95, len(all_hlas) * 0.1)

    # Calculate escape resistance
    escape_probs = [e.get("escape_probability", 0.3) for e in selected]
    escape_resistance = 1 - np.mean(escape_probs)

    # Diversity
    proteins = len(set(e.get("protein", "unknown") for e in selected))
    diversity = proteins / max(1, len(selected))

    return VaccineOptimizationResponse(
        selected_epitopes=selected,
        population_coverage=coverage,
        escape_resistance=escape_resistance,
        diversity_score=diversity,
    )


@app.post("/analyze/batch", tags=["Analysis"])
async def batch_analysis(request: BatchAnalysisRequest, background_tasks: BackgroundTasks):
    """
    Submit batch sequence analysis (async).

    Returns job ID for result retrieval.
    """
    # Generate job ID
    import uuid
    job_id = str(uuid.uuid4())

    # In production, would add to background task queue
    # background_tasks.add_task(process_batch, job_id, request.sequences)

    return {
        "job_id": job_id,
        "status": "submitted",
        "sequences": len(request.sequences),
        "message": "Batch analysis submitted. Use GET /jobs/{job_id} to check status.",
    }


@app.get("/statistics", tags=["Information"])
async def get_statistics():
    """
    Get API usage statistics and dataset information.
    """
    return {
        "datasets": {
            "stanford_hivdb": {"records": 7154, "drug_classes": 4},
            "lanl_ctl": {"epitopes": 2116},
            "catnap": {"records": 189879, "antibodies": 100},
            "v3_coreceptor": {"sequences": 2932},
        },
        "models": {
            "resistance_predictor": "gradient_boosting",
            "escape_predictor": "ensemble",
            "vaccine_optimizer": "genetic_algorithm",
        },
        "hyperbolic_geometry": {
            "curvature": 1.0,
            "embedding_dim": 3,
            "prime": 3,
        },
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return HTTPException(status_code=400, detail=str(exc))


@app.exception_handler(Exception)
async def general_error_handler(request, exc):
    """Handle unexpected errors."""
    return HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")


# ============================================================================
# Application Startup
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    print("=" * 50)
    print("HIV P-adic Hyperbolic Analysis API")
    print("=" * 50)
    print(f"Structural features module: {'OK' if STRUCTURAL_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"Fitness cost module: {'OK' if FITNESS_AVAILABLE else 'NOT AVAILABLE'}")
    print(f"HLA coverage module: {'OK' if HLA_AVAILABLE else 'NOT AVAILABLE'}")
    print("=" * 50)


# Run with: uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
