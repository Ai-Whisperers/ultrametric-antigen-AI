"""FastAPI Web Interface for Multi-Disease Drug Resistance Prediction.

Provides a REST API for drug resistance prediction across 11 disease domains:
- HIV (23 ARVs)
- SARS-CoV-2 (Paxlovid, mAbs)
- Tuberculosis (13 drugs, MDR/XDR)
- Influenza (NAIs, baloxavir)
- HCV (DAAs)
- HBV (NAs)
- Malaria (ACTs)
- MRSA (multiple antibiotics)
- Candida auris (antifungals)
- RSV (mAbs)
- Cancer (TKIs)

Usage:
    uvicorn src.api.drug_resistance_api:app --reload --port 8000

Endpoints:
    POST /predict - Predict resistance for a sequence
    POST /predict/batch - Batch prediction
    POST /predict/{disease} - Disease-specific prediction
    GET /drugs - List available drugs
    GET /diseases - List supported diseases
    GET /health - Health check
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass, field
from functools import wraps

# Add project root
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

try:
    from fastapi import FastAPI, HTTPException, Request, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from starlette.middleware.base import BaseHTTPMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")

# API Version
API_VERSION = "v1"
API_VERSION_FULL = "1.0.0"


# =============================================================================
# Rate Limiting
# =============================================================================

@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10


class RateLimiter:
    """Simple in-memory rate limiter with sliding window."""

    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def _clean_old_requests(self, client_id: str, window_seconds: int) -> None:
        """Remove requests outside the time window."""
        now = time.time()
        cutoff = now - window_seconds
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > cutoff
        ]

    def is_allowed(self, client_id: str) -> tuple[bool, dict]:
        """Check if request is allowed under rate limits.

        Returns:
            (allowed, headers) - Whether request is allowed and rate limit headers
        """
        now = time.time()

        # Clean and check minute limit
        self._clean_old_requests(client_id, 60)
        minute_count = len(self._requests[client_id])

        # Clean and check hour limit
        self._clean_old_requests(client_id, 3600)
        hour_count = len(self._requests[client_id])

        headers = {
            "X-RateLimit-Limit-Minute": str(self.config.requests_per_minute),
            "X-RateLimit-Remaining-Minute": str(max(0, self.config.requests_per_minute - minute_count)),
            "X-RateLimit-Limit-Hour": str(self.config.requests_per_hour),
            "X-RateLimit-Remaining-Hour": str(max(0, self.config.requests_per_hour - hour_count)),
        }

        if minute_count >= self.config.requests_per_minute:
            headers["Retry-After"] = "60"
            return False, headers

        if hour_count >= self.config.requests_per_hour:
            headers["Retry-After"] = "3600"
            return False, headers

        # Record this request
        self._requests[client_id].append(now)
        return True, headers


# Global rate limiter
rate_limiter = RateLimiter()

import numpy as np
import torch


# =============================================================================
# API Models
# =============================================================================

if FASTAPI_AVAILABLE:

    class SequenceInput(BaseModel):
        """Input sequence for prediction."""
        sequence: str = Field(..., description="Amino acid sequence", min_length=10)
        drug: str = Field(..., description="Drug name (e.g., AZT, LPV)")
        drug_class: Optional[str] = Field(None, description="Drug class (pi, nrti, nnrti, ini)")

    class BatchInput(BaseModel):
        """Batch of sequences for prediction."""
        sequences: List[str] = Field(..., description="List of amino acid sequences")
        drug: str = Field(..., description="Drug name")

    class PredictionOutput(BaseModel):
        """Prediction result."""
        drug: str
        resistance_score: float = Field(..., description="Resistance score (0-1)")
        confidence: Optional[float] = Field(None, description="Prediction confidence")
        interpretation: str = Field(..., description="Clinical interpretation")
        mutations_detected: Optional[List[str]] = Field(None, description="Key mutations found")

    class BatchOutput(BaseModel):
        """Batch prediction results."""
        predictions: List[PredictionOutput]
        n_sequences: int

    class DrugInfo(BaseModel):
        """Drug information."""
        name: str
        drug_class: str
        full_name: str
        key_mutations: List[str]

    class HealthResponse(BaseModel):
        """Health check response."""
        status: str
        model_loaded: bool
        version: str

    class UncertaintyOutput(BaseModel):
        """Prediction with uncertainty quantification."""
        drug: str
        mean_score: float
        std_score: float
        lower_95: float
        upper_95: float
        n_samples: int
        interpretation: str

    class CrossResistanceOutput(BaseModel):
        """Cross-resistance analysis."""
        primary_drug: str
        cross_resistance: Dict[str, float]
        high_risk_drugs: List[str]
        recommendation: str

    class ClinicalReportOutput(BaseModel):
        """Comprehensive clinical decision support report."""
        patient_id: Optional[str]
        sequence_length: int
        analysis_date: str
        drug_class_results: Dict[str, List[PredictionOutput]]
        recommended_drugs: List[str]
        avoid_drugs: List[str]
        cross_resistance_warnings: List[str]
        novel_mutations: List[str]
        overall_recommendation: str

    class NovelMutationOutput(BaseModel):
        """Novel mutation analysis."""
        position: int
        attention_score: float
        known_status: str
        drug_class: str
        recommendation: str


# =============================================================================
# Drug Database
# =============================================================================

DRUG_DATABASE = {
    # Protease Inhibitors
    "LPV": {"class": "pi", "full_name": "Lopinavir", "mutations": ["32", "47", "50", "54", "76", "82", "84"]},
    "DRV": {"class": "pi", "full_name": "Darunavir", "mutations": ["32", "47", "50", "54", "76", "84"]},
    "ATV": {"class": "pi", "full_name": "Atazanavir", "mutations": ["32", "48", "50", "54", "82", "84", "88"]},
    "NFV": {"class": "pi", "full_name": "Nelfinavir", "mutations": ["30", "46", "54", "82", "84", "88", "90"]},
    "FPV": {"class": "pi", "full_name": "Fosamprenavir", "mutations": ["32", "47", "50", "54", "76", "82", "84"]},
    "IDV": {"class": "pi", "full_name": "Indinavir", "mutations": ["32", "46", "54", "76", "82", "84"]},
    "SQV": {"class": "pi", "full_name": "Saquinavir", "mutations": ["48", "54", "82", "84", "88", "90"]},
    "TPV": {"class": "pi", "full_name": "Tipranavir", "mutations": ["33", "47", "58", "74", "82", "83", "84"]},

    # NRTIs
    "AZT": {"class": "nrti", "full_name": "Zidovudine", "mutations": ["41", "67", "70", "210", "215", "219"]},
    "D4T": {"class": "nrti", "full_name": "Stavudine", "mutations": ["41", "67", "70", "75", "210", "215", "219"]},
    "ABC": {"class": "nrti", "full_name": "Abacavir", "mutations": ["65", "74", "115", "184"]},
    "TDF": {"class": "nrti", "full_name": "Tenofovir", "mutations": ["65", "70"]},
    "DDI": {"class": "nrti", "full_name": "Didanosine", "mutations": ["65", "74"]},
    "3TC": {"class": "nrti", "full_name": "Lamivudine", "mutations": ["65", "184"]},

    # NNRTIs
    "NVP": {"class": "nnrti", "full_name": "Nevirapine", "mutations": ["100", "101", "103", "106", "181", "188", "190"]},
    "EFV": {"class": "nnrti", "full_name": "Efavirenz", "mutations": ["100", "101", "103", "106", "188", "190", "225"]},
    "ETR": {"class": "nnrti", "full_name": "Etravirine", "mutations": ["100", "101", "138", "179", "181"]},
    "DOR": {"class": "nnrti", "full_name": "Doravirine", "mutations": ["100", "101", "106", "227"]},
    "RPV": {"class": "nnrti", "full_name": "Rilpivirine", "mutations": ["100", "101", "138", "179", "181", "227"]},

    # INIs
    "RAL": {"class": "ini", "full_name": "Raltegravir", "mutations": ["66", "92", "140", "143", "148", "155"]},
    "EVG": {"class": "ini", "full_name": "Elvitegravir", "mutations": ["66", "92", "118", "121", "140", "143", "147", "148", "155"]},
    "DTG": {"class": "ini", "full_name": "Dolutegravir", "mutations": ["118", "140", "148", "263"]},
    "BIC": {"class": "ini", "full_name": "Bictegravir", "mutations": ["118", "140", "148", "263"]},
}


# =============================================================================
# Sequence Encoding
# =============================================================================

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}


def encode_sequence(sequence: str, expected_length: int) -> np.ndarray:
    """One-hot encode amino acid sequence."""
    n_aa = len(AA_ALPHABET)

    # Pad or truncate sequence
    if len(sequence) < expected_length:
        sequence = sequence + "-" * (expected_length - len(sequence))
    elif len(sequence) > expected_length:
        sequence = sequence[:expected_length]

    encoded = np.zeros((expected_length * n_aa,), dtype=np.float32)

    for i, aa in enumerate(sequence.upper()):
        if aa in AA_TO_IDX:
            encoded[i * n_aa + AA_TO_IDX[aa]] = 1.0
        else:
            encoded[i * n_aa + AA_TO_IDX["-"]] = 1.0

    return encoded


def interpret_resistance(score: float) -> str:
    """Interpret resistance score clinically."""
    if score < 0.3:
        return "Susceptible - No significant resistance expected"
    elif score < 0.5:
        return "Low-level resistance - May affect drug efficacy"
    elif score < 0.7:
        return "Intermediate resistance - Reduced drug efficacy likely"
    elif score < 0.9:
        return "High-level resistance - Drug not recommended"
    else:
        return "Very high resistance - Drug contraindicated"


# =============================================================================
# Mock Model (Replace with actual trained model)
# =============================================================================

class MockResistanceModel:
    """Mock model for demonstration. Replace with trained model."""

    def __init__(self):
        self.loaded = True

    def predict(self, x: torch.Tensor, drug: str) -> tuple:
        """Mock prediction."""
        # Simulate prediction based on input variance
        score = 0.3 + 0.4 * torch.sigmoid(x.sum(dim=-1) / 1000).item()
        confidence = 0.85 + 0.1 * np.random.random()
        return score, confidence


# Global model instance
model = MockResistanceModel()


# =============================================================================
# FastAPI Application
# =============================================================================

if FASTAPI_AVAILABLE:

    class RateLimitMiddleware(BaseHTTPMiddleware):
        """Middleware for rate limiting requests."""

        async def dispatch(self, request: Request, call_next: Callable):
            # Get client IP (consider X-Forwarded-For for proxies)
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()
            else:
                client_ip = request.client.host if request.client else "unknown"

            # Check rate limit
            allowed, headers = rate_limiter.is_allowed(client_ip)

            if not allowed:
                response = JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Please retry later."},
                    headers=headers,
                )
                return response

            # Process request
            response = await call_next(request)

            # Add rate limit headers to response
            for key, value in headers.items():
                response.headers[key] = value

            return response

    app = FastAPI(
        title="Drug Resistance Prediction API",
        description=(
            "P-adic VAE-based prediction of drug resistance from sequences. "
            "Supports HIV, SARS-CoV-2, TB, Influenza, HCV, HBV, Malaria, MRSA, "
            "Candida auris, RSV, and Cancer."
        ),
        version=API_VERSION_FULL,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=[
            {"name": "health", "description": "Health check endpoints"},
            {"name": "drugs", "description": "Drug database queries"},
            {"name": "prediction", "description": "Resistance prediction endpoints"},
            {"name": "clinical", "description": "Clinical decision support"},
            {"name": "diseases", "description": "Multi-disease support"},
        ],
    )

    # Rate limiting middleware
    app.add_middleware(RateLimitMiddleware)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint - redirect to docs."""
        return {
            "message": "Drug Resistance Prediction API",
            "version": API_VERSION_FULL,
            "docs": "/docs",
            "openapi": "/openapi.json",
        }

    @app.get(f"/api/{API_VERSION}/version")
    async def get_version():
        """Get API version information."""
        return {
            "api_version": API_VERSION,
            "version_full": API_VERSION_FULL,
            "release_date": "2025-01-01",
            "supported_diseases": list(DISEASE_DATABASE.keys()) if "DISEASE_DATABASE" in dir() else ["hiv"],
        }

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=model.loaded,
            version=API_VERSION_FULL,
        )

    @app.get("/drugs", response_model=List[DrugInfo])
    async def list_drugs():
        """List all available drugs."""
        return [
            DrugInfo(
                name=name,
                drug_class=info["class"],
                full_name=info["full_name"],
                key_mutations=info["mutations"],
            )
            for name, info in DRUG_DATABASE.items()
        ]

    @app.post("/predict", response_model=PredictionOutput)
    async def predict_resistance(input_data: SequenceInput):
        """Predict drug resistance for a sequence."""
        # Validate drug
        if input_data.drug not in DRUG_DATABASE:
            raise HTTPException(status_code=400, detail=f"Unknown drug: {input_data.drug}")

        drug_info = DRUG_DATABASE[input_data.drug]
        drug_class = drug_info["class"]

        # Get expected sequence length
        expected_lengths = {"pi": 99, "nrti": 240, "nnrti": 318, "ini": 288}
        expected_len = expected_lengths.get(drug_class, 99)

        # Encode sequence
        try:
            encoded = encode_sequence(input_data.sequence, expected_len)
            x = torch.tensor(encoded).unsqueeze(0)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Encoding error: {str(e)}")

        # Predict
        try:
            score, confidence = model.predict(x, input_data.drug)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

        return PredictionOutput(
            drug=input_data.drug,
            resistance_score=round(score, 4),
            confidence=round(confidence, 4),
            interpretation=interpret_resistance(score),
            mutations_detected=None,  # TODO: Add mutation detection
        )

    @app.post("/predict/batch", response_model=BatchOutput)
    async def predict_batch(input_data: BatchInput):
        """Batch prediction for multiple sequences."""
        if input_data.drug not in DRUG_DATABASE:
            raise HTTPException(status_code=400, detail=f"Unknown drug: {input_data.drug}")

        drug_info = DRUG_DATABASE[input_data.drug]
        drug_class = drug_info["class"]
        expected_lengths = {"pi": 99, "nrti": 240, "nnrti": 318, "ini": 288}
        expected_len = expected_lengths.get(drug_class, 99)

        predictions = []
        for seq in input_data.sequences:
            try:
                encoded = encode_sequence(seq, expected_len)
                x = torch.tensor(encoded).unsqueeze(0)
                score, confidence = model.predict(x, input_data.drug)

                predictions.append(PredictionOutput(
                    drug=input_data.drug,
                    resistance_score=round(score, 4),
                    confidence=round(confidence, 4),
                    interpretation=interpret_resistance(score),
                    mutations_detected=None,
                ))
            except Exception as e:
                predictions.append(PredictionOutput(
                    drug=input_data.drug,
                    resistance_score=0.0,
                    confidence=0.0,
                    interpretation=f"Error: {str(e)}",
                    mutations_detected=None,
                ))

        return BatchOutput(predictions=predictions, n_sequences=len(predictions))

    @app.post("/predict/uncertainty", response_model=UncertaintyOutput)
    async def predict_with_uncertainty(input_data: SequenceInput, n_samples: int = 50):
        """Predict resistance with MC Dropout uncertainty quantification."""
        if input_data.drug not in DRUG_DATABASE:
            raise HTTPException(status_code=400, detail=f"Unknown drug: {input_data.drug}")

        drug_info = DRUG_DATABASE[input_data.drug]
        drug_class = drug_info["class"]
        expected_lengths = {"pi": 99, "nrti": 240, "nnrti": 318, "ini": 288}
        expected_len = expected_lengths.get(drug_class, 99)

        encoded = encode_sequence(input_data.sequence, expected_len)
        x = torch.tensor(encoded).unsqueeze(0)

        # MC Dropout sampling
        scores = []
        for _ in range(n_samples):
            score, _ = model.predict(x, input_data.drug)
            scores.append(score)

        scores = np.array(scores)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        lower_95 = float(np.percentile(scores, 2.5))
        upper_95 = float(np.percentile(scores, 97.5))

        return UncertaintyOutput(
            drug=input_data.drug,
            mean_score=round(mean_score, 4),
            std_score=round(std_score, 4),
            lower_95=round(lower_95, 4),
            upper_95=round(upper_95, 4),
            n_samples=n_samples,
            interpretation=interpret_resistance(mean_score),
        )

    @app.post("/predict/cross-resistance", response_model=CrossResistanceOutput)
    async def predict_cross_resistance(input_data: SequenceInput):
        """Analyze cross-resistance patterns for NRTI drugs."""
        if input_data.drug not in DRUG_DATABASE:
            raise HTTPException(status_code=400, detail=f"Unknown drug: {input_data.drug}")

        drug_info = DRUG_DATABASE[input_data.drug]
        drug_class = drug_info["class"]

        if drug_class != "nrti":
            raise HTTPException(
                status_code=400,
                detail="Cross-resistance analysis currently only available for NRTI drugs"
            )

        expected_len = 240
        encoded = encode_sequence(input_data.sequence, expected_len)
        x = torch.tensor(encoded).unsqueeze(0)

        # NRTI cross-resistance matrix
        nrti_drugs = ["AZT", "D4T", "ABC", "TDF", "DDI", "3TC"]
        cross_resistance = {}
        high_risk = []

        for drug in nrti_drugs:
            score, _ = model.predict(x, drug)
            cross_resistance[drug] = round(score, 4)
            if score > 0.7:
                high_risk.append(drug)

        # Generate recommendation
        if len(high_risk) >= 4:
            recommendation = "Consider non-NRTI backbone. Multiple NRTI cross-resistance detected."
        elif len(high_risk) >= 2:
            recommendation = f"Avoid {', '.join(high_risk)}. Consider {', '.join([d for d in nrti_drugs if d not in high_risk][:2])}."
        else:
            recommendation = "Good NRTI options available."

        return CrossResistanceOutput(
            primary_drug=input_data.drug,
            cross_resistance=cross_resistance,
            high_risk_drugs=high_risk,
            recommendation=recommendation,
        )

    @app.post("/clinical-report", response_model=ClinicalReportOutput)
    async def generate_clinical_report(
        sequence: str,
        patient_id: Optional[str] = None,
    ):
        """Generate comprehensive clinical decision support report."""
        from datetime import datetime

        results_by_class = {"pi": [], "nrti": [], "nnrti": [], "ini": []}
        recommended = []
        avoid = []
        warnings = []

        expected_lengths = {"pi": 99, "nrti": 240, "nnrti": 318, "ini": 288}

        for drug, info in DRUG_DATABASE.items():
            drug_class = info["class"]
            expected_len = expected_lengths[drug_class]

            try:
                encoded = encode_sequence(sequence, expected_len)
                x = torch.tensor(encoded).unsqueeze(0)
                score, confidence = model.predict(x, drug)

                pred = PredictionOutput(
                    drug=drug,
                    resistance_score=round(score, 4),
                    confidence=round(confidence, 4),
                    interpretation=interpret_resistance(score),
                    mutations_detected=None,
                )
                results_by_class[drug_class].append(pred)

                if score < 0.3:
                    recommended.append(drug)
                elif score > 0.7:
                    avoid.append(drug)
            except Exception:
                pass

        # Generate cross-resistance warnings for NRTI
        nrti_scores = {p.drug: p.resistance_score for p in results_by_class["nrti"]}
        if nrti_scores.get("AZT", 0) > 0.5 and nrti_scores.get("D4T", 0) > 0.5:
            warnings.append("TAM-mediated cross-resistance between AZT and D4T detected")
        if nrti_scores.get("3TC", 0) > 0.5:
            warnings.append("M184V likely present - affects 3TC/FTC")

        # Overall recommendation
        if len(avoid) > 15:
            overall = "Extensive drug resistance. Consider salvage therapy with newer agents (DTG, BIC, DOR)."
        elif len(avoid) > 8:
            overall = "Moderate multi-drug resistance. Construct regimen from recommended drugs."
        elif len(avoid) > 3:
            overall = "Limited resistance. Multiple treatment options available."
        else:
            overall = "Susceptible to most drugs. Standard first-line regimens appropriate."

        return ClinicalReportOutput(
            patient_id=patient_id,
            sequence_length=len(sequence),
            analysis_date=datetime.now().isoformat(),
            drug_class_results=results_by_class,
            recommended_drugs=recommended,
            avoid_drugs=avoid,
            cross_resistance_warnings=warnings,
            novel_mutations=[],  # TODO: Add attention-based detection
            overall_recommendation=overall,
        )

    @app.get("/novel-mutations/{drug_class}", response_model=List[NovelMutationOutput])
    async def get_novel_mutations(drug_class: str):
        """Get identified novel mutation candidates for a drug class."""
        # Load from pre-computed results
        novel_candidates = {
            "nrti": [
                {"position": 105, "attention_score": 0.0173, "known_status": "NOVEL_HIGH"},
                {"position": 143, "attention_score": 0.0166, "known_status": "NOVEL_HIGH"},
                {"position": 145, "attention_score": 0.0150, "known_status": "NOVEL_HIGH"},
            ],
            "nnrti": [
                {"position": 240, "attention_score": 0.0145, "known_status": "NOVEL_HIGH"},
                {"position": 91, "attention_score": 0.0134, "known_status": "NOVEL_HIGH"},
                {"position": 289, "attention_score": 0.0130, "known_status": "NOVEL_HIGH"},
                {"position": 126, "attention_score": 0.0114, "known_status": "NOVEL_HIGH"},
            ],
            "ini": [
                {"position": 14, "attention_score": 0.0190, "known_status": "NOVEL_HIGH"},
                {"position": 208, "attention_score": 0.0161, "known_status": "NOVEL_HIGH"},
                {"position": 161, "attention_score": 0.0153, "known_status": "NOVEL_HIGH"},
                {"position": 232, "attention_score": 0.0153, "known_status": "NOVEL_HIGH"},
                {"position": 152, "attention_score": 0.0141, "known_status": "NOVEL_HIGH"},
                {"position": 135, "attention_score": 0.0133, "known_status": "NOVEL_HIGH"},
            ],
            "pi": [],  # No novel candidates found for PI
        }

        if drug_class not in novel_candidates:
            raise HTTPException(status_code=400, detail=f"Unknown drug class: {drug_class}")

        return [
            NovelMutationOutput(
                position=m["position"],
                attention_score=m["attention_score"],
                known_status=m["known_status"],
                drug_class=drug_class,
                recommendation="Investigate for structural/functional significance",
            )
            for m in novel_candidates[drug_class]
        ]


# =============================================================================
# Multi-Disease Support
# =============================================================================

DISEASE_DATABASE = {
    "hiv": {
        "name": "HIV",
        "description": "Human Immunodeficiency Virus drug resistance",
        "drugs": 23,
        "drug_classes": ["PI", "NRTI", "NNRTI", "INI"],
    },
    "sars_cov_2": {
        "name": "SARS-CoV-2",
        "description": "COVID-19 drug and antibody resistance",
        "drugs": 5,
        "drug_classes": ["Mpro inhibitors", "mAbs"],
    },
    "tuberculosis": {
        "name": "Tuberculosis",
        "description": "MDR-TB and XDR-TB drug resistance",
        "drugs": 13,
        "drug_classes": ["First-line", "Second-line", "Group A/B/C"],
    },
    "influenza": {
        "name": "Influenza",
        "description": "NAI and Cap-dependent endonuclease resistance",
        "drugs": 4,
        "drug_classes": ["NAI", "Polymerase inhibitors"],
    },
    "hcv": {
        "name": "Hepatitis C",
        "description": "DAA resistance-associated substitutions",
        "drugs": 10,
        "drug_classes": ["NS3", "NS5A", "NS5B"],
    },
    "hbv": {
        "name": "Hepatitis B",
        "description": "Nucleos(t)ide analogue resistance",
        "drugs": 5,
        "drug_classes": ["NAs"],
    },
    "malaria": {
        "name": "Malaria",
        "description": "Artemisinin and partner drug resistance",
        "drugs": 6,
        "drug_classes": ["Artemisinin", "Partner drugs"],
    },
    "mrsa": {
        "name": "MRSA",
        "description": "Methicillin-resistant S. aureus",
        "drugs": 10,
        "drug_classes": ["Beta-lactams", "Fluoroquinolones", "Others"],
    },
    "candida": {
        "name": "Candida auris",
        "description": "Multidrug-resistant fungal infection",
        "drugs": 6,
        "drug_classes": ["Echinocandins", "Azoles", "Polyenes"],
    },
    "rsv": {
        "name": "RSV",
        "description": "Monoclonal antibody escape",
        "drugs": 2,
        "drug_classes": ["mAbs"],
    },
    "cancer": {
        "name": "Cancer",
        "description": "Targeted therapy resistance",
        "drugs": 20,
        "drug_classes": ["EGFR TKIs", "BRAF inhibitors", "ALK inhibitors"],
    },
}

if FASTAPI_AVAILABLE:

    class DiseaseInfo(BaseModel):
        """Disease information."""
        id: str
        name: str
        description: str
        drugs: int
        drug_classes: List[str]

    class MultiDiseaseInput(BaseModel):
        """Input for multi-disease prediction."""
        sequence: str = Field(..., description="Amino acid sequence")
        disease: str = Field(..., description="Disease identifier")
        target: Optional[str] = Field(None, description="Specific drug/target")

    class MultiDiseaseOutput(BaseModel):
        """Output for multi-disease prediction."""
        disease: str
        target: str
        resistance_score: float
        classification: str
        confidence: float
        interpretation: str

    @app.get("/diseases", response_model=List[DiseaseInfo])
    async def list_diseases():
        """List all supported diseases."""
        return [
            DiseaseInfo(
                id=disease_id,
                name=info["name"],
                description=info["description"],
                drugs=info["drugs"],
                drug_classes=info["drug_classes"],
            )
            for disease_id, info in DISEASE_DATABASE.items()
        ]

    @app.post("/predict/{disease}", response_model=MultiDiseaseOutput)
    async def predict_disease_specific(
        disease: str,
        input_data: MultiDiseaseInput,
    ):
        """Disease-specific resistance prediction."""
        if disease not in DISEASE_DATABASE:
            raise HTTPException(status_code=400, detail=f"Unknown disease: {disease}")

        disease_info = DISEASE_DATABASE[disease]
        target = input_data.target or "primary"

        # Encode sequence
        try:
            encoded = encode_sequence(input_data.sequence, 100)
            x = torch.tensor(encoded).unsqueeze(0)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Encoding error: {str(e)}")

        # Mock prediction (replace with actual disease-specific model)
        score = 0.3 + 0.4 * torch.sigmoid(x.sum() / 1000).item()
        confidence = 0.85 + 0.1 * np.random.random()

        # Classification
        if score < 0.3:
            classification = "susceptible"
        elif score < 0.5:
            classification = "low_resistance"
        elif score < 0.7:
            classification = "intermediate"
        else:
            classification = "high_resistance"

        return MultiDiseaseOutput(
            disease=disease,
            target=target,
            resistance_score=round(score, 4),
            classification=classification,
            confidence=round(confidence, 4),
            interpretation=interpret_resistance(score),
        )

    @app.get("/diseases/{disease}/drugs")
    async def get_disease_drugs(disease: str):
        """Get drugs available for a disease."""
        if disease not in DISEASE_DATABASE:
            raise HTTPException(status_code=400, detail=f"Unknown disease: {disease}")

        # Return disease-specific drug info
        disease_info = DISEASE_DATABASE[disease]

        # For HIV, return detailed drug info
        if disease == "hiv":
            return {
                "disease": disease,
                "drugs": [
                    {"name": name, **info}
                    for name, info in DRUG_DATABASE.items()
                ],
            }

        # For other diseases, return placeholder
        return {
            "disease": disease,
            "n_drugs": disease_info["drugs"],
            "drug_classes": disease_info["drug_classes"],
            "note": "Detailed drug info coming soon",
        }


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        import uvicorn
        print("Starting Multi-Disease Drug Resistance API...")
        print("API docs available at: http://localhost:8000/docs")
        print(f"Supported diseases: {', '.join(DISEASE_DATABASE.keys())}")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
