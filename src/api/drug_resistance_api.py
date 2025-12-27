"""FastAPI Web Interface for Drug Resistance Prediction.

Provides a REST API for HIV drug resistance prediction from sequences.

Usage:
    uvicorn src.api.drug_resistance_api:app --reload --port 8000

Endpoints:
    POST /predict - Predict resistance for a sequence
    POST /predict/batch - Batch prediction
    GET /drugs - List available drugs
    GET /health - Health check
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# Add project root
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not installed. Run: pip install fastapi uvicorn")

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
    app = FastAPI(
        title="HIV Drug Resistance Prediction API",
        description="P-adic VAE-based prediction of HIV drug resistance from sequences",
        version="1.0.0",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            model_loaded=model.loaded,
            version="1.0.0",
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


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        import uvicorn
        print("Starting Drug Resistance API...")
        print("API docs available at: http://localhost:8000/docs")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("FastAPI not available. Install with: pip install fastapi uvicorn")
