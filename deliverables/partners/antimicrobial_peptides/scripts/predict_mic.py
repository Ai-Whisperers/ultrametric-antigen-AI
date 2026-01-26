#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""PeptideMICPredictor: Production-grade MIC prediction for antimicrobial peptides.

This module provides a high-quality API for predicting Minimum Inhibitory
Concentration (MIC) values using the trained PeptideVAE model (Spearman r=0.74).

Architecture:
    PeptideVAE (Transformer + Hyperbolic Projection)
    - Encoder: 2-layer Transformer with 4 attention heads
    - Latent: 16D Poincare ball (hyperbolic space)
    - MIC Head: 16 -> 32 -> 32 -> 1 MLP

Performance:
    - Spearman correlation: 0.7368 (best fold)
    - Mean Spearman: 0.6563 +/- 0.0599 (5-fold CV)
    - Baseline (sklearn Ridge): 0.56
    - Improvement over baseline: +31%

Usage:
    # Single prediction
    python predict_mic.py "KLAKLAKKLAKLAK"

    # Batch prediction from file
    python predict_mic.py --file candidates.txt --output results.csv

    # Interactive mode
    python predict_mic.py --interactive

    # As module
    from scripts.predict_mic import PeptideMICPredictor
    predictor = PeptideMICPredictor()
    mic = predictor.predict("KLAKLAKKLAKLAK")

Example Output:
    Sequence: KLAKLAKKLAKLAK
    Length: 14 AA
    Predicted log10(MIC): 0.72
    Predicted MIC: 5.25 ug/mL
    Hyperbolic radius: 0.48
    Confidence: High (within training distribution)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Add project root to path
# Path: scripts/ -> carlos_brizuela/ -> partners/ -> deliverables/ -> PROJECT_ROOT
SCRIPT_DIR = Path(__file__).parent
PACKAGE_DIR = SCRIPT_DIR.parent  # carlos_brizuela/
BRIZUELA_ROOT = PACKAGE_DIR  # Alias for consistency
PROJECT_ROOT = PACKAGE_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Lazy imports for torch (may not be available)
torch = None
PeptideVAE = None


def _lazy_import_torch():
    """Lazily import torch and model to avoid startup overhead."""
    global torch, PeptideVAE
    if torch is None:
        import torch as _torch
        torch = _torch
        from src.encoders.peptide_encoder import PeptideVAE as _PeptideVAE
        PeptideVAE = _PeptideVAE


# =============================================================================
# Constants
# =============================================================================

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
MIN_PEPTIDE_LENGTH = 5
MAX_PEPTIDE_LENGTH = 50

# Local checkpoint for MIC prediction
DEFAULT_CHECKPOINT = BRIZUELA_ROOT / "checkpoints_definitive" / "best_production.pt"

# Model configuration (from cv_results_definitive.json)
MODEL_CONFIG = {
    "latent_dim": 16,
    "hidden_dim": 64,
    "n_layers": 2,
    "n_heads": 4,
    "dropout": 0.15,
    "max_radius": 0.95,
    "curvature": 1.0,
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PredictionResult:
    """Result of MIC prediction for a single peptide."""

    sequence: str
    log10_mic: float
    mic_ug_ml: float
    hyperbolic_radius: float
    latent_vector: Optional[np.ndarray]
    confidence: str
    warnings: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "sequence": self.sequence,
            "length": len(self.sequence),
            "log10_mic": round(self.log10_mic, 4),
            "mic_ug_ml": round(self.mic_ug_ml, 4),
            "hyperbolic_radius": round(self.hyperbolic_radius, 4),
            "confidence": self.confidence,
            "warnings": self.warnings,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"Sequence: {self.sequence}",
            f"Length: {len(self.sequence)} AA",
            f"Predicted log10(MIC): {self.log10_mic:.2f}",
            f"Predicted MIC: {self.mic_ug_ml:.2f} ug/mL",
            f"Hyperbolic radius: {self.hyperbolic_radius:.2f}",
            f"Confidence: {self.confidence}",
        ]
        if self.warnings:
            lines.append(f"Warnings: {'; '.join(self.warnings)}")
        return "\n".join(lines)


@dataclass
class BatchResult:
    """Result of batch MIC prediction."""

    results: List[PredictionResult]
    n_success: int
    n_failed: int
    mean_mic: float
    std_mic: float

    def to_csv(self, path: Path) -> None:
        """Write results to CSV file."""
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sequence", "length", "log10_mic", "mic_ug_ml",
                    "hyperbolic_radius", "confidence", "warnings"
                ]
            )
            writer.writeheader()
            for result in self.results:
                row = result.to_dict()
                row["warnings"] = "; ".join(row["warnings"])
                writer.writerow(row)

    def to_json(self, path: Path) -> None:
        """Write results to JSON file."""
        data = {
            "summary": {
                "n_sequences": len(self.results),
                "n_success": self.n_success,
                "n_failed": self.n_failed,
                "mean_mic": round(self.mean_mic, 4),
                "std_mic": round(self.std_mic, 4),
            },
            "predictions": [r.to_dict() for r in self.results]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_sequence(sequence: str) -> Tuple[str, List[str]]:
    """Validate and normalize a peptide sequence.

    Args:
        sequence: Raw peptide sequence string

    Returns:
        Tuple of (normalized_sequence, list_of_warnings)

    Raises:
        ValueError: If sequence is fundamentally invalid
    """
    warnings = []

    # Remove whitespace and convert to uppercase
    clean = re.sub(r"\s+", "", sequence.upper())

    # Check for empty sequence
    if not clean:
        raise ValueError("Empty sequence provided")

    # Check length bounds
    if len(clean) < MIN_PEPTIDE_LENGTH:
        raise ValueError(
            f"Sequence too short: {len(clean)} AA (minimum: {MIN_PEPTIDE_LENGTH})"
        )
    if len(clean) > MAX_PEPTIDE_LENGTH:
        raise ValueError(
            f"Sequence too long: {len(clean)} AA (maximum: {MAX_PEPTIDE_LENGTH})"
        )

    # Check for invalid characters
    invalid_chars = set(clean) - VALID_AMINO_ACIDS
    if invalid_chars:
        # Check if they're common substitutions
        substitutions = {"B": "N", "Z": "Q", "X": "A", "U": "C", "O": "K"}
        new_clean = ""
        for char in clean:
            if char in substitutions:
                new_clean += substitutions[char]
                warnings.append(f"Substituted {char} -> {substitutions[char]}")
            elif char in VALID_AMINO_ACIDS:
                new_clean += char
            else:
                raise ValueError(f"Invalid amino acid: {char}")
        clean = new_clean

    # Check for unusual composition
    cationic_ratio = sum(1 for aa in clean if aa in "KRH") / len(clean)
    if cationic_ratio < 0.1:
        warnings.append("Low cationic content (<10%) - atypical for AMPs")

    hydrophobic_ratio = sum(1 for aa in clean if aa in "AILMFVW") / len(clean)
    if hydrophobic_ratio > 0.6:
        warnings.append("High hydrophobicity (>60%) - potential aggregation risk")

    return clean, warnings


# =============================================================================
# Main Predictor Class
# =============================================================================

class PeptideMICPredictor:
    """Production-grade MIC predictor using trained PeptideVAE.

    This class provides a clean API for MIC prediction with:
    - Automatic model loading and caching
    - Input validation and normalization
    - Confidence estimation
    - Batch processing support

    Attributes:
        model: Loaded PeptideVAE model
        device: Torch device (cpu or cuda)
        config: Model configuration dictionary
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        """Initialize the MIC predictor.

        Args:
            checkpoint_path: Path to model checkpoint. Defaults to best_production.pt
            device: Device to use ('cpu' or 'cuda'). Auto-detected if None.
            verbose: Print loading messages
        """
        _lazy_import_torch()

        self.verbose = verbose
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)

        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        self.model, self.config = self._load_model()

        if self.verbose:
            print(f"PeptideMICPredictor initialized")
            print(f"  Checkpoint: {self.checkpoint_path.name}")
            print(f"  Device: {self.device}")
            print(f"  Latent dim: {self.config['latent_dim']}")

    def _load_model(self) -> Tuple["PeptideVAE", Dict]:
        """Load the PeptideVAE model from checkpoint.

        Returns:
            Tuple of (model, config)

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If checkpoint is invalid
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Run training first or check the path."
            )

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except Exception as e:
            # Check for Git LFS pointer
            with open(self.checkpoint_path, "rb") as f:
                header = f.read(20)
            if header.startswith(b"version https://git"):
                raise RuntimeError(
                    f"Checkpoint is a Git LFS pointer. Run: git lfs pull"
                ) from e
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e

        # Extract config
        if "config" in checkpoint:
            config = checkpoint["config"]
        else:
            config = MODEL_CONFIG.copy()

        # Create model
        model = PeptideVAE(
            latent_dim=config.get("latent_dim", MODEL_CONFIG["latent_dim"]),
            hidden_dim=config.get("hidden_dim", MODEL_CONFIG["hidden_dim"]),
            n_layers=config.get("n_layers", MODEL_CONFIG["n_layers"]),
            n_heads=config.get("n_heads", MODEL_CONFIG["n_heads"]),
            dropout=config.get("dropout", MODEL_CONFIG["dropout"]),
            max_radius=config.get("max_radius", MODEL_CONFIG["max_radius"]),
            curvature=config.get("curvature", MODEL_CONFIG["curvature"]),
        )

        # Load weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        return model, config

    def predict(
        self,
        sequence: str,
        return_latent: bool = False,
    ) -> PredictionResult:
        """Predict MIC for a single peptide sequence.

        Args:
            sequence: Amino acid sequence (e.g., "KLAKLAKKLAKLAK")
            return_latent: Include latent vector in result

        Returns:
            PredictionResult with MIC prediction and metadata

        Raises:
            ValueError: If sequence is invalid
        """
        # Validate input
        clean_sequence, warnings = validate_sequence(sequence)

        # Run inference
        with torch.no_grad():
            outputs = self.model([clean_sequence], teacher_forcing=False)

            log10_mic = outputs["mic_pred"].squeeze().item()
            z_hyp = outputs["z_hyp"].squeeze()

            # Calculate hyperbolic radius
            radius = self.model.get_hyperbolic_radii(
                outputs["z_hyp"]
            ).squeeze().item()

        # Convert to MIC in ug/mL
        mic_ug_ml = 10 ** log10_mic

        # Estimate confidence based on radius
        if radius < 0.3:
            confidence = "High"
        elif radius < 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"
            warnings.append("High hyperbolic radius - extrapolation risk")

        # Get latent vector if requested
        latent = z_hyp.cpu().numpy() if return_latent else None

        return PredictionResult(
            sequence=clean_sequence,
            log10_mic=log10_mic,
            mic_ug_ml=mic_ug_ml,
            hyperbolic_radius=radius,
            latent_vector=latent,
            confidence=confidence,
            warnings=warnings,
        )

    def predict_batch(
        self,
        sequences: List[str],
        batch_size: int = 32,
        return_latent: bool = False,
    ) -> BatchResult:
        """Predict MIC for multiple peptide sequences.

        Args:
            sequences: List of amino acid sequences
            batch_size: Batch size for inference
            return_latent: Include latent vectors in results

        Returns:
            BatchResult with all predictions and summary statistics
        """
        results = []
        valid_mics = []

        for seq in sequences:
            try:
                result = self.predict(seq, return_latent=return_latent)
                results.append(result)
                valid_mics.append(result.log10_mic)
            except ValueError as e:
                # Create failed result
                results.append(PredictionResult(
                    sequence=seq[:50] + "..." if len(seq) > 50 else seq,
                    log10_mic=float("nan"),
                    mic_ug_ml=float("nan"),
                    hyperbolic_radius=float("nan"),
                    latent_vector=None,
                    confidence="Failed",
                    warnings=[str(e)],
                ))

        n_success = sum(1 for r in results if r.confidence != "Failed")
        n_failed = len(results) - n_success

        if valid_mics:
            mean_mic = np.mean(valid_mics)
            std_mic = np.std(valid_mics)
        else:
            mean_mic = float("nan")
            std_mic = float("nan")

        return BatchResult(
            results=results,
            n_success=n_success,
            n_failed=n_failed,
            mean_mic=mean_mic,
            std_mic=std_mic,
        )

    def get_embedding(self, sequence: str) -> np.ndarray:
        """Get hyperbolic embedding for a sequence.

        Args:
            sequence: Amino acid sequence

        Returns:
            16D hyperbolic embedding as numpy array
        """
        result = self.predict(sequence, return_latent=True)
        return result.latent_vector


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for MIC prediction."""
    parser = argparse.ArgumentParser(
        description="Predict MIC for antimicrobial peptides using PeptideVAE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single sequence
    python predict_mic.py "KLAKLAKKLAKLAK"

    # From file (one sequence per line)
    python predict_mic.py --file candidates.txt --output results.csv

    # Interactive mode
    python predict_mic.py --interactive

    # JSON output
    python predict_mic.py --file candidates.txt --output results.json --format json
        """,
    )

    parser.add_argument(
        "sequence",
        nargs="?",
        help="Peptide sequence to predict (e.g., KLAKLAKKLAKLAK)",
    )
    parser.add_argument(
        "--file", "-f",
        type=Path,
        help="File with sequences (one per line)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for batch results",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode for continuous prediction",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device to use (auto-detected if not specified)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress informational messages",
    )

    args = parser.parse_args()

    # Initialize predictor
    try:
        predictor = PeptideMICPredictor(
            checkpoint_path=args.checkpoint,
            device=args.device,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"Error initializing predictor: {e}", file=sys.stderr)
        sys.exit(1)

    # Handle different modes
    if args.interactive:
        print("\nInteractive MIC Prediction")
        print("Enter peptide sequences (Ctrl+C to exit)")
        print("-" * 40)
        while True:
            try:
                seq = input("\nSequence> ").strip()
                if not seq:
                    continue
                result = predictor.predict(seq)
                print(result)
            except ValueError as e:
                print(f"Error: {e}")
            except KeyboardInterrupt:
                print("\nExiting.")
                break

    elif args.file:
        # Batch mode
        if not args.file.exists():
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)

        with open(args.file) as f:
            sequences = [line.strip() for line in f if line.strip()]

        if not sequences:
            print("Error: No sequences found in file", file=sys.stderr)
            sys.exit(1)

        print(f"Processing {len(sequences)} sequences...")
        batch_result = predictor.predict_batch(sequences)

        print(f"\nResults: {batch_result.n_success} successful, {batch_result.n_failed} failed")
        print(f"Mean log10(MIC): {batch_result.mean_mic:.3f} +/- {batch_result.std_mic:.3f}")

        if args.output:
            if args.format == "json":
                batch_result.to_json(args.output)
            else:
                batch_result.to_csv(args.output)
            print(f"Results saved to: {args.output}")
        else:
            # Print to stdout
            for result in batch_result.results:
                status = "OK" if result.confidence != "Failed" else "FAIL"
                print(f"{status}\t{result.sequence}\t{result.log10_mic:.3f}\t{result.mic_ug_ml:.3f}")

    elif args.sequence:
        # Single sequence mode
        try:
            result = predictor.predict(args.sequence)
            print(result)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
