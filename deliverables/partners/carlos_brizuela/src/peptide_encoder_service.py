#!/usr/bin/env python3
"""PeptideEncoder Service for Brizuela AMP Design Package.

This module provides a service layer for the PeptideVAE model, integrating
it with the existing NSGA-II optimization scripts (B1, B8, B10).

Features:
    - Load trained PeptideVAE checkpoints
    - Predict MIC for peptide sequences
    - Generate novel peptides from latent space
    - Get hyperbolic embeddings for peptides
    - Interpolate between peptides in latent space

Usage:
    from src.peptide_encoder_service import get_peptide_encoder_service

    service = get_peptide_encoder_service()
    mic = service.predict_mic("KKLFKKILKYL")
    embedding = service.encode("KKLFKKILKYL")
    novel_peptide = service.generate_from_latent(z)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Add paths - repo_root must be first to access main src/ package
_src_dir = Path(__file__).resolve().parent
_package_dir = _src_dir.parent
_deliverables_dir = _package_dir.parent.parent
_repo_root = _deliverables_dir.parent
sys.path.insert(0, str(_repo_root))  # Must be first to find main src/

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from src.encoders.peptide_encoder import PeptideVAE
    from src.geometry import poincare_distance


# =============================================================================
# Service Configuration
# =============================================================================


DEFAULT_CHECKPOINT = _package_dir / "checkpoints_definitive" / "best_production.pt"


# =============================================================================
# Singleton Service
# =============================================================================


_peptide_encoder_service = None


def get_peptide_encoder_service(
    checkpoint_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> "PeptideEncoderService":
    """Get the singleton PeptideEncoderService instance.

    Args:
        checkpoint_path: Optional path to checkpoint (uses default if None)
        device: Device to use ('cpu', 'cuda', or None for auto)

    Returns:
        PeptideEncoderService instance
    """
    global _peptide_encoder_service

    if _peptide_encoder_service is None:
        _peptide_encoder_service = PeptideEncoderService(
            checkpoint_path=checkpoint_path,
            device=device,
        )

    return _peptide_encoder_service


# =============================================================================
# Service Class
# =============================================================================


class PeptideEncoderService:
    """Service for peptide encoding, MIC prediction, and generation."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        """Initialize the PeptideEncoderService.

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self.model: Optional[PeptideVAE] = None
        self.config: Dict = {}
        self.is_real = False

        if HAS_TORCH:
            if device is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(device)

            self._load_model()
        else:
            self.device = None
            print("Warning: PyTorch not available, using mock mode")

    def _load_model(self) -> bool:
        """Load the model from checkpoint.

        Returns:
            True if model loaded successfully, False otherwise
        """
        if not HAS_TORCH:
            return False

        if not Path(self.checkpoint_path).exists():
            print(f"Checkpoint not found: {self.checkpoint_path}")
            print("Using mock mode for predictions")
            return False

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.config = checkpoint.get('config', {})

            self.model = PeptideVAE(
                latent_dim=self.config.get('latent_dim', 16),
                hidden_dim=self.config.get('hidden_dim', 128),
                n_layers=self.config.get('n_layers', 2),
                n_heads=self.config.get('n_heads', 4),
                dropout=self.config.get('dropout', 0.1),
                max_radius=self.config.get('max_radius', 0.95),
                curvature=self.config.get('curvature', 1.0),
            ).to(self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.is_real = True

            print(f"Loaded PeptideVAE from {self.checkpoint_path}")
            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    @torch.no_grad()
    def encode(self, sequences: Union[str, List[str]]) -> np.ndarray:
        """Encode peptide sequences to hyperbolic embeddings.

        Args:
            sequences: Single sequence or list of sequences

        Returns:
            Hyperbolic embeddings (n_sequences, latent_dim)
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        if not self.is_real:
            # Mock mode: return random embeddings
            latent_dim = self.config.get('latent_dim', 16)
            return np.random.randn(len(sequences), latent_dim) * 0.5

        outputs = self.model.encode(sequences)
        return outputs['z_hyp'].cpu().numpy()

    @torch.no_grad()
    def predict_mic(
        self,
        sequences: Union[str, List[str]],
        return_log: bool = True,
    ) -> np.ndarray:
        """Predict MIC values for peptide sequences.

        Args:
            sequences: Single sequence or list of sequences
            return_log: If True, return log10(MIC), else return MIC in ug/mL

        Returns:
            Predicted MIC values (n_sequences,)
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        if not self.is_real:
            # Mock mode: return heuristic predictions
            from deliverables.shared.peptide_utils import compute_peptide_properties

            predictions = []
            for seq in sequences:
                props = compute_peptide_properties(seq)
                # Heuristic: higher charge + moderate hydrophobicity → lower MIC
                charge = props.get('net_charge', 0)
                hydro = props.get('hydrophobicity', 0)
                log_mic = 1.5 - 0.1 * charge + 0.05 * abs(hydro - 0.3)
                predictions.append(log_mic)

            log_mic = np.array(predictions)
            if return_log:
                return log_mic
            else:
                return 10 ** log_mic

        outputs = self.model(sequences, teacher_forcing=False)
        log_mic = outputs['mic_pred'].squeeze(-1).cpu().numpy()

        if return_log:
            return log_mic
        else:
            return 10 ** log_mic

    @torch.no_grad()
    def get_radii(self, sequences: Union[str, List[str]]) -> np.ndarray:
        """Get hyperbolic radii for peptide sequences.

        Lower radius = more active (central in Poincare ball)

        Args:
            sequences: Single sequence or list of sequences

        Returns:
            Radii (n_sequences,)
        """
        if isinstance(sequences, str):
            sequences = [sequences]

        if not self.is_real:
            # Mock mode: return predicted MIC as proxy for radius
            return self.predict_mic(sequences) / 3  # Scale to [0, 1]

        outputs = self.model.encode(sequences)
        z_hyp = outputs['z_hyp']
        radii = self.model.get_hyperbolic_radii(z_hyp)
        return radii.cpu().numpy()

    @torch.no_grad()
    def generate_from_latent(
        self,
        z: np.ndarray,
        temperature: float = 1.0,
    ) -> List[str]:
        """Generate peptide sequences from latent codes.

        Args:
            z: Latent codes (n_samples, latent_dim)
            temperature: Sampling temperature

        Returns:
            List of generated sequences
        """
        if not self.is_real:
            # Mock mode: return placeholder sequences
            n_samples = z.shape[0] if len(z.shape) > 1 else 1
            return ["KKLFKKILKYLGGG" for _ in range(n_samples)]

        z_tensor = torch.tensor(z, dtype=torch.float32, device=self.device)
        if z_tensor.dim() == 1:
            z_tensor = z_tensor.unsqueeze(0)

        sequences = self.model.generate(z_tensor, temperature=temperature)
        return sequences

    @torch.no_grad()
    def interpolate(
        self,
        seq1: str,
        seq2: str,
        n_steps: int = 10,
    ) -> List[Tuple[np.ndarray, str, float]]:
        """Interpolate between two peptides in latent space.

        Args:
            seq1: Starting peptide sequence
            seq2: Ending peptide sequence
            n_steps: Number of interpolation steps

        Returns:
            List of (latent_code, generated_sequence, predicted_mic) tuples
        """
        if not self.is_real:
            # Mock mode
            return [
                (np.zeros(16), f"INTERP_{i}", 1.0)
                for i in range(n_steps)
            ]

        # Encode both sequences
        z1 = self.encode(seq1)[0]
        z2 = self.encode(seq2)[0]

        results = []
        for i in range(n_steps):
            t = i / (n_steps - 1)
            z_interp = (1 - t) * z1 + t * z2

            seq = self.generate_from_latent(z_interp[np.newaxis, :])[0]
            mic = self.predict_mic(seq)[0]

            results.append((z_interp, seq, mic))

        return results

    @torch.no_grad()
    def sample_around(
        self,
        sequence: str,
        n_samples: int = 10,
        radius: float = 0.1,
    ) -> List[Tuple[str, float]]:
        """Sample peptides around a given sequence in latent space.

        Args:
            sequence: Center peptide sequence
            n_samples: Number of samples to generate
            radius: Sampling radius in latent space

        Returns:
            List of (generated_sequence, predicted_mic) tuples
        """
        if not self.is_real:
            # Mock mode
            return [
                (f"SAMPLE_{i}_{sequence[:5]}", 1.0 + 0.1 * i)
                for i in range(n_samples)
            ]

        # Encode the center sequence
        z_center = self.encode(sequence)[0]

        results = []
        for _ in range(n_samples):
            # Sample a random direction
            noise = np.random.randn(len(z_center))
            noise = noise / np.linalg.norm(noise) * radius

            z_sample = z_center + noise
            seq = self.generate_from_latent(z_sample[np.newaxis, :])[0]
            mic = self.predict_mic(seq)[0]

            results.append((seq, mic))

        return results

    def get_pathogen_activity(
        self,
        sequence: str,
        pathogen: str = "general",
    ) -> Dict[str, float]:
        """Get predicted activity metrics for a specific pathogen.

        Args:
            sequence: Peptide sequence
            pathogen: Target pathogen (general, escherichia, etc.)

        Returns:
            Dictionary with mic, radius, confidence
        """
        mic = self.predict_mic(sequence, return_log=True)[0]
        radius = self.get_radii(sequence)[0]

        # Convert MIC to activity classification
        if mic < 0.5:
            activity = "high"
            confidence = 0.9
        elif mic < 1.0:
            activity = "moderate"
            confidence = 0.7
        elif mic < 1.5:
            activity = "low"
            confidence = 0.6
        else:
            activity = "inactive"
            confidence = 0.5

        return {
            'pathogen': pathogen,
            'log_mic': mic,
            'mic_ug_ml': 10 ** mic,
            'radius': radius,
            'activity': activity,
            'confidence': confidence,
        }

    # =========================================================================
    # Mechanism-Aware Methods (via MechanismDesignService integration)
    # =========================================================================
    #
    # !! DISCLAIMER: PREMATURE MOCK - NOT PRODUCTION READY !!
    # C5 hold-out generalization test NOT RUN. R2 constraint violated.
    # These methods expose PARTIALLY-validated findings.
    # See api/amp_design_api.py for full disclaimer.
    #
    # =========================================================================

    def get_mechanism(self, sequence: str) -> Dict:
        """Classify killing mechanism based on peptide properties.

        Uses validated mechanism fingerprint from P1 investigation.

        Args:
            sequence: Peptide sequence

        Returns:
            Dict with mechanism, confidence, description, cluster_id, has_pathogen_signal
        """
        from deliverables.partners.carlos_brizuela.api import get_mechanism_service

        service = get_mechanism_service()

        # Compute properties
        length = len(sequence)
        net_charge = service._compute_net_charge(sequence)
        hydrophobicity = service._compute_hydrophobicity(sequence)

        return service.classify_mechanism(length, hydrophobicity, net_charge)

    def get_pathogen_ranking(self, sequence: str) -> List[Dict]:
        """Rank pathogens by predicted efficacy for a sequence.

        Uses mechanism-pathogen map from P1 investigation.

        Args:
            sequence: Peptide sequence

        Returns:
            List of dicts with pathogen, relative_efficacy, confidence
        """
        from deliverables.partners.carlos_brizuela.api import get_mechanism_service

        service = get_mechanism_service()
        result = service.rank_pathogens(sequence)

        return result["pathogen_ranking"]

    def get_design_rules(self, target_pathogen: str) -> Dict:
        """Get design rules for target pathogen.

        Based on validated C3 theorem findings from P1 investigation.
        NOTE: N-terminal cationic hypothesis was FALSIFIED - not included.

        Args:
            target_pathogen: One of A_baumannii, P_aeruginosa, Enterobacteriaceae,
                           S_aureus, H_pylori

        Returns:
            Dict with recommended_length, recommended_mechanism, sequence_rules,
            rationale, confidence, warning
        """
        from deliverables.partners.carlos_brizuela.api import get_mechanism_service

        service = get_mechanism_service()
        return service.get_design_rules(target_pathogen)

    def route_regime(self, sequence: str) -> Dict:
        """Determine prediction regime for a sequence.

        Based on arrow-flip threshold (hydrophobicity > 0.107).

        Args:
            sequence: Peptide sequence

        Returns:
            Dict with regime, threshold_used, expected_separation, rationale
        """
        from deliverables.partners.carlos_brizuela.api import get_mechanism_service

        service = get_mechanism_service()
        hydrophobicity = service._compute_hydrophobicity(sequence)

        return service.route_regime(hydrophobicity)


# =============================================================================
# Convenience Functions
# =============================================================================


def predict_amp_activity(
    sequences: Union[str, List[str]],
    pathogen: str = "general",
) -> List[Dict[str, float]]:
    """Convenience function to predict AMP activity.

    Args:
        sequences: Peptide sequences
        pathogen: Target pathogen

    Returns:
        List of activity dictionaries
    """
    service = get_peptide_encoder_service()

    if isinstance(sequences, str):
        sequences = [sequences]

    results = []
    for seq in sequences:
        result = service.get_pathogen_activity(seq, pathogen)
        result['sequence'] = seq
        results.append(result)

    return results


def rank_peptides_by_activity(
    sequences: List[str],
    pathogen: str = "general",
) -> List[Tuple[str, float]]:
    """Rank peptides by predicted antimicrobial activity.

    Args:
        sequences: List of peptide sequences
        pathogen: Target pathogen

    Returns:
        List of (sequence, log_mic) tuples sorted by activity (lower MIC first)
    """
    service = get_peptide_encoder_service()
    mics = service.predict_mic(sequences, return_log=True)

    ranked = sorted(zip(sequences, mics), key=lambda x: x[1])
    return ranked


# =============================================================================
# Main (Testing)
# =============================================================================


if __name__ == '__main__':
    print("PeptideEncoder Service Test")
    print("="*50)

    service = get_peptide_encoder_service()
    print(f"Is real model: {service.is_real}")
    print(f"Device: {service.device}")
    print()

    # Test sequences
    test_sequences = [
        "KKLFKKILKYL",    # BP100-like (cationic AMP)
        "GIGKFLHSAKKFGKAFVGEI",  # Magainin-like
        "LLGDFFRKSKEK",   # LL-37 fragment
    ]

    print("Test Predictions:")
    print("-"*50)
    for seq in test_sequences:
        mic = service.predict_mic(seq)[0]
        radius = service.get_radii(seq)[0]
        activity = service.get_pathogen_activity(seq)
        print(f"  {seq[:15]:15s} | MIC={10**mic:6.1f} ug/mL | r={radius:.3f} | {activity['activity']}")

    print()
    print("Ranking by activity:")
    print("-"*50)
    ranked = rank_peptides_by_activity(test_sequences)
    for i, (seq, mic) in enumerate(ranked):
        print(f"  {i+1}. {seq[:15]:15s} | log10(MIC)={mic:.3f}")

    print()
    print("Service test complete!")
