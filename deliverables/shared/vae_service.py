# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Centralized VAE service for all deliverables.

Provides a singleton VAE interface that all tools can use:
- Encode sequences to latent space
- Decode latent vectors to sequences
- Compute p-adic valuations and stability metrics
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .config import get_config
from .constants import CODON_TABLE, HYDROPHOBICITY, CHARGES

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TernaryDecoder:
    """Decode ternary operations to amino acid sequences."""

    TERNARY_TO_NUC = {-1: "T", 0: "C", 1: "A"}

    def __init__(self):
        self.codon_to_aa = CODON_TABLE

    def ternary_to_codons(self, ternary_op: np.ndarray) -> list[str]:
        """Convert 9 ternary values to 3 codons."""
        if len(ternary_op) != 9:
            raise ValueError(f"Expected 9 ternary values, got {len(ternary_op)}")

        nucs = [self.TERNARY_TO_NUC.get(int(t), "N") for t in ternary_op]
        return ["".join(nucs[i:i + 3]) for i in range(0, 9, 3)]

    def decode_to_amino_acids(self, ternary_ops: np.ndarray) -> str:
        """Decode batch of ternary operations to amino acid sequence."""
        if ternary_ops.ndim == 1:
            ternary_ops = ternary_ops.reshape(1, -1)

        amino_acids = []
        for op in ternary_ops:
            codons = self.ternary_to_codons(op)
            for codon in codons:
                aa = self.codon_to_aa.get(codon, "X")
                if aa != "*":  # Skip stop codons
                    amino_acids.append(aa)

        return "".join(amino_acids)


class VAEService:
    """Singleton VAE service for all tools."""

    _instance: Optional["VAEService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.config = get_config()
        self.model = None
        self.device = "cpu"
        self.decoder = TernaryDecoder()
        self._initialized = True

        if TORCH_AVAILABLE and self.config.has_vae:
            self._load_model()
        elif self.config.verbose:
            print("VAE Service: Running in mock mode (no checkpoint or torch unavailable)")

    def _load_model(self):
        """Load the VAE model from checkpoint."""
        try:
            from src.models import TernaryVAEV5_11_PartialFreeze

            # Determine device
            if self.config.use_gpu and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

            # Initialize model
            self.model = TernaryVAEV5_11_PartialFreeze(
                latent_dim=16,
                hidden_dim=64,
                max_radius=0.99,
                curvature=1.0,
                use_controller=True,
                use_dual_projection=True,
            )

            # Load checkpoint
            ckpt = torch.load(
                self.config.vae_checkpoint,
                map_location=self.device,
                weights_only=False
            )
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
            else:
                self.model.load_state_dict(ckpt)

            self.model.to(self.device)
            self.model.eval()

            if self.config.verbose:
                print(f"VAE Service: Loaded model from {self.config.vae_checkpoint}")
                print(f"VAE Service: Using device {self.device}")

        except Exception as e:
            error_msg = str(e)
            if self.config.verbose:
                if "invalid load key" in error_msg.lower():
                    print("VAE Service: Checkpoint file appears to be a Git LFS pointer")
                    print("  To download the actual checkpoint, run: git lfs pull")
                else:
                    print(f"VAE Service: Could not load model: {e}")
            self.model = None

    @property
    def is_real(self) -> bool:
        """Check if using real VAE (not mock)."""
        return self.model is not None

    def decode_latent(self, z: Union[np.ndarray, "torch.Tensor"]) -> str:
        """Decode latent vector to amino acid sequence.

        Args:
            z: Latent vector (16-dimensional)

        Returns:
            Amino acid sequence
        """
        if self.model is None:
            return self._mock_decode(z)

        with torch.no_grad():
            if isinstance(z, np.ndarray):
                z = torch.tensor(z, dtype=torch.float32)
            if z.dim() == 1:
                z = z.unsqueeze(0)
            z = z.to(self.device)

            # Decode through VAE
            logits = self.model.decoder(z)  # (batch, 9, 3)
            ternary_ops = torch.argmax(logits, dim=-1) - 1  # {0,1,2} -> {-1,0,1}
            ternary_ops = ternary_ops.cpu().numpy()

            return self.decoder.decode_to_amino_acids(ternary_ops)

    def decode_latent_batch(self, z_batch: np.ndarray) -> list[str]:
        """Decode batch of latent vectors to sequences.

        Args:
            z_batch: Batch of latent vectors (N, 16)

        Returns:
            List of amino acid sequences
        """
        sequences = []
        for z in z_batch:
            sequences.append(self.decode_latent(z))
        return sequences

    def encode_sequence(self, sequence: str) -> np.ndarray:
        """Encode amino acid sequence to latent space.

        Args:
            sequence: Amino acid sequence

        Returns:
            Latent vector (16-dimensional)
        """
        if self.model is None:
            return self._mock_encode(sequence)

        # Real encoding would require sequence-to-ternary mapping
        # For now, use property-based encoding
        return self._mock_encode(sequence)

    def _mock_decode(self, z: np.ndarray) -> str:
        """Mock decoding when VAE not available."""
        if isinstance(z, np.ndarray):
            z_arr = z
        else:
            z_arr = z.cpu().numpy() if hasattr(z, "cpu") else np.array(z)

        if z_arr.ndim > 1:
            z_arr = z_arr[0]

        # Use latent dimensions to bias amino acid selection
        np.random.seed(int(abs(z_arr[0] * 1000)) % (2**31 - 1))

        # Amino acid pools
        hydrophobic = list("ILVFMWYA")
        hydrophilic = list("RKDENQHST")
        cationic = list("KRH")
        aromatic = list("FWY")

        # Latent dimensions influence composition
        radius = np.linalg.norm(z_arr[:4]) if len(z_arr) >= 4 else 0.5
        charge_bias = z_arr[0] if len(z_arr) > 0 else 0
        hydro_bias = z_arr[1] if len(z_arr) > 1 else 0

        # Generate sequence
        length = 20
        sequence = []
        for i in range(length):
            r = np.random.random()
            if charge_bias > 0.3 and r < 0.3:
                sequence.append(np.random.choice(cationic))
            elif hydro_bias > 0 and r < 0.5 + hydro_bias * 0.3:
                sequence.append(np.random.choice(hydrophobic))
            else:
                sequence.append(np.random.choice(hydrophilic))

        return "".join(sequence)

    def _mock_encode(self, sequence: str) -> np.ndarray:
        """Mock encoding based on sequence properties."""
        # Compute properties
        charge = sum(CHARGES.get(aa, 0) for aa in sequence)
        hydro = sum(HYDROPHOBICITY.get(aa, 0) for aa in sequence)
        aromatic = sum(1 for aa in sequence if aa in "FWY")

        # Create latent vector
        z = np.zeros(16)
        z[0] = np.tanh(charge / 5)  # Charge dimension
        z[1] = np.tanh(hydro / len(sequence) / 2) if len(sequence) > 0 else 0  # Hydrophobicity
        z[2] = np.tanh(aromatic / 5)  # Aromatic content
        z[3] = np.tanh(len(sequence) / 30 - 1)  # Length encoding

        # Add some noise for uniqueness
        np.random.seed(hash(sequence) % (2**31 - 1))
        z[4:] = np.random.randn(12) * 0.1

        return z

    def get_radius(self, z: np.ndarray) -> float:
        """Get hyperbolic radius of latent vector."""
        if isinstance(z, np.ndarray):
            return float(np.linalg.norm(z))
        return float(z.norm().item())

    def get_padic_valuation(self, z: np.ndarray) -> int:
        """Estimate p-adic valuation from radius.

        Center (r=0) = high valuation (stable)
        Edge (r=1) = low valuation (unstable)
        """
        radius = self.get_radius(z)
        # Map radius [0, 0.99] to valuation [9, 0]
        valuation = int((1 - min(radius, 0.99) / 0.99) * 9)
        return min(9, max(0, valuation))

    def get_stability_score(self, z: np.ndarray) -> float:
        """Get stability score from latent position.

        Returns value in [0, 1] where 1 = most stable.
        """
        radius = self.get_radius(z)
        return max(0, 1 - radius / 0.99)

    def sample_latent(
        self,
        n_samples: int = 1,
        target_radius: Optional[float] = None,
        charge_bias: float = 0,
        hydro_bias: float = 0,
    ) -> np.ndarray:
        """Sample from latent space with optional biasing.

        Args:
            n_samples: Number of samples
            target_radius: Target radial position (None = random)
            charge_bias: Bias toward positive charge (positive value)
            hydro_bias: Bias toward hydrophobicity (positive value)

        Returns:
            Latent vectors (n_samples, 16)
        """
        samples = np.random.randn(n_samples, 16)

        # Apply biases
        samples[:, 0] += charge_bias
        samples[:, 1] += hydro_bias

        # Normalize to target radius if specified
        if target_radius is not None:
            norms = np.linalg.norm(samples, axis=1, keepdims=True)
            samples = samples / norms * target_radius

        return samples

    def interpolate(self, z1: np.ndarray, z2: np.ndarray, steps: int = 10) -> list[str]:
        """Interpolate between two latent vectors and decode.

        Args:
            z1: Starting latent vector
            z2: Ending latent vector
            steps: Number of interpolation steps

        Returns:
            List of decoded sequences along interpolation path
        """
        sequences = []
        for t in np.linspace(0, 1, steps):
            z_interp = (1 - t) * z1 + t * z2
            sequences.append(self.decode_latent(z_interp))
        return sequences


# Global instance
_vae_service: Optional[VAEService] = None


def get_vae_service() -> VAEService:
    """Get global VAE service instance."""
    global _vae_service
    if _vae_service is None:
        _vae_service = VAEService()
    return _vae_service
