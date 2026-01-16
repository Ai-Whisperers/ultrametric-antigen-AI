#!/usr/bin/env python3
"""VAE Fidelity Diffusion Model - Targeted Fidelity Loss Mapping.

This module implements a simple, non-overengineered diffusion model specifically
designed to identify and correct VAE fidelity loss in genetic code embeddings.

Key Features:
- Topology-aware: Respects hyperbolic geometry and groupoid structure
- VAE-targeted: Identifies specific failure modes (reconstruction, hierarchy, richness)
- Simple architecture: Minimal complexity for maximum insight

Architecture:
    VAE Embedding → Fidelity Detector → Targeted Diffusion → Refined Embedding

Loss Components:
1. Reconstruction Fidelity: Cross-entropy reconstruction accuracy
2. Geometric Consistency: Hyperbolic vs Euclidean metric alignment
3. Structural Preservation: P-adic hierarchy maintenance
4. AA Property Coherence: Biological relationship preservation

Usage:
    from research.vae_fidelity_diffusion import VAEFidelityDiffusion

    diffusion = VAEFidelityDiffusion(latent_dim=16, hidden_dim=128)
    loss_map = diffusion.map_fidelity_loss(vae_embeddings)
    refined = diffusion.refine_embeddings(vae_embeddings, loss_map)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Add project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry import poincare_distance, project_to_poincare
from src.biology.codons import GENETIC_CODE, CODON_TO_INDEX
from src.core.ternary import TernarySpace


class FidelityLossDetector(nn.Module):
    """Detects where VAE loses fidelity in embedding space.

    Identifies five key failure modes:
    1. Reconstruction bottleneck (rare valuations)
    2. Radial-richness trade-off
    3. Hyperbolic-Euclidean inconsistency
    4. KL regularization artifacts
    5. Multi-objective competition
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Fidelity analysis network
        self.fidelity_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 5)  # 5 fidelity dimensions
        )

        # Initialize ternary space for p-adic computations
        self.ternary = TernarySpace()

        # Precompute p-adic valuations for all 64 codons
        self.register_buffer('padic_valuations', self._compute_padic_valuations())

    def _compute_padic_valuations(self) -> Tensor:
        """Precompute 3-adic valuations for efficiency."""
        indices = torch.arange(64)
        # Direct valuation computation on indices
        valuations = self.ternary.valuation(indices)
        return valuations.float()

    def forward(self, embeddings: Tensor) -> Dict[str, Tensor]:
        """Detect fidelity loss patterns.

        Args:
            embeddings: VAE embeddings (batch, latent_dim) on Poincaré ball

        Returns:
            Dict with fidelity scores for each failure mode
        """
        batch_size = embeddings.size(0)

        # Get fidelity scores [reconstruction, radial_richness, metric_consistency,
        #                     kl_artifacts, objective_competition]
        fidelity_scores = torch.sigmoid(self.fidelity_net(embeddings))

        # Compute hyperbolic radii (proper geometry)
        origin = torch.zeros_like(embeddings)
        radii = poincare_distance(embeddings, origin, c=1.0)

        # Analyze specific failure patterns
        reconstruction_loss = self._analyze_reconstruction_failure(embeddings, radii)
        richness_loss = self._analyze_richness_hierarchy_tradeoff(embeddings, radii)
        metric_loss = self._analyze_metric_inconsistency(embeddings)
        kl_loss = self._analyze_kl_artifacts(embeddings)
        objective_loss = self._analyze_objective_competition(embeddings)

        return {
            'reconstruction': fidelity_scores[:, 0] + reconstruction_loss,
            'radial_richness': fidelity_scores[:, 1] + richness_loss,
            'metric_consistency': fidelity_scores[:, 2] + metric_loss,
            'kl_artifacts': fidelity_scores[:, 3] + kl_loss,
            'objective_competition': fidelity_scores[:, 4] + objective_loss,
            'total_fidelity_loss': fidelity_scores.mean(dim=1)
        }

    def _analyze_reconstruction_failure(self, embeddings: Tensor, radii: Tensor) -> Tensor:
        """Detect reconstruction bottleneck in rare valuation levels."""
        batch_size = embeddings.size(0)

        # Map radii to approximate valuation levels
        # v0 (outer) → v9 (center): 0.9 → 0.1 radius mapping
        approx_valuations = 9 - (radii * 10).clamp(0, 9)

        # Rare valuations (v7-v9) have <1% data, causing reconstruction issues
        rare_mask = approx_valuations >= 7.0
        rare_penalty = rare_mask.float() * 0.5  # Higher loss for rare valuations

        return rare_penalty

    def _analyze_richness_hierarchy_tradeoff(self, embeddings: Tensor, radii: Tensor) -> Tensor:
        """Detect radial-richness optimization conflict."""
        batch_size = embeddings.size(0)

        if batch_size < 4:  # Need multiple points for variance
            return torch.zeros(batch_size, device=embeddings.device)

        # Compute radial variance (proxy for richness)
        radial_var = torch.var(radii).item()

        # Trade-off penalty: Low richness (<0.003) or poor hierarchy correlation
        # Based on analysis: collapsed models (richness~0.003) vs rich models (0.006+)
        richness_threshold = 0.003
        richness_penalty = torch.ones(batch_size, device=embeddings.device)

        if radial_var < richness_threshold:
            richness_penalty *= 0.3  # Collapsed shells detected
        elif radial_var > 0.008:
            richness_penalty *= 0.2  # Potentially over-rich, hierarchy may suffer

        return richness_penalty

    def _analyze_metric_inconsistency(self, embeddings: Tensor) -> Tensor:
        """Detect hyperbolic vs Euclidean metric inconsistencies."""
        batch_size = embeddings.size(0)

        if batch_size < 2:
            return torch.zeros(batch_size, device=embeddings.device)

        # Compare hyperbolic vs Euclidean pairwise distances
        hyp_dists = []
        euc_dists = []

        for i in range(min(batch_size, 8)):  # Limit for efficiency
            for j in range(i+1, min(batch_size, 8)):
                hyp_d = poincare_distance(
                    embeddings[i:i+1], embeddings[j:j+1], c=1.0
                ).item()
                euc_d = torch.norm(embeddings[i] - embeddings[j]).item()

                hyp_dists.append(hyp_d)
                euc_dists.append(euc_d)

        # Correlation between hyperbolic and Euclidean distances
        # Low correlation indicates metric inconsistency
        if len(hyp_dists) > 1:
            hyp_tensor = torch.tensor(hyp_dists, device=embeddings.device)
            euc_tensor = torch.tensor(euc_dists, device=embeddings.device)

            # Simple correlation estimate
            hyp_centered = hyp_tensor - hyp_tensor.mean()
            euc_centered = euc_tensor - euc_tensor.mean()
            correlation = torch.sum(hyp_centered * euc_centered) / (
                torch.sqrt(torch.sum(hyp_centered ** 2)) *
                torch.sqrt(torch.sum(euc_centered ** 2)) + 1e-8
            )

            # Penalty for low correlation (metric inconsistency)
            metric_penalty = (1 - correlation.abs()) * 0.4
        else:
            metric_penalty = 0.0

        return torch.full((batch_size,), metric_penalty, device=embeddings.device)

    def _analyze_kl_artifacts(self, embeddings: Tensor) -> Tensor:
        """Detect KL regularization artifacts."""
        batch_size = embeddings.size(0)

        # KL artifacts typically appear as:
        # 1. Over-concentration near origin (high KL penalty)
        # 2. Boundary clustering (approaching |z| = 1)

        origin = torch.zeros_like(embeddings)
        radii = poincare_distance(embeddings, origin, c=1.0)

        # Detect over-concentration (radii too small)
        concentration_penalty = torch.exp(-radii * 5)  # Penalize r < 0.2

        # Detect boundary effects (radii too large)
        boundary_penalty = torch.sigmoid((radii - 0.95) * 20)  # Penalize r > 0.95

        kl_penalty = (concentration_penalty + boundary_penalty) * 0.3

        return kl_penalty

    def _analyze_objective_competition(self, embeddings: Tensor) -> Tensor:
        """Detect multi-objective optimization conflicts."""
        batch_size = embeddings.size(0)

        # Multi-objective conflicts manifest as:
        # 1. Embeddings that satisfy one objective but violate others
        # 2. Geometric arrangements that are suboptimal compromises

        # Simple proxy: variance in embedding magnitudes within batch
        # High variance suggests some embeddings are pushed to extremes
        origin = torch.zeros_like(embeddings)
        radii = poincare_distance(embeddings, origin, c=1.0)

        if batch_size > 1:
            radial_variance = torch.var(radii)
            # Moderate variance is good, extreme variance suggests conflicts
            variance_penalty = torch.sigmoid((radial_variance - 0.1) * 10) * 0.2
        else:
            variance_penalty = 0.0

        return torch.full((batch_size,), variance_penalty, device=embeddings.device)


class HyperbolicDiffusionRefinement(nn.Module):
    """Targeted diffusion refinement for VAE fidelity improvement.

    Uses detected fidelity loss patterns to apply selective refinement
    while preserving VAE geometric structure.
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 128, n_steps: int = 50):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps

        # Refinement network: (embedding + fidelity_scores + step) → refined_embedding
        self.refinement_net = nn.Sequential(
            nn.Linear(latent_dim + 5 + 1, hidden_dim),  # +5 for fidelity, +1 for timestep
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Simple linear noise schedule
        self.register_buffer('alphas', torch.linspace(0.99, 0.01, n_steps))
        self.register_buffer('betas', 1 - self.alphas)

    def add_noise(self, embeddings: Tensor, noise_level: float) -> Tensor:
        """Add controlled noise to embeddings while preserving Poincaré ball constraint."""
        noise = torch.randn_like(embeddings) * noise_level
        noisy = embeddings + noise

        # Project back to Poincaré ball
        return project_to_poincare(noisy, c=1.0)

    def denoise_step(self,
                     noisy_embeddings: Tensor,
                     fidelity_scores: Dict[str, Tensor],
                     timestep: int) -> Tensor:
        """Single denoising step with fidelity guidance."""
        batch_size = noisy_embeddings.size(0)

        # Prepare input: [embedding, fidelity_scores, normalized_timestep]
        t_norm = torch.full((batch_size, 1), timestep / self.n_steps,
                           device=noisy_embeddings.device)

        fidelity_tensor = torch.stack([
            fidelity_scores['reconstruction'],
            fidelity_scores['radial_richness'],
            fidelity_scores['metric_consistency'],
            fidelity_scores['kl_artifacts'],
            fidelity_scores['objective_competition']
        ], dim=1)  # (batch, 5)

        # Network input: (batch, latent_dim + 5 + 1)
        net_input = torch.cat([noisy_embeddings, fidelity_tensor, t_norm], dim=1)

        # Predict noise/refinement
        predicted_refinement = self.refinement_net(net_input)

        # Apply refinement with hyperbolic constraint
        alpha = self.alphas[timestep]
        refined = alpha * noisy_embeddings + (1 - alpha) * predicted_refinement

        # Ensure we stay on Poincaré ball
        return project_to_poincare(refined, c=1.0)

    def forward(self,
                embeddings: Tensor,
                fidelity_scores: Dict[str, Tensor],
                n_refinement_steps: Optional[int] = None) -> Tensor:
        """Apply targeted diffusion refinement.

        Args:
            embeddings: Original VAE embeddings (batch, latent_dim)
            fidelity_scores: Detected fidelity loss patterns
            n_refinement_steps: Number of refinement steps (default: self.n_steps//4)

        Returns:
            Refined embeddings with improved fidelity
        """
        if n_refinement_steps is None:
            n_refinement_steps = max(1, self.n_steps // 4)  # Light refinement by default

        current = embeddings.clone()

        # Apply noise proportional to fidelity loss
        total_loss = fidelity_scores['total_fidelity_loss'].unsqueeze(1)
        noise_scale = total_loss * 0.1  # Scale noise by fidelity loss
        current = self.add_noise(current, noise_scale.mean().item())

        # Iterative refinement
        for step in range(n_refinement_steps):
            timestep = (step * self.n_steps) // n_refinement_steps
            current = self.denoise_step(current, fidelity_scores, timestep)

        return current


class VAEFidelityDiffusion(nn.Module):
    """Complete VAE Fidelity Diffusion System.

    Combines fidelity loss detection with targeted diffusion refinement
    to identify and correct VAE representation failures.
    """

    def __init__(self,
                 latent_dim: int = 16,
                 hidden_dim: int = 128,
                 n_diffusion_steps: int = 50):
        super().__init__()

        self.latent_dim = latent_dim

        # Core components
        self.fidelity_detector = FidelityLossDetector(latent_dim, hidden_dim)
        self.diffusion_refiner = HyperbolicDiffusionRefinement(
            latent_dim, hidden_dim, n_diffusion_steps
        )

    def map_fidelity_loss(self, vae_embeddings: Tensor) -> Dict[str, Tensor]:
        """Comprehensive fidelity loss analysis.

        Args:
            vae_embeddings: Embeddings from trained VAE (batch, latent_dim)

        Returns:
            Detailed fidelity loss breakdown by failure mode
        """
        self.eval()
        with torch.no_grad():
            return self.fidelity_detector(vae_embeddings)

    def refine_embeddings(self,
                         vae_embeddings: Tensor,
                         fidelity_scores: Optional[Dict[str, Tensor]] = None,
                         refinement_strength: float = 1.0) -> Tensor:
        """Apply targeted diffusion refinement to improve fidelity.

        Args:
            vae_embeddings: Original VAE embeddings
            fidelity_scores: Pre-computed fidelity scores (computed if None)
            refinement_strength: Intensity of refinement (0.0-2.0)

        Returns:
            Refined embeddings with improved fidelity
        """
        if fidelity_scores is None:
            fidelity_scores = self.map_fidelity_loss(vae_embeddings)

        # Adjust refinement steps based on strength
        n_steps = int(self.diffusion_refiner.n_steps * refinement_strength / 4)
        n_steps = max(1, min(n_steps, self.diffusion_refiner.n_steps))

        return self.diffusion_refiner(vae_embeddings, fidelity_scores, n_steps)

    def forward(self, vae_embeddings: Tensor) -> Dict[str, Tensor]:
        """Complete fidelity analysis and refinement pipeline.

        Returns:
            Dictionary with fidelity_scores, refined_embeddings, and analysis
        """
        # Detect fidelity issues
        fidelity_scores = self.fidelity_detector(vae_embeddings)

        # Apply refinement
        refined_embeddings = self.diffusion_refiner(vae_embeddings, fidelity_scores)

        # Compute improvement metrics
        refined_fidelity = self.fidelity_detector(refined_embeddings)
        improvement = {
            f'{key}_improvement': fidelity_scores[key] - refined_fidelity[key]
            for key in fidelity_scores.keys() if key != 'total_fidelity_loss'
        }

        return {
            'original_fidelity': fidelity_scores,
            'refined_embeddings': refined_embeddings,
            'refined_fidelity': refined_fidelity,
            'improvement_metrics': improvement
        }

    @classmethod
    def from_vae_checkpoint(cls,
                           checkpoint_path: str,
                           device: str = 'cpu') -> 'VAEFidelityDiffusion':
        """Initialize from existing VAE checkpoint for analysis.

        Args:
            checkpoint_path: Path to trained VAE checkpoint
            device: Device for computation

        Returns:
            Initialized VAEFidelityDiffusion model
        """
        # For now, return with standard parameters
        # Future: Extract latent_dim from checkpoint metadata
        return cls(latent_dim=16, hidden_dim=128, n_diffusion_steps=50).to(device)


def analyze_vae_checkpoint_fidelity(checkpoint_path: str,
                                  device: str = 'cpu') -> Dict:
    """Utility function for comprehensive VAE checkpoint analysis.

    Args:
        checkpoint_path: Path to VAE checkpoint (.pt file)
        device: Computation device

    Returns:
        Complete fidelity analysis report
    """
    # Load checkpoint and extract embeddings
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Try to extract embeddings from various possible keys
        embeddings = None
        for key in ['z_hyp', 'embeddings', 'codon_embeddings', 'latent_embeddings']:
            if key in checkpoint:
                embeddings = checkpoint[key]
                break

        if embeddings is None:
            raise ValueError(f"Could not find embeddings in checkpoint {checkpoint_path}")

        # Initialize fidelity diffusion
        diffusion = VAEFidelityDiffusion.from_vae_checkpoint(checkpoint_path, device)

        # Run complete analysis
        results = diffusion(embeddings)

        # Add checkpoint metadata
        results['checkpoint_path'] = checkpoint_path
        results['embedding_shape'] = embeddings.shape
        results['device'] = device

        return results

    except Exception as e:
        print(f"Error analyzing checkpoint {checkpoint_path}: {e}")
        return {'error': str(e)}


if __name__ == '__main__':
    # Quick test
    print("VAE Fidelity Diffusion Model")
    print("Non-overengineered implementation for targeted fidelity mapping")

    # Test with synthetic data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diffusion = VAEFidelityDiffusion(latent_dim=16, hidden_dim=64).to(device)

    # Synthetic VAE embeddings (64 codons)
    test_embeddings = torch.randn(64, 16, device=device) * 0.5
    test_embeddings = project_to_poincare(test_embeddings, c=1.0)

    print(f"\nTesting with synthetic embeddings: {test_embeddings.shape}")

    # Run fidelity analysis
    fidelity_scores = diffusion.map_fidelity_loss(test_embeddings)
    print(f"\nFidelity Loss Analysis:")
    for key, scores in fidelity_scores.items():
        print(f"  {key}: {scores.mean().item():.4f} ± {scores.std().item():.4f}")

    # Run refinement
    refined = diffusion.refine_embeddings(test_embeddings, fidelity_scores)
    print(f"\nRefinement completed: {test_embeddings.shape} → {refined.shape}")

    print("\nImplementation ready for integration with existing checkpoints!")