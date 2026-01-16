"""VAE Fidelity Diffusion Package.

A targeted diffusion model for mapping and correcting VAE fidelity loss in
genetic code embeddings. Non-overengineered implementation focused on
identifying specific failure modes in TernaryVAE representations.

Key Components:
- FidelityLossDetector: Identifies 5 key VAE failure modes
- HyperbolicDiffusionRefinement: Targeted diffusion refinement
- VAEFidelityDiffusion: Complete analysis and refinement pipeline

Usage:
    from research.vae_fidelity_diffusion import VAEFidelityDiffusion

    diffusion = VAEFidelityDiffusion(latent_dim=16)
    fidelity_scores = diffusion.map_fidelity_loss(vae_embeddings)
    refined_embeddings = diffusion.refine_embeddings(vae_embeddings)
"""

from .vae_fidelity_diffusion import (
    FidelityLossDetector,
    HyperbolicDiffusionRefinement,
    VAEFidelityDiffusion,
    analyze_vae_checkpoint_fidelity
)

__version__ = '1.0.0'

__all__ = [
    'FidelityLossDetector',
    'HyperbolicDiffusionRefinement',
    'VAEFidelityDiffusion',
    'analyze_vae_checkpoint_fidelity'
]