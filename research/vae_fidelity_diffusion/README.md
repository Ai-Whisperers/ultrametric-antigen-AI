# VAE Fidelity Diffusion Model

## Overview

A targeted diffusion model designed to map where the TernaryVAE loses fidelity in representing the 64-codon genetic code. This is a non-overengineered implementation focused specifically on identifying VAE reconstruction and geometric structure failures.

## Motivation

Based on comprehensive analysis of existing checkpoints and training scripts, we identified key fidelity loss areas:

### VAE Fidelity Loss Patterns Identified

1. **Reconstruction Bottleneck**: VAE decoder struggles with rare valuation levels (v7-v9, <1% of data)
2. **Radial-Richness Trade-off**: Models achieve either hierarchy (-0.83 correlation) OR richness (0.006+ variance) but not both
3. **Hyperbolic-Euclidean Inconsistencies**: Mixed norm() vs poincare_distance() usage creates metric errors
4. **KL Regularization Pressure**: Free bits vs exact posterior matching creates compression artifacts
5. **Multi-objective Competition**: P-adic structure vs AA property loss creates geometric compromises

### Topological Hints from Existing Infrastructure

**Groupoid Structure** (from `replacement_calculus/`):
- Morphisms as geodesic paths in VAE latent space
- Path costs correlate with experimental ΔΔG (ρ=0.6+)
- Escape paths follow natural genetic code transitions

**Hyperbolic Geometry** (from `geometry/`):
- Poincaré ball with curvature=1.0
- Radial position encodes p-adic valuation hierarchy
- Geodesic distances preserve biological relationships

## Architecture

### Simple Diffusion Framework
```
VAE Embedding → Fidelity Loss Detector → Diffusion Refinement → High-Fidelity Reconstruction
```

Key design principles:
- **Non-overengineered**: Minimal complexity, maximum insight
- **Topology-aware**: Respects hyperbolic geometry and groupoid structure
- **VAE-targeted**: Specifically identifies and corrects VAE failure modes

## Usage

```python
from research.vae_fidelity_diffusion import VAEFidelityDiffusion

# Load trained VAE checkpoint
diffusion = VAEFidelityDiffusion.from_vae_checkpoint(
    'research/codon-encoder/results/v5_12_4_embeddings/best.pt'
)

# Map fidelity loss for specific codons
loss_map = diffusion.map_fidelity_loss(codon_indices=[0, 1, 2, 3])

# Generate high-fidelity reconstructions
refined_embeddings = diffusion.refine_embeddings(vae_embeddings)
```

## Implementation Status

- [x] Analysis of VAE fidelity patterns
- [x] Identification of topological hints
- [ ] Core diffusion model implementation
- [ ] Fidelity loss mapping system
- [ ] Integration with existing checkpoints
- [ ] Validation against known VAE limitations

## Key References

**Existing Infrastructure:**
- `research/codon-encoder/training/train_codon_encoder.py` - Best checkpoint (LOO ρ=0.61)
- `research/codon-encoder/replacement_calculus/` - Groupoid framework
- `src/geometry/poincare.py` - Hyperbolic operations
- `src/encoders/trainable_codon_encoder.py` - Multi-objective loss analysis