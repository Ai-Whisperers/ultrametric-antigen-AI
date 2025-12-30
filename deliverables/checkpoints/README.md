# Pre-trained VAE Checkpoints

This directory contains pre-trained Ternary VAE checkpoints for use with the bioinformatics deliverables.

## Available Checkpoints

| Checkpoint | Description | Hierarchy | Richness | Use Case |
|------------|-------------|-----------|----------|----------|
| `homeostatic_rich.pt` | Best balance model | -0.83 | 0.0079 | **Recommended** - General use |
| `v5_11_homeostasis.pt` | Strong hierarchy focus | -0.82 | 0.0014 | Hierarchy analysis |
| `v5_11_structural.pt` | Stable embeddings | -0.74 | 0.0030 | Bioinformatics encoding |

## Metrics Explained

### Hierarchy (Spearman ρ)
- Measures correlation between p-adic valuation and radial position
- **Negative = correct** (v0 at edge, v9 at center)
- Target: -0.83 (mathematical ceiling is -0.8321)

### Richness
- Average within-valuation-level variance of radii
- Higher = more meaningful geometric structure
- Zero = trivial shells (collapsed solution)

## Quick Start

```python
from shared.vae_service import get_vae_service

# Service automatically finds best available checkpoint
vae = get_vae_service()

# Check status
if vae.is_real:
    print("Using trained VAE model")
else:
    print("No checkpoint found - using mock mode")

# Encode sequence
z = vae.encode_sequence("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH")
print(f"Latent vector: {z.shape}")  # (16,)

# Get stability metrics
stability = vae.get_stability_score(z)
valuation = vae.get_padic_valuation(z)
print(f"Stability: {stability:.3f}, Valuation: {valuation}")
```

## Manual Loading

If you need to load a specific checkpoint:

```python
import torch
from src.models import TernaryVAEV5_11_PartialFreeze

model = TernaryVAEV5_11_PartialFreeze(
    latent_dim=16,
    hidden_dim=64,
    max_radius=0.99,
    curvature=1.0,
    use_controller=True,
    use_dual_projection=True
)

# Load checkpoint
ckpt = torch.load("path/to/checkpoint.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

## Checkpoint Locations

The VAE service searches for checkpoints in the following order:

1. `sandbox-training/checkpoints/homeostatic_rich/best.pt`
2. `sandbox-training/checkpoints/v5_11_homeostasis/best.pt`
3. `checkpoints/pretrained_final.pt`

## Git LFS

Checkpoints are stored using Git LFS. If you see a small text file instead of binary data:

```bash
# Install Git LFS if not installed
git lfs install

# Pull actual checkpoint files
git lfs pull
```

## Training Your Own

To train a new checkpoint with similar properties:

```bash
cd scripts/epsilon_vae
python train_homeostatic_rich.py
```

Key training parameters:
- `hierarchy_weight=5.0` - Emphasize radial ordering
- `richness_weight=2.0` - Preserve within-level variance
- `separation_weight=3.0` - Ensure level separation

## Checkpoint Contents

Each checkpoint file contains:

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch_number,
    "metrics": {
        "coverage": 1.0,
        "hierarchy_A": -0.xx,
        "hierarchy_B": -0.83,
        "richness": 0.0xxx,
    }
}
```

## Architecture Requirements

All checkpoints require the following architecture:

```python
TernaryVAEV5_11_PartialFreeze(
    latent_dim=16,
    hidden_dim=64,        # or 128 for v5_11_structural
    max_radius=0.99,
    curvature=1.0,
    use_controller=True,
    use_dual_projection=True
)
```

## Troubleshooting

### "Checkpoint file appears to be a Git LFS pointer"

The checkpoint file is a Git LFS pointer, not the actual binary. Run:

```bash
git lfs pull
```

### "Could not load model: invalid load key"

Same as above - the file is a Git LFS pointer.

### "Unknown model architecture"

Make sure you're using `TernaryVAEV5_11_PartialFreeze` with the correct parameters.

### Model runs slowly

By default, the service uses CPU. For GPU:

```python
from shared.config import get_config

config = get_config()
config.use_gpu = True  # Will use CUDA if available
```

---

## License

All checkpoints are provided under the same license as the main project:
PolyForm Noncommercial License 1.0.0

---

*Ternary VAE Bioinformatics - Checkpoint Documentation*
*December 2025*
