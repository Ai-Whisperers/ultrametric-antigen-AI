# Model Checkpoints Directory

> **Trained model checkpoints for the Ternary VAE project**

**Last Updated:** December 29, 2025

---

## Storage Location

**Model checkpoints (`.pt` files) are stored on Hugging Face Hub, NOT in git.**

- **Repository:** [`ivanpol/ternary-vae-checkpoints`](https://huggingface.co/ivanpol/ternary-vae-checkpoints)
- **Reason:** Checkpoint files are 100-200 MB each, too large for git

---

## Available Checkpoints

### Production-Ready

| Checkpoint | Coverage | Hierarchy | Richness | Use Case |
|------------|----------|-----------|----------|----------|
| `homeostatic_rich/best.pt` | 100% | -0.8321 | 0.00787 | **Recommended** - Best balance |
| `v5_11_homeostasis/best.pt` | 99.9% | -0.82 | 0.00136 | Good hierarchy |
| `v5_11_structural/best.pt` | 100% | -0.40 | 0.00304 | Bioinformatics (stable embeddings) |

### Experimental

| Checkpoint | Notes |
|------------|-------|
| `max_hierarchy/best.pt` | Maximum hierarchy (-0.8321) but low richness |
| `v5_11_overnight/best.pt` | **DO NOT USE** - Training collapsed |
| `v5_11_progressive/best.pt` | Inverted VAE-B hierarchy (+0.78) |

---

## Downloading Checkpoints

### Method 1: Using checkpoint_hub (Recommended)

```python
from src.utils.checkpoint_hub import ensure_checkpoint

# Downloads if not present, returns local path
checkpoint_path = ensure_checkpoint("homeostatic_rich/best.pt")
```

### Method 2: CLI Tool

```bash
# Download specific checkpoint
python -m src.utils.checkpoint_hub download --checkpoint homeostatic_rich/best.pt

# Download to custom directory
python -m src.utils.checkpoint_hub download --checkpoint homeostatic_rich/best.pt --output ./my_checkpoints

# List available checkpoints
python -m src.utils.checkpoint_hub list
```

### Method 3: Direct from Hugging Face

```bash
# Install huggingface_hub
pip install huggingface_hub

# Download all checkpoints
huggingface-cli download ivanpol/ternary-vae-checkpoints

# Or download specific file
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="ivanpol/ternary-vae-checkpoints",
    filename="homeostatic_rich/best.pt"
)
```

---

## Loading Checkpoints

### Basic Loading

```python
import torch
from src.models import TernaryVAEV5_11_PartialFreeze
from src.utils.checkpoint_hub import ensure_checkpoint

# Ensure checkpoint is available
ckpt_path = ensure_checkpoint("homeostatic_rich/best.pt")

# Load checkpoint
checkpoint = torch.load(ckpt_path, map_location="cpu")

# Create model
model = TernaryVAEV5_11_PartialFreeze(
    latent_dim=16,
    hidden_dim=64,
    max_radius=0.99,
    curvature=1.0,
    use_controller=True,
    use_dual_projection=True
)

# Load weights
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

### Checkpoint Contents

Each checkpoint contains:

```python
{
    "epoch": 100,                    # Training epoch
    "model_state_dict": {...},       # Model weights
    "optimizer_state_dict": {...},   # Optimizer state
    "metrics": {                     # Final metrics
        "coverage": 1.0,
        "hierarchy": -0.8321,
        "richness": 0.00787
    },
    "config": {...}                  # Training configuration
}
```

---

## Uploading Checkpoints

### Upload Single Checkpoint

```bash
python -m src.utils.checkpoint_hub upload --checkpoint outputs/models/my_model/best.pt
```

### Upload Directory

```bash
python -m src.utils.checkpoint_hub upload --directory outputs/models/my_model/
```

### From Python

```python
from src.utils.checkpoint_hub import upload_checkpoint

url = upload_checkpoint(
    local_path="outputs/models/my_model/best.pt",
    commit_message="Add my_model checkpoint"
)
```

---

## Directory Structure (Local)

When checkpoints are downloaded locally:

```
outputs/models/
├── README.md                 # This file
├── v5_5/
│   └── latest.pt            # V5.5 base model
├── v5_11/
│   └── best.pt              # V5.11 checkpoint
├── homeostatic_rich/
│   └── best.pt              # Recommended checkpoint
├── v5_11_homeostasis/
│   └── best.pt              # Homeostasis controller
├── v5_11_structural/
│   └── best.pt              # Structural training
├── codon_vae_all.pt         # Codon encoder (all sequences)
├── codon_vae_hiv.pt         # Codon encoder (HIV-specific)
└── aa_vae_all.pt            # Amino acid encoder
```

---

## Checkpoint Metrics Explained

### Coverage
- Percentage of 19,683 ternary operations correctly reconstructed
- Target: 100%
- Measured by argmax accuracy on decoder output

### Hierarchy
- Spearman correlation between 3-adic valuation and embedding radius
- Target: -0.83 to -1.0 (negative = correct ordering)
- v0 at outer edge, v9 at center
- **Ceiling: -0.8321** (mathematical limit with any variance)

### Richness
- Average within-valuation-level variance of radii
- Measures geometric diversity beyond simple ordering
- Higher = more meaningful structure preserved
- Zero = collapsed to trivial shells (bad)

---

## Training from Scratch

If you need to train models:

```bash
# Train homeostatic_rich configuration (recommended)
python scripts/experiments/epsilon_vae/train_homeostatic_rich.py

# Train with advanced modules (K-FAC, tropical geometry)
python scripts/experiments/epsilon_vae/train_with_advanced_modules.py --use-kfac

# Basic V5.11 training
python scripts/training/train_v5_11.py
```

Training typically takes:
- **V5.5 base**: ~15 minutes on GPU
- **V5.11 with homeostasis**: ~30-60 minutes on GPU
- **homeostatic_rich**: ~45 minutes on GPU

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v5.5 | Dec 2024 | Base model, 100% coverage |
| v5.11 | Dec 2024 | Dual encoder, partial freeze |
| v5.11.8 | Dec 2024 | Homeostasis controller |
| homeostatic_rich | Dec 2024 | Best hierarchy + richness balance |

---

## Troubleshooting

### "File not found" error
```bash
# Ensure checkpoint is downloaded
python -m src.utils.checkpoint_hub download --checkpoint <name>
```

### Authentication error (for uploads)
```bash
# Login to Hugging Face
huggingface-cli login
# Enter your token when prompted
```

### Large file warning
Checkpoints are 100-200 MB. Ensure sufficient disk space.

---

## See Also

- [Hugging Face Repository](https://huggingface.co/ivanpol/ternary-vae-checkpoints)
- `src/utils/checkpoint_hub.py` - Hub integration code
- `.claude/CLAUDE.md` - Checkpoint reference and warnings
