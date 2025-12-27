# Sandbox Training Directory

This directory contains training checkpoints and artifacts. **Note: Checkpoint files (`.pt`) are gitignored** to keep the repository lightweight.

## Directory Structure

```
sandbox-training/
├── checkpoints/
│   ├── v5_5/
│   │   └── latest.pt              # Base v5.5 model (100% coverage)
│   └── v5_11_11_homeostatic_ale_device/
│       ├── best.pt                # Best model (highest composite score)
│       ├── latest.pt              # Final model after training
│       └── epoch_*.pt             # Periodic checkpoints
├── aa_vae_all.pt                  # Amino acid VAE checkpoint
├── codon_vae_all.pt               # Codon VAE (all sequences)
├── codon_vae_hiv.pt               # Codon VAE (HIV-specific)
└── README.md
```

## Sharing Checkpoints

Since `.pt` files are gitignored, use one of these methods to share trained models:

### Option 1: Direct File Transfer
Copy the checkpoint files to a shared location (OneDrive, Google Drive, etc.)

### Option 2: GitHub Releases
Upload checkpoint files to GitHub Releases for versioned artifact storage.

### Option 3: Hugging Face Hub
```bash
pip install huggingface_hub
huggingface-cli upload <repo-id> sandbox-training/checkpoints/
```

## Training from Scratch

If checkpoints are not available, you can train the models:

```bash
# Train V5.11.11 Homeostatic Model
python scripts/training/train_v5_11_11_homeostatic.py

# This will:
# 1. Train v5.5 base model if missing (~15 min)
# 2. Train v5.11.11 homeostatic model (~30 min)
```

See `scripts/training/README.md` for detailed training instructions.

## Expected Model Performance

| Checkpoint | Coverage | Hierarchy (r) | Q |
|------------|----------|---------------|---|
| v5_5/latest.pt | 98%+ | ~-0.3 | N/A |
| v5_11_11_homeostatic | 95%+ | < -0.70 | > 1.5 |

## Loading Checkpoints

```python
import torch

# Load v5.11.11 homeostatic model
checkpoint = torch.load("sandbox-training/checkpoints/v5_11_11_homeostatic_ale_device/best.pt")
print(f"Coverage: {checkpoint['metrics']['coverage']*100:.1f}%")
print(f"Hierarchy: {checkpoint['metrics']['radial_corr_A']:.3f}")
```
