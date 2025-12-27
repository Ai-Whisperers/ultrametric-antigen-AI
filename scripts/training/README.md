# Training Scripts

This directory contains training scripts for the Ternary VAE models.

## V5.11.11 Homeostatic Model

The flagship model combining all advanced features:
- **Homeostatic Control**: Maintains 100% coverage while learning 3-adic structure
- **Q-gated Annealing**: Dynamic threshold relaxation based on structure capacity
- **Riemannian Optimization**: Proper gradients on the Poincare ball
- **Learnable Curvature**: Geometry adapts during training

### Quick Start

```bash
# From project root
python scripts/training/train_v5_11_11_homeostatic.py
```

### Prerequisites

1. **CUDA-enabled PyTorch** (required for GPU training):
   ```bash
   # Check your CUDA version
   nvidia-smi

   # Install PyTorch with CUDA support (adjust version as needed)
   # For CUDA 12.6:
   pip install torch --index-url https://download.pytorch.org/whl/cu126

   # For CUDA 12.4:
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```

2. **Verify CUDA installation**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device: {torch.cuda.get_device_name(0)}")
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # Or with all extras:
   pip install -e ".[all]"
   ```

### Training Pipeline

The training consists of two phases:

#### Phase 1: V5.5 Base Model (Coverage Training)
If no pre-trained v5.5 checkpoint exists, the script automatically trains one to achieve 100% reconstruction coverage. This takes ~10-20 minutes on RTX 2060 SUPER.

**Checkpoint location**: `sandbox-training/checkpoints/v5_5/latest.pt`

#### Phase 2: V5.11.11 Homeostatic Training
Using the frozen v5.5 encoder_A, trains encoder_B to learn the 3-adic hierarchical structure.

**Checkpoint location**: `sandbox-training/checkpoints/v5_11_11_homeostatic_ale_device/`
- `best.pt`: Best model by composite score
- `latest.pt`: Final model after training

### Configuration

Edit `configs/v5_11_11_homeostatic_ale_device.yaml` for:
- Batch size (default: 512 for 8GB VRAM)
- Learning rate
- Epoch count
- Homeostasis parameters
- Loss weights

### Command Line Options

```bash
# Custom epochs
python scripts/training/train_v5_11_11_homeostatic.py --epochs 200

# Custom batch size (reduce if OOM)
python scripts/training/train_v5_11_11_homeostatic.py --batch_size 256

# Resume from checkpoint
python scripts/training/train_v5_11_11_homeostatic.py --resume

# Skip v5.5 pre-training (random initialization)
python scripts/training/train_v5_11_11_homeostatic.py --skip_v5_5
```

### Expected Metrics

After successful training:
| Metric | Target | Description |
|--------|--------|-------------|
| Coverage | 95%+ | Reconstruction accuracy |
| Radial Hierarchy | < -0.70 | Spearman correlation (more negative = better) |
| Q (Structure Capacity) | > 1.5 | Learned structure capacity |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 6 GB | 8+ GB |
| System RAM | 8 GB | 16+ GB |
| CUDA | 11.8 | 12.4+ |

**Tested Hardware**:
- NVIDIA GeForce RTX 2060 SUPER (8GB VRAM)
- AMD Ryzen 3593 MHz
- 16GB RAM

### TensorBoard Monitoring

```bash
tensorboard --logdir=runs/
```

Navigate to `http://localhost:6006` to view training progress.

### Troubleshooting

**CUDA not available**:
```bash
# Uninstall CPU-only PyTorch and reinstall with CUDA
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

**Out of Memory (OOM)**:
- Reduce batch size: `--batch_size 256`
- Enable gradient checkpointing in config: `memory.gradient_checkpointing: true`

**Module 'src' not found**:
- Ensure you're running from project root
- Or install the package: `pip install -e .`

---

## Other Training Scripts

### train_universal_vae.py
Generic VAE training for biological sequences.

### train_toxicity_regressor.py
Trains a regressor model for toxicity prediction.
