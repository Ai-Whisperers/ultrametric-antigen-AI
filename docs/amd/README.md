# AMD GPU Configuration Guide

**Doc-Type:** Hardware Configuration · Version 1.0 · Created 2026-01-30 · AI Whisperers

---

## Hardware Specifications

This document details the AMD GPU configuration for a development laptop that outperforms the default training device (RTX 3050 6GB) in some metrics but requires comprehensive configuration for ROCm compatibility.

### System Specs

| Component | Specification | Notes |
|-----------|---------------|-------|
| **Discrete GPU** | AMD Radeon RX 6500M (Navi 24, RDNA 2) | **4GB GDDR6 VRAM** |
| **Integrated GPU** | AMD Cezanne (Radeon Vega) | Shared system memory |
| **CPU** | AMD Ryzen 5 5600H | 6 cores / 12 threads, 3.3-4.2 GHz |
| **RAM** | 14GB DDR4 | ~10GB available |
| **Storage** | 344GB NVMe | ~298GB free |
| **OS** | Ubuntu 24.04.3 LTS | Kernel 6.8.0-90-generic |
| **Python** | 3.12 | System Python |

### Comparison with Default Training Device

| Metric | RTX 3050 (Reference) | RX 6500M (This System) | Advantage |
|--------|---------------------|------------------------|-----------|
| VRAM | 6GB GDDR6 | 4GB GDDR6 | RTX 3050 (+50%) |
| Memory Bandwidth | 192 GB/s | 144 GB/s | RTX 3050 |
| PCIe Lanes | 16x | 4x | RTX 3050 |
| Tensor Cores | Yes (3rd gen) | No | RTX 3050 |
| FP16 Support | Native | Native | Tie |
| Driver Maturity | CUDA (mature) | ROCm (improving) | RTX 3050 |

### Key Considerations

1. **PCIe 4x Limitation**: The RX 6400/6500 series uses only 4 PCIe lanes, which can bottleneck data transfer for large batch training
2. **ROCm Support**: RDNA 2 (gfx1032) has experimental/limited ROCm support
3. **No Tensor Cores**: Cannot use TF32/mixed precision optimizations that NVIDIA provides
4. **VRAM Constraint**: 4GB vs 6GB requires smaller batch sizes and careful memory management

---

## Installation Guide

### Step 1: Install ROCm

```bash
# Add ROCm repository (Ubuntu 24.04)
wget https://repo.radeon.com/amdgpu-install/6.0/ubuntu/noble/amdgpu-install_6.0.60000-1_all.deb
sudo apt install ./amdgpu-install_6.0.60000-1_all.deb

# Install ROCm with HIP
sudo amdgpu-install --usecase=rocm,hip --no-dkms

# Add user to required groups
sudo usermod -aG video,render $USER

# Reboot required
sudo reboot
```

### Step 2: Verify ROCm Installation

```bash
# Check ROCm installation
rocminfo | grep "Name:" | head -5
rocm-smi --showproductname

# Check HIP
hipcc --version
```

### Step 3: Install PyTorch with ROCm

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch with ROCm 6.0 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'ROCm: {torch.cuda.is_available()}')"
```

### Step 4: Install Project Dependencies

```bash
# Install project requirements
pip install -r requirements.txt

# Verify full setup
python -c "from src.models import TernaryVAEV5_11_PartialFreeze; print('Model imports OK')"
```

---

## ROCm Compatibility Notes

### RX 6500M (gfx1032) Specific Issues

| Issue | Impact | Workaround |
|-------|--------|------------|
| Limited official support | May see warnings | Set `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| 4x PCIe lanes | Slower data transfer | Use smaller batch sizes, prefer on-GPU operations |
| No matrix cores | Slower mixed precision | Use FP32 or careful FP16 |
| Driver compatibility | Occasional crashes | Use stable ROCm versions |

### Environment Variables

Add to `~/.bashrc` or run before training:

```bash
# Force gfx1030 compatibility mode for gfx1032
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Disable GPU memory pre-allocation (helps with limited VRAM)
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128

# Use discrete GPU (not integrated)
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
```

---

## Training Configuration

### Recommended Settings for 4GB VRAM

```yaml
# src/configs/amd_optimized.yaml
training:
  batch_size: 128        # Conservative for 4GB VRAM
  gradient_accumulation: 4  # Effective batch 512
  mixed_precision:
    enabled: true
    dtype: float16       # Required for 4GB

torch_compile:
  enabled: false         # ROCm compile support limited

dataloader:
  num_workers: 4         # Match CPU cores
  pin_memory: true

memory:
  max_vram_usage: 0.90   # Tight but workable
  gradient_checkpointing: true  # Required for 4GB
```

### Benchmark Command

```bash
# Quick validation (5 epochs)
python src/scripts/training/train_v5_12.py \
  --config src/configs/v5_12_4_fixed_checkpoint.yaml \
  --epochs 5 \
  --batch-size 256

# Monitor GPU during training
watch -n 1 rocm-smi
```

---

## Troubleshooting

### "No GPU detected"

```bash
# Check if GPU is visible
rocm-smi --showproductname

# If not, check driver
dmesg | grep -i amdgpu

# Reinstall driver if needed
sudo amdgpu-install --uninstall
sudo amdgpu-install --usecase=rocm,hip
```

### "Out of Memory"

```bash
# Reduce batch size
--batch-size 128

# Enable gradient checkpointing in config
gradient_checkpointing: true

# Clear GPU memory cache
python -c "import torch; torch.cuda.empty_cache()"
```

### "Unsupported GPU architecture"

```bash
# Set compatibility override
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# Or for older ROCm
export HSA_OVERRIDE_GFX_VERSION=10.3.1
```

---

## Performance Expectations

Based on similar RDNA 2 configurations:

| Metric | Expected | Notes |
|--------|----------|-------|
| Training throughput | ~40-50% of RTX 3050 | Due to PCIe 4x + no tensor cores + less VRAM |
| Batch size | Up to 128-256 | With FP16, gradient checkpointing |
| Epoch time (19,683 ops) | ~3-5 minutes | vs ~1 minute on RTX 3050 |
| Stability | Good | With proper env vars |

---

## Status

- [ ] ROCm installation verified
- [ ] PyTorch ROCm working
- [x] **Environment validated** (CPU mode) - 2026-01-30
- [x] **Training script runs** (1 epoch CPU test passed)
- [ ] Full GPU training benchmarked

### Validation Log (2026-01-30)

**CPU Test Results:**
```
PyTorch: 2.10.0+cpu
Dataset: 19,683 operations
1 epoch completed successfully
Initial metrics (random init, 1 epoch):
  - Coverage: 0.01%
  - Hierarchy_B: -0.0555
  - Richness: 0.003689
```

**Next Steps:**
1. Install ROCm 6.0 (requires system reboot)
2. Install PyTorch ROCm build
3. Run full GPU training validation

**Last Updated:** 2026-01-30
