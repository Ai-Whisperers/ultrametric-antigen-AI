# Ternary VAE v5.6 Release Notes

**Release Date:** 2025-12-10
**Previous Version:** v5.5.0-srp
**Commits:** `637e971`, `57bc87c`

---

## Summary

v5.6 is a production-ready release that adds observability and performance optimization features while maintaining full backward compatibility with v5.5 checkpoints.

---

## New Features

### 1. TensorBoard Integration (Local, IP-Safe)

**Purpose:** Real-time training visualization without cloud dependencies.

**Metrics Logged:**
- Loss curves (train/val, CE, KL per VAE)
- Coverage tracking (unique operations, percentage)
- Entropy dynamics (H_A, H_B)
- Training dynamics (phase, rho, lambda weights)
- Temperature and beta schedules
- Learning rate (scheduled vs corrected)
- StateNet corrections (delta values)
- Weight histograms (every 10 epochs)

**Usage:**
```bash
# Training automatically logs to runs/
python scripts/train/train_ternary_v5_6.py --config configs/ternary_v5_6.yaml

# Launch dashboard
tensorboard --logdir runs
# Open http://localhost:6006
```

**Configuration:**
```yaml
tensorboard_dir: runs
experiment_name: null  # Auto-generates timestamp
```

### 2. TorchInductor Compilation (PyTorch 2.x)

**Purpose:** 1.4-2x training speedup via torch.compile.

**Features:**
- Automatic model compilation on initialization
- Configurable backend (inductor, cudagraphs, eager)
- Mode selection (default, reduce-overhead, max-autotune)
- Graceful fallback on compilation failure

**Configuration:**
```yaml
torch_compile:
  enabled: true
  backend: inductor
  mode: default
  fullgraph: false
```

**Modes:**
- `default`: Balanced compile time and runtime performance
- `reduce-overhead`: Lower Python overhead, good for small models
- `max-autotune`: Maximum performance, longer compile time

---

## File Changes

### Renamed Files
| Old Path | New Path |
|----------|----------|
| `scripts/train/train_ternary_v5_5_refactored.py` | `scripts/train/train_ternary_v5_6.py` |
| `src/models/ternary_vae_v5_5.py` | `src/models/ternary_vae_v5_6.py` |
| `configs/ternary_v5_5.yaml` | `configs/ternary_v5_6.yaml` |

### Modified Files (18 total)
- `src/__init__.py` - Version bump to 5.6.0
- `src/models/__init__.py` - Import path update
- `src/training/trainer.py` - TorchInductor + TensorBoard integration
- `src/training/monitor.py` - TensorBoard logging methods
- `src/utils/__init__.py`, `src/utils/data.py` - Docstring updates
- `setup.py` - Version and entry point updates
- `README.md` - Documentation updates
- `scripts/benchmark/*.py` - Import and path updates
- `tests/*.py` - Import and checkpoint path updates
- `requirements.txt` - Added tensorboard dependency

### New Checkpoint Directory
```
sandbox-training/checkpoints/v5_6/
```

---

## API Changes

### TrainingMonitor

**New Constructor Parameters:**
```python
TrainingMonitor(
    eval_num_samples: int = 100000,
    tensorboard_dir: Optional[str] = None,    # NEW
    experiment_name: Optional[str] = None     # NEW
)
```

**New Methods:**
```python
def log_tensorboard(self, epoch, train_losses, val_losses,
                    unique_A, unique_B, cov_A, cov_B) -> None
def log_histograms(self, epoch, model) -> None
def close(self) -> None
```

### TernaryVAETrainer

**New Behavior:**
- Automatically compiles model with torch.compile if enabled
- Calls `monitor.log_tensorboard()` after each epoch
- Calls `monitor.log_histograms()` every 10 epochs
- Calls `monitor.close()` at training end

**New Config Options:**
```yaml
torch_compile:
  enabled: bool
  backend: str
  mode: str
  fullgraph: bool
```

---

## Backward Compatibility

### Checkpoint Compatibility
- v5.5 checkpoints load correctly in v5.6
- Model architecture unchanged (168,770 parameters)
- StateNet structure preserved

### Config Compatibility
- v5.5 configs work with v5.6 (new options have defaults)
- `tensorboard_dir: null` disables TensorBoard
- `torch_compile.enabled: false` disables compilation

### Migration
No code changes required. Simply:
1. Update imports: `ternary_vae_v5_5` → `ternary_vae_v5_6`
2. Update config path: `ternary_v5_5.yaml` → `ternary_v5_6.yaml`

---

## Performance

### Training Speed (with torch.compile)
- **Expected:** 1.4-2x speedup after warmup
- **Warmup:** First 2-3 epochs slower due to compilation
- **Note:** Speedup varies by GPU and batch size

### Inference Speed (unchanged)
- VAE-A: ~4.4M samples/sec
- VAE-B: ~6.1M samples/sec

### Coverage (achieved at epoch 100+)
- VAE-A: 99.75% (19,634/19,683)
- VAE-B: 99.64% (19,613/19,683)

---

## Dependencies

### New Required
```
tensorboard>=2.13.0
```

### PyTorch Version
```
torch>=2.0.0  # Required for torch.compile
```

Verified with PyTorch 2.5.1+cu121.

---

## Known Issues

1. **torch.compile warmup**: First epochs are slower due to JIT compilation
2. **TensorBoard port**: Default 6006 may conflict with other services
3. **Histogram memory**: Weight histograms increase memory usage slightly

---

## Quick Start

```bash
# Install dependencies
pip install tensorboard

# Train with full observability
python scripts/train/train_ternary_v5_6.py --config configs/ternary_v5_6.yaml

# Monitor in real-time (separate terminal)
tensorboard --logdir runs
```

---

## Contributors

- Architecture: AI Whisperers
- Implementation: Claude Code

---

**Full Changelog:** v5.5.0-srp → v5.6.0
