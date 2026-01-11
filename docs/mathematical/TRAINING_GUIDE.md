# TernaryVAE Mathematical Foundation Training Guide

**Doc-Type:** Training Guide · Version 1.0 · Updated 2026-01-10 · AI Whisperers

## Quick Start

### 1. Basic Mathematical Foundation Training

```bash
# Recommended: Balanced mathematical foundation
python scripts/mathematical/train_v5_12_5.py \
    --config configs/mathematical/v5_12_5_foundation.yaml \
    --profile mathematical_foundation

# Expected results:
# Coverage: ≥ 99.99%
# Hierarchy_B: ≤ -0.832
# Richness: ≥ 0.006
# Training time: ~3-4 hours on RTX 4090
```

### 2. Quick Validation Test

```bash
# Fast test (5 epochs)
python scripts/mathematical/train_v5_12_5.py \
    --config configs/mathematical/v5_12_5_foundation.yaml \
    --test-mode

# Infrastructure validation
python scripts/mathematical/validate_mathematical_framework.py
```

### 3. Alternative Training Profiles

```bash
# Coverage-focused (perfect reconstruction)
python scripts/mathematical/train_v5_12_5.py \
    --profile coverage_focused

# Hierarchy-focused (maximum p-adic ordering)
python scripts/mathematical/train_v5_12_5.py \
    --profile hierarchy_focused

# Richness-focused (geometric diversity)
python scripts/mathematical/train_v5_12_5.py \
    --profile richness_focused
```

## Training Profiles Deep Dive

### Mathematical Foundation (Recommended)

**Use Case**: Balanced mathematical substrate for research and applications

**Configuration**:
```yaml
loss_weights:
  coverage: 1.0      # Perfect reconstruction
  hierarchy: 5.0     # Strong p-adic ordering
  richness: 2.5      # Geometric diversity preservation
  separation: 3.0    # Valuation-level distinction

targets:
  coverage: 1.0
  hierarchy_B: -0.8321
  richness: 0.006
  Q_enhanced: 2.0
```

**Expected Training Curve**:
- Epochs 0-20: Coverage reaches 99%+
- Epochs 20-50: Hierarchy develops (0 → -0.6)
- Epochs 50-100: Hierarchy ceiling approach (-0.6 → -0.83)
- Epochs 100-120: Richness stabilization and Q optimization

### Coverage Focused

**Use Case**: When perfect reconstruction is the primary goal

**Characteristics**:
- Achieves 100% coverage fastest
- Lower hierarchy development
- Suitable for downstream applications requiring perfect reconstruction

**Training Time**: ~1-2 hours

### Hierarchy Focused

**Use Case**: Maximum p-adic mathematical structure

**Characteristics**:
- Approaches hierarchy ceiling (-0.8321) most aggressively
- May sacrifice some richness
- Ideal for pure mathematical research

**Training Time**: ~2-3 hours

### Richness Focused

**Use Case**: Geometric diversity preservation research

**Characteristics**:
- Maintains high within-level variance
- Balanced hierarchy development
- Best for topological data analysis applications

**Training Time**: ~3-4 hours

## Step-by-Step Training Process

### Phase 1: Environment Setup

```bash
# 1. Verify CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 2. Check memory requirements
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')"

# 3. Validate mathematical framework
python scripts/mathematical/validate_mathematical_framework.py
```

**Requirements**:
- GPU with ≥8GB VRAM (RTX 3070, RTX 4090, V100, A100)
- PyTorch ≥2.0 with CUDA support
- 16GB+ system RAM recommended

### Phase 2: Configuration Selection

**For Research/Development**:
```bash
cp configs/mathematical/v5_12_5_foundation.yaml configs/mathematical/my_training.yaml
# Edit configuration as needed
```

**Key Configuration Parameters**:
```yaml
# Model architecture
model:
  latent_dim: 16              # Embedding dimension (fixed)
  hidden_dim: 64              # Internal layer size
  curvature: 1.0              # Hyperbolic curvature (default)
  max_radius: 0.95            # Poincaré ball boundary

# Training parameters
training:
  epochs: 120                 # Full training duration
  batch_size: 512             # Memory vs speed tradeoff
  lr: 8.0e-4                  # Learning rate (conservative)

# Mathematical precision
model:
  mathematical_precision: true  # Enhanced stability
  enhanced_controller: true     # 12-dim controller input
```

### Phase 3: Training Execution

```bash
# Start training with monitoring
python scripts/mathematical/train_v5_12_5.py \
    --config configs/mathematical/my_training.yaml \
    --profile mathematical_foundation \
    2>&1 | tee training.log
```

**Monitoring Training Progress**:

```bash
# Monitor metrics (in another terminal)
tail -f training.log | grep "Epoch"

# Check GPU utilization
nvidia-smi -l 1

# Monitor checkpoint progress
ls -lah checkpoints/v5_12_5/
```

### Phase 4: Training Validation

**Real-time Validation**:
During training, look for these patterns:

```
Epoch  10 | Loss: 2.4521 | Coverage: 0.8945 | Hierarchy_B: -0.1234 | Richness: 0.00156 | Q: 1.234
Epoch  20 | Loss: 1.8743 | Coverage: 0.9823 | Hierarchy_B: -0.3456 | Richness: 0.00234 | Q: 1.567
Epoch  50 | Loss: 1.2156 | Coverage: 0.9987 | Hierarchy_B: -0.6789 | Richness: 0.00345 | Q: 1.789
Epoch 100 | Loss: 0.8234 | Coverage: 0.9999 | Hierarchy_B: -0.8123 | Richness: 0.00567 | Q: 1.945
```

**Success Indicators**:
- ✅ Coverage consistently > 99.5% after epoch 30
- ✅ Hierarchy_B trending toward -0.83 (ceiling)
- ✅ Richness remaining > 0.002 (not collapsing)
- ✅ Q metric improving over time
- ✅ No NaN/Inf values in any metric

**Warning Signs**:
- ⚠️ Coverage drops below 95%
- ⚠️ Hierarchy becomes positive
- ⚠️ Richness drops below 0.001
- ⚠️ Training loss explodes or collapses to 0
- ⚠️ GPU memory errors

### Phase 5: Checkpoint Evaluation

```bash
# Load and evaluate best checkpoint
python scripts/mathematical/analyze_mathematical_checkpoints.py \
    checkpoints/v5_12_5/best_Q.pt
```

**Checkpoint Validation**:
```python
# Quick checkpoint test
import torch
from src.models import TernaryVAEV5_11_PartialFreeze

checkpoint = torch.load('checkpoints/v5_12_5/best_Q.pt')
print(f"Final metrics: {checkpoint['comprehensive_metrics']}")
print(f"Validation: {checkpoint['validation_results']}")

# Load model for inference
model = TernaryVAEV5_11_PartialFreeze()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Training Configuration Patterns

### Memory-Constrained Systems (8GB VRAM)

```yaml
training:
  batch_size: 256              # Reduced from 512
  gradient_accumulation_steps: 2  # Maintain effective batch size

model:
  hidden_dim: 32               # Reduced model size
  projection_layers: 1         # Simplified projections
```

### High-Performance Systems (24GB+ VRAM)

```yaml
training:
  batch_size: 1024             # Increased batch size
  num_workers: 8               # More data loading threads

model:
  hidden_dim: 128              # Larger model capacity
  enhanced_controller: true    # Full controller features
```

### Quick Development/Testing

```yaml
training:
  epochs: 20                   # Quick iteration
  batch_size: 128              # Fast startup
  lr: 1.0e-3                   # Faster convergence

targets:
  coverage: 0.95               # Relaxed targets
  hierarchy_B: -0.7
```

## Advanced Training Techniques

### Curriculum Learning

```yaml
# Progressive difficulty scheduling
curriculum:
  enabled: true
  coverage_first_epochs: 30    # Focus on reconstruction first
  hierarchy_ramp_epochs: 50    # Gradually increase hierarchy weight
  richness_preserve_epochs: 20 # Final richness optimization
```

### Learning Rate Scheduling

```yaml
scheduler:
  type: "mathematical_cosine"   # Cosine annealing with restarts
  T_0: 25                      # Initial cycle length
  T_mult: 1.5                  # Cycle length multiplier
  eta_min: 1.0e-6              # Minimum learning rate
  warmup_epochs: 5             # LR warmup
```

### Homeostatic Fine-Tuning

```yaml
homeostasis:
  enabled: true
  mathematical_precision: true

  # Dynamic thresholds
  coverage_precision_threshold: 0.9995
  hierarchy_precision_threshold: 0.0005
  Q_target: 2.0

  # Adaptive annealing
  enable_annealing: true
  annealing_step: 0.003
  annealing_patience: 5
```

## Troubleshooting Guide

### Common Issues and Solutions

**Issue: Training Loss Explodes**
```
Symptom: Loss jumps to >10.0 or NaN
Cause: Learning rate too high or numerical instability
Solution:
  - Reduce learning rate: lr: 1.0e-4
  - Enable gradient clipping: max_grad_norm: 0.5
  - Check for NaN inputs
```

**Issue: Coverage Drops During Training**
```
Symptom: Coverage < 95% after epoch 50
Cause: Encoder-A not properly frozen or loss weight imbalance
Solution:
  - Verify frozen checkpoint loading
  - Increase coverage_weight: 2.0
  - Check encoder_a_lr_scale: 0.0 (completely frozen)
```

**Issue: Hierarchy Not Developing**
```
Symptom: Hierarchy_B remains > -0.3 after epoch 100
Cause: Hierarchy weight too low or encoder-B learning too fast
Solution:
  - Increase hierarchy_weight: 8.0
  - Reduce encoder_b_lr_scale: 0.05
  - Enable homeostatic control
```

**Issue: Memory Overflow**
```
Symptom: CUDA out of memory error
Cause: Batch size too large for available GPU memory
Solution:
  - Reduce batch_size: 256 or 128
  - Use gradient accumulation
  - Enable mixed precision: use_amp: true
```

### Advanced Debugging

**Enable Detailed Logging**:
```yaml
monitoring:
  log_every_n_steps: 50        # More frequent logging
  detailed_metrics_every_n_epochs: 1  # Every epoch
  precision_tracking: true     # Track numerical precision
```

**Checkpoint Recovery**:
```bash
# Resume from specific checkpoint
python scripts/mathematical/train_v5_12_5.py \
    --checkpoint checkpoints/v5_12_5/epoch_050.pt \
    --config configs/mathematical/v5_12_5_foundation.yaml
```

## Performance Benchmarks

### Training Time Estimates

| GPU Model | Memory | Batch Size | Time/100 Epochs |
|-----------|--------|------------|-----------------|
| RTX 4090  | 24GB   | 1024       | ~2.5 hours      |
| RTX 4080  | 16GB   | 512        | ~3.5 hours      |
| RTX 3080  | 10GB   | 256        | ~5 hours        |
| RTX 3070  | 8GB    | 128        | ~7 hours        |
| V100      | 32GB   | 1024       | ~3 hours        |
| A100      | 80GB   | 2048       | ~1.5 hours      |

### Memory Usage Patterns

```
Model Size: ~15M parameters
Base Memory: ~2GB
Training Memory: 4-8GB (depends on batch size)
Peak Memory: Training memory + 2GB (gradients + optimizer)
```

## Best Practices

### 1. Always Start with Validation

```bash
# Before any training
python scripts/mathematical/validate_mathematical_framework.py
```

### 2. Monitor Key Metrics

Focus on these 4 metrics:
- **Coverage**: Must reach 99.9%+
- **Hierarchy_B**: Should approach -0.83
- **Richness**: Keep above 0.002
- **Q_enhanced**: Higher is better

### 3. Save Intermediate Checkpoints

```yaml
checkpointing:
  save_every_n_epochs: 20      # Don't lose progress
  keep_best_n: 3              # Keep multiple good checkpoints
```

### 4. Use Proven Configurations

Start with `v5_12_5_foundation.yaml` and modify incrementally.

### 5. Document Experiments

```bash
# Always save training logs
python train_v5_12_5.py --config my_config.yaml 2>&1 | tee experiment_$(date +%Y%m%d_%H%M).log
```

---

**Next**: See [MATHEMATICAL_THEORY.md](MATHEMATICAL_THEORY.md) for theoretical foundations and [API_REFERENCE.md](API_REFERENCE.md) for implementation details.