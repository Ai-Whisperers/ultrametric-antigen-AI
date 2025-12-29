# Training Analysis and Improvements

**Date:** 2025-12-27
**Analysis of:** Ternary VAE v5.10.1 (300 epochs)

---

## Executive Summary

Analysis of the latest training run reveals an **anti-grokking pattern**: best performance occurred at the START of training (epoch 0-8), not after extended training. This is the opposite of classic grokking and indicates fundamental issues with the training dynamics.

### Key Findings

| Issue | Severity | Impact |
|-------|----------|--------|
| Anti-grokking pattern | HIGH | Best correlation at epoch 0 (0.715), degraded to 0.502 |
| Coverage stagnation | HIGH | Stuck at ~5% vs target 99.7% |
| Phase transition damage | MEDIUM | Epoch 50 β-warmup caused 4x loss spike |
| v5.5 → v5.10 regression | HIGH | Coverage dropped from 86% to 5% |

---

## 1. Grokking Research Context

### What is Grokking?

Grokking is a phenomenon where neural networks suddenly generalize long after achieving near-zero training loss. First documented by [Power et al. (2022)](https://arxiv.org/abs/2201.02177).

**Classic Grokking Pattern:**
```
Epoch 0-100:    Training loss → 0, Val accuracy = random
Epoch 100-1000: Plateau (memorization)
Epoch 1000+:    SUDDEN improvement in val accuracy → 100%
```

**Key Factors (from [Liu et al., 2024](https://arxiv.org/html/2402.15555v2)):**
1. **Weight decay** - encourages compression phase
2. **Small datasets** - algorithmic datasets favor grokking
3. **Extended training** - 10x-100x past memorization
4. **Local complexity (LC)** - double-descent pattern predicts grokking

### Grokking Detection Metrics

The new `GrokDetector` module (`src/training/grokking_detector.py`) tracks:

1. **Training Phase Classification**
   - WARMUP, MEMORIZATION, PLATEAU, GROKKING, DEGRADATION, CONVERGED

2. **Local Complexity (LC)**
   - Double-descent pattern indicates impending grokking
   - Implemented in `LocalComplexityEstimator`

3. **Weight Norm Dynamics**
   - Compression phase (decreasing norms) precedes grokking
   - Tracked by `WeightNormTracker`

4. **Grokking Probability**
   - Estimated based on current training state

---

## 2. Current Training Analysis (v5.10.1)

### Training Timeline

```
Epoch   0: Loss=14.55, Corr=0.715 (BEST!), Coverage=4.9%
Epoch   8: Loss=3.95 (BEST VAL LOSS!)
Epoch  25: Corr=0.703, Coverage=5.0% (peak)
Epoch  49: Loss=5.19, stable training
Epoch  50: Loss=22.50 (4x SPIKE - β-B warmup triggered)
Epoch 100: Loss=~25, Corr=~0.55, Coverage=4.9%
Epoch 200: Loss=~22, Corr=~0.52, Coverage=4.9%
Epoch 299: Loss=20.25, Corr=0.502, Coverage=4.9%
```

### Diagnosis: Anti-Grokking

This training exhibits **anti-grokking** - the opposite of the desired phenomenon:

| Grokking (desired) | Anti-Grokking (observed) |
|--------------------|--------------------------|
| Best at END | Best at START (epoch 0-8) |
| Train loss → 0 first | Train loss never → 0 |
| Sudden improvement | Gradual degradation |
| Weight compression | No compression observed |

### Root Causes

1. **β-B Warmup Disruption (Epoch 50)**
   - Loss jumped from 5.19 → 22.50 (333% increase)
   - Model never recovered to pre-disruption quality
   - Correlation dropped from 0.70 → 0.50

2. **Curriculum Learning Interference**
   - τ schedule may conflict with natural learning dynamics
   - Radial stratification may constrain latent space exploration

3. **Missing Grokking Conditions**
   - Weight decay may be insufficient
   - Training loss never reached memorization phase (<0.1)
   - No extended plateau observed

4. **Architecture Regression (v5.5 → v5.10)**
   - v5.5 achieved 86% coverage
   - v5.10 stuck at 5% coverage
   - New features (hyperbolic prior, curriculum) may interfere

---

## 3. Improvement Recommendations

### Immediate Actions (Priority 1)

#### A. Implement Early Stopping with Grokking Awareness

```python
from src.training import GrokDetector, TrainingPhase

detector = GrokDetector()

for epoch in range(num_epochs):
    # ... training ...
    analysis = detector.update(metrics)

    # Early stop on degradation
    if analysis.current_phase == TrainingPhase.DEGRADATION:
        if epoch - analysis.best_generalization_epoch > 50:
            print(f"Early stopping: best was epoch {analysis.best_generalization_epoch}")
            break

    # Continue if potential grokking
    if analysis.current_phase == TrainingPhase.MEMORIZATION:
        if analysis.grokking_probability > 0.3:
            print("Potential grokking - extending training")
            continue
```

#### B. Checkpoint at Multiple Intervals

Current system only saves "best" and "latest". Add:
- Checkpoint every N epochs (regardless of "best")
- Track multiple "best" metrics (loss, correlation, coverage)
- Enable rollback to any checkpoint

#### C. Disable/Soften β-B Warmup

The epoch 50 spike suggests the β-B warmup is too aggressive:

```yaml
# Current (problematic)
beta_b_warmup:
  start_epoch: 50
  initial_beta: 0.0

# Recommended
beta_b_warmup:
  start_epoch: 100  # Later start
  initial_beta: 0.5  # Softer transition
  warmup_epochs: 50  # Gradual increase
```

### Medium-Term Actions (Priority 2)

#### D. Increase Weight Decay for Grokking

Weight decay is critical for grokking. Current weight decay may be too low:

```yaml
# Current
optimizer:
  weight_decay: 0.01  # (check actual value)

# Recommended for grokking
optimizer:
  weight_decay: 0.1  # 10x increase

# Or use decoupled weight decay (AdamW)
optimizer:
  type: AdamW
  weight_decay: 0.1
```

#### E. Extended Training Experiment

Grokking can require 10-100x more epochs than memorization. Try:

```
Phase 1: Train to near-zero training loss (memorization)
Phase 2: Continue for 10x additional epochs
Phase 3: Monitor for sudden generalization
```

#### F. Compare v5.5 vs v5.10 Architectures

```python
# Run controlled experiment
configs = [
    "configs/ternary_v5_5.yaml",  # 86% coverage baseline
    "configs/ternary_v5_10.yaml", # 5% coverage current
]

# Identify which component caused regression:
# - Hyperbolic prior?
# - Curriculum learning?
# - Radial stratification?
# - β-B warmup timing?
```

### Long-Term Actions (Priority 3)

#### G. Local Complexity Monitoring

Add LC estimation to training loop:

```python
lc_estimator = LocalComplexityEstimator()

for epoch in range(num_epochs):
    # ... training ...
    lc = lc_estimator.estimate(model, sample_batch)

    # Log for TensorBoard
    writer.add_scalar("training/local_complexity", lc, epoch)

    # Check for double-descent (grokking predictor)
    detected, start_epoch = lc_estimator.detect_double_descent()
    if detected:
        print(f"Double-descent detected at epoch {start_epoch}")
```

#### H. Batch Normalization Analysis

Batch normalization can prevent grokking ([Liu et al., 2024](https://arxiv.org/html/2402.15555v2)). Check:

```python
# List all BatchNorm layers
for name, module in model.named_modules():
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
        print(f"BatchNorm found: {name}")
        # Consider: LayerNorm alternative
```

#### I. Curriculum Learning Refinement

Current curriculum may be too aggressive. Consider:

1. **Slower τ warmup**: `tau_scale: 0.01` instead of `0.1`
2. **Later curriculum start**: After initial convergence
3. **Adaptive curriculum**: Based on current performance

---

## 4. Grokking Detection Usage

### Basic Usage

```python
from src.training import GrokDetector, EpochMetrics

detector = GrokDetector()

# During training loop
metrics = EpochMetrics(
    epoch=epoch,
    train_loss=train_loss,
    val_loss=val_loss,
    correlation=correlation,
    coverage=coverage,
)

analysis = detector.update(metrics)

print(f"Phase: {analysis.current_phase.value}")
print(f"Grokking probability: {analysis.grokking_probability:.2%}")
print(f"Best epoch: {analysis.best_generalization_epoch}")

for warning in analysis.warnings:
    print(f"WARNING: {warning}")

for rec in analysis.recommendations:
    print(f"RECOMMENDATION: {rec}")
```

### Analyzing Existing Logs

```python
from pathlib import Path
from src.training import analyze_training_log

log_path = Path("outputs/results/training/logs/training_full_run.log")
analysis = analyze_training_log(log_path)

print(f"Training phase: {analysis.current_phase.value}")
print(f"Trend: {analysis.trend_direction}")
```

### Integration with TensorBoard

```python
# Add to training loop
writer.add_scalar("grokking/probability", analysis.grokking_probability, epoch)
writer.add_scalar("grokking/gen_gap", analysis.current_generalization_gap, epoch)
writer.add_text("grokking/phase", analysis.current_phase.value, epoch)
```

---

## 5. Experimental Protocol

### Experiment 1: Weight Decay Sweep

```bash
# Test weight decay values
for wd in 0.001 0.01 0.1 0.5; do
    python train.py --config ternary_v5_10.yaml \
        --weight-decay $wd \
        --epochs 500 \
        --run-name "wd_sweep_${wd}"
done
```

### Experiment 2: Extended Training

```bash
# Train for 3x current epochs
python train.py --config ternary_v5_10.yaml \
    --epochs 1000 \
    --checkpoint-every 50 \
    --run-name "extended_training"
```

### Experiment 3: Disable Curriculum

```bash
# Train without curriculum learning
python train.py --config ternary_v5_10.yaml \
    --disable-curriculum \
    --disable-beta-warmup \
    --epochs 300 \
    --run-name "no_curriculum"
```

### Experiment 4: v5.5 Baseline Reproduction

```bash
# Reproduce v5.5 results for comparison
python train.py --config ternary_v5_5.yaml \
    --epochs 300 \
    --run-name "v5_5_baseline"
```

---

## 6. Summary of Changes Made

### New Module: `src/training/grokking_detector.py`

| Class | Purpose |
|-------|---------|
| `GrokDetector` | Main detector for grokking/anti-grokking patterns |
| `GrokDetectorConfig` | Configuration for detection thresholds |
| `EpochMetrics` | Data class for training metrics |
| `GrokAnalysis` | Analysis results with recommendations |
| `TrainingPhase` | Enum for training phases |
| `LocalComplexityEstimator` | Estimates LC for grokking prediction |
| `WeightNormTracker` | Tracks weight dynamics |
| `analyze_training_log` | Analyzes existing log files |

### Updated Exports

Added to `src/training/__init__.py`:
- All grokking detection classes
- `analyze_training_log` utility function

---

## 7. References

1. [Power et al. (2022) - Grokking: Generalization Beyond Overfitting](https://arxiv.org/abs/2201.02177)
2. [Nanda et al. (2023) - Progress Measures for Grokking](https://arxiv.org/abs/2301.05217)
3. [Liu et al. (2024) - Deep Networks Always Grok](https://arxiv.org/html/2402.15555v2)
4. [Wikipedia - Grokking (machine learning)](https://en.wikipedia.org/wiki/Grokking_(machine_learning))

---

## Next Steps

1. [ ] Run weight decay sweep experiment
2. [ ] Implement early stopping with grokking awareness
3. [ ] Compare v5.5 vs v5.10 to identify regression cause
4. [ ] Add LC monitoring to training loop
5. [ ] Test extended training (1000+ epochs)
6. [ ] Soften β-B warmup schedule
