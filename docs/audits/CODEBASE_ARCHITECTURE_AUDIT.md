# TernaryVAE Codebase Architecture Audit

**Doc-Type:** Codebase Audit · Version 1.0 · Updated 2026-01-12 · AI Whisperers

---

## Executive Summary

**Critical Finding**: The current TernaryVAE architecture has a fundamental dependency issue that explains the 0% coverage problem in our training runs. The V5.11+ architecture is designed to use **frozen components from v5.5 checkpoint** for coverage, while training only the geometric projection layers for hierarchy. Training from scratch without this checkpoint results in random decoder outputs → 0% coverage.

**Architecture Philosophy**:
- **Freeze what works** (100% coverage from v5.5)
- **Train only what's needed** (hyperbolic projection for hierarchy)
- **Preserve proven capabilities** while adding new geometric structure

---

## Core Architecture Analysis

### 1. **Model Hierarchy (src/models/)**

**Primary Models**:
- `TernaryVAEV5_11`: Base frozen coverage + trainable projection
- `TernaryVAEV5_11_PartialFreeze`: Extends V5.11 with dynamic freeze/unfreeze
- `ImprovedEncoder/Decoder`: V5.12.4+ components with SiLU, LayerNorm, Dropout

**Architecture Dependencies**:
```python
# CRITICAL: V5.11+ expects v5.5 checkpoint
# From src/models/ternary_vae.py lines 19-25:
# "Key insight: v5.5 achieved 100% coverage but inverted radial hierarchy.
# V5.11 freezes the coverage and learns only the geometric projection."

Input: x (batch, 9) - Ternary operation {-1, 0, 1}
    │
    ▼
┌──────────────────────────────────────────┐
│       FROZEN ENCODERS (from v5.5)       │
│  FrozenEncoder_A → mu_A, logvar_A (16D) │
│  FrozenEncoder_B → mu_B, logvar_B (16D) │
│  NO GRADIENTS - Preserves 100% coverage │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│    TRAINABLE HYPERBOLIC PROJECTION      │
│  z_euclidean → MLP(64) → z_poincare     │
│  TRAINABLE - Learns geometric structure │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│         FROZEN DECODER                   │
│  z_A → Linear(16→64→27) → logits        │
│  NO GRADIENTS - Reconstruction only     │
└──────────────────────────────────────────┘
```

### 2. **Coverage Computation Mechanism**

**Coverage Algorithm** (src/utils/metrics.py:19-39):
```python
def evaluate_coverage(samples: torch.Tensor) -> Tuple[int, float]:
    # Round to nearest ternary value
    samples_rounded = torch.round(samples).long()
    # Count unique operations
    unique_samples = torch.unique(samples_rounded, dim=0)
    unique_count = unique_samples.size(0)
    coverage_pct = (unique_count / 19683) * 100
    return unique_count, coverage_pct
```

**Coverage Computation in Training** (scripts/quick_train.py:200-206):
```python
# Forward pass through decoder
mu_A = outputs["mu_A"]
logits = model.decoder_A(mu_A)
preds = torch.argmax(logits, dim=-1) - 1
# Check perfect reconstruction
correct = (preds == x.long()).float().mean(dim=1)
coverage = (correct == 1.0).sum().item() / len(x)
```

**Why 0% Coverage Occurs**:
1. Random initialization → decoder produces random outputs
2. `torch.argmax(random_logits)` → random predictions
3. Random predictions ≠ input operations → 0% perfect reconstructions
4. Coverage = 0%

---

## Training Script Analysis

### 1. **Core Training Scripts**

| Script | Purpose | Architecture | Checkpoint Dependency |
|--------|---------|--------------|----------------------|
| `scripts/train.py` | Canonical V5.11 training | TernaryVAEV5_11 | **REQUIRES v5.5** |
| `scripts/quick_train.py` | Validation/testing | TernaryVAEV5_11_PartialFreeze | **EXPECTS v5.5** |
| `scripts/training/train_v5_12.py` | Production V5.12 | Enhanced components | **EXPECTS checkpoint** |
| `scripts/train_v5_12_4_grokking.py` | Extended grokking | Enhanced + detection | **EXPECTS checkpoint** |

### 2. **Checkpoint Loading Behavior**

**Expected Workflow**:
```python
# From scripts/quick_train.py:106-111
v5_5_path = PROJECT_ROOT / "outputs" / "models" / "v5_5" / "latest.pt"
if v5_5_path.exists():
    model.load_v5_5_checkpoint(v5_5_path, device)
else:
    print("Training with random initialization (coverage may be low)")
```

**Current Configuration Issue**:
```yaml
# From configs/v5_12_4_extended_grokking.yaml:51-55
frozen_checkpoint:
  path: null  # ← This breaks coverage!
  encoder_to_load: none
  decoder_to_load: none
```

### 3. **Training Strategy Analysis**

**Intended Design** (V5.11 architecture):
- **Phase 1**: Load v5.5 checkpoint → 100% coverage guaranteed
- **Phase 2**: Freeze coverage components (encoder_A, decoder_A)
- **Phase 3**: Train only projection layers → learn hierarchy while preserving coverage

**Current Implementation** (Our training runs):
- **Phase 1**: No checkpoint loaded → random initialization
- **Phase 2**: Train geometric projections → hierarchy learning works
- **Phase 3**: Coverage computation fails → 0% because decoder is random

---

## Configuration Analysis

### 1. **Available Configurations**

| Config | Purpose | Architecture | Expected Results |
|--------|---------|--------------|------------------|
| `v5_12_4.yaml` | Production | ImprovedEncoder/Decoder | Checkpoint required |
| `v5_12_4_extended_grokking.yaml` | Grokking detection | Extended training | **Modified for null checkpoint** |
| `ternary.yaml` | Basic config | Standard components | Checkpoint expected |

### 2. **Checkpoint Strategy Options**

**Option A: Use Existing Checkpoint**
```yaml
frozen_checkpoint:
  path: sandbox-training/checkpoints/v5_12_4/best_Q.pt
  encoder_to_load: both
  decoder_to_load: decoder_A
```

**Option B: Train Full Model from Scratch**
- Need different architecture that doesn't assume frozen components
- Requires reconstruction loss (currently missing in V5.11+)
- Would need to implement full VAE training, not just geometric projection

**Option C: Use v5.5 Checkpoint** (Intended design)
```yaml
frozen_checkpoint:
  path: outputs/models/v5_5/latest.pt
  encoder_to_load: both
  decoder_to_load: decoder_A
```

---

## Critical Issues Identified

### 1. **Architecture Mismatch** ⚠️
- **Issue**: V5.11+ designed for frozen components, not from-scratch training
- **Impact**: 0% coverage is expected behavior when training without checkpoint
- **Solution**: Either provide proper checkpoint or use different architecture

### 2. **Missing Reconstruction Loss** ⚠️
- **Issue**: Current loss functions focus on geometric structure, not reconstruction
- **Evidence**: Scripts use geodesic, radial, hierarchy losses but no reconstruction loss
- **Impact**: Model has no incentive to learn reconstruction → coverage stays 0%

### 3. **Training Strategy Confusion** ⚠️
- **Issue**: Configs modified for "from scratch" but architecture assumes frozen components
- **Evidence**: `path: null` in configs but model expects v5.5 weights
- **Impact**: Training proceeds but produces misleading results

---

## Available Checkpoints Analysis

### 1. **Checkpoint Inventory**
```
sandbox-training/checkpoints/
├── v5_12_4/best_Q.pt           # Current production (1.0MB)
├── homeostatic_rich/best.pt    # Hierarchy-richness balance (421KB)
├── v5_11_structural/best.pt    # Structural baseline (6.9MB)
├── v5_11_homeostasis/best.pt   # Homeostatic control (3.5MB)
```

### 2. **Checkpoint Compatibility**
- **v5_12_4 checkpoints**: Use ImprovedEncoder/Decoder → dimension mismatch with V5.11
- **v5_11 checkpoints**: Compatible with V5.11 architecture
- **v5_5 checkpoint**: Missing (outputs/models/v5_5/latest.pt not found)

---

## Recommendations

### 1. **Immediate Fix: Use Compatible Checkpoint**
```yaml
# Option 1: Use existing v5_11 checkpoint
frozen_checkpoint:
  path: sandbox-training/checkpoints/v5_11_homeostasis/best.pt
  encoder_to_load: both
  decoder_to_load: decoder_A
```

### 2. **Alternative: Full From-Scratch Training**
- Implement true from-scratch VAE training with reconstruction loss
- Modify training scripts to include reconstruction objectives
- Accept longer training time to achieve both coverage and hierarchy

### 3. **Pipeline Improvement**
- Create checkpoint compatibility validation
- Add reconstruction loss options for from-scratch training
- Implement progressive training: coverage first, then hierarchy

---

## Next Steps

1. **Test Compatible Checkpoint**: Use v5_11_homeostasis checkpoint for coverage
2. **Validate Full Pipeline**: Run training with proper checkpoint to achieve coverage + hierarchy
3. **Implement From-Scratch Option**: Add reconstruction loss for true from-scratch capability
4. **Create Training Guide**: Document proper checkpoint usage and training strategies

---

**Status**: Architecture audit complete - root cause identified and solutions proposed
**Priority**: High - fundamental issue affecting all training results
**Timeline**: Can be resolved immediately with checkpoint configuration fix