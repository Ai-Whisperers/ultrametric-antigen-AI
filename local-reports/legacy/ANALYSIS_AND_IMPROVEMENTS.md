# Ternary VAE Analysis and Improvement Plan

## Test Results Summary

We ran comprehensive generalization tests on the Ternary VAE v5.5 architecture (untrained baseline). The results reveal **critical weaknesses** that must be addressed through improved training.

### Performance Metrics (Untrained Model Baseline)

| Test | Result | Target | Status |
|------|--------|--------|--------|
| **Holdout Reconstruction** | 32.78% | >70% | ❌ FAIL |
| **Compositional Ops** | 27.78% | >50% | ❌ FAIL |
| **Permutation Robustness** | 0.624 corr | >0.85 | ⚠️ MODERATE |
| **Latent Variance** | 0.000 | >0.1 | ❌ **CRITICAL** |
| **Effective Dimensions** | 9.4/16 | >12/16 | ⚠️ MODERATE |
| **Sparse Op Reconstruction** | 40.74% | >70% | ❌ FAIL |
| **Dense Op Reconstruction** | 33.09% | >70% | ❌ FAIL |
| **Noise Resilience (5%)** | 0.033 dist | <0.1 | ✅ GOOD |

---

## Critical Issues Identified

### 1. **POSTERIOR COLLAPSE** (Priority: CRITICAL)

**Finding**: Latent variance = 0.000
**Cause**: KL divergence penalty overwhelming reconstruction term
**Impact**: Model cannot learn meaningful representations

**Evidence**:
- VAE-A variance: 0.000
- VAE-B variance: 0.000
- All latent codes collapsing to prior N(0,1)

**Solutions**:
- [ ] Implement β-VAE warmup: start β=0, gradually increase to target
- [ ] Use Free Bits: ignore KL below threshold (e.g., 2 nats per dimension)
- [ ] Lower initial β values (currently starts too high)
- [ ] Add reconstruction weight to balance KL
- [ ] Implement KL annealing schedule over epochs

### 2. **Poor Generalization** (Priority: HIGH)

**Finding**: 32.78% accuracy on unseen operations
**Cause**: Model memorizes training set rather than learning structure
**Impact**: Cannot generalize beyond trained examples

**Evidence**:
- Holdout accuracy: 32.78% (random = 33.33%)
- Compositional accuracy: 27.78% (worse than random!)
- Model performs barely better than guessing

**Solutions**:
- [ ] Implement data augmentation (permutations, noise injection)
- [ ] Add regularization to encourage compositional structure
- [ ] Increase training set size/diversity
- [ ] Use mixup/cutmix for ternary operations
- [ ] Add consistency losses between augmented views

### 3. **Low Reconstruction Quality** (Priority: HIGH)

**Finding**: 30-40% reconstruction accuracy
**Cause**: Model architecture or training insufficient
**Impact**: Cannot reliably reconstruct even training data

**Evidence**:
- Sparse ops: 40.74%
- Dense ops: 33.09%
- Dense ops harder (more information to encode)

**Solutions**:
- [ ] Increase model capacity (wider/deeper networks)
- [ ] Improve decoder architecture (add skip connections)
- [ ] Use better output activation (current setup may be suboptimal)
- [ ] Add auxiliary losses (e.g., per-bit reconstruction)
- [ ] Implement progressive training (easy→hard operations)

### 4. **Permutation Sensitivity** (Priority: MEDIUM)

**Finding**: 0.624 distance correlation under permutation
**Cause**: Model not learning permutation-equivariant representations
**Impact**: Latent space structure not invariant to symbol relabeling

**Solutions**:
- [ ] Add permutation augmentation during training
- [ ] Implement equivariant network layers
- [ ] Add permutation consistency loss
- [ ] Use symmetric function approximators

### 5. **Sparse vs Dense Bias** (Priority: MEDIUM)

**Finding**: Sparse ops 40.74% vs Dense ops 33.09%
**Cause**: Model finds sparse patterns easier to memorize
**Impact**: Systematic bias against information-rich operations

**Solutions**:
- [ ] Balanced sampling (ensure equal sparse/dense distribution)
- [ ] Weighted loss (penalize errors on dense ops more)
- [ ] Separate heads for sparse/dense prediction
- [ ] Curriculum learning (start with sparse, add dense)

---

## Recommended Training Improvements

### Phase 1: Fix Posterior Collapse (Immediate)

```python
# config changes needed:

vae_a:
  beta_start: 0.0  # Was too high
  beta_end: 0.5    # Reduce from 1.0
  beta_warmup_epochs: 50  # New: gradual warmup

vae_b:
  beta_start: 0.0
  beta_end: 0.3    # Even lower for VAE-B
  beta_warmup_epochs: 50

training:
  kl_free_bits: 2.0        # New: free bits per dimension
  reconstruction_weight: 1.5  # New: boost reconstruction
  kl_annealing: "cyclical"   # New: cyclical annealing
```

### Phase 2: Data Augmentation (High Priority)

Implement augmentation pipeline:
1. **Permutation augmentation**: randomly permute {-1,0,1} labels
2. **Noise injection**: add 1-3% random flips
3. **Mixup**: blend operations in latent space
4. **Curriculum**: start with sparse, gradually add dense

### Phase 3: Architecture Improvements (Medium Priority)

1. **Increase capacity**:
   - Encoder: 9 → 256 → 512 → 256 → latent
   - Decoder: latent → 256 → 512 → 256 → 27 (add depth)

2. **Add skip connections** (ResNet-style) in decoder

3. **Improve output representation**:
   - Current: 9 × 3 logits
   - Better: Separate per-position classifiers with shared backbone

### Phase 4: New Loss Components (Medium Priority)

```python
total_loss = (
    reconstruction_weight * reconstruction_loss +
    beta * kl_divergence +
    consistency_weight * consistency_loss +      # New
    permutation_weight * permutation_loss +      # New
    compositionality_weight * composition_loss   # New
)
```

**Consistency loss**: augmented views should have similar latents
**Permutation loss**: permuted inputs → equivariant latents
**Composition loss**: encode(A+B) ≈ encode(A) + encode(B)

### Phase 5: Training Schedule Improvements

1. **Curriculum learning**:
   - Epochs 0-50: Sparse operations only
   - Epochs 50-150: Mix of sparse and dense
   - Epochs 150+: Full dataset + hard examples

2. **Progressive difficulty**:
   - Start with high-frequency operations
   - Gradually add rare operations
   - Oversample hard-to-learn examples

3. **Longer training**:
   - Current: 300 epochs may be insufficient
   - Recommend: 500-1000 epochs with proper scheduling

---

## Implementation Priority

### Immediate (Fix posterior collapse):
1. Implement β-warmup schedule
2. Add free bits to KL term
3. Lower initial β values
4. Run training and verify latent variance > 0.1

### Week 1 (Core improvements):
5. Implement data augmentation pipeline
6. Add consistency losses
7. Increase model capacity
8. Retrain and measure holdout accuracy > 70%

### Week 2 (Advanced features):
9. Add compositional losses
10. Implement permutation equivariance
11. Progressive curriculum learning
12. Target: >85% reconstruction, >75% holdout

---

## Success Criteria

A successful model should achieve:

- ✅ **Latent variance** > 0.1 (no collapse)
- ✅ **Holdout accuracy** > 70% (true generalization)
- ✅ **Compositional accuracy** > 50% (structural learning)
- ✅ **Sparse reconstruction** > 80%
- ✅ **Dense reconstruction** > 70%
- ✅ **Permutation correlation** > 0.85
- ✅ **Effective dimensions** > 12/16 (good utilization)
- ✅ **Noise resilience** < 0.1 distance at 5% noise

---

## Next Steps

1. **Commit current tests and analysis**
2. **Implement β-warmup and free bits** in model
3. **Update training config** with new schedules
4. **Run short training test** (50 epochs) to verify no collapse
5. **Implement data augmentation**
6. **Full training run** with monitoring
7. **Re-run generalization tests** on trained model
8. **Iterate based on results**

---

## References

- **β-VAE**: Higgins et al. (2017) - "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
- **Free Bits**: Kingma et al. (2016) - "Improved Variational Inference with Inverse Autoregressive Flow"
- **Cyclical Annealing**: Fu et al. (2019) - "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing"
- **Curriculum Learning**: Bengio et al. (2009) - "Curriculum Learning"
