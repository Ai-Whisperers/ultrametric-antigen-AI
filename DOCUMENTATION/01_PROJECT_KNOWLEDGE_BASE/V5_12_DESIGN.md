# V5.12 Model Design Document

**Doc-Type:** Technical Design · Version 1.0 · Updated 2025-12-29 · AI Whisperers

---

## Executive Summary

V5.12 aims to be the highest quality Ternary VAE model, combining the best innovations from V5.11.x while addressing remaining gaps.

**Target Metrics:**
| Metric | V5.11 Best | V5.12 Target | Improvement |
|--------|------------|--------------|-------------|
| Coverage | 100% | 100% | Maintain |
| Hierarchy_B | -0.8321 | -0.8321 | Maintain ceiling |
| Richness | 0.00787 | >0.008 | +2% |
| r_v0 | 0.89 | 0.88-0.90 | Maintain |
| r_v9 | 0.19 | 0.12-0.15 | Improve separation |
| dist_corr | ~0.6 | >0.7 | +15% |

---

## Key Innovations in V5.12

### 1. RichHierarchyLoss Integration

The `homeostatic_rich` checkpoint achieved best results using `RichHierarchyLoss`.
V5.12 makes this the primary loss:

```python
from src.losses import RichHierarchyLoss

loss_fn = RichHierarchyLoss(
    hierarchy_weight=5.0,      # Strong hierarchy signal
    coverage_weight=1.0,       # Maintain reconstruction
    richness_weight=2.0,       # Preserve variance
    separation_weight=3.0,     # Level ordering
    min_richness_ratio=0.5,    # Don't collapse variance
)
```

### 2. ComprehensiveMetrics for Checkpoint Storage

All checkpoints store metrics using `src.core.metrics.ComprehensiveMetrics`:

```python
from src.core import compute_comprehensive_metrics

metrics = compute_comprehensive_metrics(model, device)
checkpoint = {
    'model_state_dict': model.state_dict(),
    'metrics': metrics.to_dict(),  # Standardized storage
    ...
}
```

### 3. Enhanced Radial Separation

Problem: r_v9 = 0.19 vs target 0.10
Solution: Tighter inner_radius with exponential weighting for high-v

```python
inner_radius = 0.08  # Tighter than 0.1
outer_radius = 0.90  # Slightly expanded

# Exponential weighting for high-valuation points
high_v_weight = torch.exp(valuations * 0.5)  # v=9 gets ~90x weight
```

### 4. Combined Loss Strategy

V5.12 uses a two-phase loss approach:

**Phase 1 (epochs 0-50): Establish Structure**
- RichHierarchyLoss (primary)
- RadialHierarchyLoss (auxiliary, low weight)
- GlobalRankLoss (structural constraint)

**Phase 2 (epochs 50+): Refine Geometry**
- RichHierarchyLoss (primary)
- PAdicGeodesicLoss (geodesic refinement)
- Reduced RadialHierarchyLoss

### 5. Homeostatic Control with Q-Tracking

Uses `HomeostasisController` with enhanced Q metric:

```python
Q = dist_corr + 1.5 * |hierarchy| + 0.5 * richness_ratio
```

Triggers:
- Freeze encoder_A when coverage < 99.5%
- Unfreeze encoder_B when hierarchy plateaus
- Q-gated threshold annealing

### 6. Riemannian Optimization

Uses geoopt's RiemannianAdam for proper manifold gradients:

```python
from src.geometry import get_riemannian_optimizer

optimizer = get_riemannian_optimizer(
    model.get_param_groups(lr),
    lr=1e-3,
    optimizer_type='adam',
    weight_decay=1e-4,
)
```

---

## Architecture: TernaryVAEV5_11_PartialFreeze

No model architecture changes from V5.11. V5.12 focuses on training.

```
Input (batch, 9) → FrozenEncoder_A → mu_A, logvar_A → z_A_euc (16D)
                 → FrozenEncoder_B → mu_B, logvar_B → z_B_euc (16D)

z_A_euc → HyperbolicProjection_A → z_A_hyp (Poincare ball)
z_B_euc → HyperbolicProjection_B → z_B_hyp (Poincare ball)

z_A_hyp → FrozenDecoder_A → logits_A (batch, 9, 3)
```

Trainable components:
- HyperbolicProjection_A/B (direction_net + radius_net)
- DifferentiableController (optional)
- encoder_B (when unfrozen by homeostasis)

---

## Training Configuration

### Hyperparameters

```yaml
# Model
latent_dim: 16
hidden_dim: 64
max_radius: 0.95
curvature: 1.0
use_controller: true
use_dual_projection: true
learnable_curvature: true

# Training
epochs: 200
batch_size: 512
lr: 1e-3
weight_decay: 1e-4
grad_clip: 1.0

# Loss Weights
hierarchy_weight: 5.0
coverage_weight: 1.0
richness_weight: 2.0
separation_weight: 3.0
rank_loss_weight: 0.5  # Lower than V5.11
geodesic_weight: 0.3   # Phase 2 only

# Radial Targets
inner_radius: 0.08     # Tighter for v=9
outer_radius: 0.90     # Slightly expanded

# Homeostasis
coverage_freeze_threshold: 0.995
coverage_floor: 0.95
enable_annealing: true
annealing_step: 0.003  # Slower annealing
```

### Stratified Sampling

High-valuation budget increased to 25% (from 20%):

```python
high_v_budget = int(batch_size * 0.25)  # 25% for v>=4
```

---

## Evaluation Criteria

### Primary Metrics (must pass all)

1. **Coverage >= 99.9%** - Reconstruction quality
2. **Hierarchy_B <= -0.80** - 3-adic ordering
3. **Richness >= 0.007** - Geometric diversity
4. **r_v9 <= 0.15** - High-valuation separation

### Secondary Metrics (optimization targets)

5. **dist_corr >= 0.65** - Geodesic correlation
6. **r_v0 in [0.85, 0.92]** - Low-valuation placement
7. **Q >= 1.8** - Structure capacity

---

## Implementation Checklist

- [ ] Create `configs/v5_12.yaml`
- [ ] Create `scripts/training/train_v5_12.py`
- [ ] Integrate RichHierarchyLoss as primary loss
- [ ] Add ComprehensiveMetrics to checkpoint storage
- [ ] Implement two-phase loss strategy
- [ ] Test on RTX 2060 SUPER (8GB VRAM)
- [ ] Validate against target metrics
- [ ] Document final checkpoint

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Richness collapse | min_richness_ratio=0.5 + monitoring |
| r_v9 not improving | Tighter inner_radius + exponential weighting |
| Coverage drop | Homeostasis freeze when < 99.5% |
| Training instability | Grad clipping + Riemannian optimizer |
| Memory OOM | batch_size=512, AMP if needed |

---

## Success Criteria

V5.12 is successful when:

1. All primary metrics pass
2. Checkpoint validates with `compute_comprehensive_metrics()`
3. Training completes without homeostasis collapse events
4. Richness maintained throughout training (no late collapse)

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-29 | 1.0 | Initial V5.12 design document |
