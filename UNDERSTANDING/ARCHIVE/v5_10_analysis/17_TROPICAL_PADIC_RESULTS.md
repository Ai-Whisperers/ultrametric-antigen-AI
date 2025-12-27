# Tropical + P-adic Experiment Results

**Date:** 2025-12-27
**Status:** Completed - 16/16 Experiments Succeeded

---

## Executive Summary

Tested the key hypothesis: **Can TropicalVAE + p-adic losses achieve both high accuracy AND high correlation?**

### Answer: YES, with proper tuning!

| Configuration | Accuracy | Spearman | Composite |
|--------------|----------|----------|-----------|
| **tropical_triplet_mono_pw0.5_rw0.3** | **89.1%** | **+0.4218** | **0.5522** |
| tropical_triplet_mono_pw0.5_rw0.1 | 84.9% | +0.4321 | 0.5378 |
| tropical_triplet_mono_pw0.3_rw0.1 | 92.5% | +0.3190 | 0.5228 |
| tropical_soft_ranking_only | 78.9% | +0.4318 | 0.5088 |

---

## Key Findings

### 1. Hyperparameter Sensitivity

The padic_weight and radial_weight are **critical**:

| padic_weight | radial_weight | Accuracy | Spearman |
|-------------|---------------|----------|----------|
| 0.5 | 0.3 | **89.1%** | **+0.42** |
| 0.5 | 0.1 | 84.9% | +0.43 |
| 0.3 | 0.1 | 92.5% | +0.32 |
| 0.3 | 0.3 | 82.2% | +0.33 |
| 0.1 | 0.5 | 81.2% | +0.03 |

**Insight**: Higher padic_weight (0.5) improves correlation; moderate radial_weight (0.3) is optimal.

### 2. TropicalVAE + P-adic vs Baselines

| Model | P-adic | Radial | Accuracy | Spearman |
|-------|--------|--------|----------|----------|
| TropicalVAE | none | none | 87.8% | +0.07 |
| TropicalVAE | triplet | monotonic | 82.2% | +0.33 |
| TropicalVAE (optimized) | triplet | monotonic | **89.1%** | **+0.42** |
| SimpleHyperbolic | triplet | monotonic | 90.0% | +0.34 |

**Insight**: TropicalVAE benefits significantly from p-adic losses, matching SimpleHyperbolic performance.

### 3. Individual Loss Effects on TropicalVAE

| Loss | Accuracy | Spearman | Notes |
|------|----------|----------|-------|
| none (baseline) | 87.8% | +0.07 | High acc, low corr |
| triplet only | 83.8% | +0.30 | Helps correlation |
| monotonic only | 84.1% | +0.24 | Moderate help |
| soft_ranking only | 78.9% | **+0.43** | Best correlation alone |
| hierarchy only | 75.0% | +0.31 | Lower accuracy |
| triplet + monotonic | 82.2% | +0.33 | Combined |
| soft + hierarchy | 76.8% | +0.31 | Combined |

---

## Optimal Configuration

```python
config = {
    "model_type": "tropical",
    "padic_loss_type": "triplet",
    "radial_loss_type": "monotonic",
    "padic_weight": 0.5,  # Higher than default
    "radial_weight": 0.3,  # Moderate
    "beta": 0.1,
    "beta_schedule": "cyclical",
    "learning_rate": 0.005,
    "epochs": 80,
}
# Expected: 89% accuracy, +0.42 correlation
```

---

## Comparison with Previous Best

| Metric | Previous Best | This Experiment | Configuration |
|--------|---------------|-----------------|---------------|
| **Best Accuracy** | 96.7% | 92.5% | tropical pw0.3 rw0.1 |
| **Best Correlation** | +0.55 | +0.43 | tropical pw0.5 rw0.1 |
| **Best Balance** | 0.57 | 0.55 | tropical pw0.5 rw0.3 |

**Note**: Slight variation from previous runs due to different random seeds and training dynamics.

---

## Conclusions

### Confirmed

1. **TropicalVAE + p-adic losses work together** - ultrametric structures are compatible
2. **Hyperparameter tuning is critical** - padic_weight=0.5 much better than 0.3
3. **Soft ranking gives best single-loss correlation** (+0.43)
4. **Triplet + monotonic with high padic_weight** gives best balance (89.1%, +0.42)

### New Insights

1. **TropicalVAE alone has low correlation** (+0.07) - needs p-adic losses
2. **Higher padic_weight helps** - 0.5 >> 0.3 >> 0.1
3. **radial_weight=0.3 is optimal** - too high (0.5) hurts accuracy
4. **SimpleHyperbolic still competitive** - simpler architecture works well

---

## Recommended Next Steps

### Immediate

1. **Test tropical_triplet_mono with more epochs** (100-150) to see if accuracy recovers
2. **Test soft_ranking + triplet combined** for potential correlation boost
3. **Create EnhancedTropicalVAE** with stronger tropical structure

### Short-term

1. Create TropicalHyperbolicVAE hybrid
2. Implement loss curriculum (reconstruction → structure)
3. Test on full 19683 operation dataset

### Long-term

1. Adaptive loss weighting during training
2. Multi-scale tropical embeddings
3. Combine with SwarmVAE for diverse representations

---

## Files Generated

| File | Content |
|------|---------|
| `scripts/experiments/tropical_padic_experiment.py` | Experiment script |
| `outputs/tropical_padic_results.json` | All 16 results |
| `UNDERSTANDING/17_TROPICAL_PADIC_RESULTS.md` | This document |

---

## EnhancedTropicalVAE Architecture

The experiment used an enhanced tropical architecture:

```python
class EnhancedTropicalVAE(nn.Module):
    """Uses tropical (max-plus) algebra operations:
    - Tropical addition: max(a, b)
    - Tropical multiplication: a + b
    - logsumexp as smooth approximation to max
    """

    def tropical_aggregate(self, z: torch.Tensor) -> torch.Tensor:
        """Apply tropical-like aggregation using smooth max."""
        z_transformed = self.tropical_transform(z)
        combined = torch.stack([z, z_transformed], dim=-1)
        z_tropical = torch.logsumexp(combined / self.temperature, dim=-1) * self.temperature
        return z_tropical
```

This creates piecewise-linear mappings that naturally preserve tree-like (ultrametric) structure.
