# Ternary VAE Architecture Reference

**Doc-Type:** Technical Reference · Version 1.0 · Updated 2025-12-27 · AI Whisperers

> **For Claude instances**: This document provides essential context about the VAE architecture,
> checkpoint versions, and critical warnings about training artifacts.

---

## Project Overview

This repository implements a Variational Autoencoder for learning 3-adic (p-adic) hierarchical structure over ternary operations. The model embeds 19,683 ternary operations (3^9) into a hyperbolic Poincaré ball where radial position encodes 3-adic valuation.

---

## Architecture Components

### Dual-Encoder System (v5.11+)

The TernaryVAEV5_11 architecture uses two complementary encoders:

**VAE-A (Coverage Encoder)**
- Primary role: Reconstruct all 19,683 operations (coverage)
- Behavior: Often learns frequency-based ordering (can show inverted/positive hierarchy)
- Freezing: Freeze to preserve coverage while training VAE-B for hierarchy

**VAE-B (Hierarchy Encoder)**
- Primary role: Learn correct p-adic radial ordering
- Behavior: Should achieve negative Spearman correlation (v0 at edge, v9 at center)
- Training: Benefits from slower learning rate (encoder_b_lr_scale=0.1)

**DifferentiableController**
- Architecture: 8→32→32→6 MLP
- Outputs: rho, weight_geodesic, beta_A, beta_B, tau (loss weights)
- Usage: `use_controller=True` in model config

**HomeostasisController**
- Role: Training orchestrator for freeze/unfreeze decisions
- Triggers: Based on coverage thresholds and hierarchy plateau detection
- Q-metric: `Q = dist_corr + 1.5 × |hierarchy|`

---

## Key Metrics

**Coverage**: Percentage of 19,683 operations correctly reconstructed (target: 100%)

**Hierarchy**: Spearman correlation between 3-adic valuation and radius
- Negative = correct (v0 at outer edge, v9 at center)
- Target: -0.83 to -1.0
- Ceiling: -0.8321 with any within-level variance (mathematical limit due to v=0 having 66.7% of samples)

**Richness**: Average within-valuation-level variance of radii
- Measures geometric diversity beyond ordering
- Higher = more meaningful structure preserved
- Zero = collapsed to trivial shells (bad)

---

## v5.11 Checkpoint Reference

### CRITICAL WARNING: v5_11_overnight

**DO NOT USE v5_11_overnight as a reference for good training.**

Despite appearing to have good metrics (100% coverage, -0.83 hierarchy), this checkpoint is problematic:
- Training collapsed during the run
- Coverage came purely from frozen checkpoint initialization
- The hierarchy correlation is an artifact, not learned structure
- Richness appears high (0.0085) but is not from genuine training

---

### Full Checkpoint Reference

| Checkpoint | Cov | Hier_A | Hier_B | Richness | r_v0 | r_v9 | Status |
|------------|-----|--------|--------|----------|------|------|--------|
| **v5.11.3** (v5_11_structural) | 100% | -0.69 | -0.74 | 0.00124 | 0.55 | 0.42 | Stable for bioinformatics |
| **v5.11.8** (v5_11_homeostasis) | 99.9% | -0.51 | **-0.83** | 0.00126 | 0.56 | 0.48 | Good hierarchy |
| v5_11_progressive | 99.9% | -0.04 | +0.78 | 0.00145 | 0.50 | 0.58 | VAE-B INVERTED |
| v5_11_annealing | 100% | +0.58 | +0.80 | 0.00140 | 0.42 | 0.46 | VAE-B INVERTED |
| v5_11_overnight | 100% | -0.73 | -0.83 | 0.00800 | 0.88 | 0.22 | **INVALID-COLLAPSED** |
| v5_11_11_production | 100% | +0.28 | +0.06 | 0.00135 | 0.50 | 0.55 | Weak hierarchy |
| **homeostatic_rich** | 100% | -0.69 | **-0.8321** | **0.00787** | 0.89 | 0.19 | **BEST BALANCE** |
| max_hierarchy | 100% | -0.83 | -0.8321 | 0.00028 | 0.95 | 0.16 | Ceiling, low richness |
| hierarchy_focused | 100% | -0.83 | +0.68 | 0.00015 | 0.67 | 0.29 | VAE-B INVERTED |
| radial_target | 100% | -0.83 | +0.68 | 0.00005 | 0.95 | 0.17 | VAE-B INVERTED |

**Legend**: Cov=Coverage, Hier=Hierarchy (Spearman), r_v0/r_v9=mean radius at valuation 0/9

---

### Detailed Version Notes

**v5.11.3 (v5_11_structural)**
- Architecture: hidden_dim=128, n_projection_layers=2
- Stored metrics: radial_corr_A=-0.730, radial_corr_B=-0.832
- Actual evaluation: hier_A=-0.69, hier_B=-0.74
- Use case: Bioinformatics codon encoder training (stable embeddings)

**v5.11.8 (v5_11_homeostasis)**
- Architecture: hidden_dim=64, use_controller=True
- Uses HomeostasisController for dynamic freeze management
- Stored metrics: radial_corr_A=-0.816, radial_corr_B=-0.832, radius_v0=0.932, radius_v9=0.013
- Actual evaluation: hier_A=-0.51, hier_B=-0.83
- Best radial separation (v0→v9 spread)

**homeostatic_rich (RECOMMENDED)**
- Training script: `scripts/epsilon_vae/train_homeostatic_rich.py`
- Loss weights: hierarchy=5.0, coverage=1.0, richness=2.0, separation=3.0
- Achieved: -0.8321 hierarchy (ceiling) with 5.8x more richness than v5.11.8
- Proves hierarchy and richness are NOT mutually exclusive

**v5_11_progressive**
- WARNING: VAE-B shows inverted hierarchy (+0.78)
- VAE-A shows slight negative (-0.04)
- Not suitable for p-adic applications

---

## Training Parameters for Hierarchy

To prioritize hierarchy while preserving richness:

```python
# Loss weights (from homeostatic_rich)
hierarchy_weight = 5.0      # Push toward target radii per valuation
coverage_weight = 1.0       # Maintain reconstruction
richness_weight = 2.0       # Preserve within-level variance
separation_weight = 3.0     # Ensure level ordering

# Freeze strategy
freeze_encoder_a = True     # Preserve coverage
freeze_encoder_b = False    # Allow B to learn hierarchy
encoder_b_lr_scale = 0.1    # Slower adaptation preserves structure

# Variance control
variance_weight = 50-100    # Higher = more collapse (less richness)
min_richness_ratio = 0.5    # Keep at least 50% of original variance
```

---

## Mathematical Limits

**Hierarchy Ceiling: -0.8321**

The Spearman correlation cannot exceed -0.8321 when ANY within-level variance exists. This is due to:
- v=0 contains 66.7% of all samples (13,122 of 19,683)
- Even tiny variance in this large group creates ties
- Only RadialSnapProjection (hard snapping to exact radii) achieves -1.0
- Hard snapping eliminates all richness (trivial solution)

**Richness-Hierarchy Tradeoff**

NOT mutually exclusive. homeostatic_rich proved:
- Same ceiling hierarchy (-0.8321) as collapsed models
- 5.8x more richness than v5.11.8
- 28x more richness than max_hierarchy

---

## File Locations

**Training Scripts**
- `scripts/epsilon_vae/train_homeostatic_rich.py` - Best balance approach
- `scripts/epsilon_vae/train_hierarchy_focused.py` - Hierarchy-first approach
- `scripts/epsilon_vae/analyze_all_checkpoints.py` - Checkpoint comparison tool

**Model Definitions**
- `src/models/ternary_vae.py` - TernaryVAEV5_11_PartialFreeze
- `src/models/homeostasis.py` - HomeostasisController, compute_Q

**Checkpoints**
- `sandbox-training/checkpoints/` - All training runs
- Production: Use `homeostatic_rich/best.pt` or `v5_11_homeostasis/best.pt`

---

## Quick Evaluation

```python
from src.models import TernaryVAEV5_11_PartialFreeze
from src.core import TERNARY

model = TernaryVAEV5_11_PartialFreeze(
    latent_dim=16, hidden_dim=64, max_radius=0.99,
    curvature=1.0, use_controller=True, use_dual_projection=True
)
model.load_state_dict(torch.load('path/to/best.pt')['model_state_dict'])

# Get embeddings
out = model(ops, compute_control=False)
radii_A = out['z_A_hyp'].norm(dim=-1)  # VAE-A radii
radii_B = out['z_B_hyp'].norm(dim=-1)  # VAE-B radii

# Compute hierarchy
from scipy.stats import spearmanr
valuations = TERNARY.valuation(indices)
hierarchy = spearmanr(valuations, radii_B)[0]  # Use VAE-B for p-adic
```

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-27 | 1.0 | Initial CLAUDE.md with architecture docs, checkpoint reference, overnight warning |

