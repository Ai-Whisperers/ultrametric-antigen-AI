# V5.11 Implementation Plan: Unified Hyperbolic Geometry

**Doc-Type:** Implementation Plan · Version 1.0 · 2025-12-14

---

## Executive Summary

V5.11 consolidates all architectural learnings into a unified system that:
1. Uses **v5.5 as frozen coverage base** (100% reconstruction solved)
2. Inherits **v5.10 hyperbolic infrastructure** (Poincaré geometry, HyperbolicPrior)
3. Fixes **StateNet gradient flow** (differentiable control signals)
4. Replaces competing losses with **unified PAdicGeodesicLoss**
5. Implements **Three-Body system** with position-dependent control

---

## Problem Analysis

### What v5.5 Achieved
- 100% coverage (19,683/19,683 perfect reconstructions)
- Good angular correlation (r=0.62 3-adic distance vs latent distance)
- **Problem**: Inverted radial hierarchy (+0.24 instead of negative)
- **Cause**: Euclidean reconstruction loss, no hyperbolic structure enforcement

### What v5.10 Attempted
- Hyperbolic priors, StateNet with curvature awareness
- Simultaneous coverage + correlation training
- **Problem**: Neither achieved well due to competing objectives
- **Cause**: Coverage and structure losses fight each other

### What Hyperbolic Structure Training Proved
- Freezing v5.5 and training only structure works
- Radial hierarchy flipped from +0.24 to -0.25 (correct direction)
- **Problem**: Separate ranking/radial losses still compete
- **Cause**: Ranking cares about relative distances, radial cares about absolute position

### The Core Insight
In proper hyperbolic geometry, **hierarchy IS correlation**. The Poincaré metric naturally couples them:
- Two points near origin have smaller geodesic distance than two at boundary
- Placing high-valuation points at origin automatically makes them "closer"
- The 3-adic ultrametric tree embeds naturally into hyperbolic space

---

## V5.11 Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         V5.11 ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input x ──► [FROZEN v5.5 Encoder] ──► z_euclidean (16D)           │
│                    (45,792 params, no gradients)                    │
│                                                                     │
│              ──► [HyperbolicProjection] ──► z_hyp (Poincaré ball)  │
│                    (~3K params, trainable)                          │
│                    - direction_net: angular structure               │
│                    - radius_net: radial hierarchy                   │
│                                                                     │
│              ──► [Three-Body System]                                │
│                    - VAE-A (chaotic): explores boundary             │
│                    - VAE-B (anchor): stabilizes origin              │
│                    - Controller: position-dependent signals         │
│                                                                     │
│              ──► [Unified PAdicGeodesicLoss]                        │
│                    - Single loss: d_poincare ↔ f(v_3)               │
│                    - Hierarchy + correlation unified                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Frozen v5.5 Encoder (Coverage Base)

```python
class FrozenV55Encoder(nn.Module):
    """Frozen coverage base - NEVER TRAINS."""

    def __init__(self, checkpoint_path: Path):
        # Load v5.5 encoder_A weights
        # Architecture: 9 → 256 → 128 → 64 → (μ:16, σ:16)

        # FREEZE all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Returns (mu, logvar) - deterministic, no gradients
        return mu, logvar
```

**Key**: v5.5's 100% coverage is preserved. We never risk losing it.

### Hyperbolic Projection Layer

```python
class HyperbolicProjection(nn.Module):
    """Trainable projection to Poincaré ball.

    Learns BOTH direction (angular) AND radius (hierarchy) independently.
    """

    def __init__(self, latent_dim=16, hidden_dim=64, curvature=1.0):
        # Direction network: learns angular structure
        self.direction_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Radius network: learns radial hierarchy
        self.radius_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1] → scale to [0, 0.95]
        )

    def forward(self, z_euclidean):
        direction = F.normalize(z_euclidean + self.direction_net(z_euclidean))
        radius = self.radius_net(z_euclidean) * 0.95
        return radius * direction  # z_hyp in Poincaré ball
```

**Key**: Separate networks for direction vs radius allows independent learning.

### Unified PAdicGeodesicLoss

```python
class PAdicGeodesicLoss(nn.Module):
    """THE KEY INNOVATION: Unify hierarchy and correlation in one loss.

    Instead of:
      - ranking_loss: compares triplet orderings (relative)
      - radial_loss: pushes to target radii (absolute)

    Use single geodesic loss:
      - For each pair (i,j): |d_poincare(z_i, z_j) - target(v_3(|i-j|))|²
      - Target maps valuation to hyperbolic distance
      - High valuation → small geodesic distance (automatically near origin)
    """

    def __init__(self, curvature=1.0, max_target_distance=3.0):
        self.curvature = curvature
        self.max_target = max_target_distance

    def target_distance(self, valuation):
        """Map 3-adic valuation to target hyperbolic distance.

        v_3 = 0 (not divisible by 3) → large distance (far apart)
        v_3 = 9 (divisible by 3^9)  → tiny distance (nearly same point)

        Formula: d_target = max_dist * exp(-valuation / scale)
        """
        return self.max_target * torch.exp(-valuation / 3.0)

    def forward(self, z_hyp, batch_indices):
        # Sample pairs
        i_idx, j_idx = sample_pairs(batch_indices)

        # Compute actual Poincaré distance
        d_actual = poincare_distance(z_hyp[i_idx], z_hyp[j_idx], self.curvature)

        # Compute target distance from 3-adic valuation
        diff = torch.abs(batch_indices[i_idx] - batch_indices[j_idx])
        valuation = compute_3adic_valuation(diff)
        d_target = self.target_distance(valuation)

        # Single unified loss
        return F.smooth_l1_loss(d_actual, d_target)
```

**Key**: This single loss enforces both:
- **Hierarchy**: High-valuation pairs must be close → both points near origin
- **Correlation**: Distance ordering matches valuation ordering → structure preserved

### Differentiable Three-Body Controller

```python
class ThreeBodyController(nn.Module):
    """Differentiable controller - FIXES StateNet gradient flow.

    v5.10 StateNet problem: outputs.item() breaks gradients
    v5.11 solution: outputs are TENSORS in computation graph
    """

    def __init__(self, input_dim=12, hidden_dim=32):
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6)
        )

    def forward(self, z_A, z_B, batch_stats):
        """
        Inputs (all tensors, gradients flow):
          - z_A, z_B: hyperbolic embeddings from both VAEs
          - batch_stats: [H_A, H_B, kl_A, kl_B, mean_radius_A, mean_radius_B, ...]

        Outputs (all tensors, gradients flow):
          - rho: cross-injection strength [0, 0.5]
          - beta_A, beta_B: KL weights
          - w_geodesic: geodesic loss weight
          - temp_ratio: temperature ratio for A vs B
        """
        raw = self.net(batch_stats)

        return {
            'rho': torch.sigmoid(raw[0]) * 0.5,
            'beta_A': F.softplus(raw[1]) + 0.5,
            'beta_B': F.softplus(raw[2]) + 0.5,
            'w_geodesic': F.softplus(raw[3]) + 0.1,
            'w_opposition': F.softplus(raw[4]) + 0.1,
            'temp_ratio': torch.sigmoid(raw[5]) + 0.5
        }
```

**Key**: All outputs are tensors. Gradients flow: loss → weights → controller → params.

### Position-Dependent Control

```python
def compute_position_aware_control(z_hyp, controller):
    """Control signals as functions of position on manifold.

    In hyperbolic space, the "right" learning dynamics vary with position:
    - Near origin: stable, exploit (low exploration)
    - Near boundary: unstable, explore (high exploration)
    """
    radius = torch.norm(z_hyp, dim=-1, keepdim=True)

    # Position features
    position_features = torch.cat([
        radius,                           # How far from origin
        torch.tanh(radius * 5),          # Saturated radius (boundary detector)
        1 - radius,                       # Distance to boundary
    ], dim=-1)

    # Controller sees position and outputs local control
    local_control = controller(position_features)

    # Near origin: less cross-injection (preserve structure)
    # Near boundary: more cross-injection (explore)
    return local_control
```

---

## Loss Function (Complete)

```python
def compute_v511_loss(outputs, batch_indices, controller_outputs):
    """V5.11 unified loss computation.

    Key difference from v5.10:
    - NO reconstruction loss (frozen v5.5 handles it)
    - SINGLE geodesic loss (not separate ranking + radial)
    - Controller weights are TENSORS (gradients flow)
    """
    z_A_hyp = outputs['z_A_hyp']
    z_B_hyp = outputs['z_B_hyp']

    # 1. Unified geodesic loss (THE MAIN LOSS)
    geodesic_A = padic_geodesic_loss(z_A_hyp, batch_indices)
    geodesic_B = padic_geodesic_loss(z_B_hyp, batch_indices)

    # 2. Hyperbolic KL (respects geometry)
    kl_A = hyperbolic_prior.kl_divergence(outputs['mu_A'], outputs['logvar_A'])
    kl_B = hyperbolic_prior.kl_divergence(outputs['mu_B'], outputs['logvar_B'])

    # 3. Opposition loss (maintain A/B complementarity)
    opposition = opposition_loss(z_A_hyp, z_B_hyp)

    # 4. Controller-weighted combination (ALL TENSORS)
    w = controller_outputs
    total_loss = (
        w['w_geodesic'] * (geodesic_A + geodesic_B) +
        w['beta_A'] * kl_A + w['beta_B'] * kl_B +
        w['w_opposition'] * opposition
    )

    # Gradients flow: total_loss → w → controller → controller_params
    return total_loss
```

---

## Training Strategy

### Phase 1: Pure Geometry (Epochs 0-50)
- Freeze v5.5 encoder
- Train only HyperbolicProjection
- Use unified PAdicGeodesicLoss
- Goal: Achieve negative radial hierarchy (-0.3 or better)

### Phase 2: Three-Body Integration (Epochs 50-100)
- Unfreeze Controller
- Add opposition loss
- VAE-A explores boundary, VAE-B anchors origin
- Goal: Stable dynamic equilibrium

### Phase 3: Refinement (Epochs 100+)
- Fine-tune all trainable components
- Position-dependent control active
- Goal: r > 0.8 correlation with correct hierarchy

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/models/ternary_vae_v5_11.py` | CREATE | Main V5.11 model class |
| `src/losses/padic_geodesic.py` | CREATE | Unified PAdicGeodesicLoss |
| `src/models/hyperbolic_projection.py` | CREATE | HyperbolicProjection layer |
| `src/models/three_body_controller.py` | CREATE | Differentiable controller |
| `scripts/train/train_v5_11.py` | CREATE | Training script |
| `configs/ternary_v5_11.yaml` | CREATE | Configuration |

---

## Success Metrics

| Metric | v5.5 | v5.10 | V5.11 Target |
|--------|------|-------|--------------|
| Coverage | 100% | ~80% | 100% (frozen) |
| Radial Hierarchy | +0.24 | ~0 | **< -0.3** |
| 3-adic Correlation | 0.62 | ~0.5 | **> 0.8** |
| Gradient Flow | N/A | Broken | **Full** |

---

## Key Innovations Summary

1. **Frozen Coverage**: v5.5 encoder provides 100% coverage, never regresses
2. **Unified Geodesic Loss**: Single loss couples hierarchy + correlation via geometry
3. **Differentiable Control**: Controller outputs are tensors, gradients flow
4. **Position-Dependent**: Control signals vary with manifold position
5. **Three-Body Dynamics**: VAE-A/B opposition creates balanced exploration/exploitation

---

## References

- `local-reports/THREE_BODY_REDESIGN.md` - Architecture vision
- `local-reports/ARCHITECTURE_REVIEW.md` - StateNet gradient flow analysis
- `scripts/train/train_hyperbolic_structure.py` - Proof-of-concept training
- `src/losses/padic_losses.py` - Existing p-adic loss implementations
- `src/metrics/hyperbolic.py` - Poincaré distance functions
- `sandbox-training/checkpoints/v5_5/latest.pt` - Frozen coverage checkpoint
