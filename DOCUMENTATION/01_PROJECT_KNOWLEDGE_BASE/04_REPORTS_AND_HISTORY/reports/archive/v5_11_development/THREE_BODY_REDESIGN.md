# Three-Body System: Architectural Redesign

**Doc-Type:** Architecture Vision · Version 1.1 · 2025-12-14

---

## Foundation: Pretrained Manifold

**Use existing v5.5 checkpoint as the foundation—coverage is already solved.**

```
Artifact: sandbox-training/checkpoints/v5_5/latest.pt
Epoch: 103
Coverage: 100% (19,683 / 19,683 operations - perfect reconstruction)
```

### Quality Analysis Results (2025-12-14)

| Metric | VAE-A | VAE-B | Interpretation |
|--------|-------|-------|----------------|
| **Coverage** | 100% | 100% | Perfect reconstruction |
| **3-adic Distance Correlation** | 0.616 | 0.617 | Good angular structure |
| **Radial Hierarchy Correlation** | +0.244 | +0.287 | **INVERTED** (should be negative) |
| **Radii Range** | 1.63 - 3.38 | 1.55 - 3.76 | Compressed, not using full Poincaré disk |

**Critical Finding: Radial Structure is INVERTED**

The manifold has learned excellent angular separation (3-adic neighbors close in latent space), but the radial hierarchy is backwards:

| 3-adic Valuation | Expected Radius | Actual Radius (A) | Status |
|-----------------|-----------------|-------------------|--------|
| v=0 (13,122 ops) | Large (boundary) | 2.65 | Correct position |
| v=9 (1 op: const_0) | Small (origin) | **3.09** | **WRONG - should be near origin** |

High-valuation points (more divisible by 3) should be near the ORIGIN in hyperbolic geometry, but they're currently at the BOUNDARY.

### What This Means for Three-Body

The v5.5 manifold provides:
- ✅ Perfect coverage (no re-learning needed)
- ✅ Good angular/distance structure (r≈0.62)
- ❌ Inverted radial hierarchy (needs correction)

**The goal shifts from learning coverage to correcting radial structure while preserving angular relationships.**

### Why This Matters

1. **Coverage is solved** - don't re-learn what's already learned
2. **Angular structure exists** - 3-adic distance correlation is good (0.62)
3. **Radial structure needs flipping** - high-valuation should be near origin
4. **Three-body goal** - correct radial hierarchy while preserving angular structure

---

## The Three Bodies

```
        ┌─────────────────────────────────────┐
        │           CONTROLLER                │
        │   Observes opposition, mediates     │
        └──────────────┬──────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
   ┌─────────────┐           ┌─────────────┐
   │   VAE-A     │◄────ρ────►│   VAE-B     │
   │  (Chaotic)  │           │  (Frozen)   │
   │  Explores   │           │  Anchors    │
   └─────────────┘           └─────────────┘
          │                         │
          └────────────┬────────────┘
                       │
                       ▼
              Complementary Opposition
```

---

## Purpose of Each Body

### VAE-A: The Explorer (Chaotic)
- High entropy output distribution
- Pushes into unexplored regions of latent space
- Learns diverse, spread-out representations
- Risk-taking: may overfit or collapse, but discovers structure

### VAE-B: The Anchor (Frozen)
- Low entropy, concentrated outputs
- Anchors representations near origin
- Provides stable reference point
- Conservative: maintains structure, resists drift

### Controller: The Mediator
- Observes the RELATIONSHIP between A and B
- Adjusts their interaction strength (ρ)
- Balances exploration vs exploitation
- Guides curriculum progression

---

## The Opposition Principle

VAE-A and VAE-B are not independent—they define each other through opposition:

| Aspect | VAE-A | VAE-B | Opposition Creates |
|--------|-------|-------|-------------------|
| Entropy | High | Low | Diversity gradient |
| Radius | Boundary | Origin | Radial structure |
| Risk | Exploratory | Conservative | Balanced search |
| Learning | Fast, unstable | Slow, stable | Robust convergence |

**Key Insight**: The opposition itself is informative. When A and B agree, we have confidence. When they disagree, we have uncertainty to resolve.

---

## What the Controller Should Observe

Not just individual metrics, but **relational metrics**:

### Individual (Current)
- H_A, H_B (entropies)
- kl_A, kl_B (divergences)
- r_A, r_B (correlations)

### Relational (The Opposition)
- **H_diff** = |H_A - H_B| → entropy gap (should be maintained)
- **r_diff** = |r_A - r_B| → correlation gap (who's learning better?)
- **z_similarity** = corr(z_A, z_B) → representation overlap
- **agreement** = how often A and B predict the same output
- **radius_ratio** = mean_radius_A / mean_radius_B → spatial relationship

### Temporal (Dynamics)
- Δr_A, Δr_B → who's improving?
- Δcoverage → are we exploring?
- loss_velocity → are we converging?

---

## What the Controller Should Output

All outputs must be **tensors in the computation graph**:

### 1. Cross-Injection Strength (ρ)
```python
rho = sigmoid(controller_output[0]) * rho_max
z_A_cross = (1 - rho) * z_A + rho * z_B.detach()
z_B_cross = (1 - rho) * z_B + rho * z_A.detach()
```
- High ρ: A and B share information, converge
- Low ρ: A and B diverge, explore independently

### 2. Opposition Maintenance Weight
```python
entropy_gap_loss = (target_gap - |H_A - H_B|)²
loss += controller_output[1] * entropy_gap_loss
```
- Ensures A stays chaotic, B stays frozen
- Prevents collapse to identical representations

### 3. Relative KL Weights (β_A, β_B)
```python
beta_A = softplus(controller_output[2])
beta_B = softplus(controller_output[3])
kl_loss = beta_A * kl_A + beta_B * kl_B
```
- Controls how tightly each VAE follows the prior
- Can push A to explore (low β) while B anchors (high β)

### 4. Ranking Loss Weight
```python
ranking_weight = softplus(controller_output[4])
loss += ranking_weight * ranking_loss
```
- Controls emphasis on 3-adic structure
- Curriculum: low early, high late

### 5. Temperature Ratio
```python
temp_ratio = sigmoid(controller_output[5]) * 2  # [0, 2]
temp_A = base_temp * temp_ratio
temp_B = base_temp / temp_ratio
```
- Inversely linked: if A is hot, B is cold
- Maintains opposition in exploration

---

## Differentiable Information Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                         FORWARD PASS                                │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Input x ──┬──► Encoder_A ──► μ_A, σ_A ──► z_A ──┐                │
│            │                                      │                 │
│            │                              ┌───────┴───────┐        │
│            │                              │   Controller  │        │
│            │                              │   (observes   │        │
│            │                              │   opposition) │        │
│            │                              └───────┬───────┘        │
│            │                                      │ρ, weights      │
│            │                              ┌───────┴───────┐        │
│            │                              ▼               ▼        │
│            └──► Encoder_B ──► μ_B, σ_B ──► z_B ──► Cross-Inject   │
│                                                         │          │
│                                               z_A', z_B'│          │
│                                                    │    │          │
│                                      ┌─────────────┘    │          │
│                                      ▼                  ▼          │
│                              Decoder_A(z_A')    Decoder_B(z_B')   │
│                                      │                  │          │
│                                      ▼                  ▼          │
│                                  logits_A          logits_B        │
└────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────┐
│                         LOSS COMPUTATION                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  recon_A = CE(logits_A, x)           recon_B = CE(logits_B, x)    │
│  kl_A = KL(μ_A, σ_A)                 kl_B = KL(μ_B, σ_B)          │
│  H_A = entropy(logits_A)             H_B = entropy(logits_B)       │
│                                                                     │
│  # Controller-weighted losses (ALL TENSORS - gradients flow!)     │
│  loss = (                                                          │
│      w_recon * (recon_A + recon_B) +                              │
│      β_A * kl_A + β_B * kl_B +                                    │
│      w_ranking * ranking_loss +                                    │
│      w_opposition * opposition_loss                                │
│  )                                                                 │
│                                                                     │
│  # Gradient flows: loss → weights → controller → controller_params │
└────────────────────────────────────────────────────────────────────┘
```

---

## The Opposition Loss

A new loss term that explicitly maintains the three-body dynamic:

```python
def opposition_loss(H_A, H_B, z_A, z_B, r_A, r_B):
    """Reward productive opposition, penalize collapse."""

    # 1. Entropy gap should be maintained
    entropy_gap = torch.abs(H_A - H_B)
    entropy_term = F.relu(0.5 - entropy_gap)  # Penalize if gap < 0.5

    # 2. Representations should be different but not orthogonal
    z_sim = F.cosine_similarity(z_A.mean(0), z_B.mean(0), dim=0)
    # Want similarity in [0.3, 0.7] - related but distinct
    similarity_term = F.relu(z_sim - 0.7) + F.relu(0.3 - z_sim)

    # 3. Both should learn structure (correlations)
    # But allow temporary divergence
    min_corr = torch.min(r_A, r_B)
    correlation_term = F.relu(0.5 - min_corr)  # Both should be > 0.5

    return entropy_term + similarity_term + correlation_term
```

---

## Controller Architecture

Simple, lightweight, focused on relational features:

```python
class ThreeBodyController(nn.Module):
    """Observes opposition, outputs differentiable control signals."""

    def __init__(self, hidden_dim=32):
        super().__init__()

        # Input: 12 features (individual + relational + temporal)
        # Individual: H_A, H_B, kl_A, kl_B, r_A, r_B (6)
        # Relational: H_diff, r_diff, z_sim (3)
        # Temporal: Δr_A, Δr_B, Δloss (3)
        input_dim = 12

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),  # Smooth activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6)  # 6 control signals
        )

        # Output scaling (learned)
        self.output_scales = nn.Parameter(torch.ones(6) * 0.1)

    def forward(self, features):
        raw = self.net(features)
        scaled = raw * self.output_scales

        return {
            'rho': torch.sigmoid(scaled[0]) * 0.5,        # [0, 0.5]
            'beta_A': F.softplus(scaled[1]) + 0.5,        # [0.5, ∞)
            'beta_B': F.softplus(scaled[2]) + 0.5,        # [0.5, ∞)
            'w_ranking': F.softplus(scaled[3]),           # [0, ∞)
            'w_opposition': F.softplus(scaled[4]) + 0.1,  # [0.1, ∞)
            'temp_ratio': torch.sigmoid(scaled[5]) + 0.5  # [0.5, 1.5]
        }
```

---

## Summary

| Component | Current | Redesigned |
|-----------|---------|------------|
| **StateNet** | 21D→8D, no gradients, 8.5K params | 12D→6D, differentiable, ~1.5K params |
| **Inputs** | Individual metrics only | Individual + Relational + Temporal |
| **Outputs** | Scalars via .item() | Tensors in computation graph |
| **Attention** | 3 heads, 2 unused | Removed (not needed) |
| **Training** | Never learns | Learns via backprop |
| **Opposition** | Implicit | Explicit loss term |

---

## Implementation Priority

1. **ThreeBodyController** - differentiable, lightweight
2. **Opposition loss** - explicit maintenance of A/B dynamic
3. **Relational features** - observe the relationship, not just individuals
4. **Remove StateNet** - delete dead code
5. **Verify gradients** - `torch.autograd.grad` test

---

## Key Insight: Position-Dependent Control

**Hyperparameters cannot be global scalars in non-Euclidean space.**

In hyperbolic geometry:
- Distance grows exponentially toward boundary
- Curvature effects vary with radius
- The "right" learning dynamics at r=0.1 differ from r=0.9

**Control signals must be functions of position on the manifold:**

```python
# OLD: Global scalar (wrong for hyperbolic space)
rho = 0.3  # Same everywhere

# NEW: Position-dependent (respects geometry)
def control_at_position(z):
    """Control signal as function of manifold position."""
    radius = torch.norm(z, dim=-1)
    # Near origin: less cross-injection (stable region)
    # Near boundary: more cross-injection (exploratory region)
    rho = self.rho_network(z)  # Learned function of position
    return rho
```

The Controller doesn't output 6 scalars—it outputs 6 **functions** that map positions to local control values.

---

## The Feedback Loop

All three bodies learn together through mutual influence:

```
    VAE-A output ──────►┐
                        │
                        ▼
                   Controller ────► Control signals (position-dependent)
                        ▲                    │
                        │                    │
    VAE-B output ──────►┘                    │
                                             │
         ┌───────────────────────────────────┘
         │
         ▼
    Affects VAE-A and VAE-B learning
         │
         └──► Changes their outputs ──► Controller observes ──► ...
```

**Exploitation of Exploration:** The system finds dynamic equilibrium where:
- High-quality regions are exploited (refined)
- Uncertain regions are explored (discovered)
- The controller learns WHERE to exploit vs explore based on the manifold geometry

---

## Implementation Path

1. **Load v5.5 manifold** as frozen encoder (coverage solved)
2. **Build position-aware controller** - takes z, outputs local control
3. **Make control differentiable** - gradients flow through controller
4. **Train on quality metrics** - correlation, structure, not just coverage
5. **Three-body feedback** - all components co-evolve

---

## Quality Metrics (Empirically Validated)

Based on v5.5 analysis, these metrics define manifold quality:

### Primary Metrics

| Metric | Formula | v5.5 Actual | Target | Priority |
|--------|---------|-------------|--------|----------|
| **Coverage** | % perfect reconstructions | 100% | 100% | ✅ Achieved |
| **Angular Correlation** | ρ(3-adic_dist, latent_dist) | +0.62 | >0.6 | ✅ Achieved |
| **Radial Hierarchy** | ρ(valuation, radius) | **+0.24** | **<-0.3** | ❌ Needs Fix |
| **Radius Spread** | max(r) - min(r) | 1.7 | >2.0 | ⚠️ Marginal |

### The Radial Inversion Problem

v5.5 learned **inverted** radial structure:

```
Expected (3-adic hyperbolic):          Actual (v5.5):

  Boundary (r≈1.0)                      Boundary (r≈3.5)
    └─ v=0 ops (13,122)                   └─ v≥3 ops (high valuation)

  Middle (r≈0.5)                        Middle (r≈2.7)
    └─ v=1,2 ops                          └─ v=0,1 ops (low valuation)

  Origin (r≈0.1)                        Origin (r≈1.6)
    └─ v≥3 ops (high valuation)           └─ const_0 only (special case)
```

### Quality Function (Position-Dependent)

```python
def quality_at_position(z, idx):
    """Quality score for a point on the manifold.

    Components:
    1. Reconstruction accuracy (local)
    2. Radial correctness (is radius appropriate for valuation?)
    3. Angular neighborhood (are 3-adic neighbors nearby?)
    """
    # 1. Reconstruction (already perfect, weight low)
    recon_quality = reconstruction_accuracy(z)

    # 2. Radial quality (THE MAIN ISSUE)
    actual_radius = torch.norm(z)
    expected_radius = target_radius_for_valuation(idx)
    radial_error = torch.abs(actual_radius - expected_radius)
    radial_quality = torch.exp(-radial_error)

    # 3. Angular quality (already good, preserve)
    angular_quality = neighbor_distance_correlation(z, idx)

    # Weighted combination (emphasize radial since it's broken)
    return 0.1 * recon_quality + 0.7 * radial_quality + 0.2 * angular_quality


def target_radius_for_valuation(idx: int) -> float:
    """Target radius based on 3-adic valuation.

    v_3(idx) high → small radius (near origin)
    v_3(idx) low  → large radius (near boundary)
    """
    v = compute_3adic_valuation(idx)
    max_v = 9

    # Linear mapping: v=0 → outer_radius, v=max → inner_radius
    inner_radius = 0.2  # High-valuation target
    outer_radius = 0.9  # Low-valuation target (Poincaré disk)

    return outer_radius - (v / max_v) * (outer_radius - inner_radius)
```

### Corrective Loss for Radial Hierarchy

```python
def radial_stratification_loss(z: Tensor, indices: Tensor) -> Tensor:
    """Push embeddings to correct radial positions.

    This is the MISSING loss term in v5.5 that caused inverted hierarchy.
    """
    # Compute actual radii
    actual_radii = torch.norm(z, dim=1)

    # Compute target radii based on 3-adic valuation
    valuations = compute_valuation_batch(indices)
    target_radii = outer_radius - (valuations / max_v) * (outer_radius - inner_radius)

    # SmoothL1 loss with valuation weighting (high-v points matter more)
    weights = 1.0 + valuations / max_v  # High-v weighted 2x
    loss = F.smooth_l1_loss(actual_radii, target_radii, reduction='none')

    return (loss * weights).mean()
```

---

## Experimental Validation (2025-12-14)

**Approach**: Freeze v5.5 encoder (coverage solved), train only hyperbolic projection using true Poincaré geometry.

### Training Configuration
```python
# Losses (key insight: emphasize radial over ranking)
total_loss = 0.3 * ranking_loss + 2.0 * radial_loss + 0.1 * kl_loss

# Architecture
- Frozen v5.5 encoder: 45,792 params (unchanged)
- Trainable hyperbolic projection: 2,833 params
  - direction_net: learns angular structure
  - radius_net: learns radial hierarchy (sigmoid → [0, 0.95])
```

### Results

| Metric | v5.5 (Euclidean) | Hyperbolic Training | Change |
|--------|------------------|---------------------|--------|
| **Radial Hierarchy** | +0.24 (inverted) | **-0.25** (correct) | **FLIPPED** |
| **Hyperbolic Correlation** | 0.62 | 0.57 | -0.05 (acceptable) |
| **Coverage** | 100% | 100% (frozen) | unchanged |

**Key Finding**: By emphasizing radial stratification loss (2.0 weight) over ranking loss (0.3 weight), the model learns to place high-valuation points near the origin. The radial hierarchy flipped from +0.24 to -0.25.

### Implications for Three-Body System

1. **Radial loss is essential** - v5.5 achieved coverage but wrong geometry because it lacked radial stratification
2. **Loss weighting matters** - ranking loss alone produces inverted hierarchy; radial must dominate initially
3. **Separation works** - freezing coverage (v5.5) and training structure separately is effective
4. **Trainable params**: Only 2,833 params needed for structural correction (vs 45,792 frozen)

---

## Open Questions

1. ~~What quality metrics define "good" regions?~~ **ANSWERED**: Radial hierarchy is key
2. ~~Fine-tune v5.5 or train from scratch?~~ **ANSWERED**: Freeze v5.5, train structure only
3. Should controller see raw z_A, z_B or just statistics?
4. How to initialize controller (warm start vs random)?
5. Should opposition target (entropy gap, similarity range) be learned or fixed?
6. How to integrate this back into the Three-Body system?
