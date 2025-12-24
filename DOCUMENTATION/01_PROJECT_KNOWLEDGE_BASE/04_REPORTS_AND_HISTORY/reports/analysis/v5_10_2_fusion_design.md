# v5.10.2 Fusion Design: Appetitive + Purposeful + Hyperbolic

**Doc-Type:** Architecture Design Document - Version 1.0 - Updated 2025-12-12 - Author Claude Code

---

## Executive Summary

This document proposes a fusion architecture combining three philosophical approaches:

1. **Appetitive (v5.x-appetitive)**: Bio-inspired emergent drives with metric-gated phases
2. **Purposeful (train_purposeful.py)**: Consequence awareness teaching WHY ranking matters
3. **Hyperbolic (v5.10)**: Pure Poincare ball geometry with homeostatic emergence

The fusion creates **Intentional Hyperbolic VAE** - a system that not only has appetites (what to optimize) and purpose (why it matters) but operates in geometrically consistent hyperbolic space.

---

## Part 1: Appetitive System Analysis

### 1.1 Core Philosophy

The Appetitive system treats training as satisfying biological drives along a "teleological ladder":

```
Coverage -> Metric -> Hierarchy -> Symbiosis -> Closure -> Agency
```

Key insight: **Emergent intelligence through drive satisfaction, not hand-crafted loss weighting.**

### 1.2 Hyperparameters Extracted

#### Appetite 1: Adaptive Ranking Loss
```yaml
appetite_ranking:
  base_margin: 0.1           # Base triplet margin (scales with valuation gap)
  n_triplets: 1000           # Triplets sampled per batch
  initial_weight: 0.5        # Weight in Phase 1A
```
**Code location**: `src/losses/appetitive_losses.py:20-118` (`AdaptiveRankingLoss`)

#### Appetite 2: Hierarchical Norm Loss
```yaml
appetite_hierarchy:
  n_groups: 4                # Partition latent into MSB/MID/LSB groups
  target_ratio: 2.0          # Target variance ratio between adjacent groups
  initial_weight: 0.1        # Weight in Phase 1A
```
**Code location**: `src/losses/appetitive_losses.py:121-165` (`HierarchicalNormLoss`)

#### Appetite 3: Curiosity Module (KDE Exploration)
```yaml
appetite_curiosity:
  bandwidth: 1.0             # KDE bandwidth for density estimation
  max_history: 5000          # Size of latent history buffer
  initial_weight: 0.0        # Inactive in Phase 1A (activates later)
```
**Code location**: `src/losses/appetitive_losses.py:168-235` (`CuriosityModule`)

#### Appetite 4: Symbiotic Bridge (VAE-A/VAE-B Coupling)
```yaml
appetite_symbiosis:
  hidden_dim: 32             # Hidden dim for cross-attention and MI estimator
  temperature: 0.1           # InfoNCE temperature
  initial_weight: 0.0        # Inactive in Phase 1A
  rho_base: 0.1              # Min coupling strength
  rho_max: 0.7               # Max coupling strength
  target_mi: 2.0             # Target mutual information (bits)
```
**Code location**: `src/losses/appetitive_losses.py:238-379` (`SymbioticBridge`)

#### Appetite 5: Algebraic Closure
```yaml
appetite_closure:
  num_pairs: 500             # Random pairs per batch for homomorphism test
  initial_weight: 0.0        # Inactive until late phases
  identity_idx: 9841         # Index of identity operation (all zeros)
```
**Code location**: `src/losses/appetitive_losses.py:382-560` (`AlgebraicClosureLoss`)

#### Violation Buffer (Hard Negative Mining)
```yaml
violation_buffer:
  capacity: 10000            # Max stored violations
  decay: 0.99                # Exponential decay factor
  max_age: 50                # Max epochs to retain violation
```
**Code location**: `src/losses/appetitive_losses.py:563-633` (`ViolationBuffer`)

### 1.3 Phase Gates (Metric-Triggered)

**Critical distinction**: Phases advance based on METRICS, not epochs.

```yaml
phase_gates:
  phase_1a_to_1b: 0.75       # r > 0.75 triggers transition
  phase_1b_to_2a: 0.85       # r > 0.85 triggers transition
  phase_2a_to_2b: 2.0        # MI > 2.0 bits triggers transition
  phase_2b_to_3: 0.5         # addition_accuracy > 50% triggers transition
```

### 1.4 Appetite Weight Evolution by Phase

| Appetite | Phase 1A | Phase 1B | Phase 2A | Phase 2B | Phase 3 |
|----------|----------|----------|----------|----------|---------|
| Ranking | 0.5 | 0.3 | 0.2 | 0.1 | 0.05 |
| Hierarchy | 0.1 | 0.2 | 0.1 | 0.1 | 0.05 |
| Curiosity | 0.0 | 0.1 | 0.2 | 0.1 | 0.05 |
| Symbiosis | 0.0 | 0.0 | 0.3 | 0.2 | 0.1 |
| Closure | 0.0 | 0.0 | 0.0 | 0.3 | 0.5 |

---

## Part 2: Purposeful System Analysis

### 2.1 Core Philosophy

The Purposeful system adds **consequence awareness** - teaching the model WHY improving ranking matters:

```
ranking_correlation -> predicted_addition_accuracy -> actual_addition_accuracy
```

Key insight: **Hunger without purpose is aimless optimization. Purpose = understanding that r -> closure capability.**

### 2.2 Hyperparameters Extracted

#### Consequence Predictor
```yaml
consequence_predictor:
  latent_dim: 16             # Matches model latent dim
  hidden_dim: 32             # Hidden layer dimension
  max_history: 100           # History buffer for self-model quality tracking

  # Input features (4D):
  # - ranking_correlation (0-1)
  # - z_mean_norm (latent statistics)
  # - z_std (latent variance)
  # - coverage_normalized (0-1)

  # Output: predicted_addition_accuracy (0-1) via Sigmoid
```
**Code location**: `src/losses/consequence_predictor.py:18-163` (`ConsequencePredictor`)

#### Purposeful Ranking Loss Wrapper
```yaml
purposeful_ranking:
  consequence_weight: 0.1    # Weight for consequence prediction loss
  # Wraps consequence_predictor with delayed feedback loop
```
**Code location**: `src/losses/consequence_predictor.py:241-311` (`PurposefulRankingLoss`)

### 2.3 Self-Model Quality Metric

The system tracks how well it predicts its own capabilities:

```python
# Pearson correlation between predicted and actual addition_accuracy
self_model_quality = correlation(predicted_history, actual_history)
```

**Phase transition signal**: When `self_model_quality > 0.5`, the model "understands" that ranking leads to closure.

### 2.4 Addition Accuracy Evaluation

```yaml
addition_evaluation:
  n_samples: 1000            # Test pairs for homomorphism check
  threshold: 0.1             # Error threshold for "accurate" prediction
  test_type: identity        # z_a + z_0 - z_0 == z_a (simple homomorphism)
```
**Code location**: `src/losses/consequence_predictor.py:166-238` (`evaluate_addition_accuracy`)

---

## Part 3: v5.10 Hyperbolic System Analysis

### 3.1 Core Philosophy

v5.10 operates in **pure hyperbolic space** without Euclidean contamination:

```
"Let the hyperbolic geometry do the work"
```

Key insight: **3-adic distance IS hyperbolic distance. The geometry matches the metric.**

### 3.2 Current Hyperbolic Components

#### Hyperbolic Prior (Wrapped Normal)
```yaml
hyperbolic_v10:
  use_hyperbolic_prior: true
  prior:
    homeostatic: true
    latent_dim: 16
    curvature: 2.0           # Higher = sharper tree structure
    prior_sigma: 1.0         # Spread in tangent space
    max_norm: 0.95           # Poincare ball boundary
    # Homeostatic bounds
    sigma_min: 0.3
    sigma_max: 2.0
    curvature_min: 0.5
    curvature_max: 4.0
    adaptation_rate: 0.01
```
**Code location**: `src/losses/hyperbolic_prior.py` (HyperbolicPrior)

#### Hyperbolic Reconstruction
```yaml
  use_hyperbolic_recon: true
  recon:
    homeostatic: true
    mode: weighted_ce        # Radius-weighted cross-entropy
    curvature: 2.0
    max_norm: 0.95
    geodesic_weight: 0.3     # For hybrid mode
    radius_weighting: true
    radius_power: 2.0        # Higher = more emphasis on origin
    weight: 0.5
    # Homeostatic bounds
    geodesic_weight_min: 0.1
    geodesic_weight_max: 0.8
    radius_power_min: 1.0
    radius_power_max: 4.0
    adaptation_rate: 0.01
```

#### Hyperbolic Centroid Clustering
```yaml
  use_centroid_loss: true
  centroid:
    max_level: 4             # Tree depth (3^4 = 81 clusters)
    curvature: 2.0
    max_norm: 0.95
    weight: 0.2
```

#### Hyperbolic Ranking
```yaml
ranking_hyperbolic:
  base_margin: 0.05
  margin_scale: 0.15
  n_triplets: 500
  hard_negative_ratio: 0.5
  curvature: 2.0
  radial_weight: 0.4         # Stronger hierarchy enforcement
  max_norm: 0.95
  weight: 0.5
```

### 3.3 StateNet v4 (Hyperbolic-Aware)

```yaml
# StateNet v4: 18D input, 7D output
statenet_input: 18
  # From v5.6 (12D): H_A, H_B, KL_A, KL_B, grad_A, grad_B, rho, lambda_A/B/3, coverage_A, coverage_B
  # From v5.7 (+3D): r_A, r_B, delta_ranking
  # v5.10 (+4D): mean_radius_A, mean_radius_B, prior_sigma, curvature

statenet_output: 7
  # Original (5D): lr_correction, lambda1/2/3_correction, rho_correction
  # v5.7 (+1D): ranking_weight_correction
  # v5.10 (+2D): sigma_correction, curvature_correction
```

---

## Part 4: Fusion Architecture Design

### 4.1 Core Fusion Principle

```
Appetitive (WHAT) + Purposeful (WHY) + Hyperbolic (WHERE) = Intentional Geometry
```

The fusion creates a system that:
1. **Knows what to optimize** (5 emergent appetites)
2. **Understands why it matters** (consequence prediction)
3. **Operates in correct geometry** (Poincare ball)

### 4.2 Proposed v5.10.2 Hyperparameters

#### Unified Appetite Configuration
```yaml
# v5.10.2 APPETITIVE-HYPERBOLIC FUSION
appetitive_hyperbolic:
  # Appetite 1: Hyperbolic Ranking (replaces Euclidean AdaptiveRankingLoss)
  appetite_ranking_hyp:
    base_margin: 0.05          # From v5.10 (was 0.1 in appetitive)
    margin_scale: 0.15         # Adaptive margin scaling
    n_triplets: 1000           # From appetitive (was 500 in v5.10)
    hard_negative_ratio: 0.5   # From v5.10
    curvature: 2.0             # Hyperbolic curvature
    radial_weight: 0.4         # Radial hierarchy enforcement
    max_norm: 0.95             # Poincare boundary
    initial_weight: 0.5        # Phase 1A weight

  # Appetite 2: Hyperbolic Hierarchy (Poincare radius variance)
  appetite_hierarchy_hyp:
    n_groups: 4                # Partition latent dims
    target_radius_ratio: 1.5   # MSB near origin, LSB near boundary
    curvature: 2.0
    max_norm: 0.95
    initial_weight: 0.1

  # Appetite 3: Hyperbolic Curiosity (geodesic density-based)
  appetite_curiosity_hyp:
    bandwidth: 1.0             # KDE bandwidth in tangent space
    max_history: 5000
    use_geodesic_distance: true  # KDE uses Poincare distance
    curvature: 2.0
    initial_weight: 0.0        # Activates in Phase 1B

  # Appetite 4: Symbiotic Bridge (unchanged - operates on codes)
  appetite_symbiosis:
    hidden_dim: 32
    temperature: 0.1
    rho_base: 0.1
    rho_max: 0.7
    target_mi: 2.0
    initial_weight: 0.0        # Activates in Phase 2A

  # Appetite 5: Hyperbolic Closure (geodesic homomorphism)
  appetite_closure_hyp:
    num_pairs: 500
    use_mobius_addition: true  # z_a ⊕ z_b - z_0 = z_{a∘b} in Poincare
    curvature: 2.0
    max_norm: 0.95
    initial_weight: 0.0        # Activates in Phase 2B
```

#### Purposeful Integration
```yaml
# CONSEQUENCE AWARENESS (from Purposeful)
consequence_module:
  enabled: true
  predictor:
    latent_dim: 16
    hidden_dim: 32
    max_history: 100

  # Extended input features (6D instead of 4D)
  input_features:
    - ranking_correlation      # From appetitive
    - z_mean_radius           # Hyperbolic radius (v5.10)
    - z_std                   # Latent variance
    - coverage_normalized     # Coverage metric
    - prior_sigma_current     # Homeostatic state (v5.10)
    - curvature_current       # Homeostatic state (v5.10)

  consequence_weight: 0.1
  self_model_quality_threshold: 0.5  # Triggers phase insight
```

#### Unified Phase Gates
```yaml
# METRIC-GATED PHASES (hybrid of appetitive + purposeful)
phase_gates:
  # Phase 1A -> 1B: Metric Foundation
  phase_1a_to_1b:
    ranking_threshold: 0.75    # r > 0.75
    coverage_threshold: 80.0   # coverage > 80%

  # Phase 1B -> 2A: Structural Consolidation
  phase_1b_to_2a:
    ranking_threshold: 0.85    # r > 0.85
    hierarchy_satisfied: true  # MSB > MID > LSB variance

  # Phase 2A -> 2B: Symbiotic Coupling
  phase_2a_to_2b:
    mi_threshold: 2.0          # MI > 2.0 bits
    self_model_quality: 0.3    # Model understands some consequences

  # Phase 2B -> 3: Algebraic Awakening
  phase_2b_to_3:
    addition_accuracy: 0.5     # > 50% homomorphism accuracy
    self_model_quality: 0.5    # Full consequence awareness

  # Phase 3 -> 4: Algebraic Satiation (NEW)
  phase_3_to_4:
    ranking_threshold: 0.99    # Target achieved
    addition_accuracy: 0.8     # Strong algebraic closure
    self_model_quality: 0.7    # Deep self-understanding
```

#### Homeostatic Integration
```yaml
# HOMEOSTATIC EMERGENCE (from v5.10)
homeostasis:
  enabled: true

  # Prior adaptation
  prior:
    sigma_min: 0.3
    sigma_max: 2.0
    curvature_min: 0.5
    curvature_max: 4.0
    adaptation_rate: 0.01

  # Appetite-aware adaptation (NEW)
  appetite_modulation:
    # Adapt sigma based on appetite satisfaction
    sigma_increases_when: curiosity_active  # Explore more when curious
    curvature_increases_when: hierarchy_improving  # Sharpen tree when hierarchy forms

  # Consequence-aware adaptation (NEW)
  consequence_modulation:
    # If self_model_quality low, slow down adaptation
    adaptation_rate_scale: self_model_quality  # [0, 1] multiplier
```

#### StateNet v5 (Intentional)
```yaml
# STATENET v5: Intentional Controller
statenet_v5:
  # Extended input (22D)
  input_dim: 22
  # From v5.10 (18D): H, KL, grad, rho, lambda, coverage, r_A, r_B, delta_ranking,
  #                   mean_radius_A/B, prior_sigma, curvature
  # NEW (+4D): self_model_quality, appetite_phase, mi_estimate, closure_accuracy

  # Extended output (9D)
  output_dim: 9
  # From v5.10 (7D): lr, lambda1/2/3, rho, ranking_weight, sigma, curvature
  # NEW (+2D): curiosity_weight_correction, closure_weight_correction

  scales:
    lr_scale: 0.1
    lambda_scale: 0.02
    ranking_scale: 0.3
    sigma_scale: 0.05
    curvature_scale: 0.02
    curiosity_scale: 0.1       # NEW
    closure_scale: 0.1         # NEW
```

### 4.3 Internal Code Components Required

#### From Appetitive System
| Component | File | Lines | Integration Notes |
|-----------|------|-------|-------------------|
| `AdaptiveRankingLoss` | appetitive_losses.py | 20-118 | Needs hyperbolic distance |
| `HierarchicalNormLoss` | appetitive_losses.py | 121-165 | Adapt to Poincare radius |
| `CuriosityModule` | appetitive_losses.py | 168-235 | Use geodesic KDE |
| `SymbioticBridge` | appetitive_losses.py | 238-379 | Keep as-is (latent space) |
| `AlgebraicClosureLoss` | appetitive_losses.py | 382-560 | Use Mobius addition |
| `ViolationBuffer` | appetitive_losses.py | 563-633 | Keep as-is |

#### From Purposeful System
| Component | File | Lines | Integration Notes |
|-----------|------|-------|-------------------|
| `ConsequencePredictor` | consequence_predictor.py | 18-163 | Extend input features |
| `evaluate_addition_accuracy` | consequence_predictor.py | 166-238 | Use hyperbolic distance |
| `PurposefulRankingLoss` | consequence_predictor.py | 241-311 | Merge with appetite ranking |
| `PurposefulTrainer` | train_purposeful.py | 106-197 | Template for new trainer |

#### From v5.10 System
| Component | File | Location | Integration Notes |
|-----------|------|----------|-------------------|
| `HyperbolicPrior` | hyperbolic_prior.py | Full | Core - unchanged |
| `HyperbolicReconLoss` | hyperbolic_prior.py | Full | Core - unchanged |
| `HyperbolicCentroidLoss` | hyperbolic_prior.py | Full | Core - unchanged |
| `PAdicRankingLossHyperbolic` | padic_losses.py | ~L500+ | Base for appetite_ranking_hyp |
| `PureHyperbolicTrainer` | train_ternary_v5_10.py | Full | Template for fusion trainer |

### 4.4 New Components to Create

#### 1. HyperbolicRankingAppetite
```python
class HyperbolicRankingAppetite(nn.Module):
    """Fuses AdaptiveRankingLoss + PAdicRankingLossHyperbolic + ViolationBuffer.

    Features:
    - Multi-scale margins from appetitive
    - Poincare distance from v5.10
    - Hard negative mining from both
    - Violation tracking from appetitive
    """
```

#### 2. HyperbolicHierarchyAppetite
```python
class HyperbolicHierarchyAppetite(nn.Module):
    """Hierarchy based on Poincare radius, not Euclidean variance.

    MSB dims cluster near origin (low radius = high valuation)
    LSB dims spread toward boundary (high radius = low valuation)
    """
```

#### 3. GeodesicCuriosityModule
```python
class GeodesicCuriosityModule(nn.Module):
    """KDE-based curiosity using geodesic (Poincare) distances.

    More accurate density estimation in hyperbolic space.
    """
```

#### 4. MobiusClosureLoss
```python
class MobiusClosureLoss(nn.Module):
    """Algebraic closure using Mobius addition in Poincare ball.

    z_a ⊕_c z_b ⊖_c z_0 = z_{a∘b}

    Where ⊕_c is Mobius addition with curvature c.
    """
```

#### 5. IntentionalTrainer
```python
class IntentionalTrainer:
    """Fuses PureHyperbolicTrainer + PurposefulTrainer + appetitive phases.

    Features:
    - Metric-gated phase transitions
    - Consequence prediction feedback loop
    - Homeostatic adaptation modulated by self-model quality
    - StateNet v5 with appetite/consequence awareness
    """
```

---

## Part 5: Training Flow Design

### 5.1 Phase 1A: Metric Foundation
```
Active appetites: ranking_hyp (0.5), hierarchy_hyp (0.1)
Consequence: Observing, not intervening
Homeostasis: Standard adaptation rate

Goal: Establish r > 0.75, basic tree structure
Exit: ranking_correlation > 0.75 AND coverage > 80%
```

### 5.2 Phase 1B: Structural Consolidation
```
Active appetites: ranking_hyp (0.3), hierarchy_hyp (0.2), curiosity_hyp (0.1)
Consequence: Starting to predict, learning correlation
Homeostasis: Sigma increases when curiosity active

Goal: Solidify hierarchy, explore rare operations
Exit: ranking_correlation > 0.85 AND hierarchy_satisfied
```

### 5.3 Phase 2A: Symbiotic Coupling
```
Active appetites: ranking_hyp (0.2), hierarchy_hyp (0.1), curiosity_hyp (0.2), symbiosis (0.3)
Consequence: Active prediction, self_model_quality building
Homeostasis: Curvature sharpens as MI increases

Goal: VAE-A/VAE-B mutual information > 2.0 bits
Exit: MI > 2.0 AND self_model_quality > 0.3
```

### 5.4 Phase 2B: Algebraic Awakening
```
Active appetites: ranking_hyp (0.1), all others at moderate weights, closure_hyp (0.3)
Consequence: Strong self-model, influencing adaptation
Homeostasis: Adaptation rate scaled by self_model_quality

Goal: Emergent algebraic structure
Exit: addition_accuracy > 0.5 AND self_model_quality > 0.5
```

### 5.5 Phase 3: Algebraic Satiation
```
Active appetites: closure_hyp dominant (0.5), others maintenance
Consequence: Full awareness, guiding final optimization
Homeostasis: Conservative, preserve learned structure

Goal: Target metrics achieved
Exit: r > 0.99 AND addition_accuracy > 0.8
```

### 5.6 Phase 4: Agency (Theoretical)
```
Active appetites: Balanced low weights
Consequence: Self-directed optimization
Homeostasis: Fully adaptive based on consequences

Goal: Autonomous improvement without explicit supervision
Note: This phase is aspirational - may not be achievable
```

---

## Part 6: Key Design Decisions

### 6.1 Why Hyperbolic Geometry for Appetites?

The appetitive system was designed with Euclidean assumptions:
- `AdaptiveRankingLoss` uses `torch.norm` (Euclidean L2)
- `HierarchicalNormLoss` uses variance (Euclidean)
- `CuriosityModule` uses `torch.cdist` (Euclidean)

v5.10's key insight: **3-adic metric is inherently hyperbolic**. By converting appetites to hyperbolic distances, we achieve:
1. **Geometric consistency** - All losses speak the same language
2. **Natural hierarchy** - Poincare radius encodes tree depth
3. **Stable training** - No Euclidean/hyperbolic conflicts

### 6.2 Why Consequence Awareness?

The appetitive system optimizes blindly - it doesn't know WHY ranking matters for closure. The purposeful system adds:
1. **Causal understanding** - Model learns r -> closure capability
2. **Adaptive optimization** - Can slow down when confused
3. **Self-model quality** - Measurable "understanding" metric

### 6.3 Why Metric-Gated Phases?

Epoch-based transitions (v5.6/v5.7) are brittle:
- Different seeds converge at different rates
- Hyperparameters interact unpredictably
- No guarantee phase is "ready" at epoch N

Metric-gated phases:
- Progress only when prerequisites achieved
- Natural curriculum learning
- Self-pacing based on actual capability

### 6.4 Homeostatic Modulation

v5.10's homeostasis is autonomous. v5.10.2 adds **intentional modulation**:
- When curiosity is active, increase sigma (explore more)
- When hierarchy is forming, increase curvature (sharpen tree)
- When self_model_quality is low, slow adaptation (don't optimize blindly)

---

## Part 7: Risk Analysis

### 7.1 Complexity Risk
**Problem**: v5.10.2 combines 3 systems, each with its own hyperparameters.
**Mitigation**: Use existing v5.10 as baseline, add appetites incrementally.

### 7.2 Phase Gate Deadlock
**Problem**: Model gets stuck if phase gates are too strict.
**Mitigation**: Add timeout epochs with forced transition + logging.

### 7.3 Consequence Predictor Divergence
**Problem**: Predictor learns wrong mapping, guides optimization poorly.
**Mitigation**: Monitor self_model_quality; if negative correlation, reset predictor.

### 7.4 Hyperbolic Numerical Stability
**Problem**: Operations near Poincare boundary (||z|| -> 1) are unstable.
**Mitigation**: Already handled in v5.10 with max_norm=0.95 clamping.

---

## Part 8: Implementation Roadmap

### Phase A: Foundation (Week 1)
1. Create `src/losses/hyperbolic_appetites.py` with hyperbolic versions of all 5 appetites
2. Test each appetite individually against v5.10 baseline
3. Verify numerical stability

### Phase B: Integration (Week 2)
1. Create `src/losses/intentional_loss.py` combining all components
2. Extend `ConsequencePredictor` with hyperbolic features
3. Create `StateNet v5` with extended I/O

### Phase C: Training (Week 3)
1. Create `scripts/train/train_intentional_v5_10_2.py`
2. Implement metric-gated phase transitions
3. Add comprehensive logging for all subsystems

### Phase D: Validation (Week 4)
1. Run full training to target metrics
2. Compare against v5.10 baseline
3. Ablation studies on each component

---

## Appendix A: Full Hyperparameter Reference

```yaml
# ============================================================================
# v5.10.2 INTENTIONAL HYPERBOLIC VAE - COMPLETE CONFIGURATION
# ============================================================================

model:
  input_dim: 9
  latent_dim: 16
  rho_min: 0.1
  rho_max: 0.7
  lambda3_base: 0.3
  lambda3_amplitude: 0.15
  eps_kl: 0.0005
  gradient_balance: true
  adaptive_scheduling: true
  use_statenet: true
  statenet_version: 5          # NEW: v5 intentional StateNet

# APPETITIVE-HYPERBOLIC MODULE
appetitive_hyperbolic:
  appetite_ranking_hyp:
    base_margin: 0.05
    margin_scale: 0.15
    n_triplets: 1000
    hard_negative_ratio: 0.5
    curvature: 2.0
    radial_weight: 0.4
    max_norm: 0.95

  appetite_hierarchy_hyp:
    n_groups: 4
    target_radius_ratio: 1.5
    curvature: 2.0
    max_norm: 0.95

  appetite_curiosity_hyp:
    bandwidth: 1.0
    max_history: 5000
    use_geodesic_distance: true
    curvature: 2.0

  appetite_symbiosis:
    hidden_dim: 32
    temperature: 0.1
    rho_base: 0.1
    rho_max: 0.7
    target_mi: 2.0

  appetite_closure_hyp:
    num_pairs: 500
    use_mobius_addition: true
    curvature: 2.0
    max_norm: 0.95

  violation_buffer:
    capacity: 10000
    decay: 0.99
    max_age: 50

# CONSEQUENCE AWARENESS
consequence_module:
  enabled: true
  predictor:
    latent_dim: 16
    hidden_dim: 32
    max_history: 100
    input_dim: 6               # Extended features
  consequence_weight: 0.1
  self_model_quality_threshold: 0.5

# PHASE GATES
phase_gates:
  phase_1a_to_1b:
    ranking_threshold: 0.75
    coverage_threshold: 80.0
  phase_1b_to_2a:
    ranking_threshold: 0.85
    hierarchy_satisfied: true
  phase_2a_to_2b:
    mi_threshold: 2.0
    self_model_quality: 0.3
  phase_2b_to_3:
    addition_accuracy: 0.5
    self_model_quality: 0.5
  phase_3_to_4:
    ranking_threshold: 0.99
    addition_accuracy: 0.8
    self_model_quality: 0.7
  timeout_epochs: 100          # Forced transition if stuck

# HOMEOSTASIS
homeostasis:
  enabled: true
  prior:
    sigma_min: 0.3
    sigma_max: 2.0
    curvature_min: 0.5
    curvature_max: 4.0
    adaptation_rate: 0.01
  appetite_modulation:
    sigma_curiosity_boost: 0.1
    curvature_hierarchy_boost: 0.05
  consequence_modulation:
    min_adaptation_rate: 0.001

# HYPERBOLIC CORE (from v5.10)
hyperbolic_v10:
  use_hyperbolic_prior: true
  use_hyperbolic_recon: true
  use_centroid_loss: true
  prior:
    homeostatic: true
    curvature: 2.0
    prior_sigma: 1.0
    max_norm: 0.95
  recon:
    homeostatic: true
    mode: weighted_ce
    curvature: 2.0
    radius_weighting: true
    radius_power: 2.0
  centroid:
    max_level: 4
    weight: 0.2

# STATENET v5
statenet_v5:
  input_dim: 22
  output_dim: 9
  lr_scale: 0.1
  lambda_scale: 0.02
  ranking_scale: 0.3
  sigma_scale: 0.05
  curvature_scale: 0.02
  curiosity_scale: 0.1
  closure_scale: 0.1

# TRAINING
seed: 42
batch_size: 256
num_workers: 0
total_epochs: 400              # Extended for phase transitions
patience: 200                  # Extended for metric-gated phases

# EVALUATION
eval_num_samples: 1000
eval_interval: 20
coverage_check_interval: 5

# SUCCESS METRICS
target_coverage_percent: 99.7
target_ranking_correlation: 0.99
target_addition_accuracy: 0.8
target_self_model_quality: 0.7
```

---

## Appendix B: Code File Map

```
src/
├── losses/
│   ├── appetitive_losses.py      # Original appetitive (Euclidean)
│   ├── consequence_predictor.py  # Purposeful system
│   ├── hyperbolic_prior.py       # v5.10 hyperbolic modules
│   ├── padic_losses.py           # Base p-adic losses
│   │
│   ├── hyperbolic_appetites.py   # NEW: Fused hyperbolic appetites
│   └── intentional_loss.py       # NEW: Combined loss module
│
├── models/
│   ├── ternary_vae_v5_10.py      # v5.10 model
│   ├── statenet_v4.py            # v5.10 StateNet
│   │
│   └── statenet_v5.py            # NEW: Intentional StateNet
│
scripts/train/
├── train_ternary_v5_10.py        # v5.10 trainer
├── train_purposeful.py           # Purposeful trainer
│
└── train_intentional_v5_10_2.py  # NEW: Fusion trainer

configs/
├── ternary_v5_10.yaml            # v5.10 config
├── appetitive_vae.yaml           # Appetitive config
│
└── intentional_v5_10_2.yaml      # NEW: Fusion config
```

---

**Document Status**: Design Complete - Ready for Review
**Next Steps**: Implementation requires explicit approval
**Dependencies**: v5.10 must be stable before fusion begins

---

*Generated: 2025-12-12*
*Author: Claude Code Analysis*
