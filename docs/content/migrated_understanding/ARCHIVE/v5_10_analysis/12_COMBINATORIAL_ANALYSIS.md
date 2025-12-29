# Combinatorial Analysis: All Possible Implementations

**Date:** 2025-12-27
**Status:** Comprehensive Analysis

---

## Executive Summary

The codebase contains **200+ composable components** that can be combined in thousands of ways. This document analyzes:
1. All component categories and their options
2. Valid combination patterns
3. Estimated number of unique configurations
4. Prioritized test matrix for systematic exploration

---

## 1. Component Categories & Options

### 1.1 Model Architecture (VAE Core)

| Category | Options | Count |
|----------|---------|-------|
| **Base VAE** | SimpleVAE, SimpleVAEWithHyperbolic, TernaryVAEV5_11, OptimalVAE, EpsilonVAE, TropicalVAE, SwarmVAE | 7 |
| **Encoder Type** | SimpleEncoder, FrozenEncoder, PAdicEmbedding, HybridCodonEncoder, CodonEncoder | 5 |
| **Decoder Type** | SimpleDecoder, FrozenDecoder, HolographicDecoder, TropicalDecoder | 4 |
| **Latent Dim** | 8, 16, 32, 64 | 4 |
| **Hidden Dims** | [64,32], [128,64], [256,128,64], [64,32,16] | 4 |

**Combinations:** 7 × 5 × 4 × 4 × 4 = **2,240** architectures

### 1.2 Projection Layer

| Category | Options | Count |
|----------|---------|-------|
| **Projection Type** | None (Euclidean), HyperbolicProjection, DualHyperbolicProjection, LatticeProjection | 4 |
| **Curvature** | 0.5, 1.0, 2.0 | 3 |
| **Max Radius** | 0.85, 0.90, 0.95, 0.99 | 4 |
| **Learnable Curvature** | True, False | 2 |

**Combinations:** 4 × 3 × 4 × 2 = **96** projection configs

### 1.3 Loss Functions

| Category | Options | Count |
|----------|---------|-------|
| **Reconstruction** | ReconstructionLoss, HyperbolicReconLoss, HomeostaticReconLoss | 3 |
| **KL Divergence** | KLDivergenceLoss, HyperbolicPrior, HomeostaticHyperbolicPrior | 3 |
| **P-adic Loss** | None, PAdicRankingLoss, PAdicRankingLossHyperbolic, PAdicRankingLossV2, PAdicGeodesicLoss, SoftPadicRankingLoss | 6 |
| **Radial Loss** | None, RadialStratificationLoss, RadialHierarchyLoss, MonotonicRadialLoss, GlobalRankLoss | 5 |
| **Entropy Loss** | None, EntropyRegularization, ZeroSparsityLoss | 3 |
| **Repulsion Loss** | None, RepulsionLoss | 2 |

**Combinations:** 3 × 3 × 6 × 5 × 3 × 2 = **1,620** loss configs

### 1.4 Training Strategy

| Category | Options | Count |
|----------|---------|-------|
| **Beta Schedule** | Constant, Linear, Cyclical, Warmup | 4 |
| **Beta Value** | 0.001, 0.01, 0.1, 0.5, 1.0 | 5 |
| **Learning Rate** | 1e-4, 5e-4, 1e-3, 5e-3 | 4 |
| **Optimizer** | Adam, AdamW, MixedRiemannianOptimizer | 3 |
| **Curriculum** | None, ContinuousCurriculum, AdaptiveCurriculum | 3 |
| **Early Stopping** | None, Standard, CoveragePlateau, CorrelationDrop | 4 |

**Combinations:** 4 × 5 × 4 × 3 × 3 × 4 = **2,880** training configs

### 1.5 Advanced Components

| Category | Options | Count |
|----------|---------|-------|
| **Controller** | None, DifferentiableController, HomeostasisController | 3 |
| **Feedback** | None, ContinuousFeedback, CorrelationFeedback, ExplorationBoost | 4 |
| **Uncertainty** | None, MCDropout, DeepEnsemble, Evidential | 4 |
| **Multi-Task** | None, GradNorm, UncertaintyWeighting | 3 |

**Combinations:** 3 × 4 × 4 × 3 = **144** advanced configs

---

## 2. Total Configuration Space

### Raw Combinations
```
Architecture × Projection × Loss × Training × Advanced
= 2,240 × 96 × 1,620 × 2,880 × 144
= ~1.5 × 10^15 configurations
```

### Practical Reduction

Many combinations are invalid or redundant:
- Hyperbolic losses require hyperbolic projection
- Some loss combinations are mutually exclusive
- Certain training strategies don't apply to all architectures

**Estimated valid combinations:** ~10,000 - 100,000

---

## 3. Prioritized Test Categories

### Tier 1: Core Architecture (Must Test First)

| ID | Configuration | Expected Outcome |
|----|--------------|------------------|
| A1 | SimpleVAE + Euclidean | Baseline accuracy |
| A2 | SimpleVAEWithHyperbolic + HyperbolicProjection | Best structure (verified) |
| A3 | TernaryVAEV5_11 + DualProjection | Frozen encoder transfer |
| A4 | OptimalVAE (current best) | Reference benchmark |
| A5 | TropicalVAE | Max-plus algebra baseline |
| A6 | SwarmVAE | Swarm-based exploration |

### Tier 2: Loss Combinations (High Impact)

| ID | Configuration | Hypothesis |
|----|--------------|------------|
| L1 | Hyperbolic + No P-adic | Current best (verified) |
| L2 | Hyperbolic + SoftPadicRanking | Better than triplet p-adic |
| L3 | Hyperbolic + PAdicGeodesic | Unified geodesic |
| L4 | Hyperbolic + RadialHierarchy | Explicit radial bands |
| L5 | Hyperbolic + MonotonicRadial | Strict level ordering |
| L6 | Hyperbolic + GlobalRank | All-pair ranking |
| L7 | HomeostaticHyperbolicPrior | Adaptive prior |
| L8 | HomeostaticRecon + Hyperbolic | Adaptive reconstruction |
| L9 | FisherRao + Hyperbolic | Information geometry |
| L10 | ZeroValuation + Hyperbolic | Zero-centered structure |

### Tier 3: Training Strategies (Medium Impact)

| ID | Configuration | Hypothesis |
|----|--------------|------------|
| T1 | Beta=0.01, Constant | Current best |
| T2 | Beta warmup 0.001→0.1 | Gradual regularization |
| T3 | Cyclical beta | Exploration phases |
| T4 | ContinuousCurriculum | Progressive difficulty |
| T5 | AdaptiveCurriculum | Performance-based |
| T6 | CoveragePlateau stopping | Stop at coverage plateau |
| T7 | MixedRiemannianOptimizer | Riemannian geometry |
| T8 | ExplorationBoost | Boost when stuck |

### Tier 4: Advanced Components (Exploratory)

| ID | Configuration | Hypothesis |
|----|--------------|------------|
| X1 | DifferentiableController | Learned control |
| X2 | HomeostasisController | Equilibrium seeking |
| X3 | ContinuousFeedback + Hyperbolic | Adaptive loss weights |
| X4 | DeepEnsemble | Uncertainty estimation |
| X5 | Evidential | Epistemic uncertainty |
| X6 | GradNorm + MTL | Multi-task balance |
| X7 | Dual VAE (A+B) | Complementary objectives |

---

## 4. Combination Matrix for Systematic Testing

### Phase 1: Architecture × Projection (28 experiments)

```
           | Euclidean | Hyperbolic | DualHyp | Lattice |
-----------+-----------+------------+---------+---------|
SimpleVAE  |    A1     |    A2      |   A7    |   A11   |
HypVAE     |    N/A    |    A3      |   A8    |   A12   |
V5_11      |    A4     |    A5      |   A9    |   A13   |
Optimal    |    N/A    |    A6      |   A10   |   A14   |
Tropical   |    T1     |    T2      |   T3    |   T4    |
Swarm      |    S1     |    S2      |   S3    |   S4    |
Epsilon    |    E1     |    E2      |   E3    |   E4    |
```

### Phase 2: Best Architecture × Loss Variants (24 experiments)

Using winner from Phase 1:

```
           | NoPadic | SoftRank | Geodesic | RadialHier | Monotonic | GlobalRank |
-----------+---------+----------+----------+------------+-----------+------------|
+Recon     |   L1    |    L2    |    L3    |     L4     |    L5     |     L6     |
+HypRecon  |   L7    |    L8    |    L9    |     L10    |    L11    |     L12    |
+Homeo     |   L13   |    L14   |    L15   |     L16    |    L17    |     L18    |
+FisherRao |   L19   |    L20   |    L21   |     L22    |    L23    |     L24    |
```

### Phase 3: Best Loss × Training Strategy (20 experiments)

Using winner from Phase 2:

```
           | Beta=0.01 | Warmup | Cyclical | Curriculum | Adaptive |
-----------+-----------+--------+----------+------------+----------|
Adam       |    T1     |   T2   |    T3    |     T4     |    T5    |
AdamW      |    T6     |   T7   |    T8    |     T9     |    T10   |
RiemannOpt |    T11    |   T12  |    T13   |     T14    |    T15   |
+Feedback  |    T16    |   T17  |    T18   |     T19    |    T20   |
```

### Phase 4: Best Config × Advanced (12 experiments)

Using winner from Phase 3:

```
           | NoController | DiffControl | Homeo | Feedback |
-----------+--------------+-------------+-------+----------|
Standard   |      X1      |      X2     |  X3   |    X4    |
+Ensemble  |      X5      |      X6     |  X7   |    X8    |
+Evidential|      X9      |      X10    |  X11  |    X12   |
```

---

## 5. Top 20 Most Promising Combinations

Based on theory and prior experiments:

| Rank | Configuration | Rationale |
|------|--------------|-----------|
| 1 | SimpleVAEWithHyperbolic + NoPadic + beta=0.01 | **Verified best** |
| 2 | SimpleVAEWithHyperbolic + SoftPadicRanking | Soft ranking showed promise |
| 3 | SimpleVAEWithHyperbolic + HomeostaticPrior | Adaptive regularization |
| 4 | TropicalVAE + HyperbolicProjection | Max-plus + hyperbolic |
| 5 | SimpleVAEWithHyperbolic + PAdicGeodesic | Unified geodesic loss |
| 6 | OptimalVAE + MonotonicRadial | Strict level ordering |
| 7 | SimpleVAEWithHyperbolic + FisherRao | Information geometry |
| 8 | SimpleVAE + SoftPadicRanking (no hyp) | Test soft ranking alone |
| 9 | TernaryVAEV5_11 + HyperbolicProjection | Frozen encoder + hyp |
| 10 | SimpleVAEWithHyperbolic + GlobalRank | All-pair ranking |
| 11 | SimpleVAEWithHyperbolic + CyclicalBeta | Exploration phases |
| 12 | SimpleVAEWithHyperbolic + ContinuousCurriculum | Progressive difficulty |
| 13 | SwarmVAE + HyperbolicProjection | Swarm exploration |
| 14 | SimpleVAEWithHyperbolic + DifferentiableController | Learned control |
| 15 | SimpleVAEWithHyperbolic + ZeroValuation | Zero-centered |
| 16 | EpsilonVAE + HyperbolicProjection | Meta-learner |
| 17 | DualHyperbolicProjection + ComplementaryLoss | Dual exploration |
| 18 | SimpleVAEWithHyperbolic + ExplorationBoost | Adaptive exploration |
| 19 | HybridCodonEncoder + HyperbolicProjection | Better encoder |
| 20 | PAdicEmbedding + HyperbolicProjection | P-adic input encoding |

---

## 6. Implementation Categories

### Category A: Ready to Test (Existing Code)

All components exist, just need combination:
- SimpleVAE variants
- All loss functions
- Basic training strategies
- Hyperbolic projection

### Category B: Minor Integration Needed

Components exist but need wiring:
- TropicalVAE + Hyperbolic
- SwarmVAE + P-adic losses
- Hybrid encoders + SimpleVAE

### Category C: New Implementation Required

Need to create new combinations:
- PAdicEmbedding as VAE encoder
- FisherRao with hyperbolic space
- Dual VAE with shared losses

### Category D: Research Extensions

Theoretical combinations to explore:
- Tropical + P-adic + Hyperbolic triple
- Diffusion + Hyperbolic
- Contrastive pre-training + Hyperbolic fine-tuning

---

## 7. Estimated Test Time

| Phase | Experiments | Time/Exp | Total |
|-------|-------------|----------|-------|
| Phase 1 | 28 | 5 min | 2.3 hrs |
| Phase 2 | 24 | 5 min | 2 hrs |
| Phase 3 | 20 | 5 min | 1.7 hrs |
| Phase 4 | 12 | 5 min | 1 hr |
| **Total** | **84** | | **~7 hrs** |

With parallelization (4 workers): **~2 hours**

---

## 8. Success Metrics

For each experiment, measure:

| Metric | Target | Weight |
|--------|--------|--------|
| Reconstruction Accuracy | > 95% | 40% |
| Spearman Correlation | > +0.02 | 30% |
| Silhouette Score | > 0.05 | 15% |
| Training Stability | No NaN | 10% |
| Training Time | < 5 min | 5% |

**Composite Score:** weighted sum, higher is better

---

## 9. Quick Reference: Component Compatibility

### Compatible Pairs (Green Light)

- SimpleVAE ↔ Any loss
- HyperbolicProjection ↔ Any base VAE
- PAdicRankingLoss ↔ Any architecture
- Adam/AdamW ↔ Any architecture

### Incompatible Pairs (Red Light)

- HyperbolicPrior ↔ Euclidean projection (requires hyperbolic)
- PAdicGeodesic ↔ Non-hyperbolic (requires Poincare distance)
- FrozenEncoder ↔ Training (can't backprop)
- RiemannianOptimizer ↔ Non-manifold params

### Conditional Compatibility (Yellow Light)

- TropicalVAE ↔ Hyperbolic (needs adapter)
- SwarmVAE ↔ P-adic losses (needs index tracking)
- DualProjection ↔ Single loss (needs dual loss)

---

## 10. Next Steps

1. **Run Phase 1** (Architecture × Projection) - ~28 experiments
2. **Identify winner** from Phase 1
3. **Run Phase 2** (Loss variants) with Phase 1 winner
4. **Continue cascade** through Phases 3-4
5. **Document findings** in results file
6. **Update OptimalVAE** with final best config

### Commands to Run

```bash
# Phase 1: Architecture sweep
python scripts/experiments/combination_sweep.py --phase 1 --workers 4

# Phase 2: Loss sweep (after Phase 1)
python scripts/experiments/combination_sweep.py --phase 2 --base-config phase1_winner.json

# Full cascade (automated)
python scripts/experiments/combination_sweep.py --cascade --workers 4
```
