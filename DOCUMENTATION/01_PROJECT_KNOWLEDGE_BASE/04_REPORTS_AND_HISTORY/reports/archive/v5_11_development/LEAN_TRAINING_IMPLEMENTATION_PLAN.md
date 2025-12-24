# Lean Training Implementation Plan

**Doc-Type:** Implementation Plan · Version 1.1 · Updated 2025-12-13 · Author Claude + User

---

## Executive Summary

This plan implements a **lean training pipeline** centered on the **tripartite architecture**: StateNet + VAE-A + VAE-B. These three components form a co-evolving system where:

- **VAE-A** (Chaotic Explorer): High temperature, stochastic sampling, discovers novel regions
- **VAE-B** (Consolidator): Residual connections, stabilizes discoveries, maintains coverage
- **StateNet** (Meta-Controller): Observes both VAEs, outputs corrections, coordinates exploration

**Critical insight:** StateNet is NOT a hyperparameter tuner—it's the nervous system that enables VAE-A and VAE-B to communicate indirectly and co-evolve. Any optimization that breaks this feedback loop collapses the emergent behavior that achieves 97%+ coverage.

**Goals:**
1. Retire TensorBoard DURING training (post-mortem only)
2. Ensure model learns full manifold coverage AND discovers fused/emergent operations
3. Preserve and enhance the StateNet ↔ VAE bidirectional communication
4. Provide comprehensive TODO index

**Philosophy:** Trust the mathematical boundaries (Poincaré ball, 3-adic radial stratification). Let VAEs explore naturally within those boundaries. StateNet learns the optimal exploration strategy. Detect only: hung, collapsed, or diverged.

---

## Part 0: The Tripartite Architecture (CRITICAL)

### 0.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRIPARTITE TRAINING SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐    │
│    │    VAE-A     │◄────────►│   StateNet   │◄────────►│    VAE-B     │    │
│    │  (Explorer)  │          │   (Brain)    │          │(Consolidator)│    │
│    └──────┬───────┘          └──────┬───────┘          └──────┬───────┘    │
│           │                         │                         │            │
│           │    Cross-Injection      │                         │            │
│           │◄───────(ρ)─────────────►│◄───────────────────────►│            │
│           │                         │                         │            │
│    ┌──────▼───────┐          ┌──────▼───────┐          ┌──────▼───────┐    │
│    │   z_A        │          │  Corrections │          │   z_B        │    │
│    │ (16D latent) │          │   (8D out)   │          │ (16D latent) │    │
│    └──────────────┘          └──────────────┘          └──────────────┘    │
│                                                                             │
│    MATHEMATICAL BOUNDARIES (Non-negotiable):                                │
│    ├── Poincaré Ball: ||z|| < 1 (hyperbolic geometry)                      │
│    ├── 3-Adic Radial: high valuation → near origin                         │
│    └── KL Divergence: prevents collapse to single point                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 0.2 StateNet: The Central Nervous System

**StateNet is NOT optional.** It's the mechanism by which VAE-A and VAE-B coordinate without direct gradient flow between them.

**Input (20D observation of system state):**
```python
state_vector = [
    H_A,                    # VAE-A entropy (exploration indicator)
    H_B,                    # VAE-B entropy (consolidation indicator)
    KL_A,                   # VAE-A KL divergence (regularization pressure)
    KL_B,                   # VAE-B KL divergence
    grad_ratio,             # grad_norm_B / grad_norm_A (balance indicator)
    rho,                    # Cross-injection strength (coupling)
    lambda1, lambda2, lambda3,  # Current loss weights
    coverage_A, coverage_B, # What each VAE has discovered
    missing_ops,            # Gap to full coverage
    r_A, r_B,               # Ranking correlations (structure learning)
    mean_radius_A, mean_radius_B,  # Hyperbolic position
    prior_sigma,            # Current prior spread
    curvature,              # Current hyperbolic curvature
    radial_loss,            # How well radial hierarchy is learned
    curriculum_tau          # Current curriculum position
]
```

**Output (8D corrections applied to system):**
```python
corrections = [
    delta_lr,               # Learning rate adjustment
    delta_lambda1,          # VAE-A loss weight
    delta_lambda2,          # VAE-B loss weight
    delta_lambda3,          # Coupling loss weight
    delta_ranking_weight,   # Structure enforcement
    delta_sigma,            # Prior spread (REQUIRES HyperbolicPrior enabled)
    delta_curvature,        # Hyperbolic geometry (REQUIRES HyperbolicPrior enabled)
    delta_curriculum        # Curriculum advancement
]
```

### 0.3 The Feedback Loop (DO NOT BREAK)

```
   VAE-A explores
        │
        ▼
   Generates z_A, computes H_A, KL_A, coverage_A
        │
        ▼
   StateNet OBSERVES [H_A, H_B, coverage_A, coverage_B, grad_ratio, ...]
        │
        ▼
   StateNet OUTPUTS [delta_lr, delta_lambdas, delta_curriculum, ...]
        │
        ▼
   Corrections APPLIED to next training step
        │
        ▼
   VAE-B receives cross-injected z_A (scaled by ρ)
        │
        ▼
   VAE-B consolidates, computes H_B, KL_B, coverage_B
        │
        ▼
   LOOP CONTINUES
```

**CRITICAL:** If we remove metrics that StateNet observes (coverage, entropy, correlations), we blind the meta-controller. The VAEs would still train, but they'd lose the emergent coordination that enables 97%+ coverage.

### 0.4 What the Audit Found About StateNet

From `TRAINING_OPTIMIZATION_AUDIT.md`:

| Issue | Impact | Status |
|-------|--------|--------|
| Runs once per epoch (batch_idx == 0) | Coarse control, slow adaptation | **Underutilized** |
| delta_sigma never applied | 12.5% output capacity wasted | **HyperbolicPrior disabled** |
| delta_curvature never applied | 12.5% output capacity wasted | **HyperbolicPrior disabled** |
| Corrections fight fixed schedules | Conflicting signals | **Architecture issue** |
| No verification of correction effect | Unknown if learning useful | **Missing feedback** |

**Total StateNet capacity utilization: 75%** (6 of 8 outputs used)

### 0.5 VAE-A: The Explorer

**Role:** Discover novel regions of the ternary manifold through stochastic exploration.

**Characteristics:**
- **Parameters:** 50,203 (smaller, more agile)
- **Temperature:** High (starts at 2.0, cyclic modulation)
- **Entropy:** Maximized for exploration
- **Cross-injection:** Receives z_B, but with stop-gradient (doesn't follow B blindly)

**What StateNet controls for VAE-A:**
- `lambda1`: Loss weight (exploration vs reconstruction)
- `delta_lr`: Learning rate (faster/slower adaptation)
- Temperature via curriculum_tau indirectly

### 0.6 VAE-B: The Consolidator

**Role:** Stabilize discoveries, maintain coverage, provide reliable reconstructions.

**Characteristics:**
- **Parameters:** 117,499 (larger, more capacity)
- **Temperature:** Lower (starts at 1.5, monotonic decay)
- **Residual connections:** Preserve learned structure
- **Cross-injection:** Receives z_A, integrates explorer's discoveries

**What StateNet controls for VAE-B:**
- `lambda2`: Loss weight
- `lambda3`: Coupling strength with VAE-A
- `rho`: Cross-injection permeability (via phase scheduling)

### 0.7 The Cross-Injection Mechanism

```python
# In model forward pass:
z_A_to_B = z_A.detach() * rho  # Stop-gradient, scaled by permeability
z_B_combined = z_B + z_A_to_B  # VAE-B sees VAE-A's discoveries

z_B_to_A = z_B.detach() * rho
z_A_combined = z_A + z_B_to_A  # VAE-A sees VAE-B's consolidations
```

**ρ (rho) is phase-scheduled:**
- Phase 1 (epochs 0-40): ρ = 0.1 (isolation, independent learning)
- Phase 2 (epochs 40-120): ρ → 0.3 (gradual coupling)
- Phase 3 (epochs 120-250): ρ → 0.7 (strong coupling, resonance)
- Phase 4 (epochs 250+): ρ = 0.7 + temperature boost (ultra-exploration)

**StateNet influence:** `delta_lambda3` affects how strongly the coupling loss is enforced, indirectly modulating effective ρ.

### 0.8 Why This Architecture Works

1. **Division of labor:** VAE-A explores risky regions, VAE-B doesn't have to
2. **Information sharing:** Cross-injection lets discoveries propagate
3. **Adaptive control:** StateNet learns WHEN to explore vs consolidate
4. **No direct gradient conflict:** Stop-gradient prevents VAEs from fighting
5. **Emergent curriculum:** StateNet learns the optimal training trajectory

**The 97%+ coverage emerges from the interaction, not from any single component.**

### 0.9 Preservation Requirements for Lean Training

Any optimization MUST preserve:

| Component | What to Preserve | Why |
|-----------|------------------|-----|
| StateNet input vector | All 20 dimensions | Blinding StateNet breaks coordination |
| StateNet outputs | All 8 corrections | Removing outputs limits adaptation |
| Cross-injection | ρ mechanism | VAE communication channel |
| Coverage tracking | For both VAEs | StateNet needs to see progress |
| Entropy tracking | H_A, H_B | Exploration/consolidation balance |
| Gradient ratio | grad_norm_B / grad_norm_A | Balance indicator |

**What CAN be removed:**
- TensorBoard logging (doesn't affect training)
- Histogram computation (doesn't affect training)
- Redundant metric recomputation (cache instead)
- Disabled module execution (short-circuit)

---

---

## Part 1: TensorBoard Retirement (Training-Time)

### 1.1 Current State

**TensorBoard calls per epoch:**
- `log_batch()`: 385 scalar writes (5 metrics × 77 batches)
- `log_hyperbolic_batch()`: 5 scalars + flush()
- `log_hyperbolic_epoch()`: 15 scalars + flush()
- `log_tensorboard()`: 40+ scalars + flush()
- `log_histograms()`: All params × 2 + flush()
- `log_manifold_embedding()`: 4 embedding dumps

**Total: 400+ disk writes + 5 forced syncs per epoch**

### 1.2 New Architecture: Checkpoint-Based Post-Mortem

```
DURING TRAINING:
┌─────────────────────────────────────────────────────────────┐
│  Minimal In-Memory State                                     │
│  - loss (float) → early stopping                            │
│  - coverage_ema (float) → plateau detection                  │
│  - grad_ratio (float) → health check                        │
│  - tau (float) → phase awareness                            │
│                                                             │
│  Console Output: Single line per epoch                      │
│  "Epoch 42/400 | loss=0.234 | cov=87.3% | tau=0.45 | OK"   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼ (every checkpoint_freq epochs)
┌─────────────────────────────────────────────────────────────┐
│  Rich Checkpoint (for post-mortem analysis)                  │
│  - model_state_dict                                         │
│  - optimizer_state_dict                                     │
│  - epoch, loss, coverage, correlation                       │
│  - StateNet state (all 8 outputs history)                   │
│  - Curriculum tau history                                    │
│  - Sample latent codes (1000 samples, both VAEs)            │
│  - Config snapshot                                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼ (post-training only)
┌─────────────────────────────────────────────────────────────┐
│  TensorBoard Post-Mortem Analysis                           │
│  python scripts/analysis/generate_tensorboard_from_checkpoints.py │
│  - Reconstructs full scalar history from checkpoints        │
│  - Generates histograms from saved model states             │
│  - Creates embedding visualizations                         │
│  - No training overhead                                     │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Implementation Changes

| File | Change | Impact |
|------|--------|--------|
| `src/training/monitor.py` | Remove all TensorBoard calls, keep in-memory EMA | -400 writes/epoch |
| `src/training/trainer.py` | Remove `writer` usage, add console-only logging | Simpler |
| `src/training/hyperbolic_trainer.py` | Remove `_log_*_tensorboard()` methods | -200 lines |
| `configs/ternary_v5_10.yaml` | Remove `tensorboard_dir`, `log_interval`, `histogram_interval` | Cleaner config |
| NEW: `scripts/analysis/postmortem_tensorboard.py` | Generate TB logs from checkpoints | Post-hoc analysis |

### 1.4 Minimal Health Signals (During Training)

```python
class LeanTrainingMonitor:
    """Minimal training monitor - NO TensorBoard during training."""

    def __init__(self):
        self.loss_ema = 0.0
        self.coverage_ema = 0.0
        self.loss_history = []  # Last 50 values for plateau detection
        self.alpha = 0.1

    def update(self, loss: float, coverage: float) -> str:
        """Update EMAs and return status string."""
        self.loss_ema = self.alpha * loss + (1 - self.alpha) * self.loss_ema
        self.coverage_ema = self.alpha * coverage + (1 - self.alpha) * self.coverage_ema
        self.loss_history.append(loss)
        if len(self.loss_history) > 50:
            self.loss_history.pop(0)
        return self._get_status()

    def _get_status(self) -> str:
        """Return health status: OK, PLATEAU, DIVERGING, COLLAPSED."""
        if len(self.loss_history) < 10:
            return "WARMUP"

        recent = self.loss_history[-10:]
        variance = np.var(recent)
        trend = recent[-1] - recent[0]

        if variance < 1e-6 and self.coverage_ema < 50:
            return "COLLAPSED"
        elif trend > 0.5:
            return "DIVERGING"
        elif variance < 1e-4 and self.coverage_ema > 80:
            return "PLATEAU"
        return "OK"

    def should_stop(self, patience: int) -> bool:
        """Check if training should stop."""
        status = self._get_status()
        return status in ["COLLAPSED", "DIVERGING"]
```

---

## Part 2: Model Learning Objectives

### 2.1 Current Learning Gaps

| Objective | Current Status | Gap |
|-----------|---------------|-----|
| Coverage (19,683 ops) | 97%+ | OK |
| 3-adic ranking correlation | ~0.70 | Plateau |
| Named operations (AND, OR, XOR) | Not tracked | Unknown if learned |
| Fused operations | Not defined | No metric |
| Algebraic closure (a⊕b) | Not attempted | No loss term |
| Emergent clusters | Not visualized | No observability |

### 2.2 Named Operations Definition

The 19,683 operations include canonical logic operations:

| Name | LUT (9 values) | Index | Description |
|------|----------------|-------|-------------|
| ZERO | [-1,-1,-1,-1,-1,-1,-1,-1,-1] | 0 | Always -1 |
| ONE | [1,1,1,1,1,1,1,1,1] | 19682 | Always +1 |
| IDENTITY_A | [-1,-1,-1,0,0,0,1,1,1] | 9841 | Returns first input |
| IDENTITY_B | [-1,0,1,-1,0,1,-1,0,1] | 7381 | Returns second input |
| TERNARY_MIN | [-1,-1,-1,-1,0,0,-1,0,1] | 3280 | min(a, b) |
| TERNARY_MAX | [-1,0,1,0,0,1,1,1,1] | 16402 | max(a, b) |
| TERNARY_ADD | [1,-1,0,-1,0,1,0,1,-1] | 6643 | (a + b) mod 3 |
| TERNARY_MUL | [1,0,-1,0,0,0,-1,0,1] | 6562 | (a * b) in Z₃ |
| NEGATE_A | [1,1,1,0,0,0,-1,-1,-1] | 9842 | -a |
| XOR_LIKE | [0,1,-1,1,-1,0,-1,0,1] | ... | Balanced ternary XOR |

**Action:** Create `src/data/named_operations.py` with canonical operation definitions and indices.

### 2.3 Fused Operations Definition

**Fused operations** are compositions that the model may discover as emergent clusters:

```python
# Example: a ⊕ b = (a + b) mod 3, then negate
# This is a "fused" operation that combines addition and negation

def compose_operations(op1_idx: int, op2_idx: int) -> int:
    """Compute the composition op2(op1(a, b), _).

    Note: This is NOT straightforward because op2 takes 2 inputs.
    True composition requires fixing one input or using unary operations.
    """
    # For binary operations, composition is: op2(op1(a,b), c)
    # This creates a 3-input function, not in our 2-input space
    pass
```

**Key insight:** The 3^9 space is closed under certain operations but NOT under general composition. The model may discover:
1. **Self-inverse operations:** op(op(a,b), b) = a
2. **Commutative operations:** op(a,b) = op(b,a)
3. **Associative-like patterns:** Clusters of operations that behave similarly

### 2.4 Learning Metrics to Add

```python
class ManifoldLearningMetrics:
    """Track what the model learns beyond coverage."""

    def __init__(self, named_ops: Dict[str, int]):
        self.named_ops = named_ops
        self.named_op_indices = set(named_ops.values())

    def compute_named_op_accuracy(self, model, device) -> Dict[str, float]:
        """Check if named operations are correctly reconstructed."""
        results = {}
        for name, idx in self.named_ops.items():
            op = generate_ternary_operation_by_index(idx)
            op_tensor = torch.tensor(op, dtype=torch.float32, device=device).unsqueeze(0)
            recon = model.reconstruct(op_tensor)
            accuracy = (recon.round() == op_tensor).all().item()
            results[name] = accuracy
        return results

    def compute_cluster_discovery(self, z: torch.Tensor, indices: torch.Tensor) -> Dict:
        """Analyze emergent clustering in latent space."""
        # 1. Compute pairwise distances
        # 2. Run hierarchical clustering
        # 3. Compare to 3-adic tree structure
        # 4. Identify novel clusters (not matching 3-adic prefixes)
        pass

    def compute_algebraic_closure_score(self, model, device) -> float:
        """Check if z(a⊕b) ≈ z(a) + z(b) for ternary addition."""
        # Sample pairs (a, b)
        # Compute z(a), z(b), z(a⊕b)
        # Check if z(a⊕b) ≈ f(z(a), z(b)) for some learned f
        pass
```

---

## Part 3: Ternary Definition Consistency Analysis

### 3.1 README vs Code Comparison

| Aspect | README Definition | Code Implementation | Match |
|--------|-------------------|---------------------|-------|
| Space size | 3^9 = 19,683 | `3**9` in generation.py | ✅ |
| Value domain | Z₃ = {-1, 0, +1} | `num % 3 - 1` | ✅ |
| Index formula | `Σ (digit+1) × 3^k` | `(batch_data + 1).long() * weights` | ✅ |
| Input ordering | Row-major: (-1,-1)→(-1,0)→...→(1,1) | Implicit in digit position | ✅ |
| LUT interpretation | f(a,b) = LUT[(a+1)*3 + (b+1)] | Not explicit, assumed | ⚠️ |

### 3.2 Potential Ambiguity: LUT Interpretation

**The codebase DOES NOT explicitly verify** that LUT[k] corresponds to input pair (a, b) where k = (a+1)*3 + (b+1).

**Current assumption:** The digit at position `k` in the operation vector represents the output for input index `k`. This is CONSISTENT with the README but NEVER explicitly verified in code.

**Recommendation:** Add validation function:

```python
def validate_lut_interpretation():
    """Verify that operation encoding matches documented input ordering."""
    # For operation 0 (all -1s):
    # f(a,b) = -1 for all inputs ✓

    # For IDENTITY_A (returns first input):
    # LUT should be [-1,-1,-1, 0,0,0, 1,1,1]
    # Position 0,1,2: a=-1 → output=-1 ✓
    # Position 3,4,5: a=0 → output=0 ✓
    # Position 6,7,8: a=1 → output=1 ✓

    identity_a = [-1,-1,-1, 0,0,0, 1,1,1]
    for a in [-1, 0, 1]:
        for b in [-1, 0, 1]:
            idx = (a + 1) * 3 + (b + 1)
            expected_output = a  # IDENTITY_A returns first input
            assert identity_a[idx] == expected_output, f"Mismatch at ({a},{b})"

    print("LUT interpretation validated ✓")
```

### 3.3 Uncovered Areas

| Area | Status | Impact |
|------|--------|--------|
| Named operation indices | Not precomputed | Can't track learning |
| Operation properties (commutative, etc.) | Not defined | Can't verify structure |
| Algebraic closure verification | Not implemented | Can't test homomorphism |
| Input/output relationship | Implicit | Potential bugs |
| Negative/positive operation pairs | Not tracked | Missing symmetry analysis |

---

## Part 4: Comprehensive TODO Index

### Phase A: TensorBoard Retirement (P0)

| ID | Task | File(s) | Effort | Dependencies |
|----|------|---------|--------|--------------|
| A1 | Create `LeanTrainingMonitor` class | `src/training/lean_monitor.py` | Low | None |
| A2 | Remove TensorBoard from `TernaryVAETrainer` | `src/training/trainer.py` | Low | A1 |
| A3 | Remove TensorBoard from `HyperbolicVAETrainer` | `src/training/hyperbolic_trainer.py` | Low | A1 |
| A4 | Remove TensorBoard from `TrainingMonitor` | `src/training/monitor.py` | Medium | A1 |
| A5 | Update config to remove TB settings | `configs/ternary_v5_10.yaml` | Low | A2-A4 |
| A6 | Create post-mortem TB generator | `scripts/analysis/postmortem_tensorboard.py` | Medium | None |
| A7 | Enhance checkpoint with analysis data | `src/artifacts/checkpoint_manager.py` | Medium | A6 |

### Phase B: Precomputation & Performance (P0-P1)

| ID | Task | File(s) | Effort | Dependencies |
|----|------|---------|--------|--------------|
| B1 | Create valuation LUT (19,683 values) | `src/data/precomputed.py` | Low | None |
| B2 | Create ternary representation LUT | `src/data/precomputed.py` | Low | B1 |
| B3 | Replace valuation loops with LUT lookup | Multiple loss files | Medium | B1 |
| B4 | Replace ternary conversion loops | `src/metrics/hyperbolic.py` | Low | B2 |
| B5 | GPU-resident dataset loader | `src/data/loaders.py` | Medium | None |
| B6 | Vectorize coverage evaluation | `src/training/monitor.py` | Medium | None |

### Phase C: Named Operations & Learning Metrics (P1)

| ID | Task | File(s) | Effort | Dependencies |
|----|------|---------|--------|--------------|
| C1 | Define named operations catalog | `src/data/named_operations.py` | Low | None |
| C2 | Implement named op reconstruction check | `src/metrics/learning.py` | Medium | C1 |
| C3 | Implement cluster discovery analysis | `src/metrics/learning.py` | High | None |
| C4 | Add algebraic closure score | `src/metrics/learning.py` | High | None |
| C5 | Integrate metrics into checkpoint | `src/artifacts/checkpoint_manager.py` | Low | C2-C4 |

### Phase D: Hyperbolic Prior Enablement (P1)

| ID | Task | File(s) | Effort | Dependencies |
|----|------|---------|--------|--------------|
| D1 | Enable `use_hyperbolic_prior: true` | `configs/ternary_v5_10.yaml` | Low | None |
| D2 | Verify StateNet delta_sigma applied | `src/training/trainer.py` | Medium | D1 |
| D3 | Verify StateNet delta_curvature applied | `src/training/trainer.py` | Medium | D1 |
| D4 | Test hyperbolic KL vs Euclidean KL | Benchmark script | Medium | D1-D3 |

### Phase E: Documentation & Validation (P2)

| ID | Task | File(s) | Effort | Dependencies |
|----|------|---------|--------|--------------|
| E1 | Add LUT interpretation validation | `src/data/validation.py` | Low | None |
| E2 | Document named operations in README | `README.md` | Low | C1 |
| E3 | Update architecture docs | `docs/ARCHITECTURE.md` | Medium | A-D |
| E4 | Create lean training guide | `docs/LEAN_TRAINING.md` | Medium | A-D |

### Phase F: Short-Circuit Disabled Modules (P0)

| ID | Task | File(s) | Effort | Dependencies |
|----|------|---------|--------|--------------|
| F1 | Add early return in `_compute_hyperbolic_losses` | `hyperbolic_trainer.py` | Low | None |
| F2 | Skip tensor zero creation for disabled losses | `dual_vae_loss.py` | Low | None |
| F3 | Conditional module instantiation | `trainer.py` | Low | None |

---

## Part 5: Execution Order

```
Week 1: Foundation (P0)
├── F1-F3: Short-circuit disabled modules
├── A1-A5: Remove TensorBoard from training loop
├── B1-B2: Create precomputation LUTs
└── B5: GPU-resident dataset

Week 2: Performance & Metrics (P1)
├── B3-B4: Replace loops with LUT lookups
├── B6: Vectorize coverage
├── C1-C2: Named operations catalog & check
└── D1-D3: Enable HyperbolicPrior

Week 3: Analysis & Documentation (P2)
├── A6-A7: Post-mortem TB generator
├── C3-C5: Cluster discovery & algebraic closure
├── D4: Benchmark hyperbolic vs Euclidean
└── E1-E4: Documentation updates
```

---

## Part 6: Success Criteria

### Training Runtime
- [ ] Zero TensorBoard writes during training
- [ ] Epoch time reduced by >20%
- [ ] GPU utilization >80%

### Model Learning
- [ ] Coverage >95% by epoch 100
- [ ] Named operations 100% reconstruction accuracy
- [ ] 3-adic correlation >0.85 (improved from 0.70)
- [ ] Cluster analysis shows 3-adic tree structure

### Health Detection
- [ ] Collapse detected within 5 epochs
- [ ] Divergence detected immediately
- [ ] Plateau detected with configurable patience

### Post-Mortem Analysis
- [ ] Full scalar history reconstructed from checkpoints
- [ ] Embedding visualization available
- [ ] Named operation tracking available

---

## Appendix A: Ternary Definition Verification

### Test Case: IDENTITY_A

```python
# IDENTITY_A: f(a, b) = a
# Expected LUT: [-1,-1,-1, 0,0,0, 1,1,1]

expected = []
for a in [-1, 0, 1]:  # a varies slowest
    for b in [-1, 0, 1]:  # b varies fastest
        expected.append(a)  # Output = first input

assert expected == [-1,-1,-1, 0,0,0, 1,1,1]

# Verify index computation
idx = sum((d + 1) * (3 ** i) for i, d in enumerate(expected))
# = (0)*1 + (0)*3 + (0)*9 + (1)*27 + (1)*81 + (1)*243 + (2)*729 + (2)*2187 + (2)*6561
# = 0 + 0 + 0 + 27 + 81 + 243 + 1458 + 4374 + 13122
# = 19305... wait, let me recalculate

# Actually the operation vector is [digit_0, digit_1, ..., digit_8]
# where digit_k is the output for input index k
# Index formula: Σ (digit_k + 1) * 3^k

identity_a = [-1,-1,-1, 0,0,0, 1,1,1]
idx = sum((d + 1) * (3 ** i) for i, d in enumerate(identity_a))
# = 0*1 + 0*3 + 0*9 + 1*27 + 1*81 + 1*243 + 2*729 + 2*2187 + 2*6561
# = 0 + 0 + 0 + 27 + 81 + 243 + 1458 + 4374 + 13122
# = 19305

# Verify: generate_ternary_operation_by_index(19305) should give identity_a
```

### Test Case: TERNARY_ADD (mod 3)

```python
# TERNARY_ADD: f(a, b) = (a + b) mod 3, mapped to {-1, 0, 1}
# Z₃ addition: -1+(-1)=1, -1+0=-1, -1+1=0, 0+(-1)=-1, 0+0=0, 0+1=1, 1+(-1)=0, 1+0=1, 1+1=-1

expected_add = []
for a in [-1, 0, 1]:
    for b in [-1, 0, 1]:
        # (a + b) mod 3, but in {-1, 0, 1} representation
        result = a + b
        if result < -1:
            result += 3
        elif result > 1:
            result -= 3
        expected_add.append(result)

# expected_add = [1, -1, 0, -1, 0, 1, 0, 1, -1]
```

---

## Appendix B: Removed TensorBoard Code Locations

| File | Method | Lines | Purpose |
|------|--------|-------|---------|
| `monitor.py` | `log_batch()` | 15 | Batch-level scalars |
| `monitor.py` | `log_tensorboard()` | 50 | Epoch-level scalars |
| `monitor.py` | `log_histograms()` | 20 | Weight/grad histograms |
| `monitor.py` | `log_manifold_embedding()` | 30 | Embedding projector |
| `hyperbolic_trainer.py` | `log_hyperbolic_batch()` | 15 | Hyperbolic batch scalars |
| `hyperbolic_trainer.py` | `log_hyperbolic_epoch()` | 25 | Hyperbolic epoch scalars |
| `hyperbolic_trainer.py` | `_log_standard_tensorboard()` | 40 | Standard epoch logging |

**Total removable: ~195 lines** of TensorBoard logging code.

---

**End of Plan**
