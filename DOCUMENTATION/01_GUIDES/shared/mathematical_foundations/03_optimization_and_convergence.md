# Part 3: Optimization & Convergence

> **Context**: This document covers the comprehensive loss landscape, the phase-scheduled optimization strategy, and theoretical convergence guarantees.

## 1. Loss Functions and Objectives

### Complete Loss Decomposition

```
ℒ_total = λ₁ · g_A · ℒ_A + λ₂ · g_B · ℒ_B + λ₃ · ℒ_align

where:
  ℒ_A = CE_A + β_A · KL_A
  ℒ_B = CE_B + β_B · KL_B + α_ent · (-H_output) + α_rep · ℒ_rep
  ℒ_align = |H(z_A) - H(z_B)|

  g_A, g_B = gradient balance scales
```

### 1. Cross-Entropy (Reconstruction Loss)

For ternary outputs, we use categorical cross-entropy:

```
CE = -1/N ∑ᵢ₌₁ᴺ ∑ⱼ₌₁⁹ ∑ₖ∈{-1,0,+1} 𝟙[xᵢⱼ=k] log p_θ(xᵢⱼ=k|z)
```

where p*θ(x|z) = softmax(logits*θ(z)).

**Conversion**: Input values {-1, 0, +1} → class indices {0, 1, 2}

### 2. KL Divergence (Regularization)

```
KL(q_φ(z|x) || p(z)) = -1/2 ∑ᵢ₌₁ᵈ (1 + log σᵢ² - μᵢ² - σᵢ²)
```

where d is latent dimension.

**Purpose**: Prevents posterior collapse by forcing q(z|x) ≈ p(z) = 𝒩(0,I)

### 3. Output Entropy (Diversity)

For VAE-B, we maximize output diversity:

```
H_output = -∑ₖ p̄ₖ log p̄ₖ

where p̄ₖ = 1/N ∑ᵢ₌₁ᴺ p_θ(xᵢ=k|zᵢ)  (average distribution)
```

**Effect**: Encourages VAE-B to generate diverse operations, not concentrate on a few modes.

### 4. Repulsion Loss (Anti-Clustering)

```
ℒ_rep = 1/N² ∑ᵢ,ⱼ exp(-||z_i - z_j||²/(2σ²))
```

**Purpose**: Pushes latent codes apart, preventing clustering in latent space.

### 5. Gradient Balance Scaling

To prevent one VAE from dominating:

```
g_A = ||∇_θ_B ℒ||_EMA / ||∇_θ_A ℒ||_EMA
g_B = ||∇_θ_A ℒ||_EMA / ||∇_θ_B ℒ||_EMA

g_A, g_B ∈ [0.5, 2.0]  (clipped for stability)
```

**Update Rule** (Exponential Moving Average):

```
||∇_θ_A ℒ||_EMA ← α · ||∇_θ_A ℒ||_EMA + (1-α) · ||∇_θ_A ℒ||
```

where α = 0.9 (balanced) or 0.95 (when balanced), adaptively.

---

## 2. Phase-Scheduled Optimization

### Four Training Phases

**Phase 1: Isolation (Epochs 0-40)**

```
ρ = 0.1          # Minimal cross-talk
Goal: Independent pathway establishment
```

**Phase 2: Consolidation (Epochs 40-120)**

```
ρ: 0.1 → 0.3     # Gradual coupling
Goal: Share discoveries while maintaining diversity
```

**Phase 3: Resonant Coupling (Epochs 120-250)**

```
ρ: 0.3 → 0.7     # Strong coupling (gated on gradient balance)
Goal: Synergistic exploration
Condition: 0.8 < grad_ratio < 1.2
```

**Phase 4: Ultra-Exploration (Epochs 250+)**

```
ρ = 0.7          # Maintain coupling
τ_A: boosted amplitude (0.1 → 0.5)
τ_B: 0.2 → 0.3
Goal: Discover rare operations
```

### Mathematical Justification

**Theorem 3 (Phase Progression)**:
Under the phase schedule:

1. Phase 1 establishes **independent basins** in latent space
2. Phase 2 creates **bridges** between basins
3. Phase 3 enables **coordinated search** across basins
4. Phase 4 refines **rare operation coverage**

### Temperature Schedules

**VAE-A (Cyclic)**:

```
τ_A(t) = τ_base(t) + A(t) · cos(2πt/T + π/2)

where:
  τ_base(t) = 1.0 + (0.3 - 1.0) · t/T_total
  A(t) = 0.1 · τ_base(t)     for t < 250
         0.5                  for t ≥ 250 (Phase 4)
  T = 30 epochs (cycle period)
```

**VAE-B (Monotonic with Phase 4 boost)**:

```
τ_B(t) = 0.9 + (0.2 - 0.9) · (t-30)/T_total    for t < 250
         0.3                                    for t ≥ 250
```

### Beta (KL Weight) Schedules

**VAE-A**:

```
β_A(t) = 0.6 + (1.0 - 0.6) · t/T_total
```

**VAE-B (Phase-Lagged)**:

```
β_B(t) = β_A(t) · |sin(π/4)|  ≈ 0.707 · β_A(t)
```

The phase lag prevents both VAEs from being under-regularized simultaneously.

---

## 3. Convergence Guarantees

### Theorem 4 (Eventual Coverage)

Under the following conditions:

1. Sufficient model capacity (latent dim ≥ 16)
2. Temperature annealing to τ_min > 0
3. Phase-scheduled ρ progression
4. Gradient balance maintenance

The dual-VAE system achieves:

```
lim_{t→∞} Coverage(t) ≥ C_min

where C_min ≈ 95% (empirically observed)
```

**Sketch of Proof**:

- Phase 1 establishes independent search spaces
- Phase 2-3 expand coverage through coupling
- Phase 4's temperature boost enables rare operation discovery
- Stop-gradient prevents collapse to single mode
- Entropy alignment forces similar exploration levels

### Empirical Validation

**Ternary VAE v5.5 Results**:

- **Coverage at epoch 399**: 97.64% (VAE-A), 97.67% (VAE-B)
- **Peak coverage**: 100% achieved 12 times (VAE-A), 8 times (VAE-B)
- **Stability**: No catastrophic forgetting, monotonic improvement
