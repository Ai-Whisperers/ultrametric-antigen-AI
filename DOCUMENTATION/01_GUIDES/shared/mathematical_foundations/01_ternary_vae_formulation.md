# Part 1: Ternary VAE Formulation & Architecture

> **Context**: This document covers the fundamental problem statement, the ternary operation space, and the core Dual-VAE architecture designed to solve it.

## 1. Problem Statement

### The Ternary Operation Space

A **ternary operation** is a 9-dimensional truth table where each input/output takes values from the ternary alphabet **Σ = {-1, 0, +1}**.

**Total possible operations**: 3^9 = **19,683**

**Mathematical Representation**:

```
f: Σ^9 → ℝ^9
where f = (f₁, f₂, ..., f₉) and each fᵢ ∈ {-1, 0, +1}
```

### The Challenge

**Goal**: Learn a latent representation that can:

1. **Generate** all 19,683 operations with high probability
2. **Maintain diversity** across the operation space
3. **Avoid collapse** to a subset of operations
4. **Generalize** to unseen combinations

**Why is this hard?**

- Standard VAEs suffer from **posterior collapse**: the decoder ignores the latent code
- Single-pathway models exhibit **mode collapse**: concentrate on easy-to-generate operations
- High-dimensional discrete spaces are difficult to explore systematically

---

## 2. Variational Autoencoder Theory

### Standard VAE Formulation

A Variational Autoencoder (VAE) maximizes the Evidence Lower Bound (ELBO):

```
log p(x) ≥ ELBO = 𝔼_{q_φ(z|x)}[log p_θ(x|z)] - D_KL(q_φ(z|x) || p(z))
            ^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            Reconstruction term      Regularization term
```

**Components**:

- **Encoder**: q_φ(z|x) - probabilistic mapping from data to latent space
- **Decoder**: p_θ(x|z) - probabilistic mapping from latent to data
- **Prior**: p(z) = 𝒩(0, I) - standard Gaussian
- **Parameters**: φ (encoder), θ (decoder)

### The Reparameterization Trick

To enable backpropagation through stochastic sampling:

```
z = μ_φ(x) + σ_φ(x) ⊙ ε,  where ε ~ 𝒩(0, I)
```

This separates the stochasticity (ε) from the learnable parameters (μ, σ).

### Temperature-Scaled Sampling

We extend this with temperature scaling for exploration control:

```
z = μ_φ(x) + τ · σ_φ(x) ⊙ ε
```

where **τ** (temperature) controls exploration:

- **τ > 1**: High temperature → more exploration, diverse samples
- **τ < 1**: Low temperature → less exploration, deterministic samples
- **τ = 0**: Deterministic (z = μ)

---

## 3. Dual-VAE Architecture

### Why Two VAEs?

A single VAE faces a fundamental trade-off:

- **High β** (strong regularization) → poor reconstruction, underfitting
- **Low β** (weak regularization) → posterior collapse, ignores latent

**Solution**: Use **two VAEs with complementary objectives**:

1. **VAE-A (Chaotic Regime)**:

   - High temperature → exploration
   - Moderate β → balanced KL/reconstruction
   - Task: **Discover new operations**

2. **VAE-B (Frozen Regime)**:
   - Low temperature → exploitation
   - Residual connections → strong reconstruction
   - Task: **Consolidate discoveries**

### Mathematical Formulation

**Joint Objective**:

```
ℒ_total = λ₁ · ℒ_A + λ₂ · ℒ_B + λ₃ · |H(z_A) - H(z_B)|
          ^^^^^^^^^^   ^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^
          VAE-A loss   VAE-B loss   Entropy alignment
```

Where:

```
ℒ_A = 𝔼_q_A[log p_A(x|z_A)] - β_A · D_KL(q_A(z_A|x) || p(z))

ℒ_B = 𝔼_q_B[log p_B(x|z_B)] - β_B · D_KL(q_B(z_B|x) || p(z))
      + α_ent · H_output(p_B(x|z_B))
      + α_rep · ℒ_repulsion(z_B)
```

**Entropy Alignment**:

```
H(z) = -∑ᵢ ∫ p(zᵢ) log p(zᵢ) dzᵢ   (estimated via histogram)
```

Forces VAE-A and VAE-B to explore similar entropy regimes, preventing divergence.
