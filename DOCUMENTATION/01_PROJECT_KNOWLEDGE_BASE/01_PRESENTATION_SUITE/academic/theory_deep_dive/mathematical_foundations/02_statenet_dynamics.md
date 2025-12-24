# Part 2: Dynamics & Meta-Learning

> **Context**: This document details the dynamic mechanisms that allow the Dual-VAE to function: the stop-gradient information flow and the StateNet meta-controller.

## 1. Stop-Gradient Cross-Injection

### The Information Flow Problem

If VAE-A and VAE-B train independently:

- ❌ No knowledge sharing
- ❌ Redundant exploration
- ❌ Slower convergence

If they share gradients directly:

- ❌ One pathway dominates
- ❌ Collapse to single mode
- ❌ Loss of diversity

### Solution: Stop-Gradient Cross-Injection

**Mechanism**:

```
z̃_A = (1 - ρ) · z_A + ρ · sg(z_B)
z̃_B = (1 - ρ) · z_B + ρ · sg(z_A)

where sg(·) = stop_gradient(·)  [no backprop through this term]
```

**Permeability ρ**:

- **ρ = 0**: Complete isolation (no information flow)
- **ρ = 1**: Complete mixing (averaged representations)
- **0 < ρ < 1**: Partial coupling (controlled flow)

### Mathematical Properties

**Theorem 1 (One-Way Information Flow)**:
The stop-gradient operator ensures:

```
∂ℒ_A/∂θ_B = 0   and   ∂ℒ_B/∂θ_A = 0
```

**Proof**:
Since z_B appears in ℒ_A only through sg(z_B), and ∂sg(z_B)/∂θ_B = 0 by definition, the chain rule gives ∂ℒ_A/∂θ_B = 0. Similarly for ∂ℒ_B/∂θ_A.

**Corollary**: VAE-A and VAE-B have **independent gradient flows**, preventing dominance.

**Theorem 2 (Information Transfer)**:
Despite independent gradients, information flows through:

```
I(z_A; z_B) > 0  when ρ > 0
```

where I(·;·) is mutual information.

**Proof Sketch**:
The latent codes z̃_A and z̃_B contain components from both pathways. During decoding, if p_A(x|z̃_A) reconstructs x well, it must utilize information from z_B. This creates an indirect coupling through the data space.

---

## 2. StateNet Meta-Learning

### Motivation

Training dynamics depend on multiple factors:

- Entropy levels H_A, H_B
- KL divergences KL_A, KL_B
- Gradient balance ratio
- Current phase and permeability

**Problem**: Manual tuning is brittle and suboptimal.

**Solution**: Learn to adapt hyperparameters using a meta-controller.

### StateNet Architecture

**Autodecoder Design**:

```
State Vector (9D):
s = [H_A, H_B, KL_A, KL_B, grad_ratio, ρ, λ₁, λ₂, λ₃]

Encoder:
h = Tanh(LayerNorm(Linear(s)))    # Normalize and compress
z_state = Linear(h)                # Latent state (8D)

Decoder:
corrections = Tanh(Linear(ReLU(Linear(z_state))))  # 4D output
[Δlr, Δλ₁, Δλ₂, Δλ₃] = corrections
```

### Hyperparameter Update Rules

**Learning Rate**:

```
lr_new = lr_old · (1 + α_lr · Δlr)
lr_new = clip(lr_new, 1e-6, 0.01)
```

**Loss Weights**:

```
λ₁_new = clip(λ₁_old + α_λ · Δλ₁, 0.5, 0.95)
λ₂_new = clip(λ₂_old + α_λ · Δλ₂, 0.5, 0.95)
λ₃_new = clip(λ₃_old + α_λ · Δλ₃, 0.15, 0.75)
```

where:

- **α_lr = 0.05** (5% LR modulation scale)
- **α_λ = 0.01** (1% lambda modulation scale)

### Training StateNet

**Objective**:

```
ℒ_StateNet = ||corrections||₂²   (implicit regularization)
```

StateNet learns through:

1. **Gradient signals** from main VAE loss
2. **Implicit reward** from coverage improvement
3. **Regularization** toward small corrections

**No explicit reward** is needed - StateNet learns what corrections improve training through backpropagation of the total loss.
