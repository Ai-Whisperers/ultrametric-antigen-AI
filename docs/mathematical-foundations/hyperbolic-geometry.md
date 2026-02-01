# Hyperbolic Geometry

> **Poincare ball model for hierarchical latent representations.**

---

## Overview

The latent space uses the **Poincare ball** model of hyperbolic geometry:

$$\mathcal{B}^n = \{x \in \mathbb{R}^n : \|x\| < 1\}$$

### Properties

| Property | Euclidean | Hyperbolic |
|----------|-----------|------------|
| Volume growth | Polynomial | Exponential |
| Tree embedding | Poor | Natural |
| Hierarchy | Implicit | Explicit (radial) |

---

## Poincare Ball Operations

### Distance

$$d_{\mathcal{B}}(x, y) = \text{arccosh}\left(1 + 2\frac{\|x - y\|^2}{(1 - \|x\|^2)(1 - \|y\|^2)}\right)$$

### Mobius Addition

$$x \oplus y = \frac{(1 + 2\langle x, y \rangle + \|y\|^2)x + (1 - \|x\|^2)y}{1 + 2\langle x, y \rangle + \|x\|^2\|y\|^2}$$

### Exponential Map

$$\exp_x(v) = x \oplus \left(\tanh\left(\frac{\lambda_x\|v\|}{2}\right)\frac{v}{\|v\|}\right)$$

Where $\lambda_x = \frac{2}{1 - \|x\|^2}$ is the conformal factor.

### Logarithmic Map

$$\log_x(y) = \frac{2}{\lambda_x}\text{arctanh}(\|-x \oplus y\|)\frac{-x \oplus y}{\|-x \oplus y\|}$$

---

## Implementation

```python
from src.geometry.poincare import PoincareBall

ball = PoincareBall(curvature=1.0)

# Project to ball (ensure ||x|| < 1)
z_poincare = ball.project(z_euclidean)

# Compute distance
dist = ball.distance(x, y)

# Mobius addition
z = ball.mobius_add(x, y)

# Exponential map (tangent → manifold)
z = ball.exp_map(x, v)

# Logarithmic map (manifold → tangent)
v = ball.log_map(x, y)
```

---

## Biological Interpretation

### Radial Position

| Position | Interpretation |
|----------|----------------|
| Near origin | Conserved, high-valuation sequences |
| Near boundary | Diverse, low-valuation sequences |

### Angular Position

Captures functional similarity:
- Similar function → similar angle
- Different function → different angle

---

## Curvature

Negative curvature controls embedding capacity:

| Curvature (c) | Effect |
|---------------|--------|
| c = 0 | Euclidean (flat) |
| c = 1 | Standard hyperbolic |
| c > 1 | More negative, more tree-like |

```python
# Learnable curvature
class HyperbolicVAE(nn.Module):
    def __init__(self):
        self.curvature = nn.Parameter(torch.tensor(1.0))
```

---

## Hyperbolic Prior

For VAE regularization, use wrapped normal distribution:

```python
def hyperbolic_kl_divergence(mu, logvar, curvature):
    """KL divergence for wrapped normal on Poincare ball."""
    var = torch.exp(logvar)
    mu_norm_sq = (mu ** 2).sum(dim=-1)

    kl = (mu_norm_sq + var - 1 - logvar) / (2 * curvature)
    return kl.mean()
```

---

## References

1. Nickel, M., & Kiela, D. (2017). Poincare Embeddings for Learning Hierarchical Representations
2. Ganea, O., et al. (2018). Hyperbolic Neural Networks
3. Mathieu, E., et al. (2019). Continuous Hierarchical Representations with Poincare Variational Auto-Encoders

---

_Last updated: 2025-12-28_
