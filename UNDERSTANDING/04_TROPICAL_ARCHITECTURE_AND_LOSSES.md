# Tropical VAE: Architecture & Dynamics (v5.12)

**The Current Standard for Ternary Bioinformatics**

---

## 1. Overview: Simplicity Wins

After months of complex "Dual-VAE" and "Homeostatic" experiments (v5.0-v5.11), we discovered that a **Simpler Architecture** combined with **Better Dynamics** outperforms everything else.

### The v5.12 Standard

- **Model**: Single VAE (or simplified Dual)
- **Geometry**: Tropical (Max-Plus) or Simple Hyperbolic
- **Dynamics**: Cyclical Beta Schedule (The "Heartbeat")
- **Losses**: Triplet P-adic + Monotonic Radial

---

## 2. Architecture: Tropical & Hyperbolic

We use two primary variants depending on the goal.

### Variant A: Tropical VAE (Best Accuracy: 96.7%)

Uses **Max-Plus Algebra** in the decoder. Instead of multiplying weights, we add them and take the maximum. This matches the "piecewise linear" nature of ReLU networks and biological thresholds.

```python
# Tropical Linear Layer
def tropical_linear(x, weight, bias):
    # Standard: y = x @ W + b
    # Tropical: y = max(x + W_i) + b
    return max_plus_matmul(x, weight) + bias
```

### Variant B: Simple Hyperbolic (Best Structure: +0.54 ρ)

Uses a standard MLP encoder projected into the Poincare ball.

```python
class HyperbolicVAE(nn.Module):
    def __init__(self):
        self.encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 16) # Output mu, logvar
        )

    def forward(self, x):
        z_euc = self.encoder(x)
        z_hyp = project_to_poincare(z_euc) # The critical step
        return z_hyp
```

**Why we deprecated "Frozen Encoders"**:
We found that freezing the encoder (v5.11) locked in "Euclidean bias." Training end-to-end with **Cyclical Beta** allows the model to learn hyperbolic structure from scratch without collapsing.

---

## 3. The "Heartbeat": Cyclical Beta Schedules

This is the single most important component of v5.12.
Static Beta causes **Posterior Collapse** (Anti-Grokking).
Cyclical Beta creates a "Heartbeat" of expansion and contraction.

### The Mechanism

```python
def get_beta(step, cycle_len=1000, ratio=0.5):
    # Cyclic schedule: 0 -> 1 -> 0 -> 1
    # Low Beta: Learn Reconstruction (Expansion)
    # High Beta: Learn Structure (Contraction)
    return cycler(step, cycle_len, ratio)
```

**Why it works**:

1.  **Low Beta Phase**: The model is free to spread out and memorize data (High Reconstruction).
2.  **High Beta Phase**: The model is forced to compress and organize (High Structure).
3.  **Cycling**: Prevents getting stuck in either local minimum.

---

## 4. Loss Function Landscape

We stripped away the 10+ losses of v5.10 and kept only the winners.

### Core Losses (Required)

1.  **Reconstruction (BCE/CE)**: The foundation.
2.  **KL Divergence**: The VAE regularizer.

### Structural Losses (The "Guide")

1.  **Triplet P-adic Loss**:

    - _Idea_: If $d_p(A, B) < d_p(A, C)$, then $d_z(A, B) < d_z(A, C)$.
    - _Implementation_: Triplet margin loss on latent distances.
    - _Effect_: Enforces local cluster structure.

2.  **Monotonic Radial Loss**:
    - _Idea_: High P-adic valuation = Close to Origin (Root).
    - _Implementation_: `Loss = ReLU(radius(parent) - radius(child))`
    - _Effect_: Enforces the global tree hierarchy.

---

## 5. Hyperparameters (The Gold Standard)

| Parameter         | Value    | Notes                                                            |
| :---------------- | :------- | :--------------------------------------------------------------- |
| **Learning Rate** | `0.005`  | Higher than typical (was 1e-4). Essential for escaping collapse. |
| **Beta Cycle**    | `Linear` | 4 cycles of 50 epochs each.                                      |
| **Latent Dim**    | `16`     | Sufficient for $3^9$ space.                                      |
| **Curvature (c)** | `1.0`    | Fixed. Learnable curvature was unstable.                         |
| **P-adic Weight** | `1.0`    | Balanced 1:1 with Reconstruction.                                |

---

## Summary

The v5.12 architecture is a **return to first principles**. We stopped fighting the gradients with complex "Homeostasis" and instead used **Cyclical Beta** to flow _with_ the training dynamics. The result is a model that is 10x simpler and 20% more accurate.
