# Master Synthesis: The Evolution of Ternary VAEs

**The complete narrative of how we went from a simple question to a working hyperbolic system.**

---

## 1. The Core Question

**Why does the genetic code use triplets?**
Why use 64 codons for only 20 amino acids?

**Hypothesis**: The genetic code is an **error-correcting code**. The "Wobble" hypothesis (Crick) hints at a hierarchical protection mechanism.

- **Pos 1 & 2**: High Valuation (Critical).
- **Pos 3**: Low Valuation (Flexible).

This hierarchy matches **P-adic Analysis**.

---

## 2. The Journey (History of Failure)

We iterated through 11 major versions, facing significant challenges at each step.

### Era 1: The Naive Era (v1-v3)

- **Approach**: Standard Euclidean VAEs.
- **Result**: **Posterior Collapse**. The model ignored the latent code because fitting a continuous Gaussian prior to discrete ternary data is mathematically ill-posed.

### Era 2: The Hyperbolic Era (v5.0)

- **Approach**: Projected latent space to Poincaré ball.
- **Result**: **Inverted Hierarchy**. The model used the space, but arbitrarily placed ancestors at the edge and children at the center.

### Era 3: The "Anti-Grokking" Regression (v5.10)

- **Approach**: Added "Homeostatic Control", "Curriculum Learning", and "Aggressive Warmup".
- **Result**: **Catastrophic Collapse**. The model "cheated" on the curriculum (memorizing easy patterns) and couldn't generalize.
- **Diagnostic**: We found gradients between Structure Loss and Reconstruction Loss were **orthogonal** or **conflicting**.

---

## 3. The Solution: The "Big Pivot" (v5.12)

We stripped away the complexity and fixed the core dynamics.

### A. Cyclical Beta Schedule

Instead of a linear warmup, we use a "Heartbeat" (0.0 $\to$ 0.1 $\to$ 0.0) every 10 epochs.

- **Low Beta**: Model learns to **reconstruct** (Formation).
- **High Beta**: Model learns **structure** (Organization).
- **Result**: Solved the gradient conflict.

### B. The Synergistic Loss Landscape

We stopped using "P-adic Loss" in isolation.

- **Monotonic Radial Loss**: Forces the hierarchy ($r \propto$ depth).
- **Triplet Loss**: Forces local clustering.
- **Synergy**: Combined, they fix the "Inverted Hierarchy" issue.

### C. Tropical Algebra

- **Insight**: Standard Matrix Multiplication ($Wx+b$) is continuous.
- **Solution**: Tropical Algebra ($\max(x+w)$) is the natural math of discrete decisions.
- **Result**: 96.7% Accuracy (TropicalVAE).

---

## 4. Final Recommended Configuration

### For Production (Accuracy)

- **Model**: `TropicalVAE`
- **Config**: `lr=0.005`, `beta_type="cyclical"`

### For Research (Structure)

- **Model**: `SimpleHyperbolicVAE`
- **Losses**: `triplet_padic` + `monotonic_radial`
- **Correlation**: Sperman $\rho = +0.5465$

---

## 5. Post-Mortem: Lessons Learned

1.  **Complexity Trap**: We tried to fix v5.10 by adding _more_ controls (Homeostasis). The solution was _less_ (Simple VAE + Better Schedule).
2.  **Don't Fight the Gradient**: Artificial controllers fight the training dynamics. Cyclical schedules cooperate with them.
3.  **Geometry Matters**: You cannot learn a tree in flat space. The geometry must match the data.

---

## 6. Alternative Roads Not Taken

If we hit a wall in the future:

- **Discrete Diffusion**: Using `MaskGIT` on 3-adic tokens.
- **Riemannian Flow Matching**: Continuous flows on the manifold.
- **Tropical Transformers**: Replacing Softmax with Max-Plus attention.
