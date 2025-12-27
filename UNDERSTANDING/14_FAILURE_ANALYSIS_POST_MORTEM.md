# Comprehensive Failure Log & Post-Mortem

**Date:** 2025-12-27
**Scope:** v1.0 (Inception) to v5.12 (Current)
**Purpose:** To document every dead end, broken feature, and failed hypothesis to prevent regression.

---

## 1. Chronological Failure Analysis (The "Changes" Context)

A history of _why_ we changed versions, driven by specific failures at each stage.

### v1.0 - v3.0: The "Naive" Era

- **Approach:** Standard Euclidean VAEs on discrete codon data.
- **The Failure:** **Posterior Collapse**. The KL divergence term pushed all latent code to zero because the model found it easier to predict the average codon frequency than to learn a meaningful latent map.
- **Why It Failed:** Continuous Gaussian priors cannot naturally map to discrete, sparse 64-dimensional categorical data without cleaner gradients.

### v4.0: The "Hierarchical" Era

- **Approach:** Stacked VAEs to learn hierarchy.
- **The Failure:** **Training Instability**. The lower layers collapsed before the upper layers could learn abstract concepts.
- **Why It Failed:** No explicit enforcement of hierarchy. The model merely "had capacity" for hierarchy but no incentive to use it.

### v5.0: The "Hyperbolic" Era

- **Approach:** Project latent space into Poincaré ball.
- **The Failure:** **Inverted Hierarchy**. The model placed the root at the edge and leaves at the center (or vice versa unpredictably).
- **Why It Failed:** Hyperbolic space provides _room_ for trees, but doesn't define _direction_. Without a "Radial Loss," the model utilized the space arbitrarily.

### v5.10: The "Anti-Grokking" Era (Major Regression)

- **Approach:** Combine Curriculum Learning + Aggressive Beta Warmup.
- **The Failure:** **Early Peaking & Collapse**. Correlation peaked at Epoch 8 and dropped to zero.
- **Why It Failed:** The curriculum showed "easy" patterns too early. The model memorized them. When the beta warmup kicked in (increasing regularization), the model was trapped in a local optimum (memorization) and couldn't reorganize its latent space to fit the general rule.

### v5.11: The "Frozen" Era

- **Approach:** Transfer learning. Freeze a good v5.5 encoder, train only the projection.
- **The Failure:** **Structural Rigidity**. 0.0 correlation.
- **Why It Failed:** You cannot impose geometry _post-hoc_. If the encoder learned a Euclidean map, projecting it to Hyperbolic space just distorts the Euclidean distances; it doesn't create new phylogenetic relationships.

---

## 2. Component Failures (Broken Features)

Specific modules that exist in the codebase but failed to perform/integrate.

| Component            | Status    | Failure Mode                 | Root Cause                                                                                          |
| :------------------- | :-------- | :--------------------------- | :-------------------------------------------------------------------------------------------------- |
| **SwarmVAE**         | ❌ Failed | `KeyError: 'logits'`         | Incompatible output format with the standard Trainer loop.                                          |
| **HomeostaticPrior** | ❌ Failed | `TypeError: Tensor vs Numpy` | Internal calculation mixed pytorch tensors and numpy arrays without conversion.                     |
| **MixedRiemannian**  | ❌ Failed | Optimizer Iteration Error    | `Geoopt` optimizers handle parameter groups differently than AdamW.                                 |
| **Dual-Head VAE**    | ❌ Failed | Ignored Structure Head       | The "Structure" head had no impact on the reconstruction loss, so gradients vanished.               |
| **Contrastive Loss** | ❌ Failed | Low Signal                   | Ternary data is too sparse for standard SimCLR-style contrastive views without better augmentation. |

---

## 3. Theoretical Failures (Disproven Hypotheses)

Scientific assumptions we held that turned out to be wrong.

### Hypothesis A: "P-adic Loss is Harmful"

- **Observation**: Removing p-adic loss improved reconstruction in v5.10.
- **The Reality**: **False Negative**.
- **Explanation**: P-adic loss and Reconstruction loss have "orthogonal gradients" (they want to move points in different directions). With a **Constant Beta**, they fight. With **Cyclical Beta**, they synergize. The failure was in the _schedule_, not the loss.

### Hypothesis B: "Low Learning Rate is Safer"

- **Observation**: `lr=0.001` is standard for VAEs.
- **The Reality**: **Too Slow**.
- **Explanation**: Hyperbolic space has "more volume" near the boundary. To move a point from the center to the edge (`r=0.9`) requires significant gradient magnitude. Low LR kept points stuck near the origin (Gaussian blob). `lr=0.005` was necessary to push them out.

### Hypothesis C: "Structure Requires Complex Decoders"

- **Observation**: Maybe the decoder isn't powerful enough?
- **The Reality**: **Irrelevant**.
- **Explanation**: The bottleneck was the _Latent Geometry_. Once we fixed the geometry (Hyperbolic + P-adic + Cyclical), a simple linear decoder worked perfectly. Adding complexity to the decoder just slowed down training without fixing the structural issues.

---

## 4. Summary of "What Works" (The Survivors)

Out of 98 experiments and 11 versions, only these survived:

1.  **Architecture**: `TropicalVAE` (Accuracy) or `SimpleHyperbolic` (Structure).
2.  **Geometry**: Poincaré Ball with **Monotonic Radial Loss**.
3.  **Dynamics**: **Cyclical Beta Schedule** (0.0 -> 0.1 -> 0.0).
4.  **Optimization**: High Learning Rate (0.005).

---

## 5. Deep Introspection (Psychology of Failure)

Why did we suffer so much in the v5.10 regression? It wasn't just code; it was mindset.

### The Complexity Trap (Over-Engineering)

- **The Trap**: When v5.5 worked partially, we assumed the solution was _more_ complexity (Homeostasis, Dual-VAEs, Curriculum).
- **The Result**: We built a "Rube Goldberg Machine." It was impossible to debug. When it failed, we didn't know if it was the loss, the schedule, or the architecture.
- **Lesson**: Always distill. v5.12 works because it removed 90% of the code from v5.11.

### Fighting the Gradient (Controller vs. Dynamics)

- **The Trap**: We tried to force the model to behave using "Homeostatic Controllers" (artificial rules).
- **The Result**: The model fought back. The gradients wanted to collapse, and our controller pushed them apart, leading to oscillating instability.
- **Lesson**: **Cyclical Beta** works because it _cooperates_ with the gradient. It allows collapse (rest) and then demands structure (exercise). It flows _with_ the training dynamics, not against them.

### Premature Optimization (Freezing)

- **The Trap**: We thought "The v5.5 encoder is good, let's freeze it to save time."
- **The Result**: We locked in the errors of v5.5. Geometry must be learned _during_ encoding, not tacked on after.
- **Lesson**: True representation learning must be end-to-end.

---

## 6. Alternative Approaches (The Roads Not Taken)

If `TropicalVAE` eventually hits a ceiling, here is where we should look next.

### Pathway A: Discrete Diffusion (Masked Generative Models)

- **Concept**: Instead of a continuous latent space (VAE), use a Discrete Diffusion process (like `BitDiffusion` or `MaskGIT`).
- **Why it might work**: DNA is inherently discrete. Diffusion works well on discrete tokens without the "blurriness" of VAEs.
- **Why we skipped it**: We _need_ the Hyperbolic Latent Space for vaccine targets. Diffusion doesn't give you a map; it just gives you samples.

### Pathway B: Riemannian Flow Matching

- **Concept**: A newer alternative to Diffusion that defines vector fields on manifolds (like the Hyperbolic plane).
- **Why it might work**: It could generate higher quality samples while strictly adhering to the Poincare geometry.
- **Why we skipped it**: Extremely complex math ($d_t x = u_t(x)$ on manifolds). We stuck to VAEs for accessibility.

### Pathway C: GPLVM (Gaussian Process Latent Variable Models)

- **Concept**: Non-parametric Bayesian approach.
- **Why it might work**: Handles uncertainty much better than VAEs. Perfect for small datasets (rare viruses).
- **Why we skipped it**: Does not scale to 200,000 sequences (O(N^3) complexity).

### Pathway D: Tropical Transformers

- **Concept**: Replace the standard self-attention (Softmax) with Tropical Attention (Max-Plus).
- **Why it might work**: Theoretically can model the "min/max" logic of biological thresholds perfectly.
- **Why we skipped it**: We built `TropicalVAE` (linear layers) first. A Transformer version is the logical next step.
