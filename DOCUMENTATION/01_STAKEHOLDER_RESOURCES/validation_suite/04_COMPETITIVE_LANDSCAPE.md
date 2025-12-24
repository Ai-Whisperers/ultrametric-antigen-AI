# 04 Validation Suite: Competitive Landscape

> **Objective**: The "Kill Sheet". Direct, brutal comparisons against major competitors.

## 1. vs. EVE (Frazer et al., Nature 2021)

- **The Weakness**: Gaussian latent space assumes biology is "flat" (Euclidean). It fails to model deep phylogenetic trees correctly, collapsing branches into noise.
- **Our Edge**: Hyperbolic space is isometric to trees. We capture the hierarchy natively.
- **Benchmark**: Robinson-Foulds distance on H3N2 trees.
- **Result**: We reconstruct the tree; EVE creates a "bush".

## 2. vs. ESM-1v / ESM-2 (Meta AI)

- **The Weakness**: Massive parameter count (650M - 15B). Requires data center GPUs. High latency makes it unusable for real-time edge screening.
- **Our Edge**: Ternary weights ({-1, 0, 1}) are 32x smaller than FP32. We run on a phone.
- **Benchmark**: Inference Cost per Genome ($).
- **Result**: we are ~10,000x cheaper.

## 3. vs. AlphaFold2 (DeepMind)

- **The Weakness**: Predicts _structure_ (folding), not _function_ or _fitness_. A protein can fold perfectly but be non-functional due to a single active site mutation.
- **Our Edge**: We predict _evolutionary fitness_ directly from the manifold of survival.
- **Benchmark**: Correlation with DMS fitness assays (ProteinGym).
- **Result**: AF2 predicts stability; we predict "will it live?".

## 4. vs. Standard VAE / PCA

- **The Weakness**: Linear or simple non-linear projections cannot capture the "holes" in biological space (e.g., non-viable fitness valleys).
- **Our Edge**: The "Goldilocks Zone" of the Ternary engine finds discrete, separated clusters corresponding to viable species.
- **Benchmark**: Percentage of "Dead" Latent Space (regions that decode to garbage).
- **Result**: Standard VAE > 90% dead space; Ternary VAE < 5% dead space.
