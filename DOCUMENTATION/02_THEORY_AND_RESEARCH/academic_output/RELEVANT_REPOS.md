# Relevant Repositories (The "Cheat Sheet")

> **Goal:** Don't reinvent the wheel. Steal these implementations.

These repositories solved specific problems we are facing.

---

## 1. Hyperbolic Graph Neural Networks (`hgcn`)

- **Repo:** [HazyResearch/hgcn](https://github.com/HazyResearch/hgcn) (Stanford)
- **Why:** They solved the "Hierarchy Problem" in graphs.
- **Key Files to Copy:**
  - `layers/hyp_layers.py`: Contains the `HyperbolicGraphConvolution` class. We can use this for our **StateNet controller** (treating the trajectory as a graph).
  - `manifolds/poincare.py`: Robust implementation of `mobius_add` (better than ours).

## 2. Poincaré Embeddings (C++ Optimized)

- **Repo:** [facebookresearch/poincare-embeddings](https://github.com/facebookresearch/poincare-embeddings)
- **Why:** Speed. Python is too slow for 1 Million+ nodes.
- **Key Insight:** They use "Burn-in" phases (optimizing unrelated parameters first).
- **Action:** If we scale to the full Human Genome (3B base pairs), we must port our training loop to C++ using their `dist` function.

## 3. Disentanglement Lib

- **Repo:** [google-research/disentanglement_lib](https://github.com/google-research/disentanglement_lib)
- **Why:** We need to _prove_ our features are independent (e.g., "Viral Infectivity" vs "Viral Replication").
- **Key Metric:** `mig.py` (Mutual Information Gap).
- **Action:** Copy the `compute_mig` function and run it on our latent vectors. If our features are entangled, this score will be low (bad).

## 4. RNA Velocity (`scvelo`)

- **Repo:** [theislab/scvelo](https://github.com/theislab/scvelo)
- **Why:** They predict "Future State" of cells.
- **Key Concept:** `velocity_graph`. They compute a vector field.
- **Relevance:** We can adapt this to compute "Evolutionary Velocity" for viruses. `velocity` = `fitness_gradient` in our hyperbolic space.
