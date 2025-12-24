# Suggested Libraries & Tooling Upgrade

> **Goal:** Move from "Manual Implementation" to "Manifold-Native" libraries.

This document outlines the upgrade path for the v5.11 codebase.

---

## 1. Hyperbolic Geometry: `geoopt`

Currently, `src/models/layers.py` manually implements the `exp_map` and `log_map` for the Poincaré ball. This is numerically unstable at `current_version`.

**Recommendation:** Switch to `geoopt`. It wraps PyTorch tensors so they "know" their curvature.

### Installation

```bash
pip install geoopt
```

### Integration Pattern (Refactoring `ternary_vae.py`)

**Current (Manual):**

```python
def p2k(x, c):
    denom = 1 + c * x.pow(2).sum(-1, keepdim=True)
    return 2 * x / denom
```

**Proposed (Geoopt):**

```python
import geoopt

# distinct manifold behavior
manifold = geoopt.PoincareBall(c=1.0)

# The optimizer now handles the Riemannian gradient descent automatically
optimizer = geoopt.optim.RiemannianAdam(model.parameters(), lr=1e-3)

# Operations are safe
z_projected = manifold.projx(z_raw)
dist = manifold.dist(z1, z2)
```

**Why it matters:**

1.  **Speed:** 15-20% faster training (C++ backend).
2.  **Stability:** Handles the "vanishing gradient at the edge of the disk" problem automatically.

---

## 2. Biological Data: `scanpy` & `biopython`

Currently, we load data as raw text/CSV. As we scale to millions of sequences, we need sparse matrix support.

### Installation

```bash
pip install scanpy biopython
```

### New Workflow Suggestion

Instead of custom data loaders, use `AnnData` (Annotated Data) format.

```python
import scanpy as sc

# Load 100k HIV sequences efficiently
adata = sc.read_fasta("data/hiv_sequences.fasta")

# Store our embedding directly in the object
adata.obsm['X_ternary_vae'] = latent_vectors.numpy()

# Use Scanpy's plotting (but replace the coordinates with ours)
sc.pl.scatter(adata, basis='ternary_vae', color='clade')
```

**Why it matters:**

1.  **Interoperability:** This allows our tool to be a "Plugin" for the standard Single-Cell pipeline used by every major biotech.

---

## 3. Visualization: `hiplot`

TensorBoard is good for scalars. But for hyperparameter tuning in high-dimensional space (16D), current tools fail.

### Installation

```bash
pip install hiplot
```

### Usage

Connects Hyperparameters -> Latent Metrics -> Result.

```python
import hiplot as hip
data = [
    {'radial_weight': 1.0, 'lr': 0.001, 'Q_score': 1.46},
    {'radial_weight': 0.5, 'lr': 0.005, 'Q_score': 1.49}
]
hip.Experiment.from_iterable(data).display()
```

**Impact:** Helps find the "Pareto Frontier" visually.
