# Manifold Observability Gap Analysis

**Doc-Type:** Technical Analysis · Version 1.0 · Updated 2025-12-12 · Author Claude Code

---

## Executive Summary

The current training infrastructure logs 50+ scalar metrics but provides **zero visualization of the actual manifold structure**. We track numbers that could be achieved by degenerate solutions while having no way to verify the embedding actually captures the 3^9 algebraic topology. This report identifies the gap and proposes structural visualizations that would demonstrate true geometric understanding.

---

## 1. The 3^9 Space Structure

### 1.1 Data Generation Reality

From `src/data/generation.py:12-30`:

```python
for i in range(3**9):  # 19,683 operations
    op = []
    num = i
    for _ in range(9):
        op.append(num % 3 - 1)  # Convert 0,1,2 to -1,0,1
        num //= 3
```

**Critical insight:** The index `i` IS the 3-adic representation. Operations sharing prefixes are algebraically related:

| Index Range | Shared Prefix | Count | Interpretation |
|-------------|---------------|-------|----------------|
| 0-2 | First digit | 3 | Same op(−1,−1) output |
| 0-8 | First 2 digits | 9 | Same op(−1,−1), op(−1,0) |
| 0-26 | First 3 digits | 27 | Same first row |
| 0-728 | First 6 digits | 729 | Same first two rows |

### 1.2 Ultrametric Tree Structure

The 19,683 operations form a **9-level ternary tree**:

```
                    ROOT (all ops)
                   /     |     \
            d₀=-1      d₀=0      d₀=1     (3 branches, 6561 each)
           /  |  \    /  |  \    /  |  \
        d₁∈{-1,0,1} for each     (9 branches, 2187 each)
              ...
        Level 9: individual operations (19683 leaves)
```

**3-adic distance:** `d(i,j) = 3^(-v)` where `v` = position of first differing digit.
- Adjacent in index ≠ close in 3-adic metric
- Index 0 and index 1 differ at digit 0 → distance = 1
- Index 0 and index 3 differ at digit 1 → distance = 1/3
- Index 0 and index 9 differ at digit 2 → distance = 1/9

---

## 2. Current Observability State

### 2.1 What We Log (TensorBoard)

From `src/training/monitor.py` and `src/training/hyperbolic_trainer.py`:

**Scalar Metrics (50+):**
```
Loss/Total, VAE_A/CrossEntropy, VAE_A/KL_Divergence, VAE_A/Entropy
VAE_A/Coverage_Pct, VAE_B/CrossEntropy, VAE_B/KL_Divergence
Hyperbolic/Correlation_Hyp, Hyperbolic/Correlation_Euc
Hyperbolic/MeanRadius, Hyperbolic/RankingWeight, Hyperbolic/RankingLoss
v5.10/HyperbolicKL, v5.10/CentroidLoss, v5.10/HomeostaticSigma
Dynamics/Phase, Dynamics/Rho, Dynamics/GradRatio
Lambdas/l1, Lambdas/l2, Lambdas/l3
Temperature/A, Temperature/B, Beta/A, Beta/B
```

**Histograms:**
```
Weights/{layer_name}, Gradients/{layer_name}
```

### 2.2 What We DON'T Log

**Zero `add_embedding()` calls** - The most powerful TensorBoard feature for manifold visualization is completely unused.

**Missing structural visualizations:**

| Visualization | Purpose | Current State |
|---------------|---------|---------------|
| Latent embeddings | See 19,683 points in space | NOT LOGGED |
| 3-adic prefix coloring | Verify hierarchical clustering | NOT LOGGED |
| Poincare disk projection | Confirm radial structure | NOT LOGGED |
| Distance matrix heatmaps | Compare 3-adic vs learned | NOT LOGGED |
| Frechet centroids | Verify tree node positions | NOT LOGGED |
| Geodesic fibers | See prefix-sharing connections | NOT LOGGED |
| Orbit structure | Algebraic symmetry groups | NOT LOGGED |

---

## 3. The Observability Gap

### 3.1 Scalar Metrics Can Lie

**Correlation = 0.95 could mean:**
- ✓ Perfect ultrametric embedding (what we want)
- ✗ Partial embedding with lucky triplet sampling
- ✗ Degenerate solution that clusters everything
- ✗ Overfitting to evaluation metric

**Coverage = 99% could mean:**
- ✓ True manifold coverage
- ✗ Mode collapse with noise
- ✗ Memorization without structure

**Without structural visualization, we cannot distinguish these cases.**

### 3.2 Missing Validation of Core Hypothesis

The project hypothesis: *Poincare ball naturally embeds 3-adic ultrametric structure*

To validate this, we need to SEE:
1. Operations with shared prefixes cluster together
2. Cluster hierarchy matches tree depth
3. Root operations near origin, leaves near boundary
4. Geodesic distances correlate with 3-adic distances

**None of this is currently visualized.**

---

## 4. Structural Elements to Visualize

### 4.1 Orbits (Algebraic Symmetry)

Operations related by permutation symmetry form orbits:

```python
# Example orbit: operations invariant under input swap
# op(a,b) = op(b,a) for all a,b
# These form a submanifold that should cluster in latent space

symmetric_ops = [i for i in range(19683)
                 if is_symmetric(index_to_op(i))]
# ~729 symmetric operations
```

**Orbits to track:**
- Symmetric operations (729)
- Projection operations (op(a,b) ∈ {a, b})
- Constant operations (3)
- Identity-like (op(a,a) = a)
- Idempotent (op(op(a,b), op(a,b)) = op(a,b))

### 4.2 Fibers (3-adic Prefix Clusters)

Operations sharing k-digit prefix form a fiber:

```python
# Level-1 fibers: 3 clusters of 6561 operations each
fiber_0 = range(0, 6561)      # d₀ = -1
fiber_1 = range(6561, 13122)  # d₀ = 0
fiber_2 = range(13122, 19683) # d₀ = 1

# Level-2 fibers: 9 clusters of 2187 operations each
# Level-3 fibers: 27 clusters of 729 operations each
# ...
# Level-9 fibers: 19683 clusters of 1 operation each
```

**Fiber visualization requirements:**
- Each fiber should have a **Frechet centroid** in Poincare ball
- Centroids should form hierarchical tree
- Intra-fiber variance < inter-fiber distance

### 4.3 Geodesic Structure

**Within-fiber geodesics:** Operations sharing prefix should connect via short geodesics near their centroid.

**Cross-fiber geodesics:** Operations with different prefixes should have geodesics passing through/near common ancestor centroid.

```
     C_root
    /      \
  C_0      C_1      (Level-1 centroids)
  / \      / \
C_00 C_01 C_10 C_11 (Level-2 centroids)

Geodesic from op in C_00 to op in C_11 should pass near C_root
```

### 4.4 Radial Distribution

**Hypothesis:** VAE-A explores boundary (mean_radius ~ 0.8), VAE-B anchors at origin (mean_radius ~ 0.4)

**But we need to verify:**
- Is the distribution uniform or clustered?
- Do different tree levels have different radii?
- Are fibers radially stratified?

---

## 5. Proposed Visualizations

### 5.1 TensorBoard Embeddings (Priority: CRITICAL)

```python
# In src/training/monitor.py or new src/visualization/tensorboard.py

def log_manifold_embedding(self, model, epoch, device):
    """Log latent embeddings with 3-adic metadata."""
    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, device=device)

    with torch.no_grad():
        outputs = model(x, 1.0, 1.0, 0.5, 0.5)
        z_A = project_to_poincare(outputs['z_A'])
        z_B = project_to_poincare(outputs['z_B'])

    # Metadata for coloring/filtering
    metadata = []
    for i in range(19683):
        prefix_1 = i % 3           # First digit
        prefix_2 = i % 9           # First 2 digits
        prefix_3 = i % 27          # First 3 digits
        valuation = compute_valuation_from_zero(i)
        metadata.append([i, prefix_1, prefix_2, prefix_3, valuation])

    # Log to TensorBoard
    self.writer.add_embedding(
        z_A,
        metadata=metadata,
        metadata_header=['index', 'prefix_1', 'prefix_2', 'prefix_3', 'valuation'],
        global_step=epoch,
        tag='VAE_A_Poincare'
    )
```

### 5.2 Poincare Disk Projection (Priority: HIGH)

```python
def log_poincare_disk(self, z, epoch, tag):
    """Log 2D Poincare disk projection as image."""
    # Project 16D -> 2D using hyperbolic PCA or geodesic projection
    z_2d = hyperbolic_pca(z, n_components=2)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw unit circle (Poincare boundary)
    circle = plt.Circle((0, 0), 1, fill=False, color='black')
    ax.add_patch(circle)

    # Color by 3-adic prefix
    colors = [i % 27 for i in range(len(z_2d))]  # 27 colors for level-3
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=colors, cmap='tab20', s=1, alpha=0.5)

    # Mark Frechet centroids
    for level in [1, 2, 3]:
        centroids = compute_frechet_centroids(z, level)
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   c='red', s=100, marker='*', edgecolors='black')

    self.writer.add_figure(f'Poincare/{tag}', fig, epoch)
```

### 5.3 Distance Matrix Comparison (Priority: HIGH)

```python
def log_distance_matrices(self, z, epoch):
    """Compare 3-adic vs Poincare distance matrices."""
    n_sample = 500  # Sample for visualization
    indices = np.random.choice(19683, n_sample, replace=False)

    # 3-adic distance matrix
    adic_dist = np.zeros((n_sample, n_sample))
    for i, idx_i in enumerate(indices):
        for j, idx_j in enumerate(indices):
            adic_dist[i, j] = compute_3adic_distance(idx_i, idx_j)

    # Poincare distance matrix
    z_sample = z[indices]
    poincare_dist = pairwise_poincare_distance(z_sample)

    # Log as images
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(adic_dist, cmap='viridis')
    axes[0].set_title('3-adic Distance')
    axes[1].imshow(poincare_dist, cmap='viridis')
    axes[1].set_title('Poincare Distance')
    axes[2].imshow(np.abs(adic_dist - poincare_dist), cmap='hot')
    axes[2].set_title('|Difference|')

    self.writer.add_figure('DistanceMatrix/Comparison', fig, epoch)
```

### 5.4 Fiber Coherence Visualization (Priority: MEDIUM)

```python
def log_fiber_coherence(self, z, epoch):
    """Visualize fiber (prefix cluster) coherence."""
    coherence_by_level = {}

    for level in range(1, 5):  # Levels 1-4
        n_fibers = 3 ** level
        intra_distances = []
        inter_distances = []

        for fiber_id in range(n_fibers):
            # Get operations in this fiber
            fiber_ops = get_fiber_operations(fiber_id, level)
            z_fiber = z[fiber_ops]

            # Intra-fiber: distances within fiber
            intra = pairwise_poincare_distance(z_fiber).mean()
            intra_distances.append(intra)

            # Inter-fiber: distance to other fibers
            other_ops = get_other_fiber_operations(fiber_id, level)
            z_other = z[other_ops[:100]]  # Sample
            inter = poincare_distance(z_fiber.mean(0), z_other).mean()
            inter_distances.append(inter)

        coherence = np.mean(inter_distances) / np.mean(intra_distances)
        coherence_by_level[level] = coherence

    self.writer.add_scalars('Fiber/Coherence', coherence_by_level, epoch)
```

### 5.5 Centroid Tree Visualization (Priority: MEDIUM)

```python
def log_centroid_tree(self, z, epoch):
    """Visualize Frechet centroid hierarchy."""
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Compute centroids at each level
    for level in range(4):
        centroids = compute_frechet_centroids(z, level)

        # Plot centroids
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                   s=100 / (level + 1), alpha=0.8, label=f'Level {level}')

        # Draw edges to parent centroids
        if level > 0:
            parent_centroids = compute_frechet_centroids(z, level - 1)
            for i, c in enumerate(centroids):
                parent_idx = i // 3
                ax.plot([c[0], parent_centroids[parent_idx, 0]],
                       [c[1], parent_centroids[parent_idx, 1]],
                       [c[2], parent_centroids[parent_idx, 2]],
                       'k-', alpha=0.3)

    ax.set_title('Frechet Centroid Tree (3-adic hierarchy)')
    self.writer.add_figure('Structure/CentroidTree', fig, epoch)
```

---

## 6. Implementation Roadmap

### Phase 1: Critical (Immediate)

| Task | File | Lines | Impact |
|------|------|-------|--------|
| Add `add_embedding()` for VAE-A/B | `src/training/monitor.py` | ~50 | See actual manifold |
| Add 3-adic prefix metadata | `src/data/generation.py` | ~30 | Color by structure |
| Log every N epochs | `src/training/hyperbolic_trainer.py` | ~10 | Regular snapshots |

### Phase 2: High Priority

| Task | File | Lines | Impact |
|------|------|-------|--------|
| Create `src/visualization/` module | New module | ~300 | Centralized viz |
| Poincare disk 2D projection | `src/visualization/poincare.py` | ~100 | See boundary/origin |
| Distance matrix comparison | `src/visualization/metrics.py` | ~80 | Verify isometry |

### Phase 3: Medium Priority

| Task | File | Lines | Impact |
|------|------|-------|--------|
| Fiber coherence metrics | `src/visualization/structure.py` | ~100 | Verify clustering |
| Centroid tree visualization | `src/visualization/structure.py` | ~100 | Verify hierarchy |
| Orbit highlighting | `src/visualization/algebra.py` | ~80 | Show symmetry |

---

## 7. Success Criteria

After implementing these visualizations, we should be able to answer:

| Question | Visualization | Expected Result |
|----------|---------------|-----------------|
| Do prefixes cluster? | Embedding + prefix coloring | Distinct color regions |
| Is hierarchy radial? | Poincare disk | Root at center, leaves at edge |
| Do centroids form tree? | Centroid tree | Nested hierarchy |
| Is isometry achieved? | Distance matrices | High correlation |
| Do orbits cluster? | Embedding + orbit metadata | Algebraic submanifolds |

---

## 8. Conclusion

The current observability infrastructure optimizes for scalar metrics that can be gamed. A correlation of 0.95 means nothing without seeing the actual embedding structure. By adding TensorBoard embeddings with 3-adic metadata, Poincare disk projections, and distance matrix comparisons, we can validate (or invalidate) the core hypothesis that hyperbolic geometry naturally captures ultrametric structure.

**The gap is not in our training - it's in our ability to see what we've trained.**

---

## Appendix: Key Files to Modify

```
src/
├── data/
│   └── generation.py          # Add prefix computation
├── training/
│   ├── monitor.py             # Add embedding logging
│   └── hyperbolic_trainer.py  # Call embedding logging
├── metrics/
│   └── hyperbolic.py          # Add pairwise distance functions
└── visualization/             # NEW MODULE
    ├── __init__.py
    ├── embeddings.py          # TensorBoard embedding utilities
    ├── poincare.py            # Poincare disk projections
    ├── structure.py           # Fiber/centroid visualization
    └── algebra.py             # Orbit/symmetry visualization
```
