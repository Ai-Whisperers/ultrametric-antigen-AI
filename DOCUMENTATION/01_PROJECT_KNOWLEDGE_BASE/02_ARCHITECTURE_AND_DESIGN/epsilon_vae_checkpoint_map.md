# Epsilon-VAE: A Map of Checkpoint Behavior

**Doc-Type:** Technical Architecture · Version 1.0 · Updated 2025-12-26 · Author AI Whisperers

---

## Overview

Epsilon-VAE is a meta-learning model that creates a **compressed, navigable map of checkpoint behavior**. Rather than treating checkpoints as opaque weight files, Epsilon-VAE learns to predict what embedding geometry each checkpoint produces, enabling exploration, comparison, and discovery without running expensive inference.

**Core insight**: A checkpoint's weights encode its "personality" - the geometric structure it will impose on any input data. Epsilon-VAE learns this mapping, creating a latent space where similar checkpoints cluster together and the path between any two checkpoints reveals intermediate configurations.

---

## Architecture

```
                         EPSILON-VAE ARCHITECTURE

checkpoint_weights ──► [Weight Encoder] ──► latent z (64-dim)
    (95,808 params)         │                    │
                            │     ┌──────────────┴──────────────┐
                            │     │                             │
                            │     ▼                             ▼
                            │  [Embedding Decoder]    [Metric Predictor]
                            │     │                             │
                            │     ▼                             ▼
                            │  Predicted embeddings      3 scalar metrics
                            │  (256 anchors × 16 dim)   (coverage, dist, rad)
                            │     │                             │
                            │     └──────────────┬──────────────┘
                            │                    │
                            ▼                    ▼
                    THE CHECKPOINT MAP     INTERPRETABLE SUMMARIES
```

**dimensions**:
- Input: 95,808 weight parameters (flattened projection + encoder weights)
- Latent: 64-dimensional checkpoint representation
- Embedding output: 256 × 16 = 4,096 values (anchor embeddings)
- Metric output: 3 values (coverage, distance correlation, radial hierarchy)

---

## The Checkpoint Map Concept

### What the Latent Space Represents

Each point z in the 64-dimensional latent space represents a **checkpoint personality** - a compressed encoding of how that checkpoint transforms inputs into embeddings. Checkpoints that produce similar embedding geometries will have similar z vectors.

**latent_space_properties**:
- **Continuity**: Nearby points produce similar embedding geometries
- **Interpolability**: Paths between points reveal intermediate configurations
- **Completeness**: The space covers the full range of observed checkpoint behaviors
- **Generalization**: Novel points can be sampled to discover unexplored configurations

### Why Embedding Space, Not Just Metrics

Previous approaches trained on 3 scalar metrics (coverage, distance correlation, radial hierarchy). This failed because:

**metrics_only_problems**:
- **Information loss**: 3 numbers cannot capture 4,096-dimensional geometry
- **Mode collapse**: Model learned to predict mean values for all inputs
- **No structure**: Lost the rich geometric relationships between embeddings

**hybrid_approach_benefits**:
- **Full geometry**: Reconstructs actual 256 × 16 embedding space
- **No collapse**: Must predict varied, structured outputs
- **Interpretable**: Metrics still available as auxiliary outputs
- **Validated**: 0.796 cosine similarity on held-out checkpoints

---

## Capabilities and Use Cases

### 1. Checkpoint Quality Prediction

**purpose**: Estimate embedding quality without running inference

**method**:
```python
# Load trained Epsilon-VAE
model = load_epsilon_vae("epsilon_vae_hybrid_models/best.pt")

# Extract weights from new checkpoint
weights = extract_key_weights(new_checkpoint)

# Predict metrics instantly
z = model.encode(weights)
predicted_metrics = model.predict_metrics(z)
# Returns: coverage, distance_corr, radial_hier
```

**applications**:
- Early stopping: Predict final quality from intermediate checkpoints
- Hyperparameter screening: Quickly evaluate many configurations
- Training monitoring: Track predicted quality during training

---

### 2. Checkpoint Interpolation

**purpose**: Discover intermediate configurations between known checkpoints

**method**:
```python
# Encode two checkpoints
z_frozen = model.encode(weights_frozen_encoder)
z_unfrozen = model.encode(weights_unfrozen_encoder)

# Interpolate in latent space
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    z_interp = (1 - alpha) * z_frozen + alpha * z_unfrozen

    # Predict what this intermediate would produce
    embeddings = model.decode_embeddings(z_interp)
    metrics = model.predict_metrics(z_interp)

    print(f"alpha={alpha}: coverage={metrics[0]:.3f}, dist_corr={metrics[1]:.3f}")
```

**applications**:
- Find optimal tradeoff points between different training strategies
- Understand transition dynamics between frozen/unfrozen regimes
- Generate "virtual checkpoints" that blend multiple approaches

---

### 3. Pareto Frontier Discovery

**purpose**: Find checkpoints that optimally balance multiple objectives

**method**:
```python
# Sample latent space densely
z_samples = torch.randn(10000, 64)

# Predict metrics for all samples
metrics = model.predict_metrics(z_samples)

# Find Pareto-optimal points (maximize coverage AND dist_corr)
pareto_mask = find_pareto_efficient(-metrics[:, :2])  # Negate to maximize
pareto_z = z_samples[pareto_mask]
pareto_metrics = metrics[pareto_mask]

# These represent theoretically optimal checkpoints
```

**applications**:
- Discover unexplored high-performance regions
- Guide hyperparameter search toward Pareto frontier
- Understand fundamental tradeoffs in the model family

---

### 4. Checkpoint Clustering and Similarity

**purpose**: Group checkpoints by embedding behavior, not surface-level hyperparameters

**method**:
```python
# Encode all historical checkpoints
z_all = [model.encode(weights) for weights in all_checkpoints]
z_matrix = torch.stack(z_all)

# Cluster in latent space
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=5).fit_predict(z_matrix)

# Checkpoints in same cluster produce similar embeddings
# regardless of their hyperparameter differences
```

**applications**:
- Identify redundant training runs (same cluster = similar outcome)
- Find surprising similarities between different approaches
- Organize checkpoint library by behavior, not chronology

---

### 5. Embedding Space Visualization

**purpose**: Understand what geometric structure a checkpoint will produce

**method**:
```python
# Predict full embedding space for a checkpoint
z = model.encode(weights)
anchor_embeddings = model.decode_embeddings(z)  # (256, 16)

# Visualize predicted embeddings
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
embeddings_2d = tsne.fit_transform(anchor_embeddings)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.title("Predicted Embedding Structure")
```

**applications**:
- Preview embedding geometry before committing to training
- Compare predicted vs actual embeddings for validation
- Debug unexpected checkpoint behaviors

---

### 6. Latent Space Sampling for Discovery

**purpose**: Generate hypothetical checkpoints with desired properties

**method**:
```python
# Find region of latent space with high coverage + high dist_corr
# (identified from Pareto analysis)
target_region_mean = pareto_z.mean(dim=0)
target_region_std = pareto_z.std(dim=0)

# Sample new points in this region
z_new = target_region_mean + 0.5 * target_region_std * torch.randn(100, 64)

# Predict what these hypothetical checkpoints would produce
for z in z_new:
    metrics = model.predict_metrics(z)
    embeddings = model.decode_embeddings(z)

    if metrics[0] > 0.95 and metrics[1] > 0.8:  # High coverage + dist_corr
        print(f"Promising configuration found: {metrics}")
        # Analyze embedding structure to understand what makes it good
```

**applications**:
- Guided exploration of checkpoint space
- Generate hypotheses about optimal configurations
- Identify underexplored regions worth investigating

---

### 7. Training Trajectory Analysis

**purpose**: Understand how checkpoints evolve during training

**method**:
```python
# Encode checkpoints from a single training run
trajectory = []
for epoch in [0, 10, 20, 50, 100]:
    weights = load_checkpoint(f"run/epoch_{epoch}.pt")
    z = model.encode(weights)
    trajectory.append(z)

# Analyze trajectory in latent space
trajectory = torch.stack(trajectory)

# Compute trajectory length (how much the model changed)
trajectory_length = sum(
    (trajectory[i+1] - trajectory[i]).norm()
    for i in range(len(trajectory)-1)
)

# Compute trajectory direction (where is training heading)
overall_direction = trajectory[-1] - trajectory[0]
```

**applications**:
- Detect training stagnation (short trajectory = not learning)
- Predict final outcome from early trajectory direction
- Compare training dynamics across different hyperparameter settings

---

## Validation Results

**training_data**:
- 825 training checkpoints from 48 runs
- 77 validation checkpoints from held-out runs (temporal split)
- 256 anchor ternary operations per checkpoint
- 16-dimensional hyperbolic embeddings

**performance**:

| Metric | Old (metrics-only) | New (hybrid) | Improvement |
|:-------|:-------------------|:-------------|:------------|
| Coverage MAE | 0.366 | 0.089 | 4.1× better |
| Dist Corr MAE | 0.218 | 0.126 | 1.7× better |
| Rad Hier MAE | 0.085 | 0.050 | 1.7× better |
| Embedding Cosine Sim | N/A | 0.796 | New capability |

---

## Files and Locations

**scripts**:
- `scripts/epsilon_vae/extract_embeddings.py` - Extract anchor embeddings from checkpoints
- `scripts/epsilon_vae/train_epsilon_vae_hybrid.py` - Train hybrid model

**models**:
- `sandbox-training/epsilon_vae_hybrid_models/best.pt` - Best trained model
- `sandbox-training/epsilon_vae_hybrid_models/final.pt` - Final epoch model

**data**:
- `sandbox-training/epsilon_vae_hybrid/config.json` - Dataset configuration
- `sandbox-training/epsilon_vae_hybrid/anchor_operations.npy` - Fixed anchor set
- `sandbox-training/epsilon_vae_hybrid/train_*.npy` - Training arrays
- `sandbox-training/epsilon_vae_hybrid/val_*.npy` - Validation arrays

---

## Future Directions

**immediate_extensions**:
- Add weight decoder to generate actual checkpoint weights from latent z
- Implement gradient-based optimization in latent space toward target metrics
- Build interactive visualization tool for latent space exploration

**research_directions**:
- Conditional generation: "Give me a checkpoint with coverage > 0.95"
- Transfer learning: Adapt to new model architectures
- Temporal modeling: Predict future checkpoints from training history

**integration_opportunities**:
- Hyperparameter optimization: Use latent space as search space
- Neural architecture search: Extend to architecture variations
- Continual learning: Update map as new checkpoints are created

---

## Summary

Epsilon-VAE transforms checkpoint analysis from "run inference and measure" to "encode and predict." The 64-dimensional latent space serves as a navigable map where:

- **Distance** reflects embedding similarity
- **Directions** correspond to changes in embedding geometry
- **Regions** cluster checkpoints with similar behaviors
- **Sampling** discovers unexplored configurations

This map enables rapid exploration, comparison, and discovery - turning the checkpoint space from an opaque collection of weight files into a structured, interpretable landscape.

---

**Version:** 1.0 · **Status:** Active · **Maintainer:** AI Whisperers
