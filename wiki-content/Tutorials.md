# Tutorials

Step-by-step guides for common tasks with Ternary VAE.

---

## Tutorial 1: Your First Ternary VAE

Learn the basics by training a simple model.

### Goal
Train a VAE on synthetic ternary data and visualize the latent space.

### Duration
~15 minutes

### Prerequisites
- Python environment with dependencies installed
- Basic PyTorch knowledge

### Step 1: Import Libraries

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from src.models import TernaryVAE
from src.config import TrainingConfig
from src.losses import create_registry_from_training_config
from src.geometry import RiemannianAdam
```

### Step 2: Create Synthetic Data

We'll create data with known structure to verify learning:

```python
# Create 5 clusters in "ternary space"
n_per_cluster = 200
n_clusters = 5

# Each cluster has a base operation
cluster_bases = [0, 1000, 5000, 10000, 15000]

data = []
labels = []

for i, base in enumerate(cluster_bases):
    # Add noise around base
    ops = base + np.random.randint(0, 100, n_per_cluster)
    ops = np.clip(ops, 0, 19682)
    data.extend(ops)
    labels.extend([i] * n_per_cluster)

data = torch.tensor(data, dtype=torch.long)
labels = torch.tensor(labels, dtype=torch.long)

# One-hot encode
data_onehot = torch.zeros(len(data), 19683)
data_onehot.scatter_(1, data.unsqueeze(1), 1)

print(f"Dataset: {len(data)} samples, {n_clusters} clusters")
```

### Step 3: Create DataLoader

```python
dataset = TensorDataset(data_onehot, data, labels)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
```

### Step 4: Initialize Model

```python
config = TrainingConfig(
    epochs=100,
    batch_size=64,
    geometry={"curvature": 1.0, "latent_dim": 2},  # 2D for visualization
    loss_weights={"reconstruction": 1.0, "kl_divergence": 0.1},
)

model = TernaryVAE(
    input_dim=19683,
    latent_dim=2,
    curvature=1.0,
)

optimizer = RiemannianAdam(model.parameters(), lr=1e-3)
loss_registry = create_registry_from_training_config(config)
```

### Step 5: Training Loop

```python
history = {"loss": [], "recon": [], "kl": []}

for epoch in range(config.epochs):
    epoch_loss = 0
    epoch_recon = 0
    epoch_kl = 0

    for batch_x, batch_y, _ in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_x)
        result = loss_registry.compose(outputs, batch_y)

        result.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += result.total.item()

    # Log metrics
    history["loss"].append(epoch_loss / len(train_loader))

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: Loss = {history['loss'][-1]:.4f}")
```

### Step 6: Visualize Latent Space

```python
model.eval()
all_z = []
all_labels = []

with torch.no_grad():
    for batch_x, _, batch_labels in train_loader:
        outputs = model(batch_x)
        all_z.append(outputs["z_hyperbolic"])
        all_labels.append(batch_labels)

z = torch.cat(all_z).numpy()
labels = torch.cat(all_labels).numpy()

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Latent space with clusters
ax = axes[0]
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
ax.add_patch(circle)
scatter = ax.scatter(z[:, 0], z[:, 1], c=labels, cmap='tab10', alpha=0.6, s=20)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')
ax.set_title('Latent Space (Poincare Ball)')
ax.legend(*scatter.legend_elements(), title="Cluster")

# Right: Training curve
axes[1].plot(history["loss"])
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].set_title("Training Loss")

plt.tight_layout()
plt.savefig("tutorial1_results.png", dpi=150)
plt.show()
```

### Expected Result

You should see:
- 5 distinct clusters in the Poincare ball
- Points stay within the unit circle (boundary)
- Training loss decreases smoothly

### What You Learned

- How to create and train a TernaryVAE
- Hyperbolic latent space visualization
- Using loss registry for flexible training

---

## Tutorial 2: Codon Optimization

Optimize codons for improved expression.

### Goal
Use Ternary VAE to find optimal synonymous codon substitutions.

### Step 1: Setup

```python
from src.models import TernaryVAE
from src.losses import AutoimmuneCodonRegularizer, HUMAN_CODON_RSCU
from src.config import TrainingConfig
import torch
```

### Step 2: Define Codon Tables

```python
# Amino acid to codon mapping
CODON_TABLE = {
    'A': ['GCU', 'GCC', 'GCA', 'GCG'],  # Alanine
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],  # Arginine
    'N': ['AAU', 'AAC'],  # Asparagine
    # ... (complete table)
    '*': ['UAA', 'UAG', 'UGA'],  # Stop
}

# Human codon preferences (Relative Synonymous Codon Usage)
HUMAN_RSCU = {
    'GCU': 0.26, 'GCC': 0.40, 'GCA': 0.23, 'GCG': 0.11,  # Ala
    # ... from src.losses.autoimmunity
}
```

### Step 3: Load Pre-trained Model

```python
# Load model trained on human sequences
model = TernaryVAE.load("checkpoints/human_codons.pt")
model.eval()
```

### Step 4: Encode Original Sequence

```python
def sequence_to_operations(codons):
    """Convert codon sequence to ternary operations."""
    # Implementation depends on encoding scheme
    pass

original_sequence = "AUGCCUGAA"  # Met-Pro-Glu
operations = sequence_to_operations(original_sequence)

# Encode in latent space
with torch.no_grad():
    x = one_hot_encode(operations)
    outputs = model(x)
    z_original = outputs["z_hyperbolic"]
```

### Step 5: Find Optimal Synonymous Variants

```python
def generate_synonymous_variants(sequence):
    """Generate all synonymous codon substitutions."""
    variants = []
    codons = [sequence[i:i+3] for i in range(0, len(sequence), 3)]

    for i, codon in enumerate(codons):
        aa = translate(codon)
        for alt_codon in CODON_TABLE[aa]:
            if alt_codon != codon:
                new_seq = codons.copy()
                new_seq[i] = alt_codon
                variants.append(''.join(new_seq))

    return variants

variants = generate_synonymous_variants(original_sequence)

# Score each variant
scores = []
for variant in variants:
    ops = sequence_to_operations(variant)
    x = one_hot_encode(ops)

    with torch.no_grad():
        outputs = model(x)
        z = outputs["z_hyperbolic"]

    # Score based on human codon preference
    rscu_score = sum(HUMAN_RSCU.get(c, 0) for c in variant)
    scores.append((variant, rscu_score))

# Sort by score
best_variants = sorted(scores, key=lambda x: -x[1])[:5]
print("Top 5 synonymous variants:")
for seq, score in best_variants:
    print(f"  {seq}: {score:.2f}")
```

---

## Tutorial 3: Visualizing Hyperbolic Geometry

Understand the Poincare ball through interactive visualization.

### Step 1: Create Visualization Functions

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from src.geometry import poincare_distance, exp_map_zero, mobius_add

def plot_poincare_ball(ax, title="Poincare Ball"):
    """Setup a Poincare ball plot."""
    circle = Circle((0, 0), 1, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
```

### Step 2: Visualize Exponential Map

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Euclidean vectors
ax = axes[0]
vectors = [
    torch.tensor([0.5, 0.0]),
    torch.tensor([0.0, 0.5]),
    torch.tensor([0.5, 0.5]),
    torch.tensor([1.0, 0.0]),
    torch.tensor([1.5, 0.0]),
]

for v in vectors:
    ax.arrow(0, 0, v[0], v[1], head_width=0.05, color='blue', alpha=0.7)
ax.set_xlim(-0.5, 2)
ax.set_ylim(-0.5, 1)
ax.set_title("Euclidean Vectors (Tangent Space)")
ax.set_aspect('equal')

# Right: Projected to Poincare ball
ax = axes[1]
plot_poincare_ball(ax, "After exp_map_zero")

for v in vectors:
    p = exp_map_zero(v, curvature=1.0)
    ax.plot(p[0], p[1], 'o', markersize=8)
    ax.annotate(f"||v||={v.norm():.1f}", (p[0]+0.05, p[1]+0.05))

plt.tight_layout()
plt.savefig("exp_map_visualization.png", dpi=150)
```

### Step 3: Visualize Geodesics

```python
def plot_geodesic(ax, p1, p2, n_points=50, **kwargs):
    """Plot geodesic between two points."""
    # Geodesics in Poincare ball are circular arcs
    t = np.linspace(0, 1, n_points)
    points = []

    for ti in t:
        # Interpolate using Mobius operations
        p = mobius_add(-p1, p2, curvature=1.0) * ti
        p = mobius_add(p1, p, curvature=1.0)
        points.append(p.numpy())

    points = np.array(points)
    ax.plot(points[:, 0], points[:, 1], **kwargs)

fig, ax = plt.subplots(figsize=(8, 8))
plot_poincare_ball(ax, "Geodesics in Poincare Ball")

# Plot several geodesics
p1 = torch.tensor([-0.5, 0.0])
for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
    p2 = torch.tensor([0.7 * np.cos(angle), 0.7 * np.sin(angle)])
    plot_geodesic(ax, p1, p2, color='blue', alpha=0.5)

ax.plot(p1[0], p1[1], 'ro', markersize=10, label='Origin')
plt.legend()
plt.savefig("geodesics.png", dpi=150)
```

### Step 4: Visualize Distance Distortion

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Points at different radii
radii = [0.1, 0.3, 0.5, 0.7, 0.9]
origin = torch.tensor([0.0, 0.0])

euclidean_dists = []
hyperbolic_dists = []

for r in radii:
    p = torch.tensor([r, 0.0])
    euclidean_dists.append(r)
    hyperbolic_dists.append(poincare_distance(origin, p, curvature=1.0).item())

# Plot comparison
axes[0].plot(radii, euclidean_dists, 'b-o', label='Euclidean')
axes[0].plot(radii, hyperbolic_dists, 'r-o', label='Hyperbolic')
axes[0].set_xlabel('Radius in ball')
axes[0].set_ylabel('Distance from origin')
axes[0].legend()
axes[0].set_title('Distance Comparison')

# Plot distance ratio
axes[1].plot(radii, np.array(hyperbolic_dists) / np.array(euclidean_dists))
axes[1].set_xlabel('Radius in ball')
axes[1].set_ylabel('Hyperbolic / Euclidean ratio')
axes[1].set_title('Distance Ratio (grows near boundary)')

plt.tight_layout()
plt.savefig("distance_distortion.png", dpi=150)
```

---

## Tutorial 4: Custom Loss Functions

Create your own loss component for specialized applications.

### Step 1: Understand the Protocol

```python
from src.losses import LossComponent, LossResult
import torch

class MyLoss(LossComponent):
    """Custom loss component."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def compute(self, outputs: dict, targets: torch.Tensor) -> LossResult:
        """
        Args:
            outputs: Model outputs dict with keys:
                - "reconstruction": (B, 19683) logits
                - "z_hyperbolic": (B, latent_dim) hyperbolic coords
                - "mu": (B, latent_dim) mean
                - "logvar": (B, latent_dim) log variance
            targets: (B,) target indices

        Returns:
            LossResult with total, components, metrics
        """
        loss = ...  # Your computation
        return LossResult(
            total=self.weight * loss,
            components={"my_loss": loss},
            metrics={"my_metric": some_value},
        )
```

### Step 2: Example - Diversity Loss

Encourage diverse latent representations:

```python
class DiversityLoss(LossComponent):
    """Penalize similar latent representations within a batch."""

    def __init__(self, weight: float = 0.1, min_distance: float = 0.1):
        self.weight = weight
        self.min_distance = min_distance

    def compute(self, outputs: dict, targets: torch.Tensor) -> LossResult:
        z = outputs["z_hyperbolic"]  # (B, D)

        # Compute pairwise distances
        from src.geometry import poincare_distance_matrix
        distances = poincare_distance_matrix(z, z, curvature=1.0)

        # Penalize distances below threshold
        mask = distances < self.min_distance
        mask.fill_diagonal_(False)  # Ignore self-distance

        penalty = (self.min_distance - distances[mask]).sum()
        loss = penalty / (mask.sum() + 1e-8)

        return LossResult(
            total=self.weight * loss,
            components={"diversity": loss},
            metrics={
                "min_pairwise_dist": distances[~torch.eye(len(z), dtype=bool)].min().item(),
                "pairs_too_close": mask.sum().item(),
            },
        )
```

### Step 3: Register and Use

```python
from src.losses import LossRegistry, ReconstructionLossComponent

registry = LossRegistry()
registry.register("recon", ReconstructionLossComponent(weight=1.0))
registry.register("diversity", DiversityLoss(weight=0.1, min_distance=0.2))

# Use in training
result = registry.compose(outputs, targets)
print(result.metrics["min_pairwise_dist"])
```

---

## More Tutorials

Coming soon:
- Tutorial 5: Transfer Learning
- Tutorial 6: Multi-GPU Training
- Tutorial 7: Hyperparameter Tuning
- Tutorial 8: Integration with ESM/AlphaFold

---

*See also: [[Quick-Start]], [[Training]], [[Loss-Functions]]*
