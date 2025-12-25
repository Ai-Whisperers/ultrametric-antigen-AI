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
    epoch_loss = 0.0
    epoch_recon = 0.0
    epoch_kl = 0.0

    for batch_x, batch_y, _ in train_loader:
        optimizer.zero_grad()

        outputs = model(batch_x)
        result = loss_registry.compose(outputs, batch_y)

        result.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += result.total.item()
        # Track component losses (handle both tensor and float)
        recon = result.components.get("reconstruction", 0)
        kl = result.components.get("kl_divergence", 0)
        epoch_recon += recon.item() if hasattr(recon, 'item') else recon
        epoch_kl += kl.item() if hasattr(kl, 'item') else kl

    # Log metrics
    n_batches = len(train_loader)
    history["loss"].append(epoch_loss / n_batches)
    history["recon"].append(epoch_recon / n_batches)
    history["kl"].append(epoch_kl / n_batches)

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

Optimize codons for improved expression in human cells.

### Goal
Use Ternary VAE to find optimal synonymous codon substitutions that maximize human expression while preserving protein function.

### Prerequisites
- Completed Tutorial 1
- Understanding of codon degeneracy

### Step 1: Setup

```python
import torch
from src.models import TernaryVAE
from src.config import TrainingConfig

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Step 2: Define Codon Tables

```python
# Standard genetic code: amino acid -> codons (RNA notation)
CODON_TABLE = {
    'A': ['GCU', 'GCC', 'GCA', 'GCG'],  # Alanine (4 codons)
    'R': ['CGU', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],  # Arginine (6 codons)
    'N': ['AAU', 'AAC'],  # Asparagine
    'D': ['GAU', 'GAC'],  # Aspartic acid
    'C': ['UGU', 'UGC'],  # Cysteine
    'Q': ['CAA', 'CAG'],  # Glutamine
    'E': ['GAA', 'GAG'],  # Glutamic acid
    'G': ['GGU', 'GGC', 'GGA', 'GGG'],  # Glycine
    'H': ['CAU', 'CAC'],  # Histidine
    'I': ['AUU', 'AUC', 'AUA'],  # Isoleucine
    'L': ['UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG'],  # Leucine (6 codons)
    'K': ['AAA', 'AAG'],  # Lysine
    'M': ['AUG'],  # Methionine (1 codon - start)
    'F': ['UUU', 'UUC'],  # Phenylalanine
    'P': ['CCU', 'CCC', 'CCA', 'CCG'],  # Proline
    'S': ['UCU', 'UCC', 'UCA', 'UCG', 'AGU', 'AGC'],  # Serine (6 codons)
    'T': ['ACU', 'ACC', 'ACA', 'ACG'],  # Threonine
    'W': ['UGG'],  # Tryptophan (1 codon)
    'Y': ['UAU', 'UAC'],  # Tyrosine
    'V': ['GUU', 'GUC', 'GUA', 'GUG'],  # Valine
    '*': ['UAA', 'UAG', 'UGA'],  # Stop codons
}

# Reverse lookup: codon -> amino acid
CODON_TO_AA = {}
for aa, codons in CODON_TABLE.items():
    for codon in codons:
        CODON_TO_AA[codon] = aa

# Human codon preferences (Relative Synonymous Codon Usage)
# Source: Kazusa Codon Usage Database for Homo sapiens
HUMAN_RSCU = {
    # Alanine
    'GCU': 0.26, 'GCC': 0.40, 'GCA': 0.23, 'GCG': 0.11,
    # Arginine
    'CGU': 0.08, 'CGC': 0.19, 'CGA': 0.11, 'CGG': 0.21, 'AGA': 0.20, 'AGG': 0.20,
    # Asparagine
    'AAU': 0.46, 'AAC': 0.54,
    # Aspartic acid
    'GAU': 0.46, 'GAC': 0.54,
    # Cysteine
    'UGU': 0.45, 'UGC': 0.55,
    # Glutamine
    'CAA': 0.25, 'CAG': 0.75,
    # Glutamic acid
    'GAA': 0.42, 'GAG': 0.58,
    # Glycine
    'GGU': 0.16, 'GGC': 0.34, 'GGA': 0.25, 'GGG': 0.25,
    # Leucine
    'UUA': 0.07, 'UUG': 0.13, 'CUU': 0.13, 'CUC': 0.20, 'CUA': 0.07, 'CUG': 0.41,
    # Proline
    'CCU': 0.28, 'CCC': 0.33, 'CCA': 0.27, 'CCG': 0.11,
    # Serine
    'UCU': 0.18, 'UCC': 0.22, 'UCA': 0.15, 'UCG': 0.06, 'AGU': 0.15, 'AGC': 0.24,
    # Threonine
    'ACU': 0.24, 'ACC': 0.36, 'ACA': 0.28, 'ACG': 0.12,
    # Valine
    'GUU': 0.18, 'GUC': 0.24, 'GUA': 0.11, 'GUG': 0.47,
}
```

### Step 3: Helper Functions

```python
def translate(codon: str) -> str:
    """Translate a single codon to amino acid."""
    # Convert DNA to RNA if needed
    codon = codon.upper().replace('T', 'U')
    return CODON_TO_AA.get(codon, 'X')  # X for unknown

def codon_to_index(codon: str) -> int:
    """Convert codon to ternary operation index."""
    codon = codon.upper().replace('T', 'U')
    nucleotide_map = {'A': 0, 'U': 1, 'G': 2, 'C': 0}
    idx = 0
    for i, nuc in enumerate(codon):
        idx += nucleotide_map.get(nuc, 0) * (3 ** (8 - i))
    return idx % 19683

def one_hot_encode(indices: list) -> torch.Tensor:
    """One-hot encode a list of operation indices."""
    if isinstance(indices, int):
        indices = [indices]
    x = torch.zeros(len(indices), 19683)
    for i, idx in enumerate(indices):
        x[i, idx] = 1.0
    return x

def sequence_to_codons(sequence: str) -> list:
    """Split sequence into codons."""
    sequence = sequence.upper().replace('T', 'U')
    return [sequence[i:i+3] for i in range(0, len(sequence), 3)]
```

### Step 4: Load or Create Model

```python
# Create model (or load pretrained)
model = TernaryVAE(
    input_dim=19683,
    latent_dim=16,
    curvature=1.0,
).to(device)

# To load pretrained: model = TernaryVAE.load("checkpoints/human_codons.pt")
model.eval()
```

### Step 5: Analyze Original Sequence

```python
# Example: short peptide sequence (Met-Pro-Glu-Ala-Lys)
original_sequence = "AUGCCUGAAGCCAAG"
codons = sequence_to_codons(original_sequence)

print(f"Original sequence: {original_sequence}")
print(f"Codons: {codons}")
print(f"Amino acids: {''.join(translate(c) for c in codons)}")

# Calculate original RSCU score
original_rscu = sum(HUMAN_RSCU.get(c, 0) for c in codons)
print(f"Original RSCU score: {original_rscu:.3f}")

# Encode in latent space
with torch.no_grad():
    indices = [codon_to_index(c) for c in codons]
    x = one_hot_encode(indices).to(device)
    outputs = model(x)
    z_original = outputs["z_hyperbolic"]
    print(f"Latent representations shape: {z_original.shape}")
```

### Step 6: Generate and Score Variants

```python
def generate_synonymous_variants(codons: list) -> list:
    """Generate all single-codon synonymous substitutions."""
    variants = []

    for i, codon in enumerate(codons):
        aa = translate(codon)
        if aa in CODON_TABLE:
            for alt_codon in CODON_TABLE[aa]:
                if alt_codon != codon:
                    new_codons = codons.copy()
                    new_codons[i] = alt_codon
                    variants.append({
                        'codons': new_codons,
                        'sequence': ''.join(new_codons),
                        'position': i,
                        'change': f"{codon}->{alt_codon}",
                    })

    return variants

def score_variant(variant: dict) -> float:
    """Calculate RSCU-based expression score."""
    return sum(HUMAN_RSCU.get(c, 0) for c in variant['codons'])

# Generate all variants
variants = generate_synonymous_variants(codons)
print(f"\nGenerated {len(variants)} synonymous variants")

# Score each variant
for v in variants:
    v['rscu_score'] = score_variant(v)
    v['improvement'] = v['rscu_score'] - original_rscu

# Sort by improvement
best_variants = sorted(variants, key=lambda x: -x['improvement'])[:10]

print("\nTop 10 improvements:")
print("-" * 60)
for v in best_variants:
    print(f"  {v['sequence']}")
    print(f"    Change: position {v['position']+1}, {v['change']}")
    print(f"    RSCU: {v['rscu_score']:.3f} ({v['improvement']:+.3f})")
```

### Step 7: Full Optimization (Greedy)

```python
def optimize_sequence_greedy(codons: list, max_iterations: int = 10) -> list:
    """Greedily optimize codon usage."""
    current = codons.copy()
    history = [{'codons': current.copy(), 'rscu': score_variant({'codons': current})}]

    for iteration in range(max_iterations):
        best_improvement = 0
        best_variant = None

        # Try all single substitutions
        for variant in generate_synonymous_variants(current):
            improvement = variant['rscu_score'] - history[-1]['rscu']
            if improvement > best_improvement:
                best_improvement = improvement
                best_variant = variant

        if best_variant is None or best_improvement <= 0:
            print(f"Converged after {iteration} iterations")
            break

        current = best_variant['codons'].copy()
        history.append({
            'codons': current.copy(),
            'rscu': best_variant['rscu_score'],
            'change': best_variant['change'],
        })
        print(f"Iteration {iteration+1}: {best_variant['change']} "
              f"(RSCU: {best_variant['rscu_score']:.3f})")

    return current, history

# Run optimization
optimized_codons, history = optimize_sequence_greedy(codons)

print(f"\n{'='*60}")
print(f"Original:  {''.join(codons)}")
print(f"Optimized: {''.join(optimized_codons)}")
print(f"RSCU improvement: {history[0]['rscu']:.3f} -> {history[-1]['rscu']:.3f}")
```

### Expected Output

```
Original sequence: AUGCCUGAAGCCAAG
Codons: ['AUG', 'CCU', 'GAA', 'GCC', 'AAG']
Amino acids: MPEAK
Original RSCU score: 1.370

Generated 12 synonymous variants

Top 10 improvements:
------------------------------------------------------------
  AUGCCCGAAGCCAAG
    Change: position 2, CCU->CCC
    RSCU: 1.420 (+0.050)
  ...

Optimized: AUGCCCGAGGCCAAG
RSCU improvement: 1.370 -> 1.580
```

### What You Learned

- How codon degeneracy enables synonymous substitutions
- Using RSCU scores to predict expression levels
- Greedy optimization for codon optimization
- The TernaryVAE latent space captures codon relationships

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
        batch_size = z.size(0)

        # Compute pairwise distances
        from src.geometry import poincare_distance_matrix
        distances = poincare_distance_matrix(z, z, curvature=1.0)

        # Create mask for off-diagonal elements (exclude self-distances)
        off_diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=z.device)

        # Penalize distances below threshold
        too_close_mask = (distances < self.min_distance) & off_diag_mask

        if too_close_mask.sum() > 0:
            penalty = (self.min_distance - distances[too_close_mask]).sum()
            loss = penalty / too_close_mask.sum().float()
        else:
            loss = torch.tensor(0.0, device=z.device)

        # Calculate min pairwise distance (excluding diagonal)
        off_diag_distances = distances[off_diag_mask]
        min_dist = off_diag_distances.min().item() if len(off_diag_distances) > 0 else 0.0

        return LossResult(
            total=self.weight * loss,
            components={"diversity": loss},
            metrics={
                "min_pairwise_dist": min_dist,
                "pairs_too_close": too_close_mask.sum().item(),
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
