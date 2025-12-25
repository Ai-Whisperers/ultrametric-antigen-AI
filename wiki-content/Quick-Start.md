# Quick Start

Get up and running with Ternary VAE in under 5 minutes.

## 60-Second Overview

```python
from src.models import TernaryVAE
from src.losses import create_registry_from_training_config
from src.config import TrainingConfig
import torch

# 1. Create config
config = TrainingConfig(epochs=10, batch_size=32)

# 2. Create model
model = TernaryVAE(input_dim=19683, latent_dim=16)

# 3. Create sample data (19683 = 3^9 ternary operations)
x = torch.randint(0, 19683, (32,))  # Batch of indices
x_onehot = torch.zeros(32, 19683).scatter_(1, x.unsqueeze(1), 1)

# 4. Forward pass
outputs = model(x_onehot)

# 5. Compute loss
registry = create_registry_from_training_config(config)
result = registry.compose(outputs, x)

print(f"Loss: {result.total.item():.4f}")
print(f"Latent shape: {outputs['z_hyperbolic'].shape}")
```

## Complete Training Example

### Step 1: Setup

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import TernaryVAE
from src.config import TrainingConfig
from src.losses import create_registry_from_training_config
from src.geometry import RiemannianAdam

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### Step 2: Create Synthetic Data

```python
# Generate random ternary operation indices
n_samples = 1000
data = torch.randint(0, 19683, (n_samples,))

# Create one-hot encoding
data_onehot = torch.zeros(n_samples, 19683)
data_onehot.scatter_(1, data.unsqueeze(1), 1)

# Create DataLoader
dataset = TensorDataset(data_onehot, data)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

print(f"Created {n_samples} samples")
```

### Step 3: Initialize Model

```python
# Configuration
config = TrainingConfig(
    epochs=50,
    batch_size=64,
    geometry={"curvature": 1.0, "latent_dim": 16},
    loss_weights={"reconstruction": 1.0, "kl_divergence": 0.5},
)

# Model
model = TernaryVAE(
    input_dim=19683,
    latent_dim=config.geometry.latent_dim,
    curvature=config.geometry.curvature,
).to(device)

# Optimizer (Riemannian for hyperbolic space)
optimizer = RiemannianAdam(model.parameters(), lr=1e-3)

# Loss
loss_registry = create_registry_from_training_config(config)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Step 4: Training Loop

```python
# Training
model.train()
for epoch in range(config.epochs):
    total_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)

        # Compute loss
        result = loss_registry.compose(outputs, batch_y)

        # Backward pass
        result.total.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += result.total.item()

    avg_loss = total_loss / len(train_loader)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.4f}")
```

### Step 5: Evaluate

```python
# Evaluation
model.eval()
with torch.no_grad():
    # Get latent representations
    sample_x, sample_y = next(iter(train_loader))
    sample_x = sample_x.to(device)

    outputs = model(sample_x)
    z = outputs["z_hyperbolic"]

    print(f"\nLatent space statistics:")
    print(f"  Mean norm: {z.norm(dim=1).mean():.4f}")
    print(f"  Max norm: {z.norm(dim=1).max():.4f}")

    # Reconstruction accuracy
    preds = outputs["reconstruction"].argmax(dim=1)
    accuracy = (preds == sample_y.to(device)).float().mean()
    print(f"  Reconstruction accuracy: {accuracy:.2%}")
```

## Using Configuration Files

### Create config.yaml

```yaml
# config.yaml
seed: 42
epochs: 100
batch_size: 64

geometry:
  curvature: 1.0
  max_radius: 0.95
  latent_dim: 16

optimizer:
  type: adamw
  learning_rate: 0.001
  weight_decay: 0.01

loss_weights:
  reconstruction: 1.0
  kl_divergence: 0.5
  ranking: 0.1
```

### Load and Use

```python
from src.config import load_config

config = load_config("config.yaml")
print(f"Epochs: {config.epochs}")
print(f"Curvature: {config.geometry.curvature}")
```

## Working with Real Data

### Real Example: Human vs E. coli Codon Usage

This example uses actual codon usage patterns to train a model that distinguishes organisms:

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Real codon frequencies (Relative Synonymous Codon Usage)
# Source: Kazusa Codon Usage Database (https://www.kazusa.or.jp/codon/)

HUMAN_CODONS = {
    # Alanine (Ala, A)
    'GCT': 0.26, 'GCC': 0.40, 'GCA': 0.23, 'GCG': 0.11,
    # Arginine (Arg, R)
    'CGT': 0.08, 'CGC': 0.19, 'CGA': 0.11, 'CGG': 0.21, 'AGA': 0.20, 'AGG': 0.20,
    # Leucine (Leu, L)
    'TTA': 0.07, 'TTG': 0.13, 'CTT': 0.13, 'CTC': 0.20, 'CTA': 0.07, 'CTG': 0.41,
    # Glycine (Gly, G)
    'GGT': 0.16, 'GGC': 0.34, 'GGA': 0.25, 'GGG': 0.25,
}

ECOLI_CODONS = {
    # Alanine - E. coli prefers GCG
    'GCT': 0.16, 'GCC': 0.27, 'GCA': 0.21, 'GCG': 0.36,
    # Arginine - E. coli prefers CGT
    'CGT': 0.36, 'CGC': 0.40, 'CGA': 0.06, 'CGG': 0.10, 'AGA': 0.04, 'AGG': 0.02,
    # Leucine - E. coli prefers CTG
    'TTA': 0.11, 'TTG': 0.11, 'CTT': 0.10, 'CTC': 0.10, 'CTA': 0.04, 'CTG': 0.54,
    # Glycine - E. coli prefers GGC
    'GGT': 0.35, 'GGC': 0.41, 'GGA': 0.11, 'GGG': 0.13,
}

def codon_to_index(codon: str) -> int:
    """Convert a 3-letter codon to ternary operation index.

    Encoding: Each nucleotide position mapped to base-3 digit
    A=0, T/U=1, G=2, C=3 (mod 3 for ternary)
    """
    nucleotide_map = {'A': 0, 'T': 1, 'U': 1, 'G': 2, 'C': 0}  # Simplified mapping
    idx = 0
    for i, nuc in enumerate(codon):
        idx += nucleotide_map.get(nuc, 0) * (3 ** (8 - i))  # 3^8, 3^7, 3^6...
    return idx % 19683  # Ensure valid range

def generate_organism_data(codon_freqs: dict, n_samples: int, label: int):
    """Generate samples weighted by codon usage."""
    codons = list(codon_freqs.keys())
    probs = torch.tensor(list(codon_freqs.values()))
    probs = probs / probs.sum()  # Normalize

    # Sample codons according to organism's preference
    indices = torch.multinomial(probs, n_samples, replacement=True)
    operations = torch.tensor([codon_to_index(codons[i]) for i in indices])
    labels = torch.full((n_samples,), label, dtype=torch.long)

    return operations, labels

# Generate training data
n_per_organism = 500
human_ops, human_labels = generate_organism_data(HUMAN_CODONS, n_per_organism, label=0)
ecoli_ops, ecoli_labels = generate_organism_data(ECOLI_CODONS, n_per_organism, label=1)

# Combine datasets
all_ops = torch.cat([human_ops, ecoli_ops])
all_labels = torch.cat([human_labels, ecoli_labels])

# One-hot encode
data_onehot = torch.zeros(len(all_ops), 19683)
data_onehot.scatter_(1, all_ops.unsqueeze(1), 1)

# Create DataLoader
dataset = TensorDataset(data_onehot, all_ops, all_labels)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

print(f"Dataset: {len(all_ops)} samples (500 human, 500 E. coli)")
print(f"Human codon example: GCC (Ala) → index {codon_to_index('GCC')}")
print(f"E.coli codon example: GCG (Ala) → index {codon_to_index('GCG')}")
```

**Expected output:**
```
Dataset: 1000 samples (500 human, 500 E. coli)
Human codon example: GCC (Ala) → index 4374
E.coli codon example: GCG (Ala) → index 6561
```

### Loading FASTA Files

```python
def parse_fasta(filepath: str) -> list[tuple[str, str]]:
    """Parse a FASTA file into (header, sequence) tuples."""
    sequences = []
    current_header = None
    current_seq = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_header:
                    sequences.append((current_header, ''.join(current_seq)))
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line.upper().replace('U', 'T'))

        if current_header:
            sequences.append((current_header, ''.join(current_seq)))

    return sequences

def sequence_to_codons(seq: str) -> list[str]:
    """Split DNA sequence into codons (triplets)."""
    # Trim to multiple of 3
    seq = seq[:len(seq) - len(seq) % 3]
    return [seq[i:i+3] for i in range(0, len(seq), 3)]

# Example: Load and process a FASTA file
# sequences = parse_fasta("data/spike_protein.fasta")
# for header, seq in sequences[:3]:
#     codons = sequence_to_codons(seq)
#     print(f"{header[:30]}: {len(codons)} codons")
```

### Custom Dataset Class

```python
from torch.utils.data import Dataset

class CodonSequenceDataset(Dataset):
    """Dataset for codon sequences from FASTA files."""

    def __init__(self, fasta_path: str, max_codons: int = 100):
        self.max_codons = max_codons
        self.sequences = []

        for header, seq in parse_fasta(fasta_path):
            codons = sequence_to_codons(seq)[:max_codons]
            if len(codons) >= 10:  # Minimum length filter
                ops = [codon_to_index(c) for c in codons]
                self.sequences.append({
                    'header': header,
                    'codons': codons,
                    'operations': torch.tensor(ops, dtype=torch.long),
                })

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences[idx]
        ops = item['operations']

        # Pad or truncate to fixed length
        if len(ops) < self.max_codons:
            ops = torch.cat([ops, torch.zeros(self.max_codons - len(ops), dtype=torch.long)])

        # For simplicity, use first codon as target
        target = ops[0].item()

        # One-hot encode
        onehot = torch.zeros(19683)
        onehot[target] = 1.0

        return onehot, target
```

## Visualizing Results

### Latent Space

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Get latent representations
model.eval()
all_z = []
all_y = []

with torch.no_grad():
    for batch_x, batch_y in train_loader:
        outputs = model(batch_x.to(device))
        all_z.append(outputs["z_hyperbolic"].cpu())
        all_y.append(batch_y)

z = torch.cat(all_z).numpy()
y = torch.cat(all_y).numpy()

# t-SNE projection (for high-dim latent spaces)
if z.shape[1] > 2:
    z_2d = TSNE(n_components=2).fit_transform(z)
else:
    z_2d = z

# Plot
plt.figure(figsize=(10, 8))
plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y % 10, cmap='tab10', alpha=0.5, s=10)
plt.colorbar(label='Operation class (mod 10)')
plt.title('Latent Space Visualization')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('latent_space.png', dpi=150)
plt.show()
```

### Training Curves

```python
# Track metrics during training
history = {"loss": [], "kl": [], "recon": []}

for epoch in range(config.epochs):
    epoch_metrics = {"loss": 0.0, "kl": 0.0, "recon": 0.0}

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x.to(device))
        result = loss_registry.compose(outputs, batch_y.to(device))

        result.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_metrics["loss"] += result.total.item()
        # Extract tensor values with .item() for logging
        kl_val = result.components.get("kl_divergence", torch.tensor(0.0))
        recon_val = result.components.get("reconstruction", torch.tensor(0.0))
        epoch_metrics["kl"] += kl_val.item() if hasattr(kl_val, 'item') else kl_val
        epoch_metrics["recon"] += recon_val.item() if hasattr(recon_val, 'item') else recon_val

    for key in epoch_metrics:
        history[key].append(epoch_metrics[key] / len(train_loader))

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, values) in zip(axes, history.items()):
    ax.plot(values)
    ax.set_title(name.capitalize())
    ax.set_xlabel("Epoch")
plt.tight_layout()
plt.savefig('training_curves.png')
```

## Command Line Training

```bash
# Basic training
python scripts/train/train.py --config configs/default.yaml

# With overrides
python scripts/train/train.py \
    --config configs/default.yaml \
    --epochs 200 \
    --batch_size 128 \
    --learning_rate 0.0005

# Resume from checkpoint
python scripts/train/train.py \
    --config configs/default.yaml \
    --resume checkpoints/epoch_50.pt
```

## Next Steps

| Goal | Resource |
|------|----------|
| Understand the math | [[Geometry]] |
| Customize losses | [[Loss-Functions]] |
| Full training guide | [[Training]] |
| Tune hyperparameters | [[Configuration]] |
| Debug issues | [[Troubleshooting]] |

---

*See also: [[Tutorials]] for in-depth walkthroughs*
