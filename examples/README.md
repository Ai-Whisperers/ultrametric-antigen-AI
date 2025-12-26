# Examples

Example notebooks and scripts demonstrating common workflows.

## Quick Start

### 1. Basic Training

```python
"""Train a Ternary VAE model."""
import torch
from src import TernaryVAE, TernaryVAETrainer, load_config
from src.data import generate_all_ternary_operations

# Load configuration
config = load_config("configs/ternary.yaml")

# Generate ternary operations data
x, indices = generate_all_ternary_operations()
x = torch.tensor(x, dtype=torch.float32)

# Create model
model = TernaryVAE(
    latent_dim=16,
    hidden_dim=64,
    max_radius=0.95,
    curvature=1.0,
)

# Train (see scripts/train.py for full training loop)
```

### 2. Loading a Pretrained Model

```python
"""Load and use a pretrained model."""
import torch
from src import TernaryVAE
from src.utils.checkpoint import load_checkpoint_compat

# Load checkpoint
checkpoint = load_checkpoint_compat("checkpoints/best_model.pt")

# Create model with same architecture
model = TernaryVAE(latent_dim=16, hidden_dim=64)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Get embeddings
with torch.no_grad():
    outputs = model(x)
    z_hyperbolic = outputs["z_A_hyp"]
```

### 3. Codon Encoding

```python
"""Encode biological sequences using codon encoder."""
from src.encoders.codon_encoder import CodonEncoder
from src.biology.codons import codon_to_index

# Create encoder
encoder = CodonEncoder(embedding_dim=16)

# Encode a DNA sequence
sequence = "ATGCGATCG"
codons = [sequence[i:i+3] for i in range(0, len(sequence)-2, 3)]
indices = [codon_to_index(c) for c in codons]

# Get embeddings
import torch
indices_tensor = torch.tensor(indices)
embeddings = encoder(indices_tensor)
```

### 4. P-adic Distance Computation

```python
"""Compute p-adic distances between codons."""
from src.biology.codons import codon_to_index
import torch

def compute_padic_distance(idx1: int, idx2: int, p: int = 3) -> float:
    """Compute 3-adic distance between two codon indices."""
    if idx1 == idx2:
        return 0.0

    diff = abs(idx1 - idx2)
    v = 0
    while diff % p == 0:
        v += 1
        diff //= p

    return float(p) ** (-v)

# Example: Distance between ATG and ATT
idx_atg = codon_to_index("ATG")
idx_att = codon_to_index("ATT")
distance = compute_padic_distance(idx_atg, idx_att)
print(f"P-adic distance: {distance}")
```

### 5. HIV Analysis

```python
"""Analyze HIV sequences using the codon VAE."""
from pathlib import Path
from src.research import get_data_path, get_results_path

# Get paths
hiv_data = get_data_path("external/github/HIV-data")
results = get_results_path("hiv_analysis")

# Load sequences (example)
fasta_files = list(hiv_data.glob("*.fasta"))
print(f"Found {len(fasta_files)} FASTA files")
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_quick_start.ipynb` | Getting started with Ternary VAE |
| `02_codon_encoding.ipynb` | Biological sequence encoding |
| `03_hyperbolic_geometry.ipynb` | Hyperbolic latent space visualization |
| `04_hiv_analysis.ipynb` | HIV sequence analysis workflow |

## Running Examples

```bash
# Install Jupyter
pip install jupyter

# Start notebook server
jupyter notebook examples/

# Or run as Python scripts
python examples/quick_start.py
```

## Data Requirements

Some examples require external data:

1. **HIV Analysis**: Download HIV datasets first
   ```bash
   python scripts/download_hiv_datasets.py
   ```

2. **Pretrained Models**: Train a model or use provided checkpoints
   ```bash
   python scripts/train.py --config configs/ternary.yaml
   ```

## See Also

- [scripts/README.md](../scripts/README.md) - Executable scripts
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Technical architecture
- [PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) - Project layout
