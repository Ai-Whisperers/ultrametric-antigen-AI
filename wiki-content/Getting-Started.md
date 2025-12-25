# Getting Started

This guide will help you install and run your first Ternary VAE experiment.

## Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended)
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Or using conda
conda create -n ternary-vae python=3.11
conda activate ternary-vae
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Development dependencies (for testing/linting)
pip install -r requirements-dev.txt
```

### 4. Verify Installation

```bash
python -c "from src.models import TernaryVAE; print('Installation successful!')"
```

## Quick Start

### Basic Training

```python
from src.models import TernaryVAE
from src.config import load_config, TrainingConfig
from src.losses import LossRegistry, create_registry_from_training_config

# Load configuration
config = TrainingConfig(
    epochs=100,
    batch_size=64,
    geometry={"curvature": 1.0, "latent_dim": 16}
)

# Create model
model = TernaryVAE(
    input_dim=19683,  # 3^9 ternary operations
    latent_dim=config.geometry.latent_dim,
    curvature=config.geometry.curvature
)

# Create loss registry
loss_registry = create_registry_from_training_config(config)

# Forward pass
outputs = model(input_tensor)
loss_result = loss_registry.compose(outputs, targets)
```

### Using Configuration Files

```bash
# Train with YAML config
python scripts/train/train.py --config configs/ternary.yaml
```

Example `configs/ternary.yaml`:

```yaml
epochs: 500
batch_size: 128
seed: 42

geometry:
  curvature: 1.0
  max_radius: 0.95
  latent_dim: 16
  learnable_curvature: false

optimizer:
  type: adamw
  learning_rate: 0.001
  weight_decay: 0.01
  schedule: cosine

loss_weights:
  reconstruction: 1.0
  kl_divergence: 0.5
  ranking: 0.1
```

## Project Structure

```
ternary-vaes-bioinformatics/
├── src/
│   ├── models/          # TernaryVAE, SwarmVAE architectures
│   ├── geometry/        # Poincare ball, p-adic operations
│   ├── losses/          # Loss components and registry
│   ├── training/        # Training loops, callbacks
│   ├── config/          # Configuration system
│   ├── encoders/        # Encoder architectures
│   ├── diseases/        # Disease-specific modules
│   └── ...
├── tests/               # Test suite
├── configs/             # YAML configurations
├── scripts/             # Training and utility scripts
└── DOCUMENTATION/       # Detailed documentation
```

## Next Steps

- [[Architecture]] - Understand the system design
- [[Configuration]] - Learn the config system
- [[Training]] - Full training guide
- [[Models]] - Explore model architectures

---

*See also: [[Installation]] for detailed setup instructions*
