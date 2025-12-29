# BaseVAE Abstraction

> **Unified base class for all VAE variants in the framework.**

**Module**: `src/models/base_vae.py`
**Tests**: `tests/unit/models/test_base_vae.py` (33 tests)

---

## Overview

The BaseVAE provides a unified interface for all 19+ VAE variants, reducing code duplication and ensuring consistent behavior.

### Key Features

- Standardized `encode()`, `decode()`, `reparameterize()` methods
- Common loss computation utilities
- Parameter counting and model introspection
- Hyperbolic projection support
- Configuration via dataclasses

---

## Usage

### Creating a Custom VAE

```python
from src.models.base_vae import BaseVAE, VAEConfig, VAEOutput
import torch
import torch.nn as nn
from typing import Tuple

class MyCustomVAE(BaseVAE):
    def __init__(self, config: VAEConfig):
        super().__init__(config)
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.fc_mu = nn.Linear(64, config.latent_dim)
        self.fc_logvar = nn.Linear(64, config.latent_dim)
        self.decoder = nn.Linear(config.latent_dim, config.input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
```

### Using Existing VAE Variants

```python
from src.models import TernaryVAE, SimpleVAE, MultiTaskVAE

# Simple VAE
model = SimpleVAE(input_dim=64, latent_dim=16)

# Ternary VAE (production)
model = TernaryVAE(config)

# Multi-task VAE (multiple diseases)
model = MultiTaskVAE(input_dim=64, latent_dim=16, n_tasks=11)
```

---

## API Reference

### VAEConfig

```python
@dataclass
class VAEConfig:
    input_dim: int = 64
    latent_dim: int = 16
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    dropout: float = 0.1
    use_batch_norm: bool = True
```

### VAEOutput

```python
@dataclass
class VAEOutput:
    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor
    loss: Optional[torch.Tensor] = None
    metrics: Optional[Dict[str, float]] = None
```

### BaseVAE Methods

| Method | Description |
|--------|-------------|
| `encode(x)` | Returns (mu, logvar) tuple |
| `decode(z)` | Reconstructs from latent |
| `reparameterize(mu, logvar)` | Samples z using reparameterization trick |
| `forward(x)` | Full forward pass, returns VAEOutput |
| `compute_loss(output, target)` | Computes reconstruction + KL loss |
| `count_parameters()` | Returns dict with parameter counts |

---

## Inheritance Hierarchy

```
BaseVAE
├── SimpleVAE           # Basic VAE
├── TernaryVAE          # Production dual-VAE
├── MultiTaskVAE        # Multi-disease
├── StructureAwareVAE   # AlphaFold2 integration
├── MAMLVAE             # Meta-learning
├── SwarmVAE            # Swarm intelligence
└── ... (19+ variants)
```

---

## Testing

```bash
# Run BaseVAE tests
pytest tests/unit/models/test_base_vae.py -v

# Run all model tests
pytest tests/unit/models/ -v
```

---

_Last updated: 2025-12-28_
