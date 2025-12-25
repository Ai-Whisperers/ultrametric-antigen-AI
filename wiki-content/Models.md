# Models

This page documents the core model architectures in Ternary VAE.

## TernaryVAE (V5.11)

The canonical variational autoencoder with hyperbolic latent space.

### Architecture

```python
from src.models import TernaryVAE

model = TernaryVAE(
    input_dim=19683,      # 3^9 ternary operations
    latent_dim=16,        # Hyperbolic dimension
    hidden_dims=[512, 256, 128],  # Encoder/decoder layers
    curvature=1.0,        # Poincare ball curvature
    max_radius=0.95,      # Maximum norm in ball
)
```

### Forward Pass

```python
outputs = model(x)
# Returns dict with:
# - "reconstruction": Reconstructed logits (B, 19683)
# - "mu": Mean in Euclidean space (B, latent_dim)
# - "logvar": Log variance (B, latent_dim)
# - "z_euclidean": Sampled Euclidean coords (B, latent_dim)
# - "z_hyperbolic": Projected hyperbolic coords (B, latent_dim)
```

### Key Components

| Component | Description |
|-----------|-------------|
| `encoder` | MLP: input_dim -> hidden_dims -> (μ, log σ) |
| `decoder` | MLP: latent_dim -> hidden_dims (reversed) -> input_dim |
| `projection` | `HyperbolicProjection` with exp_map_zero |
| `curvature` | Learnable or fixed curvature parameter |

## SwarmVAE

Multi-agent swarm-based architecture for collaborative learning.

### Architecture

```python
from src.models import SwarmVAE, AgentConfig, AgentRole

config = AgentConfig(
    n_agents=5,
    roles=[AgentRole.EXPLORER, AgentRole.EXPLOITER, AgentRole.SCOUT],
    pheromone_decay=0.95,
)

model = SwarmVAE(
    input_dim=19683,
    latent_dim=16,
    agent_config=config,
)
```

### Agent Roles

| Role | Behavior |
|------|----------|
| `EXPLORER` | High variance, explores latent space |
| `EXPLOITER` | Low variance, refines good regions |
| `SCOUT` | Moderate variance, identifies boundaries |

### Pheromone Field

Agents communicate via a pheromone field that guides exploration:

```python
from src.models import PheromoneField

field = PheromoneField(
    resolution=32,
    decay_rate=0.95,
    diffusion_rate=0.1,
)
```

## HyperbolicProjection

Projects Euclidean vectors to the Poincare ball.

```python
from src.models import HyperbolicProjection

projection = HyperbolicProjection(
    curvature=1.0,
    max_radius=0.95,
    learnable=False,
)

z_hyperbolic = projection(z_euclidean)
```

### Dual Projection

For separate projections of μ and σ:

```python
from src.models import DualHyperbolicProjection

dual_proj = DualHyperbolicProjection(
    curvature=1.0,
    max_radius=0.95,
)

mu_h, sigma_h = dual_proj(mu, sigma)
```

## HomeostasisController

Maintains training stability through adaptive scaling.

```python
from src.models import HomeostasisController

controller = HomeostasisController(
    target_kl=1.0,
    kl_tolerance=0.5,
    adaptation_rate=0.01,
)

# During training
beta = controller.get_beta(current_kl)
controller.update(current_kl)
```

## CurriculumScheduler

Schedules training difficulty progression.

```python
from src.models import CurriculumScheduler

scheduler = CurriculumScheduler(
    start_difficulty=0.1,
    end_difficulty=1.0,
    warmup_epochs=50,
    schedule="linear",  # or "cosine", "exponential"
)

difficulty = scheduler.get_difficulty(epoch)
```

## FrozenEncoder / FrozenDecoder

For transfer learning with frozen weights.

```python
from src.models import FrozenEncoder, TernaryVAE

# Load pretrained model
pretrained = TernaryVAE.load("checkpoint.pt")

# Freeze encoder
frozen_enc = FrozenEncoder(pretrained.encoder)
frozen_enc.requires_grad_(False)
```

## Model Selection Guide

| Use Case | Recommended Model |
|----------|-------------------|
| Standard training | `TernaryVAE` |
| Multi-objective exploration | `SwarmVAE` |
| Transfer learning | `FrozenEncoder` + new decoder |
| Stability issues | Add `HomeostasisController` |
| Progressive difficulty | Add `CurriculumScheduler` |

## See Also

- [[Architecture]] - System overview
- [[Geometry]] - Hyperbolic operations
- [[Training]] - Training procedures
