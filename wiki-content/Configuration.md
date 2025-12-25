# Configuration

The Ternary VAE uses a centralized, type-safe configuration system.

## Overview

```python
from src.config import load_config, TrainingConfig

# Load from YAML
config = load_config("configs/training.yaml")

# Or create programmatically
config = TrainingConfig(
    epochs=500,
    batch_size=128,
)
```

## Configuration Hierarchy

```
TrainingConfig
├── seed: int = 42
├── epochs: int = 100
├── batch_size: int = 64
├── geometry: GeometryConfig
│   ├── curvature: float = 1.0
│   ├── max_radius: float = 0.95
│   ├── latent_dim: int = 16
│   └── learnable_curvature: bool = False
├── optimizer: OptimizerConfig
│   ├── type: str = "adamw"
│   ├── learning_rate: float = 0.001
│   ├── weight_decay: float = 0.01
│   └── schedule: str = "constant"
└── loss_weights: LossWeights
    ├── reconstruction: float = 1.0
    ├── kl_divergence: float = 1.0
    └── ranking: float = 0.5
```

## YAML Configuration

Create a `config.yaml` file:

```yaml
# Training parameters
seed: 42
epochs: 500
batch_size: 128

# Geometry
geometry:
  curvature: 1.0
  max_radius: 0.95
  latent_dim: 16
  learnable_curvature: false

# Optimizer
optimizer:
  type: adamw
  learning_rate: 0.001
  weight_decay: 0.01
  schedule: cosine
  warmup_steps: 1000

# Loss weights
loss_weights:
  reconstruction: 1.0
  kl_divergence: 0.5
  ranking: 0.1
  entropy: 0.01
```

Load it:

```python
config = load_config("config.yaml")
```

## Environment Variables

Override any config value via environment variables:

```bash
# Format: TVAE_<FIELD> or TVAE_<SECTION>_<FIELD>
export TVAE_EPOCHS=1000
export TVAE_BATCH_SIZE=256
export TVAE_GEOMETRY_CURVATURE=2.0
export TVAE_OPTIMIZER_LEARNING_RATE=0.0005
```

Environment variables take precedence over YAML.

## Programmatic Overrides

```python
config = load_config(
    "config.yaml",
    overrides={
        "epochs": 1000,
        "geometry": {"curvature": 2.0},
    }
)
```

Priority: overrides > env vars > YAML > defaults

## Validation

Configs are validated on creation:

```python
from src.config import ConfigValidationError, GeometryConfig

try:
    config = GeometryConfig(curvature=-1.0)  # Invalid!
except ConfigValidationError as e:
    print(e)  # "curvature must be positive"
```

Validation rules:
- `curvature`: Must be positive
- `max_radius`: Must be in (0, 1)
- `latent_dim`: Must be >= 2
- `epochs`: Must be positive
- `batch_size`: Must be positive
- Loss weights: Must be non-negative

## Configuration Classes

### GeometryConfig

```python
from src.config import GeometryConfig

geom = GeometryConfig(
    curvature=1.0,        # Poincare ball curvature
    max_radius=0.95,      # Maximum norm in ball
    latent_dim=16,        # Latent space dimension
    learnable_curvature=False,  # Make curvature trainable
)
```

### OptimizerConfig

```python
from src.config import OptimizerConfig

opt = OptimizerConfig(
    type="adamw",           # adam, adamw, sgd, radam
    learning_rate=0.001,
    weight_decay=0.01,
    schedule="cosine",      # constant, cosine, linear, exponential
    warmup_steps=1000,
)
```

### LossWeights

```python
from src.config import LossWeights

weights = LossWeights(
    reconstruction=1.0,
    kl_divergence=0.5,
    ranking=0.1,
    entropy=0.01,
    repulsion=0.1,
)
```

### RankingConfig

```python
from src.config import RankingConfig

ranking = RankingConfig(
    margin=1.0,
    n_triplets=100,
    hard_negative_ratio=0.5,
)
```

## Saving Configuration

```python
from src.config import save_config

save_config(config, "output/config.yaml")
```

## Constants

Common constants are centralized:

```python
from src.config.constants import (
    EPSILON,              # 1e-8 (numerical stability)
    DEFAULT_CURVATURE,    # 1.0
    DEFAULT_MAX_RADIUS,   # 0.95
    DEFAULT_LATENT_DIM,   # 16
    N_TERNARY_OPERATIONS, # 19683 (3^9)
)
```

## Serialization

Convert to/from dictionaries:

```python
# To dict
data = config.to_dict()

# From dict
config = TrainingConfig.from_dict(data)
```

## Best Practices

1. **Use YAML for experiments**: Keep configs in version control
2. **Use env vars for secrets**: Don't commit sensitive values
3. **Use overrides for sweeps**: Programmatic hyperparameter search
4. **Validate early**: Create config at startup, not during training

## See Also

- [[Training]] - Using configs in training
- [[Architecture]] - System design
- [[Constants]] - All constant values
