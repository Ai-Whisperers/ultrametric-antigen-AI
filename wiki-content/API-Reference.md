# API Reference

Quick reference for the main modules and classes in Ternary VAE.

## Package Structure

```
src/
├── config/          # Configuration system
├── models/          # VAE architectures
├── geometry/        # Hyperbolic operations
├── losses/          # Loss functions
├── training/        # Training utilities
├── encoders/        # Encoder architectures
├── diseases/        # Disease-specific modules
├── observability/   # Logging and monitoring
├── data/            # Data loading
└── utils/           # Utilities
```

## src.config

### Functions

| Function | Description |
|----------|-------------|
| `load_config(path, overrides)` | Load config from YAML with optional overrides |
| `save_config(config, path)` | Save config to YAML file |

### Classes

| Class | Description |
|-------|-------------|
| `TrainingConfig` | Main training configuration |
| `GeometryConfig` | Curvature, radius, latent_dim |
| `OptimizerConfig` | Optimizer settings |
| `LossWeights` | Loss component weights |
| `RankingConfig` | Ranking loss settings |
| `VAEConfig` | Model architecture config |

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `EPSILON` | 1e-8 | Numerical stability |
| `DEFAULT_CURVATURE` | 1.0 | Default κ |
| `DEFAULT_MAX_RADIUS` | 0.95 | Default r_max |
| `DEFAULT_LATENT_DIM` | 16 | Default dimension |
| `N_TERNARY_OPERATIONS` | 19683 | 3^9 |

## src.models

### Classes

| Class | Description |
|-------|-------------|
| `TernaryVAE` | Main VAE (V5.11) |
| `TernaryVAE_OptionC` | Alternative architecture |
| `SwarmVAE` | Multi-agent VAE |
| `SwarmAgent` | Single swarm agent |
| `HyperbolicProjection` | Euclidean → Poincare |
| `DualHyperbolicProjection` | Dual projection |
| `HomeostasisController` | KL stability |
| `CurriculumScheduler` | Difficulty scheduling |
| `FrozenEncoder` | Transfer learning |
| `FrozenDecoder` | Transfer learning |

### Enums

| Enum | Values |
|------|--------|
| `AgentRole` | EXPLORER, EXPLOITER, SCOUT |

## src.geometry

### Functions

| Function | Description |
|----------|-------------|
| `poincare_distance(x, y, c)` | Distance in Poincare ball |
| `poincare_distance_matrix(X, Y, c)` | Pairwise distances |
| `exp_map_zero(v, c)` | Exponential map from origin |
| `log_map_zero(x, c)` | Logarithmic map to origin |
| `mobius_add(x, y, c)` | Hyperbolic addition |
| `parallel_transport(x, y, v, c)` | Transport vector |
| `project_to_poincare(x, r_max, c)` | Clip to ball |
| `lambda_x(x, c)` | Conformal factor |
| `get_manifold(c)` | Get geoopt manifold |
| `get_riemannian_optimizer(params, type, lr)` | Create optimizer |

### Classes

| Class | Description |
|-------|-------------|
| `PoincareModule` | Base for manifold modules |
| `ManifoldParameter` | Learnable manifold point |
| `ManifoldTensor` | Tensor on manifold |
| `RiemannianAdam` | Adam for manifolds |
| `RiemannianSGD` | SGD for manifolds |
| `HolographicPoincareManifold` | AdS/CFT extensions |
| `HolographicProjection` | Boundary projection |

## src.losses

### Functions

| Function | Description |
|----------|-------------|
| `create_registry_from_config(config)` | Registry from dict |
| `create_registry_from_training_config(config)` | Registry from TrainingConfig |
| `create_registry_with_plugins(config, plugins)` | Registry with plugins |

### Classes

| Class | Description |
|-------|-------------|
| `LossRegistry` | Dynamic loss composition |
| `LossResult` | Loss result dataclass |
| `LossGroup` | Group related losses |
| `ReconstructionLossComponent` | Cross-entropy |
| `KLDivergenceLossComponent` | KL with free bits |
| `PAdicRankingLossComponent` | 3-adic ranking |
| `RadialStratificationLossComponent` | Hierarchy |
| `EntropyLossComponent` | Entropy regularization |
| `RepulsionLossComponent` | Diversity |
| `CoEvolutionLoss` | Biosynthetic coherence |
| `SentinelGlycanLoss` | Glycan shield |
| `AutoimmuneCodonRegularizer` | Codon bias |

### Legacy Classes

| Class | Description |
|-------|-------------|
| `DualVAELoss` | Aggregated loss (deprecated) |
| `ReconstructionLoss` | Standalone recon |
| `KLDivergenceLoss` | Standalone KL |

## src.training

### Classes

| Class | Description |
|-------|-------------|
| `CallbackList` | Manage callbacks |
| `TrainingCallback` | Base callback |
| `EarlyStoppingCallback` | Patience stopping |
| `CheckpointCallback` | Save models |
| `CoveragePlateauCallback` | Coverage monitoring |
| `MetricsCallback` | Logging |

### Callback Methods

| Method | When Called |
|--------|-------------|
| `on_train_start()` | Training begins |
| `on_epoch_start(epoch)` | Epoch begins |
| `on_batch_end(batch, logs)` | Batch ends |
| `on_epoch_end(epoch, logs)` | Epoch ends |
| `on_train_end()` | Training ends |
| `should_stop()` | Check early stopping |

## src.encoders

### Classes

| Class | Description |
|-------|-------------|
| `DiffusionEncoder` | Diffusion-based encoding |
| `GeometricVectorPerceptron` | GVP for 3D data |
| `HolographicEncoder` | AdS/CFT encoding |

## src.diseases

### Classes

| Class | Description |
|-------|-------------|
| `MultipleSclerosisAnalyzer` | MS-specific analysis |
| `ViralEvolutionPredictor` | Evolution prediction |
| `mRNAStabilityPredictor` | mRNA stability |

## src.observability

### Functions

| Function | Description |
|----------|-------------|
| `setup_logging(log_dir)` | Initialize logging |
| `get_logger(name)` | Get named logger |

### Classes

| Class | Description |
|-------|-------------|
| `MetricsBuffer` | Buffered metrics |
| `AsyncMetricsWriter` | Async logging |

## src.data

### Classes

| Class | Description |
|-------|-------------|
| `TernaryDataset` | Base dataset |
| `CodonDataset` | Codon sequences |

## Common Patterns

### Model Initialization

```python
from src.models import TernaryVAE
from src.config import load_config

config = load_config("config.yaml")
model = TernaryVAE(
    input_dim=19683,
    latent_dim=config.geometry.latent_dim,
    curvature=config.geometry.curvature,
)
```

### Loss Composition

```python
from src.losses import create_registry_from_training_config

registry = create_registry_from_training_config(config)
result = registry.compose(outputs, targets)
```

### Riemannian Optimization

```python
from src.geometry import RiemannianAdam

optimizer = RiemannianAdam(model.parameters(), lr=0.001)
```

## See Also

- [[Architecture]] - System design
- [[Configuration]] - Config details
- [[Models]] - Model details
- [[Geometry]] - Geometry details
- [[Loss Functions]] - Loss details
- [[Training]] - Training details
