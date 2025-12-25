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

| Function | Signature | Description |
|----------|-----------|-------------|
| `load_config` | `(path: str, overrides: dict = None) -> TrainingConfig` | Load config from YAML with optional overrides |
| `save_config` | `(config: TrainingConfig, path: str) -> None` | Save config to YAML file |

### Classes

| Class | Key Attributes | Description |
|-------|----------------|-------------|
| `TrainingConfig` | `epochs`, `batch_size`, `geometry`, `optimizer`, `loss_weights` | Main training configuration |
| `GeometryConfig` | `curvature=1.0`, `max_radius=0.95`, `latent_dim=16` | Curvature, radius, latent_dim |
| `OptimizerConfig` | `type="adamw"`, `learning_rate=0.001`, `weight_decay=0.01` | Optimizer settings |
| `LossWeights` | `reconstruction=1.0`, `kl_divergence=1.0`, `ranking=0.5` | Loss component weights |
| `RankingConfig` | `margin=1.0`, `n_triplets=100` | Ranking loss settings |
| `VAEConfig` | `input_dim`, `latent_dim`, `hidden_dims` | Model architecture config |

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

| Function | Signature | Description |
|----------|-----------|-------------|
| `poincare_distance` | `(x: Tensor, y: Tensor, curvature: float = 1.0) -> Tensor` | Distance in Poincaré ball |
| `poincare_distance_matrix` | `(X: Tensor, Y: Tensor, curvature: float = 1.0) -> Tensor` | Pairwise distances (N×M) |
| `exp_map_zero` | `(v: Tensor, curvature: float = 1.0) -> Tensor` | Exponential map from origin |
| `log_map_zero` | `(x: Tensor, curvature: float = 1.0) -> Tensor` | Logarithmic map to origin |
| `mobius_add` | `(x: Tensor, y: Tensor, curvature: float = 1.0) -> Tensor` | Hyperbolic addition (x ⊕ y) |
| `parallel_transport` | `(x: Tensor, y: Tensor, v: Tensor, curvature: float = 1.0) -> Tensor` | Transport v from T_x to T_y |
| `project_to_poincare` | `(x: Tensor, max_radius: float = 0.95, curvature: float = 1.0) -> Tensor` | Clip to ball |
| `lambda_x` | `(x: Tensor, curvature: float = 1.0) -> Tensor` | Conformal factor λ(x) |
| `get_manifold` | `(curvature: float = 1.0) -> PoincareBall` | Get geoopt manifold |
| `get_riemannian_optimizer` | `(params, type: str = "adam", lr: float = 0.001) -> Optimizer` | Create Riemannian optimizer |

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
