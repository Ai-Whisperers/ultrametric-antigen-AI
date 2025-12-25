# Constants

All configuration constants defined in `src/config/constants.py`.

---

## Numerical Stability

| Constant | Value | Purpose |
|----------|-------|---------|
| `EPSILON` | 1e-8 | General numerical stability (division, sqrt) |
| `EPSILON_LOG` | 1e-10 | Log operations (prevent log(0)) |
| `EPSILON_NORM` | 1e-8 | Norm calculations |
| `EPSILON_TEMP` | 1e-6 | Temperature scaling |

**Usage**:
```python
from src.config.constants import EPSILON

# Safe division
result = x / (y + EPSILON)

# Safe log
log_prob = torch.log(prob + EPSILON_LOG)

# Safe norm
norm = torch.sqrt((x ** 2).sum() + EPSILON_NORM)
```

---

## Geometry

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_CURVATURE` | 1.0 | Poincare ball curvature (κ) |
| `CURVATURE_MIN` | 0.01 | Minimum allowed curvature |
| `CURVATURE_MAX` | 10.0 | Maximum allowed curvature |
| `DEFAULT_MAX_RADIUS` | 0.95 | Maximum norm in Poincare ball |
| `RADIUS_MIN` | 0.1 | Minimum max_radius |
| `RADIUS_MAX` | 0.999 | Maximum max_radius |
| `DEFAULT_LATENT_DIM` | 16 | Default latent space dimension |

**Interpretation**:

```
Curvature (κ):
├── 0.5  → Mild hyperbolicity, less hierarchical
├── 1.0  → Standard (default)
├── 2.0  → Strong hyperbolicity, more hierarchical
└── >5.0 → Extreme, may cause numerical issues

Max Radius:
├── 0.9  → Conservative, more stable
├── 0.95 → Default balance
├── 0.99 → Uses more capacity, less stable
└── 0.999→ Maximum capacity, numerical edge
```

---

## Ternary Space

| Constant | Value | Purpose |
|----------|-------|---------|
| `TERNARY_BASE` | 3 | Base for ternary encoding |
| `N_TERNARY_DIGITS` | 9 | Number of ternary digits |
| `N_TERNARY_OPERATIONS` | 19683 | Total operations (3^9) |
| `MAX_VALUATION` | 9 | Maximum 3-adic valuation |

**Derivation**:
```
N_TERNARY_OPERATIONS = TERNARY_BASE ^ N_TERNARY_DIGITS
                     = 3^9
                     = 19,683
```

**Usage**:
```python
from src.config.constants import N_TERNARY_OPERATIONS

# Model output dimension
model = TernaryVAE(input_dim=N_TERNARY_OPERATIONS, ...)

# Validate operation indices
assert (ops >= 0).all() and (ops < N_TERNARY_OPERATIONS).all()
```

---

## Training Defaults

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_EPOCHS` | 100 | Default training epochs |
| `DEFAULT_BATCH_SIZE` | 64 | Default batch size |
| `DEFAULT_LEARNING_RATE` | 1e-3 | Default optimizer LR |
| `DEFAULT_WEIGHT_DECAY` | 1e-2 | Default weight decay |
| `DEFAULT_PATIENCE` | 20 | Early stopping patience |
| `DEFAULT_GRAD_CLIP` | 1.0 | Gradient clipping norm |
| `DEFAULT_FREE_BITS` | 0.5 | Free bits for KL |

**Configuration override priority**:
```
Code override > Environment variable > Config file > Default constant
```

---

## Gradient Balance

| Constant | Value | Purpose |
|----------|-------|---------|
| `GRAD_EMA_MOMENTUM` | 0.99 | EMA for gradient norms |
| `GRAD_SCALE_MIN` | 0.1 | Minimum gradient scale |
| `GRAD_SCALE_MAX` | 10.0 | Maximum gradient scale |

**Usage in homeostasis**:
```python
# Exponential moving average of gradient norms
ema_grad = GRAD_EMA_MOMENTUM * ema_grad + (1 - GRAD_EMA_MOMENTUM) * current_grad

# Clip scaling factor
scale = torch.clamp(target / ema_grad, GRAD_SCALE_MIN, GRAD_SCALE_MAX)
```

---

## Loss Functions

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_N_TRIPLETS` | 100 | Triplets for ranking loss |
| `DEFAULT_RANKING_MARGIN` | 1.0 | Triplet margin |
| `DEFAULT_HARD_NEGATIVE_RATIO` | 0.5 | Hard negative mining ratio |
| `DEFAULT_REPULSION_SIGMA` | 0.1 | Repulsion kernel width |

**Usage**:
```python
from src.config.constants import DEFAULT_RANKING_MARGIN

# Triplet loss
loss = torch.relu(d_pos - d_neg + DEFAULT_RANKING_MARGIN)
```

---

## Observability

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_LOG_DIR` | "logs/" | Logging directory |
| `DEFAULT_TENSORBOARD_DIR` | "runs/" | TensorBoard directory |
| `DEFAULT_CHECKPOINT_DIR` | "checkpoints/" | Checkpoint directory |
| `DEFAULT_LOG_INTERVAL` | 100 | Log every N batches |
| `DEFAULT_CHECKPOINT_FREQ` | 10 | Checkpoint every N epochs |
| `DEFAULT_HISTOGRAM_INTERVAL` | 100 | TensorBoard histogram interval |
| `DEFAULT_EMBEDDING_INTERVAL` | 500 | TensorBoard embedding interval |
| `DEFAULT_EVAL_SAMPLES` | 1000 | Samples for evaluation |

---

## Accessing Constants

### Import Directly

```python
from src.config.constants import (
    EPSILON,
    DEFAULT_CURVATURE,
    N_TERNARY_OPERATIONS,
)
```

### Import All

```python
from src.config.constants import *

# Now all constants are in scope
print(EPSILON)
print(DEFAULT_CURVATURE)
```

### Via Config Module

```python
from src.config import EPSILON, DEFAULT_CURVATURE
```

---

## Environment Variable Overrides

Some constants can be overridden via environment variables:

| Constant | Environment Variable |
|----------|---------------------|
| `DEFAULT_EPOCHS` | `TVAE_EPOCHS` |
| `DEFAULT_BATCH_SIZE` | `TVAE_BATCH_SIZE` |
| `DEFAULT_LEARNING_RATE` | `TVAE_OPTIMIZER_LEARNING_RATE` |
| `DEFAULT_CURVATURE` | `TVAE_GEOMETRY_CURVATURE` |

**Example**:
```bash
export TVAE_EPOCHS=500
export TVAE_GEOMETRY_CURVATURE=2.0

python train.py  # Will use overridden values
```

---

## Changing Defaults

To change defaults for your project, you have three options:

### 1. Config File (Recommended)

```yaml
# config.yaml
epochs: 500
geometry:
  curvature: 2.0
```

### 2. Environment Variables

```bash
export TVAE_EPOCHS=500
```

### 3. Code Override

```python
config = TrainingConfig(
    epochs=500,
    geometry={"curvature": 2.0},
)
```

**Do NOT modify `constants.py` directly** - it affects all users.

---

*See also: [[Configuration]], [[Training]]*
