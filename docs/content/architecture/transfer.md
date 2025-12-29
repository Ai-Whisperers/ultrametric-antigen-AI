# Transfer Learning

> **Multi-disease transfer learning for drug resistance prediction.**

**Module**: `src/training/transfer_pipeline.py`
**Tests**: `tests/unit/training/test_transfer_pipeline.py` (30 tests)

---

## Overview

Transfer learning addresses data scarcity:

| Disease | Training Samples | Challenge |
|---------|------------------|-----------|
| HIV | 200,000+ | Well-studied |
| HBV | ~50,000 | Moderate data |
| Tuberculosis | ~20,000 | Limited MDR data |
| Candida auris | ~500 | Emerging pathogen |
| RSV | ~200 | Very limited |

---

## Transfer Strategies

| Strategy | Trainable Params | Best For | Risk |
|----------|------------------|----------|------|
| FROZEN_ENCODER | Head only (~5%) | <1000 samples | Under-adaptation |
| FULL_FINETUNE | All (100%) | >5000 samples | Catastrophic forgetting |
| ADAPTER | Adapters (~10%) | Moderate data | Complexity |
| LORA | Low-rank (~5%) | Large models | Rank selection |
| MAML | Inner loop | Few-shot (5-50) | Compute cost |

---

## Usage

### Basic Transfer

```python
from src.training.transfer_pipeline import (
    TransferLearningPipeline,
    TransferConfig,
    TransferStrategy,
)

# Configure
config = TransferConfig(
    latent_dim=32,
    hidden_dims=[256, 128, 64],
    strategy=TransferStrategy.FROZEN_ENCODER,
    pretrain_epochs=100,
    finetune_epochs=50,
)

# Create pipeline
pipeline = TransferLearningPipeline(config)

# Pre-train on all diseases
pretrained = pipeline.pretrain({
    "hiv": hiv_dataset,
    "hbv": hbv_dataset,
    "tb": tb_dataset,
})

# Fine-tune on target
finetuned = pipeline.finetune("candida", candida_dataset)
```

### Few-Shot with MAML

```python
config = TransferConfig(
    strategy=TransferStrategy.MAML,
    maml_inner_lr=0.01,
    maml_inner_steps=5,
)

pipeline = TransferLearningPipeline(config)
adapted = pipeline.few_shot_adapt("rsv", rsv_support_set, k_shot=10)
```

---

## Strategy Details

### Frozen Encoder

```python
# Freeze encoder, train only head
for param in model.encoder.parameters():
    param.requires_grad = False

optimizer = Adam(model.head.parameters(), lr=1e-3)
```

### Adapter Layers

```python
class AdapterModule(nn.Module):
    def __init__(self, input_dim, adapter_dim=64):
        self.down = nn.Linear(input_dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, input_dim)

    def forward(self, x):
        return x + self.up(F.relu(self.down(x)))
```

### LoRA

$$W' = W + BA$$

Where B and A are low-rank matrices (rank r << d).

### MAML

```python
def maml_adapt(model, support_set, inner_lr=0.01, inner_steps=5):
    adapted_params = dict(model.named_parameters())

    for _ in range(inner_steps):
        loss = compute_loss(model, support_set)
        grads = torch.autograd.grad(loss, adapted_params.values())
        adapted_params = {
            name: param - inner_lr * grad
            for (name, param), grad in zip(adapted_params.items(), grads)
        }

    return adapted_params
```

---

## Cross-Disease Transfer Matrix

| Source → Target | HIV | HBV | TB | Flu | COVID |
|-----------------|-----|-----|-----|-----|-------|
| HIV | 1.0 | 0.82 | 0.45 | 0.61 | 0.73 |
| HBV | 0.79 | 1.0 | 0.41 | 0.58 | 0.69 |
| TB | 0.38 | 0.35 | 1.0 | 0.42 | 0.39 |
| Flu | 0.55 | 0.52 | 0.40 | 1.0 | 0.76 |
| COVID | 0.67 | 0.64 | 0.38 | 0.71 | 1.0 |

**Interpretation**:
- High transfer (>0.7): Similar mechanisms (HIV↔HBV, Flu↔COVID)
- Low transfer (<0.5): Different mechanisms (TB↔viruses)

---

## Evaluation

```python
results = pipeline.evaluate_transfer(source="hiv", target="candida")

print(results["spearman"])      # Correlation
print(results["mse"])           # Mean squared error
print(results["transfer_gain"]) # Improvement over baseline
```

---

_Last updated: 2025-12-28_
