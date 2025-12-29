# Epistasis Modeling

> **Mutation interaction modeling for drug resistance prediction.**

**Module**: `src/models/epistasis_module.py`
**Tests**: `tests/unit/models/test_epistasis_module.py` (32 tests)

---

## Overview

Epistasis captures how mutations interact:
- **Synergistic**: Combined effect > sum of individual effects
- **Antagonistic**: Combined effect < sum of individual effects
- **Sign epistasis**: Direction of effect changes

### Biological Examples

| Mutation Pair | Interaction | Clinical Impact |
|---------------|-------------|-----------------|
| M184V + TAMs | Antagonistic | M184V resensitizes to AZT |
| T215Y + M41L | Synergistic | Additive resistance |
| K103N + Y181C | Synergistic | NNRTI cross-resistance |

---

## Usage

```python
from src.models.epistasis_module import EpistasisModule, EpistasisResult

# Create module
epistasis = EpistasisModule(
    n_positions=300,    # Protein length
    embed_dim=64,       # Embedding dimension
    n_layers=2,         # Interaction layers
)

# Analyze mutation combination
positions = torch.tensor([[65, 184, 215]])  # M65R, M184V, T215Y
result: EpistasisResult = epistasis(positions)

# Access results
print(result.interaction_score)  # Overall interaction
print(result.synergistic)        # Synergistic component
print(result.antagonistic)       # Antagonistic component
print(result.pairwise_scores)    # Position-pair scores
```

---

## Architecture

```
Mutation Positions → Pairwise Embeddings → Interaction Network → Score
       [65, 184]   →    embed(65*300+184) →       MLP          → float
```

### EpistasisModule

```python
class EpistasisModule(nn.Module):
    def __init__(self, n_positions, embed_dim=64, n_layers=2):
        # Pairwise interaction embeddings
        self.pairwise_embed = nn.Embedding(n_positions * n_positions, embed_dim)

        # Higher-order interaction network
        self.interaction_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),  # [synergistic, antagonistic]
        )

    def forward(self, positions: torch.Tensor) -> EpistasisResult:
        # Get all pairs
        pairs = self._get_pairs(positions)

        # Embed pairs
        pair_embeds = self.pairwise_embed(pairs)

        # Compute interactions
        interactions = self.interaction_net(pair_embeds.mean(dim=1))

        return EpistasisResult(
            synergistic=interactions[:, 0],
            antagonistic=interactions[:, 1],
            interaction_score=interactions[:, 0] - interactions[:, 1],
        )
```

---

## Epistasis Loss

```python
from src.losses.epistasis_loss import EpistasisLoss

loss_fn = EpistasisLoss(
    coevolution_weight=0.1,
    interaction_weight=0.1,
)

loss = loss_fn(
    epistasis_result=result,
    predicted_resistance=pred,
    target_resistance=target,
    coevolution_matrix=coev,
)
```

---

## Visualization

```python
# Get learned interaction matrix
matrix = epistasis.get_pairwise_matrix()  # (n_positions, n_positions)

import matplotlib.pyplot as plt
plt.imshow(matrix.detach().numpy(), cmap="RdBu_r")
plt.colorbar(label="Interaction Strength")
plt.xlabel("Position j")
plt.ylabel("Position i")
plt.title("Epistasis Matrix")
plt.savefig("epistasis_matrix.png")
```

---

## Biological Validation

Known HIV epistatic pairs for validation:

| Pair | Expected | Mechanism |
|------|----------|-----------|
| M41L + T215Y | Synergistic | TAM pathway |
| D67N + K70R | Synergistic | TAM pathway |
| M184V + T215Y | Antagonistic | Fitness cost |
| K103N + Y181C | Synergistic | NNRTI binding |

---

_Last updated: 2025-12-28_
