# Structure-Aware Modeling

> **AlphaFold2 integration with SE(3)-equivariant encoding.**

**Modules**: `src/models/structure_aware_vae.py`, `src/encoders/alphafold_encoder.py`
**Tests**: `tests/unit/models/test_structure_aware_vae.py` (35 tests)

---

## Overview

Protein structure affects drug resistance:
- **Drug binding**: Changes in binding pocket geometry
- **Protein stability**: Fitness cost of mutations
- **Allosteric effects**: Distant mutations affecting active site

---

## SE(3) Equivariance

SE(3) = Special Euclidean Group (rotations + translations in 3D)

A function is **SE(3) equivariant** if:
- Vectors: $f(Rx + t) = Rf(x) + t$
- Scalars: $f(Rx + t) = f(x)$

**Why it matters**: Same protein should produce same representation regardless of orientation.

---

## Usage

### StructureAwareVAE

```python
from src.models.structure_aware_vae import StructureAwareVAE, StructureConfig

config = StructureConfig(
    use_structure=True,
    structure_dim=64,
    n_structure_layers=3,
    cutoff=10.0,              # Angstroms
    use_plddt=True,           # Confidence weighting
    fusion_type="cross_attention",  # or "gated", "concat"
)

model = StructureAwareVAE(
    input_dim=128,
    latent_dim=32,
    structure_config=config,
)

# Forward pass with structure
outputs = model(
    x=sequence_embedding,
    structure=alphafold_coords,    # (batch, L, 3)
    plddt=confidence_scores,       # (batch, L)
    aa_indices=amino_acid_indices, # (batch, L)
)
```

### AlphaFold Structure Loading

```python
from src.encoders.alphafold_encoder import AlphaFoldStructureLoader

loader = AlphaFoldStructureLoader(cache_dir=".alphafold_cache")

# Download/load structure
structure = loader.get_structure("P04637")  # UniProt ID

print(structure["coords"].shape)   # (L, 3) - CA coordinates
print(structure["plddt"].shape)    # (L,) - confidence
print(structure["sequence"])       # Amino acid sequence
```

---

## Architecture

### SE3Encoder

```python
class SE3Encoder(nn.Module):
    def __init__(self, node_dim=64, edge_dim=32, n_layers=3, cutoff=10.0):
        self.cutoff = cutoff
        self.node_embed = nn.Embedding(21, node_dim)  # 20 AAs + unknown
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim),
        )
        self.layers = nn.ModuleList([
            SE3Layer(node_dim, edge_dim) for _ in range(n_layers)
        ])

    def forward(self, coords, aa_indices=None):
        # Compute pairwise distances (SE(3) invariant!)
        dists = torch.cdist(coords, coords)

        # Edge features from distances
        edge_mask = (dists < self.cutoff) & (dists > 0)
        edge_features = self.edge_mlp(dists.unsqueeze(-1))

        # Node features from amino acid type
        node_features = self.node_embed(aa_indices)

        # Message passing
        for layer in self.layers:
            node_features = layer(node_features, edge_features, edge_mask)

        return node_features.mean(dim=1)  # Pool over residues
```

### Fusion Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| Cross-Attention | Sequence attends to structure | Best performance |
| Gated | Learned gate controls contribution | Uncertain structure |
| Concatenation | Simple concat + projection | Baseline |

---

## pLDDT Interpretation

AlphaFold2 provides per-residue confidence:

| Score | Meaning | Usage |
|-------|---------|-------|
| >90 | Very high confidence | Full weight |
| 70-90 | Confident | Normal weight |
| 50-70 | Low confidence | Reduced weight |
| <50 | Very low (disordered) | Near-zero weight |

```python
# Confidence weighting
plddt_weight = torch.sigmoid(plddt / 100.0)
weighted_embed = struct_embed * plddt_weight
```

---

## Structural Insights for Resistance

| Mutation | Structural Effect | Resistance Mechanism |
|----------|-------------------|----------------------|
| K103N (NNRTI) | Disrupts hydrophobic pocket | Prevents drug binding |
| M184V (NRTI) | Steric clash with drug | Excludes 3TC/FTC |
| T215Y (NRTI) | Enhanced primer unblocking | Excision pathway |

---

_Last updated: 2025-12-28_
