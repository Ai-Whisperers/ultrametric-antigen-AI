# Hybrid Approach: Ternary VAE + AlphaFold3 Utilities

**Doc-Type:** Research Strategy · Version 1.0 · Updated 2025-12-24 · Author AI Whisperers

---

## Executive Summary

Instead of replicating AlphaFold3's massive computational requirements (A100 GPU, 630GB databases, proprietary weights), we can leverage a **hybrid approach**: use AF3's open-source utilities for database parsing and structural data, combined with our existing **ternary VAE model (v5_11_11_production/best.pt)** which already captures the discrete-continuum bridge through 3-adic geometry. This approach requires only the PDB structural database (~50GB) rather than the full MSA databases (580GB+ of redundant sequence data), since our model operates on codon-level geometry rather than sequence alignment statistics.

---

## The Insight

### What AF3 Does (and Why It's Overkill for Us)

AlphaFold3 uses:
1. **MSA databases (580GB)** - BFD, UniRef90, MGnify for evolutionary covariance
2. **Template databases** - PDB structures for structural priors
3. **Evoformer** - Massive transformer for MSA processing
4. **Structure module** - Diffusion-based 3D coordinate prediction

**Why we don't need this:** Our 3-adic codon encoder already captures the evolutionary information at a fundamental level - the ultrametric structure of codon space encodes the same hierarchical relationships that MSAs approximate through sequence statistics.

### What We Have

Our **v5_11_11_production/best.pt** model:
- Captures discrete-continuum bridge (binary↔ternary operations)
- 3-adic geometry embeds evolutionary constraints
- 16-dimensional Poincaré ball representations
- Already validated on HIV codon analysis

### The Hybrid Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    HYBRID PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  AF3 Utilities (Open Source)     Ternary VAE (Our Model)   │
│  ─────────────────────────       ──────────────────────    │
│  • PDB parsing                   • Codon embeddings        │
│  • mmCIF handling                • Hyperbolic geometry     │
│  • Chemical components           • Mutation distances      │
│  • Residue/atom constants        • Reveal score prediction │
│                                                             │
│            ↓                              ↓                 │
│       Structural Data              Geometric Analysis       │
│            ↓                              ↓                 │
│            └──────────────┬──────────────┘                 │
│                           ↓                                 │
│                  INTEGRATED PREDICTION                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Valuable AF3 Components

### USE These (Open Source, No Weights Needed)

| Component | Location | Purpose |
|:----------|:---------|:--------|
| `structure/mmcif.py` | Parsing | Read PDB/mmCIF structures |
| `constants/chemical_components.py` | Data | Amino acid properties |
| `constants/residue_names.py` | Data | Standard residue definitions |
| `constants/atom_types.py` | Data | Atom type classifications |
| `data/parsers.py` | Parsing | Sequence/structure parsing |
| `common/folding_input.py` | I/O | Input format handling |

### SKIP These (Redundant for Our Approach)

| Component | Size | Why Skip |
|:----------|:-----|:---------|
| BFD database | 270GB | MSA-based; our model captures this geometrically |
| UniRef90 | 100GB | Sequence redundancy; not needed for codon geometry |
| MGnify | 120GB | Metagenomic; irrelevant for HIV focus |
| Model weights | ~5GB | Proprietary; we use our own ternary model |
| Evoformer | - | MSA processing; replaced by 3-adic encoding |

### KEEP These (Valuable, Non-Redundant)

| Database | Size | Value |
|:---------|:-----|:------|
| PDB structures | ~50GB | Ground truth 3D coordinates |
| PDB sequences | ~2GB | Validated sequences |
| Chemical Component Dictionary | ~500MB | Ligand/PTM definitions |

---

## Implementation Plan

### Phase 1: Extract AF3 Utilities

```python
# What to copy from AF3 repo
USEFUL_MODULES = [
    "src/alphafold3/structure/",      # mmCIF parsing
    "src/alphafold3/constants/",      # Chemical data
    "src/alphafold3/common/",         # Base utilities
]
```

### Phase 2: Integrate with Ternary Model

```python
# Hybrid prediction pipeline
class HybridStructurePredictor:
    def __init__(self):
        # Load our ternary model
        self.ternary_encoder = load_model("v5_11_11_production/best.pt")

        # Use AF3 utilities for structure handling
        from alphafold3.structure import mmcif
        from alphafold3.constants import residue_names

    def predict_reveal_effect(self, wt_sequence, mutation):
        # 1. Get codon embeddings from ternary model
        wt_embedding = self.ternary_encoder.encode(wt_sequence)
        mut_embedding = self.ternary_encoder.encode(mutated_sequence)

        # 2. Compute hyperbolic distance (reveal score)
        reveal_score = poincare_distance(wt_embedding, mut_embedding)

        # 3. Map to structural context using AF3 utilities
        structural_context = self.get_pdb_context(mutation.position)

        return {
            "reveal_score": reveal_score,
            "structural_context": structural_context,
        }
```

### Phase 3: Minimal Database Setup

```bash
# Only download what we need (~50GB vs 630GB)
wget https://files.rcsb.org/download/pdb_bundle.tar.gz  # PDB structures
wget https://www.wwpdb.org/ftp/ccd/components.cif.gz    # Chemical components
```

---

## Why This Works

### Mathematical Foundation

The discrete-continuum bridge in our ternary model:

```
Binary (0,1) ←→ Ternary (0,1,2) ←→ Continuous (ℝ)
     ↓              ↓                    ↓
  Digital      3-adic metric      Hyperbolic space
  encoding     (ultrametric)      (Poincaré ball)
```

This captures the SAME evolutionary information as MSAs but through geometry rather than statistics:

- **MSA approach**: Count mutations across sequences → covariance matrix → structure
- **Our approach**: Embed codons in ultrametric space → hyperbolic geometry → structure

### Equivalence Argument

AlphaFold's MSA processing learns that:
> "Residues that co-evolve are spatially proximate"

Our 3-adic geometry encodes that:
> "Codons with similar p-adic distance encode similar structural constraints"

Both capture the same underlying biological principle, but our approach is:
1. **Computationally lighter** - No massive MSA databases
2. **More interpretable** - Geometric distances have physical meaning
3. **Directly applicable** - Reveal scores from codon distances

---

## Practical Benefits

| Aspect | AF3 Full Setup | Hybrid Approach |
|:-------|:---------------|:----------------|
| Storage | 630GB | ~50GB |
| GPU | A100 80GB | Any GPU (or CPU) |
| Setup time | Days | Hours |
| Weights | Proprietary | Our own model |
| Interpretability | Black box | Geometric insight |

---

## Next Steps

1. **Extract AF3 utilities** to `research/alphafold3/utils/`
2. **Create hybrid predictor** combining ternary model + PDB data
3. **Validate on integrase** - Compare reveal predictions to known structures
4. **Document the discrete-continuum bridge** mathematically

---

## Version History

| Version | Date | Changes |
|:--------|:-----|:--------|
| 1.0 | 2025-12-24 | Initial hybrid approach documentation |
