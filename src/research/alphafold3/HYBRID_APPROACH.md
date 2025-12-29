# Hybrid Approach: Ternary VAE + AlphaFold3 Utilities

**Doc-Type:** Research Strategy · Version 2.0 · Updated 2025-12-24 · Author AI Whisperers

---

## Executive Summary

Instead of replicating AlphaFold3's massive computational requirements (A100 GPU, 630GB databases, proprietary weights), we leverage a **hybrid approach**: use standard Python libraries (BioPython/gemmi) for structure parsing combined with our existing **ternary VAE model (v5_11_11_production/best.pt)** which captures the discrete-continuum bridge through 3-adic geometry. This approach requires only a **minimal PDB subset (~100MB)** for HIV integrase validation rather than the full 50GB PDB archive or 580GB MSA databases, since our model operates on codon-level geometry rather than sequence alignment statistics.

---

## Critical Discovery: AF3 Parsing Dependencies

**After analyzing the AF3 source code**, we discovered that AF3's Python parsing modules depend heavily on **C++ extensions** (`alphafold3.cpp`), making direct extraction impractical:

| Module | C++ Dependencies | Status |
|:-------|:-----------------|:-------|
| `structure/mmcif.py` | `cif_dict`, `mmcif_atom_site`, `mmcif_struct_conn` | Not usable without build |
| `structure/parsing.py` | `mmcif_utils`, `string_array` | Not usable without build |
| `data/parsers.py` | `fasta_iterator`, `msa_conversion` | Not usable without build |
| `constants/residue_names.py` | None | **Pure Python - usable** |
| `constants/chemical_components.py` | `cif_dict` (partial) | Requires adaptation |

**Solution**: Use established Python libraries instead of AF3's C++ extensions.

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

## Recommended Libraries (Instead of AF3 C++ Extensions)

| Task | Recommended Library | Why |
|:-----|:--------------------|:----|
| PDB/mmCIF parsing | **gemmi** or **BioPython** | Pure Python, well-tested, easy install |
| FASTA parsing | **BioPython** | Standard in bioinformatics |
| Sequence alignment | **BioPython.pairwise2** | No MSA databases needed |
| Residue constants | **AF3 residue_names.py** | Pure Python, copy directly |

### What We Extract from AF3

| Component | Location | Purpose | Status |
|:----------|:---------|:--------|:-------|
| `constants/residue_names.py` | Pure Python | CCD→one-letter mapping (800+ residues) | **Copy directly** |
| `constants/atom_types.py` | Pure Python | Atom classifications | **Copy directly** |
| `constants/chemical_component_sets.py` | Pure Python | Standard amino acids sets | **Copy directly** |

### What We Skip Entirely

| Component | Size | Why Skip |
|:----------|:-----|:---------|
| BFD database | 270GB | MSA-based; our ternary model captures this geometrically |
| UniRef90 | 100GB | Sequence redundancy; not needed for codon geometry |
| MGnify | 120GB | Metagenomic; irrelevant for HIV focus |
| UniProt | 50GB | Already encoded in our model |
| Model weights | ~5GB | Proprietary; we use our own ternary model |
| All C++ extensions | - | Use BioPython/gemmi instead |

---

## PDB Database: Redundancy Analysis

### The Full Picture

The PDB archive contains **~200,000+ structures** totaling ~50GB compressed. However, for HIV integrase validation, this is **massive overkill**:

| Dataset | Size | Structures | Relevance |
|:--------|:-----|:-----------|:----------|
| Full PDB archive | ~50GB | 200,000+ | Low (most irrelevant) |
| HIV-related structures | ~500MB | ~3,000 | Medium |
| **HIV Integrase subset** | **~100MB** | **~150** | **High (what we need)** |
| Integrase + LEDGF complexes | ~50MB | ~50 | Critical |

### Why 50GB is Redundant for Us

1. **Species irrelevance**: 95% of PDB is non-HIV (plants, bacteria, etc.)
2. **Method redundancy**: Multiple structures of same protein at different resolutions
3. **Obsolete entries**: Superseded by higher-quality structures
4. **Organism focus**: We only need HIV-1 and closely related retroviruses

### Minimal Dataset Strategy

```
HIV Integrase Validation Set (~100MB total):
├── HIV-1 Integrase structures (~50 entries)
│   ├── Apo forms (catalytic domain)
│   ├── DNA-bound complexes
│   └── Inhibitor-bound (for drug context)
├── Integrase-LEDGF complexes (~20 entries)
│   ├── Wild-type interfaces
│   └── Mutant structures (if available)
├── Related retroviral integrases (~30 entries)
│   ├── SIV, FIV, HTLV for comparison
│   └── Evolutionary context
└── Host factor structures (~20 entries)
    ├── LEDGF/p75 alone
    └── Other relevant cofactors
```

### Key PDB Entries for Validation

| PDB ID | Description | Relevance |
|:-------|:------------|:----------|
| 1EX4 | HIV-1 IN catalytic domain | Baseline structure |
| 2B4J | HIV-1 IN with LEDGF | Critical for reveal mutations |
| 3OYA | HIV-1 IN intasome | DNA-bound context |
| 4E7I | HIV-1 IN with raltegravir | Drug resistance context |
| 5U1C | HIV-1 IN strand transfer | Complete mechanism |

### Download Strategy

```bash
# Download only what we need (~100MB vs 50GB)
# Option 1: Direct PDB download
for pdb_id in 1EX4 2B4J 3OYA 4E7I 5U1C; do
    wget https://files.rcsb.org/download/${pdb_id}.cif.gz
done

# Option 2: Use RCSB search API for all HIV integrase
curl -X POST "https://search.rcsb.org/rcsbsearch/v2/query" \
  -H "Content-Type: application/json" \
  -d '{"query":{"type":"terminal","service":"text","parameters":{"value":"HIV integrase"}}}'
```

---

## Updated Implementation Plan

### Phase 1: Setup Dependencies

```bash
# Install lightweight alternatives to AF3 C++ extensions
pip install biopython gemmi

# Copy pure Python constants from AF3
cp repo/src/alphafold3/constants/residue_names.py utils/
cp repo/src/alphafold3/constants/atom_types.py utils/
cp repo/src/alphafold3/constants/chemical_component_sets.py utils/
```

### Phase 2: Download Minimal PDB Subset

```python
# scripts/download_integrase_structures.py
"""Download only the PDB structures we need for HIV integrase validation."""

import requests
import gzip
from pathlib import Path

INTEGRASE_PDBS = [
    # HIV-1 Integrase core structures
    "1EX4", "1BL3", "1BIS", "1BIU", "1BIZ",
    # Integrase-LEDGF complexes
    "2B4J", "3LPU", "3LPT", "4E7I", "4E7K",
    # Intasome structures
    "3OYA", "5U1C", "6PUT", "6PUW", "6PUY",
    # Drug resistance mutants
    "3OYM", "3OYN", "3L2T", "3L2U", "3L2V",
]

def download_structure(pdb_id: str, output_dir: Path):
    """Download a single PDB structure in mmCIF format."""
    url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_dir / f"{pdb_id}.cif.gz", "wb") as f:
            f.write(response.content)
        print(f"Downloaded {pdb_id}")
    else:
        print(f"Failed to download {pdb_id}: {response.status_code}")

# Total: ~100MB instead of 50GB
```

### Phase 3: Hybrid Predictor Implementation

```python
# src/hybrid/structure_predictor.py
"""Hybrid structure predictor using ternary VAE + PDB structures."""

from Bio.PDB import MMCIFParser
import torch
from pathlib import Path

# Import AF3 pure Python constants
from utils.residue_names import CCD_NAME_TO_ONE_LETTER

class HybridStructurePredictor:
    def __init__(self, model_path: str = "v5_11_11_production/best.pt"):
        # Load our ternary model
        self.ternary_encoder = torch.load(model_path)

        # BioPython parser for PDB structures (no C++ needed!)
        self.parser = MMCIFParser(QUIET=True)

    def get_structural_context(self, pdb_id: str, residue_pos: int):
        """Get structural context from PDB using BioPython."""
        structure = self.parser.get_structure(
            pdb_id,
            f"data/pdb/{pdb_id}.cif"
        )
        # Extract residue environment, contacts, etc.
        return self._analyze_residue_environment(structure, residue_pos)

    def predict_reveal_effect(self, wt_sequence: str, mutation: str):
        """Predict structural impact of a reveal mutation."""
        # 1. Parse mutation (e.g., "E166K")
        wt_aa, pos, mut_aa = mutation[0], int(mutation[1:-1]), mutation[-1]

        # 2. Get codon embeddings from ternary model
        wt_embedding = self.ternary_encoder.encode(wt_sequence)
        mut_sequence = wt_sequence[:pos-1] + mut_aa + wt_sequence[pos:]
        mut_embedding = self.ternary_encoder.encode(mut_sequence)

        # 3. Compute hyperbolic distance (reveal score)
        reveal_score = self._poincare_distance(wt_embedding, mut_embedding)

        # 4. Get structural context from PDB
        structural_context = self.get_structural_context("2B4J", pos)

        return {
            "reveal_score": reveal_score,
            "structural_context": structural_context,
            "mechanism": self._infer_mechanism(wt_aa, mut_aa, structural_context)
        }
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

| Aspect | AF3 Full Setup | Hybrid (Original) | Hybrid (Optimized) |
|:-------|:---------------|:------------------|:-------------------|
| Storage | 630GB | ~50GB | **~100MB** |
| GPU | A100 80GB | Any GPU (or CPU) | Any GPU (or CPU) |
| Setup time | Days | Hours | **Minutes** |
| Weights | Proprietary | Our own model | Our own model |
| Interpretability | Black box | Geometric insight | Geometric insight |
| Dependencies | C++ build, Docker | BioPython + PyTorch | BioPython + PyTorch |

---

## Storage Optimization Summary

The key insight from our analysis is that **the 50GB PDB database is still massively redundant for HIV integrase validation**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    STORAGE REDUCTION LADDER                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  AF3 Full Setup           630 GB    ████████████████████████████   │
│  ─────────────────────────────────────────────────────────────────  │
│  - MSA databases           580 GB   Replaced by ternary geometry   │
│  - PDB structures           50 GB   Still redundant                │
│                                                                     │
│  Hybrid v1 (Our Approach)   50 GB   ████████                       │
│  ─────────────────────────────────────────────────────────────────  │
│  - Full PDB                 50 GB   95% irrelevant species         │
│                                                                     │
│  Hybrid v2 (Optimized)    ~100 MB   █                              │
│  ─────────────────────────────────────────────────────────────────  │
│  - HIV Integrase subset    ~50 MB   Only what we validate          │
│  - LEDGF structures        ~20 MB   Host factor context            │
│  - Related retroviruses    ~30 MB   Evolutionary comparison        │
│                                                                     │
│  REDUCTION: 6,300x smaller than AF3 full setup                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Is This Redundancy Important?

**No, for our specific use case.** The redundancy in the full PDB comes from:

1. **Cross-species structures**: We only need HIV-1 and related retroviruses
2. **Redundant deposits**: Same protein solved multiple times (we keep best resolution)
3. **Unrelated proteins**: 99%+ of PDB has no relevance to HIV integrase

**What we lose** by using the minimal set:
- Ability to predict arbitrary proteins (not our goal)
- Template matching for de novo folding (our model doesn't need this)

**What we keep**:
- All HIV integrase structural context
- LEDGF interface geometry
- Drug resistance mutation sites
- Evolutionary comparison with related integrases

---

## Next Steps

1. **Copy AF3 pure Python constants** to `research/alphafold3/utils/`
2. **Create structure downloader** for minimal PDB subset
3. **Implement hybrid predictor** using BioPython + ternary model
4. **Validate reveal mutations** against known structures
5. **Document geometric equivalence** between MSA and 3-adic approaches

---

## References

- PDB Statistics: https://www.rcsb.org/stats/ (~200,000+ structures as of 2024)
- AlphaFold3 Repository: https://github.com/google-deepmind/alphafold3
- Our Ternary VAE: `v5_11_11_production/best.pt`

---

## Version History

| Version | Date | Changes |
|:--------|:-----|:--------|
| 2.0 | 2025-12-24 | Complete rewrite with AF3 dependency analysis and PDB redundancy study |
| 1.0 | 2025-12-24 | Initial hybrid approach documentation |
