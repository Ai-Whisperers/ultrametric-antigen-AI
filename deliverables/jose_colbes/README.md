# Deliverable Package: Dr. José Colbes
## P-adic Rotamer Stability Scoring for Protein Optimization

**Prepared for:** Dr. José Colbes
**Project:** Ternary VAE Bioinformatics - Partnership Phase 2
**Date:** December 26, 2024

---

## Overview

This package contains all materials for validating the **P-adic Geometric Rotamer Scoring Function**. The system provides a novel approach to identifying unstable protein side-chain conformations that may be missed by traditional methods like Rosetta's Dunbrack-based scoring.

---

## What's Included

### 1. Core Scripts

| File | Description |
|------|-------------|
| `scripts/ingest_pdb_rotamers.py` | PDB structure ingestion and chi angle extraction (490 lines) |
| `scripts/rotamer_stability.py` | P-adic stability analysis algorithm (385 lines) |

### 2. Interactive Notebook

| File | Description |
|------|-------------|
| `notebooks/colbes_scoring_function.ipynb` | Jupyter notebook for visualization and analysis |

### 3. Results

| File | Description |
|------|-------------|
| `results/rotamer_stability.json` | Complete analysis of 500 demo residues |

### 4. Reference Data

| File | Description |
|------|-------------|
| `data/demo_rotamers.pt` | PyTorch tensor of chi angles (500 residues) |

### 5. Documentation

| File | Description |
|------|-------------|
| `docs/TECHNICAL_PROPOSAL.md` | Original technical proposal |
| `docs/IMPLEMENTATION_GUIDE.md` | Implementation specifications |

---

## Quick Start

### Step 1: Install Dependencies

```bash
pip install numpy torch biopython matplotlib seaborn
```

### Step 2: Generate Demo Data

```bash
cd scripts
python ingest_pdb_rotamers.py --demo \
    --output ../data/demo_rotamers.pt
```

**Expected Output:**
```
Created demo rotamer data at data/demo_rotamers.pt
  500 residues from 5 demo structures
```

### Step 3: Run Stability Analysis

```bash
python rotamer_stability.py \
    --input ../data/demo_rotamers.pt \
    --output ../results/rotamer_stability.json
```

**Expected Output:**
```
Loading rotamer data from data/demo_rotamers.pt...
Loaded 500 residues
Analyzing rotamer stability...

=== Summary ===
Total residues: 500
Rare rotamers: 500 (100.0%)
Mean hyperbolic distance: 7.679
Hyp-Eucl correlation: -0.051
Exported results to results/rotamer_stability.json
```

### Step 4: Explore in Notebook

```bash
jupyter notebook notebooks/colbes_scoring_function.ipynb
```

---

## Technical Details

### Chi Angle Extraction

The system extracts side-chain dihedral angles (χ1-χ4) from PDB structures:

| Angle | Definition | Residues |
|-------|------------|----------|
| χ1 | N-CA-CB-XG | All rotameric (15 types) |
| χ2 | CA-CB-XG-XD | LEU, ILE, PHE, TYR, TRP, HIS, ASN, ASP, GLU, GLN, LYS, ARG, MET |
| χ3 | CB-XG-XD-XE | GLU, GLN, LYS, ARG, MET |
| χ4 | XG-XD-XE-XZ | LYS, ARG |

### Geometric Scoring Method

Each rotamer is scored using three complementary metrics:

#### 1. Hyperbolic Distance (d_hyp)

Maps chi angles to the Poincaré ball model and computes distance from the common rotamer centroid:

```python
def hyperbolic_distance(chi_angles):
    # Map angles to Poincaré disk
    r = np.linalg.norm(chi_radians) / (2 * np.pi)
    r = min(r, 0.99)  # Bound within disk

    # Hyperbolic metric
    return 2 * np.arctanh(r)
```

**Interpretation:** Higher distance = more unusual conformation

#### 2. P-adic Valuation (v_p)

Computes the 3-adic valuation of the angle encoding:

```python
def padic_valuation(chi_encoded, p=3):
    if chi_encoded == 0:
        return 0
    v = 0
    while chi_encoded % p == 0:
        v += 1
        chi_encoded //= p
    return v
```

**Interpretation:** Higher valuation = algebraically "deeper" structure

#### 3. Nearest Rotamer Distance

Compares to standard Dunbrack rotamer library positions:

| Rotamer Class | χ1 | χ2 |
|---------------|-----|-----|
| p (plus) | +60° | - |
| t (trans) | 180° | - |
| m (minus) | -60° | - |
| pp, pt, pm, tp, tt, tm, mp, mt, mm | combinations | |

### Proposed Energy Term

The geometric scoring function can augment Rosetta:

```
E_geom = α · d_hyp(χ) + β · v_p(χ) + γ · (1 - P_dunbrack)

Where:
- d_hyp(χ) = hyperbolic distance from common centroid
- v_p(χ) = p-adic valuation of chi encoding
- P_dunbrack = Dunbrack library probability
- α, β, γ = weighting parameters (to be fitted)
```

---

## Output Format

### JSON Structure

```json
{
  "summary": {
    "n_residues": 500,
    "n_rare": 450,
    "rare_fraction": 0.9,
    "hyperbolic_distance": {
      "mean": 7.679,
      "std": 0.803,
      "min": 7.600,
      "max": 17.329
    },
    "padic_valuation": {
      "mean": 0.434,
      "max": 6
    }
  },
  "residues": [
    {
      "pdb_id": "1CRN",
      "chain_id": "A",
      "residue_id": 5,
      "residue_name": "LEU",
      "chi_angles": [-65.2, 174.3, null, null],
      "nearest_rotamer": "mt",
      "euclidean_distance": 0.534,
      "hyperbolic_distance": 7.612,
      "padic_valuation": 1,
      "stability_score": 0.876,
      "is_rare": true
    }
  ]
}
```

### Field Descriptions

| Field | Description | Range |
|-------|-------------|-------|
| `chi_angles` | Dihedral angles in degrees | [-180, 180] or null |
| `nearest_rotamer` | Closest Dunbrack rotamer | p, t, m, pp, pt, ... |
| `euclidean_distance` | Distance to nearest rotamer | [0, ∞) |
| `hyperbolic_distance` | Poincaré ball distance | [0, ∞) |
| `padic_valuation` | 3-adic valuation | {0, 1, 2, ...} |
| `stability_score` | Combined geometric score | [0, 1] |
| `is_rare` | Flagged as unusual | true/false |

---

## Using with Real PDB Structures

### Step 1: Download Structures

```bash
python ingest_pdb_rotamers.py \
    --pdb_ids "1CRN,1TIM,4LZT,2CI2" \
    --output ../data/real_rotamers.pt \
    --cache_dir ../data/pdb_cache
```

### Step 2: Run Analysis

```bash
python rotamer_stability.py \
    --input ../data/real_rotamers.pt \
    --output ../results/real_stability.json \
    --rare_threshold 0.8
```

### Step 3: Compare with Rosetta

```python
import json

# Load our geometric scores
with open('results/real_stability.json') as f:
    geo_results = json.load(f)

# For each residue flagged as rare:
for res in geo_results['residues']:
    if res['is_rare']:
        print(f"{res['pdb_id']}:{res['chain_id']}:{res['residue_id']}")
        print(f"  Hyperbolic distance: {res['hyperbolic_distance']:.3f}")
        print(f"  P-adic valuation: {res['padic_valuation']}")
        # Compare with your Rosetta rotamer scores here
```

---

## Key Innovation

**Current Methods (Rosetta/Dunbrack):**
- Statistical potentials from PDB frequency
- May miss rare but valid conformations
- No algebraic structure

**Our Geometric Approach:**
- P-adic valuations capture hierarchical structure
- Hyperbolic distance measures "unusualness"
- Can identify "Rosetta-blind" unstable rotamers
- Complementary to existing methods

---

## Validation Checklist

- [ ] Chi angles extracted correctly from demo data
- [ ] Hyperbolic distances computed for all residues
- [ ] P-adic valuations assigned
- [ ] Rare rotamers flagged based on threshold
- [ ] JSON export is complete and parseable
- [ ] Notebook visualizations render correctly
- [ ] Real PDB ingestion works (requires internet)

---

## Expected Findings

When run on real protein structures, you should observe:

1. **Most residues** cluster near common rotamer positions
2. **Rare rotamers** often correspond to:
   - Active site residues (functional constraints)
   - Crystal packing contacts (artifacts)
   - Strained regions (potential instability)
3. **Correlation** between hyperbolic distance and Rosetta energy may reveal complementary information

---

## Questions?

- See docstrings in `rotamer_stability.py` for algorithm details
- Chi angle definitions from IUPAC conventions
- Dunbrack library reference: Shapovalov & Dunbrack (2011)

---

*Prepared as part of the Ternary VAE Bioinformatics Partnership*
