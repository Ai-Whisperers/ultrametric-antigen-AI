# Implementation Plan: P-adic Rotamer Stability Scoring

> **Technical Roadmap for Geometric Protein Scoring**

This document details the technical implementation steps for your protein rotamer stability scoring project using p-adic valuations and hyperbolic geometry.

---

## Project Focus

| Aspect | Details |
|--------|---------|
| **Domain** | Protein Side-Chain Conformations |
| **Key Deliverable** | `Geometric_Rotamer_Scoring_Report.pdf` |
| **Core Scripts** | `ingest_pdb_rotamers.py`, `rotamer_stability.py` |

---

## Data Acquisition (RCSB PDB)

We need high-resolution protein structures to calculate side-chain angles (chi angles).

- **Source:** RCSB PDB Data API
- **API Endpoint:** `https://data.rcsb.org/rest/v1/core/entry/{pdb_id}`
- **Library:** `Biopython` (`Bio.PDB`) or `biotite`
- **Target Dataset:** "Hard-to-fold" benchmark set (CASP targets or specific proteins from your research)

### Data Format

```json
{
  "pdb_id": "1CRN",
  "chain_id": "A",
  "residue_id": 5,
  "residue_name": "LEU",
  "chi_angles": [-65.2, 174.3, null, null],
  "sequence_context": "TTCCP"
}
```

---

## Implementation Components

### 1. PDB Structure Ingestion

**Script:** `scripts/ingest/ingest_pdb_structures.py`

**Command:**
```bash
python scripts/ingest/ingest_pdb_structures.py --pdb_ids "1CRN,1TIM,4LZT"
```

**Logic:**
1. Download `.cif` or `.pdb` files from RCSB
2. Parse atoms to extract N, CA, C, CB coordinates
3. Calculate chi angles (chi1, chi2, chi3, chi4) for each residue
4. Export to structured format

### 2. Rotamer Stability Analysis

**Script:** `scripts/analysis/rotamer_padic_score.py`

**Logic:**
1. Calculate side-chain dihedral angles (chi1, chi2)
2. Compute the **3-adic valuation** of the amino acid sequence context
3. Calculate **hyperbolic distance** from common rotamer positions
4. Correlate "Rare/Unstable Rotamers" with "High P-adic Shift"

---

## Chi Angle Extraction

### Dihedral Angle Definitions

| Angle | Atoms | Residues with this angle |
|-------|-------|--------------------------|
| chi1 | N-CA-CB-XG | All rotameric (15 types) |
| chi2 | CA-CB-XG-XD | LEU, ILE, PHE, TYR, TRP, HIS, ASN, ASP, GLU, GLN, LYS, ARG, MET |
| chi3 | CB-XG-XD-XE | GLU, GLN, LYS, ARG, MET |
| chi4 | XG-XD-XE-XZ | LYS, ARG |

### Standard Rotamer Classes

| Chi1 Value | Class | Description |
|------------|-------|-------------|
| +60 deg | p (plus) | Gauche+ |
| 180 deg | t (trans) | Trans |
| -60 deg | m (minus) | Gauche- |

Combined classes: pp, pt, pm, tp, tt, tm, mp, mt, mm

---

## Scoring Metrics

### 1. Hyperbolic Distance (d_hyp)

Maps chi angles to the Poincare ball model:

```python
def hyperbolic_distance(chi_angles):
    # Map angles to Poincare disk
    r = np.linalg.norm(chi_radians) / (2 * np.pi)
    r = min(r, 0.99)  # Bound within disk

    # Hyperbolic metric
    return 2 * np.arctanh(r)
```

**Interpretation:** Higher distance = more unusual conformation

### 2. P-adic Valuation (v_p)

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

### 3. Combined Stability Score

```
stability_score = 1.0 / (1.0 + d_hyp + v_p)
```

Lower stability score = more likely to be unstable

---

## Proposed Energy Term

### E_geom Formula

```
E_geom = a * d_hyp(chi) + b * v_p(chi) + c * (1 - P_dunbrack)

Where:
- d_hyp(chi) = hyperbolic distance from common centroid
- v_p(chi) = p-adic valuation of chi encoding
- P_dunbrack = Dunbrack library probability
- a, b, c = weighting parameters (to be fitted)
```

### Rosetta Integration

```python
# Add to Rosetta scoring function
def augmented_rotamer_score(residue):
    rosetta_score = rosetta.rotamer_energy(residue)
    geometric_score = calculate_e_geom(residue)

    return rosetta_score + weight * geometric_score
```

---

## Output Specification

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
  "residues": [...]
}
```

---

## Validation Workflow

1. **Demo Run**: Generate synthetic data and verify calculations
2. **Real PDB Ingestion**: Download and process actual structures
3. **Rosetta Comparison**: Compare E_geom with Rosetta rotamer energies
4. **Discrepancy Analysis**: Find cases where methods disagree
5. **Experimental Correlation**: Validate against known folding data

---

## Expected Results

With real PDB structures:

- **Correlation with instability**: High E_geom should predict folding issues
- **Rosetta discrepancies**: Some residues flagged only by E_geom
- **Active site enrichment**: Unusual rotamers may cluster at functional sites
- **Crystal packing artifacts**: Some rare rotamers due to crystal contacts

---

## Dependencies

```bash
pip install numpy torch biopython matplotlib seaborn
```

---

*Implementation Plan for Dr. Jose Colbes - Protein Rotamer Stability Scoring*
