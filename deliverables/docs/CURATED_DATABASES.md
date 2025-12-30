# Curated Bioinformatics Databases

**Version**: 1.0.0
**Updated**: 2025-12-30
**Authors**: AI Whisperers

---

## Overview

This document describes the curated experimental databases used for training machine learning models in this project. All data has been manually curated from peer-reviewed literature and public databases to ensure scientific validity.

## Why Curated Data?

Previously, the project used **synthetic/demo data** that was algorithmically generated. This had critical issues:

1. **No real-world validity**: Synthetic data cannot capture true biological complexity
2. **Overfitting artifacts**: Models showed artificially high performance (e.g., r=0.913)
3. **Non-generalizable**: Predictions would fail on real peptides/mutations

The curated databases replace synthetic data with **experimentally validated measurements**.

---

## 1. Antimicrobial Peptide Activity Database

**Location**: `carlos_brizuela/scripts/dramp_activity_loader.py`

### Statistics

| Metric | Value |
|--------|-------|
| Total entries | 224 |
| Unique peptide sequences | 155 |
| Pathogen species covered | 4 |
| MIC range | 0.5 - 1000 ug/mL |

### Pathogen Coverage

| Pathogen | Entries | Percentage |
|----------|---------|------------|
| *Escherichia coli* | 105 | 46.9% |
| *Staphylococcus aureus* | 72 | 32.1% |
| *Pseudomonas aeruginosa* | 27 | 12.1% |
| *Acinetobacter baumannii* | 20 | 8.9% |

### Data Sources

1. **APD3** (Antimicrobial Peptide Database)
   - URL: https://aps.unmc.edu/
   - Contains >3,000 validated AMPs

2. **DRAMP** (Data Repository of Antimicrobial Peptides)
   - URL: http://dramp.cpu-bioinfor.org/
   - Comprehensive activity data

3. **DBAASP** (Database of Antimicrobial Activity and Structure of Peptides)
   - URL: https://dbaasp.org/
   - Structure-activity relationships

4. **Primary Literature**
   - Peer-reviewed publications with experimental MIC values

### Peptide Categories

```
CURATED_AMPS organized by source:

1. CLASSIC AMPs (Magainins, Melittin, Cathelicidins, Cecropins, Defensins)
2. SHORT CATIONIC PEPTIDES (Indolicidin, Bactenecin, Protegrin)
3. CLINICAL/PHARMACEUTICAL (Pexiganan, Nisin, Daptomycin, Colistin)
4. AMPHIBIAN AMPs (Brevinin, Temporin, Esculentin, Buforin)
5. MARINE AMPs (Pleurocidin, Piscidin, Tachyplesin)
6. INSECT/ARTHROPOD AMPs (Mastoparan, Apidaecin, Drosocin)
7. MAMMALIAN AMPs (Dermcidin, Cathelicidin-BF, PR-39)
8. DESIGNED/ENGINEERED (WLBU2, MSI-78, LK-peptide)
9. VENOMS (Spider, Snake, Wasp, Scorpion)
10. PROLINE-RICH PEPTIDES (Bac5, Bac7, Oncocin)
11. BACTERIOCINS (Pediocin, Enterocin, Nisin)
```

### Data Format

```python
# Each entry: (name, sequence, target_organism, mic_ug_ml)
CURATED_AMPS = [
    ("Magainin 2", "GIGKFLHSAKKFGKAFVGEIMNS", "Escherichia coli", 10.0),
    ("Melittin", "GIGAVLKVLTTGLPALISWIKRKRQQ", "Staphylococcus aureus", 2.0),
    # ... 222 more entries
]
```

### Usage

```python
from carlos_brizuela.scripts.dramp_activity_loader import DRAMPLoader

loader = DRAMPLoader()
db = loader.generate_curated_database()

print(f"Total records: {len(db.records)}")
# Training data
X, y = db.get_training_data(target="Escherichia coli")
```

---

## 2. Protein Stability (DDG) Database

**Location**: `jose_colbes/scripts/protherm_ddg_loader.py`

### Statistics

| Metric | Value |
|--------|-------|
| Total mutations | 219 |
| Proteins covered | 17 |
| DDG range | -1.2 to 5.5 kcal/mol |
| Mean DDG | 1.69 kcal/mol |

### Protein Coverage

| Protein | PDB ID | Mutations | Notes |
|---------|--------|-----------|-------|
| T4 Lysozyme | 1L63 | 37 | Most studied |
| Barnase | 1BNI | 31 | RNase from *B. amyloliquefaciens* |
| Chymotrypsin Inhibitor 2 | 2CI2 | 20 | Serine protease inhibitor |
| Protein G (GB1) | 1PGA | 14 | IgG-binding domain |
| Staphylococcal Nuclease | 1STN | 13 | Classic folding model |
| Myoglobin | 1MBN | 13 | Oxygen carrier |
| SH3 Domain | 1SHG | 12 | Signaling domain |
| Cold Shock Protein | 1CSP | 12 | Thermostability studies |
| Lambda Repressor | 1LMB | 11 | DNA-binding protein |
| Hen Egg Lysozyme | 1HEL | 10 | Antimicrobial enzyme |
| Tenascin (FNIII) | 1TEN | 8 | Extracellular matrix |
| Ribonuclease A | 7RSA | 8 | Classic enzyme |
| Ubiquitin | 1UBQ | 7 | Regulatory protein |
| Thioredoxin | 1XOA | 7 | Redox protein |
| Cytochrome C | 1HRC | 7 | Electron transport |
| RNase H | 1RN1 | 6 | RNA processing |
| CheY | 3CHY | 3 | Chemotaxis regulator |

### Data Sources

1. **ProTherm**
   - URL: https://web.iitm.ac.in/bioinfo2/prothermdb/
   - Comprehensive thermodynamic database

2. **ThermoMutDB**
   - Curated mutation effects

3. **Primary Literature**
   - Original experimental publications

### Mutation Categories

```
CURATED_MUTATIONS organized by type:

1. DESTABILIZING (DDG > 0)
   - Large hydrophobic to alanine (high DDG)
   - Charged to hydrophobic (moderate DDG)
   - Aromatic substitutions

2. STABILIZING (DDG < 0)
   - Glycine to alanine (helix formation)
   - Cavity-filling mutations

3. NEUTRAL (DDG ~ 0)
   - Conservative substitutions (K->R, E->D)
   - Surface mutations

4. SPECIAL CASES
   - Proline mutations (helix breakers)
   - Disulfide-affecting mutations
```

### Data Format

```python
# Each entry: (pdb_id, chain, position, wild_type, mutant, ddg, secondary_structure)
CURATED_MUTATIONS = [
    ("1L63", "A", 3, "M", "A", 1.1, "H"),   # Helix
    ("1L63", "A", 133, "F", "A", 4.5, "E"), # Sheet
    ("1L63", "A", 9, "G", "A", -0.8, "H"),  # Stabilizing
    # ... 216 more entries
]

# Secondary structure codes:
# H = Helix
# E = Sheet (Extended)
# C = Coil
```

### Usage

```python
from jose_colbes.scripts.protherm_ddg_loader import ProThermLoader

loader = ProThermLoader()
db = loader.generate_curated_database()

print(f"Total mutations: {len(db.records)}")
# Training data
X, y, feature_names = db.get_training_data()
```

---

## 3. Hemolysis Training Data

**Location**: `shared/hemolysis_predictor.py`

### Statistics

| Metric | Value |
|--------|-------|
| Total peptides | 40 |
| Hemolytic (HC50 < 50 uM) | 12 |
| Non-hemolytic | 28 |
| HC50 range | 1.5 - 1000 uM |

### Data Categories

```
CURATED_HEMOLYSIS_DATA:

1. HIGHLY HEMOLYTIC (HC50 < 50 uM)
   - Melittin (1.5 uM) - bee venom
   - Mastoparan (15 uM) - wasp venom
   - Gramicidin S (12 uM) - cyclic peptide
   - Protegrin-1 (35 uM) - pig leukocytes

2. MODERATELY HEMOLYTIC (50-200 uM)
   - Magainin 2 (100 uM)
   - Indolicidin (80 uM)
   - LL-37 (150 uM)

3. LOW/NO HEMOLYSIS (HC50 > 200 uM)
   - Pexiganan (250 uM)
   - Cecropin A (400 uM)
   - Buforin II (500 uM)
   - Proline-rich peptides (>800 uM)
```

### Data Sources

1. **HemoPI Database**
   - Hemolytic peptide information

2. **DBAASP**
   - Structure-activity including hemolysis

3. **Primary Literature**
   - HC50 measurements from publications

---

## Data Quality Assurance

### Selection Criteria

All curated data meets these criteria:

1. **Experimental validation**: Only published experimental values
2. **Standard conditions**: Comparable assay conditions
3. **Multiple sources**: Cross-validated when possible
4. **Clear methodology**: Documented experimental methods

### Excluded Data

The following were NOT included:

1. Computationally predicted values
2. Indirect measurements
3. Non-standard conditions
4. Single unreplicated studies
5. Extreme outliers

### Cross-Validation Results

Training with curated data shows realistic performance:

| Model | Curated CV r | Previous (Synthetic) r |
|-------|--------------|----------------------|
| E. coli activity | 0.45-0.65 | 0.91 (overfit) |
| S. aureus activity | 0.40-0.60 | 0.85 (overfit) |
| DDG prediction | 0.50-0.70 | 0.91 (overfit) |

Lower correlations with real data are expected and indicate honest model performance.

---

## Updating the Databases

### Adding New AMP Entries

```python
# In dramp_activity_loader.py, add to CURATED_AMPS:
CURATED_AMPS = [
    # ... existing entries ...

    # New entry format:
    ("Peptide Name", "SEQUENCE", "Target Organism", mic_value),
]
```

### Adding New DDG Entries

```python
# In protherm_ddg_loader.py, add to CURATED_MUTATIONS:
CURATED_MUTATIONS = [
    # ... existing entries ...

    # New entry format:
    ("PDB_ID", "Chain", position, "WT", "Mut", ddg_value, "SS"),
    # SS = H (helix), E (sheet), C (coil)
]
```

### Validation After Updates

After adding new data, run:

```bash
python -m pytest deliverables/shared/tests/ -v
```

---

## References

### AMP Databases

1. Wang, G., Li, X., & Wang, Z. (2016). APD3: the antimicrobial peptide database as a tool for research and education. Nucleic acids research, 44(D1), D1087-D1093.

2. Kang, X., et al. (2019). DRAMP 2.0, an updated data repository of antimicrobial peptides. Scientific data, 6(1), 1-10.

3. Pirtskhalava, M., et al. (2021). DBAASP v3: database of antimicrobial/cytotoxic activity and structure of peptides. Nucleic acids research, 49(D1), D288-D297.

### Stability Databases

4. Nikam, R., et al. (2021). ProThermDB: thermodynamic database for proteins and mutants. Nucleic acids research, 49(D1), D420-D427.

5. Pucci, F., & Rooman, M. (2016). Improved insights into protein thermal stability. Journal of molecular biology, 428(7), 1318-1330.

### Hemolysis Data

6. Gautam, A., et al. (2014). Hemolytik: a database of experimentally determined hemolytic and non-hemolytic peptides. Nucleic acids research, 42(D1), D444-D449.
