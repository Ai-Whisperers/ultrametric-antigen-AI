# Reproducibility Guide

## Complete Instructions for Reproducing All Analyses

**Version:** 1.0
**Last Updated:** December 25, 2025
**Estimated Time:** 30-45 minutes for complete pipeline

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Data Acquisition](#3-data-acquisition)
4. [Running the Analysis Pipeline](#4-running-the-analysis-pipeline)
5. [Verifying Results](#5-verifying-results)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Prerequisites

### 1.1 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Storage | 2 GB free | 5 GB free |
| CPU | 4 cores | 8 cores |
| GPU | Not required | Optional (CUDA) |

### 1.2 Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.11+ | Runtime |
| pip | 23.0+ | Package management |
| Git | 2.30+ | Version control |

### 1.3 Operating System

- Windows 10/11 (tested)
- Linux (Ubuntu 20.04+, should work)
- macOS (12+, should work)

---

## 2. Environment Setup

### 2.1 Clone the Repository

```bash
git clone https://github.com/[your-org]/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics
```

### 2.2 Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2.3 Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Required packages:**
```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
torch>=2.0.0
biopython>=1.81
pyarrow>=12.0.0
```

### 2.4 Verify Installation

```bash
python -c "import numpy, pandas, scipy, sklearn, matplotlib, torch; print('All packages installed successfully')"
```

---

## 3. Data Acquisition

### 3.1 Data Locations

All datasets should be in these locations:

```
research/datasets/
├── stanford_hivdb_pi.txt      # Stanford PI resistance
├── stanford_hivdb_nrti.txt    # Stanford NRTI resistance
├── stanford_hivdb_nnrti.txt   # Stanford NNRTI resistance
├── stanford_hivdb_ini.txt     # Stanford INI resistance
├── ctl_summary.csv            # LANL CTL epitopes
└── catnap_assay.txt           # CATNAP neutralization

data/external/huggingface/HIV_V3_coreceptor/
├── train.parquet              # V3 tropism training data
├── test.parquet               # V3 tropism test data
└── validation.parquet         # V3 tropism validation data
```

### 3.2 Downloading Datasets

**Stanford HIVDB:**
1. Visit: https://hivdb.stanford.edu/download/GenoPhenoDatasets/
2. Download resistance datasets for PI, NRTI, NNRTI, INI
3. Place in `research/datasets/`

**LANL CTL Database:**
1. Visit: https://www.hiv.lanl.gov/content/immunology/tables/ctl_summary.html
2. Download as CSV
3. Place in `research/datasets/ctl_summary.csv`

**CATNAP:**
1. Visit: https://www.hiv.lanl.gov/components/sequence/HIV/neutralization/
2. Download assay data
3. Place in `research/datasets/catnap_assay.txt`

**V3 Coreceptor (Hugging Face):**
```python
from datasets import load_dataset
dataset = load_dataset("tnhaider/HIV_V3_coreceptor")
# Save to parquet files
```

### 3.3 Verify Data Files

```bash
python -c "
from scripts.unified_data_loader import get_dataset_summary
summary = get_dataset_summary()
print(summary)
"
```

**Expected Output:**
```
Stanford HIVDB: 7,154 records
LANL CTL: 2,115 epitopes
CATNAP: 189,879 records
V3 Coreceptor: 2,932 sequences
```

---

## 4. Running the Analysis Pipeline

### 4.1 Navigate to Scripts Directory

```bash
cd research/bioinformatics/codon_encoder_research/hiv/scripts
```

### 4.2 Option A: Run Complete Pipeline

```bash
python run_complete_analysis.py
```

**This runs all analyses sequentially:**
1. Stanford drug resistance analysis (~5 min)
2. CTL escape analysis (~3 min)
3. CATNAP neutralization analysis (~10 min)
4. Tropism switching analysis (~2 min)
5. Cross-dataset integration (~5 min)

**Total time: ~25 minutes**

### 4.3 Option B: Run Individual Analyses

**Drug Resistance:**
```bash
python analyze_stanford_resistance.py
```

**CTL Escape:**
```bash
python analyze_ctl_escape_expanded.py
```

**Antibody Neutralization:**
```bash
python analyze_catnap_neutralization.py
```

**Tropism:**
```bash
python analyze_tropism_switching.py
```

**Integration:**
```bash
python cross_dataset_integration.py
```

### 4.4 Monitor Progress

Each script outputs progress to console:
```
======================================================================
Stanford Drug Resistance Analysis
======================================================================
Loading datasets...
  PI: 2,171 records
  NRTI: 1,867 records
  ...
Extracting mutations...
  90,269 mutations extracted
Calculating geometric features...
  ...
Analysis complete!
```

---

## 5. Verifying Results

### 5.1 Check Output Files

```bash
# List all generated files
find ../results -name "*.csv" -o -name "*.png" -o -name "*.md" | wc -l
# Expected: ~35 files
```

### 5.2 Verify Key Metrics

**Drug Resistance:**
```python
import pandas as pd
df = pd.read_csv('../results/stanford_resistance/mutation_distances.csv')
print(f"Mutations: {len(df)}")  # Expected: ~90,269
print(f"Mean distance: {df['hyperbolic_distance'].mean():.3f}")  # Expected: ~0.5
```

**Tropism:**
```python
df = pd.read_csv('../results/tropism/position_importance.csv')
top_pos = df.loc[df['separation'].idxmax(), 'position']
print(f"Top position: {top_pos}")  # Expected: 22
```

**Integration:**
```python
df = pd.read_csv('../results/integrated/vaccine_targets.csv')
print(f"Safe targets: {len(df[~df['resistance_overlap']])}")  # Expected: 328
```

### 5.3 Compare to Reference Values

| Metric | Expected | Tolerance |
|--------|----------|-----------|
| Total mutations | 90,269 | ±100 |
| CTL epitopes | 2,115 | ±5 |
| CATNAP records | 189,879 | ±100 |
| V3 sequences | 2,932 | ±10 |
| Resistance-distance r | 0.34-0.41 | ±0.02 |
| Tropism accuracy | 0.85 | ±0.02 |
| Safe vaccine targets | 328 | ±5 |

---

## 6. Troubleshooting

### 6.1 Common Issues

**Issue: ModuleNotFoundError**
```
Solution: pip install [missing_module]
```

**Issue: numpy.dtype size changed**
```
Solution: pip install --upgrade numpy pandas pyarrow
```

**Issue: FileNotFoundError for datasets**
```
Solution: Verify data files are in correct locations (Section 3.1)
```

**Issue: Memory error during CATNAP analysis**
```
Solution: Close other applications, or use chunk processing
```

**Issue: KeyError 'tropism_label'**
```
Solution: V3 data format changed; use latest scripts that handle boolean columns
```

### 6.2 Getting Help

1. Check FAQ document: `documentation/faq/FAQ.md`
2. Review error messages carefully
3. Check data file formats match expected
4. Open issue on GitHub repository

---

## 7. Reproducibility Checklist

Before publishing or sharing results:

- [ ] All data files present and correct size
- [ ] Virtual environment created fresh
- [ ] Dependencies installed from requirements.txt
- [ ] Complete pipeline runs without errors
- [ ] Key metrics match reference values
- [ ] All output files generated
- [ ] Documentation reviewed and up-to-date

---

## 8. Version Information

```bash
# Record versions for reproducibility
python -c "
import numpy, pandas, scipy, sklearn, torch
print(f'numpy: {numpy.__version__}')
print(f'pandas: {pandas.__version__}')
print(f'scipy: {scipy.__version__}')
print(f'sklearn: {sklearn.__version__}')
print(f'torch: {torch.__version__}')
"
```

**Tested Versions:**
```
numpy: 1.24.3
pandas: 2.0.3
scipy: 1.11.4
sklearn: 1.3.2
torch: 2.1.0
```

---

## 9. Archiving Results

For long-term reproducibility:

```bash
# Create dated archive
DATE=$(date +%Y%m%d)
tar -czvf hiv_analysis_results_${DATE}.tar.gz ../results/
```

Include with archive:
- `requirements.txt`
- This reproducibility guide
- Git commit hash: `git rev-parse HEAD`

---

**Document Version:** 1.0
**Last Updated:** December 25, 2025
