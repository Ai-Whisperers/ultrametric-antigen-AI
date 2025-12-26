# Quick Start Guide

## Get Up and Running in 10 Minutes

**Version:** 1.0
**Last Updated:** December 25, 2025

---

## TL;DR

```bash
# 1. Navigate to scripts
cd research/bioinformatics/codon_encoder_research/hiv/scripts

# 2. Run everything
python run_complete_analysis.py

# 3. View results
ls ../results/
```

---

## For Impatient Researchers

### What This Does
Analyzes 200,000+ HIV records to find:
- Drug resistance geometric patterns
- Immune escape landscapes
- Antibody neutralization signatures
- Tropism determinants
- 328 vaccine targets

### Time Required
- Complete pipeline: ~25 minutes
- Individual analyses: 2-10 minutes each

### Output Location
All results in `results/` subdirectories

---

## Step-by-Step Quick Start

### Step 1: Check Prerequisites

```bash
python --version  # Need 3.11+
pip --version     # Need 23.0+
```

### Step 2: Verify Data Files

```bash
ls research/datasets/
# Should see: stanford_hivdb_*.txt, ctl_summary.csv, catnap_assay.txt
```

### Step 3: Run Analysis

**Option A - Everything at once:**
```bash
cd research/bioinformatics/codon_encoder_research/hiv/scripts
python run_complete_analysis.py
```

**Option B - Just what you need:**
```bash
# Drug resistance only
python analyze_stanford_resistance.py

# CTL epitopes only
python analyze_ctl_escape_expanded.py

# Antibody neutralization only
python analyze_catnap_neutralization.py

# Tropism only
python analyze_tropism_switching.py

# Integration (requires others first)
python cross_dataset_integration.py
```

### Step 4: Check Results

```bash
# View generated reports
cat ../results/stanford_resistance/ANALYSIS_REPORT.md
cat ../results/integrated/INTEGRATION_REPORT.md

# Check key numbers
head ../results/integrated/vaccine_targets.csv
```

---

## What You Get

### Reports (Markdown)
- `ANALYSIS_REPORT.md` - Drug resistance summary
- `CTL_ANALYSIS_REPORT.md` - Immune escape summary
- `NEUTRALIZATION_REPORT.md` - Antibody summary
- `TROPISM_REPORT.md` - Tropism summary
- `INTEGRATION_REPORT.md` - Combined findings

### Data Files (CSV)
- `mutation_distances.csv` - 90,269 mutation geometric features
- `epitope_data.csv` - 2,115 epitope analyses
- `vaccine_targets.csv` - 387 ranked targets

### Visualizations (PNG)
- 20 publication-ready figures
- See `documentation/figures_guide/FIGURES_GUIDE.md`

---

## Quick Answers

**Q: Something failed. What now?**
```bash
pip install --upgrade numpy pandas pyarrow
# Then try again
```

**Q: Where are the vaccine targets?**
```bash
cat ../results/integrated/vaccine_targets.csv | head -20
```

**Q: What's the tropism accuracy?**
```
85% (AUC = 0.86)
See ../results/tropism/TROPISM_REPORT.md
```

**Q: How do I cite this?**
```
HIV P-adic Hyperbolic Analysis, v1.0, 2025
See documentation/EXECUTIVE_SUMMARY.md
```

---

## Next Steps

1. **Understand results:** Read `EXECUTIVE_SUMMARY.md`
2. **Deep dive:** Read specific findings in `documentation/findings/`
3. **Check novelty:** Read `NOVELTY_ASSESSMENT.md`
4. **Reproduce:** Follow `REPRODUCIBILITY_GUIDE.md`
5. **Extend:** See `FUTURE_DIRECTIONS.md`

---

## File Structure

```
hiv/
├── scripts/           # Run these
├── results/           # Output goes here
└── documentation/     # Read these
    ├── EXECUTIVE_SUMMARY.md      ← Start here
    ├── NOVELTY_ASSESSMENT.md     ← What's new
    ├── findings/                 ← Detailed results
    ├── methodology/              ← How it works
    └── quickstart/               ← You are here
```

---

**Need more detail? See `REPRODUCIBILITY_GUIDE.md`**

**Have questions? See `FAQ.md`**

---

**Document Version:** 1.0
**Last Updated:** December 25, 2025
