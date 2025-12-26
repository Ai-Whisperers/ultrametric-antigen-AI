# Frequently Asked Questions

## Common Questions and Troubleshooting

**Version:** 1.0
**Last Updated:** December 25, 2025

---

## Table of Contents

1. [General Questions](#1-general-questions)
2. [Technical Questions](#2-technical-questions)
3. [Interpretation Questions](#3-interpretation-questions)
4. [Troubleshooting](#4-troubleshooting)
5. [Data Questions](#5-data-questions)

---

## 1. General Questions

### Q1.1: What is p-adic hyperbolic geometry?

**A:** P-adic geometry uses prime number divisibility to define distance. For our 3-adic system:
- Numbers divisible by higher powers of 3 are "closer" to each other
- This creates a hierarchical, tree-like structure
- Combined with hyperbolic geometry, it naturally represents evolutionary relationships

**Simple analogy:** Like organizing a library where books on similar topics are shelved together, but the similarity is based on a mathematical principle that respects biological hierarchy.

---

### Q1.2: Why use this approach for HIV?

**A:** Three main reasons:

1. **Codon structure:** The genetic code is triplet-based (codons), naturally fitting a base-3 (ternary) system

2. **Evolutionary hierarchy:** HIV mutations form tree-like evolutionary relationships that embed well in hyperbolic space

3. **Constraint representation:** The geometry naturally captures that some mutations are "harder" (more constrained) than others

---

### Q1.3: Is this a clinical tool?

**A:** **No.** This is a research framework. Key differences from clinical tools:

| Clinical Tool | Our Framework |
|---------------|---------------|
| Validated in trials | Research validation |
| Regulatory approved | Not approved |
| Real-time use | Batch analysis |
| Definitive answers | Probabilistic insights |

Use Stanford HIVdb, geno2pheno, etc. for clinical decisions.

---

### Q1.4: What's novel about this work?

**A:** See `NOVELTY_ASSESSMENT.md` for complete details. Key novel findings:

1. P-adic hyperbolic geometry applied to HIV (first time)
2. Position 22 as top tropism determinant
3. Breadth-centrality correlation for bnAbs
4. 328 resistance-free vaccine targets
5. Trade-off scoring for dual-pressure positions

---

### Q1.5: Who is this for?

**A:** Primary audiences:

- **HIV researchers:** Understanding evolution, resistance, immunity
- **Vaccine developers:** Identifying optimal targets
- **Computational biologists:** Novel geometric methods
- **Epidemiologists:** Population-level patterns

---

## 2. Technical Questions

### Q2.1: What Python version is required?

**A:** Python 3.11 or higher. Earlier versions may work but are not tested.

```bash
python --version  # Should be 3.11+
```

---

### Q2.2: What are the hardware requirements?

**A:**
| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| Storage | 2 GB | 5 GB |
| CPU | 4 cores | 8 cores |
| GPU | Not needed | Optional |

---

### Q2.3: How long does the analysis take?

**A:** Approximate times:

| Analysis | Time |
|----------|------|
| Drug resistance | 5 min |
| CTL escape | 3 min |
| Neutralization | 10 min |
| Tropism | 2 min |
| Integration | 5 min |
| **Total** | **25 min** |

---

### Q2.4: Can I run analyses in parallel?

**A:** Yes, the individual analyses are independent:

```bash
# Run in separate terminals
python analyze_stanford_resistance.py &
python analyze_ctl_escape_expanded.py &
python analyze_catnap_neutralization.py &
python analyze_tropism_switching.py &
wait
python cross_dataset_integration.py  # Must run last
```

---

### Q2.5: What's the 16-dimensional embedding?

**A:** Each codon is represented as a 16-dimensional vector in the Poincaré ball:
- Dimensions 1-4: 3-adic valuation features
- Dimensions 5-12: Positional encoding
- Dimensions 13-16: Biochemical properties

The choice of 16 is pragmatic; it captures key relationships without excessive complexity.

---

## 3. Interpretation Questions

### Q3.1: What does "radial position" mean?

**A:** Distance from the origin in hyperbolic space:

| Radial Position | Interpretation |
|-----------------|----------------|
| Low (0.3-0.5) | Highly constrained, essential function |
| Medium (0.5-0.7) | Moderately constrained |
| High (0.7-0.9) | Variable, tolerated changes |

**Rule of thumb:** Central = important, peripheral = variable.

---

### Q3.2: What is "escape velocity"?

**A:** A measure of how easily an epitope can accumulate mutations to escape immune recognition:

- **Low velocity:** Escape is difficult (constrained epitope)
- **High velocity:** Escape is easy (variable epitope)

Calculated from the spread of embeddings across the epitope positions.

---

### Q3.3: How do I interpret the trade-off score?

**A:** Higher score = position under stronger dual pressure:

| Score Range | Interpretation |
|-------------|----------------|
| < 2.0 | Low trade-off |
| 2.0 - 4.0 | Moderate trade-off |
| > 4.0 | High trade-off (clinically important) |

High-scoring positions may confer both drug resistance AND immune escape.

---

### Q3.4: What does "separation score" mean for tropism?

**A:** The hyperbolic distance between R5 and X4 sequence centroids at a specific position:

- **High separation:** Position strongly distinguishes tropism
- **Low separation:** Position doesn't affect tropism

Position 22 has the highest separation (0.591).

---

### Q3.5: Are the vaccine targets validated?

**A:** **Computationally validated only.** They meet these criteria:

- Broad HLA restriction
- Low escape velocity
- No drug resistance overlap
- High conservation

**Immunological validation required** before use in vaccines.

---

## 4. Troubleshooting

### T4.1: ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'xyz'
```

**Solution:**
```bash
pip install xyz
# Or install all requirements
pip install -r requirements.txt
```

---

### T4.2: numpy.dtype size changed

**Error:**
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

**Solution:**
```bash
pip install --upgrade numpy pandas pyarrow
```

This occurs when package versions are mismatched.

---

### T4.3: FileNotFoundError for data files

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'stanford_hivdb_pi.txt'
```

**Solution:**
1. Check that data files exist in `research/datasets/`
2. Download missing files (see REPRODUCIBILITY_GUIDE.md)
3. Verify file names match expected names

---

### T4.4: KeyError: 'tropism_label'

**Error:**
```
KeyError: 'tropism_label'
```

**Solution:**
This was fixed in the current version. The V3 data has boolean columns `CCR5` and `CXCR4`, not a `tropism_label` column. Update to the latest scripts.

---

### T4.5: Memory Error during CATNAP analysis

**Error:**
```
MemoryError: Unable to allocate...
```

**Solution:**
1. Close other applications
2. Increase virtual memory/swap
3. Process in chunks:
```python
# In analyze_catnap_neutralization.py
CHUNK_SIZE = 50000
for chunk in pd.read_csv(file, chunksize=CHUNK_SIZE):
    process(chunk)
```

---

### T4.6: Plots not displaying

**Error:**
Matplotlib windows don't appear or close immediately.

**Solution:**
```python
import matplotlib
matplotlib.use('Agg')  # For saving without display
# Or
matplotlib.use('TkAgg')  # For interactive display
```

---

### T4.7: Slow performance

**Symptoms:**
Analysis takes much longer than expected.

**Solutions:**
1. Check available RAM (need 8GB+)
2. Close other applications
3. Run individual analyses instead of full pipeline
4. Check disk speed (SSD recommended)

---

### T4.8: Different results on re-run

**Symptoms:**
Slight differences in numerical results between runs.

**Explanation:**
Some operations (e.g., Random Forest) have random components.

**Solution:**
Set random seeds for reproducibility:
```python
import numpy as np
np.random.seed(42)
```

---

## 5. Data Questions

### D5.1: Where do the datasets come from?

**A:**

| Dataset | Source | URL |
|---------|--------|-----|
| Stanford HIVDB | Stanford University | https://hivdb.stanford.edu |
| LANL CTL | Los Alamos | https://www.hiv.lanl.gov |
| CATNAP | Los Alamos | https://www.hiv.lanl.gov/content/sequence/CATNAP |
| V3 Coreceptor | Hugging Face | https://huggingface.co/datasets/tnhaider/HIV_V3_coreceptor |

---

### D5.2: How current is the data?

**A:** Data downloaded December 2025. External databases update periodically:
- Stanford HIVDB: Quarterly updates
- CATNAP: Ongoing additions
- LANL: Regular updates

---

### D5.3: Can I use my own data?

**A:** Yes, with formatting:

**For sequences:**
```python
# Format: CSV or tab-delimited
# Columns: SeqID, Sequence (or position columns)
my_data = pd.read_csv('my_sequences.csv')
```

**For mutations:**
```python
# Format: Standard notation
# Example: M184V, K103N
mutations = ['M184V', 'K103N', 'Y181C']
```

---

### D5.4: Why is subtype B predominant?

**A:** Historical and geographic reasons:
- Most HIV drug resistance research in North America/Europe
- Subtype B predominant in these regions
- Earlier establishment of research cohorts

**Limitation:** Results may not generalize to subtypes A, C, D, etc.

---

### D5.5: Is patient data de-identified?

**A:** Yes. All data used is:
- Publicly available from cited sources
- Already de-identified at source
- No individual patient information retained

---

## 6. Citation and Licensing

### Q6.1: How do I cite this work?

**A:**
```
HIV P-adic Hyperbolic Analysis Pipeline, Version 1.0
Ternary VAE Bioinformatics Research Group
December 2025

See also: EXECUTIVE_SUMMARY.md for detailed citation
```

---

### Q6.2: Can I use this for commercial purposes?

**A:** Check the repository license. Generally:
- Academic use: Freely permitted
- Commercial use: Contact authors
- Data sources: Have their own licenses

---

### Q6.3: Can I modify the code?

**A:** Yes, following license terms:
- Credit original work
- Share modifications openly
- Document changes

---

## 7. Getting Help

### Q7.1: Where can I ask questions?

**A:** Options:
1. GitHub Issues (preferred for technical questions)
2. Contact information in repository
3. Scientific collaborators for research questions

---

### Q7.2: How do I report bugs?

**A:** Open a GitHub Issue with:
1. Description of the problem
2. Steps to reproduce
3. Error message (full traceback)
4. Python/package versions
5. Operating system

---

### Q7.3: Can I contribute?

**A:** Yes! Contributions welcome:
1. Bug fixes
2. Documentation improvements
3. New analyses
4. Performance optimizations

See contribution guidelines in repository.

---

## Quick Reference Card

```
COMMON COMMANDS:
  python run_complete_analysis.py  # Run everything
  python analyze_tropism_switching.py  # Single analysis

KEY FILES:
  results/integrated/vaccine_targets.csv  # Vaccine targets
  results/tropism/position_importance.csv  # Tropism positions
  documentation/EXECUTIVE_SUMMARY.md  # Main summary

QUICK FIXES:
  pip install --upgrade numpy pandas pyarrow  # Version issues
  Check research/datasets/ exists  # Missing data

KEY NUMBERS:
  7,154 - Stanford records
  2,115 - CTL epitopes
  189,879 - CATNAP records
  2,932 - V3 sequences
  328 - Safe vaccine targets
  85% - Tropism accuracy
  0.41 - Resistance correlation
```

---

**Document Version:** 1.0
**Last Updated:** December 25, 2025
