# Bioinformatics Knowledge Base

> **Goal:** Validate the geometric theory on real biological data.

**Last Updated:** December 26, 2025

---

## Quick Links

| Research Area | Documents | Records | Key Findings |
|---------------|-----------|---------|--------------|
| [**HIV P-adic Analysis**](HIV_PADIC_ANALYSIS/) | 27 docs | 202,085 | Vaccine targets, MDR screening |
| [Codon Encoder Research](CODON_ENCODER_RESEARCH/) | Various | - | Multi-disease analysis |
| [P-adic Genomics](P_ADIC_GENOMICS/) | - | - | Fractal dimension ~1.58D |

---

## Code vs. Knowledge

- **Active Research**: All executable code, notebooks, and raw data are in `research/bioinformatics/`
- **Knowledge Base**: This directory contains **Summaries, Theory, and Findings** derived from that code

---

## Subdirectories

### 1. `HIV_PADIC_ANALYSIS/` (NEW - Comprehensive)

**27 documents organized into 6 sections:**

| Section | Contents |
|---------|----------|
| `00_OVERVIEW/` | Executive summary, theory, novelty assessment |
| `01_FINDINGS/` | Drug resistance, CTL escape, bnAbs, tropism, integration |
| `02_METHODOLOGY/` | Methods, API, statistics, benchmarks, architecture |
| `03_REFERENCE/` | Literature (150 papers), glossary, data dictionary |
| `04_IMPLEMENTATION/` | Improvement plans, roadmaps (75+ features) |
| `05_SUPPLEMENTARY/` | Quick start, FAQ, reproducibility, limitations |

**Key Results:**
- Top vaccine candidate: **TPQDLNTML** (Gag, priority: 0.970)
- P-adic geometry validated: Spearman r = 0.8339
- MDR marker: L63P (79.5% of MDR sequences)
- 387 vaccine targets ranked by evolutionary stability

### 2. `CODON_ENCODER_RESEARCH/`

Disease-specific codon analysis:
- **HIV**: Glycan shield and codon optimization
- **SARS-CoV-2**: Spike protein evolution
- **Neurodegeneration**: Prion-like mechanisms
- **Rheumatoid Arthritis**: HLA geometry

### 3. `P_ADIC_GENOMICS/`

- **Focus**: Whole genomes
- **Key Finding**: Fractal dimension of ~1.58D

### 4. `SPECTRAL_ANALYSIS_OVER_MODELS/`

- **Focus**: Eigenvalue spectrum analysis (Zipf's Law)

### 5. `GENETIC_CODE/`

- Fundamental genetic code theory and mappings

---

## How to Run Experiments

```bash
# Navigate to research directory
cd research/bioinformatics/

# Run HIV analysis pipelines
python ../../scripts/clinical_applications.py
python ../../scripts/research_discoveries.py
python ../../scripts/analyze_all_datasets.py
```

See individual README files in subdirectories for specific instructions.

---

## Related Resources

| Resource | Location |
|----------|----------|
| Clinical Decision Support | `results/clinical_applications/` |
| Research Discoveries | `results/research_discoveries/` |
| Raw HIV Results | `research/bioinformatics/codon_encoder_research/hiv/results/` |
| Analysis Scripts | `scripts/` |

---

_See [HIV_PADIC_ANALYSIS/README.md](HIV_PADIC_ANALYSIS/README.md) for comprehensive HIV documentation._
