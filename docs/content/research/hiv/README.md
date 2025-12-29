# HIV Research

> **Comprehensive HIV drug resistance prediction and vaccine target analysis.**

---

## Overview

Analysis of **200,000+ HIV sequences** using p-adic hyperbolic geometry.

### Key Results

| Metric | Value |
|:-------|:------|
| Drug classes analyzed | 4 (PI, NRTI, NNRTI, INI) |
| Total drugs | 23 |
| Mean Spearman correlation | +0.890 |
| Best drug (3TC) | +0.981 |
| Vaccine candidates | 387 ranked |
| Glycan sentinels | 7 positions |

---

## Drug Resistance Prediction

### Performance by Drug Class

| Drug Class | Avg Correlation | Best Drug | Architecture |
|------------|-----------------|-----------|--------------|
| PI (8 drugs) | +0.928 | LPV (+0.956) | All perform well |
| NRTI (6 drugs) | +0.887 | 3TC (+0.981) | Attention VAE |
| NNRTI (5 drugs) | +0.853 | NVP (+0.959) | Transformer VAE |
| INI (4 drugs) | +0.863 | EVG (+0.963) | Transformer VAE |

### Key Innovations

1. **P-adic ranking loss**: +0.6 correlation improvement over MSE
2. **Attention analysis**: 65-70% F1 match with known mutations
3. **Cross-resistance**: Captures TAM patterns, M184V resensitization

---

## Usage

```python
from src.diseases import HIVAnalyzer, HIVDrug

analyzer = HIVAnalyzer()

# Analyze sequences
results = analyzer.analyze(
    sequences=["PRTCLKVYLVGRSM..."],
    drugs=[HIVDrug.LAMIVUDINE, HIVDrug.EFAVIRENZ, HIVDrug.DOLUTEGRAVIR],
)

# Access results
for drug, data in results["drug_resistance"].items():
    print(f"{drug}: {data['score']:.2f} ({data['interpretation']})")
```

---

## Clinical Decision Support

| Finding | Value |
|---------|-------|
| Top Vaccine Candidate | TPQDLNTML (Gag, priority: 0.970) |
| MDR High-Risk Sequences | 2,489 (34.8% of screened) |
| Druggable Tat Targets | 247 kinases/receptors |
| MDR Mutations Identified | 1,032 enriched signatures |

---

## Quick Start

```bash
# Run full validation
python scripts/experiments/run_full_validation.py

# Run attention analysis
python scripts/experiments/run_attention_analysis.py

# Test cross-resistance
python scripts/experiments/run_cross_resistance_test.py
```

---

## Data Sources

| Source | Sequences | Description |
|--------|-----------|-------------|
| Stanford HIVDB | 200,000+ | Drug resistance with phenotypes |
| Los Alamos | 500,000+ | Sequence alignments |
| UNAIDS | - | Epidemiological data |

---

## Documentation

- [Executive Summary](../../../../research/bioinformatics/codon_encoder_research/hiv/documentation/EXECUTIVE_SUMMARY.md)
- [Literature Review](../../../../research/bioinformatics/codon_encoder_research/hiv/documentation/LITERATURE_REVIEW.md)
- [Full Module Documentation](../../../../research/bioinformatics/codon_encoder_research/hiv/README.md)

---

_Last updated: 2025-12-28_
