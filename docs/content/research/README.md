# Research

> **Research findings and disease-specific analysis.**

---

## Research Domains

| Domain | Status | Key Results |
|:-------|:-------|:------------|
| [HIV](hiv/README.md) | Active | 200K+ sequences, 23 drugs, r=0.89 |
| [Multi-Disease](multi-disease/README.md) | Active | 11 diseases unified |
| SARS-CoV-2 | Active | Spike escape, Paxlovid |
| Tuberculosis | Active | MDR/XDR classification |
| Influenza | Exploratory | NAI resistance |

---

## Key Achievements

### HIV Analysis

- **Drug Resistance**: +0.89 Spearman correlation across 23 ARVs
- **Vaccine Targets**: 387 candidates ranked by stability
- **Glycan Shield**: 7 sentinel positions identified
- **Tropism Prediction**: 85% accuracy on CCR5/CXCR4

### Multi-Disease Platform

- **Unified Framework**: 11 diseases with consistent API
- **Transfer Learning**: Cross-disease knowledge sharing
- **Uncertainty**: Clinical-grade confidence intervals

---

## Research Outputs

### Publications

See [results/research_discoveries/](../../../results/research_discoveries/)

### Clinical Applications

See [results/clinical_applications/](../../../results/clinical_applications/)

---

## Quick Start

```python
from src.diseases import HIVAnalyzer, TuberculosisAnalyzer

# HIV analysis
hiv = HIVAnalyzer()
results = hiv.analyze(sequences, drugs=["3TC", "EFV", "DTG"])

# TB analysis
tb = TuberculosisAnalyzer()
results = tb.analyze(sequences)
print(results["mdr_classification"])  # DS-TB, MDR-TB, pre-XDR-TB, or XDR-TB
```

---

_Last updated: 2025-12-28_
