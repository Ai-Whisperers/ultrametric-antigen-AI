# For Clinicians & Virologists

> **Predictions, validation, and clinical integration.**

---

## Overview

Ternary VAE provides **drug resistance predictions** with **confidence intervals** for clinical decision support.

### Supported Diseases

| Disease | Drugs | Status |
|:--------|:------|:-------|
| HIV | 23 ARVs | Validated |
| Tuberculosis | 13 first/second-line | Beta |
| Influenza | NAIs, baloxavir | Beta |
| SARS-CoV-2 | Paxlovid, mAbs | Beta |

---

## Clinical Decision Support

### HIV Example

```
Input: Patient HIV sequence (RT + PR + IN)

Output:
┌─────────────────────────────────────────────────────────┐
│ Drug         │ Resistance │ Confidence │ Recommendation │
├──────────────┼────────────┼────────────┼────────────────┤
│ Lamivudine   │ HIGH (0.92)│ 95% CI     │ AVOID          │
│ Efavirenz    │ MOD  (0.65)│ 85% CI     │ CAUTION        │
│ Dolutegravir │ LOW  (0.12)│ 98% CI     │ RECOMMEND      │
│ Tenofovir    │ LOW  (0.08)│ 96% CI     │ RECOMMEND      │
└─────────────────────────────────────────────────────────┘

Key mutations detected: M184V, K103N
Cross-resistance alert: NRTI TAMs present
```

### Interpretation

| Score | Label | Recommendation |
|:------|:------|:---------------|
| 0.0-0.3 | Susceptible | Drug effective |
| 0.3-0.7 | Intermediate | Consider with caution |
| 0.7-1.0 | Resistant | Avoid if alternatives exist |

### Confidence Levels

- **High confidence** (>90%): Clear prediction, can act
- **Moderate confidence** (70-90%): Consider additional testing
- **Low confidence** (<70%): Expert review recommended

---

## Validation

### HIV Benchmarks

| Metric | Value | Benchmark |
|:-------|:------|:----------|
| Spearman correlation | 0.89 | >0.8 clinical grade |
| Known mutation detection | 65-70% F1 | Attention analysis |
| Cross-resistance patterns | Validated | TAMs, M184V |

### Comparison to Existing Tools

| Tool | Correlation | Coverage |
|:-----|:------------|:---------|
| **Ternary VAE** | 0.89 | 23 drugs |
| Stanford HIVDB | 0.85* | 23 drugs |
| Geno2Pheno | 0.82* | Limited |

*Approximate, varies by drug

---

## Integration

### API Access

```python
from src.diseases import HIVAnalyzer

analyzer = HIVAnalyzer()
results = analyzer.analyze(
    sequences=["PRTCLKVYLVGRSM..."],
    drugs=["3TC", "EFV", "DTG"],
)

# Export for clinical record
json_output = results.to_json()
```

### Output Formats

- JSON (API integration)
- PDF report (patient record)
- HL7 FHIR (EHR integration, planned)

---

## Limitations

### Current

1. **Research use only**: Not FDA-cleared for clinical decisions
2. **Sequence quality**: Requires high-quality sequences
3. **Coverage**: Some rare mutations may not be represented

### Planned Improvements

- Regulatory submission (FDA 510(k))
- EHR integration
- Point-of-care support

---

## Contact

For clinical collaborations:
- Email: support@aiwhisperers.com
- Subject: Clinical Collaboration

---

_Last updated: 2025-12-28_
