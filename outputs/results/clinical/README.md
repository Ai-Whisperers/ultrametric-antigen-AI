# Clinical Decision Support

> **Clinical applications of p-adic bioinformatics for HIV treatment and vaccine design**

**Last Updated:** December 29, 2025

---

## Overview

This directory contains clinical decision support outputs derived from p-adic analysis, including vaccine candidates, drug resistance screening, and host-directed therapy targets.

---

## Directory Structure

```
clinical/
├── README.md                    # This file
│
├── clinical_applications/
│   ├── CLINICAL_REPORT.md       # Main clinical report
│   └── clinical_decision_support.json  # Machine-readable data
│
├── clinical_dashboard/
│   └── report_HIV-2024-001.json # Sample patient report
│
└── clinical_integration/
    └── clinical_integration.json # Integration outputs
```

---

## Key Outputs

### 1. Vaccine Candidates

**Top candidates from Gag protein:**

| Peptide | Priority | Escape Distance | HLA Coverage |
|---------|----------|-----------------|--------------|
| TPQDLNTML | 0.970 | 5.2 | 85% |
| GPGHKARVL | 0.965 | 4.8 | 82% |
| TSTLQEQIG | 0.958 | 5.0 | 79% |

### 2. MDR Screening

**Multi-Drug Resistance Alerts:**
- 2,489 sequences flagged high-risk (34.8% of dataset)
- Top mutations: I54V (4627×), L10I (1640×), L63P (1533×)

### 3. Host-Directed Therapy

**247 druggable human proteins** targeted by HIV Tat:
- Enables therapy that virus cannot escape
- Top targets: CDK9, CyclinT1, P-TEFb complex

---

## Data Formats

### Clinical Decision Support JSON

```json
{
  "vaccine_candidates": [
    {
      "peptide": "TPQDLNTML",
      "protein": "Gag",
      "priority": 0.970,
      "escape_distance": 5.2,
      "hla_coverage": 0.85,
      "conservation": 0.92
    }
  ],
  "mdr_alerts": [
    {
      "sequence_id": "HIV-2024-001",
      "risk_level": "high",
      "mutations": ["I54V", "L10I"],
      "drug_classes_affected": ["PI"]
    }
  ],
  "tat_targets": [
    {
      "human_protein": "CDK9",
      "druggability": 0.92,
      "existing_drugs": ["Flavopiridol"],
      "interaction_type": "direct"
    }
  ]
}
```

### Patient Report JSON

```json
{
  "patient_id": "HIV-2024-001",
  "analysis_date": "2025-12-26",
  "risk_assessment": {
    "mdr_risk": "moderate",
    "recommended_regimen": ["TDF", "FTC", "DTG"],
    "monitoring_frequency": "quarterly"
  }
}
```

---

## Usage

### Loading Clinical Data

```python
import json

# Load clinical decision support
with open("outputs/results/clinical/clinical_applications/clinical_decision_support.json") as f:
    clinical = json.load(f)

# Get high-priority vaccine candidates
top_candidates = [c for c in clinical["vaccine_candidates"] if c["priority"] > 0.9]

# Get MDR alerts
mdr_high_risk = [a for a in clinical["mdr_alerts"] if a["risk_level"] == "high"]
```

### Generating Reports

```python
# See scripts/clinical/ for report generation scripts
# Output format: Markdown or PDF
```

---

## Clinical Workflow

```
Sequence Input → P-adic Analysis → Risk Stratification → Report Generation
                      ↓                    ↓
              Drug Resistance        Vaccine Design
              Prediction             Recommendations
```

---

## Important Notes

1. **For Research Use Only**: These outputs are for research purposes and require clinical validation
2. **Not FDA Approved**: Predictions should be confirmed with standard clinical testing
3. **Privacy**: Patient data should be handled according to HIPAA/GDPR requirements

---

## Related Documents

- `../research/` - Underlying research findings
- `../validation/` - AlphaFold3 structural validation
- `../../reports/CLINICAL_REPORT.md` - Extended clinical report
