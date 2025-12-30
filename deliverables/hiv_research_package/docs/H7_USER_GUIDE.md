# H7: LA Injectable Selection - User Guide

**Tool:** `H7_la_injectable_selection.py`
**Version:** 1.0
**Last Updated:** December 29, 2025

---

## Introduction

The LA Injectable Selection tool assesses patient eligibility for long-acting cabotegravir/rilpivirine (CAB/RPV-LA) injectable therapy. It evaluates multiple clinical criteria to predict treatment success.

### Clinical Context
- CAB/RPV-LA: Monthly or every-2-month intramuscular injections
- Alternative to daily oral pills for suppressed patients
- Requires careful patient selection for success

---

## Quick Start

### Demo Mode
```bash
python scripts/H7_la_injectable_selection.py
```

### With Patient Data
```bash
python scripts/H7_la_injectable_selection.py \
    --patient_file data/clinic_patients.json \
    --output_dir results/la_assessment/
```

---

## Eligibility Criteria

### Absolute Requirements

| Criterion | Requirement | Rationale |
|-----------|-------------|-----------|
| Viral load | < 50 copies/mL | Must be suppressed |
| Prior virologic failure on CAB/RPV | None | Resistance risk |
| Rilpivirine resistance | None (especially E138K, H221Y) | Cross-resistance |
| Cabotegravir resistance | None (Q148 pathway) | Limited options |

### Relative Considerations

| Factor | Weight | Notes |
|--------|--------|-------|
| BMI > 30 | -10% | PK variability |
| Prior NNRTI failure | -15% | Archived resistance |
| Psychiatric history | -5% | Monitor mood |
| Injection site concerns | -10% | Absorption issues |
| Poor oral adherence | +10% | LA may help |

---

## Input Data Format

### Patient JSON Structure

```json
{
  "patients": [
    {
      "patient_id": "P001",
      "viral_load": 20,
      "cd4_count": 650,
      "current_regimen": ["TDF", "FTC", "DTG"],
      "prior_regimens": [["TDF", "FTC", "EFV"]],
      "years_on_art": 5,
      "adherence_score": 0.95,
      "bmi": 28.5,
      "prior_virologic_failure": false,
      "nnrti_mutations": [],
      "insti_mutations": [],
      "comorbidities": ["depression"],
      "contraindications": []
    }
  ]
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| patient_id | string | Unique identifier |
| viral_load | int | Latest VL (copies/mL) |
| current_regimen | list | Current ART drugs |
| prior_regimens | list of lists | Treatment history |

### Optional Fields

| Field | Type | Default |
|-------|------|---------|
| cd4_count | int | 500 |
| adherence_score | float | 0.85 |
| bmi | float | 25.0 |
| comorbidities | list | [] |

---

## Output Interpretation

### Eligibility Report

```json
{
  "patient_id": "P001",
  "eligible": true,
  "success_probability": "92.7%",
  "recommendation": "ELIGIBLE - Recommend LA injectable switch",
  "cab_resistance_risk": "0.0%",
  "rpv_resistance_risk": "0.0%",
  "pk_adequacy": "100.0%",
  "adherence_score": "95.0%",
  "risk_factors": [
    "Psychiatric history (monitor for mood changes)"
  ],
  "monitoring_plan": [
    "HIV RNA at 1, 3, and 6 months post-switch",
    "CD4 count at 6 months",
    "Psychiatric assessment at each visit"
  ],
  "contraindications": [],
  "notes": "Good candidate with strong adherence history"
}
```

### Eligibility Categories

| Category | Success Prob | Recommendation |
|----------|-------------|----------------|
| ELIGIBLE | > 85% | Proceed with switch |
| CONDITIONAL | 70-85% | Address risk factors first |
| NOT ELIGIBLE | < 70% | Continue oral therapy |
| CONTRAINDICATED | N/A | Absolute exclusion |

---

## Risk Factor Assessment

### Resistance Risk

| Factor | Risk Increase | Mitigation |
|--------|---------------|------------|
| Prior NNRTI failure | +15% RPV risk | Historical genotype |
| Prior INSTI exposure | +5% CAB risk | Check for mutations |
| VL 50-200 | +20% | Suppress first |
| VL > 200 | EXCLUDE | Not eligible |

### Pharmacokinetic Risk

| Factor | Risk | Mitigation |
|--------|------|------------|
| BMI > 35 | Reduced absorption | Consider more frequent dosing |
| Injection site issues | Variable levels | Alternative sites |
| Drug interactions | Altered levels | Review medications |

### Behavioral Risk

| Factor | Risk | Mitigation |
|--------|------|------------|
| Poor oral adherence | May actually HELP | LA removes daily pill burden |
| Appointment keeping | Critical for LA | Assess reliability |
| Psychiatric instability | Mood changes possible | Close monitoring |

---

## Monitoring Plan Generation

### Standard Monitoring (All Patients)

| Timepoint | Tests |
|-----------|-------|
| Month 1 | HIV RNA, tolerability assessment |
| Month 3 | HIV RNA, injection site evaluation |
| Month 6 | HIV RNA, CD4, comprehensive review |
| Annually | HIV RNA, CD4, resistance if needed |

### Enhanced Monitoring (High-Risk)

Added for patients with risk factors:

| Risk Factor | Additional Monitoring |
|-------------|----------------------|
| BMI > 30 | Drug level (Ctrough) at month 1-2 |
| Psychiatric | Mood assessment each visit |
| Prior NNRTI | Baseline archived resistance test |
| Borderline VL | Monthly VL for first 3 months |

---

## Clinical Workflow

### Pre-Switch Assessment

```
1. Confirm viral suppression (VL < 50 for 3+ months)
2. Review treatment history for NNRTI/INSTI exposure
3. Obtain historical genotype if available
4. Assess adherence and appointment-keeping ability
5. Screen for psychiatric conditions
6. Calculate BMI and assess injection sites
```

### Oral Lead-In (Optional)

WHO now allows direct switch, but consider oral lead-in:
- CAB 30mg + RPV 25mg daily x 1 month
- Assesses tolerability before injection

### Injection Schedule

| Regimen | Loading | Maintenance |
|---------|---------|-------------|
| Monthly | Day 1, Month 1 | Monthly |
| Every 2 months | Day 1, Month 1, Month 2 | Every 2 months |

---

## Batch Processing

### Clinic-Wide Assessment
```bash
python scripts/H7_la_injectable_selection.py \
    --patient_file data/all_suppressed_patients.json \
    --output_dir results/clinic_assessment/ \
    --format csv
```

### Summary Report
```csv
patient_id,eligible,success_prob,primary_risk_factor,recommendation
P001,true,92.7%,Psychiatric,Proceed with monitoring
P002,false,77.0%,Not suppressed,Optimize oral first
P003,true,97.0%,None,Ideal candidate
```

---

## Troubleshooting

### Issue: High-risk patient wants LA despite low eligibility

**Approach:**
1. Address modifiable risk factors
2. Consider extended oral stabilization
3. Use enhanced monitoring plan if proceeding
4. Document shared decision-making

### Issue: Previously suppressed patient with detectable VL

**Approach:**
1. Rule out resistance
2. Address adherence barriers
3. Re-suppress for 3+ months
4. Re-assess eligibility

### Issue: No historical genotype available

**Approach:**
1. Check for archived resistance (proviral DNA)
2. Review prior NNRTI exposure carefully
3. Apply conservative risk estimates
4. Consider oral lead-in to assess

---

## Validation

### Expected Outcomes

When tool predictions are validated against real-world data:

| Eligibility | Expected 48-week Success |
|-------------|--------------------------|
| > 90% predicted | > 95% actual |
| 80-90% predicted | 85-95% actual |
| 70-80% predicted | 70-85% actual |
| < 70% predicted | < 70% actual |

---

*Part of the Ternary VAE Bioinformatics Partnership*
*For HIV treatment optimization and LA injectable selection*
