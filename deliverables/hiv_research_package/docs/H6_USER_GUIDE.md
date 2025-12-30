# H6: TDR Screening - User Guide

**Tool:** `H6_tdr_screening.py`
**Version:** 1.0
**Last Updated:** December 29, 2025

---

## Introduction

The TDR (Transmitted Drug Resistance) Screening tool identifies drug resistance mutations in treatment-naive HIV patients. This helps clinicians select appropriate first-line antiretroviral therapy.

### Clinical Context
- TDR prevalence: 5-15% in most settings (PEPFAR data)
- Most common mutations: K103N (4.8%), M184V (5.2%)
- Impact: Guides regimen selection to avoid treatment failure

---

## Quick Start

### Demo Mode
```bash
python scripts/H6_tdr_screening.py
```

### With Patient Sequences
```bash
python scripts/H6_tdr_screening.py \
    --sequences data/patient_pol_sequences.fasta \
    --output_dir results/clinic_screening/
```

---

## Input Requirements

### Sequence Format
FASTA format with HIV-1 pol gene sequences:

```fasta
>PATIENT_001
CCTCAAATCACTCTTTGGCAACGACCCCTAGTTACAATA...
>PATIENT_002
CCTCAAATCACTCTTTGGCAACGACCCCTAGTTACAATA...
```

**Requirements:**
- Pol gene region (protease + RT + integrase)
- Minimum 500 bp coverage
- Sequence quality: <5% ambiguous bases

---

## WHO Surveillance Drug Resistance Mutations (SDRMs)

The tool screens for the complete WHO 2019 SDRM list:

### NRTI Mutations
| Position | Mutations | Primary Drug Impact |
|----------|-----------|---------------------|
| 41 | M41L | AZT, d4T |
| 65 | K65R | TDF, ABC, ddI |
| 67 | D67N/G/E | AZT, d4T |
| 69 | Insert | All NRTIs |
| 70 | K70R/E | AZT, TDF |
| 74 | L74V/I | ABC, ddI |
| 115 | Y115F | ABC |
| 151 | Q151M | All NRTIs |
| 184 | M184V/I | 3TC, FTC |
| 210 | L210W | AZT |
| 215 | T215Y/F | AZT, d4T |
| 219 | K219Q/E | AZT |

### NNRTI Mutations
| Position | Mutations | Primary Drug Impact |
|----------|-----------|---------------------|
| 100 | L100I | EFV, NVP |
| 101 | K101E/P | EFV, NVP |
| 103 | K103N/S | EFV, NVP |
| 106 | V106A/M | EFV, NVP |
| 181 | Y181C/I/V | NVP, EFV |
| 188 | Y188L/C/H | EFV, NVP |
| 190 | G190A/S/E | EFV, NVP |
| 230 | M230L | EFV, NVP |

### INSTI Mutations (2019 additions)
| Position | Mutations | Primary Drug Impact |
|----------|-----------|---------------------|
| 66 | T66I/A/K | EVG |
| 92 | E92Q | EVG, RAL |
| 118 | G118R | RAL, DTG |
| 121 | F121Y | RAL, EVG |
| 140 | G140S/A/C | RAL, EVG |
| 143 | Y143R/H/C | RAL |
| 147 | S147G | EVG |
| 148 | Q148H/K/R | All INSTIs |
| 155 | N155H | RAL, EVG |

---

## Output Interpretation

### Patient Report

```json
{
  "patient_id": "PATIENT_001",
  "tdr_positive": true,
  "mutations_detected": 2,
  "mutations": [
    {
      "gene": "RT",
      "position": 103,
      "wt_aa": "K",
      "mut_aa": "N",
      "mutation_string": "K103N",
      "drug_class": "NNRTI",
      "drugs_affected": ["EFV", "NVP"],
      "resistance_level": "high"
    },
    {
      "gene": "RT",
      "position": 184,
      "wt_aa": "M",
      "mut_aa": "V",
      "mutation_string": "M184V",
      "drug_class": "NRTI",
      "drugs_affected": ["3TC", "FTC"],
      "resistance_level": "high"
    }
  ],
  "resistance_summary": "NNRTI resistance (K103N), NRTI partial resistance (M184V)",
  "recommended_regimen": "TDF/3TC/DTG",
  "alternative_regimens": ["TAF/FTC/DTG", "ABC/3TC/DTG"],
  "avoid_drugs": ["EFV", "NVP"],
  "drug_susceptibility": {
    "TDF": "susceptible",
    "TAF": "susceptible",
    "ABC": "susceptible",
    "3TC": "low-level resistance",
    "FTC": "low-level resistance",
    "EFV": "high-level resistance",
    "NVP": "high-level resistance",
    "DTG": "susceptible",
    "RAL": "susceptible",
    "BIC": "susceptible"
  },
  "confidence": 0.92
}
```

### Susceptibility Levels

| Level | Interpretation | Clinical Action |
|-------|---------------|-----------------|
| Susceptible | No resistance mutations | Standard dosing |
| Potential low-level | Accessory mutations | Monitor closely |
| Low-level | Reduced susceptibility | May still be effective |
| Intermediate | Significant reduction | Consider alternatives |
| High-level | Resistance | Avoid drug |

---

## Regimen Recommendations

### For TDR-Negative Patients

**First-line (WHO 2019):**
1. TDF/3TC/DTG (preferred)
2. TDF/FTC/DTG (alternative)
3. TAF/FTC/DTG (if renal concerns)

### For TDR-Positive Patients

| TDR Pattern | Recommended Regimen | Rationale |
|-------------|---------------------|-----------|
| NNRTI only | TDF/3TC/DTG | Avoid NNRTIs |
| NRTI only (M184V) | TDF/3TC/DTG | TDF still active |
| NRTI (TAMs) | TAF/FTC/DTG | Reduced NRTI activity |
| INSTI mutations | TDF/3TC/boosted PI | Avoid INSTIs |
| Multiple classes | Specialist referral | Complex case |

---

## Batch Processing

### Multiple Patients
```bash
python scripts/H6_tdr_screening.py \
    --sequences data/batch_sequences.fasta \
    --output_dir results/batch_screening/ \
    --format csv
```

### Output Summary CSV
```csv
patient_id,tdr_positive,mutations,recommended_regimen,confidence
PATIENT_001,true,K103N;M184V,TDF/3TC/DTG,0.92
PATIENT_002,false,,TDF/3TC/DTG,0.95
PATIENT_003,true,Y181C,TDF/3TC/DTG,0.89
```

---

## Integration with Stanford HIVdb

### Comparison Mode
```bash
python scripts/H6_tdr_screening.py \
    --sequences data/patient_sequences.fasta \
    --compare_stanford \
    --output_dir results/validation/
```

This generates concordance report comparing our calls with Stanford HIVdb.

---

## Quality Control

### Sequence Quality Flags

| Flag | Meaning | Action |
|------|---------|--------|
| LOW_COVERAGE | <500 bp | Request resequencing |
| HIGH_AMBIGUITY | >5% N's | Check chromatogram |
| FRAMESHIFT | Reading frame error | Verify alignment |
| MIXTURE | Multiple variants | Report minor variants |

### Confidence Score Interpretation

| Confidence | Meaning |
|------------|---------|
| > 0.95 | High quality, clear result |
| 0.85-0.95 | Good quality, reliable |
| 0.70-0.85 | Acceptable, verify key mutations |
| < 0.70 | Low quality, recommend retest |

---

## Troubleshooting

### Issue: No mutations detected but expected TDR

**Causes:**
- Sequence doesn't cover mutation positions
- Wrong gene region sequenced
- Low viral load affecting sequencing

**Solution:** Verify sequence coverage, consider retesting

### Issue: Too many mutations detected

**Causes:**
- Sequence alignment errors
- Poor sequence quality
- Mixed infection

**Solution:** Check alignment, verify with Sanger

---

*Part of the Ternary VAE Bioinformatics Partnership*
*For HIV resistance screening and treatment optimization*
