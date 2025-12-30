# B8: Microbiome-Safe AMP Design - User Guide

**Tool:** `B8_microbiome_safe_amps.py`
**Version:** 1.0
**Last Updated:** December 29, 2025

---

## Introduction

The Microbiome-Safe AMP Design tool creates antimicrobial peptides that selectively kill pathogens while sparing beneficial commensal bacteria. This is critical for:

- Skin infection treatment (preserve *S. epidermidis*)
- Gut health (spare *Lactobacillus*, *Bacteroides*)
- Topical acne treatment (kill pathogenic *C. acnes*, spare normal flora)

---

## Quick Start

### Demo Mode
```bash
python scripts/B8_microbiome_safe_amps.py
```

### Custom Targets
```bash
python scripts/B8_microbiome_safe_amps.py \
    --pathogens "S_aureus,MRSA,P_acnes_pathogenic" \
    --commensals "S_epidermidis,C_acnes,Corynebacterium" \
    --output_dir results/my_microbiome_safe/
```

---

## Key Concept: Selectivity Index

The **Selectivity Index (SI)** measures how selective a peptide is:

```
SI = Geometric Mean(Commensal MICs) / Geometric Mean(Pathogen MICs)
```

| SI Value | Interpretation | Clinical Utility |
|----------|---------------|------------------|
| < 1.0 | Kills commensals MORE than pathogens | AVOID |
| 1.0 - 2.0 | Slight selectivity | Limited |
| 2.0 - 4.0 | Moderate selectivity | Acceptable |
| > 4.0 | High selectivity | TARGET |
| > 10.0 | Excellent selectivity | Ideal |

---

## Organism Categories

### Default Pathogens
| Organism | Associated Condition |
|----------|---------------------|
| *S. aureus* | Skin infections, abscesses |
| MRSA | Hospital-acquired infections |
| *P. acnes* (pathogenic) | Inflammatory acne |

### Default Commensals
| Organism | Beneficial Role |
|----------|----------------|
| *S. epidermidis* | Skin barrier protection |
| *C. acnes* | Normal skin flora |
| *Corynebacterium* | Competitive exclusion |
| *Malassezia* | Skin fungal balance |

---

## Understanding the Output

### Results JSON

```json
{
  "objective": "Microbiome-safe AMP design",
  "pathogens": ["S_aureus", "MRSA", "P_acnes_pathogenic"],
  "commensals": ["S_epidermidis", "C_acnes", "Corynebacterium", "Malassezia"],
  "candidates": [
    {
      "rank": 1,
      "sequence": "HNHWHMNWKKKKAYAHKPGR",
      "net_charge": 8.0,
      "selectivity_index": 1.26,
      "pathogen_mics": {
        "S_aureus": 9.5,
        "MRSA": 10.7,
        "P_acnes_pathogenic": 8.4
      },
      "commensal_mics": {
        "S_epidermidis": 13.6,
        "C_acnes": 12.1,
        "Corynebacterium": 15.3,
        "Malassezia": 8.2
      }
    }
  ]
}
```

### Interpreting MIC Values

Lower MIC = More active against that organism

For selectivity, we WANT:
- **Low pathogen MICs** (kills pathogens effectively)
- **High commensal MICs** (spares commensals)

---

## Optimization Objectives

The tool optimizes 4 objectives simultaneously:

| Objective | Goal | Weight |
|-----------|------|--------|
| Pathogen Activity | Minimize mean pathogen MIC | 1.5 |
| Commensal Sparing | Maximize mean commensal MIC | 1.5 |
| Selectivity Index | Maximize SI | 2.0 |
| Toxicity | Minimize host toxicity | 1.0 |

---

## Design Strategies

### For Skin Applications

```bash
python scripts/B8_microbiome_safe_amps.py \
    --pathogens "S_aureus,MRSA,C_acnes_pathogenic" \
    --commensals "S_epidermidis,Corynebacterium,Cutibacterium" \
    --niche "skin"
```

### For Gut Applications

```bash
python scripts/B8_microbiome_safe_amps.py \
    --pathogens "C_difficile,EHEC,Salmonella" \
    --commensals "Lactobacillus,Bacteroides,Bifidobacterium" \
    --niche "gut"
```

---

## Laboratory Validation

### Selectivity Testing Protocol

1. **Panel Preparation**
   - Obtain clinical isolates of pathogens
   - Obtain type strains of commensals (ATCC)

2. **MIC Determination**
   - Broth microdilution for each organism
   - Calculate geometric means

3. **Compute Experimental SI**
   ```
   SI_exp = GM(commensal MICs) / GM(pathogen MICs)
   ```

4. **Co-culture Assays**
   - Mix pathogen + commensal
   - Add peptide
   - Measure selective killing by CFU counting

---

## Troubleshooting

### Issue: SI < 1.0 for all candidates

**Cause:** Pathogens and commensals too similar (e.g., both Staphylococcus)
**Solution:** Target different membrane features (e.g., virulence factors)

### Issue: Duplicate sequences in output

**Cause:** Limited diversity in optimization
**Solution:** Increase population size, run longer

### Issue: High toxicity with high SI

**Cause:** Trade-off between selectivity and safety
**Solution:** Add stricter toxicity constraint

---

## Scientific Background

### Why Is Selectivity Possible?

Pathogens often have distinct features:

| Feature | Pathogen | Commensal |
|---------|----------|-----------|
| Capsule | Present (*S. aureus*) | Often absent |
| Biofilm | Aggressive | Protective |
| LTA/LPS modifications | Virulence-associated | Normal |
| Surface proteins | Invasins, toxins | Adhesins |

AMPs can potentially exploit these differences for selectivity.

---

*Part of the Ternary VAE Bioinformatics Partnership*
