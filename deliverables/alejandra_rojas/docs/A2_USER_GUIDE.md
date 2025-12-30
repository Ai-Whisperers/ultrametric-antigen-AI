# A2: Pan-Arbovirus Primer Library - User Guide

**Tool:** `A2_pan_arbovirus_primers.py`
**Version:** 1.0
**Last Updated:** December 29, 2025

---

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Detailed Usage](#detailed-usage)
5. [Understanding Outputs](#understanding-outputs)
6. [Interpreting Results](#interpreting-results)
7. [Laboratory Validation](#laboratory-validation)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## Introduction

The Pan-Arbovirus Primer Library tool designs RT-PCR primers for differential diagnosis of arboviruses circulating in Paraguay and Latin America. It generates virus-specific primers while checking for cross-reactivity to ensure diagnostic specificity.

### Target Pathogens
| Virus | Clinical Presentation | Diagnostic Challenge |
|-------|----------------------|---------------------|
| **Dengue (4 serotypes)** | Fever, rash, hemorrhage | Serotype differentiation |
| **Zika** | Mild fever, conjunctivitis, birth defects | Similar to dengue |
| **Chikungunya** | Severe joint pain | Long-term arthralgia |
| **Mayaro** | Joint pain, fever | Emerging, misdiagnosed |

### Why This Tool?
- All these viruses cause similar initial symptoms
- Cross-reactivity with other flaviviruses is common
- Serotype-specific dengue diagnosis guides clinical management
- Mayaro is underdiagnosed due to lack of specific tests

---

## Installation

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# Required packages
pip install numpy pandas
```

### Optional Dependencies
```bash
# For FASTA file handling
pip install biopython

# For advanced visualization
pip install matplotlib seaborn
```

### Verify Installation
```bash
cd deliverables/alejandra_rojas
python -c "import numpy; print('NumPy OK')"
python scripts/A2_pan_arbovirus_primers.py --help
```

---

## Quick Start

### Demo Mode (Random Sequences)
```bash
python scripts/A2_pan_arbovirus_primers.py
```

This runs with demo sequences and outputs to `results/pan_arbovirus_primers/`.

### With Real Sequences (Recommended)
```bash
python scripts/A2_pan_arbovirus_primers.py \
    --sequences data/my_sequences.fasta \
    --output_dir results/my_primers/
```

---

## Detailed Usage

### Command-Line Options

```bash
python scripts/A2_pan_arbovirus_primers.py [OPTIONS]

Options:
  --sequences PATH    Input FASTA file with viral sequences
  --output_dir PATH   Output directory (default: results/pan_arbovirus_primers/)
  --primer_length N   Primer length in nucleotides (default: 20)
  --gc_min FLOAT      Minimum GC content (default: 0.4)
  --gc_max FLOAT      Maximum GC content (default: 0.6)
  --tm_min FLOAT      Minimum melting temperature (default: 55.0)
  --tm_max FLOAT      Maximum melting temperature (default: 65.0)
  --max_cross FLOAT   Maximum cross-reactivity threshold (default: 0.7)
  --help              Show this help message
```

### Example: Custom Parameters
```bash
# Design 22nt primers with stricter GC range
python scripts/A2_pan_arbovirus_primers.py \
    --primer_length 22 \
    --gc_min 0.45 \
    --gc_max 0.55 \
    --output_dir results/strict_gc_primers/
```

### Example: Specific Virus Focus
```bash
# Focus on dengue serotypes only
python scripts/A2_pan_arbovirus_primers.py \
    --sequences data/dengue_only.fasta \
    --output_dir results/dengue_primers/
```

---

## Understanding Outputs

### Directory Structure
```
results/pan_arbovirus_primers/
├── DENV-1_primers.csv       # Dengue serotype 1 candidates
├── DENV-1_pairs.csv         # Primer pairs for DENV-1
├── DENV-1_primers.fasta     # FASTA format for ordering
├── DENV-2_primers.csv
├── DENV-2_pairs.csv
├── DENV-2_primers.fasta
├── DENV-3_primers.csv
├── ...
├── ZIKV_primers.csv
├── CHIKV_primers.csv
├── MAYV_primers.csv
└── library_summary.json     # Complete metadata
```

### CSV File Format

#### primers.csv
| Column | Description | Example |
|--------|-------------|---------|
| `rank` | Priority (1 = best) | 1 |
| `position` | Genome start position | 1234 |
| `sequence` | Primer sequence | ATGCGATCGATCGATCGATC |
| `gc_content` | GC percentage | 0.55 |
| `tm_estimate` | Melting temperature | 58.2 |
| `stability_score` | P-adic stability | 0.923 |
| `conservation_score` | Cross-sequence identity | 0.95 |
| `is_specific` | Passes cross-reactivity check | True |

#### pairs.csv
| Column | Description |
|--------|-------------|
| `pair_id` | Unique identifier |
| `forward_seq` | Forward primer |
| `reverse_seq` | Reverse primer |
| `amplicon_size` | Expected product size |
| `tm_difference` | Tm match quality |

### FASTA Format
```fasta
>DENV1_primer_001_pos1234_Tm58.2
ATGCGATCGATCGATCGATC
>DENV1_primer_002_pos2345_Tm57.8
GCTAGCTAGCTAGCTAGCTA
```

### Summary JSON
```json
{
  "targets": ["DENV-1", "DENV-2", "DENV-3", "DENV-4", "ZIKV", "CHIKV", "MAYV"],
  "statistics": {
    "DENV-1": {
      "total_primers": 10,
      "specific_primers": 8,
      "primer_pairs": 12
    }
  },
  "design_parameters": {
    "primer_length": 20,
    "gc_range": [0.4, 0.6],
    "tm_range": [55.0, 65.0]
  },
  "cross_reactivity_matrix": {
    "DENV-1_ZIKV": 0.45,
    "DENV-1_DENV-2": 0.68
  }
}
```

---

## Interpreting Results

### Primer Selection Criteria

#### Excellent Primers (Rank 1-3)
- Stability score > 0.9
- Conservation > 0.95
- GC content 45-55%
- No cross-reactivity

#### Good Primers (Rank 4-10)
- Stability score > 0.8
- Conservation > 0.90
- GC content 40-60%
- Minimal cross-reactivity

#### Consider Carefully (Rank >10)
- May have suboptimal properties
- Use for backup or multiplex

### Cross-Reactivity Interpretation

| Value | Interpretation | Action |
|-------|---------------|--------|
| <0.5 | Highly specific | Excellent choice |
| 0.5-0.7 | Some similarity | Validate empirically |
| >0.7 | High cross-reactivity | Reject or redesign |

### Stability Score Meaning

The stability score uses p-adic valuation to measure evolutionary resistance:

| Score | Interpretation |
|-------|---------------|
| >0.95 | Highly conserved region |
| 0.85-0.95 | Stable, good target |
| 0.70-0.85 | Moderate stability |
| <0.70 | Variable region, avoid |

---

## Laboratory Validation

### Recommended Validation Protocol

#### Step 1: In Silico Verification
```bash
# Check against NCBI BLAST
blastn -query DENV1_primers.fasta -db nt -outfmt 6 -max_target_seqs 100
```

#### Step 2: Primer Synthesis
Order from IDT, Sigma, or Thermo Fisher with:
- HPLC purification for qPCR
- Standard desalting for endpoint PCR

#### Step 3: Gradient PCR
```
Gradient range: Tm - 5°C to Tm + 5°C
Template: Positive control plasmid or viral RNA
Cycles: 35
```

#### Step 4: Specificity Testing
Test each primer against:
- [ ] Target virus (positive control)
- [ ] Each other arbovirus (should be negative)
- [ ] Human DNA (should be negative)
- [ ] No template control

#### Step 5: Sensitivity Testing
```
Serial dilutions: 10^6 to 10^1 copies/reaction
Expected LOD: 10-100 copies
```

### Clinical Sample Workflow
```
Patient Sample → RNA Extraction → RT-PCR → Result Interpretation
                      ↓
              Store at -80°C for confirmation
```

---

## Troubleshooting

### Issue: No Specific Primers Found
**Cause:** High sequence similarity between viruses
**Solution:**
1. Try longer primers (22-25 nt)
2. Target unique genomic regions (E, NS1)
3. Lower cross-reactivity threshold carefully

### Issue: Low Conservation Scores
**Cause:** High genetic diversity in input sequences
**Solution:**
1. Filter sequences by collection date
2. Focus on specific geographic region
3. Use consensus sequences

### Issue: Tm Out of Range
**Cause:** GC content constraints too strict
**Solution:**
```bash
python scripts/A2_pan_arbovirus_primers.py \
    --gc_min 0.35 \
    --gc_max 0.65 \
    --tm_min 52.0 \
    --tm_max 68.0
```

### Issue: Memory Error
**Cause:** Too many input sequences
**Solution:**
1. Reduce sequence count (<500)
2. Use representative sequences only
3. Increase system memory

---

## FAQ

### Q: Can I use this for other viruses?
**A:** Yes! Add virus definitions to the `ARBOVIRUS_TARGETS` dictionary in the script.

### Q: How do I add new sequences from NCBI?
**A:**
```bash
# Using NCBI Datasets CLI
datasets download virus genome taxon 12637 --geo-location "Paraguay"
```

### Q: Can primers be used for multiplex PCR?
**A:** Yes, but verify:
1. No primer-primer interactions
2. Similar Tm values (within 2°C)
3. Different amplicon sizes (at least 50 bp apart)

### Q: What about probe design for qPCR?
**A:** Current version focuses on primers. Probe design will be added in v2.0.

### Q: How often should I update the primer library?
**A:** Annually, or when new variants emerge in your surveillance area.

---

## Support

For technical assistance:
1. Check the TECHNICAL_DOCS.md for implementation details
2. Review the demo output for expected behavior
3. Contact the bioinformatics team

---

*Part of the Ternary VAE Bioinformatics Partnership*
*Prepared for IICS-UNA Arbovirus Surveillance Program*
