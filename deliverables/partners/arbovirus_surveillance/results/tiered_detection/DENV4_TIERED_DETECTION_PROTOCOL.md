# DENV-4 Tiered Detection Protocol

**Doc-Type:** Laboratory Protocol · Version 1.0 · 2026-01-04 · AI Whisperers

---

## Overview

This protocol implements a three-tier detection strategy for DENV-4, designed to address
the serotype's exceptional cryptic diversity (only 13.3% of strains can be detected by
standard consensus primers).

| Tier | Approach | Coverage | Cost | Turnaround |
|------|----------|----------|------|------------|
| 1 | Clade-specific RT-PCR | 13.3% | $5-10 | 2-4 hours |
| 2 | Pan-flavivirus + Sequencing | 86.7%+ | $20-50 | 1-2 days |
| 3 | Amplicon Sequencing (NGS) | 100% | $50-100 | 2-3 days |

---

## Decision Tree

```
Sample → Tier 1 RT-PCR
           │
           ├─ POSITIVE → Report as DENV-4 (Clade_E.3.2)
           │
           └─ NEGATIVE → Tier 2 Pan-Flavivirus RT-PCR
                            │
                            ├─ POSITIVE → Sanger Sequencing
                            │               │
                            │               └─ DENV-4 confirmed → Report
                            │
                            └─ NEGATIVE → Consider Tier 3 (if high suspicion)
                                         or report as Negative
```

---

## Tier 1: Clade-Specific RT-PCR

### Purpose
Rapid, low-cost screening for the most common DENV-4 clade in circulation.

### Primers

| Name | Sequence (5'→3') | Position | Tm |
|------|------------------|----------|-----|
| DENV4_E32_NS5_F | `AGCAGTTCCAACAGAATGGTTTCCA` | NS5 9908-9933 | 57.1°C |
| DENV4_E32_NS5_R | `ATTTTGGGGTTGTCTTCTATCC` | NS5 10036-10014 | 50.1°C |

**Expected amplicon:** 128 bp

### RT-PCR Protocol

**Reaction Mix (25 µL):**

| Component | Volume | Final Concentration |
|-----------|--------|---------------------|
| 2X One-Step RT-PCR Master Mix | 12.5 µL | 1X |
| Forward Primer (10 µM) | 0.5 µL | 0.2 µM |
| Reverse Primer (10 µM) | 0.5 µL | 0.2 µM |
| Template RNA | 5 µL | - |
| Nuclease-free water | 6.5 µL | - |

**Cycling Conditions:**

| Step | Temperature | Time | Cycles |
|------|-------------|------|--------|
| Reverse transcription | 50°C | 30 min | 1 |
| Initial denaturation | 95°C | 2 min | 1 |
| Denaturation | 95°C | 15 sec | 40 |
| Annealing | 58°C | 30 sec | 40 |
| Extension | 72°C | 30 sec | 40 |
| Final extension | 72°C | 5 min | 1 |

### Interpretation

- **Positive:** Band at expected size → Report as DENV-4
- **Negative:** Proceed to Tier 2

---

## Tier 2: Pan-Flavivirus RT-PCR + Sequencing

### Purpose
Detect divergent DENV-4 strains and other flaviviruses. Requires sequencing confirmation.

### Primers

**Primary Pair (FU1/cFD2):**

| Name | Sequence (5'→3') | Reference |
|------|------------------|-----------|
| PANFLAVI_FU1 | `TACAACATGATGGGAAAGAGAGAGAA` | Kuno et al. 1998 |
| PANFLAVI_cFD2 | `GTGTCCCAGCCGGCGGTGTCATCAGC` | Kuno et al. 1998 |

**Expected amplicon:** ~215 bp

**Hemi-Nested Pair (MAMD/cFD3):**

| Name | Sequence (5'→3') | Reference |
|------|------------------|-----------|
| PANFLAVI_MAMD | `AACATGATGGGRAARAGRGARAA` | Scaramozzino et al. 2001 |
| PANFLAVI_cFD3 | `AGCATGTCTTCCGTGGTCATCCA` | Scaramozzino et al. 2001 |

### Protocol

1. Use same RT-PCR conditions as Tier 1
2. If primary pair negative but high clinical suspicion, run hemi-nested
3. **All positives must be sequenced for species confirmation**

### Sequencing

1. Purify PCR product (gel extraction or column)
2. Send for Sanger sequencing with forward primer
3. BLAST against NCBI database
4. Report species and genotype

---

## Tier 3: Amplicon Sequencing (Reference Labs)

### Purpose
Complete genome coverage for surveillance, outbreak investigation, and novel variant detection.

### Scheme

- **Total amplicons:** 30
- **Amplicon size:** ~400 bp
- **Overlap:** 50 bp
- **Genome coverage:** 97.9%

### Platforms

- **Illumina MiSeq/NextSeq:** High accuracy, 24-48 hour turnaround
- **Oxford Nanopore MinION:** Portable, real-time, 4-8 hour turnaround

### Bioinformatics Pipeline

```bash
# 1. Quality control
fastp -i reads_R1.fq -I reads_R2.fq -o clean_R1.fq -O clean_R2.fq

# 2. Map to reference
minimap2 -ax sr DENV4_reference.fa clean_R1.fq clean_R2.fq | samtools sort > aligned.bam

# 3. Call consensus
samtools consensus -a aligned.bam > consensus.fa

# 4. Genotype assignment
blastn -query consensus.fa -db denv4_genotypes -outfmt 6

# 5. Phylogenetic placement
iqtree2 -s alignment.fa -m GTR+G -bb 1000
```

---

## Quality Control

### Required Controls

| Control | Purpose |
|---------|---------|
| DENV-4 positive (Clade_E.3.2) | Tier 1 validation |
| DENV-4 positive (divergent) | Tier 2 validation |
| DENV-1/2/3 positive | Cross-reactivity check |
| Negative (water) | Contamination check |
| Extraction control | RNA extraction validation |

### Acceptance Criteria

- Positive controls must amplify at expected Ct
- Negative controls must show no amplification
- Amplicon sizes must match expected values

---

## Troubleshooting

| Issue | Possible Cause | Solution |
|-------|---------------|----------|
| No amplification (Tier 1) | Divergent strain | Proceed to Tier 2 |
| Weak band | Low viral load | Increase template or use hemi-nested |
| Multiple bands | Non-specific amplification | Increase annealing temp |
| Sequencing fails | Mixed infection | Run Tier 3 for resolution |

---

## References

1. Kuno G et al. (1998) Universal diagnostic RT-PCR protocol for arboviruses. J Virol Methods.
2. Scaramozzino N et al. (2001) Comparison of flavivirus universal primer pairs. J Clin Microbiol.
3. Lanciotti RS et al. (1992) Rapid detection of dengue virus. J Clin Microbiol.

---

*Protocol generated: 2026-01-04 04:51:42*

*IICS-UNA Arbovirus Surveillance Program*