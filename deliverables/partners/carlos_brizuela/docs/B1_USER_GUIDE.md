# B1: Pathogen-Specific AMP Design - User Guide

**Tool:** `B1_pathogen_specific_design.py`
**Version:** 1.0
**Last Updated:** December 29, 2025

---

## Introduction

The Pathogen-Specific AMP Design tool uses NSGA-II multi-objective optimization to design antimicrobial peptides targeting specific WHO priority pathogens. It optimizes for activity, low toxicity, and stability simultaneously.

### Target Applications
- Design AMPs against carbapenem-resistant *Acinetobacter baumannii*
- Create peptides for MRSA treatment
- Develop novel antibiotics for multidrug-resistant Gram-negative bacteria

---

## Quick Start

### Demo Mode
```bash
python scripts/B1_pathogen_specific_design.py
```

### Custom Pathogen
```bash
python scripts/B1_pathogen_specific_design.py \
    --pathogen "Pseudomonas_aeruginosa" \
    --population 200 \
    --generations 100 \
    --output_dir results/pseudomonas_amps/
```

---

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pathogen` | Acinetobacter_baumannii | Target organism |
| `--population` | 100 | Population size |
| `--generations` | 50 | Number of generations |
| `--output_dir` | results/pathogen_specific/ | Output directory |
| `--latent_dim` | 16 | VAE latent dimension |
| `--seed` | None | Random seed |

---

## Supported Pathogens

| Pathogen | Key Features | Expected Peptide Properties |
|----------|--------------|----------------------------|
| *Acinetobacter baumannii* | Gram-negative, LPS-rich | High charge (+4 to +8), moderate hydrophobicity |
| *Pseudomonas aeruginosa* | Gram-negative, biofilm-forming | Cationic, amphipathic |
| *Klebsiella pneumoniae* | Gram-negative, capsule | High charge, membrane-active |
| *Staphylococcus aureus* | Gram-positive, thick PG | Moderate charge (+2 to +4) |
| *Enterococcus faecium* | Gram-positive, VRE | Cell wall-targeting |

---

## Understanding the Output

### Results JSON Structure

```json
{
  "objective": "Pathogen-specific AMP design",
  "target_pathogen": "Acinetobacter_baumannii",
  "resistance_profile": "Carbapenem-resistant (WHO Critical)",
  "candidates": [
    {
      "rank": 1,
      "sequence": "HFHTSFFFSTKVYETSHTHY",
      "length": 20,
      "net_charge": 2.0,
      "hydrophobicity": 0.09,
      "predicted_activity": 4.04,
      "toxicity_score": 0.0,
      "latent_vector": [0.23, -0.45, ...]
    }
  ]
}
```

### Interpreting Metrics

| Metric | Good Range | Interpretation |
|--------|------------|----------------|
| `predicted_activity` | 0.1 - 2.0 | Lower = better MIC (more active) |
| `net_charge` | +2 to +8 | Positive for Gram-negative targeting |
| `hydrophobicity` | 0.3 - 0.5 | Balance for membrane insertion |
| `toxicity_score` | < 0.3 | Lower = safer for mammalian cells |

---

## Scientific Background

### Why Pathogen-Specific?

Different bacteria have distinct membrane compositions:

| Bacterium Type | Membrane Feature | Optimal Peptide Property |
|----------------|------------------|-------------------------|
| Gram-negative | LPS outer membrane | High cationic charge |
| Gram-positive | Thick peptidoglycan | Cell wall penetration |
| Mycobacteria | Mycolic acid layer | Lipophilic peptides |

### Optimization Objectives

1. **Minimize Activity Score** (predicted MIC)
2. **Minimize Toxicity** (hemolysis risk)
3. **Maximize Stability** (VAE reconstruction quality)

The NSGA-II algorithm finds the Pareto front - the set of peptides where you cannot improve one objective without worsening another.

---

## Laboratory Validation

### Recommended Protocol

1. **In Silico Verification**
   - BLAST search against toxin databases
   - Check for known AMP motifs

2. **Synthesis**
   - Order from GenScript, Peptide 2.0, or similar
   - Standard Fmoc SPPS
   - HPLC purification (>95%)

3. **MIC Testing**
   - Broth microdilution (CLSI M07)
   - Test against target pathogen + controls

4. **Toxicity Assays**
   - Hemolysis assay (human RBCs)
   - MTT cytotoxicity (mammalian cells)

---

## Troubleshooting

### Issue: All peptides have high charge

**Cause:** Expected for Gram-negative targets
**Solution:** If lower charge needed, adjust objective weights

### Issue: Low diversity in Pareto front

**Cause:** Convergence to local optimum
**Solution:** Increase population size or mutation rate

### Issue: High toxicity scores

**Cause:** Peptides too hydrophobic
**Solution:** Add hydrophobicity constraint

---

*Part of the Ternary VAE Bioinformatics Partnership*
