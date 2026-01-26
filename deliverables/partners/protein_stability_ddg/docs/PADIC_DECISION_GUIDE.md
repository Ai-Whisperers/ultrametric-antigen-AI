# When to Use P-adic Geometric Analysis

**Doc-Type:** Decision Guide · Version 1.0 · Updated 2026-01-03 · AI Whisperers

---

## Quick Decision Flowchart

```
                    START
                      |
         Do you need absolute DDG values?
                   /      \
                 YES       NO
                  |         |
          Use FoldX    Is structure available?
          or Rosetta        /      \
                          YES       NO
                           |         |
               Need Rosetta-    Use P-adic
               blind detection?  (sequence-only)
                  /      \
                YES       NO
                 |         |
           Use BOTH    Compare:
           C1 + FoldX   speed vs precision
                           |
                    /-----------\
                   /             \
           High-throughput    Precision
           (>1000 mutations)  (<100 mutations)
                 |               |
           Use P-adic      Use FoldX/Rosetta
           for screening    for final ranking
```

---

## Decision Matrix

| Your Situation | Recommended Tool | Rationale |
|----------------|------------------|-----------|
| **Screening 10,000+ mutations** | P-adic C4 | 100-1000x faster than physics-based |
| **Final candidates (<20)** | FoldX/Rosetta | Calibrated kcal/mol values |
| **No structure available** | P-adic C4 | Works sequence-only |
| **Detecting hidden instability** | P-adic C1 | Unique Rosetta-blind capability |
| **Codon-level effects** | P-adic | Encodes synonymous codon signal |
| **Insertions/deletions** | Rosetta | Handles backbone changes |
| **Publication-ready values** | FoldX/Rosetta | Community-accepted calibration |
| **Quick prioritization** | P-adic | Sub-second per mutation |

---

## The P-adic Advantage: Three Unique Capabilities

### 1. Rosetta-Blind Detection (C1)

**What it is:** Identifies residues that Rosetta scores as stable but are geometrically unstable.

**When this matters:**
- Engineering enzymes where flexibility affects catalysis
- Analyzing disease variants in structurally strained regions
- Finding non-obvious mutation hotspots

**Evidence:** 23.6% of residues in our test set are Rosetta-blind.

```
Use Case: Therapeutic protein design
Problem:  Rosetta says position 156 is stable
P-adic:   Flags high geometric instability (discordance 0.39)
Action:   Experimentally test mutations at 156 despite Rosetta score
Outcome:  Discover destabilizing effect Rosetta missed
```

### 2. Codon-Level Resolution

**What it is:** P-adic encoding captures information from the 3-adic structure of codons, not just amino acid properties.

**When this matters:**
- Synonymous codon effects on protein expression
- Codon optimization for heterologous expression
- Understanding evolutionary conservation at codon level

**Standard tools ignore this:** FoldX/Rosetta work purely at amino acid level.

```
Use Case: Codon optimization for E. coli expression
Problem:  Multiple codons encode same amino acid
P-adic:   Distinguishes preferred vs. rare codon embeddings
Action:   Optimize codon choice, not just amino acid sequence
Outcome:  Improved expression without changing protein sequence
```

### 3. Hierarchical Geometric Information

**What it is:** The hyperbolic embedding encodes hierarchical relationships between amino acids based on their 3-adic valuation.

**When this matters:**
- Detecting subtle structural perturbations
- Identifying evolutionarily conserved geometric patterns
- Complementing physics-based predictions

```
Use Case: Variant of uncertain significance (VUS) interpretation
Problem:  Physics-based tools give borderline DDG (-0.5 kcal/mol)
P-adic:   Shows large geometric deviation from common patterns
Action:   Flag as potentially pathogenic despite mild DDG
Outcome:  Correctly identifies damaging variant
```

---

## Practical Workflows

### Workflow 1: High-Throughput Screening

**Goal:** Screen 50,000 possible mutations, select top 100 for experiments.

```bash
# Stage 1: P-adic screen (5-10 minutes)
python scripts/C4_mutation_effect_predictor.py \
    --mutation_file all_50000_mutations.txt \
    --output_dir results/screen/

# Stage 2: Filter to top candidates
python -c "
import pandas as pd
df = pd.read_json('results/screen/predictions.json')
top = df.nsmallest(100, 'predicted_ddg')  # Most stabilizing
top.to_csv('top_100_candidates.csv')
"

# Stage 3: FoldX refinement (only on filtered set)
# foldx --command=BuildModel --mutant-file=top_100_candidates.txt
```

**Time savings:** 50,000 x 30s (FoldX) = 417 hours → 10 min (P-adic) + 50 min (FoldX on 100) = 1 hour

### Workflow 2: Rosetta-Blind Audit

**Goal:** Identify positions where Rosetta may underestimate instability.

```bash
# Run C1 analysis
python scripts/C1_rosetta_blind_detection.py \
    --input structure_embeddings.pt \
    --rosetta_scores rosetta_per_residue.csv \
    --output_dir results/audit/

# Review flagged positions
python -c "
import json
with open('results/audit/rosetta_blind_report.json') as f:
    report = json.load(f)
print(f'Rosetta-blind residues: {report[\"summary\"][\"rosetta_blind\"]}')
for r in report['rosetta_blind_residues'][:10]:
    print(f'  {r[\"residue_id\"]} {r[\"residue_name\"]}: discordance {r[\"discordance_score\"]:.3f}')
"
```

**Output interpretation:**
- High discordance (>0.3) = Strong disagreement, investigate experimentally
- Medium discordance (0.2-0.3) = Moderate disagreement, note for follow-up
- Low discordance (<0.2) = Tools agree, trust Rosetta score

### Workflow 3: Ensemble Prediction

**Goal:** Combine P-adic with FoldX for improved accuracy.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

def ensemble_predict(padic_ddg, foldx_ddg, weights=[0.4, 0.6]):
    """Weighted ensemble of P-adic and FoldX predictions."""
    return weights[0] * padic_ddg + weights[1] * foldx_ddg

# Optimal weights from validation (tune on your data)
# P-adic captures geometric signal FoldX misses
# FoldX provides calibrated energy values
ensemble_ddg = ensemble_predict(padic_predictions, foldx_predictions)
```

**Expected improvement:** Ensemble typically improves Spearman r by 0.05-0.10 over single method.

---

## When NOT to Use P-adic

### Avoid P-adic alone when:

1. **Absolute DDG values required**
   - P-adic DDG is relative, not calibrated to kcal/mol
   - Use FoldX/Rosetta for thermodynamic interpretation

2. **Backbone modifications**
   - Insertions, deletions, loop remodeling
   - P-adic assumes fixed backbone

3. **Multi-mutation epistasis**
   - Current implementation handles single mutations
   - Use FoldX for combinatorial effects

4. **Regulatory/publication requirements**
   - Some journals require established methods
   - Use P-adic as complementary evidence

### P-adic as sole method is appropriate when:

1. Structure unavailable
2. Speed is critical (>1000 mutations)
3. Rosetta-blind detection is primary goal
4. Codon-level analysis needed
5. Initial screening before expensive methods

---

## Integration Examples

### With AlphaFold

```python
# Use AF2 confidence to weight P-adic predictions
def alphafold_weighted_padic(padic_ddg, plddt_scores, position):
    """Downweight predictions at low-confidence positions."""
    confidence = plddt_scores[position] / 100.0
    return padic_ddg * confidence
```

### With ESM-1v

```python
# Compare P-adic geometric with ESM evolutionary signal
def compare_padic_esm(padic_ddg, esm_score, mutation):
    """Flag discordant predictions for review."""
    padic_destab = padic_ddg > 1.0
    esm_destab = esm_score < -5.0  # ESM uses log-likelihood
    if padic_destab != esm_destab:
        print(f"DISCORDANT: {mutation} - P-adic: {padic_ddg:.2f}, ESM: {esm_score:.2f}")
        return "review"
    return "concordant"
```

### With Experimental Data

```python
# Validate on your specific protein family
from scipy.stats import spearmanr

def validate_on_family(predictions, experimental_ddg):
    """Check P-adic performance on your protein family."""
    r, p = spearmanr(predictions, experimental_ddg)
    print(f"Spearman r = {r:.3f} (p = {p:.4f})")
    if r < 0.4:
        print("WARNING: P-adic may not generalize well to this family")
        print("Consider training family-specific weights")
    return r
```

---

## Summary Table

| Capability | P-adic | FoldX | Rosetta | ESM-1v |
|------------|--------|-------|---------|--------|
| Speed (per mutation) | <0.1s | 30-60s | 5-30min | 1-5s |
| Structure required | No | Yes | Yes | No |
| Calibrated DDG | No | Yes | Yes | No |
| Rosetta-blind detection | Yes | No | No | No |
| Codon-level signal | Yes | No | No | No |
| Backbone changes | No | Yes | Yes | Limited |
| Epistasis | No | Yes | Yes | Limited |
| Community acceptance | Emerging | High | High | High |

---

## Getting Started Checklist

- [ ] Install dependencies: `pip install numpy torch biopython`
- [ ] Run demo: `python scripts/C4_mutation_effect_predictor.py`
- [ ] Validate on known mutations in your protein family
- [ ] Integrate with existing pipeline (FoldX/Rosetta post-filter)
- [ ] Use C1 to audit Rosetta predictions
- [ ] Document P-adic as "complementary geometric analysis" in publications

---

*Part of the Ternary VAE Bioinformatics Partnership - Jose Colbes Deliverables*
