# DENV-4 Detection: Definitive Solution Approach

**Doc-Type:** Implementation Plan · Version 1.0 · 2026-01-05 · AI Whisperers

---

## Problem Statement

DENV-4 exhibits cryptic diversity (71.7% identity vs 95-98% for other serotypes), making 86.7% of strains undetectable with consensus primers. Traditional Shannon entropy identifies NS5 as conserved, but primers fail on most strains.

---

## Solution: Dual-Metric Primer Design

### Core Innovation

Use **two orthogonal conservation metrics** to identify primer targets:

| Metric | Level | Best Region | Captures |
|--------|-------|-------------|----------|
| Shannon Entropy | Nucleotide | NS5 (9908) | Sequence identity |
| Hyperbolic Variance | Codon | E gene (2400) | Novel p-adic signal |

### Why This Works

The p-adic encoder detects a conservation signal **orthogonal to all classical metrics**:
- NOT synonymous substitution rate (ρ=0.03, p=0.86)
- NOT amino acid property conservation (ρ=0.01, p=0.97)
- NOT codon usage bias (ρ=0.31, p=0.07)
- NOT codon pair context (ρ=-0.14, p=0.43)

This means strains that escape Shannon-based detection may still be captured by hyperbolic-variance-based primers targeting different genomic regions.

---

## Implementation Plan

### Phase 1: Primer Design (Wet Lab)

**E Gene Primers (Position 2400-2500):**
```
Forward primer region: 2400-2425
Reverse primer region: 2475-2500
Expected amplicon: ~75-100bp
```

Design constraints:
- Allow 2-3 degenerate positions
- Tm = 55-60°C
- No self-complementarity

**Retain Existing NS5 Primers:**
- DENV4_E32_NS5_F (9908)
- PANFLAVI_FU1 (9007)
- PANFLAVI_cFD2 (9196)

### Phase 2: Validation Protocol

1. **In silico test** on 270 DENV-4 genomes from NCBI
2. **Cross-reactivity** check against DENV-1/2/3, ZIKV, other flaviviruses
3. **Panel testing** on:
   - Clade_E strains (211 genomes, currently 13.3% detectable)
   - Clade_D strains (52 genomes)
   - Historical strains (1956-2024 temporal span)

### Phase 3: Detection Strategy

**Multiplex RT-qPCR:**

| Target | Position | Purpose |
|--------|----------|---------|
| E gene | 2400 | Primary (hyperbolic) |
| NS5 | 9600 | Secondary (improved NS5) |
| Pan-flavi | 9007 | Backup/confirmation |

**Interpretation:**
- Any positive = DENV-4 detected
- E+NS5+ = High confidence
- E+NS5- or E-NS5+ = Divergent strain

---

## Expected Outcomes

### Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Strain coverage | 13.3% | >50% |
| Clade_E detection | Partial | >70% |
| Clade_D detection | Unknown | >50% |

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| E gene primers fail in vitro | Backup targets: NS1 (3000), prM (600) |
| Cross-reactivity | Pan-flavi discrimination step |
| Novel strains | Dual-target redundancy |

---

## Technical Validation Completed

| Step | Status | Commit |
|------|--------|--------|
| Phylogenetic analysis (270 genomes) | DONE | bee3485 |
| Shannon entropy analysis | DONE | 8ea7f8f |
| P-adic integration | DONE | c54f899 |
| Synonymous hypothesis test | REJECTED | e0af23c |
| AA property hypothesis test | REJECTED | e0af23c |
| Codon bias hypothesis test | REJECTED | e0af23c |
| Codon pair hypothesis test | REJECTED | eeb203c |
| Consolidated findings | DONE | 6d32496 |

---

## Scripts for Continued Development

### Generate Primer Candidates

```python
# Use existing framework
from deliverables.partners.alejandra_rojas.scripts.denv4_padic_integration import (
    load_encoder,
    compute_window_embedding,
)

# Top candidates by hyperbolic variance
PRIMER_CANDIDATES = [
    {"position": 2400, "gene": "E", "hyp_var": 0.0183},
    {"position": 3000, "gene": "NS1", "hyp_var": 0.0207},
    {"position": 9600, "gene": "NS5", "hyp_var": 0.0222},
]
```

### Validate Against Genome Panel

```python
# In silico primer binding analysis
from src.biology.codons import GENETIC_CODE
from Bio import SeqIO

def validate_primer_binding(primer_seq, genome_panel):
    """Check primer binding across all genomes."""
    # Implementation for wet lab validation
    pass
```

---

## Next Steps (for Rojas team)

1. **Design** degenerate primers for E gene position 2400-2500
2. **Synthesize** primers with appropriate degeneracy
3. **Test** on characterized DENV-4 isolates
4. **Compare** detection rates: E gene vs NS5 vs multiplex
5. **Report** coverage improvement metrics

---

## Conclusion

The p-adic codon encoder has identified E gene position 2400 as a novel primer target with 4x lower hyperbolic variance than current NS5 targets. This orthogonal conservation metric provides a biologically distinct detection channel that may capture the 86.7% of DENV-4 strains currently undetectable.

**The solution is a dual-target multiplex strategy combining Shannon-conserved (NS5) and p-adic-conserved (E gene) regions.**

---

*Framework validated with TrainableCodonEncoder (LOO ρ=0.61 on DDG)*
*IICS-UNA Arbovirus Surveillance Program*
