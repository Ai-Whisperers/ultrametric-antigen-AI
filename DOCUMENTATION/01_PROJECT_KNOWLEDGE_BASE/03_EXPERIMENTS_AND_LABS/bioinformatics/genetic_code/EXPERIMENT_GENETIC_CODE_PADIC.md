# Experiment Plan: Genetic Code as P-Adic Error-Correcting Code

**Doc-Type:** Experiment Plan · Version 1.0 · Updated 2025-12-16

---

## Hypothesis

**The genetic code (64 codons → 20 amino acids) is a p-adic error-correcting code, and the v1.1.0 Ternary VAE has reverse-engineered this structure.**

If true, this would:
1. Explain WHY evolution converged on this specific degeneracy pattern
2. Provide a mathematical foundation for optimal codon design
3. Enable a universal mRNA design platform with provable optimality

---

## Current Evidence (from v1.1.0 analysis)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Synonymous clustering p-value | **6.77e-05** | Codons for same AA cluster together |
| BLOSUM correlation | **r = -0.106** | Similar AAs have smaller embedding distance |
| Chemical class ANOVA | **p = 0.018** | Charged/polar/nonpolar separate by radius |
| Charged AA radius | **0.405** | Closer to origin (high valuation?) |
| Nonpolar AA radius | **0.492** | Closer to boundary (low valuation?) |

---

## Experiment Checklist

### Phase 1: P-Adic Ball Structure (Core Hypothesis)

- [ ] **1.1 Compute pairwise geodesic distances for all 64 codons**
  - Use Poincare distance, not Euclidean
  - File: `07_genetic_code_padic.py`
  - Output: 64×64 geodesic distance matrix

- [ ] **1.2 Test if synonymous codons form p-adic balls**
  - For each amino acid with k codons, check if all k lie within geodesic ε of each other
  - Compute ε_within (max intra-AA distance) vs ε_between (min inter-AA distance)
  - **SUCCESS CRITERION:** ε_within < ε_between for most amino acids

- [ ] **1.3 Measure ultrametric property for codon triplets**
  - For triplets (c1, c2, c3) of codons, check: d(c1,c3) ≤ max(d(c1,c2), d(c2,c3))
  - Count violations
  - **SUCCESS CRITERION:** <5% violations (random would give ~50%)

- [ ] **1.4 Correlate amino acid radius with degeneracy**
  - Degeneracy = number of codons per amino acid (1-6)
  - **PREDICTION:** High degeneracy → closer to origin (more "fundamental")

### Phase 2: Wobble Position Analysis

- [ ] **2.1 Analyze 3rd position (wobble) tolerance**
  - For each codon pair differing only in 3rd position, measure embedding distance
  - Compare to pairs differing in 1st or 2nd position
  - **PREDICTION:** 3rd position changes have smallest embedding distance

- [ ] **2.2 Map wobble tolerance to 3-adic valuation**
  - Compute v_3(codon_index) for all 64 codons
  - Correlate with position-specific mutation tolerance
  - **PREDICTION:** High v_3 codons are more wobble-tolerant

- [ ] **2.3 Test if wobble = p-adic neighborhood**
  - Define "wobble neighborhood" as codons differing only in 3rd position
  - Check if these form contiguous regions in embedding space

### Phase 3: Error-Correction Properties

- [ ] **3.1 Compute minimum distance between AA equivalence classes**
  - d_min(AA_i, AA_j) = min distance between any codon of AA_i and any of AA_j
  - Build 20×20 amino acid distance matrix from codon embeddings

- [ ] **3.2 Compare to BLOSUM evolutionary distances**
  - Correlate embedding-derived AA distances with BLOSUM62 scores
  - **PREDICTION:** Strong negative correlation (similar AAs cluster)

- [ ] **3.3 Test Hamming-like error detection**
  - For random single-nucleotide mutations, measure embedding displacement
  - **PREDICTION:** Synonymous mutations stay within p-adic ball, non-synonymous jump between balls

- [ ] **3.4 Compute "code distance" in p-adic terms**
  - Minimum geodesic distance between different AA classes
  - Compare to theoretical error-correcting code bounds

### Phase 4: mRNA Stability Integration

- [ ] **4.1 Obtain codon stability data**
  - Source: Presnyak et al. (2015) or similar mRNA half-life studies
  - Map stability scores to our 64 codons

- [ ] **4.2 Correlate embedding position with stability**
  - **PREDICTION:** Codons near origin = more stable
  - **RATIONALE:** "Fundamental" codons (high valuation) may be evolutionarily optimized

- [ ] **4.3 Test optimal codon selection**
  - For each amino acid, identify "optimal" codon (highest stability)
  - Check if optimal codons cluster in specific embedding region

- [ ] **4.4 Build codon optimization function**
  - Input: protein sequence
  - Output: optimized mRNA sequence maximizing "hyperbolic centrality"
  - Validate against existing codon optimization tools

### Phase 5: RNA Secondary Structure

- [ ] **5.1 Integrate RNAfold predictions**
  - For short sequences, compute MFE (minimum free energy)
  - Test if embedding distance correlates with MFE similarity

- [ ] **5.2 Test sequence context effects**
  - Do adjacent codons interact in embedding space?
  - Is there a "codon bigram" structure?

---

## Data Sources

| Data | Source | Format |
|------|--------|--------|
| v1.1.0 embeddings | `embeddings/embeddings.pt` | torch tensor (19683, 16) |
| Genetic code | Built-in GENETIC_CODE dict | 64 codons |
| BLOSUM62 | Built-in matrix | 20×20 similarity |
| Codon usage | Kazusa database | Per-organism frequencies |
| mRNA stability | Presnyak et al. (2015) | Half-life measurements |
| AA properties | Kyte-Doolittle scale | Hydrophobicity values |

---

## Success Metrics

### Tier 1: Proof of Concept (Phase 1-2)
- Synonymous codons form geodesic balls: **p < 0.001**
- Ultrametric violations: **<5%**
- Wobble position has smallest distance: **p < 0.01**

### Tier 2: Biological Validation (Phase 3-4)
- BLOSUM correlation from embeddings: **r > 0.5**
- Stability correlation with radius: **p < 0.05**
- Error detection rate for mutations: **>80%**

### Tier 3: Commercial Viability (Phase 5+)
- Codon optimizer outperforms baseline tools
- mRNA half-life prediction: **r > 0.3**
- Patent-worthy novel algorithm

---

## Implementation Order

```
Phase 1.1-1.2  →  Core p-adic ball test (1-2 hours)
     ↓
Phase 2.1-2.2  →  Wobble analysis (1 hour)
     ↓
Phase 1.3-1.4  →  Ultrametric + degeneracy (1 hour)
     ↓
Phase 3.1-3.3  →  Error-correction (2 hours)
     ↓
Phase 4.1-4.3  →  mRNA integration (2-4 hours, requires data)
     ↓
Phase 5        →  RNA structure (optional, requires RNAfold)
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `07_genetic_code_padic.py` | Main experiment script |
| `08_wobble_analysis.py` | Wobble position deep-dive |
| `09_mrna_stability.py` | mRNA half-life integration |
| `results/genetic_code_padic.json` | Numerical results |
| `results/genetic_code_balls.png` | Visualization of p-adic balls |
| `DISCOVERY_GENETIC_CODE.md` | Findings documentation |

---

## Dependencies

```python
# Core (already installed)
torch, numpy, scipy, matplotlib

# Optional for Phase 4-5
biopython          # Sequence handling
ViennaRNA          # RNAfold integration (optional)
requests           # Kazusa database access
```

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Null result (no p-adic structure) | Medium | High | Still publishable as negative result |
| Encoding artifacts dominate | Medium | Medium | Test multiple codon→ternary mappings |
| Insufficient statistical power | Low | Medium | 64 codons is small but sufficient |
| mRNA data unavailable | Low | Low | Use published datasets |

---

## Next Action

**Start with Phase 1.1-1.2:** Implement geodesic distance matrix and test if synonymous codons form p-adic balls. This is the core hypothesis test and takes ~2 hours.

---

**Status:** Ready for implementation
