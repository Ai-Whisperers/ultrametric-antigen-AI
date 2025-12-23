# Mathematical Research Program

**Doc-Type:** Research Agenda · Version 1.0 · Updated 2025-12-23 · Author AI Whisperers

Pure mathematical investigations using the 3-adic codon encoder (v5.11.3 / v1.1.0).

---

## Research Priorities

Two focused tracks, prioritized for maximum theoretical impact.

| Track | Hypothesis | Status |
|-------|-----------|--------|
| 1. Genetic Code Optimality | 64→21 degeneracy reflects optimal p-adic error-correction | Planned |
| 2. Unreasonable Effectiveness | p-adic geometry is native language of molecular evolution | Planned |

---

## Track 1: Genetic Code as Optimal P-Adic Code

**hypothesis** - The genetic code's 64→21 mapping approximates an optimal sphere-packing in p-adic space, not evolutionary accident.

### Background

The genetic code exhibits structured degeneracy:
- 64 codons → 20 amino acids + 1 stop signal
- Synonymous codons cluster by wobble position (3rd base)
- Similar amino acids often have similar codons
- Code is nearly universal across all life

### Testable Predictions

**P1** - Synonymous codons occupy minimal p-adic balls (tight clustering within amino acid groups)

**P2** - Biochemically similar amino acids have adjacent p-adic neighborhoods (graceful degradation under mutation)

**P3** - The code achieves theoretical bounds analogous to Hamming/Singleton limits in ultrametric space

**P4** - Random codes with same 64→21 structure perform worse on packing metrics

### Metrics to Compute

| Metric | Definition | Optimal Behavior |
|--------|-----------|------------------|
| Packing density | Volume ratio of codon balls to total space | Maximized |
| Covering radius | Max distance from any point to nearest codeword | Minimized |
| Minimum distance | Smallest inter-class p-adic distance | Maximized |
| Graceful degradation | Correlation between p-adic distance and biochemical similarity | Strong positive |

### Connections to Established Theory

**algebraic_coding_theory** - Reed-Solomon codes, BCH codes, algebraic geometry codes
**sphere_packing** - Kepler conjecture analogs in ultrametric spaces
**information_theory** - Channel capacity under ultrametric noise models

### Deliverables

- `genetic_code_optimality/` subdirectory
- Scripts computing packing metrics for natural vs random codes
- Comparison against theoretical bounds
- Publication-ready analysis

---

## Track 2: Unreasonable Effectiveness (Adversarial Validation)

**hypothesis** - The encoder's predictive success implies p-adic geometry is the native mathematical language of molecular evolution.

**methodology** - Unbiased adversarial approach testing both pro and counter arguments.

### Pro-Arguments to Validate

| Claim | Evidence | Validation Method |
|-------|----------|-------------------|
| High predictive accuracy | r=0.751 on HLA-disease association | Cross-validation, held-out diseases |
| Boundary semantics | 100% boundary-crossing for AA substitutions | Statistical significance vs null |
| Phylogenetic alignment | Ultrametric distances match evolutionary trees | Compare against known phylogenies |
| Cross-domain generalization | Works on RA, HIV, Tau, SARS-CoV-2 | Blind prediction on new systems |
| Structural validation | Predictions confirmed by AlphaFold3 | Correlation with experimental structures |

### Counter-Arguments to Test

| Challenge | Test |
|-----------|------|
| Euclidean sufficiency | Train equivalent model in Euclidean R^16, compare performance |
| Prime specificity | Test p=2, 5, 7, 11 encoders against p=3 |
| Overfitting to biology | Test on shuffled/randomized biological data |
| Initialization dependence | Compare against randomly-initialized encoders |
| Dimensionality confound | Test if high-dim Euclidean matches hyperbolic |
| Confirmation bias | Blind predictions scored by independent evaluator |

### Null Models Required

**null_1** - Random codon→embedding mapping (same architecture, shuffled assignments)
**null_2** - Euclidean encoder (replace Poincare ball with R^16)
**null_3** - Alternative primes (p=2, 5, 7 encoders)
**null_4** - Random biological sequences (shuffled codons preserving composition)

### Success Criteria

**strong_support** - p=3 hyperbolic significantly outperforms all null models (p < 0.01)
**weak_support** - p=3 hyperbolic outperforms but not significantly
**refutation** - Null models achieve comparable or better performance

### Deliverables

- `unreasonable_effectiveness/` subdirectory
- Null model implementations
- Comparative benchmark suite
- Statistical analysis framework
- Publication-ready findings (regardless of outcome)

---

## Shared Infrastructure

**model** - 3-Adic Codon Encoder v5.11.3
**path** - `research/genetic_code/data/codon_encoder_3adic.pt`
**architecture** - nn.Sequential(Linear(12→64) → ReLU → ReLU → Linear(64→16)) + clustering head
**embedding** - 16-dimensional Poincare ball, curvature c=1.0

**utilities** - `research/bioinformatics/codon_encoder_research/rheumatoid_arthritis/scripts/hyperbolic_utils.py`

---

## Timeline

| Phase | Focus | Dependencies |
|-------|-------|--------------|
| Phase 1 | Document hypotheses and metrics | None |
| Phase 2 | Implement null models | Phase 1 |
| Phase 3 | Run comparative benchmarks | Phase 2 |
| Phase 4 | Statistical analysis | Phase 3 |
| Phase 5 | Write findings | Phase 4 |

---

## References

**p_adic_coding** - Voloch, J.F. "Codes over p-adic numbers and finite fields"
**sphere_packing** - Conway & Sloane "Sphere Packings, Lattices and Groups"
**genetic_code_theory** - Freeland & Hurst "The Genetic Code is One in a Million" (2004)
**hyperbolic_ml** - Nickel & Kiela "Poincare Embeddings for Learning Hierarchical Representations" (2017)

---

## Notes

- All research must be reproducible with provided model weights
- Negative results are equally valuable (falsification advances knowledge)
- Focus on mathematical rigor over biological interpretation
