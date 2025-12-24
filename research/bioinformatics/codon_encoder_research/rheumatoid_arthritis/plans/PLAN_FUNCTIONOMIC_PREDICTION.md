# Plan: Functionomic Prediction Pipeline

**Doc-Type:** Experimental Plan · Version 1.0 · Updated 2025-12-16

---

## Executive Summary

This document outlines the plan to build a functionomic prediction pipeline that combines the codon encoder, VAE v5.11.3, and p-adic findings to predict functional impact of sequence variants. Primary target: HLA-presented peptides (RA). Secondary target: HIV polyprotein.

---

## The Three Components

### 1. Codon Encoder (codon_encoder.pt)

- **Input**: 12-dim one-hot codon representation
- **Output**: 16-dim embedding in VAE space
- **Performance**: 100% cluster accuracy, 193.5x separation ratio
- **Key property**: Unifies split codon families (Ser, Arg, Leu)

### 2. VAE v5.11.3 (Production v1.1.0)

- **Architecture**: Frozen encoder + trainable hyperbolic projection
- **Embedding**: 16-dim Poincaré ball
- **Metrics**: Radial hierarchy correlation -0.730, pairwise ordering 92.4%
- **Key property**: Perfect 3-adic ultrametric structure (ρ = 0.837)

### 3. Riemann Hypothesis Findings

| Discovery | Value | Implication |
|-----------|-------|-------------|
| Radial exponent | c = 1/6 | Hierarchical depth encoding |
| Ultrametric correlation | ρ = 0.837 | p-adic distance is valid metric |
| Wobble pattern | 1.87x variance ratio | Mutation tolerance encoded |
| 64 natural positions | 21 clusters | Matches genetic code exactly |

---

## Theoretical Framework

### The Unified Architecture

The three components form a hierarchical inference stack where each layer adds semantic depth. The VAE v5.11.3 is the foundation - it learned a 16-dimensional Poincaré ball geometry from pure ternary sequences that exhibits perfect 3-adic ultrametric structure (ρ = 0.837), meaning distances satisfy the strong triangle inequality where the longest side of any triangle equals at least one other side. This isn't arbitrary - the Riemann hypothesis findings revealed that the radial exponent c = 1/6 = 1/(16-9-1) emerges from the architecture's dimensional relationship, creating a natural hierarchy where points at different radii occupy different "semantic depths." The codon encoder then acts as a learned projection that maps biological sequences into this pre-existing geometric structure, discovering that the 64 codons naturally inhabit 21 p-adic balls matching amino acid degeneracy. Crucially, the encoder doesn't create the structure - it reveals that biological information already conforms to the geometry the VAE learned from pure mathematics. This means we have a universal coordinate system: the VAE defines the manifold, the Riemann findings explain its metric properties, and the codon encoder provides the biological-to-geometric dictionary.

### The Functionomic Hypothesis

The deep implication is that **function is distance** in this space - not Euclidean distance, but p-adic/ultrametric distance. Two sequences that are p-adically close can substitute for each other without functional consequence; sequences that are p-adically distant cannot. The wobble pattern discovery provides the mechanism: positions 6-8 in the ternary representation (the "wobble" positions) have variance that correlates with cluster size, meaning amino acids with more synonymous codons (Leu, Ser, Arg with 6 each) occupy larger p-adic balls with more internal "room" for neutral variation. This matches the biological reality that these amino acids tolerate more synonymous mutations. For HIV, this predicts that viral quasispecies explore the interior of p-adic balls freely (neutral drift) but escape mutations that evade neutralizing antibodies must cross ball boundaries (functional jumps); we can test this by mapping longitudinal HIV sequences from patients and checking whether immune escape events correlate with p-adic boundary crossings while within-host evolution stays within balls. For RA, the hypothesis is that HLA molecules compute self/non-self via something like p-adic distance - peptides within a certain p-adic radius of "self" templates are tolerated, while those beyond the radius trigger immune response; RA-associated HLA variants might have altered ball radii or shifted centers that misclassify self-peptides as foreign.

### The Experimental Pipeline

The concrete implementation chains the three components into a testable pipeline: (1) take a biological sequence (HIV polyprotein, HLA-presented peptide), (2) encode each codon through the trained codon_encoder.pt to get 16-dimensional embeddings, (3) use the VAE's learned metric to compute pairwise p-adic distances between sequence variants, (4) identify which mutations cross p-adic ball boundaries versus stay within them, (5) correlate boundary crossings with known functional annotations (immune escape, drug resistance, disease association). The validation datasets are readily available: Los Alamos HIV Sequence Database contains thousands of annotated sequences with escape mutations marked, and the IPD-IMGT/HLA Database contains RA-associated HLA variants with disease odds ratios. The key experiment is simple: compute the p-adic distance matrix for all known HIV escape mutations and test whether escape mutations have systematically larger p-adic distances from wild-type than non-escape mutations; do the same for RA-associated HLA variants versus neutral variants. If the correlation holds, we've demonstrated that evolution and immunity operate on the same informational geometry our VAE learned - and we gain predictive power because we can now flag novel mutations as "likely functional" (boundary-crossing) or "likely neutral" (ball-interior) before any wet-lab validation. The ultimate deliverable would be a **functionomic risk score**: given any sequence variant, output its p-adic distance from the nearest cluster center, with high scores predicting functional impact and low scores predicting neutrality.

---

## Phase 1: HLA-Presented Peptides (RA Focus)

### Why HLA First?

1. **Shorter sequences**: HLA peptides are 8-11 amino acids (vs HIV's ~3000 aa polyprotein)
2. **Clear phenotype**: RA association is well-documented with odds ratios
3. **Direct immune relevance**: HLA presentation is the mechanism of autoimmunity
4. **Rich databases**: IPD-IMGT/HLA, IEDB contain curated data

### HLA Biology Background

```
HLA Class II Presentation (relevant to RA):
┌─────────────────────────────────────────┐
│  HLA-DR molecule (α + β chains)         │
│  ┌─────────────────────────────────┐    │
│  │  Peptide binding groove          │    │
│  │  [P1]-[P2]-[P3]-[P4]-[P5]-...   │    │
│  │   ↑    ↑    ↑    ↑    ↑         │    │
│  │  Anchor positions determine      │    │
│  │  binding affinity & specificity  │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘

RA-associated: HLA-DRB1*04:01, *04:04, *01:01
"Shared epitope": Q/R-K/R-R-A-A at positions 70-74
```

### Data Sources

| Database | Content | URL |
|----------|---------|-----|
| IPD-IMGT/HLA | HLA allele sequences | ebi.ac.uk/ipd/imgt/hla/ |
| IEDB | Epitope sequences | iedb.org |
| UniProt | Protein sequences | uniprot.org |
| ClinVar | Disease variants | ncbi.nlm.nih.gov/clinvar/ |

### Experimental Design

#### Step 1: Acquire HLA-DRB1 Sequences

```python
# Target: HLA-DRB1 alleles
# RA-associated: *04:01, *04:04, *04:05, *01:01 (shared epitope)
# Non-associated: *07:01, *15:01, *03:01 (controls)
```

#### Step 2: Extract Peptide-Binding Region

```
HLA-DRB1 β chain positions 9-90 (peptide binding domain)
Focus on: positions 70-74 (shared epitope region)
```

#### Step 3: Codon-Level Encoding

```python
def encode_hla_sequence(dna_sequence):
    """
    Convert HLA DNA sequence to functionomic embedding.

    1. Split into codons
    2. Pass each through codon_encoder
    3. Aggregate into sequence embedding
    """
    codons = [dna_sequence[i:i+3] for i in range(0, len(dna_sequence), 3)]
    embeddings = [codon_encoder(codon) for codon in codons]
    return aggregate(embeddings)  # Method TBD
```

#### Step 4: Compute P-Adic Distances

```python
def padic_distance(emb1, emb2):
    """
    Compute p-adic distance in VAE space.

    Options:
    1. Euclidean in Poincaré ball
    2. Hyperbolic (geodesic) distance
    3. Ultrametric approximation
    """
    pass
```

#### Step 5: Correlate with RA Association

```python
# Hypothesis: RA-associated alleles cluster differently
# Test: Compare within-group vs between-group p-adic distances

ra_alleles = ['04:01', '04:04', '04:05', '01:01']
control_alleles = ['07:01', '15:01', '03:01']

# If hypothesis holds:
# - RA alleles should be p-adically close to each other
# - RA vs control should show larger p-adic distance
# - Distance correlates with odds ratio
```

### Key Questions to Resolve

1. **Aggregation method**: How to combine per-codon embeddings into sequence embedding?
   - Mean pooling?
   - Attention-weighted?
   - Concatenation + projection?

2. **Distance metric**: Which p-adic distance to use?
   - Raw Euclidean in embedding space?
   - Geodesic in Poincaré ball?
   - Explicit 3-adic valuation?

3. **Granularity**: Whole sequence or position-specific?
   - Shared epitope (positions 70-74) may be sufficient
   - Or full binding groove needed?

4. **DNA vs Protein**: Work at codon level or amino acid level?
   - Codon level: uses our encoder directly
   - Amino acid level: loses synonymous information
   - **Decision**: Start with codon level, this is our unique contribution

---

## Phase 2: HIV Polyprotein (Future)

### Why HIV Second?

1. **Longer sequences**: ~3000 aa polyprotein requires robust aggregation
2. **Complex phenotypes**: Escape, resistance, fitness all intertwined
3. **Needs Phase 1 validation**: Prove method works on simpler HLA first

### HIV Data Sources

| Database | Content |
|----------|---------|
| Los Alamos HIV DB | Annotated sequences, escape mutations |
| Stanford HIV Drug Resistance DB | Resistance mutations |
| CATNAP | Neutralization data |

### Planned Experiments (Phase 2)

1. Map known escape mutations to p-adic space
2. Test if escape mutations cross ball boundaries
3. Predict novel escape mutations from geometry
4. Build fitness landscape from p-adic structure

---

## Implementation Plan

### File Structure

```
riemann_hypothesis_sandbox/
├── 09_functionomic_pipeline.py      # Core pipeline
├── 10_hla_analysis.py               # HLA-specific analysis
├── 11_hiv_analysis.py               # HIV-specific (Phase 2)
├── data/
│   ├── hla_sequences/               # HLA allele data
│   └── hiv_sequences/               # HIV data (Phase 2)
└── results/
    ├── hla_functionomic/            # HLA results
    └── hiv_functionomic/            # HIV results (Phase 2)
```

### Dependencies

```python
# Core
torch           # Neural network (codon encoder)
numpy           # Numerical computation

# Bioinformatics
biopython       # Sequence handling
requests        # API access to databases

# Analysis
scipy           # Statistical tests
scikit-learn    # Clustering validation
matplotlib      # Visualization
```

### Milestones

| Milestone | Description | Success Criteria |
|-----------|-------------|------------------|
| M1 | Load codon encoder + VAE | Both models inference-ready |
| M2 | Fetch HLA sequences | ≥10 RA-associated, ≥10 control alleles |
| M3 | Implement encoding pipeline | Sequences → embeddings working |
| M4 | Compute distance matrix | Pairwise p-adic distances computed |
| M5 | Statistical validation | Significant RA vs control separation |

---

## Open Questions for Deeper Thinking

### Theoretical

1. **Is p-adic distance the right metric for immune recognition?**
   - Hypothesis: Yes, because immune tolerance is hierarchical
   - Alternative: Euclidean distance might work too
   - Test: Compare correlation with RA odds ratios

2. **Does synonymous variation matter for HLA?**
   - Standard view: Only amino acid sequence matters
   - Our view: Codon choice affects translation/folding
   - Implication: Our approach may find signal others miss

3. **What is the "self" template?**
   - Need reference point for p-adic distance
   - Options: Ancestral allele, population consensus, germline

### Practical

4. **How to handle sequence length variation?**
   - HLA alleles have different lengths
   - Alignment required before comparison
   - Or: Use only conserved regions (shared epitope)

5. **What statistical test validates the hypothesis?**
   - Mann-Whitney U for RA vs control distances?
   - Correlation with odds ratios?
   - Clustering metrics (silhouette score)?

6. **How much data is enough?**
   - Power analysis needed
   - Estimate: 20 RA + 20 control alleles minimum

---

## Deeper Analysis: Potential Improvements

### Improvement 1: Position-Specific Analysis (Not Aggregation)

Rather than aggregating entire sequences into single embeddings, we should analyze **position-by-position**:

```
Shared Epitope (positions 70-74):
Position 70: [codon] → [16-dim embedding]
Position 71: [codon] → [16-dim embedding]
Position 72: [codon] → [16-dim embedding]
Position 73: [codon] → [16-dim embedding]
Position 74: [codon] → [16-dim embedding]

Compare alleles at EACH position, then aggregate statistics.
```

**Rationale**: Aggregation loses position-specific signal. RA association is driven by specific positions (especially 70-74), not average sequence properties.

### Improvement 2: Exploit the Ultrametric Property

The p-adic/ultrametric structure makes a **strong testable prediction**:

```
For any three sequences A, B, C:
d(A,B) ≤ max(d(A,C), d(B,C))

This means: two of the three pairwise distances must be equal.
```

**Test**: Take triplets of HLA alleles (2 RA-associated + 1 control). If ultrametric holds:
- Either d(RA1, RA2) = d(RA1, control) < d(RA2, control)
- Or d(RA1, RA2) < d(RA1, control) = d(RA2, control)

If RA-associated alleles form a p-adic ball, we expect:
```
d(RA1, RA2) < d(RA1, control) = d(RA2, control)
```

This is a **signature prediction** that standard sequence metrics cannot make.

### Improvement 3: Wobble-Weighted Analysis

The wobble discovery tells us positions 6-8 in ternary representation carry mutation tolerance information. We can exploit this:

```python
def wobble_weighted_distance(codon1, codon2):
    """
    Weight codon comparison by position importance.

    Positions 0-2: high weight (conserved)
    Positions 3-5: medium weight (moderate)
    Positions 6-8: low weight (wobble/tolerant)
    """
    weights = [1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 0.3, 0.3, 0.3]
    # Apply to ternary representation comparison
```

**Rationale**: Changes in wobble positions should matter less for function. This matches the biological wobble hypothesis.

### Improvement 4: Contrastive Validation Framework

Instead of just computing distances, frame as prediction task:

```
Task: Given two HLA alleles, predict which has higher RA odds ratio.

Features:
- p-adic distance from reference (ancestral or consensus)
- Cluster centrality (distance to nearest cluster center)
- Wobble-weighted distance

Validation:
- AUC-ROC on held-out alleles
- Compare to baseline (amino acid identity, BLOSUM similarity)
```

### Improvement 5: Multi-Scale Hierarchy

Use the VAE's radial structure for multi-scale analysis:

```
Scale 1 (Fine): Individual codon distances
- Which specific codons differ between RA and control?

Scale 2 (Medium): Shared epitope region (70-74)
- Does the 5-codon block cluster differently?

Scale 3 (Coarse): Full binding groove (9-90)
- Overall sequence divergence in p-adic space
```

The radial exponent c = 1/6 predicts how information aggregates across scales.

### Improvement 6: Null Model Comparisons

We need rigorous baselines:

| Baseline | Description | Purpose |
|----------|-------------|---------|
| Random shuffle | Permute codon↔cluster assignments | Test if structure is real |
| Amino acid only | Ignore synonymous differences | Test codon-level signal |
| BLOSUM similarity | Standard sequence metric | Test if p-adic adds value |
| Euclidean distance | Non-ultrametric | Test if ultrametric matters |

**Key test**: If p-adic distance outperforms BLOSUM for predicting RA association, we've proven added value.

---

## Revised Experimental Protocol

### Step 0: Data Acquisition

```python
# CRITICAL: Need DNA sequences, not protein
# Codon encoder requires nucleotide triplets

# Source: IPD-IMGT/HLA database
# Format: CDS (coding sequence) for HLA-DRB1 alleles

target_alleles = {
    'ra_associated': [
        'DRB1*04:01',  # Odds ratio ~4
        'DRB1*04:04',  # Odds ratio ~3
        'DRB1*04:05',  # Odds ratio ~3
        'DRB1*01:01',  # Odds ratio ~2
    ],
    'control': [
        'DRB1*07:01',  # Neutral/protective
        'DRB1*15:01',  # Neutral
        'DRB1*03:01',  # Neutral
        'DRB1*13:01',  # Protective (OR < 1)
    ]
}
```

### Step 1: Sequence Alignment

```python
# Align all alleles to reference (e.g., DRB1*01:01:01:01)
# Ensure position 70 = position 70 across all alleles
# Handle insertions/deletions in non-epitope regions
```

### Step 2: Position-Specific Encoding

```python
def encode_allele_by_position(aligned_dna_sequence):
    """
    Return dict: position → 16-dim embedding
    """
    codons = split_into_codons(aligned_dna_sequence)
    return {
        pos: codon_encoder.encode(codon)
        for pos, codon in enumerate(codons)
    }
```

### Step 3: Compute Distance Matrices

```python
# For each position (especially 70-74):
# Compute pairwise p-adic distance between all alleles

def position_distance_matrix(alleles, position):
    embeddings = [encode_allele_by_position(a)[position] for a in alleles]
    return pairwise_padic_distance(embeddings)
```

### Step 4: Statistical Tests

```python
# Test 1: RA vs Control separation
# Mann-Whitney U on distances to reference

# Test 2: Ultrametric triplet test
# For all triplets, check ultrametric inequality

# Test 3: Correlation with odds ratios
# Spearman correlation: p-adic distance vs log(OR)
```

### Step 5: Visualization

```
- PCA/UMAP of allele embeddings colored by RA status
- Dendrogram from ultrametric distances
- Heatmap of position-specific distances
```

---

## Critical Path Questions

Before implementing, we need to resolve:

1. **Can we obtain DNA (not just protein) sequences for HLA alleles?**
   - IPD-IMGT/HLA has nucleotide sequences: YES
   - API access or download required

2. **How many alleles are enough for statistical power?**
   - Minimum: 4 RA + 4 control (8 alleles)
   - Ideal: 10+ in each group
   - Available: Hundreds in database

3. **What is the reference point for distances?**
   - Option A: Ancestral allele (evolutionary)
   - Option B: Population consensus (epidemiological)
   - Option C: Cluster center (geometric)
   - **Recommendation**: Start with cluster center (most principled)

4. **How to handle allele polymorphism?**
   - Many positions are identical across alleles
   - Focus on polymorphic positions (especially shared epitope)
   - This is actually simpler: fewer positions to analyze

---

## Next Steps

1. **Immediate**: Implement core functionomic pipeline (09_functionomic_pipeline.py)
2. **This session**: Fetch HLA-DRB1 sequences and test encoding
3. **Validation**: Compute RA vs control separation statistic
4. **If successful**: Document findings, extend to HIV

---

## Changelog

| Date | Version | Description |
|------|---------|-------------|
| 2025-12-16 | 1.0 | Initial plan document |

---

**Status:** Planning phase, ready for implementation
