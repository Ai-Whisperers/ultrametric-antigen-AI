# Comprehensive Methodology Documentation

## P-adic Hyperbolic Geometry for HIV Evolution Analysis

**Version:** 1.0
**Last Updated:** December 25, 2025

---

## Table of Contents

1. [Theoretical Foundation](#1-theoretical-foundation)
2. [Data Sources and Preprocessing](#2-data-sources-and-preprocessing)
3. [Codon Encoding Pipeline](#3-codon-encoding-pipeline)
4. [Analysis Methods by Dataset](#4-analysis-methods-by-dataset)
5. [Statistical Framework](#5-statistical-framework)
6. [Computational Implementation](#6-computational-implementation)
7. [Validation and Quality Control](#7-validation-and-quality-control)

---

## 1. Theoretical Foundation

### 1.1 P-adic Number Theory Background

**Definition 1.1 (P-adic Valuation):**
For a prime p and integer n ≠ 0, the p-adic valuation v_p(n) is defined as:

```
v_p(n) = max{k ∈ Z≥0 : p^k | n}
```

For our analysis, we use p = 3 (ternary base), yielding the 3-adic valuation:

```
v_3(1) = 0, v_3(3) = 1, v_3(9) = 2, v_3(27) = 3, ...
v_3(2) = 0, v_3(6) = 1, v_3(18) = 2, ...
```

**Definition 1.2 (3-adic Metric):**
The 3-adic distance between integers m and n is:

```
d_3(m, n) = 3^(-v_3(m-n))
```

This metric has the ultrametric property:
```
d_3(x, z) ≤ max{d_3(x, y), d_3(y, z)}
```

### 1.2 Codon Space Representation

**Codon Encoding Scheme:**

Each nucleotide is assigned a ternary digit:
- A = 0
- C = 1
- G = 2
- T/U = encoded via position modulation

Each codon (triplet) is encoded as a base-3 integer:
```
codon_index = n1 × 3² + n2 × 3¹ + n3 × 3⁰
```

This creates a natural hierarchy where:
- Single nucleotide changes → small 3-adic distance
- Multiple changes → larger 3-adic distance
- Synonymous codons (same amino acid) cluster together

### 1.3 Hyperbolic Embedding

**Poincare Ball Model:**

The Poincare ball B^n = {x ∈ R^n : ||x|| < 1} with the Riemannian metric:

```
ds² = (2/(1-||x||²))² × (dx₁² + dx₂² + ... + dxₙ²)
```

**Hyperbolic Distance:**

For points x, y ∈ B^n:

```
d_H(x, y) = arccosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))
```

**Properties Relevant to Biological Analysis:**

1. **Exponential volume growth:** Hyperbolic space has exponentially more volume at the boundary, naturally accommodating hierarchical biological structures

2. **Tree-like geometry:** Evolutionary relationships form approximate trees, which embed with low distortion in hyperbolic space

3. **Radial position encodes depth:** Distance from origin corresponds to depth in the evolutionary hierarchy

### 1.4 Biological Rationale

**Why Hyperbolic Geometry for HIV?**

1. **Phylogenetic Structure:** HIV evolution forms tree-like structures that embed naturally in hyperbolic space

2. **Fitness Landscapes:** The exponential volume growth captures the vast sequence space with sparse fitness peaks

3. **Constraint Gradients:** Radial position naturally encodes evolutionary constraint (conserved = central, variable = peripheral)

4. **Hierarchical Organization:** Codon degeneracy creates hierarchical clusters that respect the genetic code

---

## 2. Data Sources and Preprocessing

### 2.1 Stanford HIV Drug Resistance Database (HIVDB)

**Source:** Stanford University HIV Drug Resistance Database (https://hivdb.stanford.edu/)

**Files Used:**
```
research/datasets/
├── stanford_hivdb_pi.txt    # 2,171 Protease Inhibitor records
├── stanford_hivdb_nrti.txt  # 1,867 NRTI records
├── stanford_hivdb_nnrti.txt # 2,270 NNRTI records
└── stanford_hivdb_ini.txt   # 846 Integrase Inhibitor records
```

**Data Structure:**

| Column | Description | Type |
|--------|-------------|------|
| SeqID | Unique sequence identifier | String |
| CompMutList | Composite mutation list (e.g., "D30N, M46I") | String |
| P1-P99 / RT1-RT560 / IN1-IN288 | Amino acid at each position | Char |
| FPV, ATV, IDV, ... | Fold-change values for each drug | Float |

**Preprocessing Steps:**

1. **Load and parse:** Read tab-delimited files with pandas
2. **Extract mutations:** Parse CompMutList into individual mutations
3. **Map to codons:** Use position information to extract codon context
4. **Calculate fold-change:** Geometric mean across tested drugs per class
5. **Quality filter:** Remove sequences with >10% ambiguous positions

**Code Reference:** `scripts/unified_data_loader.py::load_stanford_hivdb()`

### 2.2 LANL CTL Epitope Database

**Source:** Los Alamos National Laboratory HIV Immunology Database

**File Used:**
```
research/datasets/ctl_summary.csv
```

**Data Structure:**

| Column | Description | Type |
|--------|-------------|------|
| Epitope | Amino acid sequence | String |
| Protein | HIV protein (Gag, Pol, Env, etc.) | String |
| HXB2 start | Start position in HXB2 reference | Int |
| HXB2 end | End position in HXB2 reference | Int |
| HLA | HLA restriction (e.g., "A*02:01") | String |
| Subtype | HIV-1 subtype | String |

**Preprocessing Steps:**

1. **Load epitopes:** Read CSV with epitope sequences and annotations
2. **Parse HLA:** Standardize HLA nomenclature to supertype level
3. **Map positions:** Convert to unified HXB2 coordinate system
4. **Extract codons:** Generate codon sequences from epitope positions
5. **Filter valid:** Remove epitopes without complete position mapping

**Code Reference:** `scripts/unified_data_loader.py::load_lanl_ctl()`

### 2.3 CATNAP Neutralization Database

**Source:** Los Alamos CATNAP (Compile, Analyze and Tally NAb Panels)

**File Used:**
```
research/datasets/catnap_assay.txt
```

**Data Structure:**

| Column | Description | Type |
|--------|-------------|------|
| Antibody | Antibody name | String |
| Virus | Virus strain identifier | String |
| IC50 | 50% inhibitory concentration (μg/mL) | Float |
| IC80 | 80% inhibitory concentration (μg/mL) | Float |
| ID50 | 50% infectious dose | Float |

**Preprocessing Steps:**

1. **Load assay data:** Parse tab-delimited neutralization records
2. **Clean IC50 values:** Handle censored values (">X" → X, "<Y" → Y)
3. **Calculate geometric means:** For antibodies with multiple measurements
4. **Annotate epitope class:** Map known bnAbs to epitope classes
5. **Calculate breadth:** Proportion of viruses neutralized at IC50 < 50

**Code Reference:** `scripts/unified_data_loader.py::load_catnap()`

### 2.4 V3 Coreceptor Tropism Data

**Source:** Hugging Face Datasets (HIV_V3_coreceptor)

**Files Used:**
```
data/external/huggingface/HIV_V3_coreceptor/
├── train.parquet
├── test.parquet
└── validation.parquet
```

**Data Structure:**

| Column | Description | Type |
|--------|-------------|------|
| Sequence | V3 loop amino acid sequence | String |
| CCR5 | Uses CCR5 coreceptor | Boolean |
| CXCR4 | Uses CXCR4 coreceptor | Boolean |

**Preprocessing Steps:**

1. **Load parquet files:** Combine train/test/validation splits
2. **Standardize tropism:** Create binary labels from boolean columns
3. **Align sequences:** Align to V3 consensus for position mapping
4. **Extract position features:** Generate per-position embeddings
5. **Quality filter:** Remove sequences with gaps or ambiguous positions

**Code Reference:** `scripts/unified_data_loader.py::load_v3_coreceptor()`

### 2.5 Reference Coordinate System

**HXB2 Reference (GenBank: K03455)**

All positions are mapped to the HXB2 reference genome for cross-dataset integration:

```python
HXB2_REGIONS = {
    "5'LTR": (1, 634),
    "gag": (790, 2292),
    "pol": (2085, 5096),
    "vif": (5041, 5619),
    "vpr": (5559, 5850),
    "tat": (5831, 6045),  # Exon 1
    "rev": (5970, 6045),  # Exon 1
    "vpu": (6062, 6310),
    "env": (6225, 8795),
    "nef": (8797, 9417),
    "3'LTR": (9086, 9719),

    # Protein subregions
    "PR": (2253, 2549),   # Protease
    "RT": (2550, 3869),   # Reverse Transcriptase
    "IN": (4230, 5096),   # Integrase
    "gp120": (6225, 7758),
    "gp41": (7759, 8795),
    "V3": (7110, 7217),   # V3 loop
}
```

**Code Reference:** `scripts/position_mapper.py`

---

## 3. Codon Encoding Pipeline

### 3.1 Amino Acid to Codon Mapping

Since protein sequences don't preserve codon information, we use a representative codon for each amino acid based on frequency in HIV:

```python
HIV_CODON_TABLE = {
    'A': 'GCT',  # Alanine - most common HIV codon
    'C': 'TGT',  # Cysteine
    'D': 'GAT',  # Aspartic acid
    'E': 'GAA',  # Glutamic acid
    'F': 'TTT',  # Phenylalanine
    'G': 'GGT',  # Glycine
    'H': 'CAT',  # Histidine
    'I': 'ATT',  # Isoleucine
    'K': 'AAA',  # Lysine
    'L': 'CTT',  # Leucine
    'M': 'ATG',  # Methionine (only codon)
    'N': 'AAT',  # Asparagine
    'P': 'CCT',  # Proline
    'Q': 'CAA',  # Glutamine
    'R': 'AGA',  # Arginine
    'S': 'TCT',  # Serine
    'T': 'ACT',  # Threonine
    'V': 'GTT',  # Valine
    'W': 'TGG',  # Tryptophan (only codon)
    'Y': 'TAT',  # Tyrosine
    '*': 'TAA',  # Stop
}
```

### 3.2 3-adic Encoding Algorithm

```python
def encode_codon_3adic(codon: str) -> np.ndarray:
    """
    Encode a codon using 3-adic valuation.

    Parameters:
    -----------
    codon : str
        Three-letter nucleotide sequence (e.g., 'ATG')

    Returns:
    --------
    np.ndarray
        16-dimensional embedding vector in Poincare ball
    """
    # Nucleotide to ternary digit
    nuc_map = {'A': 0, 'C': 1, 'G': 2, 'T': 0, 'U': 0}

    # Calculate base-3 index
    idx = (nuc_map[codon[0]] * 9 +
           nuc_map[codon[1]] * 3 +
           nuc_map[codon[2]])

    # Calculate 3-adic valuation features
    features = []
    for k in range(4):
        v_k = count_divisibility(idx, 3**k)
        features.append(v_k)

    # Add positional encoding
    for pos in range(3):
        features.extend(position_embedding(codon[pos], pos))

    # Project to Poincare ball
    embedding = project_to_poincare(features)

    return embedding
```

### 3.3 Mutation Distance Calculation

```python
def calculate_mutation_distance(
    wt_codon: str,
    mut_codon: str
) -> dict:
    """
    Calculate hyperbolic distance between wild-type and mutant codons.

    Returns:
    --------
    dict with keys:
        - 'euclidean': Euclidean distance in embedding space
        - 'hyperbolic': Hyperbolic (Poincare) distance
        - 'radial_change': Change in radial position
        - 'angular_change': Change in angular position
    """
    wt_emb = encode_codon_3adic(wt_codon)
    mut_emb = encode_codon_3adic(mut_codon)

    # Euclidean distance
    euclidean = np.linalg.norm(wt_emb - mut_emb)

    # Hyperbolic distance (Poincare)
    norm_wt = np.linalg.norm(wt_emb)
    norm_mut = np.linalg.norm(mut_emb)
    diff_norm = np.linalg.norm(wt_emb - mut_emb)

    hyperbolic = np.arccosh(
        1 + 2 * diff_norm**2 /
        ((1 - norm_wt**2) * (1 - norm_mut**2))
    )

    # Radial change
    radial_change = norm_mut - norm_wt

    # Angular change
    if norm_wt > 0 and norm_mut > 0:
        cos_angle = np.dot(wt_emb, mut_emb) / (norm_wt * norm_mut)
        angular_change = np.arccos(np.clip(cos_angle, -1, 1))
    else:
        angular_change = 0.0

    return {
        'euclidean': euclidean,
        'hyperbolic': hyperbolic,
        'radial_change': radial_change,
        'angular_change': angular_change
    }
```

### 3.4 Sequence Embedding

For full sequences (e.g., epitopes), we compute aggregate embeddings:

```python
def embed_sequence(sequence: str) -> dict:
    """
    Embed an amino acid sequence.

    Returns:
    --------
    dict with:
        - 'mean_embedding': Mean of position embeddings
        - 'per_position': List of position embeddings
        - 'mean_radius': Mean radial position
        - 'spread': Standard deviation of radii
    """
    codons = [HIV_CODON_TABLE.get(aa, 'NNN') for aa in sequence]
    embeddings = [encode_codon_3adic(c) for c in codons]

    mean_emb = np.mean(embeddings, axis=0)
    radii = [np.linalg.norm(e) for e in embeddings]

    return {
        'mean_embedding': mean_emb,
        'per_position': embeddings,
        'mean_radius': np.mean(radii),
        'spread': np.std(radii)
    }
```

**Code Reference:** `scripts/codon_extraction.py`

---

## 4. Analysis Methods by Dataset

### 4.1 Drug Resistance Analysis

#### 4.1.1 Resistance-Distance Correlation

**Hypothesis:** Mutations conferring higher fold-change resistance traverse greater hyperbolic distances.

**Method:**
1. For each mutation in Stanford HIVDB, calculate hyperbolic distance from wild-type
2. Correlate distance with log10(fold-change) for each drug
3. Report Pearson correlation with 95% CI

**Statistical Test:** Pearson correlation with permutation-based p-value (n=10,000 permutations)

**Code:**
```python
def analyze_resistance_correlation(mutations_df):
    results = []
    for drug in DRUG_COLUMNS:
        valid = mutations_df[mutations_df[drug].notna()]
        r, p = scipy.stats.pearsonr(
            valid['hyperbolic_distance'],
            np.log10(valid[drug])
        )
        results.append({
            'drug': drug,
            'correlation': r,
            'p_value': p,
            'n': len(valid)
        })
    return pd.DataFrame(results)
```

#### 4.1.2 Primary vs Accessory Classification

**Hypothesis:** Primary mutations occupy different geometric regions than accessory mutations.

**Method:**
1. Annotate mutations as primary/accessory using literature definitions
2. Calculate radial position and boundary-crossing status
3. Train classifier using geometric features
4. Evaluate with cross-validation

**Features Used:**
- Radial position (distance from origin)
- Angular position (direction in embedding space)
- Boundary crossing count
- Neighbor density

**Code Reference:** `scripts/analyze_stanford_resistance.py`

### 4.2 CTL Escape Analysis

#### 4.2.1 HLA-Stratified Escape Velocity

**Definition:** Escape velocity = mean hyperbolic distance of escape mutations weighted by selection coefficient

**Method:**
1. Group epitopes by HLA restriction
2. Calculate mean embedding and spread for each epitope
3. Compare distributions across HLA types
4. Test for significant differences using Kruskal-Wallis

**Code:**
```python
def calculate_escape_velocity(epitopes_df):
    hla_groups = epitopes_df.groupby('hla_supertype')

    velocities = {}
    for hla, group in hla_groups:
        spreads = group['embedding_spread'].values
        velocities[hla] = {
            'mean_velocity': np.mean(spreads),
            'std_velocity': np.std(spreads),
            'n_epitopes': len(group)
        }

    # Kruskal-Wallis test
    groups = [g['embedding_spread'].values for _, g in hla_groups]
    stat, p = scipy.stats.kruskal(*groups)

    return velocities, stat, p
```

#### 4.2.2 Protein Constraint Topology

**Method:**
1. Calculate mean radial position for epitopes in each protein
2. Compare distributions using Dunn's test with Bonferroni correction
3. Correlate with known functional constraints

**Code Reference:** `scripts/analyze_ctl_escape_expanded.py`

### 4.3 Antibody Neutralization Analysis

#### 4.3.1 bnAb Sensitivity Signatures

**Method:**
1. Partition viruses by sensitivity (IC50 < 1 μg/mL vs > 50 μg/mL)
2. Compare geometric features of sensitive vs resistant viruses
3. Calculate discriminative score for each bnAb

**Geometric Features:**
- Epitope mean radius
- Epitope spread
- Glycan site positions
- Variable loop lengths

#### 4.3.2 Breadth-Centrality Correlation

**Hypothesis:** Antibodies targeting geometrically central epitopes have greater breadth.

**Method:**
1. Calculate breadth = fraction of viruses neutralized at IC50 < 50
2. Calculate epitope centrality from mean embedding radius
3. Correlate breadth with centrality

**Code Reference:** `scripts/analyze_catnap_neutralization.py`

### 4.4 Tropism Analysis

#### 4.4.1 CCR5 vs CXCR4 Separation

**Method:**
1. Embed all V3 sequences
2. Calculate centroid for CCR5 and CXCR4 groups
3. Measure centroid distance
4. Test significance with Mann-Whitney U

#### 4.4.2 Position-Specific Discrimination

**Method:**
1. For each V3 position, calculate embedding separation between tropisms
2. Rank positions by discriminative power
3. Validate against known tropism determinants (11/25 rule)

**Code:**
```python
def analyze_position_importance(v3_df):
    results = []
    for pos in range(1, 36):  # V3 positions
        col = f'pos_{pos}_embedding'

        r5 = v3_df[v3_df['is_r5']][col].values
        x4 = v3_df[v3_df['is_x4']][col].values

        # Calculate separation
        r5_centroid = np.mean(r5, axis=0)
        x4_centroid = np.mean(x4, axis=0)
        separation = np.linalg.norm(r5_centroid - x4_centroid)

        # Statistical test
        stat, p = scipy.stats.mannwhitneyu(
            [np.linalg.norm(e) for e in r5],
            [np.linalg.norm(e) for e in x4]
        )

        results.append({
            'position': pos,
            'separation': separation,
            'p_value': p
        })

    return pd.DataFrame(results)
```

**Code Reference:** `scripts/analyze_tropism_switching.py`

### 4.5 Cross-Dataset Integration

#### 4.5.1 Resistance-Immunity Trade-offs

**Method:**
1. Map resistance positions to HXB2 coordinates
2. Map CTL epitopes to HXB2 coordinates
3. Find overlaps where resistance position falls within epitope
4. Calculate trade-off score:

```
trade_off_score = log10(max_fold_change) × log10(n_hla_restrictions + 1)
```

#### 4.5.2 Vaccine Target Identification

**Scoring Function:**
```
vaccine_score = (
    0.4 × log10(n_hla + 1) +           # HLA breadth
    0.3 × (1 - escape_velocity) +       # Constraint
    0.2 × (1 - resistance_overlap) +    # No resistance
    0.1 × conservation_score            # Sequence conservation
)
```

**Code Reference:** `scripts/cross_dataset_integration.py`

---

## 5. Statistical Framework

### 5.1 Multiple Testing Correction

**Bonferroni Correction:**
For m comparisons at family-wise error rate α:
```
α_individual = α / m
```

**Benjamini-Hochberg FDR:**
For ranked p-values p_(1) ≤ p_(2) ≤ ... ≤ p_(m):
```
Reject H_i if p_(i) ≤ (i/m) × q
```

where q is the desired false discovery rate (typically 0.05 or 0.10).

### 5.2 Effect Size Metrics

**Cohen's d (continuous):**
```
d = (μ₁ - μ₂) / σ_pooled
```

Interpretation: |d| < 0.2 (small), 0.2-0.8 (medium), > 0.8 (large)

**Eta-squared (ANOVA):**
```
η² = SS_between / SS_total
```

Interpretation: η² < 0.01 (small), 0.01-0.06 (medium), > 0.14 (large)

### 5.3 Cross-Validation

**Stratified K-Fold:**
- K = 5 folds
- Stratification preserves class proportions
- Reported metrics: mean ± standard deviation

**Nested Cross-Validation (for hyperparameter tuning):**
- Outer loop: 5-fold evaluation
- Inner loop: 3-fold hyperparameter selection

### 5.4 Confidence Intervals

**Bootstrap 95% CI:**
```python
def bootstrap_ci(data, statistic, n_bootstrap=1000, alpha=0.05):
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(statistic(sample))

    lower = np.percentile(boot_stats, 100 * alpha/2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha/2))
    return lower, upper
```

---

## 6. Computational Implementation

### 6.1 Software Environment

**Python Version:** 3.11

**Core Dependencies:**
```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
torch>=2.0.0
biopython>=1.81
pyarrow>=12.0.0
```

### 6.2 Code Organization

```
scripts/
├── unified_data_loader.py    # Data loading utilities
├── position_mapper.py        # HXB2 coordinate mapping
├── codon_extraction.py       # Codon encoding functions
├── analyze_stanford_resistance.py
├── analyze_ctl_escape_expanded.py
├── analyze_catnap_neutralization.py
├── analyze_tropism_switching.py
├── cross_dataset_integration.py
└── run_complete_analysis.py  # Master orchestration
```

### 6.3 Running the Analysis

**Complete Pipeline:**
```bash
cd research/bioinformatics/codon_encoder_research/hiv/scripts
python run_complete_analysis.py
```

**Individual Analyses:**
```bash
python analyze_stanford_resistance.py
python analyze_ctl_escape_expanded.py
python analyze_catnap_neutralization.py
python analyze_tropism_switching.py
python cross_dataset_integration.py
```

### 6.4 Resource Requirements

| Analysis | Memory | Time | Output Size |
|----------|--------|------|-------------|
| Stanford Resistance | 4 GB | ~5 min | 50 MB |
| CTL Escape | 2 GB | ~3 min | 30 MB |
| CATNAP | 8 GB | ~10 min | 100 MB |
| Tropism | 2 GB | ~2 min | 20 MB |
| Integration | 4 GB | ~5 min | 50 MB |
| **Total** | **8 GB peak** | **~25 min** | **250 MB** |

---

## 7. Validation and Quality Control

### 7.1 Data Validation

**Input Validation Checks:**
1. Sequence length consistency
2. Valid amino acid characters
3. Position range verification
4. Missing value handling

**Code:**
```python
def validate_sequence(seq: str) -> tuple[bool, str]:
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY*')
    invalid = set(seq.upper()) - valid_aa
    if invalid:
        return False, f"Invalid characters: {invalid}"
    return True, "Valid"
```

### 7.2 Embedding Validation

**Unit Tests:**
1. Embedding dimensionality (16D)
2. Poincare ball constraint (||x|| < 1)
3. Distance metric properties (non-negativity, triangle inequality)
4. Reproducibility (deterministic encoding)

### 7.3 Statistical Validation

**Sanity Checks:**
1. Known relationships reproduced (e.g., 11/25 rule for tropism)
2. Effect sizes in expected ranges
3. Cross-validation stability (low variance across folds)

### 7.4 Known Limitations

1. **Codon approximation:** Using representative codons loses synonymous variation
2. **HXB2 bias:** Reference-based mapping may miss insertions/deletions
3. **Subtype B focus:** Most data from subtype B; generalization uncertain
4. **Static analysis:** Cross-sectional data, no temporal dynamics

---

## Appendix A: HXB2 Reference Coordinates

Complete gene and protein coordinates for position mapping.

## Appendix B: Drug Abbreviations

| Abbreviation | Full Name | Class |
|--------------|-----------|-------|
| FPV | Fosamprenavir | PI |
| ATV | Atazanavir | PI |
| IDV | Indinavir | PI |
| LPV | Lopinavir | PI |
| NFV | Nelfinavir | PI |
| SQV | Saquinavir | PI |
| TPV | Tipranavir | PI |
| DRV | Darunavir | PI |
| ABC | Abacavir | NRTI |
| AZT | Zidovudine | NRTI |
| D4T | Stavudine | NRTI |
| DDI | Didanosine | NRTI |
| FTC | Emtricitabine | NRTI |
| 3TC | Lamivudine | NRTI |
| TDF | Tenofovir | NRTI |
| DOR | Doravirine | NNRTI |
| EFV | Efavirenz | NNRTI |
| ETR | Etravirine | NNRTI |
| NVP | Nevirapine | NNRTI |
| RPV | Rilpivirine | NNRTI |
| BIC | Bictegravir | INSTI |
| CAB | Cabotegravir | INSTI |
| DTG | Dolutegravir | INSTI |
| EVG | Elvitegravir | INSTI |
| RAL | Raltegravir | INSTI |

## Appendix C: HLA Nomenclature

HLA types are reported using standard WHO nomenclature:
- Gene*allele_group:specific_allele
- Example: A*02:01

Supertypes group alleles with similar peptide binding:
- A2 supertype: A*02:01, A*02:02, A*02:03, ...
- B57 supertype: B*57:01, B*57:02, B*57:03, ...

---

**Document Version:** 1.0
**Last Updated:** December 25, 2025
