# External Validation Guide

## Overview

This document outlines requirements and procedures for validating the p-adic VAE framework on external databases beyond Stanford HIVDB.

---

## Available External Databases

### 1. Los Alamos HIV Sequence Database

**URL**: https://www.hiv.lanl.gov/

**Data Available**:
- >1,000,000 HIV sequences
- Subtype annotations
- Geographic and temporal metadata
- Drug resistance annotations (subset)

**Access**:
- Free registration required
- Bulk download available
- FASTA and alignment formats

**Validation Use**:
- Test generalization to non-Stanford sequences
- Validate across HIV subtypes (B, C, A, etc.)
- Larger sample size for uncertainty estimation

**Preparation Steps**:
1. Download sequences with resistance annotations
2. Align to HXB2 reference
3. Extract position columns matching Stanford format
4. Convert resistance scores to compatible scale

### 2. EuResist Network

**URL**: https://www.euresist.org/

**Data Available**:
- 100,000+ treatment episodes
- Sequence-outcome pairs
- Longitudinal data

**Access**:
- Collaborative access required
- Data use agreement needed
- Contact: info@euresist.org

**Validation Use**:
- Test clinical outcome prediction
- Validate treatment response modeling
- Temporal validation (different time periods)

**Preparation Steps**:
1. Apply for data access
2. Map drug names to Stanford nomenclature
3. Extract baseline genotypes
4. Align outcome definitions

### 3. UK HIV Drug Resistance Database

**URL**: https://www.hivrdb.org.uk/

**Data Available**:
- UK clinical samples
- Longitudinal resistance data
- Treatment history

**Access**:
- Academic collaboration required
- Ethics approval may be needed

**Validation Use**:
- Geographic generalization (UK vs US data)
- Clinical population validation

### 4. ANRS (French) Databases

**URL**: https://www.anrs.fr/

**Data Available**:
- French clinical cohort data
- Resistance genotyping
- Outcome data

**Access**:
- Collaborative agreement required
- French institution partnership helpful

---

## Validation Protocol

### Phase 1: Data Preparation

```python
# Standard preprocessing pipeline
def prepare_external_data(sequences, resistance_scores, database_name):
    """Prepare external data for validation."""

    # 1. Align to HXB2 reference
    aligned = align_to_hxb2(sequences)

    # 2. Extract position columns
    # Stanford uses P1, P2, ... P99 for PI
    # Convert external format to match
    pos_cols = extract_position_columns(aligned, format="stanford")

    # 3. Normalize resistance scores
    # Stanford uses fold-change; others may use categorical
    normalized_scores = normalize_resistance(
        resistance_scores,
        source_format=database_name,
        target_format="stanford"
    )

    # 4. One-hot encode
    X = encode_sequences(aligned, pos_cols)
    y = normalized_scores

    return X, y
```

### Phase 2: Model Evaluation

```python
def evaluate_on_external(model, X_ext, y_ext, drug):
    """Evaluate trained model on external data."""

    model.eval()
    with torch.no_grad():
        predictions = model.predict(X_ext)

    # Compute metrics
    metrics = {
        "correlation": pearsonr(predictions, y_ext),
        "spearman": spearmanr(predictions, y_ext),
        "rmse": np.sqrt(np.mean((predictions - y_ext)**2)),
        "mae": np.mean(np.abs(predictions - y_ext)),
    }

    return metrics
```

### Phase 3: Subgroup Analysis

Validate performance across:
- HIV subtypes (B, C, A, CRF01_AE, etc.)
- Geographic regions
- Time periods
- Treatment-naive vs experienced

---

## Expected Results

### Acceptable Performance Drop

External validation typically shows 5-15% performance drop:

| Scenario | Expected Correlation |
|----------|---------------------|
| Same database (internal) | +0.89 |
| Different database, same population | +0.80-0.85 |
| Different database, different subtype | +0.70-0.80 |
| Different database, different era | +0.75-0.85 |

### Warning Signs

- >20% correlation drop: Potential overfitting to Stanford data
- Systematic bias: Check normalization
- Subtype-specific failures: May need subtype-specific models

---

## Recommended Validation Experiments

### Experiment 1: Cross-Database Transfer

```python
# Train on Stanford, test on Los Alamos
model_stanford = train_on_stanford(drug)
results_lanl = evaluate_on_lanl(model_stanford, drug)
```

### Experiment 2: Subtype Generalization

```python
# Train on subtype B, test on subtype C
model_b = train_on_subtype(drug, subtype="B")
results_c = evaluate_on_subtype(model_b, drug, subtype="C")
```

### Experiment 3: Temporal Transfer

```python
# Train on pre-2015, test on 2015+
model_historical = train_temporal(drug, train_end=2015)
results_recent = evaluate_temporal(model_historical, drug, test_start=2015)
```

---

## Data Use Agreements

### Required for Each Database

1. **Institutional Review Board (IRB)**
   - May be required for clinical outcome data
   - Check with database provider

2. **Data Use Agreement (DUA)**
   - Specify intended use
   - Publication requirements
   - Data destruction timeline

3. **Collaboration Agreement**
   - Co-authorship requirements
   - Data access limitations
   - Derivative work policies

---

## Contact Information

### Stanford HIVDB
- Email: hivdb@stanford.edu
- Web: https://hivdb.stanford.edu/

### Los Alamos
- Email: seq-info@lanl.gov
- Web: https://www.hiv.lanl.gov/

### EuResist
- Email: info@euresist.org
- Web: https://www.euresist.org/

---

## Checklist

Before external validation:

- [ ] Obtain data access approvals
- [ ] Verify sequence alignment compatibility
- [ ] Confirm resistance score normalization
- [ ] Prepare subtype annotations
- [ ] Document preprocessing differences
- [ ] Plan statistical analysis
- [ ] Prepare for negative results

After external validation:

- [ ] Report all metrics (not just best)
- [ ] Analyze failure cases
- [ ] Compare to published benchmarks
- [ ] Document preprocessing differences
- [ ] Archive validation data

---

## References

1. Rhee et al., 2003. Nucleic Acids Research. Stanford HIVDB.
2. Lengauer et al., 2007. AIDS Reviews. EuResist overview.
3. Kuiken et al., 2003. Nucleic Acids Research. Los Alamos Database.
4. Wensing et al., 2019. Topics in Antiviral Medicine. IAS-USA drug resistance mutations.

---

*Document version: 1.0*
*Last updated: December 2025*
