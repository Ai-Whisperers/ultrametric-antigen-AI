# Hemolysis Predictor Guide

**Version**: 1.0.0
**Updated**: 2025-12-30
**Authors**: AI Whisperers

---

## Overview

Hemolysis is the rupture of red blood cells (erythrocytes), releasing hemoglobin into the surrounding fluid. For therapeutic peptides, hemolytic activity is a major safety concern that must be evaluated before clinical development.

The `HemolysisPredictor` module predicts the hemolytic potential of antimicrobial peptides based on sequence-derived features, using machine learning models trained on experimentally validated data.

---

## Why Hemolysis Matters

Antimicrobial peptides (AMPs) kill bacteria by disrupting cell membranes. Unfortunately, this mechanism can also affect mammalian cell membranes, leading to:

1. **Red blood cell lysis** - Releases hemoglobin, causing anemia
2. **Tissue damage** - Can harm cells at injection sites
3. **Systemic toxicity** - High doses can affect multiple organs

**The Therapeutic Index** (HC50/MIC) is the key metric:
- **HC50**: Concentration causing 50% hemolysis
- **MIC**: Minimum inhibitory concentration against bacteria
- **Higher is better**: TI > 10 indicates excellent selectivity

---

## Key Metrics

### HC50 (Hemolytic Concentration 50%)

The concentration at which 50% of red blood cells are lysed.

| HC50 Range | Classification | Interpretation |
|------------|----------------|----------------|
| < 50 μM | High risk | Strongly hemolytic, unsuitable for therapy |
| 50-200 μM | Moderate risk | May need modifications |
| > 200 μM | Low risk | Favorable for development |

### Risk Categories

```
High Risk (HC50 < 50 μM):
  - Examples: Melittin (1.5 μM), Mastoparan (15 μM)
  - Action: Major redesign needed or consider non-systemic use

Moderate Risk (50-200 μM):
  - Examples: Magainin 2 (100 μM), LL-37 (150 μM)
  - Action: Optimization may improve safety margin

Low Risk (> 200 μM):
  - Examples: Pexiganan (250 μM), Cecropin A (400 μM)
  - Action: Proceed with development, monitor in vivo
```

---

## Installation & Basic Usage

```python
from shared import HemolysisPredictor

# Create predictor (trains automatically on initialization)
predictor = HemolysisPredictor()

# Predict for a single peptide
result = predictor.predict("GIGKFLHSAKKFGKAFVGEIMNS")

print(f"Sequence: {result['sequence']}")
print(f"Predicted HC50: {result['hc50_predicted']:.1f} μM")
print(f"Is hemolytic: {result['is_hemolytic']}")
print(f"Probability: {result['hemolytic_probability']:.2%}")
print(f"Risk category: {result['risk_category']}")
print(f"Note: {result['therapeutic_index_note']}")
```

**Output:**
```
Sequence: GIGKFLHSAKKFGKAFVGEIMNS
Predicted HC50: 105.4 μM
Is hemolytic: False
Probability: 23.45%
Risk category: Moderate
Note: Moderate hemolytic activity. May need optimization for therapeutic use.
```

---

## API Reference

### HemolysisPredictor Class

```python
class HemolysisPredictor:
    """Predict hemolytic activity of peptides."""

    def __init__(self):
        """Initialize and train the predictor."""

    def predict(self, sequence: str) -> dict:
        """Predict hemolytic activity.

        Args:
            sequence: Amino acid sequence (single letter code)

        Returns:
            Dictionary containing:
                - sequence: Input sequence
                - hc50_predicted: Predicted HC50 in μM
                - is_hemolytic: Boolean (HC50 < 50 μM)
                - hemolytic_probability: Probability of being hemolytic (0-1)
                - risk_category: 'High', 'Moderate', or 'Low'
                - therapeutic_index_note: Interpretation guidance
        """

    def predict_batch(self, sequences: list[str]) -> list[dict]:
        """Predict for multiple peptides.

        Args:
            sequences: List of amino acid sequences

        Returns:
            List of prediction dictionaries
        """

    def compute_therapeutic_index(
        self,
        sequence: str,
        mic_value: float,
    ) -> dict:
        """Compute therapeutic index (HC50/MIC).

        Args:
            sequence: Amino acid sequence
            mic_value: MIC in μM against target pathogen

        Returns:
            Dictionary with therapeutic index metrics
        """
```

---

## Feature Engineering

The predictor uses 17 carefully selected features relevant to membrane interactions:

### Basic Properties (5 features)
| Feature | Description | Hemolysis Effect |
|---------|-------------|------------------|
| Length | Amino acid count | Longer = more hemolytic |
| Net charge | Sum of charges at pH 7.4 | High positive = more hemolytic |
| Hydrophobicity | Mean Kyte-Doolittle | Higher = more hemolytic |
| Hydrophobic ratio | % hydrophobic AAs | Higher = more hemolytic |
| Cationic ratio | % K+R residues | Contributes to initial binding |

### Residue-Specific Features (7 features)
| Feature | Description | Hemolysis Effect |
|---------|-------------|------------------|
| Tryptophan (W) | % content | Strong membrane inserter |
| Phenylalanine (F) | % content | Aromatic, membrane anchoring |
| Leucine (L) | % content | Hydrophobic core |
| Isoleucine (I) | % content | Hydrophobic core |
| Proline (P) | % content | Helix breaker, reduces hemolysis |
| Glycine (G) | % content | Flexibility, variable effect |
| Cysteine (C) | % content | Disulfide bonds, structure |

### Structural Features (5 features)
| Feature | Description | Hemolysis Effect |
|---------|-------------|------------------|
| Aromatic content | % F+W+Y | Strong membrane interaction |
| Aliphatic content | % A+I+L+V | Hydrophobic membrane insertion |
| Charge density | Charge/length | Concentrated charge = higher |
| Hydrophobic moment proxy | Window variance | Amphipathicity = higher |
| Net hydrophobicity | Sum of all hydrophobicity | Total membrane affinity |

### Feature Calculation Example

```python
from shared.hemolysis_predictor import HemolysisPredictor

predictor = HemolysisPredictor()
features = predictor._compute_hemolysis_features("GIGAVLKVLTTGLPALISWIKRKRQQ")

print("Feature vector (17 dimensions):")
print(f"  Length: {features[0]:.0f}")
print(f"  Charge: {features[1]:.1f}")
print(f"  Hydrophobicity: {features[2]:.3f}")
print(f"  Hydrophobic ratio: {features[3]:.3f}")
print(f"  Cationic ratio: {features[4]:.3f}")
print(f"  Trp content: {features[5]:.3f}")
print(f"  Aromatic content: {features[11]:.3f}")
print(f"  Aliphatic content: {features[12]:.3f}")
```

---

## Training Data

The predictor is trained on 40 experimentally validated peptides from literature:

### Highly Hemolytic (HC50 < 50 μM) - 12 peptides

| Peptide | HC50 (μM) | Source | Mechanism |
|---------|-----------|--------|-----------|
| Melittin | 1.5 | Bee venom | Strong membrane lysis |
| Delta-lysin | 5.0 | S. aureus | Pore formation |
| Pardaxin | 8.0 | Fish | Channel formation |
| Gramicidin S | 12.0 | B. brevis | Cyclic, membrane permeation |
| Mastoparan | 15.0 | Wasp venom | Mast cell degranulation |
| Alamethicin | 18.0 | Fungal | Voltage-gated channels |
| Crabrolin | 20.0 | Hornet | Membrane disruption |
| Polybia-MP1 | 25.0 | Wasp | Selective cancer toxicity |
| Arenicin-1 | 30.0 | Lugworm | β-hairpin, broad spectrum |
| Protegrin-1 | 35.0 | Pig | Disulfide-stabilized |
| Polyphemusin I | 38.0 | Horseshoe crab | Cationic, membrane active |
| Tachyplesin I | 40.0 | Horseshoe crab | β-sheet structure |

### Moderately Hemolytic (50-200 μM) - 14 peptides

| Peptide | HC50 (μM) | Source | Notes |
|---------|-----------|--------|-------|
| Temporin A | 60.0 | Frog | Short, alpha-helical |
| Gomesin | 65.0 | Spider | Disulfide-bridged |
| Aurein 1.2 | 70.0 | Frog | Minimal AMP |
| Cathelicidin BF | 75.0 | Snake | Broad spectrum |
| Indolicidin | 80.0 | Bovine | Trp-rich |
| Lactoferricin B | 85.0 | Bovine | Derived from lactoferrin |
| BMAP-27 | 90.0 | Bovine | Cathelicidin family |
| WLBU2 | 95.0 | Designed | Engineered AMP |
| Magainin 2 | 100.0 | Frog | Classic AMP |
| Omiganan | 100.0 | Designed | Clinical trials |
| BMAP-28 | 110.0 | Bovine | Cathelicidin |
| Dermaseptin S1 | 120.0 | Frog | Alpha-helical |
| LL-37 | 150.0 | Human | Host defense peptide |
| Esculentin-1 | 180.0 | Frog | Long, multiple domains |

### Low/No Hemolysis (HC50 > 200 μM) - 14 peptides

| Peptide | HC50 (μM) | Source | Safety Feature |
|---------|-----------|--------|----------------|
| Pexiganan | 250.0 | Designed | Optimized magainin |
| Pleurocidin | 280.0 | Fish | Selective |
| Defensin HNP-1 | 300.0 | Human | Disulfide-rich |
| Cecropin B | 350.0 | Insect | Selective for bacteria |
| Nisin | 400.0 | Bacterial | Food preservative (GRAS) |
| Cecropin A | 400.0 | Insect | Model safe AMP |
| Buforin II | 500.0 | Frog | DNA-binding, non-lytic |
| Histatin 5 | 500.0 | Human | Antifungal |
| PR-39 | 600.0 | Pig | Proline-rich |
| Drosocin | 800.0 | Insect | Proline-rich, intracellular |
| Pyrrhocoricin | 900.0 | Insect | Proline-rich |
| Apidaecin IA | 1000.0 | Bee | Intracellular target |

---

## Therapeutic Index Calculation

The therapeutic index (TI) is the ratio of toxicity to efficacy:

```python
from shared import HemolysisPredictor

predictor = HemolysisPredictor()

# Magainin 2 against E. coli
ti_result = predictor.compute_therapeutic_index(
    sequence="GIGKFLHSAKKFGKAFVGEIMNS",
    mic_value=10.0  # μM
)

print(f"Peptide: Magainin 2")
print(f"HC50 (toxicity): {ti_result['hc50']:.1f} μM")
print(f"MIC (efficacy): {ti_result['mic']:.1f} μM")
print(f"Therapeutic Index: {ti_result['therapeutic_index']:.1f}")
print(f"Interpretation: {ti_result['interpretation']}")
```

**Output:**
```
Peptide: Magainin 2
HC50 (toxicity): 105.4 μM
MIC (efficacy): 10.0 μM
Therapeutic Index: 10.5
Interpretation: Excellent selectivity
```

### TI Interpretation Guide

| TI Range | Interpretation | Development Recommendation |
|----------|----------------|---------------------------|
| > 10 | Excellent selectivity | Proceed to preclinical |
| 5-10 | Good selectivity | Likely safe, monitor closely |
| 2-5 | Moderate selectivity | Consider modifications |
| < 2 | Poor selectivity | High toxicity risk, major redesign |

---

## Batch Predictions

For screening multiple peptides:

```python
from shared import HemolysisPredictor

predictor = HemolysisPredictor()

candidates = [
    "GIGKFLHSAKKFGKAFVGEIMNS",  # Magainin 2
    "GIGAVLKVLTTGLPALISWIKRKRQQ",  # Melittin
    "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",  # Cecropin A
    "ILPWKWPWWPWRR",  # Indolicidin
]

results = predictor.predict_batch(candidates)

print(f"{'Peptide':<12} {'HC50 (μM)':<12} {'Risk':<10} {'Prob':<8}")
print("-" * 45)
for i, result in enumerate(results):
    name = f"Peptide_{i+1}"
    print(f"{name:<12} {result['hc50_predicted']:<12.1f} {result['risk_category']:<10} {result['hemolytic_probability']:.2f}")
```

---

## Model Architecture

### Machine Learning Pipeline

```
Input Sequence
      ↓
Feature Extraction (17 features)
      ↓
StandardScaler (normalization)
      ↓
   ┌────────────────┬────────────────┐
   ↓                ↓                ↓
GradientBoosting   GradientBoosting
  Regressor         Classifier
   (HC50)          (Hemolytic/Not)
   ↓                ↓
Combine predictions
      ↓
Risk Assessment
      ↓
Output Dictionary
```

### Model Parameters

```python
# Regressor (HC50 prediction)
GradientBoostingRegressor(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    min_samples_leaf=2,
    random_state=42,
)

# Classifier (binary classification)
GradientBoostingClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    min_samples_leaf=2,
    random_state=42,
)
```

### Fallback Mode

When scikit-learn is not available, the predictor uses rule-based estimation:

```python
def _rule_based_hc50(self, sequence, features):
    """Heuristic HC50 estimation without ML."""
    base_hc50 = 200.0

    # Adjust based on key features
    if hydrophobicity > 1.0:
        base_hc50 /= (1 + hydro * 0.5)
    if aromatic_content > 0.2:
        base_hc50 /= (1 + aromatic * 3)
    if aliphatic_content > 0.4:
        base_hc50 /= (1 + (aliphatic - 0.3) * 2)
    if trp_content > 0.1:
        base_hc50 /= (1 + trp * 5)
    if length < 15:
        base_hc50 *= 1.5
    if length > 30:
        base_hc50 /= 1.3

    return clamp(base_hc50, 1.0, 1000.0)
```

---

## Design Guidelines

### Features That INCREASE Hemolysis

| Feature | Effect | Example Peptide |
|---------|--------|-----------------|
| High tryptophan | Membrane anchoring | Indolicidin (3W) |
| High hydrophobicity | Deeper insertion | Melittin |
| High amphipathicity | Strong helix formation | Magainin |
| Long aliphatic stretches | Membrane spanning | Alamethicin |
| Very high positive charge | Too much membrane binding | Highly charged variants |

### Features That DECREASE Hemolysis

| Feature | Effect | Example Peptide |
|---------|--------|-----------------|
| Proline residues | Breaks helical structure | PR-39, Drosocin |
| Short length (<15 aa) | Less membrane disruption | Short cationic peptides |
| Low hydrophobicity | Less insertion | Cecropins |
| Disulfide bonds | Restricts conformation | Defensins |
| Internal target | Non-lytic mechanism | Buforin II, Apidaecin |

### Optimization Strategies

```
To reduce hemolysis while maintaining activity:

1. ADD PROLINE
   - Replace L or A with P to break helical structure
   - Example: LALAL → LAPLAL

2. REDUCE HYDROPHOBICITY
   - Replace W with F or Y
   - Replace L with A
   - Add charged residues

3. SHORTEN SEQUENCE
   - Find minimal active region
   - Remove hydrophobic C-terminus

4. ADD D-AMINO ACIDS
   - Reduces hemolysis but maintains activity
   - Common at positions 2, 3

5. N-TERMINAL MODIFICATION
   - Acetylation can reduce hemolysis
   - Maintains antimicrobial activity
```

---

## Limitations

1. **Training data size**: Only 40 peptides, may not generalize to all peptide classes
2. **Sequence-only**: Does not consider 3D structure or post-translational modifications
3. **HC50 variability**: Experimental values vary between labs and assay conditions
4. **Species specificity**: Trained primarily on human RBC data
5. **No membrane composition**: Does not account for cholesterol or lipid composition

---

## Validation

### Cross-Validation Performance

Using leave-one-out cross-validation:

| Metric | Value | Notes |
|--------|-------|-------|
| Classification accuracy | ~80% | Hemolytic vs non-hemolytic |
| HC50 log10 RMSE | ~0.5 | About 3-fold error in prediction |
| Spearman correlation | ~0.65 | Reasonable rank ordering |

### Known Limitations

- May underestimate hemolysis for novel peptide scaffolds
- Proline-rich peptides well-characterized (good predictions)
- Cyclic peptides not specifically trained (less reliable)

---

## References

1. Gautam, A., et al. (2014). Hemolytik: a database of experimentally determined hemolytic and non-hemolytic peptides. Nucleic acids research, 42(D1), D444-D449.

2. Chaudhary, K., et al. (2016). A web server and mobile app for computing hemolytic potency of peptides. Scientific reports, 6(1), 1-13.

3. Plisson, F., et al. (2020). Machine learning-guided discovery and design of non-hemolytic peptides. Scientific reports, 10(1), 1-19.

4. Pirtskhalava, M., et al. (2021). DBAASP v3: database of antimicrobial/cytotoxic activity and structure of peptides. Nucleic acids research, 49(D1), D288-D297.

---

## Quick Reference

```python
from shared import HemolysisPredictor

# Initialize
predictor = HemolysisPredictor()

# Single prediction
result = predictor.predict("SEQUENCE")
# Keys: sequence, hc50_predicted, is_hemolytic, hemolytic_probability,
#       risk_category, therapeutic_index_note

# Batch prediction
results = predictor.predict_batch(["SEQ1", "SEQ2", "SEQ3"])

# Therapeutic index
ti = predictor.compute_therapeutic_index("SEQUENCE", mic_value=10.0)
# Keys: hc50, mic, therapeutic_index, interpretation, + all predict() keys
```
