# Quick Wins Implementation Guide

**Priority**: Immediate (1-2 days)
**Goal**: Improve weak drugs and fix bugs

---

## Quick Win #1: Fix Enhanced Training Scripts

### Problem
The enhanced scripts use LayerNorm + GELU which causes NaN losses.

### Solution
Replace with BatchNorm + ReLU (matching the working baseline).

### Code Changes

```python
# In run_enhanced_training.py, run_maml_evaluation.py, run_multitask_training.py

# BEFORE (unstable):
layers.append(nn.LayerNorm(h))
layers.append(nn.GELU())

# AFTER (stable):
layers.append(nn.BatchNorm1d(h))
layers.append(nn.ReLU())
```

### Files to Modify
1. `scripts/experiments/run_enhanced_training.py` - Lines 109-115
2. `scripts/experiments/run_maml_evaluation.py` - Similar locations
3. `scripts/experiments/run_multitask_training.py` - Similar locations

---

## Quick Win #2: Integrate TAM into Working Baseline

### Approach
Modify `run_on_real_data.py` to optionally use TAM features.

### Implementation

```python
# Add to run_on_real_data.py

from src.encoding.tam_aware_encoder import TAMAwareEncoder, NRTI_KEY_POSITIONS

def encode_with_tam(df: pd.DataFrame, position_cols: List[str]) -> Tuple[np.ndarray, int]:
    """Encode with TAM features for NRTI/NNRTI."""
    tam_encoder = TAMAwareEncoder(position_cols)
    X = tam_encoder.encode_dataframe(df)
    return X, tam_encoder.onehot_dim  # Return onehot_dim for reconstruction

def train_single_drug_with_tam(drug_class: str, drug: str, cfg: Config):
    """Train with TAM features."""
    df, position_cols, drugs = load_stanford_data(drug_class)

    # Filter valid samples
    df_valid = df[df[drug].notna() & (df[drug] > 0)].copy()

    # Encode with TAM
    if drug_class in ["nrti", "nnrti"]:
        X, onehot_dim = encode_with_tam(df_valid, position_cols)
        use_tam = True
    else:
        X = encode_amino_acids(df_valid, position_cols)
        onehot_dim = X.shape[1]
        use_tam = False

    # Get resistance values
    y = np.log10(df_valid[drug].values + 1).astype(np.float32)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create model with full input dim but reconstruct only onehot
    cfg.input_dim = X.shape[1]
    cfg.onehot_dim = onehot_dim  # For reconstruction loss

    # ... rest of training
```

### Key Change: Reconstruction Loss

```python
def compute_loss_with_tam(cfg, out, x, fitness, onehot_dim):
    """Compute loss, reconstructing only one-hot portion."""
    losses = {}

    # Reconstruction only for one-hot features
    x_onehot = x[:, :onehot_dim]
    recon_onehot = out["x_recon"][:, :onehot_dim]
    losses["recon"] = F.mse_loss(recon_onehot, x_onehot)

    # Rest of losses unchanged...
```

---

## Quick Win #3: Position-Weighted Loss

### Concept
Weight key resistance positions higher in reconstruction loss.

### Implementation

```python
# Key positions by drug class (from Stanford HIVDB)
KEY_POSITIONS = {
    "pi": [30, 32, 33, 46, 47, 48, 50, 54, 76, 82, 84, 88, 90],
    "nrti": [41, 62, 65, 67, 69, 70, 74, 75, 115, 151, 184, 210, 215, 219],
    "nnrti": [100, 101, 103, 106, 181, 188, 190, 225, 230],
    "ini": [66, 92, 118, 140, 143, 147, 148, 155, 263],
}

def create_position_weights(n_positions: int, n_aa: int, key_positions: List[int],
                           key_weight: float = 2.0) -> torch.Tensor:
    """Create weight tensor for position-weighted reconstruction."""
    weights = torch.ones(n_positions * n_aa)

    for pos in key_positions:
        if pos < n_positions:
            start = pos * n_aa
            end = start + n_aa
            weights[start:end] = key_weight

    return weights

def compute_weighted_recon_loss(x_recon, x, weights):
    """Compute position-weighted reconstruction loss."""
    diff = (x_recon - x) ** 2
    weighted_diff = diff * weights.to(x.device)
    return weighted_diff.mean()
```

---

## Quick Win #4: RPV-Specific Model

### Problem
RPV has complex multi-mutation resistance (r=0.588).

### Known RPV Resistance Patterns
From Stanford HIVDB:
- E138K alone: Low-level resistance
- E138K + M184V/I: Higher resistance
- Y181C/I/V: Cross-resistance from EFV/NVP
- K101E/P: Accumulation pattern

### Implementation

```python
# Add interaction features for RPV
def compute_rpv_interactions(row, position_cols):
    """Compute explicit mutation interactions for RPV."""
    features = []

    # Get mutations at key positions
    def has_mutation(pos, muts):
        col = f"P{pos}"
        if col in position_cols:
            idx = position_cols.index(col)
            aa = str(row[col]).upper() if pd.notna(row[col]) else "-"
            return any(m in aa for m in muts)
        return False

    # E138K
    e138k = has_mutation(138, ["K"])
    features.append(float(e138k))

    # M184V/I
    m184vi = has_mutation(184, ["V", "I"])
    features.append(float(m184vi))

    # Y181C/I/V
    y181 = has_mutation(181, ["C", "I", "V"])
    features.append(float(y181))

    # K101E/P
    k101 = has_mutation(101, ["E", "P"])
    features.append(float(k101))

    # Interactions
    features.append(float(e138k and m184vi))  # E138K + M184V/I
    features.append(float(e138k and y181))     # E138K + Y181
    features.append(float(y181 and k101))      # Y181 + K101

    return np.array(features, dtype=np.float32)
```

---

## Quick Win #5: Ensemble for Robustness

### Implementation

```python
def train_ensemble(drug_class: str, drug: str, n_models: int = 5) -> List[nn.Module]:
    """Train ensemble of models with different seeds."""
    models = []

    for seed in range(n_models):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Train model
        model, best_corr = train_single_drug(drug_class, drug)
        models.append(model)
        print(f"  Seed {seed}: correlation = {best_corr:+.4f}")

    return models

def predict_ensemble(models: List[nn.Module], x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict with ensemble, return mean and std."""
    predictions = []

    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(x)
            pred = out["z"][:, 0]  # First latent dimension
            predictions.append(pred)

    predictions = torch.stack(predictions)
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)

    return mean_pred, std_pred
```

---

## Expected Improvements

| Drug | Current | After Fixes | Improvement |
|------|---------|-------------|-------------|
| TDF | +0.773 | +0.82 | +0.05 (TAM) |
| RPV | +0.588 | +0.70 | +0.11 (interactions) |
| DTG | +0.756 | +0.80 | +0.04 (ensemble) |
| BIC | +0.791 | +0.82 | +0.03 (ensemble) |
| ETR | +0.799 | +0.83 | +0.03 (weights) |

---

## Testing Checklist

### After Each Fix
1. [ ] Run on one drug to verify no errors
2. [ ] Check loss is not NaN
3. [ ] Compare correlation to baseline
4. [ ] If better, run full drug class

### Validation
```bash
# Test TAM integration
python run_on_real_data_with_tam.py --drug-class nrti --drug TDF

# Test position weights
python run_on_real_data_weighted.py --drug-class nnrti --drug RPV

# Test ensemble
python run_ensemble.py --drug-class ini --drug DTG --n-models 5
```

---

## Priority Order

1. **Fix NaN bug** in enhanced scripts (30 min)
2. **TAM integration** for NRTI (2 hours)
3. **Position weights** for all classes (1 hour)
4. **RPV interactions** (2 hours)
5. **Ensemble** for low-data drugs (1 hour)

Total estimated time: **6-8 hours**
Expected overall improvement: **+0.02 to +0.05** average correlation
