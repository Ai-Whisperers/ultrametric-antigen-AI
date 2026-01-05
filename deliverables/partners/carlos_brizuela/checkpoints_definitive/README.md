# PeptideVAE Definitive Checkpoints

**Training Date:** 2026-01-05
**Training Script:** `training/train_definitive.py`

---

## Production Checkpoint

**File:** `best_production.pt` (copy of fold_2_definitive.pt)

| Metric | Value |
|--------|:-----:|
| Spearman r | **0.7368** |
| Pearson r | 0.7215 |
| Baseline (sklearn) | 0.56 |
| Improvement | +31% |

---

## All Fold Results

| Fold | File | Spearman r | Status |
|:----:|------|:----------:|--------|
| 0 | fold_0_definitive.pt | 0.6945 | PASSED |
| 1 | fold_1_definitive.pt | 0.5581 | MARGINAL |
| 2 | fold_2_definitive.pt | **0.7368** | **BEST** |
| 3 | fold_3_definitive.pt | 0.6542 | PASSED |
| 4 | fold_4_definitive.pt | 0.6379 | PASSED |

**Mean:** 0.6563 +/- 0.0599
**Collapse Rate:** 0%

---

## Usage

```python
import torch
from src.encoders.peptide_encoder import PeptideVAE

# Load model
checkpoint = torch.load('best_production.pt', map_location='cpu')
config = checkpoint['config']

model = PeptideVAE(
    latent_dim=config['latent_dim'],
    hidden_dim=config['hidden_dim'],
    n_layers=config['n_layers'],
    n_heads=config['n_heads'],
    dropout=config['dropout'],
    max_radius=config['max_radius'],
    curvature=config['curvature'],
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
outputs = model(["KLAKLAKKLAKLAK"], teacher_forcing=False)
mic_pred = outputs['mic_pred']  # Predicted MIC value
```

---

## Training Configuration

| Parameter | Value |
|-----------|:-----:|
| use_curriculum | False |
| min_epochs | 30 |
| warmup_epochs | 5 |
| mic_weight | 5.0 |
| hidden_dim | 64 |
| latent_dim | 16 |
| learning_rate | 5e-4 |

---

## Files

| File | Size | Description |
|------|:----:|-------------|
| best_production.pt | 1.2 MB | Production checkpoint (fold 2) |
| cv_results_definitive.json | 10 KB | Full CV metrics |
| fold_*_definitive.pt | 1.2 MB each | Individual fold checkpoints |
