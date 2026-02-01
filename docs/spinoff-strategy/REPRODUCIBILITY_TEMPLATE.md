# Reproducibility Guide Template

Use this template for `REPRODUCIBILITY.md` in spin-off repositories.

---

```markdown
# Reproducibility Guide

This document provides everything needed to **independently reproduce** all claimed results.

---

## Environment Setup

### Option 1: Exact Environment (Recommended)

\`\`\`bash
# Clone repository
git clone https://github.com/Ai-Whisperers/[repo-name].git
cd [repo-name]

# Create environment with exact versions
conda env create -f environment.yml
conda activate [env-name]

# Or with pip
pip install -r requirements-frozen.txt
\`\`\`

### Option 2: Minimal Requirements

\`\`\`bash
pip install -e .
\`\`\`

**Note:** Results may vary slightly with different package versions.

### Verified Environments

| OS | Python | PyTorch | Result Match |
|----|--------|---------|--------------|
| Ubuntu 22.04 | 3.11.5 | 2.1.0 | Exact |
| macOS 14.0 | 3.11.5 | 2.1.0 | Within 0.01 |
| Windows 11 | 3.11.5 | 2.1.0 | Within 0.01 |

---

## Data

### Included Data

All validation data is included in the repository:

\`\`\`
data/
├── [dataset1].json     # Description, N=X
├── [dataset2].csv      # Description, N=Y
└── README.md           # Data documentation
\`\`\`

### Data Checksums

Verify data integrity:

\`\`\`bash
python scripts/verify_data.py
# Expected output:
# [dataset1].json: SHA256 = abc123...
# [dataset2].csv: SHA256 = def456...
# All checksums match!
\`\`\`

### External Data (if any)

If external data is required:

| Dataset | Source | Download |
|---------|--------|----------|
| [Name] | [URL] | `scripts/download_[name].py` |

---

## Reproducing Main Results

### Result 1: [Metric Name] = [Value]

**Claimed:** [Metric] = [Value] ± [Std]

**Reproduce:**

\`\`\`bash
python scripts/reproduce_[metric].py
\`\`\`

**Expected output:**

\`\`\`
Running [metric] validation...
Dataset: [name] (N=[size])
Method: [description]

Results:
  [Metric]: [Value]
  95% CI: [[Lower], [Upper]]

Matches claimed value: YES
\`\`\`

**Interpretation:**
- If within CI: Successful reproduction
- If outside CI but within 10%: Environment difference, acceptable
- If >10% off: [Open an issue](https://github.com/Ai-Whisperers/[repo]/issues)

### Result 2: [Metric Name] = [Value]

[Same structure as above]

---

## Step-by-Step Validation

For users who want to understand each step:

### Step 1: Load Data

\`\`\`python
from [package] import load_validation_data

data = load_validation_data('[dataset]')
print(f"Loaded {len(data)} samples")
# Expected: Loaded [N] samples
\`\`\`

### Step 2: Load Model

\`\`\`python
from [package] import [Model]

model = [Model].from_pretrained()
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
# Expected: Model parameters: [N]
\`\`\`

### Step 3: Run Predictions

\`\`\`python
predictions = []
for sample in data:
    pred = model.predict(sample['input'])
    predictions.append(pred)

print(f"Generated {len(predictions)} predictions")
\`\`\`

### Step 4: Compute Metrics

\`\`\`python
from scipy.stats import spearmanr
import numpy as np

true_values = [s['target'] for s in data]
rho, pval = spearmanr(true_values, predictions)

print(f"Spearman ρ: {rho:.4f}")
print(f"P-value: {pval:.2e}")
# Expected: Spearman ρ: [Value]
\`\`\`

---

## Cross-Validation Reproduction

For LOO (Leave-One-Out) or k-fold results:

\`\`\`bash
# Full LOO (may take [time estimate])
python scripts/reproduce_loo.py

# Quick check (10% subset)
python scripts/reproduce_loo.py --subset 0.1
\`\`\`

**Expected runtime:**
- Full LOO: ~[X] minutes on [hardware]
- Quick check: ~[Y] seconds

---

## Hardware Requirements

| Configuration | Minimum | Recommended |
|---------------|---------|-------------|
| RAM | [X] GB | [Y] GB |
| GPU VRAM | [X] GB (optional) | [Y] GB |
| Disk | [X] GB | [Y] GB |

**Note:** GPU is optional. CPU-only runs are [X]x slower but produce identical results.

---

## Troubleshooting

### "Results don't match"

1. **Check Python version:** Must be 3.10+
2. **Check PyTorch version:** Recommend 2.0+
3. **Verify data checksums:** `python scripts/verify_data.py`
4. **Try frozen requirements:** `pip install -r requirements-frozen.txt`

### "Out of memory"

Reduce batch size:
\`\`\`bash
python scripts/validate.py --batch-size 16
\`\`\`

### "CUDA not available"

CPU-only is supported:
\`\`\`bash
python scripts/validate.py --device cpu
\`\`\`

---

## Validation Artifacts

After running validation, check `results/`:

\`\`\`
results/
├── validation_metrics.json    # All computed metrics
├── predictions.csv            # Raw predictions
├── comparison_plot.png        # Visual comparison
└── validation.log             # Detailed log
\`\`\`

---

## Reporting Issues

If reproduction fails:

1. **Run diagnostics:**
   \`\`\`bash
   python scripts/diagnostics.py > diagnostics.txt
   \`\`\`

2. **Open issue** with:
   - `diagnostics.txt` output
   - Your expected vs actual results
   - Full error message (if any)

We commit to responding within 48 hours.

---

## Certification

This reproduction guide was verified on:
- Date: [YYYY-MM-DD]
- Commit: [hash]
- By: [name/automated]

All results match within stated tolerances.

---

*Last updated: [Date]*
```

---

## Template Usage Notes

### Required Scripts

Every spin-off must include:

| Script | Purpose |
|--------|---------|
| `scripts/validate.py` | One-command full validation |
| `scripts/verify_data.py` | Data integrity check |
| `scripts/diagnostics.py` | Environment diagnostics |
| `scripts/reproduce_[metric].py` | Per-metric reproduction |

### Quality Checklist

Before release, verify:

- [ ] All scripts run without error on clean environment
- [ ] Results match within stated tolerances
- [ ] Data checksums documented
- [ ] Runtime estimates accurate
- [ ] Troubleshooting covers common issues

---

*Last updated: 2026-01-30*
