# PeptideVAE Training Improvement Plan

**Doc-Type:** Implementation Plan · Version 1.0 · Updated 2026-01-05 · AI Whisperers

---

## Current State

| Metric | Value | Target |
|--------|:-----:|:------:|
| Mean Spearman r | 0.525 | ≥0.60 |
| Std deviation | 0.196 | ≤0.10 |
| Collapse rate | 20% (1/5) | 0% |
| Best fold | 0.686 | - |
| Worst fold | 0.146 | ≥0.45 |

---

## Improvement Strategy

### Priority 1: Training Stabilization

**Goal:** Eliminate fold collapse, reduce variance

#### 1.1 Learning Rate Warmup

```python
# Add to train_improved.py

class WarmupCosineScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.base_lr * (1 + np.cos(np.pi * progress)) / 2

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# Usage
scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_epochs=5,      # 5 epoch warmup
    total_epochs=100,
    base_lr=5e-4
)
```

**Rationale:** Warmup prevents early collapse by starting with very small updates.

#### 1.2 Stochastic Weight Averaging (SWA)

```python
from torch.optim.swa_utils import AveragedModel, SWALR

# After regular training (epoch 70+)
swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=1e-4)

for epoch in range(70, 100):
    train_epoch(model, ...)
    swa_model.update_parameters(model)
    swa_scheduler.step()

# Use SWA model for inference
torch.optim.swa_utils.update_bn(train_loader, swa_model)
```

**Rationale:** SWA averages weights from multiple training steps, smoothing out local minima.

#### 1.3 Gradient Accumulation

```python
# In train_epoch function
accumulation_steps = 2  # Effective batch size = 64

optimizer.zero_grad()
for i, batch in enumerate(train_loader):
    outputs = model(batch['sequences'])
    loss, metrics = loss_manager.compute_total_loss(...)

    # Scale loss for accumulation
    (loss / accumulation_steps).backward()

    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
```

**Rationale:** Larger effective batch size = more stable gradients.

#### 1.4 Seed Averaging (Ensemble Weights)

```python
def train_with_seed_averaging(config, fold_idx, n_seeds=3):
    """Train multiple seeds and average weights."""
    models = []

    for seed in range(n_seeds):
        set_seed(config.seed + seed)
        model, metrics = train_fold(config, fold_idx)
        models.append(model)

    # Average weights
    avg_state = {}
    for key in models[0].state_dict():
        avg_state[key] = torch.stack([m.state_dict()[key].float()
                                       for m in models]).mean(0)

    final_model = PeptideVAE(...)
    final_model.load_state_dict(avg_state)
    return final_model
```

**Rationale:** Averaging across seeds reduces sensitivity to initialization.

---

### Priority 2: Regime-Aware Training

**Goal:** Improve long/hydrophobic peptide prediction

#### 2.1 Regime-Stratified Batching

```python
from torch.utils.data import WeightedRandomSampler

def get_regime_weights(dataset):
    """Weight samples inversely to regime frequency."""
    regimes = []
    for sample in dataset:
        length = len(sample['sequence'])
        hydro = sample['properties'][2]  # hydrophobicity index

        if length > 25 or hydro > 0.5:
            regime = 'hard'  # Long or hydrophobic
        else:
            regime = 'easy'
        regimes.append(regime)

    # Inverse frequency weighting
    counts = {'easy': regimes.count('easy'), 'hard': regimes.count('hard')}
    weights = [1.0 / counts[r] for r in regimes]
    return weights

# Usage
weights = get_regime_weights(train_dataset)
sampler = WeightedRandomSampler(weights, len(weights))
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
```

**Rationale:** Ensures hard regimes appear more often in training.

#### 2.2 Regime-Weighted Loss

```python
def compute_regime_weights(sequences, mic_targets):
    """Higher weight for hard-to-predict regimes."""
    weights = torch.ones(len(sequences))

    for i, seq in enumerate(sequences):
        length = len(seq)
        # Hard regime detection
        if length > 25:
            weights[i] *= 2.0  # Long peptides
        if length < 10:
            weights[i] *= 1.5  # Very short (rare)

    return weights

# In loss computation
regime_weights = compute_regime_weights(sequences, mic_targets)
mic_loss = ((mic_pred - mic_targets) ** 2 * regime_weights).mean()
```

#### 2.3 Multi-Head Architecture

```python
class RegimeAwarePeptideVAE(PeptideVAE):
    """Separate prediction heads for different regimes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Regime-specific MIC heads
        self.mic_head_short = nn.Linear(self.latent_dim, 1)
        self.mic_head_medium = nn.Linear(self.latent_dim, 1)
        self.mic_head_long = nn.Linear(self.latent_dim, 1)

        # Regime classifier
        self.regime_classifier = nn.Linear(self.latent_dim, 3)

    def forward(self, sequences, ...):
        base_outputs = super().forward(sequences, ...)
        z = base_outputs['z']

        # Predict regime
        regime_logits = self.regime_classifier(z)
        regime_probs = F.softmax(regime_logits, dim=-1)

        # Predict MIC with each head
        mic_short = self.mic_head_short(z)
        mic_medium = self.mic_head_medium(z)
        mic_long = self.mic_head_long(z)

        # Weighted combination
        mic_pred = (regime_probs[:, 0:1] * mic_short +
                    regime_probs[:, 1:2] * mic_medium +
                    regime_probs[:, 2:3] * mic_long)

        base_outputs['mic_pred'] = mic_pred
        base_outputs['regime_logits'] = regime_logits
        return base_outputs
```

---

### Priority 3: Data Augmentation

**Goal:** Expand effective training set

#### 3.1 Conservative Substitutions

```python
# Amino acid substitution groups (BLOSUM-based)
CONSERVATIVE_SUBS = {
    'K': ['R'],           # Positive charge
    'R': ['K'],
    'D': ['E'],           # Negative charge
    'E': ['D'],
    'L': ['I', 'V', 'M'], # Hydrophobic
    'I': ['L', 'V'],
    'V': ['L', 'I'],
    'F': ['Y', 'W'],      # Aromatic
    'Y': ['F', 'W'],
    'W': ['F', 'Y'],
}

def augment_sequence(seq, n_subs=1):
    """Create variant with conservative substitutions."""
    seq_list = list(seq)
    positions = [i for i, aa in enumerate(seq) if aa in CONSERVATIVE_SUBS]

    if not positions:
        return seq

    for _ in range(n_subs):
        pos = random.choice(positions)
        orig_aa = seq_list[pos]
        new_aa = random.choice(CONSERVATIVE_SUBS[orig_aa])
        seq_list[pos] = new_aa

    return ''.join(seq_list)
```

#### 3.2 Embedding Noise Injection

```python
def add_embedding_noise(z, noise_scale=0.1):
    """Add Gaussian noise to latent embeddings during training."""
    if self.training:
        noise = torch.randn_like(z) * noise_scale
        z = z + noise
    return z
```

---

### Priority 4: Pre-training on Larger Databases

**Goal:** Transfer learning from APD3/DRAMP

#### 4.1 Pre-training Pipeline

```python
# Step 1: Pre-train encoder on large unlabeled AMP database
# (reconstruction + sequence properties only, no MIC)

class PretrainingConfig:
    database: str = "APD3"  # ~3,000 sequences
    epochs: int = 50
    loss_weights = {
        'reconstruction': 1.0,
        'property': 0.5,
        'radial': 0.3,
    }
    # No MIC loss during pre-training

# Step 2: Fine-tune on labeled MIC data
class FinetuningConfig:
    pretrained_checkpoint: str = "pretrained_apd3.pt"
    freeze_encoder_epochs: int = 10  # Freeze encoder initially
    mic_weight: float = 5.0
    epochs: int = 50
```

---

## Implementation Timeline

### Session 1: Stabilization (2-3 hours)

1. Add learning rate warmup to `train_improved.py`
2. Add SWA for final 30 epochs
3. Run 5-fold CV with new settings
4. **Target:** Mean r ≥ 0.58, std ≤ 0.12

### Session 2: Regime-Aware Training (2-3 hours)

1. Implement regime-stratified batching
2. Add regime-weighted loss
3. Run 5-fold CV
4. **Target:** Long regime r ≥ 0.30, hydrophobic r ≥ 0.20

### Session 3: Data Augmentation (1-2 hours)

1. Implement conservative substitution augmentation
2. Add embedding noise
3. Run 5-fold CV with augmentation
4. **Target:** Mean r ≥ 0.62

### Session 4: Pre-training (4-6 hours)

1. Download APD3 database
2. Pre-train encoder
3. Fine-tune on MIC data
4. **Target:** Mean r ≥ 0.65, std ≤ 0.08

---

## Monitoring Checklist

During each training run, monitor:

- [ ] No fold drops below r=0.40 (collapse detection)
- [ ] Validation loss decreasing for first 30 epochs
- [ ] Learning rate follows expected schedule
- [ ] All regimes show positive correlation
- [ ] No NaN losses or gradients

---

## Rollback Criteria

If improvements fail, rollback to:

1. **Current best:** `checkpoints_improved/fold_2_improved.pt` (r=0.686)
2. **Stable fallback:** sklearn Ridge models (r=0.56)
3. **Ensemble:** sklearn + fold_2 PeptideVAE

---

## Success Metrics

| Phase | Mean r | Std r | Min fold | Long regime | Status |
|-------|:------:|:-----:|:--------:|:-----------:|--------|
| Current | 0.525 | 0.196 | 0.146 | 0.10 | BASELINE |
| Phase 1 | ≥0.58 | ≤0.12 | ≥0.40 | - | PENDING |
| Phase 2 | ≥0.60 | ≤0.10 | ≥0.45 | ≥0.30 | PENDING |
| Phase 3 | ≥0.62 | ≤0.08 | ≥0.50 | ≥0.35 | PENDING |
| Phase 4 | ≥0.65 | ≤0.05 | ≥0.55 | ≥0.40 | PENDING |

---

**Ready to implement when you have next training session.**
