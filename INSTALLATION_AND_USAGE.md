# Installation and Usage Guide

## Quick Start (5 minutes)

### 1. Install Package

```bash
cd "Ternary VAE PROD"

# Option A: Install in development mode
pip install -e .

# Option B: Install with all dependencies
pip install -e ".[all]"

# Option C: Minimal install
pip install -r requirements.txt
```

### 2. Copy Environment Template

```bash
cp .env.example .env
# Edit .env with your settings (optional)
```

### 3. Run Training

```bash
python scripts/train/train_ternary_v5_5.py --config configs/ternary_v5_5.yaml
```

### 4. Evaluate Model

```bash
python scripts/benchmark/run_benchmark.py \
    --config configs/ternary_v5_5.yaml \
    --checkpoint checkpoints/ternary_v5_5_best.pt
```

---

## Detailed Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works)
- 4GB+ GPU VRAM for training
- 8GB+ system RAM

### Step-by-Step

**1. Create Virtual Environment** (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

**2. Install Dependencies**

```bash
# Core dependencies only
pip install torch numpy scipy pyyaml tqdm

# Or install all optional dependencies
pip install -r requirements.txt
```

**3. Verify Installation**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
CUDA Available: True
```

**4. Run Tests** (optional but recommended)

```bash
pytest tests/ -v
```

---

## Usage Examples

### Example 1: Basic Training

```python
import yaml
from src.models.ternary_vae_v5_5 import DualNeuralVAEV5
from src.utils.data import generate_all_ternary_operations, TernaryOperationDataset
from torch.utils.data import DataLoader

# Load config
with open('configs/ternary_v5_5.yaml') as f:
    config = yaml.safe_load(f)

# Create dataset
operations = generate_all_ternary_operations()
dataset = TernaryOperationDataset(operations)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model
model = DualNeuralVAEV5(
    input_dim=9,
    latent_dim=16,
    rho_min=0.1,
    rho_max=0.7
)

# Train
for epoch in range(10):
    for batch in loader:
        outputs = model(batch, temp_A=1.0, temp_B=1.0, beta_A=1.0, beta_B=1.0)
        losses = model.loss_function(batch, outputs)
        # ... optimizer step
```

### Example 2: Generate Samples

```python
import torch
from src.models.ternary_vae_v5_5 import DualNeuralVAEV5

# Load trained model
model = DualNeuralVAEV5(input_dim=9, latent_dim=16)
checkpoint = torch.load('checkpoints/ternary_v5_5_best.pt')
model.load_state_dict(checkpoint['model'])
model.eval()

# Generate samples
samples_A = model.sample(1000, device='cuda', use_vae='A')
samples_B = model.sample(1000, device='cuda', use_vae='B')

print(f"VAE-A samples shape: {samples_A.shape}")  # (1000, 9)
print(f"VAE-B samples shape: {samples_B.shape}")  # (1000, 9)
```

### Example 3: Evaluate Coverage

```python
from src.utils.metrics import evaluate_coverage

# Generate many samples
samples = model.sample(195000, device='cuda', use_vae='A')

# Evaluate coverage
unique_ops, coverage_pct = evaluate_coverage(samples)
print(f"Coverage: {unique_ops} operations ({coverage_pct:.2f}%)")
```

### Example 4: Compute Metrics

```python
from src.utils.metrics import (
    compute_latent_entropy,
    compute_diversity_score,
    compute_reconstruction_accuracy
)

# Latent entropy
z = torch.randn(10000, 16)
entropy = compute_latent_entropy(z)
print(f"Latent entropy: {entropy:.3f}")

# Diversity
samples_A = model.sample(5000, device='cuda', use_vae='A')
samples_B = model.sample(5000, device='cuda', use_vae='B')
diversity = compute_diversity_score(samples_A, samples_B)
print(f"Diversity: {diversity:.3f}")

# Reconstruction accuracy
inputs = dataset.operations[:100]
outputs = model(inputs, temp_A=0.3, temp_B=0.2, beta_A=1.0, beta_B=1.0)
accuracy = compute_reconstruction_accuracy(inputs, outputs['logits_A'])
print(f"Accuracy: {accuracy:.2f}%")
```

---

## Configuration

### Config File Structure

The main configuration file is `configs/ternary_v5_5.yaml`:

```yaml
model:
  input_dim: 9
  latent_dim: 16
  rho_min: 0.1
  rho_max: 0.7
  # ... more parameters

optimizer:
  lr_start: 0.001
  lr_schedule:
    - epoch: 0
      lr: 0.001
    # ... more schedule points

vae_a:
  beta_start: 0.6
  beta_end: 1.0
  temp_start: 1.0
  temp_end: 0.3
  # ... more parameters

vae_b:
  # ... VAE-B parameters
```

### Environment Variables

Create a `.env` file from `.env.example`:

```bash
# Device
DEVICE=cuda                 # or cpu
CUDA_VISIBLE_DEVICES=0      # GPU ID

# Seeds
SEED=42
DETERMINISTIC=true

# Performance
NUM_WORKERS=4
MIXED_PRECISION=false

# Logging
LOG_LEVEL=INFO
TENSORBOARD=false
```

---

## Training

### Full Training (400 epochs)

```bash
python scripts/train/train_ternary_v5_5.py \
    --config configs/ternary_v5_5.yaml
```

Expected time: ~2.5 hours on modern GPU

### Quick Training (100 epochs)

```bash
# Modify config or create a new one
python scripts/train/train_ternary_v5_5.py \
    --config configs/ternary_v5_5_fast.yaml
```

### Resume from Checkpoint

The training script automatically saves checkpoints every 10 epochs. To resume:

```bash
# Edit the training script to load from checkpoint
# Or modify your config to include:
# resume_checkpoint: "checkpoints/epoch_100.pt"
```

---

## Evaluation and Benchmarking

### Run Full Benchmark

```bash
python scripts/benchmark/run_benchmark.py \
    --config configs/ternary_v5_5.yaml \
    --checkpoint checkpoints/ternary_v5_5_best.pt \
    --trials 10
```

Outputs:
- Inference speed (samples/sec)
- Coverage statistics (mean ± std)
- Latent entropy
- Memory usage
- Summary table

### Coverage Evaluation Only

```bash
python -c "
from src.models.ternary_vae_v5_5 import DualNeuralVAEV5
from src.utils.metrics import evaluate_coverage
import torch

model = DualNeuralVAEV5(input_dim=9, latent_dim=16)
checkpoint = torch.load('checkpoints/ternary_v5_5_best.pt')
model.load_state_dict(checkpoint['model'])
model.eval()

samples = model.sample(195000, 'cuda', 'A')
unique, cov = evaluate_coverage(samples)
print(f'Coverage: {unique} ({cov:.2f}%)')
"
```

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test

```bash
pytest tests/test_reproducibility.py -v
```

### Test Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

View coverage report: `open htmlcov/index.html`

---

## Reproducibility

### Deterministic Training

To ensure exact reproducibility:

1. **Set seeds in config**:
   ```yaml
   seed: 42
   ```

2. **Enable deterministic mode in `.env`**:
   ```bash
   DETERMINISTIC=true
   CUDNN_BENCHMARK=false
   ```

3. **Run training**:
   ```bash
   python scripts/train/train_ternary_v5_5.py --config configs/ternary_v5_5.yaml
   ```

4. **Verify with tests**:
   ```bash
   pytest tests/test_reproducibility.py -v
   ```

### Expected Results

With seed=42 and deterministic mode:
- **Epoch 100**: ~75-80% coverage
- **Epoch 200**: ~90-95% coverage
- **Epoch 400**: ~97-98% coverage (97.64% VAE-A, 97.67% VAE-B)

---

## Troubleshooting

### CUDA Out of Memory

**Solution 1**: Reduce batch size in config
```yaml
batch_size: 32  # from 64
```

**Solution 2**: Use gradient checkpointing
```python
# In .env
MEMORY_EFFICIENT=true
```

### Slow Training

**Solution 1**: Reduce num_workers
```yaml
num_workers: 0  # Use main process only
```

**Solution 2**: Enable mixed precision (experimental)
```bash
# In .env
MIXED_PRECISION=true
```

### Coverage Not Improving

**Solution**: Check phase transitions and learning rate schedule

```python
# Verify current phase
print(f"Current phase: {model.current_phase}")
print(f"Permeability ρ: {model.rho:.3f}")
print(f"Gradient balance: {model.grad_balance_achieved}")
```

### Import Errors

**Solution**: Ensure package is installed
```bash
pip install -e .
```

Or add to Python path:
```python
import sys
sys.path.append('/path/to/Ternary VAE PROD')
```

---

## Advanced Usage

### Custom Training Loop

```python
from src.models.ternary_vae_v5_5 import DualNeuralVAEV5
import torch.optim as optim

model = DualNeuralVAEV5(input_dim=9, latent_dim=16)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

for epoch in range(400):
    # Update adaptive parameters
    model.rho = model.compute_phase_scheduled_rho(epoch, phase_4_start=250)
    model.lambda3 = model.compute_cyclic_lambda3(epoch, period=30)

    # Get temperature schedules
    temp_A = get_temperature(epoch, 'A')
    temp_B = get_temperature(epoch, 'B')

    # Training loop
    for batch in loader:
        outputs = model(batch, temp_A, temp_B, beta_A, beta_B)
        losses = model.loss_function(batch, outputs)

        optimizer.zero_grad()
        losses['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        model.update_gradient_norms()
        optimizer.step()
```

### Transfer Learning

```python
# Load pre-trained weights
pretrained = torch.load('checkpoints/ternary_v5_5_best.pt')
model.load_state_dict(pretrained['model'])

# Freeze VAE-A (keep discoveries)
for param in model.encoder_A.parameters():
    param.requires_grad = False
for param in model.decoder_A.parameters():
    param.requires_grad = False

# Fine-tune VAE-B only
optimizer = optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.0001
)
```

---

## Next Steps

- **Documentation**: See `docs/` for detailed theory and implementation guides
- **Examples**: Check `examples/` for more usage patterns
- **API Reference**: See `docs/api/API_REFERENCE.md` for complete API documentation
- **Theory**: Read `docs/theory/MATHEMATICAL_FOUNDATIONS.md` for mathematical details

---

## Support

- **Issues**: File at [GitHub Issues](https://github.com/ai-whisperers/ternary-vae/issues)
- **Documentation**: See `docs/` directory
- **Email**: support@aiwhisperers.com
