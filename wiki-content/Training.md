# Training

This guide covers training workflows for Ternary VAE models.

## Basic Training Loop

```python
import torch
from src.models import TernaryVAE
from src.config import load_config
from src.losses import create_registry_from_training_config
from src.geometry import RiemannianAdam

# Load config
config = load_config("config.yaml")

# Create model
model = TernaryVAE(
    input_dim=19683,
    latent_dim=config.geometry.latent_dim,
    curvature=config.geometry.curvature,
)

# Create optimizer (Riemannian for manifold params)
optimizer = RiemannianAdam(model.parameters(), lr=config.optimizer.learning_rate)

# Create loss registry
loss_registry = create_registry_from_training_config(config)

# Training loop
for epoch in range(config.epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        outputs = model(batch)
        result = loss_registry.compose(outputs, batch)

        result.total.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Loss = {result.total.item():.4f}")
```

## Using Callbacks

Callbacks provide hooks into the training process:

```python
from src.training.callbacks import (
    CallbackList,
    EarlyStoppingCallback,
    CheckpointCallback,
    MetricsCallback,
)

callbacks = CallbackList([
    EarlyStoppingCallback(patience=20, monitor="val_loss"),
    CheckpointCallback(save_dir="checkpoints/", save_interval=10),
    MetricsCallback(log_interval=100),
])

# In training loop
callbacks.on_epoch_start(epoch)

for batch_idx, batch in enumerate(dataloader):
    # ... training step ...
    callbacks.on_batch_end(batch_idx, {"loss": loss.item()})

callbacks.on_epoch_end(epoch, {"val_loss": val_loss})

if callbacks.should_stop():
    break
```

## Callback Types

### EarlyStoppingCallback

Stops training when metric stops improving:

```python
early_stop = EarlyStoppingCallback(
    patience=20,           # Epochs to wait
    monitor="val_loss",    # Metric to watch
    mode="min",            # "min" or "max"
    min_delta=1e-4,        # Minimum improvement
)
```

### CheckpointCallback

Saves model checkpoints:

```python
checkpoint = CheckpointCallback(
    save_dir="checkpoints/",
    save_interval=10,      # Save every N epochs
    save_best=True,        # Save best model
    monitor="val_loss",
)
```

### CoveragePlateauCallback

Monitors latent space coverage:

```python
coverage = CoveragePlateauCallback(
    patience=10,
    threshold=0.95,        # Target coverage
    min_improvement=0.01,
)
```

## Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

# Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

# With warmup
warmup = LinearLR(optimizer, start_factor=0.1, total_iters=1000)
main_scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup, main_scheduler],
    milestones=[1000],
)
```

## Gradient Clipping

```python
from src.config import DEFAULT_GRAD_CLIP

# After backward, before optimizer step
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=DEFAULT_GRAD_CLIP,  # 1.0
)
```

## Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        outputs = model(batch)
        result = loss_registry.compose(outputs, batch)

    scaler.scale(result.total).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize
dist.init_process_group("nccl")
local_rank = dist.get_rank()

# Wrap model
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Use DistributedSampler
sampler = torch.utils.data.DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)
```

## Curriculum Learning

Gradually increase training difficulty:

```python
from src.models import CurriculumScheduler

scheduler = CurriculumScheduler(
    start_difficulty=0.1,
    end_difficulty=1.0,
    warmup_epochs=50,
)

for epoch in range(epochs):
    difficulty = scheduler.get_difficulty(epoch)

    # Use difficulty to filter data or adjust loss weights
    config.loss_weights.ranking = 0.1 * difficulty
```

## Homeostasis Controller

Maintains KL divergence stability:

```python
from src.models import HomeostasisController

controller = HomeostasisController(
    target_kl=1.0,
    kl_tolerance=0.5,
)

for epoch in range(epochs):
    # Get adaptive beta
    beta = controller.get_beta(current_kl)

    # Use in loss
    loss = recon_loss + beta * kl_loss

    # Update controller
    controller.update(current_kl)
```

## Logging & Monitoring

```python
from src.observability import setup_logging, get_logger, MetricsBuffer

# Setup logging
setup_logging(log_dir="logs/")
logger = get_logger(__name__)

# Metrics buffer for TensorBoard
buffer = MetricsBuffer(tensorboard_dir="runs/")

# During training
logger.info(f"Epoch {epoch}: loss={loss:.4f}")
buffer.add("train/loss", loss, step=global_step)
buffer.add("train/kl", kl.item(), step=global_step)
buffer.flush()
```

## Checkpointing

```python
# Save
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": config.to_dict(),
}, "checkpoint.pt")

# Load
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch = checkpoint["epoch"]
```

## Validation

```python
def validate(model, val_loader, loss_registry):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch)
            result = loss_registry.compose(outputs, batch)
            total_loss += result.total.item()

    return total_loss / len(val_loader)
```

## Best Practices

1. **Use Riemannian optimizer** for manifold parameters
2. **Gradient clipping** prevents instability
3. **Early stopping** saves compute
4. **Checkpoint regularly** to recover from failures
5. **Monitor KL divergence** for posterior collapse
6. **Use curriculum learning** for complex tasks

## See Also

- [[Configuration]] - Training config options
- [[Loss Functions]] - Available losses
- [[Models]] - Model architectures
