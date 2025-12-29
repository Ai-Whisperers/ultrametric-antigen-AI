# Data Module

Data generation and loading for ternary operations.

## Purpose

This module handles the generation and loading of ternary operation data, which represents all 19,683 possible ternary logic operations (3^9 combinations).

## Ternary Operations

A ternary operation maps {-1, 0, 1}³ → {-1, 0, 1}, represented as a 9-element vector.

```python
from src.data import generate_all_ternary_operations, count_ternary_operations

# Count: 3^9 = 19,683 operations
count = count_ternary_operations()

# Generate all operations as numpy array (19683, 9)
operations = generate_all_ternary_operations()
```

## Dataset Classes

### Standard Dataset

```python
from src.data import TernaryOperationDataset

dataset = TernaryOperationDataset()
print(len(dataset))  # 19683

# Get a single operation
x = dataset[0]  # torch.Tensor of shape (9,)
```

### GPU-Resident Dataset (P2 Optimization)

For high-performance training, keep data resident on GPU:

```python
from src.data import GPUResidentTernaryDataset, create_gpu_resident_loaders

# Create GPU-resident dataset (zero host-device transfer during training)
dataset = GPUResidentTernaryDataset(device="cuda")

# Or use the factory function
train_loader, val_loader = create_gpu_resident_loaders(
    batch_size=256,
    val_split=0.1,
    device="cuda"
)
```

## DataLoaders

```python
from src.data import create_ternary_data_loaders, get_data_loader_info

# Create train/validation loaders
train_loader, val_loader = create_ternary_data_loaders(
    batch_size=256,
    val_split=0.1,
    shuffle=True,
    num_workers=4
)

# Get loader statistics
info = get_data_loader_info(train_loader)
print(info)  # {"num_batches": ..., "batch_size": ..., "total_samples": ...}
```

## Files

| File | Description |
|------|-------------|
| `generation.py` | Ternary operation generation functions |
| `dataset.py` | TernaryOperationDataset class |
| `loaders.py` | DataLoader creation utilities |
| `gpu_resident.py` | GPU-resident dataset for P2 optimization |

## Performance Notes

- **Standard loading**: Good for development and small-scale experiments
- **GPU-resident**: Recommended for training, eliminates data transfer bottleneck
- **Batch size**: 256-512 works well for most GPUs
- **Validation split**: 10% is typical (0.1)
