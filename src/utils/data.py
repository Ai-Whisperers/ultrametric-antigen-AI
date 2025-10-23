"""Data generation utilities for Ternary VAE v5.5."""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Tuple


def generate_all_ternary_operations() -> np.ndarray:
    """Generate all 19,683 possible ternary operations.

    A ternary operation is a 9-dimensional truth table where each
    dimension can take values {-1, 0, +1}.

    Returns:
        np.ndarray: Array of shape (19683, 9) with dtype float32
                    Each row is a unique ternary operation
    """
    operations = []
    for i in range(3**9):
        op = []
        num = i
        for _ in range(9):
            op.append(num % 3 - 1)  # Convert {0,1,2} → {-1,0,+1}
            num //= 3
        operations.append(op)
    return np.array(operations, dtype=np.float32)


def validate_ternary_operation(op: np.ndarray) -> bool:
    """Validate that an operation is a valid ternary operation.

    Args:
        op: Array of shape (9,) to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if op.shape != (9,):
        return False
    return np.all(np.isin(op, [-1, 0, 1]))


class TernaryOperationDataset(Dataset):
    """Dataset of ternary operations.

    Args:
        operations: Array of ternary operations, shape (N, 9)
        transform: Optional transformation to apply to each sample
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        operations: Optional[np.ndarray] = None,
        transform: Optional[callable] = None,
        seed: int = 42
    ):
        if operations is None:
            operations = generate_all_ternary_operations()

        self.operations = torch.FloatTensor(operations)
        self.transform = transform
        self.seed = seed

        # Validate
        assert self.operations.shape[1] == 9, f"Expected 9 dims, got {self.operations.shape[1]}"

    def __len__(self) -> int:
        return len(self.operations)

    def __getitem__(self, idx: int) -> torch.Tensor:
        op = self.operations[idx]

        if self.transform:
            op = self.transform(op)

        return op

    def get_statistics(self) -> dict:
        """Compute dataset statistics.

        Returns:
            dict: Statistics including value distribution, sparsity, etc.
        """
        ops = self.operations.numpy()

        # Value distribution
        unique, counts = np.unique(ops, return_counts=True)
        value_dist = dict(zip(unique.tolist(), counts.tolist()))

        # Sparsity (fraction of zeros)
        sparsity = np.mean(ops == 0)

        # Balance (how evenly distributed are -1, 0, +1)
        balance = np.std(list(value_dist.values())) / np.mean(list(value_dist.values()))

        return {
            'num_operations': len(self),
            'dimension': ops.shape[1],
            'value_distribution': value_dist,
            'sparsity': float(sparsity),
            'balance': float(balance),
            'mean': float(ops.mean()),
            'std': float(ops.std())
        }


def split_dataset(
    dataset: TernaryOperationDataset,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """Split dataset into train/val/test.

    Args:
        dataset: Dataset to split
        train_frac: Fraction for training
        val_frac: Fraction for validation
        test_frac: Fraction for testing
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    from torch.utils.data import random_split

    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1.0"

    total_size = len(dataset)
    train_size = int(train_frac * total_size)
    val_size = int(val_frac * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 42
) -> torch.utils.data.DataLoader:
    """Create a DataLoader with deterministic settings.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed

    Returns:
        DataLoader instance
    """
    generator = torch.Generator().manual_seed(seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        generator=generator if shuffle else None,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id)
    )


def sample_operations(
    num_samples: int,
    replacement: bool = False,
    seed: int = 42
) -> np.ndarray:
    """Sample random ternary operations.

    Args:
        num_samples: Number of operations to sample
        replacement: Sample with replacement
        seed: Random seed

    Returns:
        np.ndarray: Sampled operations, shape (num_samples, 9)
    """
    np.random.seed(seed)
    all_ops = generate_all_ternary_operations()

    if replacement:
        indices = np.random.choice(len(all_ops), num_samples, replace=True)
    else:
        assert num_samples <= len(all_ops), f"Cannot sample {num_samples} without replacement from {len(all_ops)}"
        indices = np.random.choice(len(all_ops), num_samples, replace=False)

    return all_ops[indices]
