# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Data loading and sampling strategies for training.

This module provides data loading utilities and sampling strategies,
including stratified sampling for ternary operations.

Extracted from scripts/train/train.py for reusability.

Key Features:
    - StratifiedBatchSampler: Ensures valuation-balanced batches
    - TernaryDataset: Dataset wrapper for ternary operations
    - Configurable oversampling for rare high-valuation points
"""

from typing import Callable, Iterator, List, Optional

import torch
from torch.utils.data import Dataset, Sampler


class TernaryDataset(Dataset):
    """Dataset for ternary operations.

    Wraps ternary operation tensors for use with DataLoader.

    Args:
        operations: Tensor of ternary operations
        indices: Tensor of operation indices
        valuations: Optional pre-computed valuations
        valuation_fn: Function to compute valuations if not provided
    """

    def __init__(
        self,
        operations: torch.Tensor,
        indices: torch.Tensor,
        valuations: Optional[torch.Tensor] = None,
        valuation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.operations = operations
        self.indices = indices

        if valuations is not None:
            self.valuations = valuations
        elif valuation_fn is not None:
            self.valuations = valuation_fn(indices)
        else:
            self.valuations = None

    def __len__(self) -> int:
        return len(self.operations)

    def __getitem__(self, idx: int):
        item = {
            "operation": self.operations[idx],
            "index": self.indices[idx],
        }
        if self.valuations is not None:
            item["valuation"] = self.valuations[idx]
        return item


class StratifiedBatchSampler(Sampler):
    """Stratified batch sampler ensuring valuation-balanced batches.

    V5.11.2 FIX: High-valuation points are extremely rare (v>=7 is ~9 out of 19683).
    Random sampling means most batches have NO high-valuation points.

    Solution: Stratified sampling - ensure each batch contains points from
    all valuation levels, with oversampling of rare high-valuation points.

    Args:
        valuations: Tensor of valuation levels for each sample
        batch_size: Target batch size
        high_v_threshold: Valuation level above which to oversample (default: 4)
        high_v_budget_ratio: Fraction of batch reserved for high-v (default: 0.2)
        drop_last: Whether to drop last incomplete batch (default: False)
    """

    def __init__(
        self,
        valuations: torch.Tensor,
        batch_size: int,
        high_v_threshold: int = 4,
        high_v_budget_ratio: float = 0.2,
        drop_last: bool = False,
    ):
        self.valuations = valuations.cpu()
        self.batch_size = batch_size
        self.high_v_threshold = high_v_threshold
        self.high_v_budget_ratio = high_v_budget_ratio
        self.drop_last = drop_last
        self.n_samples = len(valuations)

        # Pre-compute valuation groups
        self._build_valuation_groups()

    def _build_valuation_groups(self):
        """Group indices by valuation level."""
        self.valuation_groups = {}
        valuations_np = self.valuations.numpy()

        for i, v in enumerate(valuations_np):
            v = int(v)
            if v not in self.valuation_groups:
                self.valuation_groups[v] = []
            self.valuation_groups[v].append(i)

        # Convert to tensors
        for v in self.valuation_groups:
            self.valuation_groups[v] = torch.tensor(
                self.valuation_groups[v], dtype=torch.long
            )

        # Separate high and low valuation levels
        self.high_v_levels = [
            v for v in self.valuation_groups if v >= self.high_v_threshold
        ]
        self.low_v_levels = [
            v for v in self.valuation_groups if v < self.high_v_threshold
        ]

    def __iter__(self) -> Iterator[List[int]]:
        """Generate stratified batch indices."""
        high_v_budget = int(self.batch_size * self.high_v_budget_ratio)
        low_v_budget = self.batch_size - high_v_budget

        n_batches = len(self)

        for _ in range(n_batches):
            batch_indices = []

            # Sample from high-valuation levels (with replacement if needed)
            if self.high_v_levels:
                per_high_v = max(1, high_v_budget // len(self.high_v_levels))
                for v in self.high_v_levels:
                    group = self.valuation_groups[v]
                    # Sample with replacement for rare groups
                    sample_idx = torch.randint(0, len(group), (per_high_v,))
                    batch_indices.extend(group[sample_idx].tolist())

            # Sample from low-valuation levels (proportional to size)
            if self.low_v_levels:
                total_low = sum(
                    len(self.valuation_groups[v]) for v in self.low_v_levels
                )
                for v in self.low_v_levels:
                    group = self.valuation_groups[v]
                    n_to_sample = max(1, int(low_v_budget * len(group) / total_low))
                    sample_idx = torch.randint(0, len(group), (n_to_sample,))
                    batch_indices.extend(group[sample_idx].tolist())

            # Trim to exact batch size or pad if needed
            if len(batch_indices) > self.batch_size:
                indices = torch.randperm(len(batch_indices))[: self.batch_size]
                batch_indices = [batch_indices[i] for i in indices.tolist()]
            elif len(batch_indices) < self.batch_size:
                # Pad with random samples
                extra = torch.randint(
                    0, self.n_samples, (self.batch_size - len(batch_indices),)
                )
                batch_indices.extend(extra.tolist())

            yield batch_indices

    def __len__(self) -> int:
        """Number of batches."""
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def create_stratified_batches(
    indices: torch.Tensor,
    valuations: torch.Tensor,
    batch_size: int,
    device: str = "cpu",
    high_v_threshold: int = 4,
    high_v_budget_ratio: float = 0.2,
) -> List[torch.Tensor]:
    """Create stratified batch indices ensuring all valuation levels represented.

    This is the functional version of StratifiedBatchSampler for use in
    training scripts that don't use DataLoader.

    Args:
        indices: All operation indices
        valuations: Valuation level for each index
        batch_size: Target batch size
        device: Torch device for output tensors
        high_v_threshold: Valuation level above which to oversample
        high_v_budget_ratio: Fraction of batch reserved for high-valuation

    Returns:
        List of batch index tensors, each containing stratified samples
    """
    sampler = StratifiedBatchSampler(
        valuations=valuations,
        batch_size=batch_size,
        high_v_threshold=high_v_threshold,
        high_v_budget_ratio=high_v_budget_ratio,
    )

    batches = []
    for batch_indices in sampler:
        batch = torch.tensor(batch_indices, dtype=torch.long, device=device)
        batches.append(batch)

    return batches


__all__ = [
    "TernaryDataset",
    "StratifiedBatchSampler",
    "create_stratified_batches",
]
