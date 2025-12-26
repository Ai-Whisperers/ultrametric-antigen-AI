# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Data generation and loading components.

This module handles ternary operation data:
- generation: Generate all possible ternary operations
- dataset: Dataset classes for ternary operations
- loaders: DataLoader creation and configuration
- gpu_resident: GPU-resident dataset for zero-transfer training (P2 optimization)
"""

from .dataset import TernaryOperationDataset
from .generation import (count_ternary_operations,
                         generate_all_ternary_operations,
                         generate_ternary_operation_by_index)
from .gpu_resident import (GPUBatchIterator, GPUResidentTernaryDataset,
                           create_gpu_resident_loaders)
from .loaders import create_ternary_data_loaders, get_data_loader_info
from .stratified import (
    TernaryDataset,
    StratifiedBatchSampler,
    create_stratified_batches,
    get_valuation_distribution,
)

__all__ = [
    # Generation
    "generate_all_ternary_operations",
    "count_ternary_operations",
    "generate_ternary_operation_by_index",
    # Dataset
    "TernaryOperationDataset",
    # Loaders (CPU-based, standard)
    "create_ternary_data_loaders",
    "get_data_loader_info",
    # GPU-Resident (P2 optimization)
    "GPUResidentTernaryDataset",
    "GPUBatchIterator",
    "create_gpu_resident_loaders",
    # Stratified sampling
    "TernaryDataset",
    "StratifiedBatchSampler",
    "create_stratified_batches",
    "get_valuation_distribution",
]
