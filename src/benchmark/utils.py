"""Shared utilities for benchmark scripts.

Extracted from measure_coupled_resolution.py and measure_manifold_resolution.py
to eliminate code duplication (D1.5 from DUPLICATION_REPORT).
"""

import torch
import numpy as np
from typing import Any

from src.data import generate_all_ternary_operations


class BenchmarkBase:
    """Base class for VAE benchmarks with common initialization pattern.

    Provides:
    - Model setup (to device, eval mode)
    - All ternary operations preloaded as tensor
    - Operation count
    """

    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        """Initialize benchmark with model and operations.

        Args:
            model: VAE model to benchmark
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Generate all operations
        self.all_ops = torch.FloatTensor(generate_all_ternary_operations()).to(device)
        self.n_ops = len(self.all_ops)


def convert_to_python_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object to convert (dict, list, numpy type, or other)

    Returns:
        Object with numpy types converted to native Python types
    """
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
