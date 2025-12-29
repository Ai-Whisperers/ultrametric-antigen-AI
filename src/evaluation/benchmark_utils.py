# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Shared utilities for benchmark scripts.

Extracted from measure_coupled_resolution.py and measure_manifold_resolution.py
to eliminate code duplication (D1.5 from DUPLICATION_REPORT).
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

from src.data import generate_all_ternary_operations

if TYPE_CHECKING:
    from src.models.ternary_vae_v5_6 import DualNeuralVAEV5


class BenchmarkBase:
    """Base class for VAE benchmarks with common initialization pattern.

    Provides:
    - Model setup (to device, eval mode)
    - All ternary operations preloaded as tensor
    - Operation count
    """

    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
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


def load_config(config_path: str = "configs/ternary_v5_6.yaml") -> Dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to config file

    Returns:
        Parsed config dictionary
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_device() -> str:
    """Get available compute device.

    Returns:
        'cuda' if available, else 'cpu'
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_v5_6_model(config: Dict) -> "DualNeuralVAEV5":
    """Create DualNeuralVAEV5 model from config.

    Args:
        config: Configuration dict with 'model' section

    Returns:
        Initialized model (not yet on device)
    """
    # Lazy import to allow sys.path.append in calling script
    from src.models.ternary_vae_v5_6 import DualNeuralVAEV5

    mc = config["model"]
    return DualNeuralVAEV5(
        input_dim=mc["input_dim"],
        latent_dim=mc["latent_dim"],
        rho_min=mc["rho_min"],
        rho_max=mc["rho_max"],
        lambda3_base=mc["lambda3_base"],
        lambda3_amplitude=mc["lambda3_amplitude"],
        eps_kl=mc["eps_kl"],
        gradient_balance=mc.get("gradient_balance", True),
        adaptive_scheduling=mc.get("adaptive_scheduling", True),
        use_statenet=mc.get("use_statenet", True),
    )


def load_checkpoint_safe(
    model: torch.nn.Module,
    checkpoint_dir: str,
    device: str,
    checkpoint_name: str = "best",
) -> Dict:
    """Load checkpoint with error handling.

    Args:
        model: Model to load weights into
        checkpoint_dir: Directory containing checkpoints
        device: Device to load to
        checkpoint_name: Which checkpoint to load ('best' or 'latest')

    Returns:
        Checkpoint dict with 'epoch' key (or {'epoch': 'init'} on failure)
    """
    # Lazy import to allow sys.path.append in calling script
    from src.training import CheckpointManager

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        logger.info("No checkpoint found, using random initialization")
        return {"epoch": "init"}

    try:
        manager = CheckpointManager(checkpoint_path)
        checkpoint = manager.load_checkpoint(model, checkpoint_name=checkpoint_name, device=device)
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint
    except Exception as e:
        logger.warning(f"Could not load checkpoint: {e}")
        return {"epoch": "init"}


def save_results(
    results: Dict,
    output_name: str,
    epoch: Any,
    output_dir: str = "reports/benchmarks",
) -> Path:
    """Save benchmark results to JSON file.

    Args:
        results: Results dictionary to save
        output_name: Base name for output file (e.g., 'manifold_resolution')
        epoch: Epoch identifier for filename
        output_dir: Directory to save to

    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"{output_name}_{epoch}.json"
    with open(output_file, "w") as f:
        json.dump(convert_to_python_types(results), f, indent=2)

    logger.info(f"Results saved to: {output_file}")
    return output_file
