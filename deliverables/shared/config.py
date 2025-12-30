# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Configuration management for all deliverables.

Provides centralized configuration including:
- Paths to checkpoints, data, and outputs
- API endpoints and credentials
- Runtime settings
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class Config:
    """Central configuration for all deliverables."""

    # Base paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    deliverables_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    # VAE checkpoints
    vae_checkpoint: Optional[str] = None
    fallback_checkpoints: list[str] = field(default_factory=lambda: [
        "sandbox-training/checkpoints/homeostatic_rich/best.pt",
        "sandbox-training/checkpoints/v5_11_homeostasis/best.pt",
        "checkpoints/pretrained_final.pt",
    ])

    # Data directories
    data_cache_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "cache")
    results_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "results")

    # API configurations
    ncbi_email: str = "researcher@university.edu"
    ncbi_api_key: Optional[str] = None
    stanford_hivdb_url: str = "https://hivdb.stanford.edu/graphql"

    # Runtime settings
    use_gpu: bool = True
    batch_size: int = 32
    random_seed: int = 42

    # Demo mode (use mock data when real data unavailable)
    demo_mode: bool = True
    verbose: bool = True

    def __post_init__(self):
        """Initialize paths and validate configuration."""
        # Ensure directories exist
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Find valid VAE checkpoint
        if self.vae_checkpoint is None:
            self.vae_checkpoint = self._find_checkpoint()

        # Load NCBI API key from environment
        if self.ncbi_api_key is None:
            self.ncbi_api_key = os.environ.get("NCBI_API_KEY")

    def _find_checkpoint(self) -> Optional[str]:
        """Find the first available VAE checkpoint."""
        for ckpt_path in self.fallback_checkpoints:
            full_path = self.project_root / ckpt_path
            if full_path.exists() and self._is_valid_checkpoint(full_path):
                return str(full_path)
        return None

    def _is_valid_checkpoint(self, path: Path) -> bool:
        """Check if a file is a valid PyTorch checkpoint (not Git LFS pointer)."""
        try:
            with open(path, "rb") as f:
                header = f.read(20)
                # Git LFS pointer files start with "version https://git-lfs"
                if header.startswith(b"version https://git"):
                    if self.verbose:
                        print(f"Warning: {path.name} is a Git LFS pointer, not actual checkpoint")
                        print("  Run: git lfs pull")
                    return False
                # Valid PyTorch files start with ZIP magic number (PK) or pickle
                return True
        except Exception:
            return False

    @property
    def has_vae(self) -> bool:
        """Check if a valid VAE checkpoint is available."""
        return self.vae_checkpoint is not None and Path(self.vae_checkpoint).exists()

    def get_partner_dir(self, partner: str) -> Path:
        """Get directory for a specific partner."""
        partner_dirs = {
            "alejandra": self.deliverables_root / "alejandra_rojas",
            "carlos": self.deliverables_root / "carlos_brizuela",
            "jose": self.deliverables_root / "jose_colbes",
            "hiv": self.deliverables_root / "hiv_research_package",
        }
        return partner_dirs.get(partner.lower(), self.deliverables_root)

    def get_cache_path(self, name: str) -> Path:
        """Get path for cached data file."""
        return self.data_cache_dir / name

    def save(self, path: Path):
        """Save configuration to JSON file."""
        config_dict = {
            "project_root": str(self.project_root),
            "vae_checkpoint": self.vae_checkpoint,
            "ncbi_email": self.ncbi_email,
            "stanford_hivdb_url": self.stanford_hivdb_url,
            "demo_mode": self.demo_mode,
        }
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load configuration from JSON file."""
        with open(path) as f:
            config_dict = json.load(f)
        return cls(**config_dict)


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config):
    """Set global configuration instance."""
    global _config
    _config = config
