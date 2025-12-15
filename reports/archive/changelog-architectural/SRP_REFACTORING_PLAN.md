# Single Responsibility Principle - Refactoring Plan

**Analysis Date:** 2025-11-23
**Codebase Version:** Ternary VAE v5.5
**Branch:** `refactor/srp-implementation`
**Status:** Aggressive Refactoring (No Backward Compatibility)

---

## Executive Summary

The Ternary VAE v5.5 codebase has a **well-structured foundation** but contains **critical SRP violations** that impact maintainability, testability, and extensibility. This document provides a comprehensive refactoring plan to improve code organization by separating:

1. **Training orchestration** from artifact management
2. **Model architecture** from loss computation and gradient tracking
3. **Raw training outputs** from validated/production models
4. **Configuration management** from execution logic

**Refactoring Approach:**
- **Clean rewrite in dedicated branch** (`refactor/srp-implementation`)
- **No backward compatibility patches** - aggressive, clean changes
- **Complete restructuring** - no temporary dual implementations
- **Merge to main when complete** - after full validation

**Key Metrics:**

| Component | Current Size | Target Size | Reduction |
|-----------|-------------|-------------|-----------|
| Trainer Class | 350+ lines | <200 lines | 43% |
| Model Class | 632 lines | <400 lines | 37% |
| Main Function | 50+ lines | <30 lines | 40% |

---

## Critical SRP Violations

### 1. Trainer Class: Multiple Responsibilities

**File:** `scripts/train/train_ternary_v5_5.py`
**Class:** `DNVAETrainerV5` (Lines 79-548)
**Severity:** HIGH

**Current Responsibilities (7+):**
1. Training loop orchestration
2. Checkpoint saving/loading
3. Metrics tracking and history
4. Validation execution
5. Console logging/monitoring
6. Configuration scheduling (temperature, beta, LR)
7. Coverage evaluation

**Example Violation (Checkpoint Management in Trainer):**

```python
# scripts/train/train_ternary_v5_5.py:445-461
def save_checkpoint(self, is_best=False):
    """Save checkpoint."""
    checkpoint = {
        'epoch': self.epoch,
        'model': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'best_val_loss': self.best_val_loss,
        'H_A_history': self.H_A_history,
        'H_B_history': self.H_B_history,
        # ... 15+ more fields
    }
    torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')
    if is_best:
        torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
    if self.epoch % self.config['checkpoint_freq'] == 0:
        torch.save(checkpoint, self.checkpoint_dir / f'epoch_{self.epoch}.pt')
```

**Impact:**
- Trainer is 350+ lines (should be <200)
- Cannot test checkpointing independently
- Cannot swap checkpoint backends (S3, database)
- Mixing training logic with I/O operations

**Proposed Fix:** Extract `CheckpointManager` class

---

### 2. Model Class: Architecture + Loss + Tracking

**File:** `src/models/ternary_vae_v5_5.py`
**Class:** `DualNeuralVAEV5` (Lines 150-632)
**Severity:** HIGH

**Current Responsibilities (9+):**
1. Model architecture definition (encoders/decoders)
2. Forward pass implementation
3. Loss computation (multi-term complex loss)
4. Gradient tracking and EMA updates
5. StateNet control and corrections
6. Sampling/generation logic
7. Entropy computation
8. Phase scheduling (permeability)
9. Adaptive scheduling updates

**Example Violation (Loss Computation in Model):**

```python
# src/models/ternary_vae_v5_5.py:540-603
def loss_function(self, x, outputs, entropy_weight_B=0.05,
                  repulsion_weight_B=0.01, free_bits=0.0):
    """Compute total loss with adaptive regime-aware scaling."""
    # Reconstruction losses
    ce_A = F.cross_entropy(...)
    ce_B = F.cross_entropy(...)

    # KL divergence losses
    kl_A = self.compute_kl_divergence(...)
    kl_B = self.compute_kl_divergence(...)

    # Entropy regularization for VAE-B
    probs_B = F.softmax(outputs['logits_B'], dim=-1).mean(dim=0)
    entropy_B = -(probs_B * torch.log(probs_B + 1e-10)).sum()
    entropy_loss_B = -entropy_B

    # Repulsion loss
    repulsion_B = self.repulsion_loss(outputs['z_B'])

    # Aggregate with gradient normalization
    # ... 40+ more lines
```

**Impact:**
- Model is 632 lines (should be <400)
- Cannot test loss function independently
- Cannot swap loss computation strategies
- Difficult to add new loss terms
- Mixes neural network definition with training objectives

**Proposed Fix:** Extract `DualVAELoss` class

---

### 3. Main Function: Configuration + Data + Orchestration

**File:** `scripts/train/train_ternary_v5_5.py`
**Function:** `main()` (Lines 509-593)
**Severity:** HIGH

**Current Responsibilities (6+):**
1. Configuration loading from YAML
2. Random seed initialization
3. Dataset generation
4. Data splitting (train/val/test)
5. DataLoader creation
6. Trainer initialization and execution

**Example Violation:**

```python
# scripts/train/train_ternary_v5_5.py:535-556
# Generate dataset
print("\nGenerating dataset...")
operations = generate_all_ternary_operations()
dataset = TernaryOperationDataset(operations)

# Split dataset
train_split = config['train_split']
val_split = config['val_split']
test_split = config['test_split']

train_size = int(train_split * len(dataset))
val_size = int(val_split * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(...)
```

**Impact:**
- `main()` is too long and handles too many concerns
- Cannot reuse dataset generation without running training
- Testing requires mocking entire pipeline
- Hard to inject dependencies for testing

**Proposed Fix:** Extract factory functions and configuration manager

---

### 4. Utils Modules: Overly Broad Scope

**Files:**
- `src/utils/data.py` (199 lines)
- `src/utils/metrics.py` (276 lines)

**Severity:** MEDIUM

**data.py Responsibilities (6+):**
1. Data generation (`generate_all_ternary_operations()`)
2. Validation (`validate_ternary_operation()`)
3. Dataset definition (`TernaryOperationDataset`)
4. Dataset utilities (`split_dataset()`, `create_dataloader()`)
5. Sampling (`sample_operations()`)
6. Statistics (`get_statistics()`)

**metrics.py Responsibilities (6+):**
1. Coverage evaluation
2. Entropy computation
3. Diversity analysis
4. Reconstruction accuracy
5. Distribution analysis
6. Stateful tracking (`CoverageTracker`)

**Impact:**
- Modules are too large and multipurpose
- Difficult to find specific functionality
- Tight coupling between unrelated functions
- Hard to test individual components

**Proposed Fix:** Split into focused modules by domain

---

## Proposed Directory Structure

### Current Structure Issues

```
ternary-vaes/
├── src/
│   ├── models/           # ❌ Model + Loss + Tracking mixed
│   └── utils/            # ❌ Too broad, mixed concerns
├── sandbox-training/     # ❌ Raw outputs only, no separation
└── reports/              # ✓ Good separation
```

### Proposed Structure

```
ternary-vaes/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ternary_vae_v5_5.py    # REFACTOR: Only architecture
│   │   ├── encoders.py             # NEW: Extract encoders
│   │   ├── decoders.py             # NEW: Extract decoders
│   │   └── statenet.py             # NEW: Extract StateNet
│   │
│   ├── training/                   # NEW: Training orchestration
│   │   ├── __init__.py
│   │   ├── trainer.py              # REFACTOR: Only training loop
│   │   ├── schedulers.py           # NEW: Beta, temp, LR schedulers
│   │   ├── monitor.py              # NEW: Logging and monitoring
│   │   └── validators.py           # NEW: Validation logic
│   │
│   ├── artifacts/                  # NEW: Artifact management
│   │   ├── __init__.py
│   │   ├── checkpoint_manager.py   # NEW: Save/load checkpoints
│   │   ├── metadata.py             # NEW: Checkpoint metadata
│   │   └── repository.py           # NEW: Artifact versioning
│   │
│   ├── data/                       # REFACTOR: Split from utils
│   │   ├── __init__.py
│   │   ├── generation.py           # Ternary operations generation
│   │   ├── validation.py           # Data validation
│   │   ├── dataset.py              # Dataset classes
│   │   └── loaders.py              # DataLoader creation
│   │
│   ├── metrics/                    # REFACTOR: Split from utils
│   │   ├── __init__.py
│   │   ├── coverage.py             # Coverage metrics
│   │   ├── entropy.py              # Entropy computation
│   │   ├── reconstruction.py       # Accuracy metrics
│   │   └── tracking.py             # CoverageTracker
│   │
│   └── losses/                     # NEW: Separate loss functions
│       ├── __init__.py
│       ├── reconstruction.py       # Cross-entropy losses
│       ├── regularization.py       # KL, entropy, repulsion
│       └── aggregate.py            # Combined loss computation
│
├── scripts/
│   └── train/
│       └── train_ternary_v5_5.py  # REFACTOR: Simpler orchestration
│
├── artifacts/                      # NEW: Organized artifact storage
│   ├── raw/                        # Direct training outputs
│   │   └── v5_5_20251123/
│   │       ├── checkpoints/
│   │       │   ├── manifest.json   # Training metadata
│   │       │   ├── latest.pt
│   │       │   └── epoch_*.pt
│   │       ├── metrics.json        # Training metrics
│   │       └── training.log        # Full training log
│   │
│   ├── validated/                  # Passed validation tests
│   │   └── v5_5_epoch103_validated/
│   │       ├── model.pt
│   │       ├── validation_report.json
│   │       ├── test_results.json
│   │       └── config.yaml
│   │
│   └── production/                 # Deployment-ready models
│       └── v5_5_prod_v1.0/
│           ├── model.pt
│           ├── config.yaml
│           ├── performance_metrics.json
│           ├── deployment_notes.md
│           └── README.md
│
├── configs/
│   ├── ternary_v5_5.yaml          # Main configuration
│   ├── schema.json                # NEW: Validation schema
│   └── defaults.yaml              # NEW: Default values
│
├── tests/
│   ├── test_models/
│   ├── test_training/             # NEW: Training component tests
│   │   ├── test_trainer.py
│   │   ├── test_schedulers.py
│   │   └── test_monitor.py
│   ├── test_artifacts/            # NEW: Artifact management tests
│   │   └── test_checkpoint_manager.py
│   ├── test_losses/               # NEW: Loss function tests
│   │   └── test_loss_computation.py
│   └── test_integration/          # NEW: End-to-end tests
│       └── test_training_pipeline.py
│
├── reports/
│   ├── training/                  # Training session reports
│   │   └── 2025-11-23/
│   │       ├── training_report.md
│   │       └── metrics.json
│   ├── analysis/                  # Deep-dive analyses
│   │   ├── coverage_analysis.md
│   │   ├── entropy_analysis.md
│   │   └── gradient_analysis.md
│   └── README.md
│
└── docs/
    ├── architecture/
    │   ├── SEPARATION_OF_CONCERNS.md  # NEW: SoC guide
    │   ├── MODULE_RESPONSIBILITIES.md # NEW: Module guide
    │   └── REFACTORING_GUIDE.md       # NEW: Implementation guide
    └── ...
```

---

## Detailed Refactoring Proposals

### Refactoring 1: Extract CheckpointManager

**Objective:** Separate checkpoint I/O from training logic

**Current Code Location:** `scripts/train/train_ternary_v5_5.py:445-461`

**Proposed Implementation:**

```python
# src/artifacts/checkpoint_manager.py
from pathlib import Path
from typing import Dict, Optional, List
import torch
import json
from datetime import datetime

class CheckpointManager:
    """Manages checkpoint saving, loading, and lifecycle."""

    def __init__(self, directory: Path, strategy: str = 'best+periodic'):
        """
        Args:
            directory: Base directory for checkpoints
            strategy: 'best+periodic', 'all', 'best_only', 'latest_only'
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.strategy = strategy
        self.manifest_path = self.directory / 'manifest.json'
        self._load_or_create_manifest()

    def save(self,
             state: Dict,
             epoch: int,
             is_best: bool = False,
             metadata: Optional[Dict] = None):
        """Save checkpoint with metadata tracking."""
        checkpoint_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'metadata': metadata or {}
        }

        # Always save latest
        latest_path = self.directory / 'latest.pt'
        torch.save(checkpoint_data, latest_path)

        # Save best if applicable
        if is_best:
            best_path = self.directory / 'best.pt'
            torch.save(checkpoint_data, best_path)
            self._update_manifest('best', epoch, best_path, metadata)

        # Save periodic based on strategy
        if self._should_save_periodic(epoch):
            periodic_path = self.directory / f'epoch_{epoch}.pt'
            torch.save(checkpoint_data, periodic_path)
            self._update_manifest('periodic', epoch, periodic_path, metadata)

    def load(self, checkpoint_id: str = 'latest') -> Dict:
        """Load checkpoint by ID (latest, best, epoch_N)."""
        if checkpoint_id == 'latest':
            path = self.directory / 'latest.pt'
        elif checkpoint_id == 'best':
            path = self.directory / 'best.pt'
        elif checkpoint_id.startswith('epoch_'):
            path = self.directory / f'{checkpoint_id}.pt'
        else:
            raise ValueError(f"Unknown checkpoint ID: {checkpoint_id}")

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        return torch.load(path, map_location='cpu', weights_only=False)

    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints with metadata."""
        manifest = self._load_manifest()
        return manifest.get('checkpoints', [])

    def _should_save_periodic(self, epoch: int) -> bool:
        """Determine if periodic checkpoint should be saved."""
        if self.strategy == 'all':
            return True
        elif self.strategy == 'best_only' or self.strategy == 'latest_only':
            return False
        elif self.strategy == 'best+periodic':
            return epoch % 10 == 0  # Every 10 epochs
        return False

    def _update_manifest(self, checkpoint_type: str, epoch: int,
                        path: Path, metadata: Optional[Dict]):
        """Update manifest with new checkpoint info."""
        manifest = self._load_manifest()
        manifest['checkpoints'].append({
            'type': checkpoint_type,
            'epoch': epoch,
            'path': str(path),
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        })
        self._save_manifest(manifest)

    def _load_or_create_manifest(self):
        """Load existing manifest or create new one."""
        if not self.manifest_path.exists():
            self._save_manifest({'checkpoints': []})

    def _load_manifest(self) -> Dict:
        """Load manifest from disk."""
        with open(self.manifest_path, 'r') as f:
            return json.load(f)

    def _save_manifest(self, manifest: Dict):
        """Save manifest to disk."""
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)


# Usage in trainer:
class DNVAETrainerV5:
    def __init__(self, config, model, train_loader, val_loader, device):
        # ... other initialization
        self.checkpoint_manager = CheckpointManager(
            directory=config['checkpoint_dir'],
            strategy='best+periodic'
        )

    def train_epoch(self, epoch):
        # ... training logic

        # Save checkpoint
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_val_loss': self.best_val_loss
        }

        metadata = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'coverage_A': coverage_A,
            'coverage_B': coverage_B
        }

        is_best = val_loss < self.best_val_loss
        self.checkpoint_manager.save(state, epoch, is_best, metadata)
```

**Benefits:**
- Trainer reduced by ~40 lines
- Checkpoint logic testable independently
- Easy to change storage backend (add S3, database support)
- Manifest tracking enables checkpoint browsing
- Clear separation of concerns

---

### Refactoring 2: Extract Schedulers

**Objective:** Separate parameter scheduling from training loop

**Current Code Location:** `scripts/train/train_ternary_v5_5.py:138-198`

**Proposed Implementation:**

```python
# src/training/schedulers.py
from abc import ABC, abstractmethod
from typing import Dict

class Scheduler(ABC):
    """Base class for parameter schedulers."""

    @abstractmethod
    def get_value(self, epoch: int) -> float:
        """Get parameter value for given epoch."""
        pass


class LinearScheduler(Scheduler):
    """Linear interpolation between start and end values."""

    def __init__(self, start_val: float, end_val: float,
                 total_epochs: int, start_epoch: int = 0):
        self.start_val = start_val
        self.end_val = end_val
        self.total_epochs = total_epochs
        self.start_epoch = start_epoch

    def get_value(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            return self.start_val
        progress = min((epoch - self.start_epoch) / self.total_epochs, 1.0)
        return self.start_val + (self.end_val - self.start_val) * progress


class CyclicScheduler(Scheduler):
    """Cyclic scheduling with cosine function."""

    def __init__(self, base_val: float, amplitude: float, period: int):
        self.base_val = base_val
        self.amplitude = amplitude
        self.period = period

    def get_value(self, epoch: int) -> float:
        import math
        phase = (epoch % self.period) / self.period * 2 * math.pi
        return self.base_val + self.amplitude * math.cos(phase)


class MultiStageScheduler(Scheduler):
    """Multi-stage scheduling from list of (epoch, value) pairs."""

    def __init__(self, schedule: list):
        """
        Args:
            schedule: List of {'epoch': int, 'value': float} dicts
        """
        self.schedule = sorted(schedule, key=lambda x: x['epoch'])

    def get_value(self, epoch: int) -> float:
        value = self.schedule[0]['value']
        for entry in self.schedule:
            if epoch >= entry['epoch']:
                value = entry['value']
        return value


class TemperatureScheduler:
    """Manages temperature scheduling for both VAEs with Phase 4 support."""

    def __init__(self, config: Dict):
        self.config = config

        # VAE-A: Linear + Cyclic
        self.vae_a_linear = LinearScheduler(
            start_val=config['vae_a']['temp_start'],
            end_val=config['vae_a']['temp_end'],
            total_epochs=config['total_epochs']
        )

        # VAE-B: Linear only
        self.vae_b_linear = LinearScheduler(
            start_val=config['vae_b']['temp_start'],
            end_val=config['vae_b']['temp_end'],
            total_epochs=config['total_epochs']
        )

        self.phase4_start = config['phase_transitions']['ultra_exploration_start']

    def get_temperature(self, epoch: int, vae: str = 'A') -> float:
        """Get temperature for given epoch and VAE."""
        if vae == 'A':
            base_temp = self.vae_a_linear.get_value(epoch)

            # Phase 4: Add cyclic boost
            if epoch >= self.phase4_start and self.config['vae_a'].get('temp_cyclic', False):
                amplitude = self.config['vae_a'].get('temp_boost_amplitude', 0.1)
                period = 20
                cyclic = CyclicScheduler(0, amplitude, period)
                base_temp += cyclic.get_value(epoch - self.phase4_start)

            return base_temp

        elif vae == 'B':
            # Phase 4: Override temperature
            if epoch >= self.phase4_start:
                return self.config['vae_b'].get('temp_phase4',
                                                 self.vae_b_linear.get_value(epoch))
            return self.vae_b_linear.get_value(epoch)

        else:
            raise ValueError(f"Unknown VAE: {vae}")


class BetaScheduler:
    """Manages beta (KL weight) scheduling with warmup."""

    def __init__(self, config: Dict):
        self.config = config

        # VAE-A warmup
        self.vae_a_warmup = LinearScheduler(
            start_val=config['vae_a']['beta_start'],
            end_val=config['vae_a']['beta_end'],
            total_epochs=config['vae_a']['beta_warmup_epochs']
        )

        # VAE-B warmup
        self.vae_b_warmup = LinearScheduler(
            start_val=config['vae_b']['beta_start'],
            end_val=config['vae_b']['beta_end'],
            total_epochs=config['vae_b']['beta_warmup_epochs']
        )

    def get_beta(self, epoch: int, vae: str = 'A') -> float:
        """Get beta value for given epoch and VAE."""
        if vae == 'A':
            return self.vae_a_warmup.get_value(epoch)
        elif vae == 'B':
            return self.vae_b_warmup.get_value(epoch)
        else:
            raise ValueError(f"Unknown VAE: {vae}")


class LearningRateScheduler:
    """Manages learning rate scheduling."""

    def __init__(self, lr_schedule: list):
        """
        Args:
            lr_schedule: List of {'epoch': int, 'lr': float} dicts
        """
        self.scheduler = MultiStageScheduler(
            [{'epoch': entry['epoch'], 'value': entry['lr']}
             for entry in lr_schedule]
        )

    def get_lr(self, epoch: int) -> float:
        """Get learning rate for given epoch."""
        return self.scheduler.get_value(epoch)


# Usage in trainer:
class DNVAETrainerV5:
    def __init__(self, config, model, train_loader, val_loader, device):
        # ... other initialization

        self.temp_scheduler = TemperatureScheduler(config)
        self.beta_scheduler = BetaScheduler(config)
        self.lr_scheduler = LearningRateScheduler(config['optimizer']['lr_schedule'])

    def train_epoch(self, epoch):
        # Get scheduled values
        temp_A = self.temp_scheduler.get_temperature(epoch, 'A')
        temp_B = self.temp_scheduler.get_temperature(epoch, 'B')
        beta_A = self.beta_scheduler.get_beta(epoch, 'A')
        beta_B = self.beta_scheduler.get_beta(epoch, 'B')
        lr = self.lr_scheduler.get_lr(epoch)

        # Apply to model and optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # ... training logic
```

**Benefits:**
- Trainer reduced by ~60 lines
- Schedulers testable independently
- Easy to add new scheduling strategies
- Clear separation of scheduling logic from training
- Reusable across different models

---

### Refactoring 3: Extract Loss Computation

**Objective:** Separate loss computation from model architecture

**Current Code Location:** `src/models/ternary_vae_v5_5.py:540-603`

**Proposed Implementation:**

```python
# src/losses/reconstruction.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    """Categorical cross-entropy reconstruction loss."""

    def __init__(self, num_categories: int = 3):
        super().__init__()
        self.num_categories = num_categories

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, seq_len, num_categories]
            target: [batch, seq_len] with values in {-1, 0, 1}

        Returns:
            Reconstruction loss (scalar)
        """
        # Convert {-1, 0, 1} to {0, 1, 2}
        target_idx = (target + 1).long()

        batch_size, seq_len, _ = logits.shape
        logits_flat = logits.reshape(-1, self.num_categories)
        target_flat = target_idx.reshape(-1)

        return F.cross_entropy(logits_flat, target_flat)


# src/losses/regularization.py
import torch
import torch.nn as nn

class KLDivergenceLoss(nn.Module):
    """KL divergence loss with free bits support."""

    def __init__(self, free_bits: float = 0.0):
        super().__init__()
        self.free_bits = free_bits

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]

        Returns:
            KL divergence (scalar)
        """
        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        # Free bits: allow first `free_bits` nats without penalty
        if self.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)

        return kl_per_dim.sum(dim=1).mean()


class EntropyRegularization(nn.Module):
    """Entropy regularization to encourage output diversity."""

    def __init__(self, num_categories: int = 3):
        super().__init__()
        self.num_categories = num_categories

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, seq_len, num_categories]

        Returns:
            Negative entropy (to minimize -> maximize entropy)
        """
        # Average probabilities across batch and sequence
        probs = F.softmax(logits, dim=-1).mean(dim=(0, 1))

        # Entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + 1e-10)).sum()

        # Return negative (to minimize -> maximize entropy)
        return -entropy


class RepulsionLoss(nn.Module):
    """Repulsion loss to encourage diverse latent representations."""

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, latent_dim]

        Returns:
            Repulsion loss (scalar)
        """
        # Compute pairwise distances
        distances = torch.cdist(z, z, p=2)

        # Penalize small distances (exclude diagonal)
        mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        distances_masked = distances[mask]

        # Inverse distance penalty
        repulsion = 1.0 / (distances_masked + 1e-6)

        return repulsion.mean()


# src/losses/aggregate.py
import torch
import torch.nn as nn
from .reconstruction import ReconstructionLoss
from .regularization import KLDivergenceLoss, EntropyRegularization, RepulsionLoss

class DualVAELoss(nn.Module):
    """Aggregated loss for Dual VAE system."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # Loss components
        self.recon_loss = ReconstructionLoss(num_categories=3)
        self.kl_loss = KLDivergenceLoss(free_bits=config.get('free_bits', 0.0))
        self.entropy_reg = EntropyRegularization(num_categories=3)
        self.repulsion_loss = RepulsionLoss()

    def forward(self, x: torch.Tensor, outputs: dict, weights: dict) -> dict:
        """
        Args:
            x: Input [batch, seq_len] with values in {-1, 0, 1}
            outputs: Model outputs containing:
                - logits_A, logits_B
                - mu_A, logvar_A, mu_B, logvar_B
                - z_A, z_B
                - beta_A, beta_B
            weights: Loss weights:
                - lambda1, lambda2, lambda3
                - entropy_weight_B, repulsion_weight_B

        Returns:
            Dictionary with:
                - loss: Total loss
                - ce_A, ce_B: Reconstruction losses
                - kl_A, kl_B: KL divergence losses
                - entropy_B, repulsion_B: Regularization losses
        """
        # Reconstruction losses
        ce_A = self.recon_loss(outputs['logits_A'], x)
        ce_B = self.recon_loss(outputs['logits_B'], x)

        # KL divergence losses
        kl_A = self.kl_loss(outputs['mu_A'], outputs['logvar_A'])
        kl_B = self.kl_loss(outputs['mu_B'], outputs['logvar_B'])

        # VAE-A loss
        loss_A = ce_A + outputs['beta_A'] * kl_A

        # VAE-B loss with regularization
        entropy_B = self.entropy_reg(outputs['logits_B'])
        repulsion_B = self.repulsion_loss(outputs['z_B'])

        loss_B = (ce_B +
                  outputs['beta_B'] * kl_B +
                  weights.get('entropy_weight_B', 0.05) * entropy_B +
                  weights.get('repulsion_weight_B', 0.01) * repulsion_B)

        # Weighted combination
        total_loss = (weights['lambda1'] * loss_A +
                      weights['lambda2'] * loss_B +
                      weights['lambda3'] * outputs.get('entropy_alignment', 0.0))

        return {
            'loss': total_loss,
            'ce_A': ce_A.item(),
            'ce_B': ce_B.item(),
            'kl_A': kl_A.item(),
            'kl_B': kl_B.item(),
            'entropy_B': entropy_B.item(),
            'repulsion_B': repulsion_B.item(),
            'loss_A': loss_A.item(),
            'loss_B': loss_B.item()
        }


# Usage in model:
class DualNeuralVAEV5(nn.Module):
    def __init__(self, input_dim=9, latent_dim=16, config=None):
        super().__init__()
        # ... model architecture
        self.loss_fn = DualVAELoss(config)

    def compute_loss(self, x, outputs, weights):
        """Delegate to loss function."""
        return self.loss_fn(x, outputs, weights)
```

**Benefits:**
- Model reduced by ~65 lines
- Loss components testable independently
- Easy to add new loss terms
- Clear separation of architecture vs objectives
- Reusable loss components

---

### Refactoring 4: Artifact Management Structure

**Objective:** Separate raw outputs from validated/production models

**Current Structure:**
```
sandbox-training/
└── checkpoints/v5_5/
    ├── latest.pt
    ├── best.pt
    └── epoch_*.pt
```

**Proposed Structure:**
```
artifacts/
├── raw/                           # Direct training outputs
│   ├── v5_5_20251123_103epochs/  # Session identifier
│   │   ├── checkpoints/
│   │   │   ├── manifest.json     # Checkpoint registry
│   │   │   ├── latest.pt
│   │   │   ├── best.pt           # Best val loss
│   │   │   └── epoch_*.pt
│   │   ├── metrics/
│   │   │   ├── training_metrics.json
│   │   │   ├── coverage_history.json
│   │   │   └── loss_history.json
│   │   ├── config.yaml           # Exact config used
│   │   ├── training.log          # Full training log
│   │   └── README.md             # Session summary
│   │
│   └── README.md                 # Raw artifacts documentation
│
├── validated/                    # Passed validation tests
│   ├── v5_5_epoch70_validated/  # Validated checkpoint
│   │   ├── model.pt             # Cleaned model (no optimizer)
│   │   ├── config.yaml          # Original config
│   │   ├── validation/
│   │   │   ├── test_results.json
│   │   │   ├── coverage_report.json
│   │   │   ├── entropy_analysis.json
│   │   │   └── generalization_report.json
│   │   ├── metadata.json        # Model card
│   │   └── README.md
│   │
│   └── README.md                # Validation criteria
│
└── production/                   # Deployment-ready models
    ├── v5_5_prod_v1.0/          # Production version
    │   ├── model.pt             # Optimized model
    │   ├── config.yaml
    │   ├── performance/
    │   │   ├── benchmarks.json
    │   │   ├── latency_profile.json
    │   │   └── resource_usage.json
    │   ├── deployment/
    │   │   ├── docker/
    │   │   ├── kubernetes/
    │   │   └── deployment_notes.md
    │   ├── metadata.json        # Complete model card
    │   ├── CHANGELOG.md
    │   └── README.md
    │
    └── README.md                # Production deployment guide
```

**Implementation - Artifact Promotion Workflow:**

```python
# src/artifacts/repository.py
from pathlib import Path
from typing import Dict, Optional
import shutil
import json
from datetime import datetime

class ArtifactRepository:
    """Manages artifact lifecycle: raw → validated → production."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / 'raw'
        self.validated_dir = self.base_dir / 'validated'
        self.production_dir = self.base_dir / 'production'

        # Create directories
        for dir in [self.raw_dir, self.validated_dir, self.production_dir]:
            dir.mkdir(parents=True, exist_ok=True)

    def register_training_session(self,
                                   session_id: str,
                                   checkpoint_dir: Path,
                                   config: Dict,
                                   metrics: Dict) -> Path:
        """Register a completed training session."""
        session_dir = self.raw_dir / session_id
        session_dir.mkdir(exist_ok=True)

        # Copy checkpoints
        checkpoint_dest = session_dir / 'checkpoints'
        shutil.copytree(checkpoint_dir, checkpoint_dest)

        # Save config
        with open(session_dir / 'config.yaml', 'w') as f:
            import yaml
            yaml.dump(config, f)

        # Save metrics
        metrics_dir = session_dir / 'metrics'
        metrics_dir.mkdir(exist_ok=True)
        with open(metrics_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Create README
        self._create_session_readme(session_dir, session_id, config, metrics)

        return session_dir

    def promote_to_validated(self,
                              session_id: str,
                              checkpoint_name: str,
                              validation_results: Dict) -> Path:
        """Promote a checkpoint to validated status."""
        # Load raw checkpoint
        raw_session = self.raw_dir / session_id
        checkpoint_path = raw_session / 'checkpoints' / f'{checkpoint_name}.pt'

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Create validated artifact
        validated_id = f"{session_id}_{checkpoint_name}_validated"
        validated_dir = self.validated_dir / validated_id
        validated_dir.mkdir(exist_ok=True)

        # Copy model (clean - no optimizer state)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        clean_model = {
            'model': checkpoint['model'],
            'epoch': checkpoint['epoch'],
            'config': checkpoint.get('config', {})
        }
        torch.save(clean_model, validated_dir / 'model.pt')

        # Copy config
        shutil.copy(raw_session / 'config.yaml', validated_dir / 'config.yaml')

        # Save validation results
        validation_dir = validated_dir / 'validation'
        validation_dir.mkdir(exist_ok=True)

        for key, value in validation_results.items():
            with open(validation_dir / f'{key}.json', 'w') as f:
                json.dump(value, f, indent=2)

        # Create metadata (model card)
        metadata = self._create_model_card(
            session_id, checkpoint_name, validation_results
        )
        with open(validated_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        return validated_dir

    def promote_to_production(self,
                               validated_id: str,
                               version: str,
                               deployment_config: Dict) -> Path:
        """Promote validated model to production."""
        validated_dir = self.validated_dir / validated_id

        if not validated_dir.exists():
            raise FileNotFoundError(f"Validated artifact not found: {validated_id}")

        # Create production artifact
        prod_id = f"v{version.replace('.', '_')}"
        prod_dir = self.production_dir / prod_id
        prod_dir.mkdir(exist_ok=True)

        # Copy model and config
        shutil.copy(validated_dir / 'model.pt', prod_dir / 'model.pt')
        shutil.copy(validated_dir / 'config.yaml', prod_dir / 'config.yaml')
        shutil.copy(validated_dir / 'metadata.json', prod_dir / 'metadata.json')

        # Add deployment configuration
        deployment_dir = prod_dir / 'deployment'
        deployment_dir.mkdir(exist_ok=True)

        with open(deployment_dir / 'config.json', 'w') as f:
            json.dump(deployment_config, f, indent=2)

        # Create changelog
        self._create_changelog(prod_dir, version, validated_id)

        return prod_dir

    def list_artifacts(self, stage: str = 'all') -> list:
        """List all artifacts at given stage."""
        if stage == 'raw' or stage == 'all':
            raw = [d.name for d in self.raw_dir.iterdir() if d.is_dir()]
        if stage == 'validated' or stage == 'all':
            validated = [d.name for d in self.validated_dir.iterdir() if d.is_dir()]
        if stage == 'production' or stage == 'all':
            production = [d.name for d in self.production_dir.iterdir() if d.is_dir()]

        if stage == 'all':
            return {
                'raw': raw,
                'validated': validated,
                'production': production
            }
        elif stage == 'raw':
            return raw
        elif stage == 'validated':
            return validated
        elif stage == 'production':
            return production

    def _create_session_readme(self, session_dir, session_id, config, metrics):
        """Create README for training session."""
        content = f"""# Training Session: {session_id}

**Date:** {datetime.now().isoformat()}
**Status:** Raw (not validated)

## Configuration

- Model: {config.get('model', {}).get('latent_dim', 'N/A')}-dim latent space
- Epochs: {metrics.get('epoch', 'N/A')}
- Device: {metrics.get('device', 'N/A')}

## Performance

- Best Val Loss: {metrics.get('best_val_loss', 'N/A')}
- Coverage: {metrics.get('coverage', 'N/A')}

## Files

- `checkpoints/` - Model checkpoints
- `metrics/` - Training metrics
- `config.yaml` - Complete configuration
- `training.log` - Full training log

## Next Steps

1. Run validation tests
2. Promote to validated/ if tests pass
3. Deploy to production/ if approved
"""
        with open(session_dir / 'README.md', 'w') as f:
            f.write(content)

    def _create_model_card(self, session_id, checkpoint, validation):
        """Create model card metadata."""
        return {
            'model_id': f"{session_id}_{checkpoint}",
            'created': datetime.now().isoformat(),
            'source': session_id,
            'checkpoint': checkpoint,
            'validation_status': 'passed',
            'validation_results': validation,
            'tags': ['ternary-vae', 'v5.5', 'validated']
        }

    def _create_changelog(self, prod_dir, version, source):
        """Create changelog for production version."""
        content = f"""# Changelog - Version {version}

## {version} - {datetime.now().date()}

### Promoted From
- Source: {source}

### Changes
- Initial production release

### Performance
- See metadata.json for benchmarks

### Deployment
- See deployment/ directory for configs
"""
        with open(prod_dir / 'CHANGELOG.md', 'w') as f:
            f.write(content)


# Usage example:
repo = ArtifactRepository(Path('artifacts'))

# After training completes:
session_dir = repo.register_training_session(
    session_id='v5_5_20251123_103epochs',
    checkpoint_dir=Path('sandbox-training/checkpoints/v5_5'),
    config=config,
    metrics={'best_val_loss': 0.3836, 'coverage': 0.9886, 'epoch': 103}
)

# After validation tests pass:
validated_dir = repo.promote_to_validated(
    session_id='v5_5_20251123_103epochs',
    checkpoint_name='epoch_70',
    validation_results={
        'test_results': {...},
        'coverage_report': {...},
        'generalization_report': {...}
    }
)

# After approval for production:
prod_dir = repo.promote_to_production(
    validated_id='v5_5_20251123_103epochs_epoch_70_validated',
    version='1.0.0',
    deployment_config={'docker_image': 'ternary-vae:v1.0.0', ...}
)
```

**Benefits:**
- Clear separation: raw → validated → production
- Traceability: Each stage links to previous
- Metadata tracking: Complete model cards at each stage
- Safe promotion: Validation required before production
- Organized: Easy to find and manage artifacts

---

## Implementation Roadmap (Aggressive)

**Branch Strategy:**
- All work in `refactor/srp-implementation` branch
- No backward compatibility - clean rewrite
- Merge to `main` when complete and validated
- Delete old code, no dual implementation period

---

### Phase 1: Core Extraction & New Structure (Days 1-5)

**Goal:** Create new structure and extract core components

**Tasks:**
1. **Day 1:** Create complete new directory structure
   - `src/training/`, `src/artifacts/`, `src/losses/`, `src/data/`, `src/metrics/`
   - `artifacts/raw/`, `artifacts/validated/`, `artifacts/production/`

2. **Days 2-3:** Implement and test core components
   - `CheckpointManager` class (full implementation)
   - Scheduler classes (`TemperatureScheduler`, `BetaScheduler`, `LearningRateScheduler`)
   - `TrainingMonitor` class
   - Unit tests for all components

3. **Days 4-5:** Refactor trainer
   - Strip trainer to <200 lines (only training loop)
   - Delegate to new components
   - Remove all checkpoint, logging, scheduling logic
   - Integration tests

**Expected Impact:**
- Trainer: 350 → <180 lines (-49%)
- New modules: ~500 lines
- Tests: ~300 lines

**Validation:**
- Run refactored trainer
- Compare training curves (should match exactly)
- Verify checkpoint loading/saving works

---

### Phase 2: Model & Loss Separation (Days 6-10)

**Goal:** Clean model separation - architecture only

**Tasks:**
1. **Days 6-7:** Extract loss computation
   - Create complete `src/losses/` module
   - Implement all loss components (reconstruction, KL, entropy, repulsion)
   - Implement `DualVAELoss` aggregator
   - Unit tests for each loss component

2. **Days 8-9:** Refactor model class
   - Remove all loss computation from model
   - Remove gradient tracking (move to trainer/monitor)
   - Extract encoder/decoder duplicates to base classes
   - Keep only architecture and forward pass
   - Model tests

3. **Day 10:** Validation
   - Run full training
   - Verify loss values identical
   - Check gradients match
   - Performance profiling

**Expected Impact:**
- Model: 632 → <380 lines (-40%)
- New loss modules: ~400 lines
- Tests: ~350 lines

**Validation:**
- Bit-identical loss values
- Same gradient flow
- No performance regression

---

### Phase 3: Utils Reorganization (Days 11-14)

**Goal:** Split and organize utility modules

**Tasks:**
1. **Days 11-12:** Split data.py
   - Create `src/data/` module
   - `generation.py` - Ternary operation generation
   - `validation.py` - Data validation
   - `dataset.py` - Dataset classes
   - `loaders.py` - DataLoader creation
   - Delete old `utils/data.py`
   - Unit tests

2. **Days 13-14:** Split metrics.py
   - Create `src/metrics/` module
   - `coverage.py` - Coverage evaluation
   - `entropy.py` - Entropy computation
   - `reconstruction.py` - Accuracy metrics
   - `tracking.py` - CoverageTracker class
   - Delete old `utils/metrics.py`
   - Delete entire `utils/` directory (no longer needed)
   - Unit tests

**Expected Impact:**
- Utils: 475 → 0 lines (deleted, replaced with focused modules)
- New modules: 8 files (~60-80 lines each)
- Tests: ~250 lines

**Validation:**
- All tests pass
- Data generation identical
- Metrics calculations unchanged

---

### Phase 4: Artifact Management & Integration (Days 15-18)

**Goal:** Professional artifact lifecycle + full integration

**Tasks:**
1. **Days 15-16:** Artifact repository
   - Create `artifacts/` structure (raw/validated/production)
   - Implement `ArtifactRepository` class
   - Add metadata tracking
   - Promotion workflow
   - Migrate existing checkpoints from sandbox-training/

2. **Days 17-18:** Integration testing
   - End-to-end training pipeline test
   - Artifact promotion workflow test
   - Checkpoint compatibility validation
   - Full regression test suite
   - Performance benchmarks

**Expected Impact:**
- New repository module: ~400 lines
- Integration tests: ~400 lines
- Migration complete

**Validation:**
- Train full model end-to-end
- Promote through all artifact stages
- All metrics match baseline

---

### Phase 5: Documentation & Finalization (Days 19-21)

**Goal:** Complete documentation and final cleanup

**Tasks:**
1. **Day 19:** Module documentation
   - Document all new modules
   - Architecture decision records (ADRs)
   - Module responsibility guide
   - API documentation

2. **Day 20:** Code quality
   - Add type hints throughout
   - Run black formatter
   - Run pylint/flake8
   - Fix all warnings
   - Optimize imports

3. **Day 21:** Final review
   - Code review
   - Test coverage check (target: >85%)
   - Performance validation
   - Update README
   - Prepare merge to main

**Expected Impact:**
- Documentation: ~800 lines
- Type hints added throughout
- Code quality score: A+

**Validation:**
- All tests pass (>85% coverage)
- No performance regression
- Documentation complete

---

## Branch Workflow

### Development Strategy

**Approach:** Aggressive clean rewrite in dedicated branch

1. **All work in `refactor/srp-implementation` branch**
   - No backward compatibility layers
   - Clean deletion of old code as we go
   - No deprecation warnings needed
   - No dual implementation periods

2. **Continuous validation**
   - Run tests after each phase
   - Compare results to baseline (main branch)
   - Track metrics in separate document
   - Performance benchmarks against main

3. **Merge when complete**
   - All tests passing (>85% coverage)
   - Performance validated (no regression)
   - Documentation complete
   - Full code review approved
   - Squash merge or detailed commit history (TBD)

### Risk Mitigation

**Checkpoint Compatibility:**
- Training outputs go to new `artifacts/` structure
- Old checkpoints in `sandbox-training/` remain untouched
- Migration script to convert old → new format
- Validation that old checkpoints load correctly

**Configuration:**
- Config files remain compatible
- Add schema validation
- Any breaking changes documented
- Migration guide for users

**Testing:**
- Comprehensive regression suite
- Compare training runs: main vs refactor branch
- Bit-identical results for deterministic operations
- Performance profiling at each phase

**Rollback Plan:**
- Keep main branch stable
- Tag before merge: `v5.5-pre-refactor`
- Can revert merge if issues found
- Separate branch for any critical fixes

---

## Success Metrics

### Code Quality

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Average function length | 35 lines | <25 lines | Static analysis |
| Average class length | 400 lines | <250 lines | Static analysis |
| Cyclomatic complexity | 15 | <10 | Code metrics |
| Test coverage | 60% | >80% | pytest-cov |
| Module coupling | High | Low | Dependency analysis |

### Maintainability

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Time to add new loss term | 2 hours | 30 min | 75% faster |
| Time to add new scheduler | 3 hours | 45 min | 75% faster |
| Time to change checkpoint format | 4 hours | 1 hour | 75% faster |
| Time to add new metric | 2 hours | 30 min | 75% faster |

### Development Velocity

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| New feature development time | Baseline | -30% | Faster |
| Bug fix time | Baseline | -40% | Faster |
| Test writing time | Baseline | -50% | Faster |
| Code review time | Baseline | -35% | Faster |

---

## Conclusion

This refactoring plan addresses critical SRP violations in the Ternary VAE codebase through an **aggressive, clean rewrite** in a dedicated branch. No backward compatibility cruft - just clean, well-structured code.

**Key Benefits:**

1. **Improved Testability:** Each component tested independently (>85% coverage)
2. **Better Maintainability:** Clear module boundaries and responsibilities
3. **Enhanced Extensibility:** Easy to add features without modifying existing code
4. **Clearer Architecture:** Separation of concerns throughout
5. **Professional Artifact Management:** Clear lifecycle (raw → validated → production)
6. **No Technical Debt:** Clean rewrite eliminates all compatibility patches

**Aggressive Approach Advantages:**

- **Faster:** 3 weeks vs 6 weeks (50% time savings)
- **Cleaner:** No dual implementations or deprecation warnings
- **Simpler:** Straightforward path - rewrite and validate
- **Better quality:** No compromises for backward compatibility
- **Branch safety:** Main branch remains stable during refactoring

**Next Steps:**

1. Review and approve this aggressive plan
2. Begin Phase 1 in `refactor/srp-implementation` branch
3. Track progress with daily commits
4. Validate after each phase
5. Merge to main when complete (3 weeks timeline)

**Development Workflow:**

```
main branch (stable) ────────────────────────── merge (after validation)
                 \                             /
                  refactor/srp-implementation
                  (aggressive changes, 21 days)
```

---

**Document Version:** 2.0 (Aggressive)
**Last Updated:** 2025-11-23
**Branch:** `refactor/srp-implementation`
**Status:** Ready for Aggressive Implementation
**Estimated Timeline:** 3 weeks (21 days)
**Estimated Effort:** 80-100 hours
**Approach:** Clean rewrite, no backward compatibility

---

*For questions or clarifications, refer to the inline code examples or consult the existing codebase documentation.*
