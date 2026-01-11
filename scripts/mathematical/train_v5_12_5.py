#!/usr/bin/env python3
"""TernaryVAE v5.12.5 Mathematical Foundation Training Pipeline.

Unified training pipeline for mathematical foundation combining:
- V5.12.4 improved components (SiLU, LayerNorm, Dropout)
- Homeostatic_rich proven balance (coverage + hierarchy + richness)
- Enhanced mathematical precision and validation
- Hyperbolic geometry as default

Usage:
    python scripts/mathematical/train_v5_12_5.py \
        --config configs/mathematical/v5_12_5_foundation.yaml \
        --profile mathematical_foundation
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import from mathematical framework
try:
    from src_math.core import TERNARY, padic_distance
    from src_math.geometry import poincare_distance
    from src_math.models import TernaryVAEV5_11_PartialFreeze, HomeostasisController
    from src_math.losses import RichHierarchyLoss
    from src_math.training import Trainer, TrainingMonitor
except ImportError:
    # Fallback to original structure
    from src.core import TERNARY
    from src.geometry import poincare_distance
    from src.models import TernaryVAEV5_11_PartialFreeze
    from src.models.homeostasis import HomeostasisController, compute_Q
    from src.losses import RichHierarchyLoss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MathematicalFoundationTrainer:
    """Unified trainer for TernaryVAE v5.12.5 mathematical foundation."""

    MATHEMATICAL_PROFILES = {
        'mathematical_foundation': {
            'description': 'Balanced mathematical foundation',
            'loss_weights': {
                'coverage': 1.0,
                'hierarchy': 5.0,
                'richness': 2.5,
                'separation': 3.0
            },
            'targets': {
                'coverage': 1.0,
                'hierarchy_B': -0.8321,
                'richness': 0.006,
                'Q_enhanced': 2.0
            }
        },
        'coverage_focused': {
            'description': 'Pure reconstruction accuracy',
            'loss_weights': {
                'coverage': 10.0,
                'hierarchy': 0.1,
                'richness': 0.1,
                'separation': 0.1
            }
        },
        'hierarchy_focused': {
            'description': 'P-adic ordering optimization',
            'loss_weights': {
                'coverage': 1.0,
                'hierarchy': 8.0,
                'richness': 0.5,
                'separation': 1.0
            }
        },
        'richness_focused': {
            'description': 'Geometric diversity preservation',
            'loss_weights': {
                'coverage': 1.0,
                'hierarchy': 3.0,
                'richness': 5.0,
                'separation': 2.0
            }
        }
    }

    def __init__(self, config: Dict[str, Any], profile: str = 'mathematical_foundation'):
        self.config = config
        self.profile = self.MATHEMATICAL_PROFILES[profile]
        self.device = torch.device(f"cuda:{config['device']['cuda_device']}" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing v5.12.5 Mathematical Foundation with profile: {profile}")
        logger.info(f"Profile: {self.profile['description']}")
        logger.info(f"Device: {self.device}")

        self.setup_mathematical_framework()

    def setup_mathematical_framework(self):
        """Initialize mathematical components."""

        # Model with v5.12.5 enhancements
        model_config = self.config['model']
        self.model = TernaryVAEV5_11_PartialFreeze(
            latent_dim=model_config['latent_dim'],
            hidden_dim=model_config['hidden_dim'],
            max_radius=model_config['max_radius'],
            curvature=model_config['curvature'],
            use_controller=model_config['use_controller'],
            use_dual_projection=model_config['use_dual_projection'],
            encoder_type=model_config.get('encoder_type', 'improved'),
            decoder_type=model_config.get('decoder_type', 'improved'),
        ).to(self.device)

        # Enhanced mathematical loss
        loss_config = self.config['loss']['rich_hierarchy']
        self.loss_fn = RichHierarchyLoss(
            hierarchy_weight=self.profile['loss_weights']['hierarchy'],
            coverage_weight=self.profile['loss_weights']['coverage'],
            richness_weight=self.profile['loss_weights']['richness'],
            separation_weight=self.profile['loss_weights']['separation'],
        ).to(self.device)

        # Enhanced homeostasis controller
        if self.config['homeostasis']['enabled']:
            self.homeostasis = HomeostasisController(
                coverage_freeze_threshold=self.config['homeostasis']['coverage_freeze_threshold'],
                enable_annealing=self.config['homeostasis']['enable_annealing'],
                annealing_step=self.config['homeostasis']['annealing_step'],
            )

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=self.config['training']['weight_decay']
        )

        # Training monitor
        self.monitor = TrainingMonitor()

    def load_frozen_checkpoint(self):
        """Load v5.5 frozen checkpoint for coverage preservation."""
        frozen_config = self.config.get('frozen_checkpoint')
        if frozen_config:
            checkpoint_path = Path(frozen_config['path'])
            if checkpoint_path.exists():
                logger.info(f"Loading frozen checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_frozen_components(checkpoint, strict=False)
            else:
                logger.warning(f"Frozen checkpoint not found: {checkpoint_path}")

    def compute_comprehensive_metrics(self, all_ops: torch.Tensor, device: torch.device) -> Dict[str, float]:
        """Compute comprehensive mathematical metrics."""
        self.model.eval()
        batch_size = 4096

        all_radii_A = []
        all_radii_B = []
        all_correct = []
        all_valuations = []

        with torch.no_grad():
            for i in range(0, len(all_ops), batch_size):
                batch = all_ops[i:i + batch_size].to(device)
                batch_indices = TERNARY.from_ternary(batch)
                batch_valuations = TERNARY.valuation(batch_indices)

                outputs = self.model(batch, compute_control=False)

                # Use hyperbolic distance for radii (V5.12.2+ correct approach)
                origin_A = torch.zeros_like(outputs['z_A_hyp'])
                origin_B = torch.zeros_like(outputs['z_B_hyp'])
                radii_A = poincare_distance(outputs['z_A_hyp'], origin_A, c=1.0)
                radii_B = poincare_distance(outputs['z_B_hyp'], origin_B, c=1.0)

                # Compute reconstruction accuracy
                predicted = torch.argmax(outputs['reconstructed'], dim=-1)
                correct = (predicted == batch).float().mean(dim=1)

                all_radii_A.extend(radii_A.cpu().numpy())
                all_radii_B.extend(radii_B.cpu().numpy())
                all_correct.extend(correct.cpu().numpy())
                all_valuations.extend(batch_valuations.cpu().numpy())

        # Compute metrics
        coverage = np.mean([c > 0.999 for c in all_correct])
        hierarchy_A = spearmanr(all_valuations, all_radii_A)[0]
        hierarchy_B = spearmanr(all_valuations, all_radii_B)[0]

        # Compute richness (within-level variance)
        richness = 0.0
        for v in range(10):
            v_mask = np.array(all_valuations) == v
            if v_mask.sum() > 1:
                richness += np.var(np.array(all_radii_B)[v_mask])
        richness /= 10

        # Enhanced Q metric
        dist_corr = np.corrcoef(all_radii_A, all_radii_B)[0, 1]
        Q = abs(dist_corr) + 1.5 * abs(hierarchy_B) + 0.5 * np.log(richness + 1e-8)

        return {
            'coverage': coverage,
            'hierarchy_A': hierarchy_A,
            'hierarchy_B': hierarchy_B,
            'richness': richness,
            'Q_enhanced': Q,
            'r_v0': np.mean(np.array(all_radii_B)[np.array(all_valuations) == 0]),
            'r_v9': np.mean(np.array(all_radii_B)[np.array(all_valuations) >= 8]),
        }

    def validate_mathematical_foundation(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Validate mathematical foundation criteria."""
        targets = self.profile.get('targets', self.config['targets']['tier_1_mathematical_foundation'])

        validation_results = {}
        validation_results['coverage_precision'] = metrics['coverage'] >= targets.get('coverage', 0.9999)
        validation_results['hierarchy_ceiling'] = metrics['hierarchy_B'] <= targets.get('hierarchy_B', -0.82)
        validation_results['richness_preservation'] = metrics['richness'] >= targets.get('richness', 0.002)
        validation_results['Q_enhanced'] = metrics['Q_enhanced'] >= targets.get('Q_enhanced', 1.8)

        # Mathematical stability checks
        validation_results['numerical_stability'] = not any(
            np.isnan(v) or np.isinf(v) for v in metrics.values() if isinstance(v, (int, float))
        )

        # P-adic structure preservation
        validation_results['p_adic_structure'] = abs(metrics['hierarchy_B']) > 0.1  # Non-trivial hierarchy

        # Overall foundation readiness
        validation_results['mathematical_foundation_ready'] = all(validation_results.values())

        return validation_results

    def train(self):
        """Execute mathematical foundation training."""
        logger.info("Starting TernaryVAE v5.12.5 Mathematical Foundation Training")

        # Load frozen components
        self.load_frozen_checkpoint()

        # Generate training data
        logger.info("Generating ternary operations dataset")
        all_ops = torch.tensor(TERNARY.generate_all_operations(), dtype=torch.long)
        dataset = TensorDataset(all_ops)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            pin_memory=True
        )

        epochs = self.config['training']['epochs']
        best_Q = float('-inf')
        best_metrics = {}

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []

            for batch_idx, (batch,) in enumerate(dataloader):
                batch = batch.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch, compute_control=True)
                loss_dict = self.loss_fn(outputs, batch, TERNARY)

                total_loss = loss_dict['total_loss']
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])

                self.optimizer.step()
                epoch_losses.append(total_loss.item())

                if batch_idx % 10 == 0:
                    logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.4f}")

            # Comprehensive evaluation
            if epoch % 10 == 0 or epoch == epochs - 1:
                metrics = self.compute_comprehensive_metrics(all_ops, self.device)
                validation = self.validate_mathematical_foundation(metrics)

                logger.info(f"Epoch {epoch:3d} | "
                           f"Loss: {np.mean(epoch_losses):.4f} | "
                           f"Coverage: {metrics['coverage']:.4f} | "
                           f"Hierarchy_B: {metrics['hierarchy_B']:.4f} | "
                           f"Richness: {metrics['richness']:.6f} | "
                           f"Q: {metrics['Q_enhanced']:.4f}")

                # Save best checkpoint
                if metrics['Q_enhanced'] > best_Q:
                    best_Q = metrics['Q_enhanced']
                    best_metrics = metrics.copy()

                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'comprehensive_metrics': metrics,
                        'validation_results': validation,
                        'config': self.config,
                        'profile': self.profile,
                        'mathematical_foundation_version': '5.12.5'
                    }

                    checkpoint_dir = Path('checkpoints/v5_12_5')
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)

                    torch.save(checkpoint, checkpoint_dir / 'best_Q.pt')
                    logger.info(f"Saved best checkpoint (Q={best_Q:.4f})")

                    if validation['mathematical_foundation_ready']:
                        torch.save(checkpoint, checkpoint_dir / 'mathematical_foundation_ready.pt')
                        logger.info("✓ Mathematical foundation criteria met!")

                # Homeostatic control
                if hasattr(self, 'homeostasis'):
                    should_freeze = self.homeostasis.should_freeze(metrics)
                    if should_freeze:
                        logger.info("Homeostatic controller triggered freeze")

        logger.info("Training completed!")
        logger.info(f"Best Q metric: {best_Q:.4f}")
        logger.info(f"Final metrics: {best_metrics}")

        return best_metrics

def main():
    parser = argparse.ArgumentParser(description="TernaryVAE v5.12.5 Mathematical Foundation Training")
    parser.add_argument('--config', type=str, default='configs/mathematical/v5_12_5_foundation.yaml',
                       help='Configuration file path')
    parser.add_argument('--profile', type=str, default='mathematical_foundation',
                       choices=['mathematical_foundation', 'coverage_focused', 'hierarchy_focused', 'richness_focused'],
                       help='Training profile to use')
    parser.add_argument('--checkpoint', type=str, help='Resume from checkpoint')
    parser.add_argument('--test-mode', action='store_true', help='Run quick test (5 epochs)')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.test_mode:
        config['training']['epochs'] = 5
        config['training']['batch_size'] = 256
        logger.info("Running in test mode (5 epochs)")

    # Initialize and run trainer
    trainer = MathematicalFoundationTrainer(config, args.profile)

    if args.checkpoint:
        logger.info(f"Resuming from checkpoint: {args.checkpoint}")
        # Checkpoint loading logic here

    # Execute training
    final_metrics = trainer.train()

    # Final validation
    logger.info("=== MATHEMATICAL FOUNDATION TRAINING COMPLETE ===")
    logger.info(f"Profile: {args.profile}")
    logger.info(f"Final Coverage: {final_metrics.get('coverage', 0):.4f}")
    logger.info(f"Final Hierarchy_B: {final_metrics.get('hierarchy_B', 0):.4f}")
    logger.info(f"Final Richness: {final_metrics.get('richness', 0):.6f}")
    logger.info(f"Final Q_enhanced: {final_metrics.get('Q_enhanced', 0):.4f}")

if __name__ == "__main__":
    main()