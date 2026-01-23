#!/usr/bin/env python3
"""Train Codon Encoder with Grokking Detection and Soft Topologies.

This script implements an improved training regimen for the TrainableCodonEncoder:
1.  **Soft Topologies**: Reduces rigid structural enforcement (p-adic/property weights)
    to allow the model to "grok" the natural structure.
2.  **Grokking Detection**: Monitors for phase transitions in validation performance (S669 DDG).
3.  **Extended Training**: Runs for longer epochs to allow for emergence.

Usage:
    python research/codon-encoder/training/train_codon_grokking.py --epochs 5000 --device cuda
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.encoders.trainable_codon_encoder import TrainableCodonEncoder
from src.encoders.codon_encoder import AA_PROPERTIES
from src.geometry import poincare_distance

# =============================================================================
# Grokking Detector
# =============================================================================

class GrokkingDetector:
    """Detect grokking and phase transition phenomena during training."""

    def __init__(
        self,
        window_size: int = 20,
        plateau_threshold: float = 0.001,
        plateau_patience: int = 50,
        accuracy_jump_threshold: float = 0.05,
    ):
        self.window_size = window_size
        self.plateau_threshold = plateau_threshold
        self.plateau_patience = plateau_patience
        self.accuracy_jump_threshold = accuracy_jump_threshold

        self.loss_history: List[float] = []
        self.metric_history: List[float] = []
        
        self.in_plateau = False
        self.plateau_start_epoch = None
        self.potential_grokking_events = []

    def update(self, epoch: int, loss: float, metric: float) -> Dict:
        self.loss_history.append(loss)
        self.metric_history.append(metric)

        analysis = {
            "in_plateau": False,
            "plateau_duration": 0,
            "potential_grokking": False,
        }

        if len(self.loss_history) < self.window_size:
            return analysis

        # Plateau detection
        recent_losses = self.loss_history[-self.plateau_patience:]
        if len(recent_losses) >= self.plateau_patience:
            loss_change = max(recent_losses) - min(recent_losses)

            if loss_change < self.plateau_threshold:
                if not self.in_plateau:
                    self.in_plateau = True
                    self.plateau_start_epoch = epoch - self.plateau_patience + 1

                analysis["in_plateau"] = True
                analysis["plateau_duration"] = epoch - self.plateau_start_epoch + 1
            else:
                if self.in_plateau:
                    # Exiting plateau - potential grokking?
                    # Check for metric jump
                    if len(self.metric_history) >= self.window_size:
                        pre_plateau_metric = np.mean(self.metric_history[-self.window_size:-self.plateau_patience])
                        current_metric = self.metric_history[-1]
                        metric_jump = current_metric - pre_plateau_metric

                        if metric_jump > self.accuracy_jump_threshold:
                            analysis["potential_grokking"] = True
                            self.potential_grokking_events.append({
                                "epoch": epoch,
                                "plateau_duration": analysis["plateau_duration"],
                                "metric_jump": metric_jump,
                                "loss_before": np.mean(recent_losses),
                                "loss_after": loss,
                            })

                self.in_plateau = False
                self.plateau_start_epoch = None

        return analysis

    def get_summary(self) -> Dict:
        return {
            "total_grokking_events": len(self.potential_grokking_events),
            "grokking_events": self.potential_grokking_events,
        }

# =============================================================================
# Data & Evaluation
# =============================================================================

def load_s669(filepath: Path) -> list[dict]:
    """Load S669 dataset."""
    mutations = []
    if not filepath.exists():
        print(f"WARNING: S669 dataset not found at {filepath}")
        return []
        
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 6:
            try:
                mutations.append({
                    'wild_type': parts[3].upper(),
                    'mutant': parts[4].upper(),
                    'ddg_exp': float(parts[5])
                })
            except (ValueError, IndexError):
                continue
    return mutations

def evaluate_ddg_fast(
    encoder: TrainableCodonEncoder,
    mutations: list[dict],
    device: str = 'cpu',
) -> float:
    """Fast DDG evaluation returning just the Spearman correlation."""
    if not mutations:
        return 0.0
        
    encoder.eval()
    aa_embeddings = encoder.get_all_amino_acid_embeddings()

    X = []
    y = []

    for mut in mutations:
        wt = mut['wild_type']
        mt = mut['mutant']
        if wt not in aa_embeddings or mt not in aa_embeddings:
            continue

        wt_emb = aa_embeddings[wt]
        mt_emb = aa_embeddings[mt]

        # Features
        hyp_dist = poincare_distance(wt_emb.unsqueeze(0), mt_emb.unsqueeze(0), c=encoder.curvature).item()
        
        origin = torch.zeros(1, encoder.latent_dim, device=wt_emb.device)
        wt_norm = poincare_distance(wt_emb.unsqueeze(0), origin, c=encoder.curvature).item()
        mt_norm = poincare_distance(mt_emb.unsqueeze(0), origin, c=encoder.curvature).item()
        delta_norm = mt_norm - wt_norm

        wt_props = AA_PROPERTIES.get(wt, (0, 0, 0, 0))
        mt_props = AA_PROPERTIES.get(mt, (0, 0, 0, 0))
        delta_hydro = mt_props[0] - wt_props[0]
        delta_charge = abs(mt_props[1] - wt_props[1])

        X.append([hyp_dist, delta_norm, delta_hydro, delta_charge])
        y.append(mut['ddg_exp'])

    if not X:
        return 0.0
        
    X = np.array(X)
    y = np.array(y)

    # Simple Ridge regression on full set (faster than LOO for monitoring)
    # We care about the trend, so training fit is an okay proxy for monitoring,
    # but to be strict we should use a hold-out or CV.
    # Let's use 5-fold CV for a balance of speed and honesty.
    try:
        model = Ridge(alpha=1.0)
        y_pred = cross_val_predict(model, X, y, cv=5)
        spearman, _ = spearmanr(y_pred, y)
        return float(spearman)
    except Exception:
        return 0.0

# =============================================================================
# Main Training
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Grokking Training for Codon Encoder")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--output_dir", type=str, default="research/codon-encoder/training/results_grokking")
    args = parser.parse_args()

    # Setup
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load S669 for validation
    data_path = PROJECT_ROOT / "deliverables/partners/jose_colbes/reproducibility/data/s669.csv"
    mutations = load_s669(data_path)
    print(f"Loaded {len(mutations)} validation mutations")

    # Model
    encoder = TrainableCodonEncoder(
        latent_dim=16,
        hidden_dim=64,
        curvature=1.0,
        max_radius=0.9,
        dropout=0.1
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Scheduler: Cosine with Warm Restarts to encourage escaping local minima
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=500, T_mult=2, eta_min=1e-5
    )

    # Grokking Detector
    detector = GrokkingDetector()

    # Soft Topology Weights
    # We reduce the rigid p-adic and property weights to allow "softness"
    weights = {
        'radial_weight': 1.0,
        'cohesion_weight': 1.0,
        'separation_weight': 0.5,
        'padic_weight': 0.1,
        'property_weight': 0.1,
    }
    
    print("\nStarting Training with Soft Topologies:")
    print(json.dumps(weights, indent=2))
    print(f"Target Epochs: {args.epochs}")

    best_spearman = -1.0
    start_time = time.time()

    for epoch in range(args.epochs):
        encoder.train()
        optimizer.zero_grad()

        losses = encoder.compute_total_loss(**weights)
        
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Monitoring
        if (epoch + 1) % 50 == 0:
            val_spearman = evaluate_ddg_fast(encoder, mutations, device)
            
            # Update detector
            analysis = detector.update(epoch, losses['total'].item(), val_spearman)
            
            # Log
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:4d} | Loss: {losses['total'].item():.4f} | "
                  f"Val Spearman: {val_spearman:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.0f}s")
            
            if analysis.get("potential_grokking"):
                print(f"  >>> POTENTIAL GROKKING DETECTED! Jump: {analysis['potential_grokking_events'][-1]['metric_jump']:.3f}")
            
            if val_spearman > best_spearman:
                best_spearman = val_spearman
                torch.save({
                    'epoch': epoch,
                    'model_state': encoder.state_dict(),
                    'val_spearman': val_spearman,
                    'config': weights
                }, output_dir / "best_grokking_model.pt")
                print(f"  [New Best Model Saved: {best_spearman:.4f}]")

    print("\nTraining Complete")
    print(f"Best Validation Spearman: {best_spearman:.4f}")
    
    # Save final summary
    summary = {
        'best_spearman': best_spearman,
        'grokking_events': detector.get_summary(),
        'weights': weights,
        'epochs': args.epochs
    }
    with open(output_dir / "grokking_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
