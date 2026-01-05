# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DENV-4 Clade Classification using TrainableCodonEncoder.

This script trains a clade classifier for 270 DENV-4 genomes using:
1. TrainableCodonEncoder for codon-level embeddings
2. Mean pooling for genome-level representation
3. Classification head for 5 clades

Architecture:
    Genome → Codons → TrainableCodonEncoder → Mean Pool → MLP → 5 classes

Challenge: Severe class imbalance (Clade_E=211, others=59 combined)
Solution: Stratified K-Fold, class weights, focal loss

Usage:
    python train_clade_classifier.py --epochs 100 --folds 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# Project imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[5]  # ternary-vaes root
sys.path.insert(0, str(PROJECT_ROOT))

from src.encoders.trainable_codon_encoder import TrainableCodonEncoder
from src.biology.codons import CODON_TO_INDEX, codon_index_to_triplet

# Paths (reuse PROJECT_ROOT from above)
ROJAS_DIR = PROJECT_ROOT / "deliverables" / "partners" / "alejandra_rojas"
ML_READY_DIR = ROJAS_DIR / "results" / "ml_ready"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_denv4_data() -> tuple[list[str], list[str], list[str]]:
    """Load DENV-4 sequences and clade assignments.

    Returns:
        accessions: List of accession IDs
        sequences: List of genome sequences
        clades: List of clade labels
    """
    # Load metadata
    with open(ML_READY_DIR / "denv4_genome_metadata.json") as f:
        metadata = json.load(f)

    # Load sequences
    with open(ML_READY_DIR / "denv4_genome_sequences.json") as f:
        seq_data = json.load(f)

    accessions = []
    sequences = []
    clades = []

    for acc, meta in metadata["data"].items():
        if acc in seq_data["data"]:
            accessions.append(acc)
            sequences.append(seq_data["data"][acc])
            clades.append(meta["clade"])

    return accessions, sequences, clades


def sequence_to_codon_indices(sequence: str) -> list[int]:
    """Convert nucleotide sequence to codon indices.

    Args:
        sequence: Nucleotide sequence (ACGT)

    Returns:
        List of codon indices (0-63)
    """
    sequence = sequence.upper().replace('U', 'T')
    indices = []

    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3]
        if codon in CODON_TO_INDEX:
            indices.append(CODON_TO_INDEX[codon])

    return indices


class GenomeEncoder(nn.Module):
    """Encode full genome using TrainableCodonEncoder + pooling."""

    def __init__(
        self,
        codon_encoder: TrainableCodonEncoder,
        pool_type: str = "mean",
    ):
        super().__init__()
        self.codon_encoder = codon_encoder
        self.pool_type = pool_type

    def forward(self, codon_indices: torch.Tensor) -> torch.Tensor:
        """Encode genome from codon indices.

        Args:
            codon_indices: (batch, seq_len) codon indices

        Returns:
            (batch, latent_dim) genome embeddings
        """
        batch_size, seq_len = codon_indices.shape

        # Flatten for encoding
        flat_indices = codon_indices.view(-1)

        # Encode all codons
        codon_embeddings = self.codon_encoder(flat_indices)  # (batch*seq, latent)

        # Reshape back
        latent_dim = codon_embeddings.shape[-1]
        codon_embeddings = codon_embeddings.view(batch_size, seq_len, latent_dim)

        # Pool
        if self.pool_type == "mean":
            genome_emb = codon_embeddings.mean(dim=1)
        elif self.pool_type == "max":
            genome_emb = codon_embeddings.max(dim=1)[0]
        else:
            genome_emb = codon_embeddings.mean(dim=1)

        return genome_emb


class CladeClassifier(nn.Module):
    """Full classification model: Genome → Clade."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        n_classes: int = 5,
        dropout: float = 0.3,
        freeze_codon_encoder: bool = True,
    ):
        super().__init__()

        # Codon encoder (can be frozen)
        self.codon_encoder = TrainableCodonEncoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

        if freeze_codon_encoder:
            for param in self.codon_encoder.parameters():
                param.requires_grad = False

        # Genome encoder (pooling)
        self.genome_encoder = GenomeEncoder(self.codon_encoder, pool_type="mean")

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

        self.n_classes = n_classes
        self.latent_dim = latent_dim

    def forward(self, codon_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            codon_indices: (batch, seq_len) codon indices

        Returns:
            (batch, n_classes) logits
        """
        genome_emb = self.genome_encoder(codon_indices)
        logits = self.classifier(genome_emb)
        return logits

    def load_pretrained_codon_encoder(self, checkpoint_path: str):
        """Load pretrained codon encoder weights."""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt

        # Filter to codon encoder keys
        codon_keys = {k: v for k, v in state.items() if k.startswith("codon_encoder.") or not k.startswith("classifier.")}
        self.codon_encoder.load_state_dict(codon_keys, strict=False)
        print(f"Loaded pretrained codon encoder from {checkpoint_path}")


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def prepare_batch(
    sequences: list[str],
    max_len: int = 3500,
    device: str = "cpu",
) -> torch.Tensor:
    """Convert sequences to padded codon index tensor.

    Args:
        sequences: List of nucleotide sequences
        max_len: Maximum number of codons
        device: Device to place tensor on

    Returns:
        (batch, max_len) tensor of codon indices
    """
    batch_indices = []

    for seq in sequences:
        indices = sequence_to_codon_indices(seq)

        # Truncate or pad
        if len(indices) > max_len:
            indices = indices[:max_len]
        elif len(indices) < max_len:
            indices = indices + [0] * (max_len - len(indices))

        batch_indices.append(indices)

    return torch.tensor(batch_indices, dtype=torch.long, device=device)


def train_fold(
    train_seqs: list[str],
    train_labels: np.ndarray,
    val_seqs: list[str],
    val_labels: np.ndarray,
    class_weights: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    pretrained_path: Optional[str] = None,
) -> dict:
    """Train one fold.

    Returns:
        Dictionary with training history and best metrics
    """
    # Create model
    model = CladeClassifier(
        latent_dim=16,
        hidden_dim=64,
        n_classes=5,
        dropout=0.3,
        freeze_codon_encoder=True,  # Start with frozen encoder
    ).to(device)

    # Load pretrained if available
    if pretrained_path and Path(pretrained_path).exists():
        model.load_pretrained_codon_encoder(pretrained_path)

    # Loss and optimizer
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Prepare data
    train_X = prepare_batch(train_seqs, device=device)
    train_y = torch.tensor(train_labels, dtype=torch.long, device=device)
    val_X = prepare_batch(val_seqs, device=device)
    val_y = torch.tensor(val_labels, dtype=torch.long, device=device)

    # Training loop
    best_val_acc = 0.0
    best_state = None
    history = {"train_loss": [], "val_acc": [], "val_balanced_acc": []}

    for epoch in range(epochs):
        model.train()

        # Forward
        logits = model(train_X)
        loss = criterion(logits, train_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X)
            val_preds = val_logits.argmax(dim=-1).cpu().numpy()
            val_labels_np = val_y.cpu().numpy()

            acc = accuracy_score(val_labels_np, val_preds)
            bal_acc = balanced_accuracy_score(val_labels_np, val_preds)

        history["train_loss"].append(loss.item())
        history["val_acc"].append(acc)
        history["val_balanced_acc"].append(bal_acc)

        if bal_acc > best_val_acc:
            best_val_acc = bal_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}, acc={acc:.3f}, bal_acc={bal_acc:.3f}")

    # Final evaluation with best model
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(val_X)
        val_preds = val_logits.argmax(dim=-1).cpu().numpy()
        val_labels_np = val_y.cpu().numpy()

    return {
        "history": history,
        "best_val_balanced_acc": best_val_acc,
        "final_preds": val_preds.tolist(),
        "final_labels": val_labels_np.tolist(),
        "best_state": best_state,
    }


def main():
    parser = argparse.ArgumentParser(description="Train DENV-4 clade classifier")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained codon encoder checkpoint")
    args = parser.parse_args()

    print("=" * 70)
    print("DENV-4 CLADE CLASSIFICATION WITH TRAINABLE CODON ENCODER")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Folds: {args.folds}")
    print()

    # Load data
    print("Loading DENV-4 data...")
    accessions, sequences, clades = load_denv4_data()
    print(f"Loaded {len(sequences)} sequences")

    # Encode labels
    clade_to_idx = {c: i for i, c in enumerate(sorted(set(clades)))}
    idx_to_clade = {i: c for c, i in clade_to_idx.items()}
    labels = np.array([clade_to_idx[c] for c in clades])

    print(f"Clade mapping: {clade_to_idx}")
    print(f"Class distribution: {Counter(labels)}")

    # Compute class weights (inverse frequency)
    class_counts = np.bincount(labels)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    print(f"Class weights: {class_weights.tolist()}")
    print()

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    all_results = []
    all_preds = np.zeros(len(labels), dtype=int)

    for fold, (train_idx, val_idx) in enumerate(skf.split(sequences, labels)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1}/{args.folds}")
        print(f"{'='*70}")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

        train_seqs = [sequences[i] for i in train_idx]
        train_labels = labels[train_idx]
        val_seqs = [sequences[i] for i in val_idx]
        val_labels = labels[val_idx]

        result = train_fold(
            train_seqs=train_seqs,
            train_labels=train_labels,
            val_seqs=val_seqs,
            val_labels=val_labels,
            class_weights=class_weights,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            pretrained_path=args.pretrained,
        )

        all_results.append(result)
        all_preds[val_idx] = result["final_preds"]

        print(f"\nFold {fold + 1} Best Balanced Accuracy: {result['best_val_balanced_acc']:.4f}")

    # Overall metrics
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    overall_acc = accuracy_score(labels, all_preds)
    overall_bal_acc = balanced_accuracy_score(labels, all_preds)

    print(f"\nOverall Accuracy: {overall_acc:.4f}")
    print(f"Overall Balanced Accuracy: {overall_bal_acc:.4f}")

    print("\nClassification Report:")
    target_names = [idx_to_clade[i] for i in range(len(clade_to_idx))]
    print(classification_report(labels, all_preds, target_names=target_names))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, all_preds)
    print(cm)

    # Save results
    results_dict = {
        "_metadata": {
            "analysis_type": "clade_classification",
            "description": "DENV-4 clade classification using TrainableCodonEncoder",
            "created": datetime.now(timezone.utc).isoformat(),
            "parameters": {
                "epochs": args.epochs,
                "folds": args.folds,
                "lr": args.lr,
                "device": args.device,
            },
        },
        "summary": {
            "overall_accuracy": overall_acc,
            "overall_balanced_accuracy": overall_bal_acc,
            "n_samples": len(labels),
            "n_classes": len(clade_to_idx),
        },
        "clade_mapping": clade_to_idx,
        "class_weights": class_weights.tolist(),
        "fold_results": [
            {
                "fold": i + 1,
                "best_balanced_acc": r["best_val_balanced_acc"],
            }
            for i, r in enumerate(all_results)
        ],
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            labels, all_preds, target_names=target_names, output_dict=True
        ),
        "predictions": {
            "labels": labels.tolist(),
            "predictions": all_preds.tolist(),
        },
    }

    results_path = RESULTS_DIR / "clade_classification_results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Per-fold summary
    mean_bal_acc = np.mean([r["best_val_balanced_acc"] for r in all_results])
    std_bal_acc = np.std([r["best_val_balanced_acc"] for r in all_results])
    print(f"\nMean Balanced Accuracy: {mean_bal_acc:.4f} ± {std_bal_acc:.4f}")

    return results_dict


if __name__ == "__main__":
    main()
