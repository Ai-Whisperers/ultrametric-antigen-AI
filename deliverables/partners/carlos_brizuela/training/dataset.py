#!/usr/bin/env python3
"""PyTorch Dataset for AMP Activity Prediction.

This module provides a PyTorch Dataset for training the PeptideVAE model
on antimicrobial peptide (AMP) activity data.

Features:
- Stratified cross-validation splits (pathogen-balanced)
- Automatic feature computation from sequences
- Support for train/val/test splits
- Property normalization with fit/transform pattern

Usage:
    from deliverables.partners.carlos_brizuela.training.dataset import (
        AMPDataset, create_stratified_dataloaders
    )

    train_loader, val_loader = create_stratified_dataloaders(
        fold_idx=0, n_folds=5, batch_size=32
    )
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Add paths
_script_dir = Path(__file__).parent
_deliverables_dir = _script_dir.parent.parent.parent
sys.path.insert(0, str(_deliverables_dir))
sys.path.insert(0, str(_script_dir.parent))

# Import DRAMP loader
from scripts.dramp_activity_loader import DRAMPLoader


# =============================================================================
# Pathogen Label Mapping
# =============================================================================

PATHOGEN_TO_LABEL = {
    'escherichia': 0,
    'pseudomonas': 1,
    'staphylococcus': 2,
    'acinetobacter': 3,
}

LABEL_TO_PATHOGEN = {v: k for k, v in PATHOGEN_TO_LABEL.items()}


# =============================================================================
# Dataset
# =============================================================================


@dataclass
class AMPSample:
    """Single AMP sample with all features."""
    sequence: str
    mic_value: float  # log10(MIC)
    pathogen: str
    pathogen_label: int
    properties: np.ndarray  # Physicochemical features


class AMPDataset(Dataset):
    """PyTorch Dataset for AMP activity prediction."""

    def __init__(
        self,
        sequences: List[str],
        mic_values: np.ndarray,
        pathogen_labels: np.ndarray,
        properties: np.ndarray,
        pathogens: List[str],
        normalize_properties: bool = True,
        property_mean: Optional[np.ndarray] = None,
        property_std: Optional[np.ndarray] = None,
    ):
        """Initialize AMP dataset.

        Args:
            sequences: List of peptide sequences
            mic_values: log10(MIC) values (n_samples,)
            pathogen_labels: Pathogen class labels (n_samples,)
            properties: Feature matrix (n_samples, n_features)
            pathogens: Pathogen names (n_samples,)
            normalize_properties: Whether to normalize properties
            property_mean: Pre-computed mean for normalization
            property_std: Pre-computed std for normalization
        """
        self.sequences = sequences
        self.mic_values = torch.tensor(mic_values, dtype=torch.float32)
        self.pathogen_labels = torch.tensor(pathogen_labels, dtype=torch.long)
        self.pathogens = pathogens

        # Normalize properties
        if normalize_properties:
            if property_mean is None:
                self.property_mean = properties.mean(axis=0)
                self.property_std = properties.std(axis=0) + 1e-8
            else:
                self.property_mean = property_mean
                self.property_std = property_std

            properties_normalized = (properties - self.property_mean) / self.property_std
            self.properties = torch.tensor(properties_normalized, dtype=torch.float32)
        else:
            self.property_mean = None
            self.property_std = None
            self.properties = torch.tensor(properties, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Returns:
            Dictionary with sequence, mic, pathogen_label, properties
        """
        return {
            'sequence': self.sequences[idx],
            'mic': self.mic_values[idx],
            'pathogen_label': self.pathogen_labels[idx],
            'properties': self.properties[idx],
            'pathogen': self.pathogens[idx],
        }

    def get_stats(self) -> Dict[str, float]:
        """Get dataset statistics."""
        return {
            'n_samples': len(self),
            'mic_mean': self.mic_values.mean().item(),
            'mic_std': self.mic_values.std().item(),
            'n_pathogens': len(set(self.pathogens)),
        }


def collate_amp_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for AMP batches.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary
    """
    return {
        'sequences': [s['sequence'] for s in batch],
        'mic': torch.stack([s['mic'] for s in batch]),
        'pathogen_labels': torch.stack([s['pathogen_label'] for s in batch]),
        'properties': torch.stack([s['properties'] for s in batch]),
        'pathogens': [s['pathogen'] for s in batch],
    }


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_amp_data(
    target: Optional[str] = None,
    deduplicate: bool = True,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load AMP data from DRAMP loader.

    Args:
        target: Optional pathogen target filter
        deduplicate: Whether to remove duplicates

    Returns:
        Tuple of (sequences, mic_values, pathogen_labels, properties, pathogen_names)
    """
    loader = DRAMPLoader()
    db = loader.generate_curated_database()

    if deduplicate:
        db = db.deduplicate()

    # Get training data and labels
    X, y = db.get_training_data(target=target)
    pathogen_labels = np.array(db.get_pathogen_labels())

    # Get sequences and pathogen names
    sequences = []
    pathogen_names = []

    for record in db.records:
        if record.mic_value and record.mic_value > 0:
            sequences.append(record.sequence)
            org = record.target_organism or ''
            org_lower = org.lower()
            # Map to canonical name
            if 'escherichia' in org_lower or 'e. coli' in org_lower:
                pathogen_names.append('escherichia')
            elif 'pseudomonas' in org_lower or 'p. aeruginosa' in org_lower:
                pathogen_names.append('pseudomonas')
            elif 'staphylococcus' in org_lower or 's. aureus' in org_lower:
                pathogen_names.append('staphylococcus')
            elif 'acinetobacter' in org_lower or 'a. baumannii' in org_lower:
                pathogen_names.append('acinetobacter')
            else:
                pathogen_names.append('other')

    return sequences, y, pathogen_labels, X, pathogen_names


def create_stratified_datasets(
    fold_idx: int = 0,
    n_folds: int = 5,
    random_state: int = 42,
    normalize_properties: bool = True,
) -> Tuple[AMPDataset, AMPDataset]:
    """Create stratified train/val datasets for a specific fold.

    Args:
        fold_idx: Fold index (0 to n_folds-1)
        n_folds: Total number of folds
        random_state: Random seed
        normalize_properties: Whether to normalize properties

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    loader = DRAMPLoader()
    db = loader.generate_curated_database().deduplicate()

    # Get splits
    X_train, y_train, X_val, y_val = db.get_training_data_split(
        fold_idx=fold_idx,
        n_folds=n_folds,
        random_state=random_state,
    )

    # Get train/val indices
    splits = db.get_stratified_splits(n_folds=n_folds, random_state=random_state)
    train_indices, val_indices = splits[fold_idx]

    # Get sequences and pathogens for train/val
    all_records = [r for r in db.records if r.mic_value and r.mic_value > 0]

    def get_subset_data(indices):
        sequences = []
        pathogens = []
        pathogen_labels = []

        for idx in indices:
            record = all_records[idx]
            sequences.append(record.sequence)

            org = (record.target_organism or '').lower()
            if 'escherichia' in org or 'e. coli' in org:
                pathogen = 'escherichia'
            elif 'pseudomonas' in org or 'p. aeruginosa' in org:
                pathogen = 'pseudomonas'
            elif 'staphylococcus' in org or 's. aureus' in org:
                pathogen = 'staphylococcus'
            elif 'acinetobacter' in org or 'a. baumannii' in org:
                pathogen = 'acinetobacter'
            else:
                pathogen = 'other'

            pathogens.append(pathogen)
            pathogen_labels.append(PATHOGEN_TO_LABEL.get(pathogen, 4))

        return sequences, pathogens, np.array(pathogen_labels)

    train_seqs, train_pathogens, train_labels = get_subset_data(train_indices)
    val_seqs, val_pathogens, val_labels = get_subset_data(val_indices)

    # Create train dataset first (to compute normalization stats)
    train_dataset = AMPDataset(
        sequences=train_seqs,
        mic_values=y_train,
        pathogen_labels=train_labels,
        properties=X_train,
        pathogens=train_pathogens,
        normalize_properties=normalize_properties,
    )

    # Create val dataset with train stats
    val_dataset = AMPDataset(
        sequences=val_seqs,
        mic_values=y_val,
        pathogen_labels=val_labels,
        properties=X_val,
        pathogens=val_pathogens,
        normalize_properties=normalize_properties,
        property_mean=train_dataset.property_mean,
        property_std=train_dataset.property_std,
    )

    return train_dataset, val_dataset


def create_stratified_dataloaders(
    fold_idx: int = 0,
    n_folds: int = 5,
    batch_size: int = 32,
    num_workers: int = 0,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create stratified train/val dataloaders.

    Args:
        fold_idx: Fold index
        n_folds: Number of folds
        batch_size: Batch size
        num_workers: DataLoader workers
        random_state: Random seed

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset, val_dataset = create_stratified_datasets(
        fold_idx=fold_idx,
        n_folds=n_folds,
        random_state=random_state,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_amp_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_amp_batch,
    )

    return train_loader, val_loader


def create_full_dataset(
    normalize_properties: bool = True,
) -> AMPDataset:
    """Create dataset with all data (no train/val split).

    Args:
        normalize_properties: Whether to normalize properties

    Returns:
        Full dataset
    """
    sequences, mic_values, pathogen_labels, properties, pathogens = load_amp_data(
        deduplicate=True
    )

    return AMPDataset(
        sequences=sequences,
        mic_values=mic_values,
        pathogen_labels=pathogen_labels,
        properties=properties,
        pathogens=pathogens,
        normalize_properties=normalize_properties,
    )


# =============================================================================
# Main (Testing)
# =============================================================================


if __name__ == '__main__':
    print("Testing AMP Dataset...")
    print()

    # Test stratified datasets
    train_ds, val_ds = create_stratified_datasets(fold_idx=0, n_folds=5)

    print(f"Train dataset: {len(train_ds)} samples")
    print(f"Val dataset: {len(val_ds)} samples")
    print()

    # Check first sample
    sample = train_ds[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"  Sequence: {sample['sequence'][:20]}...")
    print(f"  MIC: {sample['mic'].item():.3f}")
    print(f"  Pathogen: {sample['pathogen']}")
    print(f"  Properties shape: {sample['properties'].shape}")
    print()

    # Test dataloader
    train_loader, val_loader = create_stratified_dataloaders(
        fold_idx=0, batch_size=16
    )

    batch = next(iter(train_loader))
    print(f"Batch keys: {list(batch.keys())}")
    print(f"  Sequences: {len(batch['sequences'])} samples")
    print(f"  MIC shape: {batch['mic'].shape}")
    print(f"  Pathogen labels shape: {batch['pathogen_labels'].shape}")
    print(f"  Properties shape: {batch['properties'].shape}")
    print()

    # Check pathogen distribution
    all_pathogens = train_ds.pathogens + val_ds.pathogens
    from collections import Counter
    print("Pathogen distribution:")
    for pathogen, count in Counter(all_pathogens).most_common():
        print(f"  {pathogen}: {count}")

    print()
    print("Dataset tests passed!")
