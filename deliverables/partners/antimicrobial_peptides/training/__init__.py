"""Training module for PeptideVAE.

This module provides:
- PyTorch Dataset for AMP activity data
- Training pipeline with stratified cross-validation
- Curriculum learning for 6-component loss
"""

from .dataset import (
    AMPDataset,
    AMPSample,
    create_stratified_dataloaders,
    create_stratified_datasets,
    create_full_dataset,
    collate_amp_batch,
    PATHOGEN_TO_LABEL,
    LABEL_TO_PATHOGEN,
)

__all__ = [
    'AMPDataset',
    'AMPSample',
    'create_stratified_dataloaders',
    'create_stratified_datasets',
    'create_full_dataset',
    'collate_amp_batch',
    'PATHOGEN_TO_LABEL',
    'LABEL_TO_PATHOGEN',
]
