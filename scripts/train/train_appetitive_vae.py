"""Training script for Appetitive Dual-VAE.

Bio-inspired VAE training with emergent drives toward:
- r > 0.99 3-adic correlation (ultrametric structure)
- Full 19683 operation coverage
- Algebraic closure (emergent addition)

Appetitive modules:
1. AdaptiveRankingLoss - Multi-scale triplet margins
2. HierarchicalNormLoss - MSB/LSB variance structure
3. CuriosityModule - Density-based exploration
4. SymbioticBridge - VAE-A/VAE-B mutual information
5. AlgebraicClosureLoss - Homomorphism constraint

Metric-gated phase transitions (not epoch-based).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import yaml
import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.ternary_vae_v5_6 import DualNeuralVAEV5
from src.models.appetitive_vae import AppetitiveDualVAE
from src.training import AppetitiveVAETrainer
from src.data import generate_all_ternary_operations, TernaryOperationDataset


def main():
    parser = argparse.ArgumentParser(description='Train Appetitive Dual-VAE')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"{'='*80}")
    print("Appetitive Dual-VAE - Bio-Inspired Training")
    print(f"{'='*80}")
    print(f"Config: {args.config}")

    # Set seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Generate dataset
    print("\nGenerating dataset...")
    operations = generate_all_ternary_operations()
    dataset = TernaryOperationDataset(operations)
    print(f"Total operations: {len(dataset):,}")

    # Split dataset
    train_split = config['train_split']
    val_split = config['val_split']
    test_split = config['test_split']

    train_size = int(train_split * len(dataset))
    val_size = int(val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    # Initialize base model
    model_config = config['model']
    base_model = DualNeuralVAEV5(
        input_dim=model_config['input_dim'],
        latent_dim=model_config['latent_dim'],
        rho_min=model_config['rho_min'],
        rho_max=model_config['rho_max'],
        lambda3_base=model_config['lambda3_base'],
        lambda3_amplitude=model_config['lambda3_amplitude'],
        eps_kl=model_config['eps_kl'],
        gradient_balance=model_config.get('gradient_balance', True),
        adaptive_scheduling=model_config.get('adaptive_scheduling', True),
        use_statenet=model_config.get('use_statenet', True),
        statenet_lr_scale=model_config.get('statenet_lr_scale', 0.05),
        statenet_lambda_scale=model_config.get('statenet_lambda_scale', 0.01)
    )

    # Build appetitive config from YAML sections
    appetite_config = {
        'latent_dim': model_config['latent_dim'],

        # Ranking
        'ranking_margin': config.get('appetite_ranking', {}).get('base_margin', 0.1),
        'ranking_n_triplets': config.get('appetite_ranking', {}).get('n_triplets', 1000),
        'appetite_ranking': config.get('appetite_ranking', {}).get('initial_weight', 0.5),

        # Hierarchy
        'hierarchy_n_groups': config.get('appetite_hierarchy', {}).get('n_groups', 4),
        'appetite_hierarchy': config.get('appetite_hierarchy', {}).get('initial_weight', 0.1),

        # Curiosity
        'curiosity_bandwidth': config.get('appetite_curiosity', {}).get('bandwidth', 1.0),
        'curiosity_max_history': config.get('appetite_curiosity', {}).get('max_history', 5000),
        'appetite_curiosity': config.get('appetite_curiosity', {}).get('initial_weight', 0.0),

        # Symbiosis
        'symbiosis_hidden_dim': config.get('appetite_symbiosis', {}).get('hidden_dim', 32),
        'appetite_symbiosis': config.get('appetite_symbiosis', {}).get('initial_weight', 0.0),

        # Closure
        'appetite_closure': config.get('appetite_closure', {}).get('initial_weight', 0.0),

        # Violation buffer
        'violation_capacity': config.get('violation_buffer', {}).get('capacity', 10000),

        # Phase gates
        'phase_1a_gate': config.get('phase_gates', {}).get('phase_1a_to_1b', 0.75),
        'phase_1b_gate': config.get('phase_gates', {}).get('phase_1b_to_2a', 0.85),
        'phase_2a_gate': config.get('phase_gates', {}).get('phase_2a_to_2b', 2.0),
        'phase_2b_gate': config.get('phase_gates', {}).get('phase_2b_to_3', 0.5),
    }

    # Create appetitive model
    model = AppetitiveDualVAE(base_model, appetite_config)

    print(f"\nAppetitive modules initialized:")
    print(f"  Ranking: margin={appetite_config['ranking_margin']}, triplets={appetite_config['ranking_n_triplets']}")
    print(f"  Hierarchy: groups={appetite_config['hierarchy_n_groups']}")
    print(f"  Curiosity: bandwidth={appetite_config['curiosity_bandwidth']}, history={appetite_config['curiosity_max_history']}")
    print(f"  Symbiosis: hidden_dim={appetite_config['symbiosis_hidden_dim']}")
    print(f"  Closure: active in Phase 2B+")
    print(f"\nPhase gates (metric-based):")
    print(f"  1A -> 1B: r > {appetite_config['phase_1a_gate']}")
    print(f"  1B -> 2A: r > {appetite_config['phase_1b_gate']}")
    print(f"  2A -> 2B: MI > {appetite_config['phase_2a_gate']}")
    print(f"  2B -> 3:  add > {appetite_config['phase_2b_gate']}")

    # Initialize trainer
    trainer = AppetitiveVAETrainer(model, config, device)

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
