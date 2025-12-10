"""Purposeful VAE Training: Backbone + Consequence Awareness.

Uses the proven v5.6 backbone with ranking loss enabled, adding a
consequence predictor that learns WHY improving ranking matters.

Key insight: Don't parallel the hunger, give it purpose.
The consequence predictor learns: ranking_correlation → addition_accuracy
This teaches the model that better metric structure enables algebraic closure.

Architecture:
- Base: DualNeuralVAEV5 (unchanged)
- Loss: DualVAELoss with enable_ranking_loss=true (unchanged)
- Purpose: ConsequencePredictor observes and learns (new)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import argparse
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.ternary_vae_v5_6 import DualNeuralVAEV5
from src.training import TernaryVAETrainer
from src.data import generate_all_ternary_operations, TernaryOperationDataset
from src.losses.consequence_predictor import (
    ConsequencePredictor,
    evaluate_addition_accuracy
)


def compute_ranking_correlation(model, device, n_samples=5000):
    """Compute 3-adic ranking correlation.

    Returns the concordance rate between 3-adic distance ordering
    and latent space distance ordering.
    """
    model.eval()

    with torch.no_grad():
        indices = torch.randint(0, 19683, (n_samples,), device=device)

        # Convert to ternary
        ternary_data = torch.zeros(n_samples, 9, device=device)
        for i in range(9):
            ternary_data[:, i] = ((indices // (3**i)) % 3) - 1

        # Forward pass
        outputs = model(ternary_data.float(), 1.0, 1.0, 0.5, 0.5)
        z_A = outputs['z_A']
        z_B = outputs['z_B']

        # Sample triplets
        n_triplets = 1000
        i_idx = torch.randint(n_samples, (n_triplets,), device=device)
        j_idx = torch.randint(n_samples, (n_triplets,), device=device)
        k_idx = torch.randint(n_samples, (n_triplets,), device=device)

        # Filter distinct
        valid = (i_idx != j_idx) & (i_idx != k_idx) & (j_idx != k_idx)
        i_idx, j_idx, k_idx = i_idx[valid], j_idx[valid], k_idx[valid]

        if len(i_idx) < 100:
            return 0.5, 0.5

        # Compute 3-adic valuations
        def compute_valuation(diff):
            val = torch.zeros_like(diff, dtype=torch.float32)
            remaining = diff.clone()
            for _ in range(10):
                mask = (remaining % 3 == 0) & (remaining > 0)
                val[mask] += 1
                remaining[mask] = remaining[mask] // 3
            val[diff == 0] = 10.0
            return val

        diff_ij = torch.abs(indices[i_idx] - indices[j_idx])
        diff_ik = torch.abs(indices[i_idx] - indices[k_idx])

        v_ij = compute_valuation(diff_ij)
        v_ik = compute_valuation(diff_ik)

        # 3-adic: larger valuation = smaller distance
        padic_closer_ij = (v_ij > v_ik).float()

        # Latent distances
        d_A_ij = torch.norm(z_A[i_idx] - z_A[j_idx], dim=1)
        d_A_ik = torch.norm(z_A[i_idx] - z_A[k_idx], dim=1)
        d_B_ij = torch.norm(z_B[i_idx] - z_B[j_idx], dim=1)
        d_B_ik = torch.norm(z_B[i_idx] - z_B[k_idx], dim=1)

        latent_A_closer = (d_A_ij < d_A_ik).float()
        latent_B_closer = (d_B_ij < d_B_ik).float()

        corr_A = (padic_closer_ij == latent_A_closer).float().mean().item()
        corr_B = (padic_closer_ij == latent_B_closer).float().mean().item()

    return corr_A, corr_B


class PurposefulTrainer:
    """Wrapper around TernaryVAETrainer with consequence awareness."""

    def __init__(self, base_trainer, model, device, config):
        self.base_trainer = base_trainer
        self.model = model
        self.device = device
        self.config = config

        # Consequence predictor - learns WHY ranking matters
        self.consequence_predictor = ConsequencePredictor(
            latent_dim=config['model']['latent_dim']
        ).to(device)

        self.consequence_optimizer = optim.Adam(
            self.consequence_predictor.parameters(),
            lr=0.001
        )

        # History
        self.correlation_history = []
        self.addition_accuracy_history = []
        self.predicted_accuracy_history = []
        self.self_model_quality_history = []

    def train_epoch_with_purpose(self, train_loader, val_loader, epoch):
        """Train one epoch with consequence learning."""

        # Use base trainer for main training
        train_losses = self.base_trainer.train_epoch(train_loader)
        val_losses = self.base_trainer.validate(val_loader)

        # Compute ranking correlation
        corr_A, corr_B = compute_ranking_correlation(self.model, self.device)
        corr_mean = (corr_A + corr_B) / 2
        self.correlation_history.append(corr_mean)

        # Evaluate addition accuracy (ground truth for consequence prediction)
        addition_acc = evaluate_addition_accuracy(self.model, self.device)
        self.addition_accuracy_history.append(addition_acc)

        # Get coverage for predictor input
        unique_A, cov_A = self.base_trainer.monitor.evaluate_coverage(
            self.model, self.config['eval_num_samples'], self.device, 'A'
        )

        # Train consequence predictor
        self.consequence_optimizer.zero_grad()

        # Forward: predict addition accuracy from current ranking
        with torch.no_grad():
            # Get z for statistics
            sample_indices = torch.randint(0, 19683, (1000,), device=self.device)
            ternary_data = torch.zeros(1000, 9, device=self.device)
            for i in range(9):
                ternary_data[:, i] = ((sample_indices // (3**i)) % 3) - 1
            outputs = self.model(ternary_data.float(), 1.0, 1.0, 0.5, 0.5)
            z = outputs['z_A']

        predicted_acc = self.consequence_predictor(
            corr_mean, z, cov_A
        )
        self.predicted_accuracy_history.append(predicted_acc.item())

        # Compute consequence loss (prediction error)
        consequence_loss = self.consequence_predictor.compute_loss(
            predicted_acc, addition_acc
        )

        # Backprop consequence learning
        consequence_loss.backward()
        self.consequence_optimizer.step()

        # Update history
        self.consequence_predictor.update_history(
            predicted_acc.item(), addition_acc
        )

        # Track self-model quality
        self_model_quality = self.consequence_predictor.get_prediction_quality()
        self.self_model_quality_history.append(self_model_quality)

        return {
            **train_losses,
            'corr_A': corr_A,
            'corr_B': corr_B,
            'corr_mean': corr_mean,
            'addition_accuracy': addition_acc,
            'predicted_accuracy': predicted_acc.item(),
            'consequence_loss': consequence_loss.item(),
            'self_model_quality': self_model_quality
        }


def main():
    parser = argparse.ArgumentParser(description='Train Purposeful VAE')
    parser.add_argument('--config', type=str, default='configs/ternary_v5_6.yaml',
                        help='Path to config file')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override: enable ranking loss (the proven approach)
    config['padic_losses']['enable_ranking_loss'] = True
    config['padic_losses']['ranking_loss_weight'] = 0.5
    config['padic_losses']['ranking_n_triplets'] = 500

    # Override: use purposeful checkpoint dir
    config['checkpoint_dir'] = 'sandbox-training/checkpoints/purposeful'
    config['experiment_name'] = 'purposeful_vae'
    config['total_epochs'] = args.epochs

    print(f"{'='*80}")
    print("Purposeful VAE Training")
    print("Backbone: v5.6 + Ranking Loss")
    print("Purpose: Consequence Predictor (r -> addition_accuracy)")
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
    train_size = int(config['train_split'] * len(dataset))
    val_size = int(config['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

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

    # Initialize model
    model_config = config['model']
    model = DualNeuralVAEV5(
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

    # Initialize base trainer
    base_trainer = TernaryVAETrainer(model, config, device)

    # Wrap with purposeful trainer
    trainer = PurposefulTrainer(base_trainer, model, device, config)

    print(f"\n{'='*80}")
    print("Starting Purposeful Training")
    print(f"{'='*80}\n")

    total_epochs = config['total_epochs']
    best_corr = 0.0

    for epoch in range(total_epochs):
        base_trainer.epoch = epoch

        # Train with consequence awareness
        losses = trainer.train_epoch_with_purpose(train_loader, val_loader, epoch)

        # Check for best model
        is_best = base_trainer.monitor.check_best(losses['loss'])

        # Evaluate coverage
        unique_A, cov_A = base_trainer.monitor.evaluate_coverage(
            model, config['eval_num_samples'], device, 'A'
        )
        unique_B, cov_B = base_trainer.monitor.evaluate_coverage(
            model, config['eval_num_samples'], device, 'B'
        )

        # Update histories
        base_trainer.monitor.update_histories(
            losses['H_A'], losses['H_B'], unique_A, unique_B
        )

        # Track best correlation
        if losses['corr_mean'] > best_corr:
            best_corr = losses['corr_mean']

        # Print epoch summary
        print(f"\nEpoch {epoch}/{total_epochs}")
        print(f"  Loss: {losses['loss']:.4f} | Coverage: A={cov_A:.1f}% B={cov_B:.1f}%")
        print(f"  3-Adic Correlation: A={losses['corr_A']:.3f} B={losses['corr_B']:.3f} (best={best_corr:.3f})")
        print(f"  PURPOSE:")
        print(f"    Addition Accuracy: {losses['addition_accuracy']:.3f}")
        print(f"    Predicted Accuracy: {losses['predicted_accuracy']:.3f}")
        print(f"    Consequence Loss: {losses['consequence_loss']:.4f}")
        print(f"    Self-Model Quality: {losses['self_model_quality']:.3f}")

        # Phase transition insight
        if losses['self_model_quality'] > 0.5:
            print(f"    -> Model understands: ranking -> closure")

        # Log to TensorBoard if available
        if base_trainer.monitor.writer is not None:
            base_trainer.monitor.writer.add_scalars('Purpose/Correlation', {
                'VAE_A': losses['corr_A'],
                'VAE_B': losses['corr_B']
            }, epoch)
            base_trainer.monitor.writer.add_scalar(
                'Purpose/AdditionAccuracy', losses['addition_accuracy'], epoch
            )
            base_trainer.monitor.writer.add_scalar(
                'Purpose/PredictedAccuracy', losses['predicted_accuracy'], epoch
            )
            base_trainer.monitor.writer.add_scalar(
                'Purpose/SelfModelQuality', losses['self_model_quality'], epoch
            )
            base_trainer.monitor.writer.flush()

        # Save checkpoint
        if epoch % config['checkpoint_freq'] == 0:
            base_trainer.checkpoint_manager.save_checkpoint(
                epoch, model, base_trainer.optimizer,
                {
                    **base_trainer.monitor.get_metadata(),
                    'correlation_history': trainer.correlation_history,
                    'addition_accuracy_history': trainer.addition_accuracy_history,
                    'self_model_quality': losses['self_model_quality']
                },
                is_best
            )

        # Early stopping
        if base_trainer.monitor.should_stop(config['patience']):
            print(f"\nEarly stopping triggered")
            break

    # Summary
    print(f"\n{'='*80}")
    print("Training Complete")
    print(f"{'='*80}")
    print(f"Best 3-adic correlation: {best_corr:.4f}")
    print(f"Final addition accuracy: {trainer.addition_accuracy_history[-1]:.4f}")
    print(f"Final self-model quality: {trainer.self_model_quality_history[-1]:.4f}")

    base_trainer.monitor.close()


if __name__ == '__main__':
    main()
