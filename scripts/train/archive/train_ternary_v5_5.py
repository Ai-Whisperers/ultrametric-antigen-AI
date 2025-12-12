"""Training script for Ternary VAE v5.5 - Clean implementation with full config integration.

Key improvements over v5.3/v5.4:
- ALL config parameters properly used (temp_boost_amplitude, temp_phase4)
- Proper Phase 4 ultra-exploration implementation
- Consistent 1-epoch monitoring
- No dead code or config mismatches
- Clean, maintainable codebase

This training script guarantees that every parameter in the config file is actually used.
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
import math
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.ternary_vae_v5_5 import DualNeuralVAEV5


def generate_all_ternary_operations():
    """Generate all 19,683 possible ternary operations."""
    operations = []
    for i in range(3**9):
        op = []
        num = i
        for _ in range(9):
            op.append(num % 3 - 1)
            num //= 3
        operations.append(op)
    return np.array(operations, dtype=np.float32)


class TernaryOperationDataset(Dataset):
    """Dataset of ternary operations."""

    def __init__(self, operations):
        self.operations = torch.FloatTensor(operations)

    def __len__(self):
        return len(self.operations)

    def __getitem__(self, idx):
        return self.operations[idx]


def linear_schedule(epoch, start_val, end_val, total_epochs, start_epoch=0):
    """Linear scheduling from start_val to end_val."""
    if epoch < start_epoch:
        return start_val
    progress = min((epoch - start_epoch) / total_epochs, 1.0)
    return start_val + (end_val - start_val) * progress


def cyclic_schedule(epoch, base_val, amplitude, period):
    """Cyclic scheduling: base ± amplitude with given period."""
    phase = (epoch % period) / period * 2 * np.pi
    return base_val + amplitude * np.cos(phase)


def get_lr_from_schedule(epoch, schedule):
    """Get learning rate from schedule."""
    lr = schedule[0]['lr']
    for entry in schedule:
        if epoch >= entry['epoch']:
            lr = entry['lr']
    return lr


class DNVAETrainerV5:
    """Trainer for Dual-Neural VAE v5.5 with complete config integration."""

    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device

        # Initialize model
        model_config = config['model']
        self.model = DualNeuralVAEV5(
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
        ).to(device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['optimizer']['lr_start'],
            weight_decay=config['optimizer'].get('weight_decay', 0.0001)
        )

        # Controller parameters
        self.temp_lag = config['controller']['temp_lag']
        self.beta_phase_lag = config['controller']['beta_phase_lag']

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Coverage tracking
        self.coverage_A_history = []
        self.coverage_B_history = []
        self.H_A_history = []
        self.H_B_history = []

        # Checkpoint directory
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Phase transitions
        self.phase_4_start = config['phase_transitions']['ultra_exploration_start']

        print(f"\n{'='*80}")
        print(f"DN-VAE v5.5 Initialized")
        print(f"{'='*80}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if model_config.get('use_statenet', True):
            statenet_params = sum(p.numel() for p in self.model.state_net.parameters())
            print(f"StateNet parameters: {statenet_params:,} ({statenet_params/sum(p.numel() for p in self.model.parameters())*100:.2f}%)")
        print(f"Device: {device}")
        print(f"Gradient balance: {model_config.get('gradient_balance', True)}")
        print(f"Adaptive scheduling: {model_config.get('adaptive_scheduling', True)}")
        print(f"StateNet enabled: {model_config.get('use_statenet', True)}")

    def get_temperature(self, epoch, vae='A'):
        """Get temperature with proper Phase 4 support.

        FIXED: Now actually uses config parameters temp_boost_amplitude and temp_phase4
        """
        if vae == 'A':
            # Chaotic regime: cyclic with boost in Phase 4
            base_temp = linear_schedule(
                epoch,
                self.config['vae_a']['temp_start'],
                self.config['vae_a']['temp_end'],
                self.config['total_epochs']
            )

            if self.config['vae_a'].get('temp_cyclic', False):
                # Phase 1-3: Small cyclic modulation
                amplitude = 0.1 * base_temp

                # Phase 4: Enhanced exploration with temp_boost_amplitude
                if epoch >= self.phase_4_start and 'temp_boost_amplitude' in self.config['vae_a']:
                    amplitude = self.config['vae_a']['temp_boost_amplitude']

                period = 30
                return max(0.1, cyclic_schedule(epoch, base_temp, amplitude, period))

            return base_temp
        else:
            # Frozen regime: monotonic with Phase 4 boost
            epoch_lagged = max(0, epoch - self.temp_lag)

            # Phase 1-3: Normal annealing
            if epoch < self.phase_4_start:
                return linear_schedule(
                    epoch_lagged,
                    self.config['vae_b']['temp_start'],
                    self.config['vae_b']['temp_end'],
                    self.config['total_epochs']
                )
            else:
                # Phase 4: Use temp_phase4 if specified
                if 'temp_phase4' in self.config['vae_b']:
                    return self.config['vae_b']['temp_phase4']
                else:
                    return self.config['vae_b']['temp_end']

    def get_beta(self, epoch, vae='A'):
        """Get beta with KL warmup and phase offset for VAE-B.

        Implements β-VAE warmup to prevent posterior collapse:
        - Warmup phase: β increases from 0 to target over warmup_epochs
        - After warmup: β follows configured schedule
        """
        if vae == 'A':
            # Get warmup parameters
            warmup_epochs = self.config['vae_a'].get('beta_warmup_epochs', 0)

            if warmup_epochs > 0 and epoch < warmup_epochs:
                # Warmup: linearly increase from 0 to beta_start
                beta_target = self.config['vae_a']['beta_start']
                return (epoch / warmup_epochs) * beta_target
            else:
                # Normal schedule after warmup
                return linear_schedule(
                    epoch - warmup_epochs,
                    self.config['vae_a']['beta_start'],
                    self.config['vae_a']['beta_end'],
                    self.config['total_epochs'] - warmup_epochs
                )
        else:
            # VAE-B warmup
            warmup_epochs = self.config['vae_b'].get('beta_warmup_epochs', 0)

            if warmup_epochs > 0 and epoch < warmup_epochs:
                beta_target = self.config['vae_b']['beta_start']
                return (epoch / warmup_epochs) * beta_target
            else:
                # After warmup, use phase offset from VAE-A
                beta_A = self.get_beta(epoch, 'A')
                return beta_A * abs(math.sin(self.beta_phase_lag))

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.model.train()
        self.epoch = epoch
        self.model.epoch = epoch

        # Update adaptive parameters
        self.model.rho = self.model.compute_phase_scheduled_rho(epoch, self.phase_4_start)
        self.model.lambda3 = self.model.compute_cyclic_lambda3(epoch, period=30)

        grad_ratio = (self.model.grad_norm_B_ema / (self.model.grad_norm_A_ema + 1e-8)).item()
        self.model.update_adaptive_ema_momentum(grad_ratio)

        if len(self.coverage_A_history) > 0:
            coverage_A = self.coverage_A_history[-1]
            coverage_B = self.coverage_B_history[-1]
            self.model.update_adaptive_lambdas(grad_ratio, coverage_A, coverage_B)

        # Get schedules
        temp_A = self.get_temperature(epoch, 'A')
        temp_B = self.get_temperature(epoch, 'B')
        beta_A = self.get_beta(epoch, 'A')
        beta_B = self.get_beta(epoch, 'B')

        lr_scheduled = get_lr_from_schedule(epoch, self.config['optimizer']['lr_schedule'])
        entropy_weight = self.config['vae_b']['entropy_weight']
        repulsion_weight = self.config['vae_b']['repulsion_weight']
        free_bits = self.config.get('free_bits', 0.0)  # Free bits for KL

        epoch_losses = defaultdict(float)
        num_batches = 0

        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(self.device)

            outputs = self.model(batch_data, temp_A, temp_B, beta_A, beta_B)
            losses = self.model.loss_function(batch_data, outputs,
                                            entropy_weight, repulsion_weight, free_bits)

            # Apply StateNet corrections once per epoch
            if self.model.use_statenet and batch_idx == 0:
                corrected_lr, *deltas = self.model.apply_statenet_corrections(
                    lr_scheduled,
                    losses['H_A'].item() if torch.is_tensor(losses['H_A']) else losses['H_A'],
                    losses['H_B'].item() if torch.is_tensor(losses['H_B']) else losses['H_B'],
                    losses['kl_A'].item(),
                    losses['kl_B'].item(),
                    grad_ratio
                )

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = corrected_lr

                epoch_losses['lr_corrected'] = corrected_lr
                epoch_losses['delta_lr'] = deltas[0]
                epoch_losses['delta_lambda1'] = deltas[1]
                epoch_losses['delta_lambda2'] = deltas[2]
                epoch_losses['delta_lambda3'] = deltas[3]
            else:
                if batch_idx == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr_scheduled

            # Backward and optimize
            self.optimizer.zero_grad()
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.model.update_gradient_norms()
            self.optimizer.step()

            # Accumulate losses
            for key, val in losses.items():
                if isinstance(val, torch.Tensor):
                    epoch_losses[key] += val.item()
                else:
                    epoch_losses[key] += val

            num_batches += 1

        # Average losses
        for key in epoch_losses:
            if key not in ['lr_corrected', 'delta_lr', 'delta_lambda1', 'delta_lambda2', 'delta_lambda3']:
                epoch_losses[key] /= num_batches

        # Update history
        self.H_A_history.append(epoch_losses['H_A'])
        self.H_B_history.append(epoch_losses['H_B'])

        # Store schedule info
        epoch_losses['temp_A'] = temp_A
        epoch_losses['temp_B'] = temp_B
        epoch_losses['beta_A'] = beta_A
        epoch_losses['beta_B'] = beta_B
        epoch_losses['lr_scheduled'] = lr_scheduled
        epoch_losses['grad_ratio'] = grad_ratio
        epoch_losses['ema_momentum'] = self.model.grad_ema_momentum

        return epoch_losses

    def validate(self, val_loader):
        """Validation pass."""
        self.model.eval()
        epoch_losses = defaultdict(float)
        num_batches = 0

        temp_A = self.get_temperature(self.epoch, 'A')
        temp_B = self.get_temperature(self.epoch, 'B')
        beta_A = self.get_beta(self.epoch, 'A')
        beta_B = self.get_beta(self.epoch, 'B')
        entropy_weight = self.config['vae_b']['entropy_weight']
        repulsion_weight = self.config['vae_b']['repulsion_weight']
        free_bits = self.config.get('free_bits', 0.0)

        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(self.device)
                outputs = self.model(batch_data, temp_A, temp_B, beta_A, beta_B)
                losses = self.model.loss_function(batch_data, outputs,
                                                entropy_weight, repulsion_weight, free_bits)

                for key, val in losses.items():
                    if isinstance(val, torch.Tensor):
                        epoch_losses[key] += val.item()
                    else:
                        epoch_losses[key] += val

                num_batches += 1

        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def evaluate_coverage(self, num_samples, vae='A'):
        """Evaluate operation coverage."""
        self.model.eval()
        unique_ops = set()

        with torch.no_grad():
            batch_size = 1000
            num_batches = num_samples // batch_size

            for _ in range(num_batches):
                samples = self.model.sample(batch_size, self.device, vae)
                samples_rounded = torch.round(samples).long()

                for i in range(batch_size):
                    lut = samples_rounded[i]
                    lut_tuple = tuple(lut.cpu().tolist())
                    unique_ops.add(lut_tuple)

        coverage_pct = (len(unique_ops) / 19683) * 100
        return len(unique_ops), coverage_pct

    def save_checkpoint(self, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'H_A_history': self.H_A_history,
            'H_B_history': self.H_B_history,
            'coverage_A_history': self.coverage_A_history,
            'coverage_B_history': self.coverage_B_history,
            'lambda1': self.model.lambda1,
            'lambda2': self.model.lambda2,
            'lambda3': self.model.lambda3,
            'rho': self.model.rho,
            'phase': self.model.current_phase,
            'grad_balance_achieved': self.model.grad_balance_achieved,
            'grad_norm_A_ema': self.model.grad_norm_A_ema.item(),
            'grad_norm_B_ema': self.model.grad_norm_B_ema.item(),
            'grad_ema_momentum': self.model.grad_ema_momentum,
            'statenet_enabled': self.model.use_statenet
        }

        if self.model.use_statenet:
            checkpoint['statenet_corrections'] = self.model.statenet_corrections

        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')

        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')

        if self.epoch % self.config['checkpoint_freq'] == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{self.epoch}.pt')

    def train(self, train_loader, val_loader):
        """Main training loop with 1-epoch monitoring."""
        print(f"\n{'='*80}")
        print("Starting DN-VAE v5.5 Training")
        print(f"{'='*80}\n")

        total_epochs = self.config['total_epochs']

        for epoch in range(total_epochs):
            # Train
            train_losses = self.train_epoch(train_loader, epoch)

            # Validate every epoch
            val_losses = self.validate(val_loader)

            # Check for best
            is_best = val_losses['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Evaluate coverage every epoch
            unique_A, cov_A = self.evaluate_coverage(self.config['eval_num_samples'], 'A')
            unique_B, cov_B = self.evaluate_coverage(self.config['eval_num_samples'], 'B')

            self.coverage_A_history.append(unique_A)
            self.coverage_B_history.append(unique_B)

            # Log every epoch
            print(f"\nEpoch {epoch}/{total_epochs}")
            print(f"  Loss: Train={train_losses['loss']:.4f} Val={val_losses['loss']:.4f}")
            print(f"  VAE-A: CE={train_losses['ce_A']:.4f} KL={train_losses['kl_A']:.4f} H={train_losses['H_A']:.3f}")
            print(f"  VAE-B: CE={train_losses['ce_B']:.4f} KL={train_losses['kl_B']:.4f} H={train_losses['H_B']:.3f}")
            print(f"  Weights: λ1={train_losses['lambda1']:.3f} λ2={train_losses['lambda2']:.3f} λ3={train_losses['lambda3']:.3f}")
            print(f"  Phase {train_losses['phase']}: ρ={train_losses['rho']:.3f} (balance: {'✓' if self.model.grad_balance_achieved else '○'})")
            print(f"  Grad: ratio={train_losses['grad_ratio']:.3f} EMA_α={train_losses['ema_momentum']:.2f}")
            print(f"  Temp: A={train_losses['temp_A']:.3f} B={train_losses['temp_B']:.3f} | β: A={train_losses['beta_A']:.3f} B={train_losses['beta_B']:.3f}")

            if self.model.use_statenet and 'lr_corrected' in train_losses:
                print(f"  LR: {train_losses['lr_scheduled']:.6f} → {train_losses['lr_corrected']:.6f} (Δ={train_losses.get('delta_lr', 0):+.3f})")
                print(f"  StateNet: Δλ1={train_losses.get('delta_lambda1', 0):+.3f} Δλ2={train_losses.get('delta_lambda2', 0):+.3f} Δλ3={train_losses.get('delta_lambda3', 0):+.3f}")
            else:
                print(f"  LR: {train_losses['lr_scheduled']:.6f}")

            print(f"  Coverage: A={unique_A} ({cov_A:.2f}%) | B={unique_B} ({cov_B:.2f}%)")

            if is_best:
                print(f"  ✓ Best val loss: {self.best_val_loss:.4f}")

            # Save checkpoint
            self.save_checkpoint(is_best)

            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\n⚠️  Early stopping triggered (patience={self.config['patience']})")
                break

        print(f"\n{'='*80}")
        print("Training Complete")
        print(f"{'='*80}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Final Coverage: A={self.coverage_A_history[-1]} ({self.coverage_A_history[-1]/19683*100:.2f}%)")
        print(f"                B={self.coverage_B_history[-1]} ({self.coverage_B_history[-1]/19683*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Train Ternary VAE v5.5')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"{'='*80}")
    print("Ternary VAE v5.5 - Clean Implementation")
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

    # Initialize trainer
    trainer = DNVAETrainerV5(config, device)

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
