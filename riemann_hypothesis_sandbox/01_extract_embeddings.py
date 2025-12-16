"""
01_extract_embeddings.py - Extract hyperbolic embeddings from trained Ternary VAE

This script loads a trained checkpoint and extracts the hyperbolic embeddings
for all 19,683 ternary operations, saving them for spectral analysis.

Usage:
    python 01_extract_embeddings.py [--checkpoint PATH] [--output DIR]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from tqdm import tqdm

from src.data.generation import generate_all_ternary_operations
from src.models.ternary_vae import TernaryVAEV5_11


def load_v5_5_checkpoint(checkpoint_path: Path, device: str = 'cpu'):
    """Load a v5.5 style checkpoint (encoder/decoder only, no projection)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    print(f"Checkpoint keys: {checkpoint.keys()}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")

    # Check what model type this is
    model_state = checkpoint.get('model_state', checkpoint.get('model', checkpoint.get('model_state_dict', {})))

    # Detect model type from keys
    has_projection = any('projection' in k for k in model_state.keys())
    has_encoder_A = any('encoder_A' in k for k in model_state.keys())

    print(f"Has projection layer: {has_projection}")
    print(f"Has encoder_A: {has_encoder_A}")

    return checkpoint, model_state


def extract_embeddings_v5_5(model_state: dict, operations: torch.Tensor, device: str = 'cpu'):
    """Extract Euclidean embeddings from v5.5 encoder (no hyperbolic projection)."""
    from src.models.ternary_vae import FrozenEncoder

    # Build encoder manually
    encoder = FrozenEncoder(input_dim=9, latent_dim=16)

    # Load weights
    enc_state = {}
    for key, value in model_state.items():
        if key.startswith('encoder_A.'):
            new_key = key.replace('encoder_A.', '')
            enc_state[new_key] = value

    if not enc_state:
        # Try alternate naming
        for key, value in model_state.items():
            if 'encoder' in key.lower() and 'A' in key:
                parts = key.split('.')
                new_key = '.'.join(parts[1:])
                enc_state[new_key] = value

    encoder.load_state_dict(enc_state, strict=False)
    encoder.to(device)
    encoder.eval()

    # Extract embeddings
    operations = operations.to(device)

    with torch.no_grad():
        mu, logvar = encoder(operations)
        # Use mean (deterministic)
        z_euclidean = mu

    return z_euclidean


def extract_embeddings_v5_11(checkpoint_path: Path, operations: torch.Tensor, device: str = 'cpu'):
    """Extract hyperbolic embeddings from v5.11 model with projection."""
    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    model_state = checkpoint.get('model_state', {})

    # Detect dual projection from checkpoint
    has_dual = any('proj_B' in k for k in model_state.keys())
    print(f"Detected dual projection: {has_dual}")

    # Get projection params from config (matching train.py)
    proj_hidden_dim = config.get('projection_hidden_dim', 64)
    proj_layers = config.get('projection_layers', 1)
    proj_dropout = config.get('projection_dropout', 0.0)

    print(f"Projection config: hidden_dim={proj_hidden_dim}, layers={proj_layers}, dropout={proj_dropout}")

    # Create model with matching config (using Option C if encoder_b present)
    has_encoder_b = any('encoder_B' in k for k in model_state.keys())

    if has_encoder_b and config.get('option_c', False):
        from src.models import TernaryVAEV5_11_OptionC
        print("Using TernaryVAEV5_11_OptionC")
        model = TernaryVAEV5_11_OptionC(
            latent_dim=16,
            hidden_dim=proj_hidden_dim,
            max_radius=config.get('max_radius', 0.95),
            curvature=config.get('curvature', 1.0),
            use_controller=False,
            use_dual_projection=has_dual,
            n_projection_layers=proj_layers,
            projection_dropout=proj_dropout
        )
    else:
        print("Using TernaryVAEV5_11")
        model = TernaryVAEV5_11(
            latent_dim=16,
            hidden_dim=proj_hidden_dim,
            max_radius=config.get('max_radius', 0.95),
            curvature=config.get('curvature', 1.0),
            use_controller=False,
            use_dual_projection=has_dual,
            n_projection_layers=proj_layers,
            projection_dropout=proj_dropout
        )

    # Load full state dict
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()

    # Extract embeddings
    operations = operations.to(device)

    with torch.no_grad():
        outputs = model(operations, compute_control=False)
        z_A_hyp = outputs['z_A_hyp']
        z_B_hyp = outputs.get('z_B_hyp', z_A_hyp)
        z_A_euc = outputs['z_A_euc']
        z_B_euc = outputs.get('z_B_euc', z_A_euc)

    return {
        'z_A_hyp': z_A_hyp,
        'z_B_hyp': z_B_hyp,
        'z_A_euc': z_A_euc,
        'z_B_euc': z_B_euc
    }


def project_to_poincare(z_euclidean: torch.Tensor, max_radius: float = 0.95) -> torch.Tensor:
    """Project Euclidean embeddings to Poincaré ball (simple approach)."""
    # Normalize to unit ball, then scale
    norms = torch.norm(z_euclidean, dim=-1, keepdim=True)
    max_norm = norms.max()

    if max_norm > 0:
        # Scale to fit in ball with max_radius
        z_poincare = z_euclidean / (max_norm / max_radius + 1e-8)
    else:
        z_poincare = z_euclidean

    # Ensure we're strictly inside the ball
    norms = torch.norm(z_poincare, dim=-1, keepdim=True)
    z_poincare = torch.where(
        norms > max_radius,
        z_poincare * max_radius / (norms + 1e-8),
        z_poincare
    )

    return z_poincare


def main():
    parser = argparse.ArgumentParser(description='Extract hyperbolic embeddings')
    parser.add_argument('--checkpoint', type=str,
                       default='sandbox-training/checkpoints/v5_5/best.pt',
                       help='Path to checkpoint')
    parser.add_argument('--output', type=str,
                       default='riemann_hypothesis_sandbox/embeddings',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Paths
    checkpoint_path = PROJECT_ROOT / args.checkpoint
    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Device: {args.device}")

    # Generate all operations
    print("Generating all 19,683 ternary operations...")
    operations = generate_all_ternary_operations()
    operations = torch.tensor(operations, dtype=torch.float32)
    print(f"Operations shape: {operations.shape}")

    # Load checkpoint and detect type
    checkpoint, model_state = load_v5_5_checkpoint(checkpoint_path, args.device)

    # Check if this is a v5.11+ model with projection
    has_projection = any('projection' in k for k in model_state.keys())

    if has_projection:
        print("\nDetected v5.11+ model with hyperbolic projection")
        embeddings = extract_embeddings_v5_11(checkpoint_path, operations, args.device)
        z_A_hyp = embeddings['z_A_hyp']
        z_B_hyp = embeddings['z_B_hyp']
        z_A_euc = embeddings['z_A_euc']
        z_B_euc = embeddings['z_B_euc']
        z_hyp = z_A_hyp  # Default to A for compatibility
        z_euc = z_A_euc
        has_dual = True
    else:
        print("\nDetected v5.5 model (Euclidean only)")
        print("Will project to Poincaré ball manually")
        z_euc = extract_embeddings_v5_5(model_state, operations, args.device)
        z_hyp = project_to_poincare(z_euc, max_radius=0.95)
        z_A_hyp = z_B_hyp = z_hyp
        z_A_euc = z_B_euc = z_euc
        has_dual = False

    # Move to CPU for saving
    z_hyp = z_hyp.cpu()
    z_euc = z_euc.cpu()
    z_A_hyp = z_A_hyp.cpu()
    z_B_hyp = z_B_hyp.cpu()
    z_A_euc = z_A_euc.cpu()
    z_B_euc = z_B_euc.cpu()

    # Stats
    print(f"\n=== Embedding Statistics ===")
    print(f"Shape: {z_hyp.shape}")

    radii_A = torch.norm(z_A_hyp, dim=-1)
    radii_B = torch.norm(z_B_hyp, dim=-1)
    print(f"VAE-A radii: min={radii_A.min():.4f}, max={radii_A.max():.4f}, mean={radii_A.mean():.4f}")
    print(f"VAE-B radii: min={radii_B.min():.4f}, max={radii_B.max():.4f}, mean={radii_B.mean():.4f}")

    # Save embeddings
    output_file = output_dir / 'embeddings.pt'
    torch.save({
        'z_hyperbolic': z_hyp,  # Default (A) for compatibility
        'z_euclidean': z_euc,
        'z_A_hyp': z_A_hyp,
        'z_B_hyp': z_B_hyp,
        'z_A_euc': z_A_euc,
        'z_B_euc': z_B_euc,
        'operations': operations,
        'n_operations': len(operations),
        'latent_dim': z_hyp.shape[1],
        'checkpoint_path': str(checkpoint_path),
        'has_projection': has_projection,
        'has_dual': has_dual
    }, output_file)

    print(f"\nSaved embeddings to: {output_file}")

    # Also save as numpy for easy analysis
    np.save(output_dir / 'z_hyperbolic.npy', z_hyp.numpy())
    np.save(output_dir / 'z_euclidean.npy', z_euc.numpy())
    np.save(output_dir / 'z_A_hyp.npy', z_A_hyp.numpy())
    np.save(output_dir / 'z_B_hyp.npy', z_B_hyp.numpy())
    print(f"Saved numpy arrays to: {output_dir}")

    return z_hyp, z_euc


if __name__ == '__main__':
    main()
