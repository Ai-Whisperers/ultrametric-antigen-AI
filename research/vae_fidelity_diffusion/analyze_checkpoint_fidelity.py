#!/usr/bin/env python3
"""Analyze VAE Checkpoint Fidelity with Diffusion Model.

This script demonstrates how to use the VAE Fidelity Diffusion model to:
1. Analyze existing VAE checkpoints for fidelity loss patterns
2. Identify specific failure modes (reconstruction, geometry, objectives)
3. Apply targeted refinement to improve representation quality
4. Generate detailed analysis reports

Usage:
    python analyze_checkpoint_fidelity.py --checkpoint path/to/best.pt
    python analyze_checkpoint_fidelity.py --checkpoint v5_11_3 --synthetic
    python analyze_checkpoint_fidelity.py --compare-all
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from vae_fidelity_diffusion import (
    VAEFidelityDiffusion,
    analyze_vae_checkpoint_fidelity
)
from src.core.ternary import TernarySpace
from src.geometry import poincare_distance, project_to_poincare


# Known checkpoint paths from exploration
AVAILABLE_CHECKPOINTS = {
    'v5_11_3': 'research/bioinformatics/genetic_code/data/v5_11_3_embeddings.pt',
    'codon_encoder_3adic': 'research/bioinformatics/genetic_code/data/codon_encoder_3adic.pt',
    'fused_embeddings': 'research/bioinformatics/genetic_code/data/fused_embeddings.pt',
    'v5_11_structural': 'research/contact-prediction/checkpoints/v5_11_structural_best.pt',
    'homeostatic_rich': 'research/contact-prediction/checkpoints/homeostatic_rich_best.pt',
    'final_rich_lr5e5': 'research/contact-prediction/checkpoints/final_rich_lr5e5_best.pt',
}


def create_synthetic_embeddings(embedding_type: str = 'balanced',
                               device: str = 'cpu') -> torch.Tensor:
    """Create synthetic embeddings with known fidelity patterns.

    Args:
        embedding_type: Type of synthetic embedding
            'balanced': Good hierarchy and richness
            'collapsed': Hierarchy but no richness (shells)
            'chaotic': Richness but no hierarchy
            'compressed': KL regularization artifacts
            'conflicted': Multi-objective competition artifacts

    Returns:
        Synthetic embeddings (64, 16) on Poincaré ball
    """
    ternary = TernarySpace()

    if embedding_type == 'balanced':
        # Good hierarchy with moderate richness
        embeddings = torch.randn(64, 16, device=device) * 0.3
        for i in range(64):
            # Set radial position by p-adic valuation with some variance
            valuation = ternary.valuation(torch.tensor([i])).item()
            target_radius = 0.9 - valuation * 0.08  # v0→0.9, v9→0.18
            target_radius += torch.randn(1).item() * 0.05  # Add richness
            target_radius = max(0.05, min(0.95, target_radius))

            # Scale embedding to target radius
            current_norm = torch.norm(embeddings[i])
            if current_norm > 0:
                embeddings[i] = embeddings[i] / current_norm * target_radius

    elif embedding_type == 'collapsed':
        # Perfect hierarchy, no richness (collapsed shells)
        embeddings = torch.randn(64, 16, device=device) * 0.1
        for i in range(64):
            valuation = ternary.valuation(torch.tensor([i])).item()
            target_radius = 0.9 - valuation * 0.08
            # No additional variance - perfect shells
            current_norm = torch.norm(embeddings[i])
            if current_norm > 0:
                embeddings[i] = embeddings[i] / current_norm * target_radius

    elif embedding_type == 'chaotic':
        # No hierarchy, high richness
        embeddings = torch.randn(64, 16, device=device) * 0.6
        # Random radii, no correlation with p-adic valuation

    elif embedding_type == 'compressed':
        # KL artifacts: over-concentrated near origin
        embeddings = torch.randn(64, 16, device=device) * 0.15
        # Most points very close to origin

    elif embedding_type == 'conflicted':
        # Multi-objective conflicts: some very inner, some very outer
        embeddings = torch.randn(64, 16, device=device) * 0.2
        for i in range(64):
            if i % 2 == 0:
                # Push to boundary
                embeddings[i] = embeddings[i] / torch.norm(embeddings[i]) * 0.95
            else:
                # Keep near origin
                embeddings[i] = embeddings[i] * 0.1

    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

    # Ensure all embeddings are on Poincaré ball
    return project_to_poincare(embeddings, c=1.0)


def compute_traditional_metrics(embeddings: torch.Tensor) -> Dict[str, float]:
    """Compute traditional VAE evaluation metrics.

    Args:
        embeddings: Codon embeddings (64, latent_dim)

    Returns:
        Dictionary with hierarchy correlation, richness, etc.
    """
    ternary = TernarySpace()

    # Compute p-adic valuations for all 64 codons
    indices = torch.arange(64, device=embeddings.device)
    valuations = ternary.valuation(indices).float()

    # Compute hyperbolic radii (proper geometry)
    origin = torch.zeros_like(embeddings)
    radii = poincare_distance(embeddings, origin, c=1.0)

    # Hierarchy correlation (Spearman)
    hierarchy_corr, _ = spearmanr(valuations.cpu().numpy(), radii.cpu().numpy())

    # Richness (radial variance)
    richness = torch.var(radii).item()

    # Coverage (mock - would need actual reconstruction)
    coverage = 1.0  # Assume perfect for analysis

    return {
        'hierarchy_correlation': float(hierarchy_corr),
        'richness_variance': richness,
        'coverage': coverage,
        'mean_radius': radii.mean().item(),
        'radius_std': radii.std().item()
    }


def analyze_single_checkpoint(checkpoint_path: str,
                            device: str = 'cpu',
                            save_report: bool = True) -> Dict:
    """Comprehensive analysis of single checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Computation device
        save_report: Whether to save detailed report

    Returns:
        Complete analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    try:
        # Run fidelity analysis
        results = analyze_vae_checkpoint_fidelity(checkpoint_path, device)

        if 'error' in results:
            print(f"Error: {results['error']}")
            return results

        # Compute traditional metrics
        embeddings = results['refined_embeddings']  # Use refined for comparison
        traditional_metrics = compute_traditional_metrics(embeddings)

        # Combine results
        analysis_results = {
            'checkpoint_path': checkpoint_path,
            'traditional_metrics': traditional_metrics,
            'fidelity_analysis': results,
            'summary': {
                'total_fidelity_loss': results['original_fidelity']['total_fidelity_loss'].mean().item(),
                'improvement_achieved': (
                    results['original_fidelity']['total_fidelity_loss'].mean() -
                    results['refined_fidelity']['total_fidelity_loss'].mean()
                ).item(),
                'hierarchy_correlation': traditional_metrics['hierarchy_correlation'],
                'richness_variance': traditional_metrics['richness_variance']
            }
        }

        # Print summary
        print(f"\nTraditional Metrics:")
        print(f"  Hierarchy Correlation: {traditional_metrics['hierarchy_correlation']:.4f}")
        print(f"  Richness Variance: {traditional_metrics['richness_variance']:.6f}")
        print(f"  Mean Radius: {traditional_metrics['mean_radius']:.4f}")

        print(f"\nFidelity Analysis:")
        orig_fidelity = results['original_fidelity']
        for key in ['reconstruction', 'radial_richness', 'metric_consistency',
                   'kl_artifacts', 'objective_competition']:
            if key in orig_fidelity:
                print(f"  {key.replace('_', ' ').title()}: {orig_fidelity[key].mean():.4f}")

        print(f"\nTotal Fidelity Loss: {analysis_results['summary']['total_fidelity_loss']:.4f}")
        print(f"Improvement Achieved: {analysis_results['summary']['improvement_achieved']:.4f}")

        # Save detailed report
        if save_report:
            output_path = Path(checkpoint_path).stem + '_fidelity_report.json'
            output_dir = Path(__file__).parent / 'reports'
            output_dir.mkdir(exist_ok=True)

            # Convert tensors to lists for JSON serialization
            json_results = {}
            for key, value in analysis_results.items():
                if key == 'fidelity_analysis':
                    # Skip complex nested tensors for now
                    json_results[key] = {'summary': 'detailed_tensors_omitted'}
                else:
                    json_results[key] = value

            with open(output_dir / output_path, 'w') as f:
                json.dump(json_results, f, indent=2)

            print(f"\nDetailed report saved: {output_dir / output_path}")

        return analysis_results

    except Exception as e:
        print(f"Error analyzing checkpoint: {e}")
        return {'error': str(e)}


def compare_synthetic_embeddings(device: str = 'cpu') -> Dict:
    """Compare fidelity analysis across different synthetic embedding types."""
    print(f"\n{'='*60}")
    print("Comparing Synthetic Embedding Types")
    print(f"{'='*60}")

    embedding_types = ['balanced', 'collapsed', 'chaotic', 'compressed', 'conflicted']
    results = {}

    diffusion = VAEFidelityDiffusion(latent_dim=16, hidden_dim=64).to(device)

    for emb_type in embedding_types:
        print(f"\nAnalyzing {emb_type} embeddings...")

        # Create synthetic embeddings
        embeddings = create_synthetic_embeddings(emb_type, device)

        # Run fidelity analysis
        fidelity_scores = diffusion.map_fidelity_loss(embeddings)
        traditional_metrics = compute_traditional_metrics(embeddings)

        results[emb_type] = {
            'traditional_metrics': traditional_metrics,
            'fidelity_scores': {
                key: scores.mean().item()
                for key, scores in fidelity_scores.items()
            }
        }

        # Print summary
        print(f"  Hierarchy: {traditional_metrics['hierarchy_correlation']:.4f}")
        print(f"  Richness: {traditional_metrics['richness_variance']:.6f}")
        print(f"  Fidelity Loss: {fidelity_scores['total_fidelity_loss'].mean():.4f}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("Synthetic Embedding Comparison")
    print(f"{'='*80}")
    print(f"{'Type':<12} {'Hierarchy':<10} {'Richness':<10} {'Fidelity':<10} {'Pattern'}")
    print(f"{'-'*80}")

    for emb_type in embedding_types:
        r = results[emb_type]
        hierarchy = r['traditional_metrics']['hierarchy_correlation']
        richness = r['traditional_metrics']['richness_variance']
        fidelity = r['fidelity_scores']['total_fidelity_loss']

        # Classify pattern
        if hierarchy < -0.7 and richness > 0.005:
            pattern = "✓ Good"
        elif hierarchy < -0.7 and richness < 0.003:
            pattern = "⚠ Collapsed"
        elif hierarchy > -0.3 and richness > 0.01:
            pattern = "⚠ Chaotic"
        elif fidelity > 0.6:
            pattern = "✗ Poor"
        else:
            pattern = "~ Mixed"

        print(f"{emb_type:<12} {hierarchy:<10.3f} {richness:<10.6f} {fidelity:<10.3f} {pattern}")

    return results


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(description='VAE Checkpoint Fidelity Analysis')
    parser.add_argument('--checkpoint', type=str,
                       help='Checkpoint path or name from AVAILABLE_CHECKPOINTS')
    parser.add_argument('--synthetic', action='store_true',
                       help='Analyze synthetic embeddings instead')
    parser.add_argument('--compare-all', action='store_true',
                       help='Compare all available checkpoints')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Computation device')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving detailed reports')

    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    print("VAE Fidelity Diffusion Analysis")
    print(f"Device: {device}")

    if args.synthetic:
        # Analyze synthetic embeddings
        compare_synthetic_embeddings(device)

    elif args.compare_all:
        # Compare all available checkpoints
        print(f"\n{'='*60}")
        print("Comparing All Available Checkpoints")
        print(f"{'='*60}")

        checkpoint_results = {}
        for name, path in AVAILABLE_CHECKPOINTS.items():
            full_path = Path(PROJECT_ROOT) / path
            if full_path.exists():
                try:
                    result = analyze_single_checkpoint(str(full_path), device, not args.no_save)
                    checkpoint_results[name] = result
                except Exception as e:
                    print(f"Error analyzing {name}: {e}")
            else:
                print(f"Checkpoint not found: {full_path}")

        # Summary comparison
        if checkpoint_results:
            print(f"\n{'='*80}")
            print("Checkpoint Comparison Summary")
            print(f"{'='*80}")
            print(f"{'Checkpoint':<20} {'Hierarchy':<10} {'Richness':<10} {'Fidelity':<10}")
            print(f"{'-'*80}")

            for name, result in checkpoint_results.items():
                if 'summary' in result:
                    s = result['summary']
                    print(f"{name:<20} {s['hierarchy_correlation']:<10.3f} "
                          f"{s['richness_variance']:<10.6f} {s['total_fidelity_loss']:<10.3f}")

    elif args.checkpoint:
        # Analyze single checkpoint
        if args.checkpoint in AVAILABLE_CHECKPOINTS:
            checkpoint_path = Path(PROJECT_ROOT) / AVAILABLE_CHECKPOINTS[args.checkpoint]
        else:
            checkpoint_path = Path(args.checkpoint)

        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            print(f"Available checkpoints: {list(AVAILABLE_CHECKPOINTS.keys())}")
            return

        analyze_single_checkpoint(str(checkpoint_path), device, not args.no_save)

    else:
        # Show help and available options
        print("\nAvailable analysis options:")
        print("1. Analyze specific checkpoint: --checkpoint path/to/checkpoint.pt")
        print("2. Analyze synthetic patterns: --synthetic")
        print("3. Compare all checkpoints: --compare-all")
        print(f"\nAvailable named checkpoints: {list(AVAILABLE_CHECKPOINTS.keys())}")


if __name__ == '__main__':
    main()