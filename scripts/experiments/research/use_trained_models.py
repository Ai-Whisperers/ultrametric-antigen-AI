#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Research Application Guide: Using Trained Models.

This script demonstrates how to use the trained multitask disease model
and SOTA components for various research applications.

Applications Covered:
1. HIV drug resistance prediction
2. Viral escape prediction (EVEscape-style)
3. Cross-disease transfer learning
4. Few-shot adaptation to new variants
5. Codon optimization for therapeutics
6. Structure-conditioned sequence design

Usage:
    python scripts/research/use_trained_models.py --application hiv_resistance
    python scripts/research/use_trained_models.py --application escape_prediction
    python scripts/research/use_trained_models.py --application transfer_learning
    python scripts/research/use_trained_models.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Add project root
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

# Import components
from src.diseases import VariantEscapeHead, MetaLearningEscapeHead
from src.evaluation import ProteinGymEvaluator
from src.losses import CodonUsageLoss, CodonUsageConfig, CodonOptimalityScore, Organism
from src.contrastive import CodonPositiveSampler, CodonSamplerConfig
from src.encoders import HybridCodonEncoder, HybridEncoderConfig, PLMBackend


def load_multitask_model(checkpoint_path: Path, device: str = "cpu"):
    """Load trained multitask disease model.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, config)
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best validation loss: {checkpoint.get('val_loss', 'N/A')}")
        return checkpoint
    except Exception as e:
        print(f"Could not load full checkpoint: {e}")
        print("Loading state_dict only...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        return checkpoint


def demo_hiv_resistance_prediction():
    """Demo: HIV drug resistance prediction using trained model."""
    print("\n" + "=" * 70)
    print("  APPLICATION 1: HIV DRUG RESISTANCE PREDICTION")
    print("=" * 70)

    # Load model checkpoint
    checkpoint_path = PROJECT_ROOT / "sandbox-training/checkpoints/multitask_disease/best.pt"

    if checkpoint_path.exists():
        checkpoint = load_multitask_model(checkpoint_path)
        print("\nModel configuration:")
        if "config" in checkpoint:
            config = checkpoint["config"]
            print(f"  Diseases trained: {getattr(config, 'diseases', 'unknown')}")
            print(f"  Tasks: {getattr(config, 'tasks', 'unknown')}")
    else:
        print(f"\nCheckpoint not found at {checkpoint_path}")
        print("Run: python scripts/training/train_multitask_disease.py --diseases hiv")

    # Create VariantEscapeHead for drug resistance
    print("\nCreating VariantEscapeHead for HIV...")
    escape_head = VariantEscapeHead(
        latent_dim=256,
        hidden_dim=128,
        disease="hiv",
        n_drug_classes=6,  # NRTIs, NNRTIs, PIs, INIs, EIs, MIs
        n_antibody_classes=10,
        n_tcell_epitopes=20,
    )

    # Example: Predict resistance for a variant
    batch_size = 4
    latent_z = torch.randn(batch_size, 256)  # Simulated encoder output

    predictions = escape_head(latent_z, return_components=True)

    print("\nDrug resistance predictions:")
    print(f"  Overall resistance score: {predictions['drug_resistance'].mean():.3f}")
    print(f"  Per-drug resistance shape: {predictions['per_drug_resistance'].shape}")

    drug_classes = ["NRTIs", "NNRTIs", "PIs", "INIs", "EIs", "MIs"]
    for i, drug in enumerate(drug_classes):
        score = predictions["per_drug_resistance"][:, i].mean()
        print(f"    {drug}: {score:.3f}")

    return predictions


def demo_escape_prediction():
    """Demo: Viral escape prediction (EVEscape-inspired)."""
    print("\n" + "=" * 70)
    print("  APPLICATION 2: VIRAL ESCAPE PREDICTION (EVEscape-Style)")
    print("=" * 70)

    # Create escape head
    escape_head = VariantEscapeHead(
        latent_dim=64,
        hidden_dim=128,
        disease="hiv",
        n_drug_classes=6,
        n_antibody_classes=10,
        n_tcell_epitopes=20,
    )

    # Simulate variant embeddings
    n_variants = 10
    variant_embeddings = torch.randn(n_variants, 64)

    # Predict escape potential
    predictions = escape_head(variant_embeddings, return_components=True)

    print("\nEscape predictions for 10 simulated variants:")
    print(f"  Fitness scores: mean={predictions['fitness'].mean():.3f}, std={predictions['fitness'].std():.3f}")
    print(f"  Immune escape: mean={predictions['immune_escape'].mean():.3f}")
    print(f"  Antibody escape: shape={predictions['antibody_escape'].shape}")
    print(f"  T-cell escape: shape={predictions['tcell_escape'].shape}")

    # Rank variants by escape potential
    escape_scores = predictions["escape_score"].squeeze()
    top_variants = escape_scores.argsort(descending=True)[:3]
    print(f"\nTop 3 variants by escape potential: {top_variants.tolist()}")

    return predictions


def demo_transfer_learning():
    """Demo: Cross-disease transfer learning."""
    print("\n" + "=" * 70)
    print("  APPLICATION 3: CROSS-DISEASE TRANSFER LEARNING")
    print("=" * 70)

    print("\nUse Case: Transfer HIV model to SARS-CoV-2 variants")
    print("\nApproach:")
    print("  1. Load HIV-trained encoder from multitask checkpoint")
    print("  2. Freeze encoder layers")
    print("  3. Train new task heads for SARS-CoV-2")
    print("  4. Fine-tune with small learning rate")

    # Code example
    code = '''
# Load pretrained HIV model
checkpoint = torch.load("checkpoints/multitask_disease/best.pt")
encoder_state = checkpoint["model_state_dict"]

# Extract encoder weights
encoder_weights = {k: v for k, v in encoder_state.items() if k.startswith("encoder.")}

# Create new model for SARS-CoV-2
new_model = MultiTaskPredictor(config)
new_model.encoder.load_state_dict(encoder_weights, strict=False)

# Freeze encoder initially
for param in new_model.encoder.parameters():
    param.requires_grad = False

# Train only task heads on SARS-CoV-2 data
optimizer = AdamW(new_model.task_heads.parameters(), lr=1e-3)
    '''
    print(code)

    return None


def demo_few_shot_adaptation():
    """Demo: Few-shot adaptation to new variants."""
    print("\n" + "=" * 70)
    print("  APPLICATION 4: FEW-SHOT ADAPTATION TO NEW VARIANTS")
    print("=" * 70)

    print("\nUse Case: Rapidly adapt to emerging HIV variant with limited data")

    # Create MetaLearningEscapeHead
    meta_head = MetaLearningEscapeHead(
        latent_dim=64,
        hidden_dim=128,
    )

    # Simulate support set (5-shot learning)
    k_shot = 5
    support_embeddings = torch.randn(k_shot, 64)
    support_labels = torch.tensor([0, 0, 1, 1, 1])  # Binary escape labels

    # Simulate query set
    query_embeddings = torch.randn(10, 64)

    # Predict on query set
    predictions = meta_head(query_embeddings)

    print(f"\n5-shot adaptation results:")
    print(f"  Support set size: {k_shot}")
    print(f"  Query set size: 10")
    print(f"  Fitness predictions: {predictions['fitness'].squeeze().mean():.3f}")
    print(f"  Immune escape predictions: {predictions['immune_escape'].squeeze().mean():.3f}")

    print("\nBenefits of Meta-Learning:")
    print("  - Rapid adaptation to new variants")
    print("  - Works with limited labeled data")
    print("  - Preserves knowledge from related variants")

    return predictions


def demo_codon_optimization():
    """Demo: Codon optimization for therapeutics."""
    print("\n" + "=" * 70)
    print("  APPLICATION 5: CODON OPTIMIZATION FOR THERAPEUTICS")
    print("=" * 70)

    print("\nUse Case: Optimize mRNA vaccine codons for human expression")

    # Create codon usage loss for human optimization
    config = CodonUsageConfig(
        organism=Organism.HUMAN,
        tai_weight=0.3,
        cai_weight=0.3,
        rare_penalty_weight=0.2,
        cpg_penalty_weight=0.1,
        gc_weight=0.1,
    )

    codon_loss = CodonUsageLoss(weight=0.1, config=config)
    scorer = CodonOptimalityScore(organism=Organism.HUMAN)

    # Example: Score wild-type vs optimized codons
    # Simulate some codon sequences
    wt_codons = torch.randint(0, 64, (1, 100))  # Wild-type
    opt_codons = torch.randint(0, 64, (1, 100))  # "Optimized"

    # Score both
    wt_scores = scorer(wt_codons)
    opt_scores = scorer(opt_codons)

    print("\nCodon optimization scores:")
    print(f"  Wild-type tAI: {wt_scores['tai'].mean():.4f}")
    print(f"  Optimized tAI: {opt_scores['tai'].mean():.4f}")
    print(f"  Wild-type CAI: {wt_scores['cai'].mean():.4f}")
    print(f"  Optimized CAI: {opt_scores['cai'].mean():.4f}")

    print("\nOptimization targets:")
    print("  - Maximize tAI (tRNA adaptation index)")
    print("  - Maximize CAI (Codon adaptation index)")
    print("  - Minimize CpG dinucleotides (reduce immunogenicity)")
    print("  - Target 40-60% GC content")
    print("  - Avoid rare codons")

    return wt_scores, opt_scores


def demo_structure_conditioned():
    """Demo: Structure-conditioned sequence design."""
    print("\n" + "=" * 70)
    print("  APPLICATION 6: STRUCTURE-CONDITIONED SEQUENCE DESIGN")
    print("=" * 70)

    print("\nUse Case: Design codon sequences compatible with target structure")

    print("\nWorkflow:")
    print("  1. Load AlphaFold3 structure predictions from research/")
    print("  2. Extract backbone coordinates")
    print("  3. Use structure-conditioned diffusion model")
    print("  4. Generate compatible codon sequences")
    print("  5. Evaluate with ProteinGym metrics")

    # Show available structure data
    structure_dir = PROJECT_ROOT / "research/bioinformatics/codon_encoder_research/hiv/data/structures"
    if structure_dir.exists():
        variants = [d.name for d in structure_dir.iterdir() if d.is_dir()][:5]
        print(f"\nAvailable HIV structure variants:")
        for v in variants:
            print(f"    - {v}")

    print("\nRun structure-conditioned training:")
    print("  python scripts/training/train_diffusion_codon.py --disease hiv --structure-conditioned")

    return None


def demo_proteingym_evaluation():
    """Demo: Comprehensive evaluation with ProteinGym metrics."""
    print("\n" + "=" * 70)
    print("  APPLICATION 7: PROTEINGYM-STYLE COMPREHENSIVE EVALUATION")
    print("=" * 70)

    # Generate some sequences
    n_train = 100
    n_generated = 50
    seq_len = 100

    training_seqs = torch.randint(0, 64, (n_train, seq_len))
    generated_seqs = torch.randint(0, 64, (n_generated, seq_len))

    # Create evaluator
    evaluator = ProteinGymEvaluator(
        training_sequences=training_seqs,
        organism=Organism.HUMAN,
    )

    # Evaluate
    metrics = evaluator.evaluate(generated_seqs)

    print("\nProteinGym-style evaluation results:")
    print(f"\n  QUALITY METRICS:")
    print(f"    Mean tAI: {metrics.quality.mean_tai:.4f}")
    print(f"    Mean CAI: {metrics.quality.mean_cai:.4f}")

    print(f"\n  NOVELTY METRICS:")
    print(f"    Unique fraction: {metrics.novelty.unique_fraction:.4f}")
    print(f"    Novel fraction: {metrics.novelty.novel_fraction:.4f}")

    print(f"\n  DIVERSITY METRICS:")
    print(f"    Mean pairwise distance: {metrics.diversity.mean_pairwise_distance:.4f}")
    print(f"    Codon coverage: {metrics.diversity.codon_coverage:.4f}")
    print(f"    Amino acid coverage: {metrics.diversity.amino_acid_coverage:.4f}")

    print(f"\n  BIOLOGICAL VALIDITY:")
    print(f"    No internal stops: {metrics.validity.no_stop_codons:.4f}")
    print(f"    Valid start codon: {metrics.validity.valid_start_codon:.4f}")

    return metrics


def list_available_data():
    """List available research data."""
    print("\n" + "=" * 70)
    print("  AVAILABLE RESEARCH DATA")
    print("=" * 70)

    research_base = PROJECT_ROOT / "research/bioinformatics/codon_encoder_research"

    diseases = {
        "hiv": "HIV/AIDS - Drug resistance, glycan shield, immune escape",
        "rheumatoid_arthritis": "RA - Citrullination, HLA interactions, autoimmunity",
        "neurodegeneration": "Alzheimer's - Tau phosphorylation, aggregation",
        "sars_cov_2": "COVID-19 - Spike glycan shield, variant analysis",
    }

    for disease, description in diseases.items():
        disease_dir = research_base / disease
        if disease_dir.exists():
            print(f"\n  {disease.upper()}")
            print(f"    {description}")

            # Count files
            data_dir = disease_dir / "data"
            scripts_dir = disease_dir / "scripts"

            if data_dir.exists():
                n_files = sum(1 for _ in data_dir.rglob("*") if _.is_file())
                print(f"    Data files: {n_files}")

            if scripts_dir.exists():
                scripts = list(scripts_dir.glob("*.py"))
                print(f"    Analysis scripts: {len(scripts)}")

    # Checkpoints
    print("\n  TRAINED CHECKPOINTS:")
    checkpoint_dir = PROJECT_ROOT / "sandbox-training/checkpoints"
    if checkpoint_dir.exists():
        for d in checkpoint_dir.iterdir():
            if d.is_dir():
                checkpoints = list(d.glob("*.pt"))
                print(f"    {d.name}: {len(checkpoints)} checkpoint(s)")


def main():
    parser = argparse.ArgumentParser(description="Research Applications Guide")
    parser.add_argument(
        "--application",
        type=str,
        choices=[
            "hiv_resistance",
            "escape_prediction",
            "transfer_learning",
            "few_shot",
            "codon_optimization",
            "structure_conditioned",
            "evaluation",
            "list_data",
        ],
        help="Application to demonstrate",
    )
    parser.add_argument("--all", action="store_true", help="Run all demos")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  RESEARCH APPLICATIONS: USING TRAINED MODELS")
    print("  Based on SOTA Bioinformatics (2024-2025)")
    print("=" * 70)

    if args.all or args.application is None:
        list_available_data()
        demo_hiv_resistance_prediction()
        demo_escape_prediction()
        demo_transfer_learning()
        demo_few_shot_adaptation()
        demo_codon_optimization()
        demo_structure_conditioned()
        demo_proteingym_evaluation()
    elif args.application == "hiv_resistance":
        demo_hiv_resistance_prediction()
    elif args.application == "escape_prediction":
        demo_escape_prediction()
    elif args.application == "transfer_learning":
        demo_transfer_learning()
    elif args.application == "few_shot":
        demo_few_shot_adaptation()
    elif args.application == "codon_optimization":
        demo_codon_optimization()
    elif args.application == "structure_conditioned":
        demo_structure_conditioned()
    elif args.application == "evaluation":
        demo_proteingym_evaluation()
    elif args.application == "list_data":
        list_available_data()

    print("\n" + "=" * 70)
    print("  NEXT STEPS")
    print("=" * 70)
    print("""
  1. Train with real data:
     python scripts/training/train_multitask_disease.py --diseases hiv cancer ra --use-escape-head --evaluate

  2. Run meta-learning for variant adaptation:
     python scripts/training/train_meta_learning.py --disease hiv --use-escape-head --evaluate

  3. Generate optimized sequences:
     python scripts/training/train_diffusion_codon.py --disease hiv --use-codon-loss --evaluate

  4. Explore research analysis scripts:
     research/bioinformatics/codon_encoder_research/hiv/scripts/
     research/bioinformatics/codon_encoder_research/rheumatoid_arthritis/scripts/
    """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
