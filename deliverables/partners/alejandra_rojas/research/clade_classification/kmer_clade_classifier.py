# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DENV-4 Clade Classification using K-mer Features.

Simplified approach for highly imbalanced data:
1. Extract k-mer frequency vectors (k=6)
2. Train RandomForest with class balancing
3. Stratified cross-validation

This provides a baseline before attempting neural approaches.

Usage:
    python kmer_clade_classifier.py --k 6 --folds 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[5]
ROJAS_DIR = PROJECT_ROOT / "deliverables" / "partners" / "alejandra_rojas"
ML_READY_DIR = ROJAS_DIR / "results" / "ml_ready"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_denv4_data() -> tuple[list[str], list[str], list[str]]:
    """Load DENV-4 sequences and clade assignments."""
    with open(ML_READY_DIR / "denv4_genome_metadata.json") as f:
        metadata = json.load(f)

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


def generate_kmers(k: int = 6) -> list[str]:
    """Generate all possible k-mers."""
    bases = ['A', 'C', 'G', 'T']
    if k == 1:
        return bases
    kmers = []
    for b in bases:
        for suffix in generate_kmers(k - 1):
            kmers.append(b + suffix)
    return kmers


def sequence_to_kmer_vector(sequence: str, k: int = 6) -> np.ndarray:
    """Convert sequence to k-mer frequency vector.

    Args:
        sequence: Nucleotide sequence
        k: k-mer length

    Returns:
        Normalized k-mer frequency vector (4^k dimensions)
    """
    sequence = sequence.upper().replace('U', 'T')
    valid_bases = set('ACGT')

    # Count k-mers
    kmer_counts = Counter()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if all(b in valid_bases for b in kmer):
            kmer_counts[kmer] += 1

    # Create vector
    all_kmers = generate_kmers(k)
    kmer_to_idx = {km: i for i, km in enumerate(all_kmers)}
    vector = np.zeros(len(all_kmers), dtype=np.float32)

    for kmer, count in kmer_counts.items():
        if kmer in kmer_to_idx:
            vector[kmer_to_idx[kmer]] = count

    # Normalize
    total = vector.sum()
    if total > 0:
        vector = vector / total

    return vector


def compute_padic_kmer_features(sequence: str, k: int = 3) -> np.ndarray:
    """Compute p-adic inspired codon features.

    For each position, compute codon-level features:
    - Codon class (synonymous groups)
    - Position-wise entropy proxy

    Args:
        sequence: Nucleotide sequence
        k: Codon length (always 3)

    Returns:
        Feature vector capturing codon-level patterns
    """
    from itertools import product

    sequence = sequence.upper().replace('U', 'T')

    # Define codon to amino acid mapping
    genetic_code = {
        'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
        'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
        'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
        'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
        'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
        'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
        'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
        'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
        'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
        'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
        'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
        'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
        'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    }

    # Count codon usage
    codon_counts = Counter()
    aa_counts = Counter()
    wobble_base_counts = Counter()  # 3rd position

    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3]
        if codon in genetic_code:
            codon_counts[codon] += 1
            aa_counts[genetic_code[codon]] += 1
            wobble_base_counts[codon[2]] += 1  # 3rd position

    # Features:
    # 1. Codon usage frequencies (64 dim)
    all_codons = list(genetic_code.keys())
    codon_freq = np.array([codon_counts.get(c, 0) for c in all_codons], dtype=np.float32)
    total = codon_freq.sum()
    if total > 0:
        codon_freq = codon_freq / total

    # 2. Codon usage bias per amino acid (compute RSCU-like metric)
    rscu_features = []
    for aa in sorted(set(genetic_code.values())):
        if aa == '*':
            continue
        synonymous = [c for c, a in genetic_code.items() if a == aa]
        observed = [codon_counts.get(c, 0) for c in synonymous]
        total_aa = sum(observed)
        if total_aa > 0:
            # Compute relative synonymous codon usage
            expected = total_aa / len(synonymous)
            rscu = [o / expected if expected > 0 else 0 for o in observed]
            rscu_features.extend(rscu)
        else:
            rscu_features.extend([0] * len(synonymous))

    # 3. Wobble position bias (4 dim)
    wobble_freq = np.array([wobble_base_counts.get(b, 0) for b in 'ACGT'], dtype=np.float32)
    wobble_total = wobble_freq.sum()
    if wobble_total > 0:
        wobble_freq = wobble_freq / wobble_total

    # Combine features
    features = np.concatenate([codon_freq, np.array(rscu_features), wobble_freq])
    return features


def main():
    parser = argparse.ArgumentParser(description="K-mer based clade classifier")
    parser.add_argument("--k", type=int, default=6, help="K-mer length")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--include_padic", action="store_true", help="Include p-adic codon features")
    args = parser.parse_args()

    print("=" * 70)
    print("DENV-4 CLADE CLASSIFICATION - K-MER BASELINE")
    print("=" * 70)
    print(f"K-mer length: {args.k}")
    print(f"Folds: {args.folds}")
    print(f"Include p-adic features: {args.include_padic}")
    print()

    # Load data
    print("Loading DENV-4 data...")
    accessions, sequences, clades = load_denv4_data()
    print(f"Loaded {len(sequences)} sequences")

    # Encode labels
    clade_to_idx = {c: i for i, c in enumerate(sorted(set(clades)))}
    idx_to_clade = {i: c for c, i in clade_to_idx.items()}
    y = np.array([clade_to_idx[c] for c in clades])

    print(f"Clade mapping: {clade_to_idx}")
    print(f"Class distribution: {Counter(y)}")
    print()

    # Extract features
    print(f"Extracting {args.k}-mer features...")
    X_kmer = np.array([sequence_to_kmer_vector(s, k=args.k) for s in sequences])
    print(f"K-mer feature shape: {X_kmer.shape}")

    if args.include_padic:
        print("Extracting p-adic codon features...")
        X_padic = np.array([compute_padic_kmer_features(s) for s in sequences])
        print(f"P-adic feature shape: {X_padic.shape}")
        X = np.hstack([X_kmer, X_padic])
        print(f"Combined feature shape: {X.shape}")
    else:
        X = X_kmer

    # Train classifier
    print("\nTraining RandomForest classifier...")

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ))
    ])

    # Stratified cross-validation
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    # Cross-val predict
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    # Metrics
    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nOverall Accuracy: {acc:.4f}")
    print(f"Overall Balanced Accuracy: {bal_acc:.4f}")

    print("\nClassification Report:")
    target_names = [idx_to_clade[i] for i in range(len(clade_to_idx))]
    print(classification_report(y, y_pred, target_names=target_names, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(cm)

    # Per-clade accuracy
    print("\nPer-Clade Accuracy:")
    for i, clade in enumerate(target_names):
        mask = y == i
        if mask.sum() > 0:
            clade_acc = (y_pred[mask] == y[mask]).mean()
            print(f"  {clade}: {clade_acc:.3f} ({mask.sum()} samples)")

    # Feature importance (top 20)
    print("\nTraining final model for feature importance...")
    clf.fit(X, y)
    importances = clf.named_steps['clf'].feature_importances_

    # Map features to names
    feature_names = generate_kmers(args.k)
    if args.include_padic:
        # Add padic feature names
        genetic_code = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
        }
        feature_names.extend([f"codon_{c}" for c in genetic_code.keys()])
        # RSCU features
        for aa in sorted(set(genetic_code.values())):
            if aa != '*':
                synonymous = [c for c, a in genetic_code.items() if a == aa]
                feature_names.extend([f"rscu_{aa}_{c}" for c in synonymous])
        feature_names.extend(['wobble_A', 'wobble_C', 'wobble_G', 'wobble_T'])

    top_idx = np.argsort(importances)[::-1][:20]
    print("\nTop 20 Important Features:")
    for rank, idx in enumerate(top_idx, 1):
        if idx < len(feature_names):
            print(f"  {rank}. {feature_names[idx]}: {importances[idx]:.4f}")

    # Save results
    results = {
        "_metadata": {
            "analysis_type": "clade_classification_kmer",
            "description": "DENV-4 clade classification using k-mer features",
            "created": datetime.now(timezone.utc).isoformat(),
            "parameters": {
                "k": args.k,
                "folds": args.folds,
                "include_padic": args.include_padic,
                "classifier": "RandomForest",
            },
        },
        "summary": {
            "overall_accuracy": float(acc),
            "overall_balanced_accuracy": float(bal_acc),
            "n_samples": len(y),
            "n_features": X.shape[1],
            "n_classes": len(clade_to_idx),
        },
        "clade_mapping": clade_to_idx,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y, y_pred, target_names=target_names, output_dict=True, zero_division=0
        ),
        "top_features": [
            {"rank": rank, "feature": feature_names[idx] if idx < len(feature_names) else f"feature_{idx}",
             "importance": float(importances[idx])}
            for rank, idx in enumerate(top_idx[:20], 1)
        ],
        "predictions": {
            "labels": y.tolist(),
            "predictions": y_pred.tolist(),
        },
    }

    suffix = "_padic" if args.include_padic else ""
    results_path = RESULTS_DIR / f"kmer{args.k}_classification_results{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    main()
