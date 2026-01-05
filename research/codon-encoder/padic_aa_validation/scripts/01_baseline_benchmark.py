#!/usr/bin/env python3
"""Standardized Benchmark Suite for P-adic Amino Acid Encoders.

This module provides a comprehensive, future-proof testing framework for:
1. Single-prime p-adic encoders (p=2,3,5,7,11,...)
2. Multi-prime ensemble approaches
3. Baseline comparisons (random, physicochemical-only)
4. Proteomics-only scenarios (no codon information)

Metrics computed:
- Spearman correlation (rank-based, robust to outliers)
- Pearson correlation (linear relationship)
- MAE (mean absolute error)
- RMSE (root mean squared error)
- R² (coefficient of determination)

Usage:
    python padic_encoder_benchmark.py --all-tests
    python padic_encoder_benchmark.py --test ensemble
    python padic_encoder_benchmark.py --test baselines
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.encoders.codon_encoder import AA_PROPERTIES
from src.geometry import poincare_distance


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class MetricResult:
    """Single metric result with confidence interval."""
    value: float
    std: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0

    def __str__(self):
        if self.std > 0:
            return f"{self.value:.4f} ± {self.std:.4f}"
        return f"{self.value:.4f}"


@dataclass
class BenchmarkResult:
    """Complete benchmark result for one model."""
    model_name: str
    spearman: MetricResult
    pearson: MetricResult
    mae: MetricResult
    rmse: MetricResult
    r2: MetricResult
    n_samples: int
    n_features: int
    training_time: float = 0.0
    inference_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'spearman': self.spearman.value,
            'spearman_std': self.spearman.std,
            'pearson': self.pearson.value,
            'pearson_std': self.pearson.std,
            'mae': self.mae.value,
            'mae_std': self.mae.std,
            'rmse': self.rmse.value,
            'rmse_std': self.rmse.std,
            'r2': self.r2.value,
            'r2_std': self.r2.std,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'training_time': self.training_time,
            'metadata': self.metadata,
        }


# =============================================================================
# Amino Acid Encoder (from training script)
# =============================================================================

AA_TO_IDX = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
}


def compute_padic_valuation(n: int, p: int) -> int:
    """Compute p-adic valuation v_p(n)."""
    if n == 0:
        return p
    if n < 0:
        n = abs(n)
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


class AminoAcidPadicEncoder(nn.Module):
    """Amino acid encoder using p-adic structure."""

    def __init__(self, prime: int = 5, latent_dim: int = 16, hidden_dim: int = 64,
                 curvature: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.prime = prime
        self.latent_dim = latent_dim
        self.curvature = curvature
        self.n_aa = 20

        self.aa_embedding = nn.Parameter(torch.randn(self.n_aa, latent_dim) * 0.01)

        self.property_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        from src.geometry import exp_map_zero, project_to_poincare
        self.exp_map_zero = exp_map_zero
        self.project_to_poincare = project_to_poincare

        self._setup_hierarchy()
        self._setup_properties()

    def _setup_hierarchy(self):
        hierarchy = [min(compute_padic_valuation(i, self.prime), 4) for i in range(self.n_aa)]
        self.register_buffer('hierarchy_levels', torch.tensor(hierarchy, dtype=torch.long))
        max_level = max(hierarchy) if max(hierarchy) > 0 else 1
        target_radii = [0.85 - (h / max_level) * 0.6 for h in hierarchy]
        self.register_buffer('target_radii', torch.tensor(target_radii, dtype=torch.float32))

    def _setup_properties(self):
        props = [list(AA_PROPERTIES.get(aa, (0, 0, 0, 0))) for aa in 'ACDEFGHIKLMNPQRSTVWY']
        self.register_buffer('aa_properties', torch.tensor(props, dtype=torch.float32))

    def forward(self, aa_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = aa_indices.shape[0]
        z_base = self.aa_embedding[aa_indices]
        props = self.aa_properties[aa_indices]
        z_prop = self.property_encoder(props)
        z_fused = self.fusion(torch.cat([z_base, z_prop], dim=-1))
        z_hyp = self.exp_map_zero(z_fused, c=self.curvature)
        z_hyp = self.project_to_poincare(z_hyp, max_norm=0.95, c=self.curvature)
        origin = torch.zeros_like(z_hyp)
        radii = poincare_distance(z_hyp, origin, c=self.curvature)
        return {'z_hyp': z_hyp, 'radii': radii}

    def get_aa_embeddings(self) -> Dict[str, torch.Tensor]:
        indices = torch.arange(self.n_aa, device=self.aa_embedding.device)
        out = self.forward(indices)
        return {aa: out['z_hyp'][i] for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}


# =============================================================================
# Feature Extractors
# =============================================================================

class FeatureExtractor:
    """Base class for feature extraction."""

    def __init__(self, name: str):
        self.name = name

    def extract(self, wt: str, mt: str) -> np.ndarray:
        raise NotImplementedError

    def feature_names(self) -> List[str]:
        raise NotImplementedError


class PhysicochemicalFeatures(FeatureExtractor):
    """Physicochemical property features only (baseline)."""

    def __init__(self):
        super().__init__("physicochemical")

    def extract(self, wt: str, mt: str) -> np.ndarray:
        wt_props = np.array(AA_PROPERTIES.get(wt, (0, 0, 0, 0)))
        mt_props = np.array(AA_PROPERTIES.get(mt, (0, 0, 0, 0)))

        delta = mt_props - wt_props
        return np.concatenate([
            delta,  # Delta properties
            [np.abs(delta[1])],  # Abs charge change
            [np.linalg.norm(delta)],  # L2 distance
        ])

    def feature_names(self) -> List[str]:
        return ['delta_hydro', 'delta_charge', 'delta_volume', 'delta_polarity',
                'abs_charge_change', 'property_distance']


class RandomFeatures(FeatureExtractor):
    """Random features (negative control)."""

    def __init__(self, n_features: int = 6, seed: int = 42):
        super().__init__("random")
        self.n_features = n_features
        self.rng = np.random.RandomState(seed)

    def extract(self, wt: str, mt: str) -> np.ndarray:
        return self.rng.randn(self.n_features)

    def feature_names(self) -> List[str]:
        return [f'random_{i}' for i in range(self.n_features)]


class PadicEncoderFeatures(FeatureExtractor):
    """Features from trained p-adic encoder."""

    def __init__(self, encoder: AminoAcidPadicEncoder, prime: int):
        super().__init__(f"padic_p{prime}")
        self.encoder = encoder
        self.prime = prime
        self.encoder.eval()
        self._cache_embeddings()

    def _cache_embeddings(self):
        with torch.no_grad():
            self.embeddings = self.encoder.get_aa_embeddings()

    def extract(self, wt: str, mt: str) -> np.ndarray:
        if wt not in self.embeddings or mt not in self.embeddings:
            return np.zeros(6)

        wt_emb = self.embeddings[wt].cpu().numpy()
        mt_emb = self.embeddings[mt].cpu().numpy()

        # Hyperbolic distance
        wt_t = torch.tensor(wt_emb).unsqueeze(0)
        mt_t = torch.tensor(mt_emb).unsqueeze(0)
        hyp_dist = poincare_distance(wt_t, mt_t, c=self.encoder.curvature).item()

        # Radii
        origin = torch.zeros(1, self.encoder.latent_dim)
        wt_radius = poincare_distance(wt_t, origin, c=self.encoder.curvature).item()
        mt_radius = poincare_distance(mt_t, origin, c=self.encoder.curvature).item()

        # Physicochemical (always include as baseline)
        wt_props = AA_PROPERTIES.get(wt, (0, 0, 0, 0))
        mt_props = AA_PROPERTIES.get(mt, (0, 0, 0, 0))

        return np.array([
            hyp_dist,
            mt_radius - wt_radius,
            mt_props[0] - wt_props[0],  # delta hydro
            abs(mt_props[1] - wt_props[1]),  # abs delta charge
            mt_props[2] - wt_props[2] if len(mt_props) > 2 else 0,  # delta volume
            np.linalg.norm(np.array(mt_props) - np.array(wt_props)),  # prop dist
        ])

    def feature_names(self) -> List[str]:
        return ['hyp_dist', 'delta_radius', 'delta_hydro', 'abs_delta_charge',
                'delta_volume', 'property_distance']


class EnsembleFeatures(FeatureExtractor):
    """Ensemble features from multiple primes."""

    def __init__(self, encoders: Dict[int, AminoAcidPadicEncoder]):
        super().__init__("ensemble")
        self.encoders = encoders
        self.extractors = {
            p: PadicEncoderFeatures(enc, p) for p, enc in encoders.items()
        }

    def extract(self, wt: str, mt: str) -> np.ndarray:
        features = []
        for p in sorted(self.extractors.keys()):
            features.append(self.extractors[p].extract(wt, mt))
        return np.concatenate(features)

    def feature_names(self) -> List[str]:
        names = []
        for p in sorted(self.extractors.keys()):
            for name in self.extractors[p].feature_names():
                names.append(f'p{p}_{name}')
        return names


# =============================================================================
# Data Loading
# =============================================================================

def load_s669(filepath: Path) -> List[Dict]:
    """Load S669 dataset."""
    mutations = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip().split(',')
    ddg_idx = header.index('Experimental_DDG_dir') if 'Experimental_DDG_dir' in header else 11

    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) <= ddg_idx:
            continue
        try:
            seq_mut = parts[2] if len(parts) > 2 else parts[1]
            if len(seq_mut) >= 3:
                wt = seq_mut[0].upper()
                mt = seq_mut[-1].upper()
                if wt in AA_TO_IDX and mt in AA_TO_IDX and wt != mt:
                    ddg = float(parts[ddg_idx])
                    mutations.append({'wild_type': wt, 'mutant': mt, 'ddg_exp': ddg})
        except (ValueError, IndexError):
            continue
    return mutations


# =============================================================================
# Evaluation Functions
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all metrics."""
    spearman_r, _ = spearmanr(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'spearman': spearman_r,
        'pearson': pearson_r,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
    }


def evaluate_extractor(
    extractor: FeatureExtractor,
    mutations: List[Dict],
    n_folds: int = 5,
    n_repeats: int = 3,
    regressor_class: type = Ridge,
    regressor_kwargs: Dict = None,
) -> BenchmarkResult:
    """Evaluate a feature extractor with cross-validation."""

    if regressor_kwargs is None:
        regressor_kwargs = {'alpha': 1.0}

    # Extract features
    X = []
    y = []
    for mut in mutations:
        features = extractor.extract(mut['wild_type'], mut['mutant'])
        if not np.any(np.isnan(features)):
            X.append(features)
            y.append(mut['ddg_exp'])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        return BenchmarkResult(
            model_name=extractor.name,
            spearman=MetricResult(0.0),
            pearson=MetricResult(0.0),
            mae=MetricResult(float('inf')),
            rmse=MetricResult(float('inf')),
            r2=MetricResult(0.0),
            n_samples=0,
            n_features=0,
        )

    # Repeated k-fold CV
    all_metrics = {k: [] for k in ['spearman', 'pearson', 'mae', 'rmse', 'r2']}

    for repeat in range(n_repeats):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42 + repeat)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Cross-validation
        model = regressor_class(**regressor_kwargs)
        y_pred = cross_val_predict(model, X_scaled, y, cv=kf)

        metrics = compute_metrics(y, y_pred)
        for k, v in metrics.items():
            all_metrics[k].append(v)

    # Aggregate results
    def make_metric(values):
        arr = np.array(values)
        return MetricResult(
            value=np.mean(arr),
            std=np.std(arr),
            ci_low=np.percentile(arr, 2.5),
            ci_high=np.percentile(arr, 97.5),
        )

    return BenchmarkResult(
        model_name=extractor.name,
        spearman=make_metric(all_metrics['spearman']),
        pearson=make_metric(all_metrics['pearson']),
        mae=make_metric(all_metrics['mae']),
        rmse=make_metric(all_metrics['rmse']),
        r2=make_metric(all_metrics['r2']),
        n_samples=len(y),
        n_features=X.shape[1],
        metadata={'n_folds': n_folds, 'n_repeats': n_repeats},
    )


# =============================================================================
# Training Functions
# =============================================================================

def train_encoder(prime: int, n_epochs: int = 200, device: str = 'cpu') -> AminoAcidPadicEncoder:
    """Train a single-prime encoder."""
    encoder = AminoAcidPadicEncoder(prime=prime, latent_dim=16, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    for epoch in range(n_epochs):
        encoder.train()
        optimizer.zero_grad()

        # Compute losses
        indices = torch.arange(20, device=device)
        out = encoder(indices)

        # Hierarchy loss
        hierarchy_loss = torch.mean((out['radii'] - encoder.target_radii) ** 2)

        # Property alignment (simplified)
        z = out['z_hyp']
        prop_dist = torch.cdist(encoder.aa_properties, encoder.aa_properties, p=2)
        prop_dist = prop_dist / (prop_dist.max() + 1e-8)

        emb_dist = torch.zeros(20, 20, device=device)
        for i in range(20):
            for j in range(i+1, 20):
                d = poincare_distance(z[i:i+1], z[j:j+1], c=encoder.curvature).squeeze()
                d = d.clamp(0, 10)
                emb_dist[i, j] = emb_dist[j, i] = d
        emb_dist = emb_dist / (emb_dist.max() + 1e-8)

        mask = torch.triu(torch.ones(20, 20, device=device), diagonal=1).bool()
        property_loss = torch.mean((emb_dist[mask] - prop_dist[mask]) ** 2)

        loss = hierarchy_loss * 2.0 + property_loss * 1.0

        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

    return encoder


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkSuite:
    """Run comprehensive benchmarks."""

    def __init__(self, data_path: Path, output_dir: Path, device: str = 'cpu'):
        self.data_path = data_path
        self.output_dir = output_dir
        self.device = device
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self.mutations = load_s669(data_path)
        print(f"Loaded {len(self.mutations)} mutations from {data_path}")

    def run_baseline_tests(self) -> Dict[str, BenchmarkResult]:
        """Run baseline tests (physicochemical, random)."""
        print("\n" + "="*60)
        print("Running Baseline Tests")
        print("="*60)

        results = {}

        # Physicochemical baseline
        print("\n[1/2] Physicochemical features...")
        phys_extractor = PhysicochemicalFeatures()
        results['physicochemical'] = evaluate_extractor(phys_extractor, self.mutations)
        print(f"  Spearman: {results['physicochemical'].spearman}")

        # Random baseline (negative control)
        print("\n[2/2] Random features (negative control)...")
        rand_extractor = RandomFeatures(n_features=6)
        results['random'] = evaluate_extractor(rand_extractor, self.mutations)
        print(f"  Spearman: {results['random'].spearman}")

        return results

    def run_single_prime_tests(self, primes: List[int] = [2, 3, 5, 7, 11],
                                n_epochs: int = 200) -> Dict[str, BenchmarkResult]:
        """Run single-prime encoder tests."""
        print("\n" + "="*60)
        print("Running Single-Prime Tests")
        print("="*60)

        results = {}
        self.trained_encoders = {}

        for i, p in enumerate(primes):
            print(f"\n[{i+1}/{len(primes)}] Training p={p} encoder...")
            encoder = train_encoder(p, n_epochs=n_epochs, device=self.device)
            self.trained_encoders[p] = encoder

            extractor = PadicEncoderFeatures(encoder, p)
            results[f'padic_p{p}'] = evaluate_extractor(extractor, self.mutations)
            print(f"  Spearman: {results[f'padic_p{p}'].spearman}")

        return results

    def run_ensemble_tests(self) -> Dict[str, BenchmarkResult]:
        """Run multi-prime ensemble tests."""
        print("\n" + "="*60)
        print("Running Ensemble Tests")
        print("="*60)

        if not hasattr(self, 'trained_encoders') or len(self.trained_encoders) == 0:
            print("No trained encoders found. Running single-prime tests first...")
            self.run_single_prime_tests()

        results = {}

        # Full ensemble
        print("\n[1/3] Full ensemble (all primes)...")
        full_extractor = EnsembleFeatures(self.trained_encoders)
        results['ensemble_full'] = evaluate_extractor(full_extractor, self.mutations)
        print(f"  Spearman: {results['ensemble_full'].spearman}")
        print(f"  Features: {results['ensemble_full'].n_features}")

        # Top-2 ensemble (best performing primes)
        print("\n[2/3] Top-2 ensemble (p=2, p=5)...")
        top2_encoders = {p: self.trained_encoders[p] for p in [2, 5] if p in self.trained_encoders}
        if len(top2_encoders) == 2:
            top2_extractor = EnsembleFeatures(top2_encoders)
            results['ensemble_top2'] = evaluate_extractor(top2_extractor, self.mutations)
            print(f"  Spearman: {results['ensemble_top2'].spearman}")

        # With different regressors
        print("\n[3/3] Ensemble with GradientBoosting...")
        results['ensemble_gb'] = evaluate_extractor(
            full_extractor, self.mutations,
            regressor_class=GradientBoostingRegressor,
            regressor_kwargs={'n_estimators': 100, 'max_depth': 3, 'random_state': 42}
        )
        print(f"  Spearman: {results['ensemble_gb'].spearman}")

        return results

    def run_regressor_comparison(self) -> Dict[str, BenchmarkResult]:
        """Compare different regression models."""
        print("\n" + "="*60)
        print("Running Regressor Comparison")
        print("="*60)

        if not hasattr(self, 'trained_encoders'):
            self.run_single_prime_tests(primes=[5], n_epochs=200)

        extractor = EnsembleFeatures(self.trained_encoders)

        regressors = [
            ('Ridge', Ridge, {'alpha': 1.0}),
            ('Lasso', Lasso, {'alpha': 0.1}),
            ('ElasticNet', ElasticNet, {'alpha': 0.1, 'l1_ratio': 0.5}),
            ('RandomForest', RandomForestRegressor, {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}),
            ('GradientBoosting', GradientBoostingRegressor, {'n_estimators': 100, 'max_depth': 3, 'random_state': 42}),
        ]

        results = {}
        for i, (name, cls, kwargs) in enumerate(regressors):
            print(f"\n[{i+1}/{len(regressors)}] {name}...")
            results[f'regressor_{name.lower()}'] = evaluate_extractor(
                extractor, self.mutations,
                regressor_class=cls,
                regressor_kwargs=kwargs
            )
            print(f"  Spearman: {results[f'regressor_{name.lower()}'].spearman}")

        return results

    def run_all_tests(self) -> Dict[str, BenchmarkResult]:
        """Run all benchmark tests."""
        all_results = {}

        # Baselines
        all_results.update(self.run_baseline_tests())

        # Single prime
        all_results.update(self.run_single_prime_tests())

        # Ensemble
        all_results.update(self.run_ensemble_tests())

        # Regressor comparison
        all_results.update(self.run_regressor_comparison())

        return all_results

    def save_results(self, results: Dict[str, BenchmarkResult], filename: str = 'benchmark_results.json'):
        """Save results to JSON."""
        output = {
            'timestamp': datetime.now().isoformat(),
            'data_path': str(self.data_path),
            'n_mutations': len(self.mutations),
            'results': {k: v.to_dict() for k, v in results.items()},
        }

        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to {output_path}")
        return output_path

    def print_summary(self, results: Dict[str, BenchmarkResult]):
        """Print summary table."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(f"{'Model':<30} {'Spearman':>12} {'Pearson':>12} {'MAE':>10} {'Features':>10}")
        print("-"*80)

        # Sort by Spearman
        sorted_results = sorted(results.items(), key=lambda x: -x[1].spearman.value)

        for name, result in sorted_results:
            print(f"{name:<30} {result.spearman.value:>12.4f} {result.pearson.value:>12.4f} "
                  f"{result.mae.value:>10.4f} {result.n_features:>10}")

        print("="*80)

        # Best model
        best = sorted_results[0]
        print(f"\nBest model: {best[0]} (Spearman = {best[1].spearman})")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='P-adic Encoder Benchmark Suite')
    parser.add_argument('--test', type=str, choices=['baselines', 'single', 'ensemble', 'regressors', 'all'],
                        default='all', help='Which tests to run')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--primes', type=str, default='2,3,5,7,11', help='Primes to test (comma-separated)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    args = parser.parse_args()

    # Paths
    data_dir = PROJECT_ROOT / 'deliverables' / 'partners' / 'jose_colbes' / 'reproducibility' / 'data'
    data_path = data_dir / 's669_full.csv'
    if not data_path.exists():
        data_path = data_dir / 'S669' / 'S669.csv'

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / 'research' / 'codon-encoder' / 'results' / 'benchmarks'

    # Parse primes
    primes = [int(p) for p in args.primes.split(',')]

    # Run benchmarks
    suite = BenchmarkSuite(data_path, output_dir, device=args.device)

    if args.test == 'baselines':
        results = suite.run_baseline_tests()
    elif args.test == 'single':
        results = suite.run_single_prime_tests(primes=primes, n_epochs=args.epochs)
    elif args.test == 'ensemble':
        suite.run_single_prime_tests(primes=primes, n_epochs=args.epochs)
        results = suite.run_ensemble_tests()
    elif args.test == 'regressors':
        suite.run_single_prime_tests(primes=primes, n_epochs=args.epochs)
        results = suite.run_regressor_comparison()
    else:  # all
        results = suite.run_all_tests()

    # Save and summarize
    suite.save_results(results)
    suite.print_summary(results)


if __name__ == '__main__':
    main()
