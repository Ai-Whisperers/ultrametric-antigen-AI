"""Comprehensive benchmarking suite for Ternary VAE v5.5."""

import argparse
import torch
import yaml
import time
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
from tabulate import tabulate
from typing import Dict, List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.ternary_vae_v5_5 import DualNeuralVAEV5
from src.utils.data import generate_all_ternary_operations, TernaryOperationDataset
from src.utils.metrics import evaluate_coverage, compute_latent_entropy


class TernaryVAEBenchmark:
    """Benchmark suite for Ternary VAE v5.5."""

    def __init__(self, config_path: str, checkpoint_path: str = None, device: str = 'cuda'):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Require checkpoint for valid benchmarking
        if checkpoint_path is None:
            raise ValueError(
                "Checkpoint is required for benchmarking. "
                "Running benchmarks on untrained weights produces meaningless results. "
                "Please provide --checkpoint argument."
            )

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize model
        model_config = self.config['model']
        self.model = DualNeuralVAEV5(
            input_dim=model_config['input_dim'],
            latent_dim=model_config['latent_dim'],
            rho_min=model_config['rho_min'],
            rho_max=model_config['rho_max'],
            lambda3_base=model_config['lambda3_base'],
            lambda3_amplitude=model_config['lambda3_amplitude'],
            eps_kl=model_config['eps_kl']
        ).to(self.device)

        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        self.model.eval()

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best val loss: {checkpoint.get('best_val_loss', 'unknown')}")

    def benchmark_inference_speed(self, num_samples: int = 10000, num_trials: int = 10) -> Dict:
        """Benchmark inference speed.

        Args:
            num_samples: Number of samples to generate per trial
            num_trials: Number of trials to average

        Returns:
            dict: Timing statistics
        """
        print(f"\n{'='*80}")
        print("Benchmarking Inference Speed")
        print(f"{'='*80}")

        times_A = []
        times_B = []

        for trial in range(num_trials):
            # Warm-up
            if trial == 0:
                _ = self.model.sample(100, self.device, 'A')
                _ = self.model.sample(100, self.device, 'B')
                if self.device == 'cuda':
                    torch.cuda.synchronize()

            # VAE-A
            start = time.time()
            _ = self.model.sample(num_samples, self.device, 'A')
            if self.device == 'cuda':
                torch.cuda.synchronize()
            times_A.append(time.time() - start)

            # VAE-B
            start = time.time()
            _ = self.model.sample(num_samples, self.device, 'B')
            if self.device == 'cuda':
                torch.cuda.synchronize()
            times_B.append(time.time() - start)

        results = {
            'num_samples': num_samples,
            'num_trials': num_trials,
            'vae_a_mean_time': np.mean(times_A),
            'vae_a_std_time': np.std(times_A),
            'vae_a_samples_per_sec': num_samples / np.mean(times_A),
            'vae_b_mean_time': np.mean(times_B),
            'vae_b_std_time': np.std(times_B),
            'vae_b_samples_per_sec': num_samples / np.mean(times_B),
        }

        # Print results
        print(f"\nVAE-A: {results['vae_a_mean_time']:.4f}s ± {results['vae_a_std_time']:.4f}s")
        print(f"       {results['vae_a_samples_per_sec']:.0f} samples/sec")
        print(f"\nVAE-B: {results['vae_b_mean_time']:.4f}s ± {results['vae_b_std_time']:.4f}s")
        print(f"       {results['vae_b_samples_per_sec']:.0f} samples/sec")

        return results

    def benchmark_coverage(self, num_samples: int = 195000, num_trials: int = 5) -> Dict:
        """Benchmark operation coverage.

        Args:
            num_samples: Number of samples to generate (10x exhaustive)
            num_trials: Number of trials to average

        Returns:
            dict: Coverage statistics
        """
        print(f"\n{'='*80}")
        print("Benchmarking Coverage")
        print(f"{'='*80}")

        results_A = []
        results_B = []

        for trial in range(num_trials):
            print(f"\nTrial {trial+1}/{num_trials}...")

            # VAE-A
            samples_A = self.model.sample(num_samples, self.device, 'A')
            unique_A, cov_A = evaluate_coverage(samples_A)
            results_A.append((unique_A, cov_A))

            # VAE-B
            samples_B = self.model.sample(num_samples, self.device, 'B')
            unique_B, cov_B = evaluate_coverage(samples_B)
            results_B.append((unique_B, cov_B))

            print(f"  VAE-A: {unique_A} ops ({cov_A:.2f}%)")
            print(f"  VAE-B: {unique_B} ops ({cov_B:.2f}%)")

        # Aggregate
        unique_A_values = [r[0] for r in results_A]
        cov_A_values = [r[1] for r in results_A]
        unique_B_values = [r[0] for r in results_B]
        cov_B_values = [r[1] for r in results_B]

        results = {
            'num_samples': num_samples,
            'num_trials': num_trials,
            'vae_a_mean_unique': np.mean(unique_A_values),
            'vae_a_std_unique': np.std(unique_A_values),
            'vae_a_mean_coverage': np.mean(cov_A_values),
            'vae_a_std_coverage': np.std(cov_A_values),
            'vae_a_max_coverage': np.max(cov_A_values),
            'vae_b_mean_unique': np.mean(unique_B_values),
            'vae_b_std_unique': np.std(unique_B_values),
            'vae_b_mean_coverage': np.mean(cov_B_values),
            'vae_b_std_coverage': np.std(cov_B_values),
            'vae_b_max_coverage': np.max(cov_B_values),
        }

        # Print summary
        print(f"\n{'='*40}")
        print("Coverage Summary")
        print(f"{'='*40}")
        print(f"VAE-A: {results['vae_a_mean_coverage']:.2f}% ± {results['vae_a_std_coverage']:.2f}%")
        print(f"       Max: {results['vae_a_max_coverage']:.2f}%")
        print(f"       ({results['vae_a_mean_unique']:.0f} ± {results['vae_a_std_unique']:.0f} ops)")
        print(f"\nVAE-B: {results['vae_b_mean_coverage']:.2f}% ± {results['vae_b_std_coverage']:.2f}%")
        print(f"       Max: {results['vae_b_max_coverage']:.2f}%")
        print(f"       ({results['vae_b_mean_unique']:.0f} ± {results['vae_b_std_unique']:.0f} ops)")

        return results

    def benchmark_latent_entropy(self, num_samples: int = 10000) -> Dict:
        """Benchmark latent space entropy.

        Args:
            num_samples: Number of samples to analyze

        Returns:
            dict: Entropy statistics
        """
        print(f"\n{'='*80}")
        print("Benchmarking Latent Entropy")
        print(f"{'='*80}")

        # Generate real ternary operations and encode them to get actual latent codes
        all_ops = generate_all_ternary_operations()
        # Sample subset if requested samples < total operations
        if num_samples < len(all_ops):
            indices = torch.randperm(len(all_ops))[:num_samples]
            sampled_ops = all_ops[indices]
        else:
            sampled_ops = all_ops[:num_samples]

        # Convert to tensor if needed and move to device
        if not isinstance(sampled_ops, torch.Tensor):
            sampled_ops = torch.FloatTensor(sampled_ops)
        sampled_ops = sampled_ops.to(self.device)

        # Encode to get real latent codes
        with torch.no_grad():
            mu_A, _ = self.model.encoder_A(sampled_ops)
            mu_B, _ = self.model.encoder_B(sampled_ops)

            H_A = compute_latent_entropy(mu_A)
            H_B = compute_latent_entropy(mu_B)

        results = {
            'num_samples': num_samples,
            'entropy_A': H_A.item(),
            'entropy_B': H_B.item(),
            'entropy_diff': abs(H_A.item() - H_B.item()),
        }

        print(f"\nVAE-A Entropy: {results['entropy_A']:.3f}")
        print(f"VAE-B Entropy: {results['entropy_B']:.3f}")
        print(f"Difference: {results['entropy_diff']:.3f}")

        return results

    def benchmark_memory_usage(self) -> Dict:
        """Benchmark memory usage.

        Returns:
            dict: Memory statistics
        """
        print(f"\n{'='*80}")
        print("Benchmarking Memory Usage")
        print(f"{'='*80}")

        if not torch.cuda.is_available():
            print("CUDA not available, skipping memory benchmark")
            return {}

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Baseline
        baseline = torch.cuda.memory_allocated() / 1e9

        # Generate samples
        _ = self.model.sample(10000, self.device, 'A')
        torch.cuda.synchronize()

        # Peak memory
        peak = torch.cuda.max_memory_allocated() / 1e9
        current = torch.cuda.memory_allocated() / 1e9

        results = {
            'baseline_gb': baseline,
            'current_gb': current,
            'peak_gb': peak,
            'overhead_gb': peak - baseline,
        }

        print(f"\nBaseline: {results['baseline_gb']:.3f} GB")
        print(f"Current: {results['current_gb']:.3f} GB")
        print(f"Peak: {results['peak_gb']:.3f} GB")
        print(f"Overhead: {results['overhead_gb']:.3f} GB")

        return results

    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite.

        Returns:
            dict: All benchmark results
        """
        print(f"\n{'#'*80}")
        print(f"# Ternary VAE v5.5 - Full Benchmark Suite")
        print(f"# Config: {self.config_path}")
        print(f"# Checkpoint: {self.checkpoint_path or 'None'}")
        print(f"# Device: {self.device}")
        print(f"{'#'*80}")

        results = {}

        # Inference speed
        results['inference'] = self.benchmark_inference_speed(num_samples=10000, num_trials=10)

        # Coverage
        results['coverage'] = self.benchmark_coverage(num_samples=195000, num_trials=5)

        # Latent entropy
        results['entropy'] = self.benchmark_latent_entropy(num_samples=10000)

        # Memory usage
        results['memory'] = self.benchmark_memory_usage()

        # Print summary table
        self.print_summary_table(results)

        return results

    def print_summary_table(self, results: Dict):
        """Print formatted summary table."""
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}\n")

        # Prepare table data
        table_data = []

        # Inference
        if 'inference' in results:
            table_data.append(['Inference Speed (VAE-A)', f"{results['inference']['vae_a_samples_per_sec']:.0f} samples/sec"])
            table_data.append(['Inference Speed (VAE-B)', f"{results['inference']['vae_b_samples_per_sec']:.0f} samples/sec"])

        # Coverage
        if 'coverage' in results:
            table_data.append(['Coverage (VAE-A)', f"{results['coverage']['vae_a_mean_coverage']:.2f}% ± {results['coverage']['vae_a_std_coverage']:.2f}%"])
            table_data.append(['Coverage (VAE-B)', f"{results['coverage']['vae_b_mean_coverage']:.2f}% ± {results['coverage']['vae_b_std_coverage']:.2f}%"])

        # Entropy
        if 'entropy' in results:
            table_data.append(['Latent Entropy (VAE-A)', f"{results['entropy']['entropy_A']:.3f}"])
            table_data.append(['Latent Entropy (VAE-B)', f"{results['entropy']['entropy_B']:.3f}"])

        # Memory
        if 'memory' in results and results['memory']:
            table_data.append(['Peak Memory Usage', f"{results['memory']['peak_gb']:.3f} GB"])

        print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))

    def save_results(self, results: Dict, output_path: str = None):
        """Save benchmark results to JSON file with metadata.

        Args:
            results: Benchmark results dictionary
            output_path: Path to save results (default: benchmarks/coverage_vs_entropy.json)
        """
        if output_path is None:
            output_path = Path(__file__).parent.parent.parent / "benchmarks" / "coverage_vs_entropy.json"
        else:
            output_path = Path(output_path)

        # Create benchmarks directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Compute checkpoint hash
        checkpoint_hash = self._compute_file_hash(self.checkpoint_path)
        config_hash = self._compute_file_hash(self.config_path)

        # Prepare output with metadata
        output = {
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'checkpoint_path': str(self.checkpoint_path),
                'checkpoint_hash': checkpoint_hash,
                'config_path': str(self.config_path),
                'config_hash': config_hash,
                'device': self.device,
            },
            'results': results,
        }

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Results saved to: {output_path}")
        print(f"Checkpoint hash: {checkpoint_hash}")
        print(f"Config hash: {config_hash}")
        print(f"{'='*80}\n")

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Benchmark Ternary VAE v5.5')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint (required)')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials for coverage benchmark')
    parser.add_argument('--output', type=str, default=None, help='Output path for JSON results (default: benchmarks/coverage_vs_entropy.json)')
    args = parser.parse_args()

    # Run benchmark
    benchmark = TernaryVAEBenchmark(args.config, args.checkpoint, args.device)
    results = benchmark.run_full_benchmark()

    # Save results to JSON
    benchmark.save_results(results, args.output)

    print(f"\n{'='*80}")
    print("Benchmark Complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
