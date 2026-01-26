#!/usr/bin/env python3
"""Master Analysis: Complete p-adic structure characterization of viral space.

This script orchestrates all three analyses:
    1. Multi-prime ultrametric test (which prime, if any?)
    2. Projection/deformation analysis (is there a mapping from 3-adic?)
    3. Adelic decomposition (is it multi-prime?)

And produces a FINAL VERDICT on whether:
    A) Viral space has p-adic structure (and which prime)
    B) An output module adjusting p-adic projections is mathematically justified
    C) We must honestly report no p-adic structure

The key principle: WE MUST NOT LIE about the mathematics.

Usage:
    python 04_master_analysis.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Setup paths
_script_dir = Path(__file__).resolve().parent
_results_dir = _script_dir / "results"


def run_analysis(script_name: str) -> dict:
    """Run an analysis script and load its results."""
    script_path = _script_dir / script_name
    results_name = script_name.replace('.py', '_results.json').replace(
        '01_multi_prime_ultrametric_test', 'multi_prime_ultrametric'
    ).replace(
        '02_projection_deformation_analysis', 'projection_deformation'
    ).replace(
        '03_adelic_decomposition_test', 'adelic_decomposition'
    )

    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print('='*60)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=300
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        # Load results
        results_file = _results_dir / results_name
        if results_file.exists():
            with open(results_file) as f:
                return json.load(f)
        else:
            print(f"WARNING: Results file not found: {results_file}")
            return {}

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {script_name} took too long")
        return {}
    except Exception as e:
        print(f"ERROR running {script_name}: {e}")
        return {}


def synthesize_verdict(
    ultrametric_results: dict,
    projection_results: dict,
    adelic_results: dict
) -> dict:
    """Synthesize all results into a final verdict."""

    verdict = {
        'timestamp': datetime.now().isoformat(),
        'analyses_completed': {
            'ultrametric': bool(ultrametric_results),
            'projection': bool(projection_results),
            'adelic': bool(adelic_results)
        },
        'findings': {},
        'verdict': None,
        'recommendation': None,
        'output_module_justified': False,
        'honest_conclusion': None
    }

    # === ULTRAMETRIC FINDINGS ===
    if ultrametric_results:
        best = ultrametric_results.get('best_ultrametric', {})
        verdict['findings']['ultrametric'] = {
            'best_prime': best.get('prime'),
            'best_compliance': best.get('compliance'),
            'has_structure': best.get('compliance', 0) > 0.85
        }

    # === PROJECTION FINDINGS ===
    if projection_results:
        linear = projection_results.get('linear_projection', {})
        nonlinear = projection_results.get('nonlinear_analysis', {})
        verdict['findings']['projection'] = {
            'linear_r2': linear.get('r2_score'),
            'best_nonlinear_rho': nonlinear.get('best_rho'),
            'has_relationship': (linear.get('r2_score', 0) > 0.2 or
                                abs(nonlinear.get('best_rho', 0)) > 0.3)
        }

    # === ADELIC FINDINGS ===
    if adelic_results:
        adelic = adelic_results.get('adelic_analysis', {})
        verdict['findings']['adelic'] = {
            'is_adelic': adelic.get('is_genuinely_adelic'),
            'best_single_r2': adelic.get('best_single_r2'),
            'adelic_r2': adelic.get('adelic_r2'),
            'dominant_primes': adelic.get('dominant_primes')
        }

    # === SYNTHESIZE VERDICT ===
    has_ultrametric = verdict['findings'].get('ultrametric', {}).get('has_structure', False)
    has_projection = verdict['findings'].get('projection', {}).get('has_relationship', False)
    is_adelic = verdict['findings'].get('adelic', {}).get('is_adelic', False)

    if is_adelic:
        verdict['verdict'] = 'ADELIC_STRUCTURE'
        verdict['output_module_justified'] = True
        verdict['recommendation'] = """
OUTPUT MODULE DESIGN (Adelic):

The viral space exhibits multi-prime structure. An output module IS justified:

class AdelicProjectionModule(nn.Module):
    def __init__(self, primes=[2, 3, 5, 7], hidden_dim=32):
        super().__init__()
        self.primes = primes
        # Learned weights for each prime's contribution
        self.prime_weights = nn.Parameter(torch.ones(len(primes)))
        self.mlp = nn.Sequential(
            nn.Linear(len(primes), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, padic_distances: dict) -> torch.Tensor:
        # Weighted combination of p-adic distances
        features = torch.stack([padic_distances[p] for p in self.primes], dim=-1)
        weighted = features * F.softmax(self.prime_weights, dim=0)
        return self.mlp(weighted)
"""
        verdict['honest_conclusion'] = "Statistics support adelic structure. Output module is mathematically justified."

    elif has_ultrametric and has_projection:
        best_prime = verdict['findings'].get('ultrametric', {}).get('best_prime')
        verdict['verdict'] = f'SINGLE_PRIME_{best_prime}'
        verdict['output_module_justified'] = True
        verdict['recommendation'] = f"""
OUTPUT MODULE DESIGN (Single Prime):

The viral space appears {best_prime}-adic. A simple transformation IS justified:

class PadicProjectionModule(nn.Module):
    def __init__(self, prime={best_prime}):
        super().__init__()
        self.prime = prime
        # Learnable non-linear transformation
        self.transform = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

    def forward(self, hyperbolic_dist: torch.Tensor) -> torch.Tensor:
        # Transform 3-adic (hyperbolic) to {best_prime}-adic (viral)
        return self.transform(hyperbolic_dist.unsqueeze(-1)).squeeze(-1)
"""
        verdict['honest_conclusion'] = f"Statistics support {best_prime}-adic structure with monotonic relationship to 3-adic."

    elif has_projection:
        verdict['verdict'] = 'WEAK_RELATIONSHIP'
        verdict['output_module_justified'] = False
        verdict['recommendation'] = """
NO OUTPUT MODULE RECOMMENDED:

A weak statistical relationship exists but is insufficient to justify
a dedicated "p-adic projection adjustment" module.

Alternative approaches:
1. Ensemble: Keep p-adic + add separate evolution model
2. Feature: Use p-adic as additional feature, not primary signal
3. Acknowledge: Two orthogonal signals is informative, not a failure
"""
        verdict['honest_conclusion'] = "Weak relationship exists but output module would overstate the mathematics."

    else:
        verdict['verdict'] = 'NO_PADIC_STRUCTURE'
        verdict['output_module_justified'] = False
        verdict['recommendation'] = """
DO NOT BUILD OUTPUT MODULE:

The statistics clearly show viral evolutionary distance does NOT
follow p-adic geometry for any prime or combination tested.

An "output module" claiming to adjust p-adic projections would be
MATHEMATICALLY DISHONEST.

HONEST CONCLUSIONS:
1. TrainableCodonEncoder captures GENETIC CODE GRAMMAR (universal, 3-adic)
2. Viral evolution operates in a DIFFERENT geometric space
3. These are ORTHOGONAL information axes - both valuable, but independent

RECOMMENDATIONS FOR ALEJANDRA ROJAS:
1. Use hyperbolic variance as ONE signal among many (not primary)
2. Combine with Shannon entropy (different conservation aspect)
3. Do NOT claim p-adic primer design "adjusts" for viral evolution
4. The dual-metric approach (Shannon + hyperbolic) is scientifically valid
   precisely BECAUSE they capture different things
"""
        verdict['honest_conclusion'] = "No p-adic structure detected. We must not pretend otherwise."

    return verdict


def main():
    """Run complete master analysis."""

    print("=" * 70)
    print("MASTER ANALYSIS: VIRAL COMBINATORIAL SPACE P-ADIC STRUCTURE")
    print("=" * 70)
    print("""
This analysis will determine:
1. Does viral sequence space have p-adic structure? (which prime?)
2. Is there a projection from 3-adic codon space to viral space?
3. Is viral space multi-prime (adelic)?

Based on rigorous statistics, we will determine whether an "output module"
for adjusting p-adic projections is mathematically justified.

WE WILL NOT LIE ABOUT THE MATHEMATICS.
""")

    _results_dir.mkdir(parents=True, exist_ok=True)

    # Run all analyses
    ultrametric_results = run_analysis('01_multi_prime_ultrametric_test.py')
    projection_results = run_analysis('02_projection_deformation_analysis.py')
    adelic_results = run_analysis('03_adelic_decomposition_test.py')

    # Synthesize verdict
    print("\n" + "=" * 70)
    print("SYNTHESIZING FINAL VERDICT")
    print("=" * 70)

    verdict = synthesize_verdict(ultrametric_results, projection_results, adelic_results)

    # Print verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    print(f"\nStructure detected: {verdict['verdict']}")
    print(f"Output module justified: {verdict['output_module_justified']}")
    print(f"\nHonest conclusion:\n{verdict['honest_conclusion']}")
    print(f"\nRecommendation:\n{verdict['recommendation']}")

    # Save verdict
    verdict_file = _results_dir / "FINAL_VERDICT.json"
    with open(verdict_file, 'w') as f:
        json.dump(verdict, f, indent=2)

    print(f"\nFinal verdict saved to: {verdict_file}")

    # Also write a markdown summary
    md_file = _results_dir / "PADIC_STRUCTURE_ANALYSIS_REPORT.md"
    with open(md_file, 'w') as f:
        f.write(f"""# P-adic Structure Analysis Report

**Generated:** {verdict['timestamp']}

## Executive Summary

**Verdict:** {verdict['verdict']}
**Output Module Justified:** {'YES' if verdict['output_module_justified'] else 'NO'}

## Honest Conclusion

{verdict['honest_conclusion']}

## Analyses Completed

| Analysis | Completed | Key Finding |
|----------|-----------|-------------|
| Ultrametric | {'Yes' if verdict['analyses_completed']['ultrametric'] else 'No'} | {verdict['findings'].get('ultrametric', {}).get('has_structure', 'N/A')} |
| Projection | {'Yes' if verdict['analyses_completed']['projection'] else 'No'} | R²={verdict['findings'].get('projection', {}).get('linear_r2', 'N/A')} |
| Adelic | {'Yes' if verdict['analyses_completed']['adelic'] else 'No'} | {verdict['findings'].get('adelic', {}).get('is_adelic', 'N/A')} |

## Detailed Findings

### Ultrametric Compliance
{json.dumps(verdict['findings'].get('ultrametric', {}), indent=2)}

### Projection Analysis
{json.dumps(verdict['findings'].get('projection', {}), indent=2)}

### Adelic Decomposition
{json.dumps(verdict['findings'].get('adelic', {}), indent=2)}

## Recommendation

{verdict['recommendation']}

---

*This analysis follows the principle: We must not lie about the mathematics.*
""")

    print(f"Markdown report saved to: {md_file}")

    return verdict


if __name__ == "__main__":
    main()
