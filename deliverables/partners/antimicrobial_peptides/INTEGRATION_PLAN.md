# Integration Plan: PeptideVAE for Carlos Brizuela Package

**Doc-Type:** Technical Integration Plan · Version 1.0 · Updated 2026-01-05 · AI Whisperers

---

## Problem Statement

The NSGA-II optimization tools (B1, B8, B10) produce 3-character sequences instead of real AMPs because they use the wrong VAE model.

**Current Architecture (Broken):**
```
Latent (16D) → TernaryVAE Decoder → 3 amino acids

TernaryVAE is designed for:
  - Input: 9 ternary values (-1, 0, 1)
  - Output: 9 ternary values → 3 codons → 3 amino acids
  - Purpose: P-adic structure learning on genetic code
```

**Required Architecture:**
```
Latent (16D) → PeptideVAE Decoder → 10-50 amino acids

PeptideVAE is designed for:
  - Input: Peptide sequences (10-50 AA)
  - Output: MIC prediction (antimicrobial activity)
  - Purpose: Antimicrobial peptide activity prediction
```

---

## Root Cause Analysis

| Component | Current Behavior | Expected Behavior |
|-----------|-----------------|-------------------|
| `shared/config.py` | Defaults to `homeostatic_rich/best.pt` | Should default to `best_production.pt` |
| `shared/vae_service.py` | Loads `TernaryVAEV5_11_PartialFreeze` | Should load `PeptideVAE` |
| `decode_latent()` | Decodes 9 ternary ops → 3 AA | Should decode to 10-50 AA sequence |
| Activity prediction | Heuristic formula | Should use PeptideVAE MIC head |

---

## Architecture Comparison

### TernaryVAE (Wrong Model)

```python
class TernaryVAEV5_11_PartialFreeze:
    """Encodes/decodes 19,683 ternary operations to learn p-adic structure."""

    latent_dim = 16
    input = torch.Tensor([9])  # 9 ternary values
    output = torch.Tensor([9, 3])  # 9 positions, 3 classes each

    # Decoding produces 3 amino acids:
    # 9 ternary values → 3 codons → 3 amino acids
```

### PeptideVAE (Correct Model)

```python
class PeptideVAE:
    """Encodes peptide sequences, predicts MIC activity."""

    latent_dim = 16
    max_seq_len = 50
    vocab_size = 22  # 20 AA + stop + pad

    # Encoding:
    sequence (str) → tokens → transformer → latent (16D)

    # MIC Prediction:
    latent (16D) → MIC head → scalar MIC value

    # Decoding (for generation):
    latent (16D) → transformer decoder → sequence (10-50 AA)
```

---

## Integration Options

### Option A: Sequence-Space Evolution (Recommended)

Bypass latent-space optimization entirely. Use PeptideVAE only for scoring.

**Pros:**
- No decoder needed
- Real peptide sequences throughout
- Immediate implementation

**Cons:**
- Constrained to mutation space
- May miss novel peptides

**Flow:**
```
1. Load seed sequences from DRAMP database
2. Apply mutations (single AA substitutions)
3. Score with PeptideVAE (MIC prediction)
4. NSGA-II selects best mutations
5. Output: Real peptide sequences with predicted MIC
```

**Files to Create:**
- `scripts/sequence_nsga2.py` - Sequence-space optimizer
- `src/mutators.py` - Peptide mutation operators

### Option B: PeptideVAE Latent Optimization

Keep latent-space optimization but use PeptideVAE instead of TernaryVAE.

**Pros:**
- Explores full latent space
- Can generate novel peptides

**Cons:**
- Requires working decoder
- May produce invalid sequences
- Decoder training may be unstable

**Flow:**
```
1. Sample latent vectors (16D)
2. Decode with PeptideVAE decoder
3. Score with PeptideVAE MIC head
4. NSGA-II optimizes latent vectors
5. Output: Generated sequences with predicted MIC
```

**Files to Modify:**
- `shared/vae_service.py` - Replace TernaryVAE with PeptideVAE
- `shared/config.py` - Update checkpoint path

### Option C: Hybrid Approach

Use sequence seeds + latent perturbations.

**Flow:**
```
1. Encode real seed sequences with PeptideVAE
2. Perturb latent vectors (Gaussian noise)
3. Decode back to sequences
4. Score with PeptideVAE MIC head
5. Output: Novel variations of known AMPs
```

---

## Recommended Implementation: Option A

Given the validation findings, Option A (Sequence-Space Evolution) is recommended:

1. **Fastest to implement** - PeptideVAE encoder + MIC head already work (r=0.74)
2. **No decoder issues** - Avoids autoregressive generation problems
3. **Interpretable** - Every mutation is traceable to parent sequence
4. **Biologically valid** - All sequences are real peptide variations

---

## Implementation Tasks

### Phase 1: Prediction API (Quick Win)

| Task | File | Effort | Output |
|------|------|--------|--------|
| Create prediction script | `scripts/predict_mic.py` | 30 min | CLI tool |
| Create batch evaluator | `scripts/evaluate_candidates.py` | 1 hr | CSV results |
| Update DRAMP loader | `scripts/dramp_activity_loader.py` | 30 min | PeptideVAE integration |

**Deliverable:** Working MIC prediction for any peptide sequence

### Phase 2: Sequence Evolution (NSGA-II Fix)

| Task | File | Effort | Output |
|------|------|--------|--------|
| Create mutation operators | `src/mutators.py` | 2 hr | AA substitution/insertion/deletion |
| Create sequence NSGA-II | `scripts/sequence_nsga2.py` | 3 hr | Pareto-optimal peptides |
| Update B1 tool | `scripts/B1_pathogen_specific_design.py` | 2 hr | Fixed tool |
| Update B8 tool | `scripts/B8_microbiome_safe_amps.py` | 1 hr | Fixed tool |
| Update B10 tool | `scripts/B10_synthesis_optimization.py` | 1 hr | Fixed tool |

**Deliverable:** Working NSGA-II optimization producing real AMPs

### Phase 3: Integration & Testing

| Task | File | Effort | Output |
|------|------|--------|--------|
| Create integration tests | `tests/test_peptide_optimization.py` | 2 hr | Test suite |
| Benchmark vs random | `validation/benchmark_optimization.py` | 1 hr | Performance metrics |
| Documentation | `docs/OPTIMIZATION_API.md` | 1 hr | User guide |

**Deliverable:** Production-ready optimization pipeline

---

## File Changes Required

### shared/vae_service.py

**Current:**
```python
from src.models import TernaryVAEV5_11_PartialFreeze
self.model = TernaryVAEV5_11_PartialFreeze(...)
```

**For Option A (Prediction Only):**
```python
from src.encoders.peptide_encoder import PeptideVAE

class PeptideVAEService:
    """Singleton PeptideVAE service for activity prediction."""

    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        checkpoint = torch.load('checkpoints_definitive/best_production.pt')
        config = checkpoint['config']
        model = PeptideVAE(
            latent_dim=config['latent_dim'],
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            dropout=config['dropout'],
            max_radius=config['max_radius'],
            curvature=config['curvature'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def predict_mic(self, sequence: str) -> float:
        """Predict MIC for a peptide sequence."""
        with torch.no_grad():
            outputs = self.model([sequence], teacher_forcing=False)
            return outputs['mic_pred'].item()

    def predict_batch(self, sequences: list[str]) -> list[float]:
        """Predict MIC for multiple sequences."""
        with torch.no_grad():
            outputs = self.model(sequences, teacher_forcing=False)
            return outputs['mic_pred'].squeeze(-1).tolist()
```

### shared/config.py

**Current:**
```python
fallback_checkpoints: list[str] = field(default_factory=lambda: [
    "checkpoints/homeostatic_rich/best.pt",
    ...
])
```

**Updated:**
```python
# Ternary VAE (for p-adic structure)
ternary_vae_checkpoint: str = "checkpoints/homeostatic_rich/best.pt"

# PeptideVAE (for AMP activity prediction)
peptide_vae_checkpoint: str = "checkpoints_definitive/best_production.pt"
```

---

## New Files to Create

### scripts/predict_mic.py

```python
#!/usr/bin/env python3
"""Predict MIC activity for peptide sequences.

Usage:
    python predict_mic.py "KLAKLAKKLAKLAK"
    python predict_mic.py --file candidates.txt
    python predict_mic.py --interactive
"""

import argparse
from pathlib import Path
import torch
from src.encoders.peptide_encoder import PeptideVAE

def load_model():
    ckpt = torch.load('checkpoints_definitive/best_production.pt', map_location='cpu')
    config = ckpt['config']
    model = PeptideVAE(
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        dropout=config['dropout'],
        max_radius=config['max_radius'],
        curvature=config['curvature'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model

def predict(model, sequence: str) -> float:
    with torch.no_grad():
        outputs = model([sequence], teacher_forcing=False)
        return outputs['mic_pred'].item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sequence', nargs='?', help='Peptide sequence')
    parser.add_argument('--file', '-f', help='File with sequences (one per line)')
    parser.add_argument('--interactive', '-i', action='store_true')
    args = parser.parse_args()

    model = load_model()

    if args.file:
        with open(args.file) as f:
            for line in f:
                seq = line.strip()
                if seq:
                    mic = predict(model, seq)
                    print(f"{seq}\t{mic:.3f}")
    elif args.interactive:
        print("Enter peptide sequences (Ctrl+C to exit):")
        while True:
            seq = input("> ").strip()
            if seq:
                mic = predict(model, seq)
                print(f"Predicted MIC: {mic:.3f}")
    elif args.sequence:
        mic = predict(model, args.sequence)
        print(f"Sequence: {args.sequence}")
        print(f"Predicted MIC: {mic:.3f}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

### scripts/sequence_nsga2.py (Skeleton)

```python
#!/usr/bin/env python3
"""Sequence-space NSGA-II optimization for AMP design.

Uses real peptide mutations instead of latent-space exploration.
Scores candidates with trained PeptideVAE (r=0.74).
"""

from dataclasses import dataclass
from typing import List, Tuple
import random

from deap import base, creator, tools, algorithms

@dataclass
class Peptide:
    sequence: str
    mic_pred: float
    toxicity_pred: float
    stability_score: float

class SequenceNSGA2:
    """NSGA-II optimizer in sequence space."""

    def __init__(
        self,
        seed_sequences: List[str],
        model_path: str = "checkpoints_definitive/best_production.pt",
        population_size: int = 100,
        generations: int = 50,
    ):
        self.seeds = seed_sequences
        self.model = self._load_model(model_path)
        self.pop_size = population_size
        self.n_gen = generations

    def mutate(self, sequence: str) -> str:
        """Apply single amino acid mutation."""
        # Implementation: random substitution, insertion, deletion
        ...

    def evaluate(self, sequence: str) -> Tuple[float, float, float]:
        """Evaluate objectives: activity, toxicity, stability."""
        mic = self.model.predict_mic(sequence)
        toxicity = self._predict_toxicity(sequence)
        stability = self._predict_stability(sequence)
        return (-mic, -toxicity, stability)  # Minimize MIC, minimize toxicity, maximize stability

    def run(self) -> List[Peptide]:
        """Run optimization, return Pareto front."""
        ...
```

---

## Timeline Estimate

| Phase | Tasks | Effort | Cumulative |
|-------|-------|--------|------------|
| Phase 1 | Prediction API | 2 hours | 2 hours |
| Phase 2 | Sequence NSGA-II | 9 hours | 11 hours |
| Phase 3 | Integration | 4 hours | 15 hours |

**Total:** ~2 working days

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Sequence length | 3 AA | 10-50 AA |
| Activity prediction | Heuristic | ML (r=0.74) |
| Valid AMP sequences | 0% | 100% |
| NSGA-II produces Pareto front | Yes | Yes |
| Integration tests pass | N/A | 100% |

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Decoder instability | Medium | High | Use Option A (no decoder) |
| Mutation space too small | Low | Medium | Add recombination operators |
| Slow inference | Low | Low | Batch predictions |
| Invalid sequences | Low | Medium | Validate AA composition |

---

## Decision Required

Before proceeding, confirm implementation approach:

- [ ] **Option A: Sequence Evolution** (Recommended) - Use PeptideVAE for scoring only
- [ ] **Option B: Latent Optimization** - Replace TernaryVAE with PeptideVAE decoder
- [ ] **Option C: Hybrid** - Encode seeds + latent perturbation

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-05 | 1.0 | Initial integration plan |
