# n_pairs Hyperparameter Analysis

**Doc-Type:** Experimental Analysis · Version 1.0 · Updated 2025-12-23 · Author AI Whisperers

Analysis of geodesic loss `n_pairs` parameter impact on V5.11 training metrics.

---

## Experiment Setup

**baseline** - n_pairs=2000 (default)
**variants** - n_pairs=4000, n_pairs=8000
**epochs** - 10 (short run for signal detection)
**architecture** - Option C + dual projection + controller

---

## Results Summary

| Metric | n=2000 | n=4000 | n=8000 | Winner |
|:-------|-------:|-------:|-------:|:-------|
| Hierarchy A | -0.6543 | -0.6588 | -0.6560 | 8k (marginal) |
| Hierarchy B | -0.8315 | - | -0.8313 | 2k (marginal) |
| Dist Corr A | 0.4645 | 0.4531 | 0.4857 | **8k (+4.5%)** |
| Dist Corr B | 0.9198 | - | 0.9134 | 2k (-0.6%) |
| Radius v=0 | 0.8304 | - | 0.8215 | 2k |
| Radius v=9 | 0.1354 | 0.3117 | 0.1944 | 2k |
| Radius Range | 0.8728 | - | 0.8992 | 8k |

---

## Key Findings

**n_pairs=4000** - Inconclusive, appeared to be stochastic dip in distance correlation

**n_pairs=8000** - Clear signal above noise floor (~1%):
- VAE-A distance correlation improved +4.5% (0.465 → 0.486)
- VAE-A hierarchy marginally improved
- VAE-A radius range improved (better spread)
- VAE-B metrics slightly degraded
- Radius v=9 placement worse (0.135 → 0.194, target 0.10)

---

## Tradeoff Analysis

**benefits** - Better global pairwise structure learning in primary VAE (distance correlation computed over thousands of pairs)

**costs** - Worse placement of rarest point (v=9 is single data point), slight VAE-B degradation

**interpretation** - Since distance correlation is a global metric over all pairs while v=9 is one point, improved geodesic learning likely outweighs single-point regression. However, if both VAEs matter equally, the VAE-B degradation partially offsets the gain.

---

## Stochastic Noise Observations

Training runs show ~1-3% variation in most metrics due to:
- Random weight initialization
- Batch sampling randomness
- CUDA non-deterministic algorithms
- Potential hardware state influence (temperature, etc.)

Changes must exceed this noise floor to be considered significant.

---

## Recommendations

**if prioritizing VAE-A** - Use n_pairs=8000, commit as new default
**if both VAEs equal** - Keep n_pairs=2000, tradeoff unclear
**for further testing** - Run longer epochs (100+) to see if effects amplify or converge

---

## Related Parameters

**n_pairs** - Number of point pairs sampled per batch for geodesic loss
**location** - `scripts/train/train.py` line 112
**default** - 2000
**config key** - `loss.geodesic.n_pairs` in `configs/ternary.yaml`
