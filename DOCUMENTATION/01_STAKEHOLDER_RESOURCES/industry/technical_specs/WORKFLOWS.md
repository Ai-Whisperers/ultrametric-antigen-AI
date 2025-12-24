# Workflows and Scripts

This guide explains how to perform common tasks using the scripts provided in the `scripts/` directory.

## 1. Training a Model

The core workflow is training the Ternary VAE to learn the 3-adic structure.

**Command:**

```bash
python -m scripts.train.train_ternary_v5_10 --config configs/ternary.yaml
```

**What it does:**

1.  Loads configuration from `configs/ternary.yaml`.
2.  Initializes `TernaryVAEV5_11` (Frozen Encoder).
3.  Runs the `HyperbolicVAETrainer`.
4.  Saves checkpoints to `results/checkpoints/`.
5.  Logs metrics to `results/training_runs/`.

## 2. Evaluating Coverage

Verify if the model has learned all 19,683 ternary operations.

**Command:**

```bash
python -m scripts.eval.evaluate_coverage --checkpoint results/checkpoints/best.pt
```

**Output:**

- Coverage percentage (Target: 100% for v5.11).
- Entropy metrics.

## 3. Benchmarking

Run performance benchmarks to test inference speed and stability.

**Command:**

```bash
python -m scripts.benchmark.run_benchmark
```

## 4. Visualization

Generate visualizations of the Poincare embeddings.

**Command:**

```bash
python -m scripts.visualization.plot_poincare --checkpoint results/checkpoints/best.pt
```

**Output:**

- Save plots to `results/discoveries/` (or configured output).

---

## Development Workflows

### Running Tests

We use `pytest` for integration testing.

```bash
pytest tests/
```

### Visualizing Training

Launch TensorBoard to view training curves.

```bash
tensorboard --logdir results/training_runs
```
