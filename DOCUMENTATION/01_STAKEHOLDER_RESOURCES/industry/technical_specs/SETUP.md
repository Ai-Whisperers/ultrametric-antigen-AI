# Setup and Dependencies

This guide covers how to set up the Ternary VAE v5.11 environment involving the Frozen Encoder architecture.

**Version:** 5.11.0
**Python:** 3.8+ (Tested on 3.10)

## Quick Start (Production)

```bash
# 1. Clone
git clone https://github.com/ai-whisperers/ternary-vae.git
cd ternary-vaes-bioinformatics

# 2. Virtual Environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# 3. Install Dependencies
pip install -r requirements.txt
```

## Developer Installation (Editable)

If you plan to modify the code in `src/`:

```bash
pip install -e .[dev]
```

This installs the package in editable mode with development tools (pytest, etc.).

## Environment Configuration

The repository uses `configs/ternary.yaml` for model config, but environment variables for hardware settings.
Copy the example template:

```bash
cp configs/env.example .env
```

**Key Variables (.env):**

- `DEVICE=cuda` (or `cpu`)
- `CHECKPOINT_DIR=results/checkpoints` (Updated location)
- `LOG_DIR=results/logs`

## Verifying Installation

Run the coverage evaluation script to check if the model loads.

```bash
python -m scripts.eval.evaluate_coverage --help
```

## Dependencies Explained

- **Core:** `torch` (Neural Nets), `numpy` (Math), `scipy` (Stats).
- **Config:** `pyyaml`.
- **Logging:** `tensorboard` (Visualize training).

See [requirements.txt](../../requirements.txt) for exact versions.
