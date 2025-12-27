# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Training commands for Ternary VAE CLI."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config.paths import CHECKPOINTS_DIR, RESULTS_DIR

app = typer.Typer(help="Training commands for Ternary VAE models")
console = Console()


@app.command("run")
def train_run(
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to YAML configuration file",
    ),
    epochs: int = typer.Option(
        100,
        "--epochs", "-e",
        help="Number of training epochs",
    ),
    batch_size: int = typer.Option(
        512,
        "--batch-size", "-b",
        help="Training batch size",
    ),
    learning_rate: float = typer.Option(
        1e-3,
        "--lr",
        help="Learning rate",
    ),
    device: str = typer.Option(
        "cuda",
        "--device", "-d",
        help="Device to train on (cuda/cpu)",
    ),
    save_dir: Path = typer.Option(
        RESULTS_DIR / "training",
        "--save-dir", "-o",
        help="Directory to save checkpoints",
    ),
    partial_freeze: bool = typer.Option(
        True,
        "--partial-freeze/--no-partial-freeze",
        help="Use partial freeze architecture (frozen encoder_A)",
    ),
    curvature: float = typer.Option(
        1.0,
        "--curvature",
        help="Hyperbolic curvature parameter",
    ),
    max_radius: float = typer.Option(
        0.95,
        "--max-radius",
        help="Maximum Poincare ball radius",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose output",
    ),
):
    """Train a Ternary VAE model.

    Example:
        ternary-vae train run --config configs/ternary.yaml
        ternary-vae train run --epochs 200 --lr 5e-4
    """
    import torch

    # Check device availability
    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]CUDA not available, falling back to CPU[/yellow]")
        device = "cpu"

    console.print("[bold blue]Starting Ternary VAE Training[/bold blue]")
    console.print(f"  Device: {device}")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Learning rate: {learning_rate}")
    console.print(f"  Architecture: {'PartialFreeze' if partial_freeze else 'Standard'}")

    # Load config if provided
    training_config = {}
    if config and config.exists():
        import yaml
        with open(config) as f:
            training_config = yaml.safe_load(f)
        console.print(f"[green]Loaded config from {config}[/green]")

    # Import training components
    from src.data import generate_all_ternary_operations
    from src.models import TernaryVAEV5_11, TernaryVAEV5_11_PartialFreeze
    from src.training import TernaryVAETrainer
    from src import TrainingConfig

    # Generate data
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating ternary operations...", total=None)
        x, indices = generate_all_ternary_operations()
        x_tensor = torch.tensor(x, dtype=torch.float32)
        progress.update(task, completed=True)

    console.print(f"[green]Generated {len(x_tensor)} ternary operations[/green]")

    # Create model
    ModelClass = TernaryVAEV5_11_PartialFreeze if partial_freeze else TernaryVAEV5_11
    model = ModelClass(
        latent_dim=training_config.get("model", {}).get("latent_dim", 16),
        hidden_dim=training_config.get("model", {}).get("hidden_dim", 64),
        curvature=curvature,
        max_radius=max_radius,
    )
    model = model.to(device)

    console.print(f"[green]Model created with {sum(p.numel() for p in model.parameters()):,} parameters[/green]")

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build training config
    train_cfg = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )

    # Create trainer and train
    trainer = TernaryVAETrainer(model, train_cfg, device=device)

    console.print("[bold]Starting training...[/bold]")
    trainer.train(x_tensor)

    # Save final model
    final_path = save_dir / "final_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": training_config,
        "epochs": epochs,
    }, final_path)

    console.print(f"[bold green]Training complete! Model saved to {final_path}[/bold green]")


@app.command("hiv")
def train_hiv(
    config: Optional[Path] = typer.Option(
        None,
        "--config", "-c",
        help="Path to YAML configuration file",
    ),
    epochs: int = typer.Option(
        100,
        "--epochs", "-e",
        help="Number of training epochs",
    ),
    save_dir: Path = typer.Option(
        RESULTS_DIR / "hiv_training",
        "--save-dir", "-o",
        help="Directory to save results",
    ),
):
    """Train HIV-specific codon VAE model.

    Example:
        ternary-vae train hiv --epochs 200
    """
    console.print("[bold blue]HIV Codon VAE Training[/bold blue]")
    console.print("This command wraps scripts/train_codon_vae_hiv.py")

    import subprocess
    import sys

    cmd = [
        sys.executable,
        "scripts/train_codon_vae_hiv.py",
        "--epochs", str(epochs),
        "--save_dir", str(save_dir),
    ]
    if config:
        cmd.extend(["--config", str(config)])

    subprocess.run(cmd)


@app.command("resume")
def train_resume(
    checkpoint: Path = typer.Argument(
        ...,
        help="Path to checkpoint to resume from",
    ),
    epochs: int = typer.Option(
        50,
        "--epochs", "-e",
        help="Additional epochs to train",
    ),
):
    """Resume training from a checkpoint.

    Example:
        ternary-vae train resume results/training/checkpoint_epoch_50.pt --epochs 50
    """
    import torch

    if not checkpoint.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold blue]Resuming training from {checkpoint}[/bold blue]")

    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location="cpu")
    console.print(f"[green]Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}[/green]")

    # TODO: Implement full resume logic
    console.print("[yellow]Full resume implementation pending[/yellow]")


@app.callback()
def callback():
    """Training commands for Ternary VAE models."""
    pass
