# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for codon diffusion models."""

from __future__ import annotations

import pytest
import torch

from src.diffusion import (
    CodonDiffusion,
    ConditionalCodonDiffusion,
    PositionalEncoding,
    TimestepEmbedding,
    TransformerDenoiser,
)


class TestPositionalEncoding:
    """Tests for PositionalEncoding."""

    def test_init(self):
        """Test initialization."""
        pe = PositionalEncoding(d_model=64)
        assert pe.pe.shape == (1, 5000, 64)

    def test_forward_shape(self, device):
        """Test forward pass shape."""
        pe = PositionalEncoding(d_model=64)
        pe = pe.to(device)

        x = torch.randn(4, 100, 64, device=device)
        result = pe(x)

        assert result.shape == x.shape

    def test_different_positions_different(self, device):
        """Test that different positions have different encodings."""
        pe = PositionalEncoding(d_model=64)
        pe = pe.to(device)

        # First and second position should be different
        assert not torch.allclose(pe.pe[0, 0], pe.pe[0, 1])


class TestTimestepEmbedding:
    """Tests for TimestepEmbedding."""

    def test_init(self):
        """Test initialization."""
        emb = TimestepEmbedding(d_model=64)
        assert emb.d_model == 64

    def test_forward_shape(self, device):
        """Test forward pass shape."""
        emb = TimestepEmbedding(d_model=64)
        emb = emb.to(device)

        t = torch.randint(0, 1000, (4,), device=device)
        result = emb(t)

        assert result.shape == (4, 64)

    def test_different_timesteps_different(self, device):
        """Test that different timesteps have different embeddings."""
        emb = TimestepEmbedding(d_model=64)
        emb = emb.to(device)

        t = torch.tensor([0, 100, 500, 999], device=device)
        result = emb(t)

        # All embeddings should be different
        for i in range(4):
            for j in range(i + 1, 4):
                assert not torch.allclose(result[i], result[j])


class TestTransformerDenoiser:
    """Tests for TransformerDenoiser."""

    def test_init(self):
        """Test initialization."""
        model = TransformerDenoiser(vocab_size=64, d_model=128, n_layers=2)
        assert model.vocab_size == 64
        assert model.d_model == 128

    def test_forward_shape(self, device, codon_sequence, timesteps):
        """Test forward pass shape."""
        model = TransformerDenoiser(vocab_size=64, d_model=64, n_layers=2)
        model = model.to(device)

        logits = model(codon_sequence, timesteps)
        assert logits.shape == (4, 50, 64)

    def test_forward_with_context(self, device, codon_sequence, timesteps):
        """Test forward with context conditioning."""
        model = TransformerDenoiser(vocab_size=64, d_model=64, n_layers=2)
        model = model.to(device)

        context = torch.randn(4, 10, 64, device=device)
        logits = model(codon_sequence, timesteps, context=context)

        assert logits.shape == (4, 50, 64)


class TestCodonDiffusion:
    """Tests for CodonDiffusion model."""

    def test_init(self):
        """Test initialization."""
        model = CodonDiffusion(n_steps=100, vocab_size=64, hidden_dim=64, n_layers=2)
        assert model.n_steps == 100
        assert model.vocab_size == 64

    def test_forward_returns_loss(self, device, codon_sequence):
        """Test forward returns loss dictionary."""
        model = CodonDiffusion(n_steps=100, vocab_size=64, hidden_dim=32, n_layers=2)
        model = model.to(device)

        result = model.forward(codon_sequence)

        assert "loss" in result
        assert "accuracy" in result
        assert "logits" in result
        assert result["logits"].shape == (4, 50, 64)

    def test_forward_loss_is_finite(self, device, codon_sequence):
        """Test that loss is finite."""
        model = CodonDiffusion(n_steps=100, vocab_size=64, hidden_dim=32, n_layers=2)
        model = model.to(device)

        result = model.forward(codon_sequence)

        assert torch.isfinite(result["loss"])
        assert result["loss"] > 0

    def test_forward_with_timesteps(self, device, codon_sequence, timesteps):
        """Test forward with explicit timesteps."""
        model = CodonDiffusion(n_steps=1000, vocab_size=64, hidden_dim=32, n_layers=2)
        model = model.to(device)

        result = model.forward(codon_sequence, t=timesteps)
        assert torch.isfinite(result["loss"])

    def test_training_step(self, device, codon_sequence):
        """Test training step."""
        model = CodonDiffusion(n_steps=100, vocab_size=64, hidden_dim=32, n_layers=2)
        model = model.to(device)

        result = model.training_step(codon_sequence)
        assert "loss" in result

    def test_sample_shape(self, device):
        """Test sampling output shape."""
        model = CodonDiffusion(n_steps=10, vocab_size=64, hidden_dim=32, n_layers=2)
        model = model.to(device)
        model.eval()

        samples = model.sample(n_samples=2, seq_length=20, device=device)

        assert samples.shape == (2, 20)
        assert samples.dtype == torch.long
        assert torch.all(samples >= 0)
        assert torch.all(samples < 64)

    def test_sample_ddim_shape(self, device):
        """Test DDIM sampling output shape."""
        model = CodonDiffusion(n_steps=100, vocab_size=64, hidden_dim=32, n_layers=2)
        model = model.to(device)
        model.eval()

        samples = model.sample_ddim(n_samples=2, seq_length=20, n_steps=10, device=device)

        assert samples.shape == (2, 20)
        assert torch.all(samples >= 0)
        assert torch.all(samples < 64)

    def test_gradient_flow(self, device, codon_sequence):
        """Test that gradients flow through model."""
        model = CodonDiffusion(n_steps=100, vocab_size=64, hidden_dim=32, n_layers=2)
        model = model.to(device)

        result = model.forward(codon_sequence)
        result["loss"].backward()

        # Check gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad


class TestConditionalCodonDiffusion:
    """Tests for ConditionalCodonDiffusion."""

    def test_init(self):
        """Test initialization."""
        model = ConditionalCodonDiffusion(
            context_dim=128, n_steps=100, vocab_size=64, hidden_dim=64, n_layers=2
        )
        assert model.context_dim == 128

    def test_forward_with_context(self, device, codon_sequence):
        """Test forward with context."""
        model = ConditionalCodonDiffusion(
            context_dim=32, n_steps=100, vocab_size=64, hidden_dim=32, n_layers=2
        )
        model = model.to(device)

        context = torch.randn(4, 10, 32, device=device)
        result = model.forward(codon_sequence, context=context)

        assert "loss" in result
        assert torch.isfinite(result["loss"])

    def test_forward_without_context(self, device, codon_sequence):
        """Test forward without context."""
        model = ConditionalCodonDiffusion(
            context_dim=32, n_steps=100, vocab_size=64, hidden_dim=32, n_layers=2
        )
        model = model.to(device)

        result = model.forward(codon_sequence)
        assert "loss" in result

    def test_sample_with_context(self, device):
        """Test sampling with context."""
        model = ConditionalCodonDiffusion(
            context_dim=32, n_steps=10, vocab_size=64, hidden_dim=32, n_layers=2
        )
        model = model.to(device)
        model.eval()

        context = torch.randn(2, 5, 32, device=device)
        samples = model.sample(n_samples=2, seq_length=20, context=context, device=device)

        assert samples.shape == (2, 20)
