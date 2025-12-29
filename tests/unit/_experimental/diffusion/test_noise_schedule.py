# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for noise schedulers."""

from __future__ import annotations

import pytest
import torch

from src.diffusion import DiscreteNoiseScheduler, NoiseScheduler


class TestNoiseScheduler:
    """Tests for continuous NoiseScheduler."""

    def test_init_default(self):
        """Test default initialization."""
        scheduler = NoiseScheduler()
        assert scheduler.n_steps == 1000
        assert scheduler.schedule_type == "cosine"

    def test_init_linear(self):
        """Test linear schedule initialization."""
        scheduler = NoiseScheduler(n_steps=500, schedule_type="linear")
        assert scheduler.n_steps == 500
        assert scheduler.betas.shape == (500,)

    def test_init_sigmoid(self):
        """Test sigmoid schedule."""
        scheduler = NoiseScheduler(schedule_type="sigmoid")
        assert torch.all(scheduler.betas > 0)
        assert torch.all(scheduler.betas < 1)

    def test_init_exponential(self):
        """Test exponential schedule."""
        scheduler = NoiseScheduler(schedule_type="exponential")
        assert torch.all(scheduler.betas > 0)

    def test_betas_valid_range(self):
        """Test that betas are in valid range."""
        for schedule in ["linear", "cosine", "sigmoid", "exponential"]:
            scheduler = NoiseScheduler(schedule_type=schedule)
            assert torch.all(scheduler.betas >= 0.0001)
            assert torch.all(scheduler.betas <= 0.999)

    def test_alphas_cumprod_decreasing(self):
        """Test that alpha_bar decreases over time."""
        scheduler = NoiseScheduler()
        diffs = scheduler.alphas_cumprod[1:] - scheduler.alphas_cumprod[:-1]
        assert torch.all(diffs <= 0)

    def test_add_noise_shape(self, device, continuous_data):
        """Test add_noise output shape."""
        scheduler = NoiseScheduler(n_steps=100)
        scheduler = scheduler.to(device)

        t = torch.randint(0, 100, (4,), device=device)
        noised, noise = scheduler.add_noise(continuous_data, t)

        assert noised.shape == continuous_data.shape
        assert noise.shape == continuous_data.shape

    def test_add_noise_with_custom_noise(self, device, continuous_data):
        """Test add_noise with pre-generated noise."""
        scheduler = NoiseScheduler(n_steps=100)
        scheduler = scheduler.to(device)

        t = torch.randint(0, 100, (4,), device=device)
        custom_noise = torch.ones_like(continuous_data)

        noised, noise = scheduler.add_noise(continuous_data, t, noise=custom_noise)

        assert torch.allclose(noise, custom_noise)

    def test_remove_noise(self, device, continuous_data):
        """Test noise removal prediction."""
        scheduler = NoiseScheduler(n_steps=100)
        scheduler = scheduler.to(device)

        t = torch.randint(0, 100, (4,), device=device)
        noise = torch.randn_like(continuous_data)

        # Add noise then remove it
        noised, _ = scheduler.add_noise(continuous_data, t, noise=noise)
        reconstructed = scheduler.remove_noise(noised, noise, t)

        # Should approximately recover original
        assert torch.allclose(reconstructed, continuous_data, atol=1e-5)

    def test_posterior_mean(self, device, continuous_data):
        """Test posterior mean computation."""
        scheduler = NoiseScheduler(n_steps=100)
        scheduler = scheduler.to(device)

        t = torch.randint(1, 100, (4,), device=device)  # t > 0 for posterior
        x_start = continuous_data
        x_t = torch.randn_like(continuous_data)

        mean = scheduler.posterior_mean(x_start, x_t, t)
        assert mean.shape == continuous_data.shape

    def test_step(self, device, continuous_data):
        """Test reverse diffusion step."""
        scheduler = NoiseScheduler(n_steps=100)
        scheduler = scheduler.to(device)

        t = torch.full((4,), 50, dtype=torch.long, device=device)
        model_output = torch.randn_like(continuous_data)  # Predicted noise

        x_prev = scheduler.step(model_output, t, continuous_data)
        assert x_prev.shape == continuous_data.shape


class TestDiscreteNoiseScheduler:
    """Tests for DiscreteNoiseScheduler."""

    def test_init_default(self):
        """Test default initialization."""
        scheduler = DiscreteNoiseScheduler()
        assert scheduler.n_steps == 1000
        assert scheduler.vocab_size == 64
        assert scheduler.absorbing_state == 63

    def test_init_custom(self):
        """Test custom initialization."""
        scheduler = DiscreteNoiseScheduler(n_steps=500, vocab_size=32)
        assert scheduler.n_steps == 500
        assert scheduler.vocab_size == 32
        assert scheduler.absorbing_state == 31

    def test_stay_probs_decreasing(self):
        """Test that stay probabilities decrease over time."""
        scheduler = DiscreteNoiseScheduler()
        diffs = scheduler.stay_probs_cumprod[1:] - scheduler.stay_probs_cumprod[:-1]
        assert torch.all(diffs <= 0)

    def test_add_noise_preserves_some_tokens(self, device, codon_sequence):
        """Test that some tokens are preserved (not all become absorbing)."""
        scheduler = DiscreteNoiseScheduler(n_steps=100)
        scheduler = scheduler.to(device)

        # Low timestep should preserve most tokens
        t = torch.zeros(4, dtype=torch.long, device=device)
        noised = scheduler.add_noise(codon_sequence, t)

        # At t=0, most tokens should be preserved
        preserved = (noised == codon_sequence).float().mean()
        assert preserved > 0.5

    def test_add_noise_more_noise_at_high_t(self, device, codon_sequence):
        """Test that more tokens become absorbing at higher t."""
        scheduler = DiscreteNoiseScheduler(n_steps=100)
        scheduler = scheduler.to(device)

        # Compare low vs high timestep
        t_low = torch.full((4,), 10, dtype=torch.long, device=device)
        t_high = torch.full((4,), 90, dtype=torch.long, device=device)

        noised_low = scheduler.add_noise(codon_sequence, t_low)
        noised_high = scheduler.add_noise(codon_sequence, t_high)

        absorbing_low = (noised_low == scheduler.absorbing_state).float().mean()
        absorbing_high = (noised_high == scheduler.absorbing_state).float().mean()

        # Higher timestep should have more absorbing tokens
        assert absorbing_high > absorbing_low

    def test_add_noise_shape(self, device, codon_sequence):
        """Test that shape is preserved."""
        scheduler = DiscreteNoiseScheduler()
        scheduler = scheduler.to(device)

        t = torch.randint(0, 1000, (4,), device=device)
        noised = scheduler.add_noise(codon_sequence, t)

        assert noised.shape == codon_sequence.shape
        assert noised.dtype == codon_sequence.dtype

    def test_posterior_distribution_shape(self, device, codon_sequence):
        """Test posterior distribution computation."""
        scheduler = DiscreteNoiseScheduler(n_steps=100, vocab_size=64)
        scheduler = scheduler.to(device)

        t = torch.randint(1, 100, (4,), device=device)
        logits = torch.randn(4, 50, 64, device=device)

        posterior = scheduler.posterior_distribution(codon_sequence, logits, t)
        assert posterior.shape == (4, 50, 64)
