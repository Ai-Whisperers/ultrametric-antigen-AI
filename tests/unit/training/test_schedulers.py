"""Tests for src/training/schedulers.py - Parameter schedulers for training.

This module provides comprehensive tests for:
- linear_schedule: Linear interpolation between values
- cyclic_schedule: Periodic oscillation scheduling
- TemperatureScheduler: Temperature annealing for VAEs
- BetaScheduler: KL weight scheduling with warmup
- LearningRateScheduler: Step-based learning rate scheduling
"""

import math
from typing import Any, Dict

import pytest

from src.training.schedulers import (
    BetaScheduler,
    LearningRateScheduler,
    TemperatureScheduler,
    cyclic_schedule,
    linear_schedule,
)


# =============================================================================
# Linear Schedule Tests
# =============================================================================


class TestLinearSchedule:
    """Tests for the linear_schedule function."""

    def test_start_value_at_epoch_zero(self):
        """At epoch 0, should return start_val."""
        result = linear_schedule(epoch=0, start_val=1.0, end_val=0.0, total_epochs=100)
        assert result == pytest.approx(1.0)

    def test_end_value_at_final_epoch(self):
        """At total_epochs, should return end_val."""
        result = linear_schedule(epoch=100, start_val=1.0, end_val=0.0, total_epochs=100)
        assert result == pytest.approx(0.0)

    def test_midpoint_value(self):
        """At midpoint, should return average of start and end."""
        result = linear_schedule(epoch=50, start_val=1.0, end_val=0.0, total_epochs=100)
        assert result == pytest.approx(0.5)

    def test_increasing_schedule(self):
        """Should work for increasing values (start < end)."""
        result = linear_schedule(epoch=50, start_val=0.0, end_val=1.0, total_epochs=100)
        assert result == pytest.approx(0.5)

    def test_negative_values(self):
        """Should handle negative values correctly."""
        result = linear_schedule(epoch=50, start_val=-1.0, end_val=1.0, total_epochs=100)
        assert result == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "epoch,expected",
        [
            (0, 1.0),
            (25, 0.75),
            (50, 0.5),
            (75, 0.25),
            (100, 0.0),
        ],
    )
    def test_linear_interpolation_parametrized(self, epoch: int, expected: float):
        """Verify linear interpolation at multiple points."""
        result = linear_schedule(epoch=epoch, start_val=1.0, end_val=0.0, total_epochs=100)
        assert result == pytest.approx(expected)

    def test_start_epoch_before_threshold(self):
        """Before start_epoch, should return start_val."""
        result = linear_schedule(epoch=5, start_val=1.0, end_val=0.0, total_epochs=100, start_epoch=10)
        assert result == pytest.approx(1.0)

    def test_start_epoch_at_threshold(self):
        """At start_epoch, should return start_val (progress = 0)."""
        result = linear_schedule(epoch=10, start_val=1.0, end_val=0.0, total_epochs=100, start_epoch=10)
        assert result == pytest.approx(1.0)

    def test_start_epoch_after_threshold(self):
        """After start_epoch, schedule should progress normally."""
        result = linear_schedule(epoch=60, start_val=1.0, end_val=0.0, total_epochs=100, start_epoch=10)
        # progress = (60 - 10) / 100 = 0.5
        assert result == pytest.approx(0.5)

    def test_epoch_beyond_total_epochs(self):
        """Epochs beyond total_epochs should clamp to end_val."""
        result = linear_schedule(epoch=150, start_val=1.0, end_val=0.0, total_epochs=100)
        assert result == pytest.approx(0.0)

    def test_zero_total_epochs_returns_end_val(self):
        """Edge case: zero total epochs should return end_val (progress = 1)."""
        # This would cause division by zero without min(..., 1.0) clamping
        # But (epoch - start_epoch) / 0 would be inf, min(inf, 1.0) = 1.0
        # Actually Python raises ZeroDivisionError
        with pytest.raises(ZeroDivisionError):
            linear_schedule(epoch=0, start_val=1.0, end_val=0.0, total_epochs=0)


# =============================================================================
# Cyclic Schedule Tests
# =============================================================================


class TestCyclicSchedule:
    """Tests for the cyclic_schedule function."""

    def test_peak_at_epoch_zero(self):
        """At epoch 0, cos(0) = 1, so value = base + amplitude."""
        result = cyclic_schedule(epoch=0, base_val=1.0, amplitude=0.5, period=10)
        assert result == pytest.approx(1.5)

    def test_trough_at_half_period(self):
        """At half period, cos(pi) = -1, so value = base - amplitude."""
        result = cyclic_schedule(epoch=5, base_val=1.0, amplitude=0.5, period=10)
        assert result == pytest.approx(0.5)

    def test_return_to_peak_at_full_period(self):
        """At full period, should return to initial value."""
        result = cyclic_schedule(epoch=10, base_val=1.0, amplitude=0.5, period=10)
        assert result == pytest.approx(1.5)

    def test_quarter_period(self):
        """At quarter period, cos(pi/2) = 0, so value = base."""
        result = cyclic_schedule(epoch=2, base_val=1.0, amplitude=0.5, period=8)
        # phase = (2 % 8) / 8 * 2 * pi = 0.25 * 2 * pi = pi/2
        # cos(pi/2) = 0
        assert result == pytest.approx(1.0)

    @pytest.mark.parametrize("period", [1, 5, 10, 50, 100])
    def test_periodicity(self, period: int):
        """Value should repeat after each period."""
        val_0 = cyclic_schedule(epoch=0, base_val=1.0, amplitude=0.5, period=period)
        val_period = cyclic_schedule(epoch=period, base_val=1.0, amplitude=0.5, period=period)
        val_2period = cyclic_schedule(epoch=2 * period, base_val=1.0, amplitude=0.5, period=period)

        assert val_0 == pytest.approx(val_period)
        assert val_0 == pytest.approx(val_2period)

    def test_zero_amplitude_returns_base(self):
        """With zero amplitude, should always return base value."""
        for epoch in [0, 5, 10, 25, 100]:
            result = cyclic_schedule(epoch=epoch, base_val=1.0, amplitude=0.0, period=10)
            assert result == pytest.approx(1.0)

    def test_negative_base_value(self):
        """Should work with negative base values."""
        result = cyclic_schedule(epoch=0, base_val=-1.0, amplitude=0.5, period=10)
        assert result == pytest.approx(-0.5)  # -1.0 + 0.5

    def test_large_epoch_value(self):
        """Should handle large epoch values correctly (modulo operation)."""
        result = cyclic_schedule(epoch=1000, base_val=1.0, amplitude=0.5, period=10)
        result_0 = cyclic_schedule(epoch=0, base_val=1.0, amplitude=0.5, period=10)
        assert result == pytest.approx(result_0)  # 1000 % 10 = 0


# =============================================================================
# TemperatureScheduler Tests
# =============================================================================


class TestTemperatureScheduler:
    """Tests for the TemperatureScheduler class."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Create a basic configuration for testing."""
        return {
            "total_epochs": 100,
            "vae_a": {
                "temp_start": 1.0,
                "temp_end": 0.1,
                "temp_cyclic": False,
            },
            "vae_b": {
                "temp_start": 1.0,
                "temp_end": 0.1,
            },
        }

    @pytest.fixture
    def cyclic_config(self) -> Dict[str, Any]:
        """Create a configuration with cyclic temperature."""
        return {
            "total_epochs": 100,
            "vae_a": {
                "temp_start": 1.0,
                "temp_end": 0.1,
                "temp_cyclic": True,
                "temp_boost_amplitude": 0.3,
            },
            "vae_b": {
                "temp_start": 1.0,
                "temp_end": 0.1,
            },
        }

    @pytest.fixture
    def phase4_config(self) -> Dict[str, Any]:
        """Create a configuration with Phase 4 settings."""
        return {
            "total_epochs": 100,
            "vae_a": {
                "temp_start": 1.0,
                "temp_end": 0.1,
                "temp_cyclic": True,
                "temp_boost_amplitude": 0.5,
            },
            "vae_b": {
                "temp_start": 1.0,
                "temp_end": 0.1,
                "temp_phase4": 0.5,
            },
        }

    def test_vae_a_temperature_at_start(self, basic_config: Dict[str, Any]):
        """VAE-A should start at temp_start."""
        scheduler = TemperatureScheduler(basic_config, phase_4_start=80)
        temp = scheduler.get_temperature(epoch=0, vae="A")
        assert temp == pytest.approx(1.0)

    def test_vae_a_temperature_at_end(self, basic_config: Dict[str, Any]):
        """VAE-A should end at temp_end."""
        scheduler = TemperatureScheduler(basic_config, phase_4_start=80)
        temp = scheduler.get_temperature(epoch=100, vae="A")
        assert temp == pytest.approx(0.1)

    def test_vae_a_temperature_linear_midpoint(self, basic_config: Dict[str, Any]):
        """VAE-A should be at midpoint value at midpoint epoch."""
        scheduler = TemperatureScheduler(basic_config, phase_4_start=80)
        temp = scheduler.get_temperature(epoch=50, vae="A")
        expected = 1.0 + (0.1 - 1.0) * 0.5  # = 0.55
        assert temp == pytest.approx(expected)

    def test_vae_b_temperature_at_start(self, basic_config: Dict[str, Any]):
        """VAE-B should start at temp_start."""
        scheduler = TemperatureScheduler(basic_config, phase_4_start=80)
        temp = scheduler.get_temperature(epoch=0, vae="B")
        assert temp == pytest.approx(1.0)

    def test_vae_b_temperature_with_lag(self, basic_config: Dict[str, Any]):
        """VAE-B temperature should be delayed by temp_lag."""
        scheduler = TemperatureScheduler(basic_config, phase_4_start=80, temp_lag=10)
        # At epoch 10 with lag 10, effective epoch = 0
        temp = scheduler.get_temperature(epoch=10, vae="B")
        assert temp == pytest.approx(1.0)

    def test_vae_b_temperature_lag_clamps_to_zero(self, basic_config: Dict[str, Any]):
        """VAE-B lagged epoch should not go negative."""
        scheduler = TemperatureScheduler(basic_config, phase_4_start=80, temp_lag=50)
        # At epoch 10 with lag 50, effective epoch = max(0, 10-50) = 0
        temp = scheduler.get_temperature(epoch=10, vae="B")
        assert temp == pytest.approx(1.0)

    def test_vae_a_cyclic_temperature(self, cyclic_config: Dict[str, Any]):
        """VAE-A with cyclic should oscillate around base temperature."""
        scheduler = TemperatureScheduler(cyclic_config, phase_4_start=80)
        temp_0 = scheduler.get_temperature(epoch=0, vae="A")
        temp_15 = scheduler.get_temperature(epoch=15, vae="A")

        # Should be different due to cyclic modulation
        assert temp_0 != temp_15

    def test_vae_a_cyclic_minimum_temperature(self, cyclic_config: Dict[str, Any]):
        """Cyclic temperature should not go below 0.1."""
        scheduler = TemperatureScheduler(cyclic_config, phase_4_start=80)
        # Test at various epochs
        for epoch in range(100):
            temp = scheduler.get_temperature(epoch=epoch, vae="A")
            assert temp >= 0.1

    def test_vae_a_phase4_boost(self, phase4_config: Dict[str, Any]):
        """VAE-A should use temp_boost_amplitude in Phase 4."""
        scheduler = TemperatureScheduler(phase4_config, phase_4_start=80)

        # In Phase 4, amplitude should be temp_boost_amplitude
        temp_phase4 = scheduler.get_temperature(epoch=85, vae="A")
        # This should use the boosted amplitude
        assert temp_phase4 > 0

    def test_vae_b_phase4_uses_temp_phase4(self, phase4_config: Dict[str, Any]):
        """VAE-B should use temp_phase4 value in Phase 4."""
        scheduler = TemperatureScheduler(phase4_config, phase_4_start=80)
        temp = scheduler.get_temperature(epoch=85, vae="B")
        assert temp == pytest.approx(0.5)

    def test_vae_b_phase4_falls_back_to_temp_end(self, basic_config: Dict[str, Any]):
        """VAE-B should use temp_end if temp_phase4 not specified."""
        scheduler = TemperatureScheduler(basic_config, phase_4_start=80)
        temp = scheduler.get_temperature(epoch=85, vae="B")
        assert temp == pytest.approx(0.1)


# =============================================================================
# BetaScheduler Tests
# =============================================================================


class TestBetaScheduler:
    """Tests for the BetaScheduler class."""

    @pytest.fixture
    def basic_config(self) -> Dict[str, Any]:
        """Create a basic configuration without warmup."""
        return {
            "total_epochs": 100,
            "vae_a": {
                "beta_start": 1.0,
                "beta_end": 0.5,
            },
            "vae_b": {
                "beta_start": 0.8,
                "beta_end": 0.4,
            },
        }

    @pytest.fixture
    def warmup_config(self) -> Dict[str, Any]:
        """Create a configuration with warmup."""
        return {
            "total_epochs": 100,
            "vae_a": {
                "beta_start": 1.0,
                "beta_end": 0.5,
                "beta_warmup_epochs": 10,
            },
            "vae_b": {
                "beta_start": 0.8,
                "beta_end": 0.4,
                "beta_warmup_epochs": 20,
            },
        }

    def test_vae_a_beta_without_warmup_at_start(self, basic_config: Dict[str, Any]):
        """VAE-A without warmup should start at beta_start."""
        scheduler = BetaScheduler(basic_config)
        beta = scheduler.get_beta(epoch=0, vae="A")
        assert beta == pytest.approx(1.0)

    def test_vae_a_beta_without_warmup_at_end(self, basic_config: Dict[str, Any]):
        """VAE-A without warmup should end at beta_end."""
        scheduler = BetaScheduler(basic_config)
        beta = scheduler.get_beta(epoch=100, vae="A")
        assert beta == pytest.approx(0.5)

    def test_vae_a_beta_with_warmup_starts_at_zero(self, warmup_config: Dict[str, Any]):
        """VAE-A with warmup should start at 0."""
        scheduler = BetaScheduler(warmup_config)
        beta = scheduler.get_beta(epoch=0, vae="A")
        assert beta == pytest.approx(0.0)

    def test_vae_a_beta_warmup_progression(self, warmup_config: Dict[str, Any]):
        """VAE-A beta should linearly increase during warmup."""
        scheduler = BetaScheduler(warmup_config)

        # At warmup midpoint
        beta_mid = scheduler.get_beta(epoch=5, vae="A")
        expected_mid = (5 / 10) * 1.0  # = 0.5
        assert beta_mid == pytest.approx(expected_mid)

    def test_vae_a_beta_after_warmup(self, warmup_config: Dict[str, Any]):
        """VAE-A beta should reach beta_start at end of warmup."""
        scheduler = BetaScheduler(warmup_config)
        beta = scheduler.get_beta(epoch=10, vae="A")
        # After warmup, should start at beta_start and decay
        # At epoch 10, warmup just ended, progress = 0
        assert beta == pytest.approx(1.0)

    def test_vae_b_beta_with_phase_lag(self, basic_config: Dict[str, Any]):
        """VAE-B beta should be scaled by sin(beta_phase_lag)."""
        scheduler = BetaScheduler(basic_config, beta_phase_lag=math.pi / 2)
        # sin(pi/2) = 1, so VAE-B should equal VAE-A
        beta_a = scheduler.get_beta(epoch=50, vae="A")
        beta_b = scheduler.get_beta(epoch=50, vae="B")
        assert beta_b == pytest.approx(beta_a)

    def test_vae_b_beta_phase_lag_zero(self, basic_config: Dict[str, Any]):
        """VAE-B with zero phase lag should be zero after warmup."""
        scheduler = BetaScheduler(basic_config, beta_phase_lag=0.0)
        # sin(0) = 0
        beta = scheduler.get_beta(epoch=50, vae="B")
        assert beta == pytest.approx(0.0)

    def test_vae_b_beta_warmup(self, warmup_config: Dict[str, Any]):
        """VAE-B should warm up independently."""
        scheduler = BetaScheduler(warmup_config, beta_phase_lag=math.pi / 2)

        # During warmup
        beta_5 = scheduler.get_beta(epoch=5, vae="B")
        expected = (5 / 20) * 0.8  # = 0.2
        assert beta_5 == pytest.approx(expected)

    @pytest.mark.parametrize("epoch", [0, 25, 50, 75, 100])
    def test_beta_monotonic_decay_no_warmup(self, basic_config: Dict[str, Any], epoch: int):
        """Beta should monotonically decay without warmup."""
        scheduler = BetaScheduler(basic_config)
        beta = scheduler.get_beta(epoch=epoch, vae="A")

        # Should be between start and end
        assert 0.5 <= beta <= 1.0

    def test_beta_warmup_linearity(self, warmup_config: Dict[str, Any]):
        """Beta warmup should be strictly linear."""
        scheduler = BetaScheduler(warmup_config)

        betas = [scheduler.get_beta(epoch=e, vae="A") for e in range(10)]
        # Each step should increase by same amount
        diffs = [betas[i + 1] - betas[i] for i in range(len(betas) - 1)]

        for diff in diffs:
            assert diff == pytest.approx(diffs[0])


# =============================================================================
# LearningRateScheduler Tests
# =============================================================================


class TestLearningRateScheduler:
    """Tests for the LearningRateScheduler class."""

    @pytest.fixture
    def simple_schedule(self):
        """Create a simple step schedule."""
        return [
            {"epoch": 0, "lr": 0.001},
            {"epoch": 50, "lr": 0.0005},
            {"epoch": 80, "lr": 0.0001},
        ]

    @pytest.fixture
    def single_lr_schedule(self):
        """Create a schedule with single learning rate."""
        return [{"epoch": 0, "lr": 0.001}]

    def test_initial_learning_rate(self, simple_schedule):
        """Should return initial LR at epoch 0."""
        scheduler = LearningRateScheduler(simple_schedule)
        lr = scheduler.get_lr(epoch=0)
        assert lr == pytest.approx(0.001)

    def test_first_step(self, simple_schedule):
        """Should use new LR after first step."""
        scheduler = LearningRateScheduler(simple_schedule)
        lr = scheduler.get_lr(epoch=50)
        assert lr == pytest.approx(0.0005)

    def test_second_step(self, simple_schedule):
        """Should use final LR after second step."""
        scheduler = LearningRateScheduler(simple_schedule)
        lr = scheduler.get_lr(epoch=80)
        assert lr == pytest.approx(0.0001)

    def test_between_steps(self, simple_schedule):
        """Should maintain previous LR between steps."""
        scheduler = LearningRateScheduler(simple_schedule)
        lr = scheduler.get_lr(epoch=65)
        assert lr == pytest.approx(0.0005)

    def test_before_first_step(self, simple_schedule):
        """Should use first LR before any steps."""
        scheduler = LearningRateScheduler(simple_schedule)
        lr = scheduler.get_lr(epoch=25)
        assert lr == pytest.approx(0.001)

    def test_after_last_step(self, simple_schedule):
        """Should maintain final LR after all steps."""
        scheduler = LearningRateScheduler(simple_schedule)
        lr = scheduler.get_lr(epoch=100)
        assert lr == pytest.approx(0.0001)

    def test_single_lr(self, single_lr_schedule):
        """Should work with single LR (no steps)."""
        scheduler = LearningRateScheduler(single_lr_schedule)

        for epoch in [0, 50, 100, 500]:
            lr = scheduler.get_lr(epoch=epoch)
            assert lr == pytest.approx(0.001)

    @pytest.mark.parametrize(
        "epoch,expected_lr",
        [
            (0, 0.001),
            (25, 0.001),
            (49, 0.001),
            (50, 0.0005),
            (51, 0.0005),
            (79, 0.0005),
            (80, 0.0001),
            (100, 0.0001),
        ],
    )
    def test_lr_at_boundaries(self, simple_schedule, epoch: int, expected_lr: float):
        """Test LR values at and around step boundaries."""
        scheduler = LearningRateScheduler(simple_schedule)
        lr = scheduler.get_lr(epoch=epoch)
        assert lr == pytest.approx(expected_lr)

    def test_increasing_lr_schedule(self):
        """Should support increasing LR (warmup pattern)."""
        schedule = [
            {"epoch": 0, "lr": 0.0001},
            {"epoch": 10, "lr": 0.001},
            {"epoch": 50, "lr": 0.0001},
        ]
        scheduler = LearningRateScheduler(schedule)

        lr_0 = scheduler.get_lr(0)
        lr_15 = scheduler.get_lr(15)
        lr_60 = scheduler.get_lr(60)

        assert lr_0 == pytest.approx(0.0001)
        assert lr_15 == pytest.approx(0.001)
        assert lr_60 == pytest.approx(0.0001)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_linear_schedule_same_start_end(self):
        """Linear schedule with same start and end should be constant."""
        for epoch in [0, 25, 50, 100]:
            result = linear_schedule(epoch=epoch, start_val=0.5, end_val=0.5, total_epochs=100)
            assert result == pytest.approx(0.5)

    def test_cyclic_schedule_period_one(self):
        """Cyclic schedule with period 1 should always be at peak."""
        for epoch in range(10):
            result = cyclic_schedule(epoch=epoch, base_val=1.0, amplitude=0.5, period=1)
            assert result == pytest.approx(1.5)

    def test_temperature_scheduler_phase_4_at_epoch_zero(self):
        """Phase 4 starting at epoch 0 should work."""
        config = {
            "total_epochs": 100,
            "vae_a": {"temp_start": 1.0, "temp_end": 0.1, "temp_cyclic": False},
            "vae_b": {"temp_start": 1.0, "temp_end": 0.1, "temp_phase4": 0.5},
        }
        scheduler = TemperatureScheduler(config, phase_4_start=0)

        # VAE-B should use temp_phase4 from epoch 0
        temp = scheduler.get_temperature(epoch=0, vae="B")
        assert temp == pytest.approx(0.5)

    def test_beta_scheduler_zero_warmup_epochs(self):
        """Zero warmup epochs should skip warmup phase."""
        config = {
            "total_epochs": 100,
            "vae_a": {"beta_start": 1.0, "beta_end": 0.5, "beta_warmup_epochs": 0},
            "vae_b": {"beta_start": 0.8, "beta_end": 0.4, "beta_warmup_epochs": 0},
        }
        scheduler = BetaScheduler(config)

        # Should start at beta_start, not 0
        beta = scheduler.get_beta(epoch=0, vae="A")
        assert beta == pytest.approx(1.0)

    def test_lr_scheduler_empty_schedule_raises(self):
        """Empty LR schedule should raise IndexError."""
        scheduler = LearningRateScheduler([])
        with pytest.raises(IndexError):
            scheduler.get_lr(epoch=0)

    def test_linear_schedule_negative_epoch(self):
        """Negative epochs before start_epoch should return start_val."""
        result = linear_schedule(epoch=-5, start_val=1.0, end_val=0.0, total_epochs=100)
        # epoch=-5 < start_epoch=0, so returns start_val
        assert result == pytest.approx(1.0)

    def test_temperature_scheduler_vae_b_no_lag(self):
        """VAE-B with no lag should behave like VAE-A in early phases."""
        config = {
            "total_epochs": 100,
            "vae_a": {"temp_start": 1.0, "temp_end": 0.1, "temp_cyclic": False},
            "vae_b": {"temp_start": 1.0, "temp_end": 0.1},
        }
        scheduler = TemperatureScheduler(config, phase_4_start=80, temp_lag=0)

        temp_a = scheduler.get_temperature(epoch=50, vae="A")
        temp_b = scheduler.get_temperature(epoch=50, vae="B")

        assert temp_a == pytest.approx(temp_b)

    def test_beta_scheduler_very_large_epoch(self):
        """Scheduler should handle very large epoch values."""
        config = {
            "total_epochs": 100,
            "vae_a": {"beta_start": 1.0, "beta_end": 0.5},
            "vae_b": {"beta_start": 0.8, "beta_end": 0.4},
        }
        scheduler = BetaScheduler(config)

        # Should clamp to end value
        beta = scheduler.get_beta(epoch=10000, vae="A")
        assert beta == pytest.approx(0.5)


# =============================================================================
# Integration-like Tests
# =============================================================================


class TestSchedulerIntegration:
    """Integration tests for scheduler interactions."""

    def test_full_training_simulation(self):
        """Simulate a full training run with all schedulers."""
        config = {
            "total_epochs": 100,
            "vae_a": {
                "temp_start": 1.0,
                "temp_end": 0.1,
                "temp_cyclic": True,
                "beta_start": 1.0,
                "beta_end": 0.5,
                "beta_warmup_epochs": 10,
            },
            "vae_b": {
                "temp_start": 1.0,
                "temp_end": 0.1,
                "beta_start": 0.8,
                "beta_end": 0.4,
                "beta_warmup_epochs": 10,
            },
        }

        lr_schedule = [
            {"epoch": 0, "lr": 0.001},
            {"epoch": 50, "lr": 0.0005},
            {"epoch": 80, "lr": 0.0001},
        ]

        temp_scheduler = TemperatureScheduler(config, phase_4_start=80)
        beta_scheduler = BetaScheduler(config)
        lr_scheduler = LearningRateScheduler(lr_schedule)

        # Run through all epochs
        for epoch in range(100):
            temp_a = temp_scheduler.get_temperature(epoch, "A")
            temp_b = temp_scheduler.get_temperature(epoch, "B")
            beta_a = beta_scheduler.get_beta(epoch, "A")
            beta_b = beta_scheduler.get_beta(epoch, "B")
            lr = lr_scheduler.get_lr(epoch)

            # All values should be positive and finite
            assert temp_a > 0 and math.isfinite(temp_a)
            assert temp_b > 0 and math.isfinite(temp_b)
            assert beta_a >= 0 and math.isfinite(beta_a)
            assert beta_b >= 0 and math.isfinite(beta_b)
            assert lr > 0 and math.isfinite(lr)

    def test_scheduler_consistency_across_vae_types(self):
        """Both VAE types should be handled consistently."""
        config = {
            "total_epochs": 100,
            "vae_a": {"temp_start": 1.0, "temp_end": 0.1, "temp_cyclic": False},
            "vae_b": {"temp_start": 1.0, "temp_end": 0.1},
        }

        scheduler = TemperatureScheduler(config, phase_4_start=80)

        # Get temperatures for both VAEs
        temps_a = [scheduler.get_temperature(e, "A") for e in range(80)]
        temps_b = [scheduler.get_temperature(e, "B") for e in range(80)]

        # Without cyclic and lag, they should be identical before phase 4
        for t_a, t_b in zip(temps_a, temps_b):
            assert t_a == pytest.approx(t_b)

    def test_warmup_then_decay_pattern(self):
        """Verify common warmup-then-decay training pattern."""
        config = {
            "total_epochs": 100,
            "vae_a": {
                "beta_start": 1.0,
                "beta_end": 0.1,
                "beta_warmup_epochs": 20,
            },
            "vae_b": {
                "beta_start": 1.0,
                "beta_end": 0.1,
            },
        }

        scheduler = BetaScheduler(config)

        # Warmup phase: monotonically increasing
        warmup_betas = [scheduler.get_beta(e, "A") for e in range(20)]
        for i in range(1, len(warmup_betas)):
            assert warmup_betas[i] >= warmup_betas[i - 1]

        # After warmup: monotonically decreasing
        decay_betas = [scheduler.get_beta(e, "A") for e in range(20, 101)]
        for i in range(1, len(decay_betas)):
            assert decay_betas[i] <= decay_betas[i - 1]
