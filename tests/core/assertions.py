# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Custom assertions for testing.

This module provides domain-specific assertions for:
- Tensor properties (shape, device, dtype, numerical stability)
- Geometric properties (Poincaré disk, hyperbolic distances)
- Model outputs (VAE outputs, loss components)
- Training artifacts (gradients, checkpoints)

Usage:
    from tests.core.assertions import TensorAssertions, GeometryAssertions

    TensorAssertions.assert_shape(tensor, (32, 16))
    GeometryAssertions.assert_on_poincare_disk(points)
"""

from typing import Any, Dict, Tuple

import torch


class TensorAssertions:
    """Assertions for tensor properties."""

    @staticmethod
    def assert_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], msg: str = ""):
        """Assert tensor has expected shape."""
        if tensor.shape != expected_shape:
            raise AssertionError(f"Shape mismatch: expected {expected_shape}, got {tensor.shape}. {msg}")

    @staticmethod
    def assert_device(tensor: torch.Tensor, device: str, msg: str = ""):
        """Assert tensor is on expected device."""
        actual_device = str(tensor.device)
        if device not in actual_device:
            raise AssertionError(f"Device mismatch: expected {device}, got {actual_device}. {msg}")

    @staticmethod
    def assert_dtype(tensor: torch.Tensor, dtype: torch.dtype, msg: str = ""):
        """Assert tensor has expected dtype."""
        if tensor.dtype != dtype:
            raise AssertionError(f"Dtype mismatch: expected {dtype}, got {tensor.dtype}. {msg}")

    @staticmethod
    def assert_finite(tensor: torch.Tensor, msg: str = ""):
        """Assert tensor contains only finite values (no NaN or Inf)."""
        if not torch.isfinite(tensor).all():
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            raise AssertionError(f"Tensor contains non-finite values: {nan_count} NaN, {inf_count} Inf. {msg}")

    @staticmethod
    def assert_no_nan(tensor: torch.Tensor, msg: str = ""):
        """Assert tensor contains no NaN values."""
        if torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            raise AssertionError(f"Tensor contains {nan_count} NaN values. {msg}")

    @staticmethod
    def assert_positive(tensor: torch.Tensor, msg: str = ""):
        """Assert all tensor values are positive."""
        if (tensor <= 0).any():
            min_val = tensor.min().item()
            raise AssertionError(f"Tensor contains non-positive values. Min: {min_val}. {msg}")

    @staticmethod
    def assert_non_negative(tensor: torch.Tensor, msg: str = ""):
        """Assert all tensor values are non-negative."""
        if (tensor < 0).any():
            min_val = tensor.min().item()
            raise AssertionError(f"Tensor contains negative values. Min: {min_val}. {msg}")

    @staticmethod
    def assert_in_range(
        tensor: torch.Tensor,
        low: float,
        high: float,
        inclusive: bool = True,
        msg: str = "",
    ):
        """Assert all tensor values are within range."""
        if inclusive:
            if (tensor < low).any() or (tensor > high).any():
                min_val, max_val = tensor.min().item(), tensor.max().item()
                raise AssertionError(f"Values outside range [{low}, {high}]. " f"Actual range: [{min_val}, {max_val}]. {msg}")
        else:
            if (tensor <= low).any() or (tensor >= high).any():
                min_val, max_val = tensor.min().item(), tensor.max().item()
                raise AssertionError(f"Values outside range ({low}, {high}). " f"Actual range: [{min_val}, {max_val}]. {msg}")

    @staticmethod
    def assert_close(
        actual: torch.Tensor,
        expected: torch.Tensor,
        atol: float = 1e-6,
        rtol: float = 1e-5,
        msg: str = "",
    ):
        """Assert tensors are element-wise close within tolerance."""
        if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
            max_diff = (actual - expected).abs().max().item()
            raise AssertionError(f"Tensors not close. Max diff: {max_diff}, atol: {atol}, rtol: {rtol}. {msg}")

    @staticmethod
    def assert_equal(actual: torch.Tensor, expected: torch.Tensor, msg: str = ""):
        """Assert tensors are exactly equal."""
        if not torch.equal(actual, expected):
            diff_count = (actual != expected).sum().item()
            raise AssertionError(f"Tensors not equal. {diff_count} elements differ. {msg}")


class GeometryAssertions:
    """Assertions for geometric properties."""

    @staticmethod
    def assert_on_poincare_disk(
        tensor: torch.Tensor,
        max_norm: float = 1.0,
        tolerance: float = 1e-5,
        msg: str = "",
    ):
        """Assert all points have norm < max_norm (on Poincaré disk)."""
        norms = torch.norm(tensor, dim=-1)
        if not (norms < max_norm + tolerance).all():
            max_val = norms.max().item()
            violations = (norms >= max_norm).sum().item()
            raise AssertionError(f"Points outside Poincaré disk! Max norm: {max_val:.6f} >= {max_norm}. " f"{violations} violations. {msg}")

    @staticmethod
    def assert_distance_symmetry(
        dist_ab: torch.Tensor,
        dist_ba: torch.Tensor,
        atol: float = 1e-5,
        msg: str = "",
    ):
        """Assert distance is symmetric: d(a,b) = d(b,a)."""
        if not torch.allclose(dist_ab, dist_ba, atol=atol):
            max_diff = (dist_ab - dist_ba).abs().max().item()
            raise AssertionError(f"Distance asymmetry! Max diff: {max_diff}. {msg}")

    @staticmethod
    def assert_distance_identity(dist_aa: torch.Tensor, atol: float = 1e-6, msg: str = ""):
        """Assert distance(x, x) is approximately 0."""
        if not torch.allclose(dist_aa, torch.zeros_like(dist_aa), atol=atol):
            max_val = dist_aa.max().item()
            raise AssertionError(f"Identity distance violation: max self-distance {max_val}. {msg}")

    @staticmethod
    def assert_triangle_inequality(
        dist_ab: torch.Tensor,
        dist_bc: torch.Tensor,
        dist_ac: torch.Tensor,
        atol: float = 1e-5,
        msg: str = "",
    ):
        """Assert triangle inequality: d(a,c) <= d(a,b) + d(b,c)."""
        violations = dist_ac > dist_ab + dist_bc + atol
        if violations.any():
            violation_count = violations.sum().item()
            max_violation = (dist_ac - dist_ab - dist_bc).max().item()
            raise AssertionError(f"Triangle inequality violated! {violation_count} violations. " f"Max violation: {max_violation}. {msg}")

    @staticmethod
    def assert_radial_ordering(
        radii: torch.Tensor,
        valuations: torch.Tensor,
        inverse: bool = True,
        msg: str = "",
    ):
        """Assert radial ordering respects valuation hierarchy.

        If inverse=True: higher valuation -> smaller radius (closer to origin)
        """
        # Check correlation direction
        if radii.numel() < 2 or valuations.numel() < 2:
            return  # Can't check with single point

        correlation = torch.corrcoef(torch.stack([radii.flatten(), valuations.float().flatten()]))[0, 1]

        if inverse:
            if correlation > 0:
                raise AssertionError(f"Expected negative correlation (inverse radial ordering). " f"Got correlation: {correlation:.4f}. {msg}")
        else:
            if correlation < 0:
                raise AssertionError(f"Expected positive correlation (direct radial ordering). " f"Got correlation: {correlation:.4f}. {msg}")


class ModelAssertions:
    """Assertions for model properties."""

    @staticmethod
    def assert_output_keys(outputs: Dict[str, Any], required_keys: list, msg: str = ""):
        """Assert model outputs contain required keys."""
        missing = [k for k in required_keys if k not in outputs]
        if missing:
            raise AssertionError(f"Missing required output keys: {missing}. " f"Available keys: {list(outputs.keys())}. {msg}")

    @staticmethod
    def assert_logits_shape(
        logits: torch.Tensor,
        batch_size: int,
        n_digits: int = 9,
        n_classes: int = 3,
        msg: str = "",
    ):
        """Assert logits have correct shape for ternary classification."""
        expected = (batch_size, n_digits, n_classes)
        if logits.shape != expected:
            raise AssertionError(f"Logits shape mismatch: expected {expected}, got {logits.shape}. {msg}")

    @staticmethod
    def assert_latent_shape(z: torch.Tensor, batch_size: int, latent_dim: int, msg: str = ""):
        """Assert latent code has correct shape."""
        expected = (batch_size, latent_dim)
        if z.shape != expected:
            raise AssertionError(f"Latent shape mismatch: expected {expected}, got {z.shape}. {msg}")

    @staticmethod
    def assert_gradients_exist(model: torch.nn.Module, check_all: bool = False, msg: str = ""):
        """Assert gradients exist for trainable parameters."""
        no_grad_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                no_grad_params.append(name)

        if no_grad_params:
            if check_all or len(no_grad_params) == sum(1 for p in model.parameters() if p.requires_grad):
                raise AssertionError(f"No gradients for parameters: {no_grad_params[:5]}... {msg}")

    @staticmethod
    def assert_gradients_finite(model: torch.nn.Module, msg: str = ""):
        """Assert all gradients are finite."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    nan_count = torch.isnan(param.grad).sum().item()
                    inf_count = torch.isinf(param.grad).sum().item()
                    raise AssertionError(f"Non-finite gradients in {name}: " f"{nan_count} NaN, {inf_count} Inf. {msg}")


class LossAssertions:
    """Assertions for loss values."""

    @staticmethod
    def assert_loss_valid(loss: torch.Tensor, msg: str = ""):
        """Assert loss is a valid scalar (finite, non-negative)."""
        if loss.numel() != 1:
            raise AssertionError(f"Loss is not scalar. Shape: {loss.shape}. {msg}")

        if not torch.isfinite(loss):
            raise AssertionError(f"Loss is not finite: {loss.item()}. {msg}")

        if loss.item() < 0:
            raise AssertionError(f"Loss is negative: {loss.item()}. {msg}")

    @staticmethod
    def assert_loss_decreases(
        losses: list,
        tolerance: float = 0.0,
        strict: bool = False,
        msg: str = "",
    ):
        """Assert loss generally decreases over iterations.

        Args:
            losses: List of loss values
            tolerance: Allowed increase per step
            strict: If True, every step must decrease
        """
        if len(losses) < 2:
            return

        if strict:
            for i in range(1, len(losses)):
                if losses[i] > losses[i - 1] + tolerance:
                    raise AssertionError(f"Loss increased at step {i}: {losses[i-1]:.6f} -> {losses[i]:.6f}. {msg}")
        else:
            # Check overall trend
            if losses[-1] > losses[0]:
                raise AssertionError(f"Loss did not decrease overall: {losses[0]:.6f} -> {losses[-1]:.6f}. {msg}")

    @staticmethod
    def assert_loss_components_valid(loss_dict: Dict[str, Any], required_components: list, msg: str = ""):
        """Assert loss dictionary has valid components."""
        missing = [k for k in required_components if k not in loss_dict]
        if missing:
            raise AssertionError(f"Missing loss components: {missing}. {msg}")

        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all():
                    raise AssertionError(f"Loss component '{key}' is not finite. {msg}")


class TernaryAssertions:
    """Assertions for ternary operations."""

    @staticmethod
    def assert_valid_ternary(tensor: torch.Tensor, msg: str = ""):
        """Assert tensor contains only valid ternary values {-1, 0, 1}."""
        valid_values = {-1.0, 0.0, 1.0}
        unique_values = set(tensor.unique().tolist())
        invalid = unique_values - valid_values

        if invalid:
            raise AssertionError(f"Invalid ternary values: {invalid}. Expected only {valid_values}. {msg}")

    @staticmethod
    def assert_ternary_shape(tensor: torch.Tensor, batch_size: int = None, msg: str = ""):
        """Assert tensor has valid ternary operation shape (B, 9)."""
        if len(tensor.shape) != 2:
            raise AssertionError(f"Expected 2D tensor, got {len(tensor.shape)}D. Shape: {tensor.shape}. {msg}")

        if tensor.shape[1] != 9:
            raise AssertionError(f"Expected 9 digits, got {tensor.shape[1]}. Shape: {tensor.shape}. {msg}")

        if batch_size is not None and tensor.shape[0] != batch_size:
            raise AssertionError(f"Expected batch size {batch_size}, got {tensor.shape[0]}. {msg}")

    @staticmethod
    def assert_index_valid(indices: torch.Tensor, msg: str = ""):
        """Assert indices are valid ternary operation indices [0, 19682]."""
        max_index = 3**9 - 1  # 19682

        if (indices < 0).any():
            min_val = indices.min().item()
            raise AssertionError(f"Negative index found: {min_val}. {msg}")

        if (indices > max_index).any():
            max_val = indices.max().item()
            raise AssertionError(f"Index exceeds max ({max_index}): {max_val}. {msg}")


# Convenience function to import all assertions
def get_all_assertions():
    """Returns all assertion classes for easy importing."""
    return {
        "tensor": TensorAssertions,
        "geometry": GeometryAssertions,
        "model": ModelAssertions,
        "loss": LossAssertions,
        "ternary": TernaryAssertions,
    }
