# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tests for Functor and NaturalTransformation classes."""

from __future__ import annotations

import pytest
import torch

from src.categorical import TensorType, Morphism, Functor


class TestFunctor:
    """Tests for Functor class."""

    def test_functor_creation(self):
        """Test functor creation."""
        # Identity functor
        functor = Functor(
            object_map=lambda x: x,
            morphism_map=lambda m: m,
            name="identity",
        )
        assert functor.name == "identity"

    def test_functor_apply_object(self):
        """Test applying functor to object."""
        # Doubling functor for TensorTypes
        def double_shape(t: TensorType) -> TensorType:
            return TensorType(tuple(2 * s for s in t.shape))

        functor = Functor(object_map=double_shape, morphism_map=lambda m: m)

        t = TensorType((4,))
        result = functor.apply_object(t)
        assert result.shape == (8,)

    def test_functor_apply_morphism(self):
        """Test applying functor to morphism."""
        # Functor that scales transformations
        def scale_morphism(m: Morphism) -> Morphism:
            return Morphism(
                source=m.source,
                target=m.target,
                name=f"scaled_{m.name}",
                transform=lambda x: 2 * m(x) if m.transform else None,
            )

        functor = Functor(
            object_map=lambda x: x,
            morphism_map=scale_morphism,
        )

        m = Morphism(
            source=TensorType((4,)),
            target=TensorType((4,)),
            name="id",
            transform=lambda x: x,
        )
        result = functor.apply_morphism(m)

        x = torch.randn(5, 4)
        assert torch.allclose(result(x), 2 * x)

    def test_functor_compose(self):
        """Test functor composition."""
        # Functor that doubles shapes
        f1 = Functor(
            object_map=lambda t: TensorType(tuple(2 * s for s in t.shape)),
            morphism_map=lambda m: m,
            name="double",
        )
        # Functor that adds 1 to each dimension
        f2 = Functor(
            object_map=lambda t: TensorType(tuple(s + 1 for s in t.shape)),
            morphism_map=lambda m: m,
            name="plus1",
        )

        composed = f1.compose(f2)

        t = TensorType((4,))
        result = composed.apply_object(t)
        # 4 -> 8 -> 9
        assert result.shape == (9,)


class TestFunctorPreservation:
    """Tests for functor preservation properties."""

    def test_identity_preservation(self):
        """Test identity morphism preservation."""
        # An identity functor should preserve identity morphisms
        t = TensorType((4,))
        identity_morph = Morphism(
            source=t,
            target=t,
            name="id",
            transform=lambda x: x,
        )

        functor = Functor(
            object_map=lambda x: x,
            morphism_map=lambda m: m,
        )

        result = functor.apply_morphism(identity_morph)
        x = torch.randn(5, 4)
        # Should still be identity
        assert torch.allclose(result(x), x)
