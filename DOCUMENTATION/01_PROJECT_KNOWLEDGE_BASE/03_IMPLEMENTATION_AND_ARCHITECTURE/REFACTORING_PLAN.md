# Comprehensive Refactoring Plan: Code Centralization & Modularization

## Executive Summary

This document outlines a complete refactoring strategy to eliminate code duplication, centralize utilities, and establish professional-grade architecture for the ternary VAE bioinformatics project.

**Key Metrics:**
- **8 duplicate implementations** of p-adic valuation
- **5 duplicate implementations** of vectorized distance computation
- **4+ scattered configuration patterns**
- **Estimated code reduction:** 2,000+ lines

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Target Architecture](#2-target-architecture)
3. [Centralized Module Specifications](#3-centralized-module-specifications)
4. [Refactoring Phases](#4-refactoring-phases)
5. [Migration Guide](#5-migration-guide)
6. [Deprecation Strategy](#6-deprecation-strategy)
7. [Testing Requirements](#7-testing-requirements)
8. [Documentation Standards](#8-documentation-standards)

---

## 1. Current State Analysis

### 1.1 Duplication Inventory

| Function/Pattern | Occurrences | Files | Priority |
|-----------------|-------------|-------|----------|
| `padic_valuation` | 8 | core, utils, analysis, encoders, physics, models | CRITICAL |
| `padic_distance` | 3 | core, utils, analysis | CRITICAL |
| `padic_distance_vectorized` | 5 | core, utils, analysis, contrastive | CRITICAL |
| `PAdicShiftResult` | 2 | core, utils | HIGH |
| `padic_digits` | 4 | core, utils, encoders | HIGH |
| `goldilocks_score` | 2 | core, analysis | MEDIUM |
| Tensor broadcasting patterns | 3+ | core, utils, contrastive | MEDIUM |
| Prime/p initialization | 4+ | contrastive, physics, crispr | LOW |

### 1.2 Impact Assessment

**Maintenance Burden:**
- Bug fixes require changes in 8 places
- Inconsistent return types (`float("inf")` vs `100` vs `PADIC_INFINITY`)
- Different naming conventions (`padic_valuation` vs `compute_padic_valuation` vs `_padic_valuation`)

**Performance Impact:**
- Redundant computations not optimized uniformly
- Some implementations use O(n) loops, others O(log n)
- GPU optimization applied inconsistently

**Developer Experience:**
- Import confusion (which module to use?)
- No clear documentation of canonical source
- Test coverage fragmented

---

## 2. Target Architecture

### 2.1 Module Hierarchy

```
src/
├── core/                          # CENTRALIZED UTILITIES
│   ├── __init__.py               # Public API exports
│   ├── padic_math.py             # P-adic mathematics (EXISTS - enhance)
│   ├── tensor_utils.py           # Tensor operations (NEW)
│   ├── geometry_utils.py         # Geometric utilities (NEW)
│   ├── config_base.py            # Base configuration classes (NEW)
│   └── types.py                  # Type definitions (NEW)
│
├── utils/                         # LEGACY (deprecate gradually)
│   ├── padic_shift.py            # -> Redirect to core/padic_math.py
│   └── ...
│
├── analysis/
│   └── immunology/
│       └── padic_utils.py        # -> Redirect to core/padic_math.py
│
└── [other modules]/              # Use centralized imports
```

### 2.2 Import Pattern

**Before (scattered):**
```python
# File 1
from src.utils.padic_shift import padic_valuation

# File 2
from src.analysis.immunology.padic_utils import compute_padic_valuation

# File 3
def _padic_valuation(self, n):  # Inline implementation
    ...
```

**After (centralized):**
```python
# ALL files use same import
from src.core import padic_valuation, padic_distance, padic_distance_matrix
```

---

## 3. Centralized Module Specifications

### 3.1 `src/core/padic_math.py` (Enhanced)

**Status:** EXISTS - needs minor enhancements

**Current exports:**
- `padic_valuation`, `padic_norm`, `padic_distance`, `padic_digits`
- `padic_valuation_vectorized`, `padic_distance_vectorized`, `padic_distance_matrix`
- `compute_goldilocks_score`, `compute_goldilocks_tensor`
- `compute_hierarchical_embedding`, `PAdicShiftResult`

**Enhancements needed:**
```python
# Add float-safe valuation (for physics module)
def padic_valuation_float(x: float, p: int = 3, precision: int = 1000) -> int:
    """Compute p-adic valuation for float values."""
    x_int = int(round(x * precision))
    if x_int == 0:
        return PADIC_INFINITY_INT
    return padic_valuation(x_int, p)

# Add neighborhood building (for meta-learning)
def build_padic_neighborhoods(
    indices: torch.Tensor,
    valuation_threshold: int = 2,
    p: int = 3,
) -> Dict[int, List[int]]:
    """Build p-adic neighborhoods for efficient sampling."""
    ...

# Add positive pair sampling (for contrastive)
def sample_padic_positives(
    anchor_idx: int,
    all_indices: torch.Tensor,
    valuation_threshold: int = 2,
    p: int = 3,
) -> torch.Tensor:
    """Sample positive pairs based on p-adic distance."""
    ...
```

### 3.2 `src/core/tensor_utils.py` (NEW)

**Purpose:** Centralize tensor manipulation patterns

```python
"""Tensor utilities - Centralized tensor operations.

This module provides optimized tensor operations used across the codebase,
eliminating duplicate implementations of common patterns.

Usage:
    from src.core.tensor_utils import (
        pairwise_broadcast,
        batch_index_select,
        safe_normalize,
    )
"""

def pairwise_broadcast(
    tensor: torch.Tensor,
    dim1: int = -2,
    dim2: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create pairwise broadcasted tensors for distance computations.

    Replaces the repeated pattern:
        i_idx = indices.unsqueeze(2).expand(-1, -1, seq_len)
        j_idx = indices.unsqueeze(1).expand(-1, seq_len, -1)
    """
    ...

def batch_index_select(
    matrix: torch.Tensor,
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
) -> torch.Tensor:
    """Efficient batch indexing into a 2D matrix.

    Replaces:
        flat_i = i_idx.reshape(-1)
        flat_j = j_idx.reshape(-1)
        result = matrix[flat_i, flat_j].reshape(...)
    """
    ...

def safe_normalize(
    tensor: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Safely normalize tensor avoiding division by zero."""
    ...

def clamp_norm(
    tensor: torch.Tensor,
    max_norm: float,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Clamp tensor norm to maximum value."""
    ...
```

### 3.3 `src/core/geometry_utils.py` (NEW)

**Purpose:** Centralize geometric operations

```python
"""Geometry utilities - Centralized geometric operations.

Provides unified geometric operations for Euclidean, hyperbolic,
and other geometric spaces used throughout the project.

Usage:
    from src.core.geometry_utils import (
        project_to_ball,
        exp_map_zero,
        log_map_zero,
    )
"""

def project_to_ball(
    x: torch.Tensor,
    max_norm: float = 0.95,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Project points to interior of unit ball."""
    ...

def project_to_poincare(
    z: torch.Tensor,
    max_norm: float = 0.95,
    curvature: float = 1.0,
) -> torch.Tensor:
    """Project to Poincare ball with curvature."""
    ...

def mobius_add(
    x: torch.Tensor,
    y: torch.Tensor,
    curvature: float = 1.0,
) -> torch.Tensor:
    """Mobius addition in hyperbolic space."""
    ...

def exp_map_zero(
    v: torch.Tensor,
    curvature: float = 1.0,
) -> torch.Tensor:
    """Exponential map from tangent space at origin."""
    ...

def log_map_zero(
    x: torch.Tensor,
    curvature: float = 1.0,
) -> torch.Tensor:
    """Logarithmic map to tangent space at origin."""
    ...
```

### 3.4 `src/core/config_base.py` (NEW)

**Purpose:** Base classes for configuration dataclasses

```python
"""Configuration base classes.

Provides standardized configuration patterns with validation,
serialization, and documentation support.

Usage:
    from src.core.config_base import BaseConfig, PAdicConfig

    @dataclass
    class MyConfig(BaseConfig):
        learning_rate: float = 0.001
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict

@dataclass
class BaseConfig:
    """Base configuration with common utilities."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """Create from dictionary."""
        return cls(**data)

    def validate(self) -> None:
        """Validate configuration. Override in subclasses."""
        pass


@dataclass
class PAdicConfig(BaseConfig):
    """Standard p-adic configuration."""

    prime: int = 3
    max_valuation: int = 9
    use_padic_structure: bool = True

    def validate(self) -> None:
        if self.prime < 2:
            raise ValueError(f"Prime must be >= 2, got {self.prime}")
        if self.max_valuation < 1:
            raise ValueError(f"Max valuation must be >= 1")


@dataclass
class ContrastiveConfig(BaseConfig):
    """Standard contrastive learning configuration."""

    temperature: float = 0.07
    projection_dim: int = 128
    hidden_dim: int = 256
    momentum: float = 0.999
    queue_size: int = 65536
```

### 3.5 `src/core/types.py` (NEW)

**Purpose:** Centralized type definitions

```python
"""Type definitions for the project.

Provides type aliases and protocols for consistent typing across modules.
"""

from typing import Protocol, TypeVar, Union, List, Dict, Tuple, Optional
import torch
import numpy as np

# Type aliases
Tensor = torch.Tensor
Array = np.ndarray
Number = Union[int, float]
TensorOrArray = Union[Tensor, Array]

# P-adic types
ValuationType = Union[int, float]  # int for finite, float('inf') for infinite
PAdicIndex = int
PAdicDigits = List[int]

# Geometry types
Curvature = float
Radius = float
Point = Tensor  # Point in manifold

# Protocol for manifolds
class Manifold(Protocol):
    """Protocol for geometric manifolds."""

    def expmap(self, x: Tensor, v: Tensor) -> Tensor: ...
    def logmap(self, x: Tensor, y: Tensor) -> Tensor: ...
    def dist(self, x: Tensor, y: Tensor) -> Tensor: ...
    def projx(self, x: Tensor) -> Tensor: ...
```

---

## 4. Refactoring Phases

### Phase 1: Foundation (Week 1)

**Goal:** Create centralized modules without breaking existing code

| Task | Priority | Effort |
|------|----------|--------|
| Enhance `src/core/padic_math.py` with missing functions | HIGH | 4h |
| Create `src/core/tensor_utils.py` | HIGH | 3h |
| Create `src/core/geometry_utils.py` | MEDIUM | 3h |
| Create `src/core/config_base.py` | MEDIUM | 2h |
| Create `src/core/types.py` | LOW | 1h |
| Update `src/core/__init__.py` with exports | HIGH | 1h |
| Write unit tests for new modules | HIGH | 4h |

**Deliverables:**
- All new centralized modules created
- 100% test coverage for core modules
- No existing code modified yet

### Phase 2: Gradual Migration (Week 2)

**Goal:** Update modules to use centralized imports with deprecation warnings

| Module | Files to Update | Effort |
|--------|-----------------|--------|
| Extension modules | meta, physics, categorical, tropical, information, topology, contrastive, graphs | 4h |
| Utils | padic_shift.py | 2h |
| Analysis | immunology/padic_utils.py, crispr/padic_distance.py | 2h |
| Encoders | codon_encoder.py, geometric_vector_perceptron.py | 2h |
| Models | predictors/base_predictor.py | 1h |

**Migration pattern:**
```python
# In src/utils/padic_shift.py

import warnings
from src.core.padic_math import (
    padic_valuation as _padic_valuation,
    padic_distance as _padic_distance,
    # ... other imports
)

def padic_valuation(n: int, p: int = 3) -> int:
    """DEPRECATED: Use src.core.padic_math.padic_valuation instead."""
    warnings.warn(
        "padic_valuation from src.utils.padic_shift is deprecated. "
        "Use src.core.padic_math.padic_valuation instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _padic_valuation(n, p)
```

### Phase 3: Code Cleanup (Week 3)

**Goal:** Remove deprecated code paths and inline implementations

| Task | Description | Effort |
|------|-------------|--------|
| Remove inline `_padic_valuation` methods | Replace with imports | 3h |
| Remove duplicate dataclasses | Keep only in core | 2h |
| Consolidate vectorized operations | Use core implementations | 3h |
| Update all imports project-wide | Automated refactoring | 2h |
| Remove empty/redirect modules | Final cleanup | 1h |

### Phase 4: Documentation & Testing (Week 4)

**Goal:** Complete documentation and comprehensive testing

| Task | Deliverable | Effort |
|------|-------------|--------|
| API documentation | Sphinx docs for core modules | 4h |
| Usage examples | Example notebooks | 3h |
| Integration tests | Cross-module tests | 4h |
| Performance benchmarks | Benchmark suite | 2h |
| Migration guide | Developer documentation | 2h |

---

## 5. Migration Guide

### 5.1 Import Changes

**P-adic Operations:**
```python
# OLD (deprecated)
from src.utils.padic_shift import padic_valuation, padic_distance
from src.analysis.immunology.padic_utils import compute_padic_valuation

# NEW (canonical)
from src.core import padic_valuation, padic_distance
# or
from src.core.padic_math import padic_valuation, padic_distance
```

**Tensor Operations:**
```python
# OLD (inline code)
i_idx = indices.unsqueeze(2).expand(-1, -1, seq_len)
j_idx = indices.unsqueeze(1).expand(-1, seq_len, -1)
distances = matrix[i_idx.reshape(-1), j_idx.reshape(-1)].reshape(...)

# NEW (centralized)
from src.core.tensor_utils import pairwise_broadcast, batch_index_select
i_idx, j_idx = pairwise_broadcast(indices)
distances = batch_index_select(matrix, i_idx, j_idx)
```

**Configuration:**
```python
# OLD (scattered dataclasses)
@dataclass
class ContrastiveConfig:
    temperature: float = 0.07
    ...

# NEW (inherit from base)
from src.core.config_base import BaseConfig

@dataclass
class ContrastiveConfig(BaseConfig):
    temperature: float = 0.07
    ...
```

### 5.2 Inline Method Replacement

**Before:**
```python
class MyClass:
    def _padic_valuation(self, n: int) -> int:
        if n == 0:
            return 100
        v = 0
        while n % self.prime == 0:
            n //= self.prime
            v += 1
        return v
```

**After:**
```python
from src.core import padic_valuation

class MyClass:
    def _compute_valuation(self, n: int) -> int:
        return padic_valuation(n, self.prime)
```

---

## 6. Deprecation Strategy

### 6.1 Timeline

| Phase | Duration | Action |
|-------|----------|--------|
| Soft deprecation | 2 weeks | Add warnings, keep functionality |
| Hard deprecation | 2 weeks | Warnings + log errors |
| Removal | 1 week | Remove deprecated code |

### 6.2 Warning Template

```python
import warnings
from functools import wraps

def deprecated(replacement: str, removal_version: str = "2.0.0"):
    """Decorator to mark functions as deprecated."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated and will be removed in "
                f"version {removal_version}. Use {replacement} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage
@deprecated("src.core.padic_math.padic_valuation")
def padic_valuation(n: int, p: int = 3) -> int:
    from src.core.padic_math import padic_valuation as core_valuation
    return core_valuation(n, p)
```

---

## 7. Testing Requirements

### 7.1 Core Module Tests

```python
# tests/unit/core/test_padic_math.py

class TestPadicValuation:
    """Test p-adic valuation computation."""

    def test_valuation_zero(self):
        assert padic_valuation(0) == PADIC_INFINITY_INT

    def test_valuation_prime_power(self):
        assert padic_valuation(9, 3) == 2
        assert padic_valuation(27, 3) == 3

    def test_valuation_coprime(self):
        assert padic_valuation(5, 3) == 0

    @pytest.mark.parametrize("n,p,expected", [
        (6, 3, 1),
        (12, 3, 1),
        (18, 3, 2),
    ])
    def test_valuation_parametrized(self, n, p, expected):
        assert padic_valuation(n, p) == expected


class TestPadicDistanceVectorized:
    """Test vectorized distance computation."""

    def test_zero_distance(self):
        a = torch.tensor([1, 2, 3])
        b = torch.tensor([1, 2, 3])
        distances = padic_distance_vectorized(a, b)
        assert torch.all(distances == 0)

    def test_gpu_compatibility(self):
        if torch.cuda.is_available():
            a = torch.randint(0, 100, (1000,)).cuda()
            b = torch.randint(0, 100, (1000,)).cuda()
            distances = padic_distance_vectorized(a, b)
            assert distances.device.type == "cuda"
```

### 7.2 Integration Tests

```python
# tests/integration/test_centralization.py

class TestCentralizedImports:
    """Verify all modules use centralized imports."""

    def test_no_local_padic_valuation(self):
        """Ensure no module defines local padic_valuation."""
        import ast
        from pathlib import Path

        for py_file in Path("src").rglob("*.py"):
            if "core/padic_math.py" in str(py_file):
                continue
            content = py_file.read_text()
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    assert node.name != "padic_valuation", \
                        f"Found local padic_valuation in {py_file}"
```

### 7.3 Coverage Requirements

| Module | Minimum Coverage |
|--------|-----------------|
| `src/core/padic_math.py` | 95% |
| `src/core/tensor_utils.py` | 90% |
| `src/core/geometry_utils.py` | 90% |
| `src/core/config_base.py` | 85% |

---

## 8. Documentation Standards

### 8.1 Module Documentation

Every centralized module must have:

1. **Module docstring** with:
   - Purpose and scope
   - Usage examples
   - Import patterns
   - References to consolidated modules

2. **Function docstrings** with:
   - Google-style format
   - Type hints in signature
   - Examples section
   - Mathematical formulas where applicable

3. **Inline comments** for:
   - Non-obvious algorithms
   - Performance considerations
   - Edge cases

### 8.2 Example Documentation

```python
def padic_valuation(n: int, p: int = DEFAULT_P) -> int:
    """Compute p-adic valuation v_p(n).

    The p-adic valuation is the largest power of p that divides n.
    Mathematically: v_p(n) = max{k : p^k | n}

    This function consolidates implementations from:
    - src/utils/padic_shift.py (deprecated)
    - src/analysis/immunology/padic_utils.py (deprecated)
    - src/encoders/codon_encoder.py (deprecated)

    Args:
        n: Integer to compute valuation for. Can be positive, negative, or zero.
        p: Prime base (default: 3 for ternary/codon structure)

    Returns:
        The p-adic valuation as an integer.
        Returns PADIC_INFINITY_INT (100) for n=0.

    Raises:
        None (always returns a valid integer)

    Examples:
        Basic usage:
        >>> padic_valuation(9, 3)   # 9 = 3^2
        2
        >>> padic_valuation(6, 3)   # 6 = 2 * 3^1
        1
        >>> padic_valuation(5, 3)   # 5 is coprime to 3
        0

        Zero handling:
        >>> padic_valuation(0, 3)
        100

    Note:
        For vectorized operations on tensors, use padic_valuation_vectorized().
        For float inputs (e.g., from physics simulations), use padic_valuation_float().

    See Also:
        padic_norm: Compute |n|_p = p^(-v_p(n))
        padic_distance: Compute d_p(a,b) = |a-b|_p
    """
```

---

## Appendix A: File Change Summary

### Files to Create
- `src/core/tensor_utils.py`
- `src/core/geometry_utils.py`
- `src/core/config_base.py`
- `src/core/types.py`
- `tests/unit/core/test_tensor_utils.py`
- `tests/unit/core/test_geometry_utils.py`

### Files to Modify
- `src/core/padic_math.py` (enhance)
- `src/core/__init__.py` (update exports)
- `src/utils/padic_shift.py` (deprecation wrappers)
- `src/analysis/immunology/padic_utils.py` (deprecation wrappers)
- All extension modules (update imports)

### Files to Eventually Remove
- Inline `_padic_valuation` methods in 8+ files
- Duplicate `PAdicShiftResult` in utils
- Duplicate vectorized operations in 5+ files

---

## Appendix B: Verification Checklist

- [ ] All core modules created with 90%+ test coverage
- [ ] No duplicate implementations remain
- [ ] All imports use canonical paths
- [ ] Deprecation warnings in place
- [ ] Documentation complete with examples
- [ ] Performance benchmarks show no regression
- [ ] CI/CD updated to check for new duplications
