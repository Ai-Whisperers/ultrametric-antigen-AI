# Code Health Dashboard
**Date:** 2025-12-24 06:40

**Total Issues Found:** 387
- **Ruff (Lint/Style):** 21
- **Mypy (Type Safety):** 366

## Top Offenders
| File | Issue Count | Primary Issues |
| :--- | :---: | :--- |
| `scripts\benchmark\measure_manifold_resolution.py` | 58 | Need type annotation for "bit_errors" (hint: "b... |
| `src\training\trainer.py` | 46 | Incompatible types in assignment (expression ha... |
| `scripts\benchmark\measure_coupled_resolution.py` | 38 | "Tensor" not callable  [operator] |
| `src\models\archive\appetitive_vae.py` | 25 | "Tensor" not callable  [operator] |
| `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\project_diagrams_generator.py` | 18 | Blank line contains whitespace |
| `src\models\archive\ternary_vae_v5_10.py` | 17 | Incompatible types in assignment (expression ha... |
| `scripts\benchmark\run_benchmark.py` | 16 | Library stubs not installed for "yaml"  [import... |
| `scripts\visualization\analyze_3adic_structure.py` | 15 | Incompatible types in assignment (expression ha... |
| `src\models\homeostasis.py` | 11 | Need type annotation for "coverage_history"  [v... |
| `src\models\curriculum.py` | 11 | Need type annotation for "tau_history" (hint: "... |

## Detailed Verification Audit
> Issues grouped by file. Fix priority: Type Errors > Syntax Errors > Style.

### 📄 `scripts\benchmark\measure_manifold_resolution.py` (58 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 34 | **mypy** | `TYPE` | Need type annotation for "bit_errors" (hint: "bit_errors: list[<type>] = ...")  [var-annotated] |
| 41 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 43 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 45 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 47 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 54 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "int | float", variable has type "int")  [assignment] |
| 61 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "ndarray[Any, dtype[Any]]", variable has type "list[Any]")  [assignment] |
| 65 | **mypy** | `TYPE` | "list[Any]" has no attribute "mean"  [attr-defined] |
| 67 | **mypy** | `TYPE` | "list[Any]" has no attribute "max"  [attr-defined] |
| 68 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 69 | **mypy** | `TYPE` | No overload variant of "unique" matches argument types "list[Any]", "bool"  [call-overload] |
| 69 | **mypy** | `TYPE` | Possible overload variants: |
| 69 | **mypy** | `TYPE` | def [_SCT: generic] unique(ar: _SupportsArray[dtype[_SCT]] | _NestedSequence[_SupportsArray[dtype[_SCT]]], return_index: Literal[False] = ..., return_inverse: Literal[False] = ..., return_counts: Literal[False] = ..., axis: SupportsIndex | None = ..., *, equal_nan: bool = ...) -> ndarray[Any, dtype[_SCT]] |
| 69 | **mypy** | `TYPE` | def unique(ar: _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes], return_index: Literal[False] = ..., return_inverse: Literal[False] = ..., return_counts: Literal[False] = ..., axis: SupportsIndex | None = ..., *, equal_nan: bool = ...) -> ndarray[Any, dtype[Any]] |
| 69 | **mypy** | `TYPE` | def [_SCT: generic] unique(ar: _SupportsArray[dtype[_SCT]] | _NestedSequence[_SupportsArray[dtype[_SCT]]], return_index: Literal[True] = ..., return_inverse: Literal[False] = ..., return_counts: Literal[False] = ..., axis: SupportsIndex | None = ..., *, equal_nan: bool = ...) -> tuple[ndarray[Any, dtype[_SCT]], ndarray[Any, dtype[signedinteger[Any]]]] |
| 69 | **mypy** | `TYPE` | def unique(ar: _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes], return_index: Literal[True] = ..., return_inverse: Literal[False] = ..., return_counts: Literal[False] = ..., axis: SupportsIndex | None = ..., *, equal_nan: bool = ...) -> tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[signedinteger[Any]]]] |
| 69 | **mypy** | `TYPE` | def [_SCT: generic] unique(ar: _SupportsArray[dtype[_SCT]] | _NestedSequence[_SupportsArray[dtype[_SCT]]], return_index: Literal[False] = ..., return_inverse: Literal[True] = ..., return_counts: Literal[False] = ..., axis: SupportsIndex | None = ..., *, equal_nan: bool = ...) -> tuple[ndarray[Any, dtype[_SCT]], ndarray[Any, dtype[signedinteger[Any]]]] |
| 69 | **mypy** | `TYPE` | def unique(ar: _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes], return_index: Literal[False] = ..., return_inverse: Literal[True] = ..., return_counts: Literal[False] = ..., axis: SupportsIndex | None = ..., *, equal_nan: bool = ...) -> tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[signedinteger[Any]]]] |
| 69 | **mypy** | `TYPE` | def [_SCT: generic] unique(ar: _SupportsArray[dtype[_SCT]] | _NestedSequence[_SupportsArray[dtype[_SCT]]], return_index: Literal[False] = ..., return_inverse: Literal[False] = ..., return_counts: Literal[True] = ..., axis: SupportsIndex | None = ..., *, equal_nan: bool = ...) -> tuple[ndarray[Any, dtype[_SCT]], ndarray[Any, dtype[signedinteger[Any]]]] |
| 69 | **mypy** | `TYPE` | def unique(ar: _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes], return_index: Literal[False] = ..., return_inverse: Literal[False] = ..., return_counts: Literal[True] = ..., axis: SupportsIndex | None = ..., *, equal_nan: bool = ...) -> tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[signedinteger[Any]]]] |
| 69 | **mypy** | `TYPE` | def [_SCT: generic] unique(ar: _SupportsArray[dtype[_SCT]] | _NestedSequence[_SupportsArray[dtype[_SCT]]], return_index: Literal[True] = ..., return_inverse: Literal[True] = ..., return_counts: Literal[False] = ..., axis: SupportsIndex | None = ..., *, equal_nan: bool = ...) -> tuple[ndarray[Any, dtype[_SCT]], ndarray[Any, dtype[signedinteger[Any]]], ndarray[Any, dtype[signedinteger[Any]]]] |
| 69 | **mypy** | `TYPE` | def unique(ar: _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes], return_index: Literal[True] = ..., return_inverse: Literal[True] = ..., return_counts: Literal[False] = ..., axis: SupportsIndex | None = ..., *, equal_nan: bool = ...) -> tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[signedinteger[Any]]], ndarray[Any, dtype[signedinteger[Any]]]] |
| 69 | **mypy** | `TYPE` | def [_SCT: generic] unique(ar: _SupportsArray[dtype[_SCT]] | _NestedSequence[_SupportsArray[dtype[_SCT]]], return_index: Literal[True] = ..., return_inverse: Literal[False] = ..., return_counts: Literal[True] = ..., axis: SupportsIndex | None = ..., *, equal_nan: bool = ...) -> tuple[ndarray[Any, dtype[_SCT]], ndarray[Any, dtype[signedinteger[Any]]], ndarray[Any, dtype[signedinteger[Any]]]] |
| 69 | **mypy** | `TYPE` | def unique(ar: _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes], return_index: Literal[True] = ..., return_inverse: Literal[False] = ..., return_counts: Literal[True] = ..., axis: SupportsIndex | None = ..., *, equal_nan: bool = ...) -> tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[signedinteger[Any]]], ndarray[Any, dtype[signedinteger[Any]]]] |
| 69 | **mypy** | `TYPE` | def [_SCT: generic] unique(ar: _SupportsArray[dtype[_SCT]] | _NestedSequence[_SupportsArray[dtype[_SCT]]], return_index: Literal[False] = ..., return_inverse: Literal[True] = ..., return_counts: Literal[True] = ..., axis: SupportsIndex | None = ..., *, equal_nan: bool = ...) -> tuple[ndarray[Any, dtype[_SCT]], ndarray[Any, dtype[signedinteger[Any]]], ndarray[Any, dtype[signedinteger[Any]]]] |
| 69 | **mypy** | `TYPE` | def unique(ar: _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes], return_index: Literal[False] = ..., return_inverse: Literal[True] = ..., return_counts: Literal[True] = ..., axis: SupportsIndex | None = ..., *, equal_nan: bool = ...) -> tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[signedinteger[Any]]], ndarray[Any, dtype[signedinteger[Any]]]] |
| 69 | **mypy** | `TYPE` | def [_SCT: generic] unique(ar: _SupportsArray[dtype[_SCT]] | _NestedSequence[_SupportsArray[dtype[_SCT]]], return_index: Literal[True] = ..., return_inverse: Literal[True] = ..., return_counts: Literal[True] = ..., axis: SupportsIndex | None = ..., *, equal_nan: bool = ...) -> tuple[ndarray[Any, dtype[_SCT]], ndarray[Any, dtype[signedinteger[Any]]], ndarray[Any, dtype[signedinteger[Any]]], ndarray[Any, dtype[signedinteger[Any]]]] |
| 69 | **mypy** | `TYPE` | def unique(ar: _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes], return_index: Literal[True] = ..., return_inverse: Literal[True] = ..., return_counts: Literal[True] = ..., axis: SupportsIndex | None = ..., *, equal_nan: bool = ...) -> tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[signedinteger[Any]]], ndarray[Any, dtype[signedinteger[Any]]], ndarray[Any, dtype[signedinteger[Any]]]] |
| 80 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 82 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 85 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "Tensor", variable has type "list[Any]")  [assignment] |
| 103 | **mypy** | `TYPE` | "list[Any]" has no attribute "shape"  [attr-defined] |
| 115 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 145 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 146 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 148 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 149 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 160 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 162 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 194 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 196 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 199 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "Tensor", variable has type "list[Any]")  [assignment] |
| 212 | **mypy** | `TYPE` | Argument 1 to "cdist" has incompatible type "list[Any]"; expected "Tensor"  [arg-type] |
| 212 | **mypy** | `TYPE` | Argument 2 to "cdist" has incompatible type "list[Any]"; expected "Tensor"  [arg-type] |
| 239 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 241 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 280 | **mypy** | `TYPE` | Dict entry 0 has incompatible type "str": "dict[Any, Any]"; expected "str": "int | Tensor | Module"  [dict-item] |
| 281 | **mypy** | `TYPE` | Dict entry 1 has incompatible type "str": "dict[Any, Any]"; expected "str": "int | Tensor | Module"  [dict-item] |
| 282 | **mypy** | `TYPE` | Dict entry 2 has incompatible type "str": "dict[Any, Any]"; expected "str": "int | Tensor | Module"  [dict-item] |
| 283 | **mypy** | `TYPE` | Dict entry 3 has incompatible type "str": "dict[Any, Any]"; expected "str": "int | Tensor | Module"  [dict-item] |
| 284 | **mypy** | `TYPE` | Dict entry 4 has incompatible type "str": "dict[Any, Any]"; expected "str": "int | Tensor | Module"  [dict-item] |
| 285 | **mypy** | `TYPE` | Dict entry 5 has incompatible type "str": "dict[Any, Any]"; expected "str": "int | Tensor | Module"  [dict-item] |
| 290 | **mypy** | `TYPE` | Dict entry 0 has incompatible type "str": "dict[Any, Any]"; expected "str": "int | Tensor | Module"  [dict-item] |
| 291 | **mypy** | `TYPE` | Dict entry 1 has incompatible type "str": "dict[Any, Any]"; expected "str": "int | Tensor | Module"  [dict-item] |
| 292 | **mypy** | `TYPE` | Dict entry 2 has incompatible type "str": "dict[Any, Any]"; expected "str": "int | Tensor | Module"  [dict-item] |
| 293 | **mypy** | `TYPE` | Dict entry 3 has incompatible type "str": "dict[Any, Any]"; expected "str": "int | Tensor | Module"  [dict-item] |
| 294 | **mypy** | `TYPE` | Dict entry 4 has incompatible type "str": "dict[Any, Any]"; expected "str": "int | Tensor | Module"  [dict-item] |
| 295 | **mypy** | `TYPE` | Dict entry 5 has incompatible type "str": "dict[Any, Any]"; expected "str": "int | Tensor | Module"  [dict-item] |

### 📄 `src\training\trainer.py` (46 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 56 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "def (*Any, **Any) -> Any", variable has type Module)  [assignment] |
| 122 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "RadialStratificationLoss")  [assignment] |
| 136 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "ContinuousCurriculumModule")  [assignment] |
| 165 | **mypy** | `TYPE` | Generator has incompatible item type "Any | int"; expected "bool"  [misc] |
| 165 | **mypy** | `TYPE` | Item "Tensor" of "Tensor | Module" has no attribute "parameters"  [union-attr] |
| 247 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "int", variable has type "Tensor | Module")  [assignment] |
| 248 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 257 | **mypy** | `TYPE` | Unsupported operand types for * (Module and "float")  [operator] |
| 257 | **mypy** | `TYPE` | Left operand is of type "Tensor | Module" |
| 257 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "Tensor | float", variable has type "Tensor | Module")  [assignment] |
| 259 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 261 | **mypy** | `TYPE` | Item "float" of "Tensor | float" has no attribute "item"  [union-attr] |
| 261 | **mypy** | `TYPE` | Argument 1 has incompatible type Module; expected "Tensor | int | float | bool | complex"  [arg-type] |
| 261 | **mypy** | `TYPE` | Unsupported operand types for / (Module and "float")  [operator] |
| 261 | **mypy** | `TYPE` | Both left and right operands are unions |
| 261 | **mypy** | `TYPE` | Unsupported operand types for + (Module and "float")  [operator] |
| 261 | **mypy** | `TYPE` | Left operand is of type "Tensor | Module" |
| 262 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 267 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 403 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 409 | **mypy** | `TYPE` | Item "float" of "Tensor | float" has no attribute "item"  [union-attr] |
| 409 | **mypy** | `TYPE` | Argument 1 has incompatible type Module; expected "Tensor | int | float | bool | complex"  [arg-type] |
| 409 | **mypy** | `TYPE` | Unsupported operand types for / (likely involving Union)  [operator] |
| 409 | **mypy** | `TYPE` | Both left and right operands are unions |
| 409 | **mypy** | `TYPE` | Unsupported operand types for + (likely involving Union)  [operator] |
| 409 | **mypy** | `TYPE` | Left operand is of type "Tensor | Module" |
| 436 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 442 | **mypy** | `TYPE` | Item "float" of "Tensor | float" has no attribute "item"  [union-attr] |
| 442 | **mypy** | `TYPE` | Argument 1 has incompatible type Module; expected "Tensor | int | float | bool | complex"  [arg-type] |
| 442 | **mypy** | `TYPE` | Unsupported operand types for / (likely involving Union)  [operator] |
| 442 | **mypy** | `TYPE` | Both left and right operands are unions |
| 442 | **mypy** | `TYPE` | Unsupported operand types for + (likely involving Union)  [operator] |
| 442 | **mypy** | `TYPE` | Left operand is of type "Tensor | Module" |
| 475 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 511 | **mypy** | `TYPE` | Item "float" of "Tensor | float" has no attribute "item"  [union-attr] |
| 511 | **mypy** | `TYPE` | Argument 1 has incompatible type Module; expected "Tensor | int | float | bool | complex"  [arg-type] |
| 511 | **mypy** | `TYPE` | Unsupported operand types for / (Module and "float")  [operator] |
| 511 | **mypy** | `TYPE` | Both left and right operands are unions |
| 511 | **mypy** | `TYPE` | Unsupported operand types for + (Module and "float")  [operator] |
| 511 | **mypy** | `TYPE` | Left operand is of type "Tensor | Module" |
| 518 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 533 | **mypy** | `TYPE` | Need type annotation for "epoch_losses"  [var-annotated] |
| 633 | **mypy** | `TYPE` | Argument 10 to "log_epoch" of "TrainingMonitor" has incompatible type "Tensor | Module"; expected "bool"  [arg-type] |
| 634 | **mypy** | `TYPE` | Argument 11 to "log_epoch" of "TrainingMonitor" has incompatible type "Tensor | Module"; expected "bool"  [arg-type] |
| 664 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 665 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |

### 📄 `scripts\benchmark\measure_coupled_resolution.py` (38 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 45 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 46 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 49 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 50 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 59 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "int | float", variable has type "int")  [assignment] |
| 72 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "int | float", variable has type "int")  [assignment] |
| 82 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "int | float", variable has type "int")  [assignment] |
| 108 | **mypy** | `TYPE` | No overload variant of "randn" matches argument types "Any", "Tensor | Module", "str"  [call-overload] |
| 108 | **mypy** | `TYPE` | Possible overload variants: |
| 108 | **mypy** | `TYPE` | def randn(size: Sequence[int | SymInt], *, generator: Generator | None, names: Sequence[str | EllipsisType | None] | None, dtype: dtype | None = ..., layout: layout | None = ..., device: str | device | int | None = ..., pin_memory: bool | None = ..., requires_grad: bool | None = ...) -> Tensor |
| 108 | **mypy** | `TYPE` | def randn(*size: int | SymInt, generator: Generator | None, names: Sequence[str | EllipsisType | None] | None, dtype: dtype | None = ..., layout: layout | None = ..., device: str | device | int | None = ..., pin_memory: bool | None = ..., requires_grad: bool | None = ...) -> Tensor |
| 108 | **mypy** | `TYPE` | def randn(size: Sequence[int | SymInt], *, generator: Generator | None, out: Tensor | None = ..., dtype: dtype | None = ..., layout: layout | None = ..., device: str | device | int | None = ..., pin_memory: bool | None = ..., requires_grad: bool | None = ...) -> Tensor |
| 108 | **mypy** | `TYPE` | def randn(*size: int | SymInt, generator: Generator | None, out: Tensor | None = ..., dtype: dtype | None = ..., layout: layout | None = ..., device: str | device | int | None = ..., pin_memory: bool | None = ..., requires_grad: bool | None = ...) -> Tensor |
| 108 | **mypy** | `TYPE` | def randn(size: Sequence[int | SymInt], *, out: Tensor | None = ..., dtype: dtype | None = ..., layout: layout | None = ..., device: str | device | int | None = ..., pin_memory: bool | None = ..., requires_grad: bool | None = ...) -> Tensor |
| 108 | **mypy** | `TYPE` | def randn(*size: int | SymInt, out: Tensor | None = ..., dtype: dtype | None = ..., layout: layout | None = ..., device: str | device | int | None = ..., pin_memory: bool | None = ..., requires_grad: bool | None = ...) -> Tensor |
| 108 | **mypy** | `TYPE` | def randn(size: Sequence[int | SymInt], *, names: Sequence[str | EllipsisType | None] | None, dtype: dtype | None = ..., layout: layout | None = ..., device: str | device | int | None = ..., pin_memory: bool | None = ..., requires_grad: bool | None = ...) -> Tensor |
| 108 | **mypy** | `TYPE` | def randn(*size: int | SymInt, names: Sequence[str | EllipsisType | None] | None, dtype: dtype | None = ..., layout: layout | None = ..., device: str | device | int | None = ..., pin_memory: bool | None = ..., requires_grad: bool | None = ...) -> Tensor |
| 109 | **mypy** | `TYPE` | No overload variant of "randn" matches argument types "Any", "Tensor | Module", "str"  [call-overload] |
| 109 | **mypy** | `TYPE` | Possible overload variants: |
| 109 | **mypy** | `TYPE` | def randn(size: Sequence[int | SymInt], *, generator: Generator | None, names: Sequence[str | EllipsisType | None] | None, dtype: dtype | None = ..., layout: layout | None = ..., device: str | device | int | None = ..., pin_memory: bool | None = ..., requires_grad: bool | None = ...) -> Tensor |
| 109 | **mypy** | `TYPE` | def randn(*size: int | SymInt, generator: Generator | None, names: Sequence[str | EllipsisType | None] | None, dtype: dtype | None = ..., layout: layout | None = ..., device: str | device | int | None = ..., pin_memory: bool | None = ..., requires_grad: bool | None = ...) -> Tensor |
| 109 | **mypy** | `TYPE` | def randn(size: Sequence[int | SymInt], *, generator: Generator | None, out: Tensor | None = ..., dtype: dtype | None = ..., layout: layout | None = ..., device: str | device | int | None = ..., pin_memory: bool | None = ..., requires_grad: bool | None = ...) -> Tensor |
| 109 | **mypy** | `TYPE` | def randn(*size: int | SymInt, generator: Generator | None, out: Tensor | None = ..., dtype: dtype | None = ..., layout: layout | None = ..., device: str | device | int | None = ..., pin_memory: bool | None = ..., requires_grad: bool | None = ...) -> Tensor |
| 109 | **mypy** | `TYPE` | def randn(size: Sequence[int | SymInt], *, out: Tensor | None = ..., dtype: dtype | None = ..., layout: layout | None = ..., device: str | device | int | None = ..., pin_memory: bool | None = ..., requires_grad: bool | None = ...) -> Tensor |
| 109 | **mypy** | `TYPE` | def randn(*size: int | SymInt, out: Tensor | None = ..., dtype: dtype | None = ..., layout: layout | None = ..., device: str | device | int | None = ..., pin_memory: bool | None = ..., requires_grad: bool | None = ...) -> Tensor |
| 109 | **mypy** | `TYPE` | def randn(size: Sequence[int | SymInt], *, names: Sequence[str | EllipsisType | None] | None, dtype: dtype | None = ..., layout: layout | None = ..., device: str | device | int | None = ..., pin_memory: bool | None = ..., requires_grad: bool | None = ...) -> Tensor |
| 109 | **mypy** | `TYPE` | def randn(*size: int | SymInt, names: Sequence[str | EllipsisType | None] | None, dtype: dtype | None = ..., layout: layout | None = ..., device: str | device | int | None = ..., pin_memory: bool | None = ..., requires_grad: bool | None = ...) -> Tensor |
| 116 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 117 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 158 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 159 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 161 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 162 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 210 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 211 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 219 | **mypy** | `TYPE` | Argument 1 to "range" has incompatible type "Tensor | Module"; expected "SupportsIndex"  [arg-type] |
| 223 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "ndarray[Any, dtype[Any]]", variable has type "list[Any]")  [assignment] |
| 280 | **mypy** | `TYPE` | Argument 1 to "float" has incompatible type "Tensor | Module"; expected "str | Buffer | SupportsFloat | SupportsIndex"  [arg-type] |

### 📄 `src\models\archive\appetitive_vae.py` (25 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 127 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 128 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 129 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 130 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 131 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 134 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 135 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 136 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 137 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 138 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 141 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 142 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 143 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 144 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 145 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 148 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 149 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 150 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 151 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 152 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 155 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 156 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 157 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 158 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 159 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\scripts\project_diagrams_generator.py` (18 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 37 | **ruff** | `W293` | Blank line contains whitespace |
| 48 | **ruff** | `W293` | Blank line contains whitespace |
| 119 | **ruff** | `W293` | Blank line contains whitespace |
| 144 | **ruff** | `W293` | Blank line contains whitespace |
| 196 | **ruff** | `W293` | Blank line contains whitespace |
| 204 | **ruff** | `W293` | Blank line contains whitespace |
| 211 | **ruff** | `W293` | Blank line contains whitespace |
| 292 | **ruff** | `W293` | Blank line contains whitespace |
| 297 | **ruff** | `W293` | Blank line contains whitespace |
| 300 | **ruff** | `W293` | Blank line contains whitespace |
| 303 | **ruff** | `W293` | Blank line contains whitespace |
| 306 | **ruff** | `W293` | Blank line contains whitespace |
| 310 | **ruff** | `W293` | Blank line contains whitespace |
| 314 | **ruff** | `W293` | Blank line contains whitespace |
| 351 | **ruff** | `W293` | Blank line contains whitespace |
| 394 | **ruff** | `W293` | Blank line contains whitespace |
| 408 | **ruff** | `W293` | Blank line contains whitespace |
| 412 | **ruff** | `W293` | Blank line contains whitespace |

### 📄 `src\models\archive\ternary_vae_v5_10.py` (17 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 517 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "StateNetV4", variable has type "StateNetV5")  [assignment] |
| 519 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "StateNetV5")  [assignment] |
| 558 | **mypy** | `TYPE` | Need type annotation for "statenet_corrections"  [var-annotated] |
| 688 | **mypy** | `TYPE` | Cannot determine type of "r_A_ema"  [has-type] |
| 689 | **mypy** | `TYPE` | Cannot determine type of "r_B_ema"  [has-type] |
| 696 | **mypy** | `TYPE` | Cannot determine type of "mean_radius_A"  [has-type] |
| 697 | **mypy** | `TYPE` | Cannot determine type of "mean_radius_B"  [has-type] |
| 698 | **mypy** | `TYPE` | Cannot determine type of "prior_sigma"  [has-type] |
| 699 | **mypy** | `TYPE` | Cannot determine type of "curvature"  [has-type] |
| 1122 | **mypy** | `TYPE` | Cannot determine type of "r_A_ema"  [has-type] |
| 1123 | **mypy** | `TYPE` | Cannot determine type of "r_B_ema"  [has-type] |
| 1134 | **mypy** | `TYPE` | Cannot determine type of "mean_radius_A"  [has-type] |
| 1135 | **mypy** | `TYPE` | Cannot determine type of "mean_radius_B"  [has-type] |
| 1161 | **mypy** | `TYPE` | Cannot determine type of "loss_ema"  [has-type] |
| 1164 | **mypy** | `TYPE` | Cannot determine type of "loss_prev"  [has-type] |
| 1165 | **mypy** | `TYPE` | Cannot determine type of "loss_grad_ema"  [has-type] |
| 1166 | **mypy** | `TYPE` | Cannot determine type of "loss_prev"  [has-type] |

### 📄 `scripts\benchmark\run_benchmark.py` (16 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 5 | **mypy** | `TYPE` | Library stubs not installed for "yaml"  [import-untyped] |
| 13 | **mypy** | `TYPE` | Library stubs not installed for "tabulate"  [import-untyped] |
| 13 | **mypy** | `TYPE` | Hint: "python3 -m pip install types-tabulate" |
| 29 | **mypy** | `TYPE` | Incompatible default for argument "checkpoint_path" (default has type "None", argument has type "str")  [assignment] |
| 29 | **mypy** | `TYPE` | PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True |
| 29 | **mypy** | `TYPE` | Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase |
| 353 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "dict[Any, Any]", target has type "str")  [assignment] |
| 356 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "dict[Any, Any]", target has type "str")  [assignment] |
| 359 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "dict[Any, Any]", target has type "str")  [assignment] |
| 363 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "dict[Any, Any]", target has type "str")  [assignment] |
| 366 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "dict[Any, Any]", target has type "str")  [assignment] |
| 408 | **mypy** | `TYPE` | Incompatible default for argument "output_path" (default has type "None", argument has type "str")  [assignment] |
| 408 | **mypy** | `TYPE` | PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True |
| 408 | **mypy** | `TYPE` | Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase |
| 413 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "Path", variable has type "str")  [assignment] |
| 415 | **mypy** | `TYPE` | "str" has no attribute "parent"  [attr-defined] |

### 📄 `scripts\visualization\analyze_3adic_structure.py` (15 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 156 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "ndarray[Any, dtype[Any]]", variable has type "list[int]")  [assignment] |
| 157 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "ndarray[Any, dtype[Any]]", variable has type "list[int]")  [assignment] |
| 158 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "ndarray[Any, Any]", variable has type "list[floating[Any]]")  [assignment] |
| 159 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "ndarray[Any, Any]", variable has type "list[floating[Any]]")  [assignment] |
| 180 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 181 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 191 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 192 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 216 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 217 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 226 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 227 | **mypy** | `TYPE` | "bool" has no attribute "sum"  [attr-defined] |
| 267 | **mypy** | `TYPE` | Need type annotation for "neighbor_dists_A"  [var-annotated] |
| 268 | **mypy** | `TYPE` | Need type annotation for "neighbor_dists_B"  [var-annotated] |
| 490 | **mypy** | `TYPE` | Module has no attribute "tab10"  [attr-defined] |

### 📄 `src\models\homeostasis.py` (11 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 94 | **mypy** | `TYPE` | Need type annotation for "coverage_history"  [var-annotated] |
| 95 | **mypy** | `TYPE` | Need type annotation for "hierarchy_A_history"  [var-annotated] |
| 96 | **mypy** | `TYPE` | Need type annotation for "hierarchy_B_history"  [var-annotated] |
| 97 | **mypy** | `TYPE` | Need type annotation for "controller_grad_history"  [var-annotated] |
| 98 | **mypy** | `TYPE` | Need type annotation for "Q_history"  [var-annotated] |
| 173 | **mypy** | `TYPE` | Dict entry 3 has incompatible type "str": "float"; expected "str": "bool"  [dict-item] |
| 174 | **mypy** | `TYPE` | Dict entry 4 has incompatible type "str": "float"; expected "str": "bool"  [dict-item] |
| 175 | **mypy** | `TYPE` | Dict entry 5 has incompatible type "str": "list[str]"; expected "str": "bool"  [dict-item] |
| 228 | **mypy** | `TYPE` | Dict entry 3 has incompatible type "str": "float"; expected "str": "bool"  [dict-item] |
| 229 | **mypy** | `TYPE` | Dict entry 4 has incompatible type "str": "float"; expected "str": "bool"  [dict-item] |
| 230 | **mypy** | `TYPE` | Dict entry 5 has incompatible type "str": "list[str]"; expected "str": "bool"  [dict-item] |

### 📄 `src\models\curriculum.py` (11 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 64 | **mypy** | `TYPE` | Need type annotation for "tau_history" (hint: "tau_history: list[<type>] = ...")  [var-annotated] |
| 85 | **mypy** | `TYPE` | Cannot determine type of "tau"  [has-type] |
| 94 | **mypy** | `TYPE` | Cannot determine type of "tau_ema"  [has-type] |
| 137 | **mypy** | `TYPE` | Function "builtins.any" is not valid as a type  [valid-type] |
| 137 | **mypy** | `TYPE` | Perhaps you meant "typing.Any" instead of "any"? |
| 143 | **mypy** | `TYPE` | Function "builtins.any" is not valid as a type  [valid-type] |
| 143 | **mypy** | `TYPE` | Perhaps you meant "typing.Any" instead of "any"? |
| 173 | **mypy** | `TYPE` | Need type annotation for "delta_history" (hint: "delta_history: list[<type>] = ...")  [var-annotated] |
| 174 | **mypy** | `TYPE` | Need type annotation for "radial_loss_history" (hint: "radial_loss_history: list[<type>] = ...")  [var-annotated] |
| 175 | **mypy** | `TYPE` | Need type annotation for "ranking_loss_history" (hint: "ranking_loss_history: list[<type>] = ...")  [var-annotated] |
| 196 | **mypy** | `TYPE` | Dict entry 2 has incompatible type "str": "str"; expected "str": "float"  [dict-item] |

### 📄 `src\training\hyperbolic_trainer.py` (11 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 273 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "PAdicRankingLossHyperbolic")  [assignment] |
| 316 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "HomeostaticHyperbolicPrior")  [assignment] |
| 317 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "HomeostaticHyperbolicPrior")  [assignment] |
| 340 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "HomeostaticReconLoss")  [assignment] |
| 341 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "HomeostaticReconLoss")  [assignment] |
| 355 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "HyperbolicCentroidLoss")  [assignment] |
| 397 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "RadialStratificationLoss")  [assignment] |
| 398 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "RadialStratificationLoss")  [assignment] |
| 415 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "ContinuousCurriculumModule")  [assignment] |
| 449 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "float", variable has type "None")  [assignment] |
| 460 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "float", variable has type "None")  [assignment] |

### 📄 `src\losses\dual_vae_loss.py` (10 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 354 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "float", variable has type "Tensor")  [assignment] |
| 355 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "float", variable has type "Tensor")  [assignment] |
| 451 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 452 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 454 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 455 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 456 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 457 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 480 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 481 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |

### 📄 `src\losses\appetitive_losses.py` (8 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 61 | **mypy** | `TYPE` | Value of type "Tensor | Module" is not indexable  [index] |
| 112 | **mypy** | `TYPE` | Value of type "Tensor | Module" is not indexable  [index] |
| 113 | **mypy** | `TYPE` | Value of type "Tensor | Module" is not indexable  [index] |
| 192 | **mypy** | `TYPE` | Cannot determine type of "z_history"  [has-type] |
| 377 | **mypy** | `TYPE` | Dict entry 3 has incompatible type "str": "float"; expected "str": "Tensor"  [dict-item] |
| 378 | **mypy** | `TYPE` | Dict entry 4 has incompatible type "str": "float"; expected "str": "Tensor"  [dict-item] |
| 452 | **mypy** | `TYPE` | Argument 2 to "full_like" has incompatible type "Tensor | Module"; expected "int | float | bool | complex"  [arg-type] |
| 480 | **mypy** | `TYPE` | "bool" has no attribute "any"  [attr-defined] |

### 📄 `scripts\analysis\verify_mathematical_proofs.py` (8 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 36 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "Callable[[str, str], Any]")  [assignment] |
| 40 | **mypy** | `TYPE` | Incompatible default for argument "checkpoint_path" (default has type "None", argument has type "str")  [assignment] |
| 40 | **mypy** | `TYPE` | PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True |
| 40 | **mypy** | `TYPE` | Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase |
| 41 | **mypy** | `TYPE` | Incompatible default for argument "model_config" (default has type "None", argument has type "dict[str, Any]")  [assignment] |
| 41 | **mypy** | `TYPE` | PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True |
| 41 | **mypy** | `TYPE` | Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase |
| 48 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "device", variable has type "str")  [assignment] |

### 📄 `src\utils\metrics.py` (7 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 170 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "int | float", target has type "int")  [assignment] |
| 189 | **mypy** | `TYPE` | Dict entry 0 has incompatible type "str": "dict[int, float]"; expected "str": "float"  [dict-item] |
| 191 | **mypy** | `TYPE` | Dict entry 2 has incompatible type "str": "floating[Any]"; expected "str": "float"  [dict-item] |
| 192 | **mypy** | `TYPE` | Dict entry 3 has incompatible type "str": "floating[Any]"; expected "str": "float"  [dict-item] |
| 211 | **mypy** | `TYPE` | Incompatible default for argument "intersection" (default has type "None", argument has type "int")  [assignment] |
| 211 | **mypy** | `TYPE` | PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True |
| 211 | **mypy** | `TYPE` | Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase |

### 📄 `src\losses\padic_geodesic.py` (6 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 578 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "Tensor", variable has type "list[Any]")  [assignment] |
| 590 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "Tensor", variable has type "list[float]")  [assignment] |
| 595 | **mypy** | `TYPE` | Unsupported left operand type for - ("list[Any]")  [operator] |
| 611 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "Tensor", variable has type "list[float]")  [assignment] |
| 613 | **mypy** | `TYPE` | Argument 1 to "mse_loss" has incompatible type "list[Any]"; expected "Tensor"  [arg-type] |
| 613 | **mypy** | `TYPE` | Argument 2 to "mse_loss" has incompatible type "list[float]"; expected "Tensor"  [arg-type] |

### 📄 `scripts\visualization\viz_v59_hyperbolic.py` (5 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 38 | **mypy** | `TYPE` | Argument "where" to "fill_between" of "Axes" has incompatible type "ndarray[Any, dtype[bool_]]"; expected "Sequence[bool] | None"  [arg-type] |
| 51 | **mypy** | `TYPE` | Argument "xy" to "annotate" of "Axes" has incompatible type "tuple[signedinteger[Any], Any]"; expected "tuple[float, float]"  [arg-type] |
| 51 | **mypy** | `TYPE` | Argument "xytext" to "annotate" of "Axes" has incompatible type "tuple[signedinteger[Any], Any]"; expected "tuple[float, float] | None"  [arg-type] |
| 95 | **mypy** | `TYPE` | Argument "where" to "fill_between" of "Axes" has incompatible type "ndarray[Any, dtype[bool_]]"; expected "Sequence[bool] | None"  [arg-type] |
| 96 | **mypy** | `TYPE` | Argument "where" to "fill_between" of "Axes" has incompatible type "ndarray[Any, dtype[bool_]]"; expected "Sequence[bool] | None"  [arg-type] |

### 📄 `src\training\monitor.py` (5 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 33 | **mypy** | `TYPE` | Cannot assign to a type  [misc] |
| 33 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "type[SummaryWriter]")  [assignment] |
| 88 | **mypy** | `TYPE` | Argument 1 to "_setup_file_logging" of "TrainingMonitor" has incompatible type "str | None"; expected "str"  [arg-type] |
| 521 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 526 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "Tensor", variable has type "list[Tensor]")  [assignment] |

### 📄 `src\losses\hyperbolic_recon.py` (5 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 297 | **mypy** | `TYPE` | Cannot determine type of "loss_ema"  [has-type] |
| 298 | **mypy** | `TYPE` | Cannot determine type of "coverage_ema"  [has-type] |
| 307 | **mypy** | `TYPE` | Cannot determine type of "adaptive_geodesic_weight"  [has-type] |
| 320 | **mypy** | `TYPE` | Cannot determine type of "adaptive_radius_power"  [has-type] |
| 510 | **mypy** | `TYPE` | Value of type "Tensor | Module" is not indexable  [index] |

### 📄 `src\losses\consequence_predictor.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 54 | **mypy** | `TYPE` | Need type annotation for "actual_accuracy_history" (hint: "actual_accuracy_history: list[<type>] = ...")  [var-annotated] |
| 55 | **mypy** | `TYPE` | Need type annotation for "predicted_accuracy_history" (hint: "predicted_accuracy_history: list[<type>] = ...")  [var-annotated] |
| 205 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 214 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |

### 📄 `src\losses\hyperbolic_prior.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 314 | **mypy** | `TYPE` | Cannot determine type of "mean_radius_ema"  [has-type] |
| 315 | **mypy** | `TYPE` | Cannot determine type of "kl_ema"  [has-type] |
| 323 | **mypy** | `TYPE` | Cannot determine type of "adaptive_sigma"  [has-type] |
| 335 | **mypy** | `TYPE` | Cannot determine type of "adaptive_curvature"  [has-type] |

### 📄 `src\observability\coverage.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 88 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 93 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "Tensor", variable has type "list[Tensor]")  [assignment] |
| 122 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |
| 127 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "Tensor", variable has type "list[Tensor]")  [assignment] |

### 📄 `src\losses\registry.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 62 | **mypy** | `TYPE` | By default the bodies of untyped functions are not checked, consider using --check-untyped-defs  [annotation-unchecked] |
| 63 | **mypy** | `TYPE` | By default the bodies of untyped functions are not checked, consider using --check-untyped-defs  [annotation-unchecked] |
| 64 | **mypy** | `TYPE` | By default the bodies of untyped functions are not checked, consider using --check-untyped-defs  [annotation-unchecked] |
| 205 | **mypy** | `TYPE` | Item "None" of "Tensor | None" has no attribute "device"  [union-attr] |

### 📄 `src\benchmark\utils.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 8 | **mypy** | `TYPE` | Library stubs not installed for "yaml"  [import-untyped] |
| 8 | **mypy** | `TYPE` | Hint: "python3 -m pip install types-PyYAML" |
| 8 | **mypy** | `TYPE` | (or run "mypy --install-types" to install all missing stub packages) |
| 8 | **mypy** | `TYPE` | See https://mypy.readthedocs.io/en/stable/running_mypy.html#missing-imports |

### 📄 `scripts\train\train.py` (4 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 33 | **mypy** | `TYPE` | Library stubs not installed for "yaml"  [import-untyped] |
| 380 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "float", variable has type "None")  [assignment] |
| 381 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "int", variable has type "None")  [assignment] |
| 445 | **mypy** | `TYPE` | Need type annotation for "valuation_groups" (hint: "valuation_groups: dict[<type>, <type>] = ...")  [var-annotated] |

### 📄 `src\training\config_schema.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 241 | **mypy** | `TYPE` | Incompatible default for argument "defaults" (default has type "None", argument has type "dict[str, Any]")  [assignment] |
| 241 | **mypy** | `TYPE` | PEP 484 prohibits implicit Optional. Accordingly, mypy has changed its default to no_implicit_optional=True |
| 241 | **mypy** | `TYPE` | Use https://github.com/hauntsaninja/no_implicit_optional to automatically upgrade your codebase |

### 📄 `scripts\visualization\visualize_ternary_manifold.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 336 | **mypy** | `TYPE` | Argument "extent" to "imshow" of "Axes" has incompatible type "list[Any]"; expected "tuple[float, float, float, float] | None"  [arg-type] |
| 684 | **mypy** | `TYPE` | Argument "extent" to "imshow" of "Axes" has incompatible type "list[Any]"; expected "tuple[float, float, float, float] | None"  [arg-type] |
| 696 | **mypy** | `TYPE` | Argument "extent" to "imshow" of "Axes" has incompatible type "list[Any]"; expected "tuple[float, float, float, float] | None"  [arg-type] |

### 📄 `scripts\visualization\calabi_yau_v58_fast.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 35 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "Tensor", variable has type "list[list[int]]")  [assignment] |
| 362 | **mypy** | `TYPE` | Module has no attribute "viridis"  [attr-defined] |
| 375 | **mypy** | `TYPE` | Module has no attribute "plasma"  [attr-defined] |

### 📄 `scripts\visualization\calabi_yau_v58_extended.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 38 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "Tensor", variable has type "list[list[int]]")  [assignment] |
| 369 | **mypy** | `TYPE` | Cannot call function of unknown type  [operator] |
| 439 | **mypy** | `TYPE` | Module has no attribute "viridis"  [attr-defined] |

### 📄 `src\observability\async_writer.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 42 | **mypy** | `TYPE` | Cannot assign to a type  [misc] |
| 42 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "type[SummaryWriter]")  [assignment] |
| 83 | **mypy** | `TYPE` | Need type annotation for "_queue"  [var-annotated] |

### 📄 `src\models\archive\ternary_vae_v5_7.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 320 | **mypy** | `TYPE` | Need type annotation for "statenet_corrections"  [var-annotated] |
| 392 | **mypy** | `TYPE` | Cannot determine type of "r_A_ema"  [has-type] |
| 393 | **mypy** | `TYPE` | Cannot determine type of "r_B_ema"  [has-type] |

### 📄 `src\losses\zero_structure.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 163 | **mypy** | `TYPE` | Incompatible return value type (got "tuple[Tensor, dict[str, Any | float]]", expected "Tensor")  [return-value] |
| 240 | **mypy** | `TYPE` | Incompatible return value type (got "tuple[Tensor, dict[str, Any]]", expected "Tensor")  [return-value] |
| 312 | **mypy** | `TYPE` | Incompatible return value type (got "tuple[Any, dict[Any, Any]]", expected "Tensor")  [return-value] |

### 📄 `src\training\environment.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 125 | **mypy** | `TYPE` | "dict[Any, Any]" has no attribute "log_dir"  [attr-defined] |
| 193 | **mypy** | `TYPE` | Assignment to variable "e" outside except: block  [misc] |
| 194 | **mypy** | `TYPE` | Trying to read deleted variable "e"  [misc] |

### 📄 `src\data\loaders.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 98 | **mypy** | `TYPE` | Incompatible return value type (got "tuple[DataLoader[Any], DataLoader[Any] | None, DataLoader[Any] | None]", expected "tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any] | None]")  [return-value] |
| 113 | **mypy** | `TYPE` | Argument 1 to "len" has incompatible type "Dataset[Any]"; expected "Sized"  [arg-type] |
| 115 | **mypy** | `TYPE` | Argument 1 to "len" has incompatible type "Dataset[Any]"; expected "Sized"  [arg-type] |

### 📄 `src\losses\padic_losses.py` (3 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 364 | **mypy** | `TYPE` | Need type annotation for "hard_anchors" (hint: "hard_anchors: list[<type>] = ...")  [var-annotated] |
| 774 | **mypy** | `TYPE` | Need type annotation for "hard_anchors" (hint: "hard_anchors: list[<type>] = ...")  [var-annotated] |
| 940 | **mypy** | `TYPE` | "Tensor" not callable  [operator] |

### 📄 `src\artifacts\checkpoint_manager.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 36 | **mypy** | `TYPE` | Need type annotation for "_queue"  [var-annotated] |
| 175 | **mypy** | `TYPE` | Item "None" of "AsyncCheckpointSaver | None" has no attribute "save_async"  [union-attr] |

### 📄 `src\models\ternary_vae.py` (2 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 241 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "HyperbolicProjection", variable has type "DualHyperbolicProjection")  [assignment] |
| 255 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "None", variable has type "DifferentiableController")  [assignment] |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\benchmark\utils.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 12 | **ruff** | `F401` | `typing.Optional` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\training\base.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 17 | **ruff** | `F401` | `pathlib.Path` imported but unused |

### 📄 `C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\src\training\trainer.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 23 | **ruff** | `F401` | `.base.STATENET_KEYS` imported but unused |

### 📄 `scripts\analysis\run_metrics.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 136 | **mypy** | `TYPE` | Item "TextIO" of "TextIO | Any" has no attribute "reconfigure"  [union-attr] |

### 📄 `scripts\analysis\analyze_external_tools.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 152 | **mypy** | `TYPE` | Item "TextIO" of "TextIO | Any" has no attribute "reconfigure"  [union-attr] |

### 📄 `scripts\analysis\comprehensive_audit.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 235 | **mypy** | `TYPE` | Item "TextIO" of "TextIO | Any" has no attribute "reconfigure"  [union-attr] |

### 📄 `src\models\archive\ternary_vae_v5_6.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 274 | **mypy** | `TYPE` | Need type annotation for "statenet_corrections"  [var-annotated] |

### 📄 `src\data\dataset.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 30 | **mypy** | `TYPE` | Incompatible types in assignment (expression has type "Tensor", variable has type "FloatTensor")  [assignment] |

### 📄 `src\losses\radial_stratification.py` (1 issues)
| Line | Tool | Code | Message |
| :--- | :--- | :--- | :--- |
| 149 | **mypy** | `TYPE` | Incompatible return value type (got "tuple[Tensor, dict[str, Any]]", expected "Tensor")  [return-value] |

