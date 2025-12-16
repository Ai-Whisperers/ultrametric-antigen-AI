# Ternary Hyperbolic Bottleneck Layer - Implementation Plan

**Doc-Type:** Implementation Plan · Version 1.0 · Updated 2025-12-16 · Author Ternary VAE Team

---

## Executive Summary

Bridge the Python VAE (`src/core/ternary.py`, `src/models/ternary_vae.py`) with the C++ ternary engine (`engine/`) to enable:
- **8x memory reduction**: 16D FP32 (64 bytes) → Dense243 (4 bytes)
- **10-100x speedup**: Replace Python valuation loops with engine LUTs
- **Hardware-agnostic inference**: TritNet GEMM runs on CPU SIMD / GPU matmul

---

## Phase 1: Core Bridge Layer

### 1.1 Create `src/core/ternary_bridge.py`

**Status:** [ ] Not Started

**Purpose:** Python bindings to engine's Dense243 and TritNet modules

**Tasks:**
- [ ] Import `ternary_dense243_module` (engine binding)
- [ ] Import `ternary_tritnet_gemm` (engine binding)
- [ ] Create `to_dense243(z_hyp: Tensor) -> bytes` function
- [ ] Create `from_dense243(packed: bytes) -> Tensor` function
- [ ] Create `tritnet_linear(x: Tensor, weights_packed: bytes) -> Tensor`

**Code location:** `src/core/ternary_bridge.py` (new file)

**Dependencies:**
- Engine must be built: `python engine/build/build_dense243.py`
- Engine must be built: `python engine/build/build_tritnet_gemm.py` (if exists)

---

### 1.2 Implement Hyperbolic → Ternary Quantization

**Status:** [ ] Not Started

**Purpose:** Map continuous Poincaré ball coordinates to ternary {-1, 0, +1}

**Algorithm:**
```python
def quantize_hyperbolic_to_ternary(z_hyp: Tensor, threshold: float = 0.3) -> Tensor:
    """
    Quantize 16D hyperbolic coordinates to ternary.

    Strategy: Use radius-aware thresholding
    - Near origin (flat region): → 0
    - Negative curvature direction: → -1
    - Positive curvature direction: → +1

    Args:
        z_hyp: (batch, 16) hyperbolic embeddings in Poincaré ball
        threshold: Dead zone around 0

    Returns:
        (batch, 16) ternary values in {-1, 0, +1}
    """
    ternary = torch.zeros_like(z_hyp)
    ternary[z_hyp > threshold] = 1
    ternary[z_hyp < -threshold] = -1
    return ternary
```

**Tasks:**
- [ ] Implement basic threshold quantization
- [ ] Add learned threshold (per-dimension)
- [ ] Add straight-through estimator for gradient flow
- [ ] Test round-trip: quantize → dequantize → measure error

**Files to modify:**
- `src/core/ternary_bridge.py` (new)
- `src/models/hyperbolic_projection.py` (add quantization layer)

---

### 1.3 Pack Ternary to Dense243 Format

**Status:** [ ] Not Started

**Purpose:** Pack 16D ternary vector into 4 bytes (Dense243 = 5 trits/byte)

**Math:**
- 16 trits ÷ 5 trits/byte = 3.2 bytes → **4 bytes** (pad with zeros)
- Compression: 64 bytes (FP32) → 4 bytes = **16x reduction**

**Tasks:**
- [ ] Implement `pack_to_dense243(ternary: Tensor) -> bytes`
- [ ] Implement `unpack_from_dense243(packed: bytes) -> Tensor`
- [ ] Use engine's `dense243_pack()` via Python binding
- [ ] Benchmark pack/unpack throughput

**Integration point:** `engine/src/engine/dense243/ternary_dense243.h`

---

## Phase 2: Replace Decoder with TritNet GEMM

### 2.1 Convert Decoder Weights to Ternary

**Status:** [ ] Not Started

**Purpose:** Quantize FrozenDecoder's weights to ternary format

**Current decoder architecture (`src/models/ternary_vae.py:125-131`):**
```python
self.decoder = nn.Sequential(
    nn.Linear(16, 32),   # W1: 16×32 = 512 weights
    nn.ReLU(),
    nn.Linear(32, 64),   # W2: 32×64 = 2048 weights
    nn.ReLU(),
    nn.Linear(64, 27)    # W3: 64×27 = 1728 weights
)
# Total: 4,288 weights → 858 bytes in Dense243
```

**Tasks:**
- [ ] Extract decoder weights from checkpoint
- [ ] Quantize weights to ternary (threshold or learned)
- [ ] Pack weights to Dense243 format
- [ ] Store as `.ternary` binary file

**Code:**
```python
def quantize_decoder_weights(decoder: FrozenDecoder) -> Dict[str, bytes]:
    """Quantize all decoder weights to Dense243 format."""
    packed_weights = {}
    for name, param in decoder.named_parameters():
        if 'weight' in name:
            ternary_w = quantize_weights(param.data)
            packed_w = pack_to_dense243(ternary_w.flatten())
            packed_weights[name] = packed_w
    return packed_weights
```

---

### 2.2 Implement TernaryLinear Layer

**Status:** [ ] Not Started

**Purpose:** Drop-in replacement for nn.Linear using TritNet GEMM

**Tasks:**
- [ ] Create `TernaryLinear(nn.Module)` class
- [ ] Load Dense243-packed weights
- [ ] Forward: call `ternary_tritnet_gemm.gemm()`
- [ ] Support bias (add after GEMM)
- [ ] Support batched inference

**Code skeleton:**
```python
class TernaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Packed weights: (in_features/5, out_features) in Dense243
        self.register_buffer('weights_packed', None)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def load_ternary_weights(self, packed: bytes):
        """Load Dense243-packed weights."""
        # Convert bytes to numpy array
        arr = np.frombuffer(packed, dtype=np.uint8)
        self.weights_packed = torch.from_numpy(arr)

    def forward(self, x: Tensor) -> Tensor:
        """Forward using TritNet GEMM."""
        import ternary_tritnet_gemm as gemm

        batch_size = x.shape[0]
        # Call engine GEMM
        out = gemm.gemm(
            x.numpy().astype(np.float32),
            self.weights_packed.numpy(),
            batch_size,
            self.out_features,
            self.in_features
        )
        return torch.from_numpy(out) + self.bias
```

**File:** `src/layers/ternary_linear.py` (new)

---

### 2.3 Create TernaryDecoder

**Status:** [ ] Not Started

**Purpose:** Full decoder using TernaryLinear layers

**Tasks:**
- [ ] Create `TernaryDecoder` class
- [ ] Replace all nn.Linear with TernaryLinear
- [ ] Keep ReLU activations (unchanged)
- [ ] Load from quantized checkpoint
- [ ] Benchmark vs FrozenDecoder

**Expected results:**
- Memory: 4,288 × 4 bytes = 17KB → 858 bytes = **20x reduction**
- Speed: Depends on TritNet GEMM vs PyTorch Linear (benchmark needed)

---

## Phase 3: Accelerate P-adic Valuation

### 3.1 Replace Python Valuation Loop with Engine LUT

**Status:** [ ] Not Started

**Current code (`src/core/ternary.py:77-90`):**
```python
def _build_valuation_lut(self) -> torch.Tensor:
    valuations = []
    for n in range(self.N_OPERATIONS):  # 19,683 iterations
        if n == 0:
            valuations.append(self.MAX_VALUATION)
        else:
            v = 0
            m = n
            while m % 3 == 0:  # Up to 9 iterations each
                v += 1
                m //= 3
            valuations.append(v)
    return torch.tensor(valuations, dtype=torch.long)
```

**Replacement:** Use engine's precomputed LUTs

**Tasks:**
- [ ] Create Python binding to access engine's valuation LUT
- [ ] Replace `_build_valuation_lut()` with LUT load
- [ ] Verify correctness (compare outputs)
- [ ] Benchmark: Python loop vs engine LUT

**Expected speedup:** 10-100x for LUT initialization

---

### 3.2 Accelerate `compute_3adic_valuation` in Metrics

**Status:** [ ] Not Started

**Current code (`src/metrics/hyperbolic.py:64-88`):**
```python
def compute_3adic_valuation(diff: torch.Tensor, max_depth: int = 10) -> torch.Tensor:
    val = torch.zeros_like(diff, dtype=torch.float32)
    remaining = diff.clone()

    for _ in range(max_depth):  # 10 iterations
        mask = (remaining % 3 == 0) & (remaining > 0)
        val[mask] += 1
        remaining[mask] = remaining[mask] // 3

    val[diff == 0] = float(max_depth)
    return val
```

**Tasks:**
- [ ] Replace loop with engine's vectorized valuation
- [ ] Use engine's batch operations if available
- [ ] Benchmark improvement

---

## Phase 4: End-to-End Integration

### 4.1 Create `TernaryHyperbolicBottleneck` Module

**Status:** [ ] Not Started

**Purpose:** Single module combining quantization + packing + GEMM

**Architecture:**
```
z_hyp (16D float)
    → quantize_hyperbolic_to_ternary()
    → pack_to_dense243()
    → TernaryLinear (first decoder layer)
    → ...rest of decoder
    → output logits
```

**Tasks:**
- [ ] Create `TernaryHyperbolicBottleneck(nn.Module)`
- [ ] Implement forward with quantization + GEMM
- [ ] Implement backward with straight-through estimator
- [ ] Add toggle: ternary mode vs float mode (for comparison)

**File:** `src/models/ternary_bottleneck.py` (new)

---

### 4.2 Integrate into TernaryVAEV5_11

**Status:** [ ] Not Started

**Tasks:**
- [ ] Add `use_ternary_bottleneck: bool` config option
- [ ] Replace `decoder_A` with `TernaryDecoder` when enabled
- [ ] Add quantization after `z_A_hyp` projection
- [ ] Test training with ternary bottleneck

**File to modify:** `src/models/ternary_vae.py`

---

### 4.3 Benchmark Accuracy Retention

**Status:** [ ] Not Started

**Tasks:**
- [ ] Train baseline model (float decoder)
- [ ] Train with ternary bottleneck
- [ ] Compare metrics:
  - [ ] Reconstruction accuracy
  - [ ] Coverage (must stay 100%)
  - [ ] Hierarchy correlation
- [ ] **Target: <5% accuracy loss**

---

## Phase 5: Hardware Deployment

### 5.1 CPU SIMD Inference Path

**Status:** [ ] Not Started

**Tasks:**
- [ ] Verify AVX2 backend is active
- [ ] Benchmark inference throughput (samples/sec)
- [ ] Compare vs PyTorch CPU

---

### 5.2 GPU Inference Path (Future)

**Status:** [ ] Not Started

**Tasks:**
- [ ] Implement CUDA kernel for Dense243 unpack
- [ ] Use cuBLAS for GEMM after unpack
- [ ] OR: Keep ternary weights on GPU, use custom CUDA GEMM
- [ ] Benchmark vs PyTorch CUDA

---

## File Checklist

**New files to create:**
- [ ] `src/core/ternary_bridge.py` - Engine bindings
- [ ] `src/layers/__init__.py` - New layers module
- [ ] `src/layers/ternary_linear.py` - TernaryLinear layer
- [ ] `src/models/ternary_bottleneck.py` - Bottleneck module
- [ ] `src/models/ternary_decoder.py` - Full ternary decoder

**Files to modify:**
- [ ] `src/core/ternary.py` - Use engine LUTs
- [ ] `src/models/ternary_vae.py` - Add bottleneck option
- [ ] `src/metrics/hyperbolic.py` - Accelerate valuation

**Engine builds required:**
- [ ] `ternary_dense243_module.pyd` (Dense243 pack/unpack)
- [ ] `ternary_tritnet_gemm.pyd` (GEMM operations)

---

## Success Criteria

| Metric | Baseline | Target | Status |
|--------|----------|--------|--------|
| Embedding size | 64 bytes | 4 bytes | [ ] |
| Decoder weights | 17 KB | 858 bytes | [ ] |
| Valuation init | ~100ms | <1ms | [ ] |
| Inference speed | 1x | 2-5x | [ ] |
| Accuracy loss | 0% | <5% | [ ] |
| Coverage | 100% | 100% | [ ] |

---

## Dependencies

**Python packages:**
- torch >= 2.0
- numpy >= 1.19
- pybind11 (for engine builds)

**Engine modules (must build):**
```bash
cd engine
python build/build_dense243.py
python build/build_tritnet_gemm.py  # If available
```

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Accuracy degradation from quantization | High | Use learned thresholds, straight-through estimator |
| Engine build failures | Medium | Test on Windows x64 first (validated platform) |
| TritNet GEMM slower than PyTorch | Medium | Benchmark early, fallback to float if needed |
| Gradient flow through quantization | High | Implement proper STE, test gradient magnitudes |

---

## Next Steps (Immediate Actions)

1. **Build engine modules** - Ensure Dense243 and GEMM bindings compile
2. **Create ternary_bridge.py** - Basic import and smoke test
3. **Implement quantization** - Threshold-based first, learned later
4. **Benchmark pack/unpack** - Verify 16x compression achieved
5. **Replace one decoder layer** - Single TernaryLinear, measure accuracy

---

**Version:** 1.0 · **Updated:** 2025-12-16 · **Project:** Ternary VAE + Engine Integration
