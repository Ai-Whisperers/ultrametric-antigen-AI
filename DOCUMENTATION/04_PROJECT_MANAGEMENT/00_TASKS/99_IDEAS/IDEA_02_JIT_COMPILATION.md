# Idea: JIT Compilation for Numeric Kernels

**Status:** Concept
**Driver:** Performance bottlenecks in 3-adic math (p-adic losses).

## Concept

The 3-adic distance and valuation calculations are heavily iterative and slow in pure Python.
We can accelerate them by 10-100x using JIT compilation.

## Candidates

- [ ] **`numba.jit`**: Decorate the valuation functions in `padic_losses.py`.
- [ ] **`torch.compile`**: Use PyTorch 2.0 compiler for the hyperbolic operations.
- [ ] **Custom CUDA Kernel**: (Long term) Write a specialized kernel for 3-adic distance matrix calculation.
