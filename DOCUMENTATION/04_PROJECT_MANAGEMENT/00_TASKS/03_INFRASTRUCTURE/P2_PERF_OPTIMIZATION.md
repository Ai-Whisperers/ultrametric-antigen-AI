# P2: Performance Optimization

**Status:** Open
**Source:** TECHNICAL_DEBT_AUDIT (Category 1)
**Area:** Infrastructure / Performance

## Bottlenecks

- [ ] **Vectorize 3-Adic Matrix**: `src/losses/padic_losses.py` (35-48). Currently $O(n^2)$ nested loops for 19k items. Blocks startup.
- [ ] **Vectorize Poincaré Distances**: `src/losses/padic_losses.py` (834-838). Hard negative mining uses a slow python loop.
- [ ] **Batch Valuation**: `src/losses/padic_losses.py` (847-857). Replace inner loop with `compute_3adic_valuation_batch()`.
