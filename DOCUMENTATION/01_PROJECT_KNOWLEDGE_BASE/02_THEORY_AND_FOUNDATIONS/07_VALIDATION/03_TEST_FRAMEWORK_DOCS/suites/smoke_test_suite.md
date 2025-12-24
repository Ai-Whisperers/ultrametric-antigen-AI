# Smoke Test Suite

**Quick sanity checks to run on every commit.**

1.  **Environment Check**: Can we import `torch` and `src`?
2.  **Model Init**: Can `TernaryVAEV5_11(config)` be instantiated?
3.  **Inference**: Can the model run `forward(zeros)` without error?
