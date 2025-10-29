# Ternary dual-VAEs current broken status

**Status = trained but poorly measured.**
The errors are in **metrics and testing**, not in the architecture or data.
→ That’s gold: it means the model *already learns*, but the system evaluating it lies.

---

### 🧩 2️⃣ Repair Plan (exact tasks)

**Total time:** ≈ 3–5 focused hours.

| Step | File                                    | Action                                                                                                     | Result                                       |
| ---- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| ✅ 1  | `train_ternary_v5_5.py` L357            | Replace `softmax(logits)` → `torch.distributions.Categorical(logits=logits).sample()` or `.argmax(dim=-1)` | True coverage (~86 %) measured correctly     |
| ✅ 2  | `run_benchmark.py` L24–49               | Require `--checkpoint`; abort if missing; feed `latent = model.encode(data)` into entropy computation      | Valid and reproducible metrics               |
| ✅ 3  | `tests/test_generalization.py` L116–215 | Remove prints; add asserts like `acc > 0.9`, `unique_ops >= 0.8 * total_ops`                               | Automated tests that truly validate learning |
| ✅ 4  | Choose checkpoint                       | Keep `epoch 106`; document SHA hash + true coverage                                                        | Certified baseline                           |
| ⚙️ 5 | Log outputs                             | Save `benchmarks/coverage_vs_entropy.json` with timestamp                                                  | Traceable benchmark                          |

---

### 🧠 3️⃣ After the Repair

* **Honest benchmark:** will report ~86–95 % real coverage, coherent entropy.
* **Reproducible tests:** valid CI.
* **Usable checkpoint:** `latest.pt (epoch 106)` with signed hash.
* **Stable fluxttice:** ready to abstract as *foundation branch* (`dev/fluxttice-foundation`).

---

### 🚀 4️⃣ Minimal Viable Production

Once repaired:

1. Package the VAE as a module (`fluxttice_core/` with `encode`, `decode`, `coverage_report()`).
2. Publish reproducible benchmark → first *scientific release* or API.
3. **Short-term monetization:** “fluxttice explorer” / embedding API / technical paper.

---

### 💡 5️⃣ Final Diagnosis

👉 The *model* is healthy.
👉 The *metrics* are broken.
👉 The *fix* is cheap and linear.

---

# DIAGNOSIS MADE BY CODEX

• Key Findings

  - Coverage numbers are overstated because sample returns the expectation
    of softmax(logits) instead of drawing categorical samples, and both the
    trainer and benchmark round those expectations to integers (src/models/
    ternary_vae_v5_5.py:614, scripts/train/train_ternary_v5_5.py:357, scripts/
    benchmark/run_benchmark.py:117). When I decoded directly with argmax from
    the v5_5 latest.pt checkpoint (epoch 106) the model produced 16 976–16 921
    unique operations—only ≈86% of the ternary space—versus the reported ≥99%.
  - The benchmark suite currently gives misleading results: it happily
    runs on random weights if --checkpoint is omitted (scripts/benchmark/
    run_benchmark.py:24-49), and its latent-entropy metric is computed on
    fresh Gaussian noise rather than latent codes from data (scripts/benchmark/
    run_benchmark.py:184-214), so those numbers say nothing about the trained
    model.
  - The “generalization” tests don’t validate success—they instantiate a brand-
    new, untrained model and only assert trivial conditions like accuracy
    >= 0 (tests/test_generalization.py:116-215). They function as print-only
    diagnostics, so the automated test suite never verifies that training
    learned anything.
  - Utility check: loading sandbox-training/checkpoints/v5_5/latest.pt shows
    the model does memorize operations well—hold-out reconstruction on 10%
    of ternary ops hits 100% for both VAEs, but generation tops out around
    86% unique ops (hash-counted) and later checkpoints in checkpoints/
    ternary_v5_5_best.pt regress to ≈95% coverage, so the usable checkpoint
    needs to be chosen carefully.

  Next Steps

  1. Replace sample with categorical sampling (or argmax) and recalc coverage/
     benchmarks; store both sampled and expectation-based metrics so you can
     compare.
  2. Require a checkpoint in run_benchmark.py, feed real reconstructions
     into the entropy metric, and persist benchmark outputs alongside config/
     checkpoint hashes.
  3. Convert tests/test_generalization.py into real assertions that load the
     trained checkpoint; remove or relocate the print-only diagnostics.
  4. Decide which checkpoint is production-ready (epoch 106 vs later epochs)
     and document the true hash-based coverage so downstream consumers know the
     actual capability.