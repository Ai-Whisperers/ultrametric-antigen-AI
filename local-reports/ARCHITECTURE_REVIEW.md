# Architecture Review: Critical Issues in StateNet Design

**Doc-Type:** Technical Audit · Version 1.0 · Updated 2025-12-14

---

## Executive Summary

The StateNet component is **architecturally dead** - it cannot learn because all gradient flow is severed by `.item()` calls. All features built on top of StateNet (attention heads, curriculum control, adaptive LR, cross-VAE correlation) are adding complexity to a fundamentally broken component.

---

## Critical Issue #1: No Gradient Flow to StateNet

**Location:** `src/models/ternary_vae_v5_10.py:1061-1079`

**Problem:**
```python
corrections, latent, attention = self.state_net(state_vec)  # Forward pass
delta_lr = corrections[0, 0]

# EVERY output is converted to Python scalar - BREAKS GRADIENT
corrected_lr = lr * (1 + effective_lr_scale * delta_lr.item())  # .item() = no grad!
self.lambda1 = ... + self.statenet_lambda_scale * delta_lambda1.item()  # no grad!
```

**Impact:**
- StateNet parameters are initialized randomly
- No gradient ever flows back to StateNet
- StateNet weights NEVER UPDATE during training
- All "corrections" are random noise from untrained weights

**Why This Happened:**
StateNet was designed to output hyperparameter adjustments (lr, lambdas), which are used as multipliers/addends for the optimizer. But hyperparameter adjustments can't flow gradients - they're applied outside the loss computation.

---

## Critical Issue #2: Unused Attention Heads

**Location:** `src/models/ternary_vae_v5_10.py:368-382`

**Problem:**
```python
# These are computed:
metric_attention = self.metric_attention(normalized_state)
hyp_attention = self.hyperbolic_attention(normalized_state)     # COMPUTED
curriculum_attention = self.curriculum_attention(normalized_state)  # COMPUTED

# But only metric_attention is used:
attended_state = normalized_state * metric_attention_expanded

# hyp_attention and curriculum_attention just returned for logging
attention = {
    'metric': metric_attention,
    'hyperbolic': hyp_attention,      # NEVER USED in computation
    'curriculum': curriculum_attention  # NEVER USED in computation
}
```

**Impact:**
- `hyperbolic_attention` (6 neurons) - dead weight
- `curriculum_attention` (4 neurons) - dead weight
- GAP 5 "fix" added learnable scales to these unused heads

---

## Critical Issue #3: Semantically Meaningless Attention

**Location:** `src/models/ternary_vae_v5_10.py:384-399`

**Problem:**
```python
# metric_attention produces 13 values that sum to 1.0 (softmax)
metric_attention = self.metric_attention(normalized_state)  # [batch, 13], sums to 1

# These 13 values are expanded to 21 by repeating:
metric_attention_expanded = torch.cat([
    metric_attention[:, 0:2].repeat(1, 1),   # 2 values for H_A, H_B
    metric_attention[:, 2:4].repeat(1, 1),   # 2 values for kl_A, kl_B
    metric_attention[:, 4:5],                # 1 value for grad_ratio
    metric_attention[:, 4:5],                # SAME value reused for rho (why?)
    metric_attention[:, 5:6].repeat(1, 3),   # 1 value repeated 3x for lambdas
    ...
], dim=1)

# Then multiplied element-wise:
attended_state = normalized_state * metric_attention_expanded
```

**Issues:**
1. Softmax produces 13 values that sum to 1.0 - this is a probability distribution
2. Expanding to 21D by repeating values makes no semantic sense
3. Using the same attention value (index 4) for both grad_ratio AND rho
4. Multiplying state by softmax values is not standard attention

**What Attention Should Be:**
```python
# Standard attention: Query-Key-Value
attention_weights = softmax(Q @ K.T / sqrt(d))
output = attention_weights @ V  # Weighted sum of values
```

---

## Critical Issue #4: Stale Input Data

**Location:** `src/training/trainer.py:365-387`

**Problem:**
```python
# StateNet called once per EPOCH (at batch_idx == 0)
if self.model.use_statenet and batch_idx == 0:
    # Uses coverage from PREVIOUS epoch
    coverage_A = self.monitor.coverage_A_history[-1] if self.monitor.coverage_A_history else 0

    # Uses correlation from PREVIOUS epoch
    r_A = ...  # from monitor history
```

**Impact:**
- StateNet sees data that's potentially one full epoch old
- Decisions made based on stale information
- Even if StateNet could learn, it would be optimizing for outdated state

---

## Critical Issue #5: Delta Outputs Never Applied

**Location:** `src/models/ternary_vae_v5_10.py:1065-1067`

**Problem:**
```python
delta_sigma = corrections[0, 5]
delta_curvature = corrections[0, 6]
delta_curriculum = corrections[0, 7]

# These are recorded for logging but...
# delta_sigma and delta_curvature are passed to hyperbolic_trainer
# But curriculum.update_tau() is the only one actually used
```

Let me verify: Are delta_sigma and delta_curvature actually applied?

---

## Critical Issue #6: Complexity Without Purpose

All "gap fixes" added more complexity to a broken foundation:

| "Fix" | What It Added | Why It's Useless |
|-------|---------------|------------------|
| Gap 6 (r_AB) | 21st input dimension | StateNet can't learn from any inputs |
| Gap 4 (adaptive LR) | Loss plateau detection, boost | Boosting broken corrections is still broken |
| Gap 5 (dynamic arch) | Learnable attention scales | Scaling unused attention heads |

---

## Root Cause Analysis

**The fundamental design flaw:**

StateNet was conceived as a "meta-controller" that observes training metrics and outputs hyperparameter adjustments. But:

1. **Hyperparameters are not differentiable** - you can't backprop through `optimizer.lr = new_lr`
2. **StateNet has no reward signal** - there's no loss that measures "good hyperparameter adjustment"
3. **The architecture assumes learning happens magically** - but learning requires gradients

**What Would Be Needed:**
1. A reward signal for StateNet (e.g., improvement in correlation after N steps)
2. Reinforcement learning or separate optimization loop for StateNet
3. Or: Remove StateNet entirely and use fixed/scheduled hyperparameters

---

## Recommendations

### Option A: Remove StateNet (Simplest)
- Delete StateNet entirely
- Use fixed or scheduled hyperparameters
- Reduces model parameters by ~5%
- Removes source of random noise

### Option B: Fix StateNet with RL (Complex)
- Implement policy gradient or actor-critic for StateNet
- Define reward as improvement in target metrics
- Requires significant architectural changes

### Option C: Differentiable Hyperparameters (Research)
- Make hyperparameters part of the computation graph
- Use techniques like hypergradient descent
- Academic research territory

---

## Summary

**Stop adding features to StateNet.** It's a dead component. Either remove it or fundamentally redesign it with proper gradient flow or RL-based training.

The current architecture adds 8,595 parameters (4.88% of model) that do nothing but produce random noise.
