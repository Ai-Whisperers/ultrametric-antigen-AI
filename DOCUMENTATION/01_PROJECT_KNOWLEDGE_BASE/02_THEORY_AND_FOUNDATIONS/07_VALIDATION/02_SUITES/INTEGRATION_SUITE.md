# Integration Test Suite Definition

## Scope

Verification of component interaction.

## Test Plan

| Workflow     | Components              | Test Case                                                          | ID               |
| :----------- | :---------------------- | :----------------------------------------------------------------- | :--------------- |
| `Train Loop` | Model + Data + Loss     | Verify loss decreases on toy data (1 epoch).                       | `INT-001`        |
| `Save/Load`  | Checkpointing           | Verify weights persist after reload.                               | `INT-002`        |
| `Train Loop` | VAE + HyperbolicLoss    | Loss decreases after 5 steps on random batch.                      | `INT-PIPE-001`   |
| `Train Loop` | Gradients               | Check grads are non-zero for Projection, zero for Frozen Encoder.  | `INT-PIPE-002`   |
| `Checkpoint` | Save/Load               | Save VAE, reload, verify weights match exactly.                    | `INT-IO-001`     |
| `Inference`  | `downstream_validation` | Run full validation script on "Toy" dataset (n=100) without error. | `INT-SCRIPT-001` |
