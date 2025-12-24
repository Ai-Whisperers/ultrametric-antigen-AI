# Scientific (E2E) Test Suite

**Location:** `tests/suites/e2e/`
**Goal:** Verify scientific validity and downstream utility of the embeddings.

## Included Tests

### 1. Scientific Pipeline (`test_scientific.py`)

Runs the full validation suite used to grade model performance against biological benchmarks.

| Function Test                           | Description                                                                                                                                                                       |
| :-------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_scientific_pipeline_smoke`        | **SCI-E2E-001**: Executes the full validation suite (Nearest Neighbor, Hierarchy, Arithmetic) on synthetic embeddings to ensure the _pipeline_ (not necessarily the model) works. |
| `test_production_readiness_score_logic` | Verifies the scoring logic that determines if a model is "Production Ready" based on passing thresholds (e.g., Spearman correlation > 0.8).                                       |
