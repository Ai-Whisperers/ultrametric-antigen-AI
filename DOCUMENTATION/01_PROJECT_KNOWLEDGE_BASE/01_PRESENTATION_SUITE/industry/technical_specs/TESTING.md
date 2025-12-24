# Testing Strategy

The project primarily uses **Integration Tests** to ensure the entire pipeline (Model -> Loss -> Trainer) works correctly in the hyperbolic space.

## Test Directory: `tests/`

| File                          | Purpose                                                                                |
| ----------------------------- | -------------------------------------------------------------------------------------- |
| `test_generalization.py`      | Checks if the model generalizes structure to unseen data (though v5.11 sees all data). |
| `test_reproducibility.py`     | Ensures deterministic runs when seeds are set. Critical for scientific validity.       |
| `test_training_validation.py` | faster end-to-end check of a training loop (1-2 epochs) to catch runtime errors.       |

## Running Tests

Prerequisites:

- Install dev dependencies: `pip install -r requirements.txt`

Run all tests:

```bash
pytest
```

Run specific test:

```bash
pytest tests/test_reproducibility.py
```

## Writing New Tests

When adding features (e.g., new Loss function):

1.  Create a test in `tests/`.
2.  Should mock the data loader for speed.
3.  Focus on **geometric invariants** (e.g., "Distance A > Distance B") rather than exact pixel values.
