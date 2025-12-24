# Validation Documentation

> **Test strategies, verification frameworks, and quality assurance.**

---

## Purpose

This section documents:
- Test planning and strategy
- Test suite documentation
- CI/CD configuration
- Quality metrics

---

## Contents

| Section | Description |
|:--------|:------------|
| [01_PLANS/](01_PLANS/) | Test strategies and CI/CD plans |
| [02_SUITES/](02_SUITES/) | Test suite documentation |
| [03_TEST_FRAMEWORK_DOCS/](03_TEST_FRAMEWORK_DOCS/) | Framework architecture |

---

## Key Documents

### Planning
- [MASTER_TEST_STRATEGY.md](01_PLANS/MASTER_TEST_STRATEGY.md) - Overall testing approach
- [CI_CD_PLAN.md](01_PLANS/CI_CD_PLAN.md) - Continuous integration setup

### Test Suites
- [UNIT_TESTS.md](02_SUITES/UNIT_TESTS.md) - Unit test documentation
- [INTEGRATION_TESTS.md](02_SUITES/INTEGRATION_TESTS.md) - Integration tests
- [SCIENTIFIC_TESTS.md](02_SUITES/SCIENTIFIC_TESTS.md) - Scientific validation tests

---

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific suite
pytest tests/unit/
```

---

## Related Sections

- [02_CODE_HEALTH_METRICS/](../02_PROJECT_MANAGEMENT/02_CODE_HEALTH_METRICS/) - Code quality tracking
- [validation_suite/](../01_PROJECT_KNOWLEDGE_BASE/02_THEORY_AND_FOUNDATIONS/validation_suite/) - Scientific validation

---

*Last updated: 2025-12-24*
