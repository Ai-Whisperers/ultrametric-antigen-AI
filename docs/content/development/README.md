# Development

> **Contributing, testing, and project roadmap.**

---

## Documents

| Topic | Document |
|:------|:---------|
| Contributing Guide | [contributing.md](contributing.md) |
| Testing Guide | [testing.md](testing.md) |
| Project Roadmap | [roadmap.md](roadmap.md) |

---

## Quick Links

### Setup

```bash
git clone https://github.com/Ai-Whisperers/ternary-vaes-bioinformatics.git
cd ternary-vaes-bioinformatics
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Code Quality

```bash
# Format
black src/ tests/ --line-length 120

# Lint
ruff check src/ tests/

# Type check
mypy src/

# Test
pytest tests/
```

---

## Project Health

| Metric | Value |
|:-------|:------|
| Test count | 231 |
| Pass rate | 97.4% |
| Coverage target | 80% |

---

_Last updated: 2025-12-28_
