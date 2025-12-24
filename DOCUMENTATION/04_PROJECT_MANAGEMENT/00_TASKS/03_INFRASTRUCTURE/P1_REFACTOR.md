# P1: Unify Trainers Refactor

**Status:** Open
**Source:** IMPROVEMENT_PLAN.md (Sec 5)
**Area:** Infrastructure

## Objective

Reduce technical debt by unifying forked trainer logic (`trainer.py` vs `appetitive_trainer.py`).

## Tasks

- [ ] **Base Trainer Class**: Create `src/training/base.py` containing `__init__`, checkpointing, and loop boilerplate.
- [ ] **Inheritance Refactor**: Refactor `TernaryVAETrainer` and `AppetitiveVAETrainer` to inherit from the base class.
