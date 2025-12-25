# AI Environment Specification

**Version**: 1.0.0
**Last Updated**: 2025-12-24
**Status**: Active

## Overview

This document defines the Artificial Intelligence Coding Environment (AICE) for the **Ternary VAEs Bioinformatics** repository. It standardizes the context, tools, and rules for all AI agents (Claude, Gemini, Cursor, Antigravity) to ensure scientific rigor and code quality.

## 1. Context Anchors

| Agent      | Context File     | Location | Purpose                                          |
| :--------- | :--------------- | :------- | :----------------------------------------------- |
| **All**    | `ANTIGRAVITY.md` | `ROOT`   | High-level architectural goals and constraints.  |
| **Claude** | `CLAUDE.md`      | `ROOT`   | Project entry point, quick start, and CLI rules. |
| **Gemini** | `GEMINI.md`      | `ROOT`   | Project memory for Gemini CLI.                   |

## 2. Configuration (`.claude/`)

- **Settings**: `settings.json` configured for **PyTorch/Python**.
- **MCP Servers**:
  - `filesystem`: Read/Write access.
  - `terminal`: Command execution.
  - `brave-search`: Access to external documentation.
- **Commands** (`.claude/commands/`):
  - `/lint`: `ruff check .`
  - `/doc`: `sphinx-build ...`
  - `/test`: `pytest --cov=src tests/`
- **Exemplars** (`.claude/exemplars/`):
  - `pytorch_module.py`: Reference `nn.Module` with strict typing.
  - `test_pattern.py`: Reference `pytest` fixture setup.

## 3. Rules (`.cursor/rules/`)

- `antigravity.mdc`: Persona alignment for high-agency/autonomy.
- `bioinformatics.mdc`: Reproducibility (seeds) and data validation standards.
- `documentation.mdc`: Google-style docstrings enforcement.
- `pytorch.mdc`: PyTorch 2.0+, typed tensors, device-agnostic code.

## 4. Gemini Configuration (`.gemini/`)

- **System Prompt**: `system_instructions.md` (Senior Bioinformatics Researcher).
- **MCPs**: `mcp_config.json` (Parity with Claude).

## 5. Maintenance

- **Update Protocol**: When project dependencies change (`requirements.txt`), review `CLAUDE.md` and Rules.
- **Exemplars**: Add new patterns to `exemplars/` as they emerge.
