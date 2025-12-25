# GitHub Organization Configuration Guide

This guide explains the special repositories and files created in `org-defaults/` to standardize the **Ai-Whisperers** organization.

## 1. Special Repositories

### The `.github` Repository (Public Defaults)

This repository is the command center for your organization's public presence and default community health standards. Files placed here apply to **all** repositories in the organization that do not define their own.

- **Action:** Create a public repo named `.github` and push the contents of `org-defaults/.github` to it.

### The `.github-private` Repository (Internal Defaults)

Used for internal documentation (`profile/README.md` visible only to members).

## 2. Default Files (The `.github` Repo)

We have generated these files in `c:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\org-defaults\.github`:

- **Profile (Home Page):** `profile/README.md`
  - _Path:_ `c:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\org-defaults\.github\profile\README.md`
- **Issue Templates:**
  - _Bug:_ `c:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\org-defaults\.github\ISSUE_TEMPLATE\bug_report.yml`
  - _Feature:_ `c:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\org-defaults\.github\ISSUE_TEMPLATE\feature_request.yml`
- **PR Template:** `PULL_REQUEST_TEMPLATE.md`
  - _Path:_ `c:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\org-defaults\.github\PULL_REQUEST_TEMPLATE.md`
- **Health Files:**
  - _Security:_ `c:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\org-defaults\.github\SECURITY.md`
  - _Funding:_ `c:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\org-defaults\.github\FUNDING.yml`
  - _Conduct:_ `c:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\org-defaults\.github\CODE_OF_CONDUCT.md`
  - _Contributing:_ `c:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\org-defaults\.github\CONTRIBUTING.md`

## 3. Tooling Template Repository (The `organization-template` Repo)

GitHub cannot enforce `ruff` or `flake8` configs globally via `.github`. You must use a **Template Repository**.

- **Action:** Create a repo named `organization-template` in your organization. Check **"Template repository"** in Settings. Push the contents of `org-defaults/template/` to it.
- **Usage:** When creating a _new_ project, select `Ai-Whisperers/organization-template` as the starter.

### Included Configuration Files

Copy these from `org-defaults/template/` to your new `organization-template` repo:

- **Ruff Config (Linting):**
  - `c:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\org-defaults\template\ruff.toml`
- **Flake8 Config (Style):**
  - `c:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\org-defaults\template\.flake8`
- **CSpell Config (Spelling):**
  - `c:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\org-defaults\template\cspell.json`
- **Project Vocabulary:**
  - `c:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\org-defaults\template\configs\project-words.txt`
- **Instructions:**
  - `c:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\org-defaults\template\README.md`
