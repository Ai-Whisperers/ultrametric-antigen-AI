# Risk Register – Project Management

## Overview

The risk register captures potential threats to the **Ternary VAE Bioinformatics** project, especially those that could enable external parties (e.g., large pharmaceutical companies) to appropriate or monopolize our code, data, or results. Each risk is described, assigned a severity, likelihood, and a concrete mitigation strategy.

## Risks

| ID  | Risk Description                                                                                                      | Category               | Severity (1‑5) | Likelihood (1‑5) | Mitigation                                                                                                                              |
| --- | --------------------------------------------------------------------------------------------------------------------- | ---------------------- | -------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| R1  | **IP Leakage – Unlicensed commercial use of source code**                                                             | Legal / Licensing      | 5              | 3                | Enforce PolyForm Non‑Commercial license, add SPDX headers to every file, monitor public forks, and require signed CLA for contributors. |
| R2  | **Model Weights Redistribution** – Companies download pretrained checkpoints and embed them in proprietary pipelines. | Legal / Data           | 4              | 4                | CC‑BY‑4.0 results license, watermark checkpoints, embed license file in `results/` and require attribution.                             |
| R3  | **Patent Claims on Hyperbolic Embedding** – Third‑party patents could block our methodology.                          | Patent / Legal         | 4              | 2                | File defensive patent applications for core algorithms (hyperbolic loss, 3‑adic ranking).                                               |
| R4  | **Export‑Control Violations** – Sharing virus‑related data with embargoed countries.                                  | Compliance             | 5              | 2                | Include export‑control notice in `OPEN_MEDICINE_POLICY.md`; restrict data distribution to approved institutions.                        |
| R5  | **Dependency Vulnerabilities** – Third‑party libraries with security flaws.                                           | Security               | 3              | 3                | Use `requirements.txt` pinned versions, run `pip-audit` in CI, and update regularly.                                                    |
| R6  | **Data Privacy Breach** – Patient‑derived datasets inadvertently contain PII.                                         | Privacy                | 4              | 2                | Anonymize all clinical datasets, store them under `data/` with access controls, and run privacy‑audit scripts.                          |
| R7  | **Fork‑and‑Abandon** – External forks become a competing closed source version.                                       | Community / Reputation | 3              | 3                | Monitor GitHub forks, add a “Contributor License Agreement” that requires upstream contributions to remain open.                        |
| R8  | **Insufficient Documentation** – Ambiguous licensing leads to misuse.                                                 | Process                | 3              | 4                | Add SPDX headers, badge, and explicit licensing sections in every README and documentation file.                                        |
| R9  | **Funding Dependency** – Reliance on a single commercial sponsor could force a license change.                        | Financial              | 4              | 2                | Diversify funding sources (grants, non‑profit partnerships) and keep a “non‑commercial only” clause in the license.                     |
| R10 | **Academic Publication Delay** – Results not published promptly, allowing others to claim priority.                   | Reputation             | 2              | 3                | Maintain a public pre‑print repository (arXiv) and timestamp all major findings.                                                        |

## Mitigation Workflow

1. **Risk Identification** – Quarterly review by the Project Lead and Legal Advisor.
2. **Assessment** – Update severity/likelihood scores in this register.
3. **Action Planning** – Assign owners (e.g., `Legal`, `Security`, `Data Steward`).
4. **Monitoring** – Track mitigation status in the `02_PROJECT_MANAGEMENT/02_CODE_HEALTH_METRICS` dashboard.
5. **Escalation** – If a risk reaches severity ≥ 4 and likelihood ≥ 4, trigger a formal risk‑response meeting.

---

_Prepared on 2025‑12‑24 as part of the “Analyze Theory Foundations and Update Project Management” task._
