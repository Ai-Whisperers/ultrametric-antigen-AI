# 🎯 Tier 1 Presentation – Executive Overview (Pasteur Molecule‑Binding Project)

> **Speaker Note:** Begin with a brief story about the economic impact of parasitic worms in cattle and the need for rapid therapeutic discovery.

---

## 1️⃣ Problem Statement

- **Target disease:** Gastro‑intestinal nematodes affecting bovine livestock (e.g., _Haemonchus contortus_, _Cooperia_ spp.).
- **Economic burden:** > US $2 billion annual losses in the global cattle industry due to reduced weight gain, milk production, and mortality.
- **Current bottleneck:** Drug discovery is **manual, slow, and costly** – screening thousands of compounds in vitro takes months.

---

## 2️⃣ Project Vision (Institut Pasteur Collaboration)

- **Goal:** Build an **AI‑augmented pipeline** that predicts how well candidate small‑molecule compounds bind to a curated set of worm‑specific protein targets **before** any wet‑lab work.
- **Outcome:** A ranked shortlist of the **top 10‑20** molecules ready for rapid in‑vitro validation, cutting discovery time by > 80 %.

---

## 3️⃣ High‑Level Approach

| Step                           | What we do                                                                                                                                                       | Why it matters                                                                                  |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **A. Target Curation**         | Identify and retrieve 3‑D structures of essential worm proteins (e.g., glutamate‑gated chloride channels, acetylcholine receptors).                              | Provides a **specific therapeutic window** – host‑selective targets reduce off‑target toxicity. |
| **B. Ligand Library**          | Assemble a virtual library (≈ 500 k compounds) from public sources (ZINC, ChEMBL) and Pasteur’s proprietary collections.                                         | Broad chemical space ensures we do not miss novel scaffolds.                                    |
| **C. Fast Docking + Scoring**  | Use GPU‑accelerated docking (AutoDock‑GPU) to generate pose ensembles, followed by a **machine‑learning rescoring model** trained on known worm‑protein binders. | Balances speed (hours) with predictive accuracy (≈ R² 0.65).                                    |
| **D. Generative VAE**          | Train a **Ternary VAE** on the top‑scoring ligands to **explore nearby chemical space** and propose optimized analogues.                                         | Enables **in‑silico optimisation** of potency, solubility, and selectivity.                     |
| **E. Experimental Validation** | Rapid‑turnaround biochemical assays (fluorescence‑polarisation) on the top 10‑20 hits.                                                                           | Confirms computational predictions and feeds back into the model.                               |

---

## 4️⃣ Expected Impact

- **Time‑to‑lead:** < 6 weeks vs. > 6 months traditionally.
- **Cost reduction:** ≈ 90 % fewer reagents and personnel hours.
- **Strategic advantage:** Early‑stage **IP‑ready** candidates for licensing or internal development.

---

## 5️⃣ Timeline (12 weeks total)

| Week  | Milestone                                                        |
| ----- | ---------------------------------------------------------------- |
| 1‑2   | Target protein selection & structure retrieval.                  |
| 3‑4   | Build ligand library & set up docking pipeline.                  |
| 5‑6   | Run high‑throughput docking; train ML rescoring model.           |
| 7‑8   | Train Ternary VAE on top‑500 ligands; generate analogues.        |
| 9‑10  | In‑vitro binding assays on top 20 candidates.                    |
| 11‑12 | Data‑driven model refinement & final report to Institut Pasteur. |

---

## 6️⃣ Next Steps (Immediate)

1. **Kick‑off meeting** with Pasteur scientists to finalize target list.
2. Grant/contract paperwork (if required).
3. Allocate GPU resources on the Ternary VAE cluster.

---

_Prepared for senior stakeholders – all technical details are expanded in the Tier 2 deck._
