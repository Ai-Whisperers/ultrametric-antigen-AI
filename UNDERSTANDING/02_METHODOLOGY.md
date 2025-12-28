# Methodology: The Engineering Implementation

**How we translate biology into geometry.**

---

## 1. Data Ecosystem

We utilize a multi-tiered data strategy involving over **440,000 sequences** and **state-of-the-art language models**.

### Primary Datasets

| Source             | Size  | Content               | Utility                              |
| :----------------- | :---- | :-------------------- | :----------------------------------- |
| **Stanford HIVDB** | 200k+ | Pol Gene + Phenotypes | Training Ground Truth for Resistance |
| **LANL CATNAP**    | 190k  | Env + Neutralization  | Immune Escape & Glycan Shield        |
| **LANL CTL**       | 2,115 | Epitopes + HLA        | T-Cell Pressure Mapping              |
| **AlphaFold DB**   | 50+   | 3D Structures         | Spatial Binding Sites                |

### The ESM-2 Integration (2025 Upgrade)

We integrated Meta's **ESM-2 (650M)** protein language model to replace strict One-Hot encoding.

- **Model**: `facebook/esm2_t33_650M_UR50D`
- **dim**: 1280
- **Why**: Implicitly captures evolutionary constraints trained on 250M+ sequences.

---

## 2. The Modeling Pipeline

### Step 1: Ternary Encoding (The "Old" Way)

- **Input**: DNA Sequence.
- **Transform**: Codon Indexing (0-63).
- **Logic**: Captures the "Wobble" structure (P-adic distances).

### Step 2: Language Embedding (The "New" Way)

- **Input**: Amino Acid Sequence.
- **Transform**: Transformer Attention Output (`[Batch, Seq_Len, 1280]`).
- **Logic**: Captures long-range dependencies and biophysical properties.

### Step 3: Hyperbolic VAE

- **Encoder**: Maps input $\to$ Poincar\'e Ball ($\mathbb{B}^n$).
- **Latent Space**: Structured by P-adic losses (Triplet + Radial).
- **Decoder**: Reconstructs original sequence.

---

## 3. Training Infrastructure

- **Hardware**: A100 (40GB) / RTX 4090.
- **Optimizer**: Riemannian Adam (Geoopt).
- **Scheduler**: **Cyclical Beta** (The "Heartbeat").
  - $\beta$ ramps $0 \to 0.1 \to 0$ over 10 epochs.
  - Allows alternating phases of **Extension** (Learning structure) and **Consolidation** (Learning reconstruction).

---

## 4. The Validation Harness

No model is deployed without passing the **Geometric Stress Tests**:

1.  **Triangle Inequality**: Verifies latent space metric axioms.
2.  **Gradient Orthogonality**: Checks if Structure Loss $\perp$ Reconstruction Loss.
3.  **Boundary Collapse**: Ensures distribution isn't degenerate ($r=0$ or $r=1$).

---

## 5. API Integrations

### Stanford HIVDB API

- **Endpoint**: `https://hivdb.stanford.edu/page/webservice`
- **Use**: Real-time resistance scoring for validation.

### ESM Inference

- **Library**: HuggingFace Transformers (`EsmModel`).
- **Use**: Generating embeddings on the fly for new sequences.
