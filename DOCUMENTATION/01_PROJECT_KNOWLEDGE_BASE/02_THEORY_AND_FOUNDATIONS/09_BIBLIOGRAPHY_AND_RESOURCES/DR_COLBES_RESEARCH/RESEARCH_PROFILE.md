# Research Profile: Dr. José Domingo Colbes Sanabria

**Focus Areas:** Combinatorial Optimization, Protein Structure Prediction, Deep Learning.

---

## Key Papers & Findings

### 1. "Protein side-chain packing problem: is there still room for improvement?" (2016)

- **Focus:** Benchmarking side-chain packing algorithms.
- **Key Insight:** Classical heuristics (SCWRL, etc.) are hitting a performance ceiling. New scoring functions or optimization methods are needed.
- **Relevance:** Validates our "Geometric Scoring Function" approach as a necessary novelty.

### 2. "Scoring of side-chain packings: An analysis of weight factors" (JCIM 2018)

- **Focus:** Evaluated how different terms in the energy function (Van der Waals, Electrostatics) are weighted.
- **Relevance:** He deeply understands the _components_ of a scoring function. We can present "3-adic distance" as a new, orthogonal term $E_{geom}$ to be added to the standard $E_{phys}$.

### 3. "Optimization in Positioning of Police Resources" (CLEI 2022)

- **Method:** Used **Tabu Search** to solve a coverage problem.
- **Relevance:** Shows he is an expert in _metaheuristics_ (Tabu Search, Genetic Algorithms). He thinks in terms of "escaping local optima".

### 4. "Performance Analysis of Deep Learning Methods for Protein Contact Prediction in CASP13" (CLEI 2021)

- **Focus:** Analyzed DeepMetaPSICOV and ResNets.
- **Key Insight:** Acknowledges the shift from classical optimization to Deep Learning.
- **Relevance:** Bridges the gap to our VAE. He knows Deep Learning is the future of folding.

---

## Algorithmic Palette

- **Optimization:** Tabu Search, Genetic Algorithms (GA), Simulated Annealing.
- **Machine Learning:** Deep Learning (CNNs, ResNets), Support Vector Machines.
- **Bioinformatics:** CASP benchmarks, Side-chain packing (PSCPP).

## Strategic Hooks for Presentation

1.  **"Natural Genetic Algorithms"**: Frame the evolution of the genetic code as a multi-objective GA (minimizing 3-adic distance + maximizing stability).
2.  **"Escape from Local Optima"**: Frame the VAE's latent space as a smoothed landscape that avoids the local optima plaguing classical side-chain packing.
3.  **"The Missing Term"**: Propose $E_{total} = w_1 E_{physical} + w_2 E_{geometric}$ (where $E_{geometric}$ is our p-adic score).
