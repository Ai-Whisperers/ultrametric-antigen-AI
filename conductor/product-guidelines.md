# Product Guidelines: Ternary VAE v5.11

## Documentation Style
- **Primary Tone**: **Scientific & Rigorous**. Documentation should prioritize mathematical accuracy and clear theoretical grounding, suitable for an academic and advanced research audience.
- **Secondary Tone**: **Practical & Modular**. Code documentation (docstrings, READMEs) must be clear, standard-compliant (Google/NumPy style), and focused on usability to support modular development.
- **Narrative Elements**: Complex architectural choices should be justified with reference to experimental insights or theoretical necessities (e.g., "why hyperbolic space?").

## Code Standards
- **Architectural Pattern**: **Modular & Component-Based**. Adhere strictly to the Single Responsibility Principle (SRP). Components (Models, Trainers, Losses) must be decoupled and independently testable.
- **Reproducibility**: All experiments must be deterministic where possible. Seeds must be fixed, and configurations should drive all hyperparameters—no magic numbers in code.
- **Testing**: Maintain high test coverage (>90%). Critical mathematical components (metrics, losses) require unit tests verifying edge cases and numerical stability.
- **Typing**: Strict static typing (`mypy`) is required for all core logic to ensure interface contracts are respected.

## Design Philosophy
- **Geometry First**: The architecture should reflect the underlying geometry. Operations should respect the Poincare ball constraints (e.g., using Mobius addition).
- **Zero-Transfer Efficiency**: Prioritize GPU-resident data structures to eliminate CPU-GPU bottlenecks.
- **Observability**: The system must be "transparent." Complex dynamics (StateNet decisions, loss components) must be logged extensively to TensorBoard for analysis.

## Workflow Protocols
- **Atomic Commits**: Changes should be small, focused, and verified by the test suite.
- **Configuration-Driven Development**: New features or experiments should be toggleable via YAML configuration files, preserving backward compatibility where feasible.
