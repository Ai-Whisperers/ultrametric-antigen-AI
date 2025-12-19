"""Loss function components.

This module contains loss computation separated from model architecture.

Architecture:
    - base.py: LossComponent protocol and LossResult dataclass
    - registry.py: LossRegistry for dynamic composition
    - components.py: LossComponent wrappers for all loss types

Legacy classes (for backwards compatibility):
    - DualVAELoss: Aggregated loss (deprecated, use LossRegistry)
    - ReconstructionLoss, KLDivergenceLoss, etc.

New pattern (recommended):
    from src.losses import LossRegistry, ReconstructionLossComponent

    registry = LossRegistry()
    registry.register('recon', ReconstructionLossComponent())
    result = registry.compose(outputs, targets)

Loss Types:
    - Reconstruction: Cross-entropy for ternary operations
    - KL Divergence: With free bits support
    - Entropy: Regularization for diversity
    - Repulsion: Latent space diversity
    - p-Adic: Metric alignment, ranking, hierarchical norms
    - Hyperbolic: Poincare ball geometry (v5.10)
    - Radial: Stratification for 3-adic hierarchy
"""

from .dual_vae_loss import (
    ReconstructionLoss,
    KLDivergenceLoss,
    EntropyRegularization,
    RepulsionLoss,
    DualVAELoss
)

from .padic_losses import (
    PAdicMetricLoss,
    PAdicRankingLoss,
    PAdicRankingLossV2,
    PAdicRankingLossHyperbolic,
    PAdicNormLoss
)

from .appetitive_losses import (
    AdaptiveRankingLoss,
    HierarchicalNormLoss,
    CuriosityModule,
    SymbioticBridge,
    AlgebraicClosureLoss,
    ViolationBuffer
)

from .hyperbolic_prior import (
    HyperbolicPrior,
    HomeostaticHyperbolicPrior
)

from .hyperbolic_recon import (
    HyperbolicReconLoss,
    HomeostaticReconLoss,
    HyperbolicCentroidLoss
)

from .radial_stratification import (
    RadialStratificationLoss,
    compute_single_index_valuation
)

from .padic_geodesic import (
    poincare_distance,
    PAdicGeodesicLoss,
    RadialHierarchyLoss,
    CombinedGeodesicLoss,
    GlobalRankLoss,
    MonotonicRadialLoss
)

from .consequence_predictor import (
    ConsequencePredictor,
    evaluate_addition_accuracy,
    PurposefulRankingLoss
)

# New structural components (LossRegistry pattern)
from .base import (
    LossResult,
    LossComponent,
    DualVAELossComponent
)

from .registry import (
    LossRegistry,
    LossGroup,
    create_registry_from_config
)

from .components import (
    ReconstructionLossComponent,
    KLDivergenceLossComponent,
    EntropyLossComponent,
    RepulsionLossComponent,
    EntropyAlignmentComponent,
    PAdicRankingLossComponent,
    PAdicHyperbolicLossComponent,
    RadialStratificationLossComponent
)

__all__ = [
    # Legacy classes (backwards compatibility)
    'ReconstructionLoss',
    'KLDivergenceLoss',
    'EntropyRegularization',
    'RepulsionLoss',
    'DualVAELoss',
    'PAdicMetricLoss',
    'PAdicRankingLoss',
    'PAdicRankingLossV2',
    'PAdicRankingLossHyperbolic',
    'PAdicNormLoss',
    'AdaptiveRankingLoss',
    'HierarchicalNormLoss',
    'CuriosityModule',
    'SymbioticBridge',
    'AlgebraicClosureLoss',
    'ViolationBuffer',
    'HyperbolicPrior',
    'HomeostaticHyperbolicPrior',
    'HyperbolicReconLoss',
    'HomeostaticReconLoss',
    'HyperbolicCentroidLoss',
    'RadialStratificationLoss',
    'compute_single_index_valuation',
    # V5.11 Unified Geodesic Loss
    'poincare_distance',
    'PAdicGeodesicLoss',
    'RadialHierarchyLoss',
    'CombinedGeodesicLoss',
    'GlobalRankLoss',
    'MonotonicRadialLoss',
    'ConsequencePredictor',
    'evaluate_addition_accuracy',
    'PurposefulRankingLoss',
    # New structural components (LossRegistry pattern)
    'LossResult',
    'LossComponent',
    'DualVAELossComponent',
    'LossRegistry',
    'LossGroup',
    'create_registry_from_config',
    'ReconstructionLossComponent',
    'KLDivergenceLossComponent',
    'EntropyLossComponent',
    'RepulsionLossComponent',
    'EntropyAlignmentComponent',
    'PAdicRankingLossComponent',
    'PAdicHyperbolicLossComponent',
    'RadialStratificationLossComponent'
]
