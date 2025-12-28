# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Multi-disease framework for unified pathogen/disease modeling.

This module provides a disease-agnostic framework for training models
across multiple disease domains including:
- HIV (drug resistance, immune escape, glycan shielding)
- Rheumatoid Arthritis (citrullination, HLA interactions)
- Neurodegeneration (tau phosphorylation, amyloid aggregation)
- Cancer (neoantigen prediction, immune checkpoint)
- Infectious diseases (emerging pathogens, variant tracking)

Example:
    from src.diseases import DiseaseRegistry, get_disease_config

    # Get all registered diseases
    diseases = DiseaseRegistry.list_diseases()

    # Get specific disease config
    hiv_config = get_disease_config("hiv")

    # Train multi-disease model
    from src.diseases.training import MultiDiseaseTrainer
    trainer = MultiDiseaseTrainer(diseases=["hiv", "ra", "neuro"])
    trainer.train()
"""

from src.diseases.registry import DiseaseRegistry, get_disease_config
from src.diseases.base import DiseaseConfig, DiseaseDataset
from src.diseases.losses import MultiDiseaseLoss
from src.diseases.variant_escape import (
    DiseaseType,
    DrugResistancePredictor,
    EscapePrediction,
    FitnessPredictor,
    ImmuneEscapePredictor,
    MetaLearningEscapeHead,
    ReceptorBindingPredictor,
    VariantEscapeHead,
)
from src.diseases.sars_cov2_analyzer import (
    SARSCoV2Analyzer,
    SARSCoV2Config,
    SARSCoV2Gene,
    SARSCoV2Variant,
    create_sars_cov2_dataset,
)
from src.diseases.tuberculosis_analyzer import (
    TuberculosisAnalyzer,
    TuberculosisConfig,
    TBDrug,
    TBGene,
    ResistanceLevel,
    create_tb_synthetic_dataset,
)
from src.diseases.influenza_analyzer import (
    InfluenzaAnalyzer,
    InfluenzaConfig,
    InfluenzaSubtype,
    InfluenzaGene,
    InfluenzaDrug,
    create_influenza_synthetic_dataset,
)

__all__ = [
    "DiseaseRegistry",
    "get_disease_config",
    "DiseaseConfig",
    "DiseaseDataset",
    "MultiDiseaseLoss",
    # Variant escape prediction
    "VariantEscapeHead",
    "MetaLearningEscapeHead",
    "EscapePrediction",
    "DiseaseType",
    "FitnessPredictor",
    "ImmuneEscapePredictor",
    "DrugResistancePredictor",
    "ReceptorBindingPredictor",
    # SARS-CoV-2 specific
    "SARSCoV2Analyzer",
    "SARSCoV2Config",
    "SARSCoV2Gene",
    "SARSCoV2Variant",
    "create_sars_cov2_dataset",
    # Tuberculosis specific
    "TuberculosisAnalyzer",
    "TuberculosisConfig",
    "TBDrug",
    "TBGene",
    "ResistanceLevel",
    "create_tb_synthetic_dataset",
    # Influenza specific
    "InfluenzaAnalyzer",
    "InfluenzaConfig",
    "InfluenzaSubtype",
    "InfluenzaGene",
    "InfluenzaDrug",
    "create_influenza_synthetic_dataset",
]
