● Bioinformatics Scripts - Ingestion & Models Summary

  Directory Structure

  bioinformatics/
  ├── rheumatoid_arthritis/   # 29 scripts - HLA alleles, citrullination
  ├── neurodegeneration/alzheimers/  # 6 scripts - tau phosphorylation
  ├── hiv/                    # 5 scripts - CTL escape, glycan shield
  └── sars_cov_2/             # 2 scripts - spike glycan analysis

  Primary Model: 3-Adic Codon Encoder (V5.11.3)

  Path: ../genetic_code/data/codon_encoder_3adic.pt

  Architecture:
  - Input: 12-dim one-hot codon encoding (3 nucleotides × 4 bases)
  - Output: 16-dim hyperbolic embeddings (Poincaré ball)
  - Network: nn.Sequential(Linear → ReLU → ReLU → Linear) + clustering head

  Loading pattern in hyperbolic_utils.py:
  encoder, mapping, native_hyperbolic = load_codon_encoder(
      device='cpu',
      version='3adic'
  )

  Data Ingestion Patterns

  | Source Type         | Examples                                          |
  |---------------------|---------------------------------------------------|
  | Hardcoded sequences | HLA-DRB1 alleles, HIV epitopes, tau sites         |
  | Downloaded          | Human proteome (UniProt) → results/proteome_wide/ |
  | JSON configs        | AlphaFold3 batch inputs/outputs                   |
  | Model checkpoints   | torch.load() for codon encoder                    |

  AlphaFold3 Integration

  All modules generate AF3 inputs and consume .cif structure predictions:
  - HIV: BG505 gp120 glycan variants
  - SARS-CoV-2: RBD-ACE2 complexes
  - RA: HLA-peptide-TCR complexes

  Key Dependencies

  torch, numpy, scipy, sklearn, pandas, matplotlib, seaborn

  Shared Infrastructure

  rheumatoid_arthritis/scripts/hyperbolic_utils.py - used by all disease modules for:
  - codon_to_onehot(), poincare_distance(), encode_sequence_hyperbolic()
  - 21 clusters (20 amino acids + stop codon)
  - Hyperbolic curvature c=1.0