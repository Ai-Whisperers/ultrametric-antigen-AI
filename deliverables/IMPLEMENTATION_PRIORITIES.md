# Implementation Priorities
## Quick Wins to Maximum Impact

**Created:** December 29, 2025
**Goal:** Prioritize work for maximum "wow factor" per effort

---

## Priority Matrix

```
                        HIGH IMPACT
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         │   B1: Activity   │   A2: Real       │
         │   Predictors     │   Sequences      │
         │   ★★★★★          │   ★★★★★          │
         │                  │                  │
         │   C4: ProTherm   │   H6: Stanford   │
         │   Training       │   Integration    │
         │   ★★★★☆          │   ★★★★☆          │
    LOW  │──────────────────┼──────────────────│ HIGH
   EFFORT│                  │                  │ EFFORT
         │   VAE Service    │   C1: PyRosetta  │
         │   Integration    │   Pipeline       │
         │   ★★★★★          │   ★★★☆☆          │
         │                  │                  │
         │   B8/B10: Model  │   H7: PK Model   │
         │   Refinement     │   + EHR          │
         │   ★★★☆☆          │   ★★☆☆☆          │
         │                  │                  │
         └──────────────────┼──────────────────┘
                            │
                       LOW IMPACT
```

---

## Top 5 Quick Wins (Do First!)

### 1. VAE Service Integration (2-3 hours)
**Impact:** Transforms ALL tools from mock to real
**Effort:** Low - just connecting existing code

```python
# File: deliverables/shared/vae_service.py

import torch
import sys
sys.path.insert(0, '../..')

from src.models import TernaryVAEV5_11_PartialFreeze

class VAEService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TernaryVAEV5_11_PartialFreeze(
            latent_dim=16,
            hidden_dim=64,
            max_radius=0.99,
            curvature=1.0,
            use_controller=True,
            use_dual_projection=True
        )

        checkpoint = torch.load(
            'sandbox-training/checkpoints/homeostatic_rich/best.pt',
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def encode(self, sequence):
        """Sequence → latent vector"""
        with torch.no_grad():
            # Convert sequence to ternary operations
            ops = self._sequence_to_ops(sequence)
            output = self.model(ops)
            return output['z_B_hyp']  # Use VAE-B for hierarchy

    def decode(self, z):
        """Latent vector → sequence"""
        with torch.no_grad():
            # Decode from latent space
            return self.model.decode(z)

    def get_padic_valuation(self, z):
        """Get p-adic valuation from radial position"""
        radius = z.norm(dim=-1)
        # Map radius [0, 0.99] to valuation [9, 0]
        valuation = 9 * (1 - radius / 0.99)
        return valuation.round().int()
```

### 2. NCBI Arbovirus Downloader (4-6 hours)
**Impact:** Makes A2 use real viral sequences
**Effort:** Medium - API calls and caching

```python
# File: deliverables/alejandra_rojas/scripts/ncbi_downloader.py

from Bio import Entrez
import json
import os

Entrez.email = "researcher@university.edu"

ARBOVIRUS_TAXIDS = {
    'DENV-1': 11053,
    'DENV-2': 11060,
    'DENV-3': 11069,
    'DENV-4': 11070,
    'ZIKV': 64320,
    'CHIKV': 37124,
    'MAYV': 59301
}

def download_sequences(virus, max_seqs=100):
    """Download complete genomes from NCBI"""
    taxid = ARBOVIRUS_TAXIDS[virus]

    # Search for complete genomes
    handle = Entrez.esearch(
        db="nucleotide",
        term=f"txid{taxid}[Organism] AND complete genome[Title]",
        retmax=max_seqs
    )
    record = Entrez.read(handle)
    ids = record['IdList']

    # Fetch sequences
    handle = Entrez.efetch(
        db="nucleotide",
        id=ids,
        rettype="fasta",
        retmode="text"
    )
    return handle.read()

def build_local_database():
    """Download all arboviruses to local cache"""
    cache_dir = "../data/ncbi_cache"
    os.makedirs(cache_dir, exist_ok=True)

    for virus, taxid in ARBOVIRUS_TAXIDS.items():
        print(f"Downloading {virus}...")
        sequences = download_sequences(virus)
        with open(f"{cache_dir}/{virus}_genomes.fasta", 'w') as f:
            f.write(sequences)
```

### 3. DRAMP Activity Data (4-6 hours)
**Impact:** Real MIC training data for B1
**Effort:** Medium - download and process

```python
# File: deliverables/carlos_brizuela/scripts/dramp_downloader.py

import pandas as pd
import requests
from bs4 import BeautifulSoup

def download_dramp_data():
    """Download antimicrobial peptide data from DRAMP"""
    # DRAMP provides downloadable datasets
    url = "http://dramp.cpu-bioinfor.org/downloads/"

    # Get general AMPs with activity data
    amps_df = pd.read_csv(
        "http://dramp.cpu-bioinfor.org/downloads/download_data/DRAMP_general_amps.csv"
    )

    # Filter for relevant pathogens
    pathogens = [
        'Acinetobacter baumannii',
        'Pseudomonas aeruginosa',
        'Klebsiella pneumoniae',
        'Staphylococcus aureus'
    ]

    activity_data = amps_df[
        amps_df['Target_Organism'].isin(pathogens)
    ]

    return activity_data

def prepare_training_data(df):
    """Extract features for ML training"""
    features = []
    for _, row in df.iterrows():
        seq = row['Sequence']
        features.append({
            'sequence': seq,
            'length': len(seq),
            'charge': calculate_charge(seq),
            'hydrophobicity': calculate_hydrophobicity(seq),
            'target': row['Target_Organism'],
            'mic': row['MIC_Value']
        })
    return pd.DataFrame(features)
```

### 4. Stanford HIVdb Integration (3-4 hours)
**Impact:** Professional resistance reports for H6
**Effort:** Low - REST API calls

```python
# File: deliverables/hiv_research_package/scripts/stanford_integration.py

import requests
import json

STANFORD_GRAPHQL = "https://hivdb.stanford.edu/graphql"

def analyze_sequence(sequence):
    """Send sequence to Stanford HIVdb for analysis"""

    query = """
    mutation AnalyzeSequence($sequences: [UnalignedSequenceInput]!) {
      viewer {
        sequenceAnalysis(sequences: $sequences) {
          inputSequence {
            header
          }
          bestMatchingSubtype {
            display
          }
          drugResistance {
            gene {
              name
            }
            drugScores {
              drug {
                name
                displayAbbr
              }
              score
              level
              text
            }
            mutationsByTypes {
              mutationType
              mutations {
                text
                primaryType
              }
            }
          }
        }
      }
    }
    """

    variables = {
        "sequences": [{
            "header": "patient_sequence",
            "sequence": sequence
        }]
    }

    response = requests.post(
        STANFORD_GRAPHQL,
        json={"query": query, "variables": variables}
    )

    return response.json()

def format_resistance_report(stanford_result):
    """Convert Stanford output to clinical report"""
    analysis = stanford_result['data']['viewer']['sequenceAnalysis'][0]

    report = {
        'subtype': analysis['bestMatchingSubtype']['display'],
        'drugs': []
    }

    for drug_class in analysis['drugResistance']:
        for drug in drug_class['drugScores']:
            report['drugs'].append({
                'name': drug['drug']['displayAbbr'],
                'score': drug['score'],
                'level': drug['level'],
                'interpretation': drug['text']
            })

    return report
```

### 5. ProTherm Training Data (4-6 hours)
**Impact:** Real ΔΔG predictions for C4
**Effort:** Medium - data processing

```python
# File: deliverables/jose_colbes/scripts/protherm_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def load_protherm():
    """Load and process ProTherm database"""
    # ProTherm can be downloaded from:
    # https://web.iitm.ac.in/bioinfo2/prothermdb/

    df = pd.read_csv("data/protherm_single_mutations.csv")

    # Filter quality entries
    df = df[df['ddG'].notna()]
    df = df[df['Measurement_Method'] != 'unknown']

    return df

def extract_features(row):
    """Extract features for each mutation"""
    wild = row['Wild_Type']
    mut = row['Mutant']
    pos = row['Position']

    return {
        'volume_change': VOLUMES[mut] - VOLUMES[wild],
        'hydrophobicity_change': HYDROPHOBICITY[mut] - HYDROPHOBICITY[wild],
        'charge_change': CHARGES[mut] - CHARGES[wild],
        'position': pos,
        'secondary_structure': row['Secondary_Structure'],
        'solvent_accessibility': row['RSA']
    }

def train_ddg_predictor():
    """Train ΔΔG predictor on ProTherm data"""
    df = load_protherm()

    # Extract features
    X = pd.DataFrame([extract_features(row) for _, row in df.iterrows()])
    y = df['ddG']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05
    )
    model.fit(X_train, y_train)

    # Evaluate
    from scipy.stats import pearsonr
    y_pred = model.predict(X_test)
    r, p = pearsonr(y_test, y_pred)
    print(f"Pearson r: {r:.3f}, p-value: {p:.2e}")

    return model
```

---

## Implementation Order

### Week 1: Foundation + Quick Wins

| Day | Task | Tools Affected |
|-----|------|---------------|
| 1-2 | VAE Service Integration | All |
| 3 | NCBI Downloader | A2 |
| 4 | DRAMP Downloader | B1, B8 |
| 5 | Stanford API Integration | H6 |

**End of Week 1 Checkpoint:**
- All tools connected to real VAE
- A2 running on real arbovirus sequences
- B1 training on real AMP data
- H6 producing Stanford-quality reports

### Week 2: Model Training

| Day | Task | Tools Affected |
|-----|------|---------------|
| 1-2 | Activity predictor training | B1 |
| 3 | Selectivity model refinement | B8 |
| 4 | ProTherm ΔΔG training | C4 |
| 5 | Validation and benchmarking | All |

### Week 3: Polish + Validation

| Day | Task | Tools Affected |
|-----|------|---------------|
| 1-2 | In-silico primer validation | A2 |
| 3 | Rosetta integration (optional) | C1 |
| 4 | Report generation | All |
| 5 | Documentation updates | All |

---

## Minimum Viable "Amazing" (MVA)

If time is limited, focus on these for maximum impact:

### Must-Have (3 days)
1. ✅ VAE Service (connects all tools to real model)
2. ✅ NCBI sequences for A2 (real primers)
3. ✅ Stanford API for H6 (professional reports)

### Should-Have (5 days)
4. ⬜ DRAMP training data for B1 (validated activity)
5. ⬜ ProTherm training for C4 (real ΔΔG)

### Nice-to-Have (7+ days)
6. ⬜ PyRosetta for C1 (real Rosetta scores)
7. ⬜ PK modeling for H7 (drug level predictions)
8. ⬜ Multiplex panel design for A2 (advanced feature)

---

## "Wow Factor" Demonstrations

### For Alejandra Rojas
**Demo:** "Here are primers that work on real Paraguay dengue sequences from 2024"
- Show alignment of 50+ DENV-1 Paraguay isolates
- Highlight conserved regions where primers bind
- Display cross-reactivity matrix proving specificity

### For Carlos Brizuela
**Demo:** "These peptides are predicted to kill A. baumannii with MIC < 4 μg/mL"
- Show activity predictions from trained model
- Display ESMFold structure of top candidate
- Compare to existing AMPs in literature

### For Jose Colbes
**Demo:** "We found 23% of residues where Rosetta is wrong - here's the experimental proof"
- Show correlation with ProTherm ΔΔG values
- Highlight specific residues where p-adic succeeds
- Visualize discordance in 3D structure

### For HIV Package
**Demo:** "This report matches Stanford HIVdb exactly, plus adds p-adic stability"
- Side-by-side comparison with Stanford
- Show additional p-adic predictions
- Display eligibility assessment for LA therapy

---

## Files to Create

```
deliverables/
├── shared/
│   ├── __init__.py
│   ├── vae_service.py           # Centralized VAE interface
│   ├── config.py                # Shared configuration
│   └── utils.py                 # Common utilities
│
├── alejandra_rojas/
│   ├── scripts/
│   │   ├── ncbi_downloader.py   # NEW: Real sequence fetching
│   │   └── primer_validator.py  # NEW: In-silico validation
│   └── data/
│       └── ncbi_cache/          # Downloaded sequences
│
├── carlos_brizuela/
│   ├── scripts/
│   │   ├── dramp_downloader.py  # NEW: Training data
│   │   └── activity_trainer.py  # NEW: ML training
│   └── models/
│       └── activity_models/     # Trained predictors
│
├── jose_colbes/
│   ├── scripts/
│   │   ├── protherm_loader.py   # NEW: Training data
│   │   └── ddg_trainer.py       # NEW: ΔΔG training
│   └── models/
│       └── ddg_model.pkl        # Trained predictor
│
└── hiv_research_package/
    ├── scripts/
    │   ├── stanford_integration.py  # NEW: API integration
    │   └── report_generator.py      # NEW: Professional reports
    └── templates/
        └── resistance_report.html   # Report template
```

---

## Success Checklist

### A2: Pan-Arbovirus Primers
- [ ] Real NCBI sequences downloaded (500+)
- [ ] Primers designed from conserved regions
- [ ] In-silico PCR validation passed
- [ ] Cross-reactivity confirmed zero
- [ ] Laboratory protocol generated

### B1: Pathogen-Specific AMPs
- [ ] DRAMP training data loaded (10,000+)
- [ ] Activity predictors trained (AUROC > 0.85)
- [ ] Top 10 candidates identified
- [ ] Synthesis feasibility confirmed
- [ ] ESMFold structures generated

### B8: Microbiome-Safe AMPs
- [ ] Selectivity Index > 4.0 achieved
- [ ] Multi-species MIC predictions
- [ ] Skin microbiome validated
- [ ] Top candidates ready for synthesis

### B10: Synthesis Optimization
- [ ] Real synthesis costs integrated
- [ ] Success rate prediction validated
- [ ] Scale-up recommendations ready

### C1: Rosetta-Blind Detection
- [ ] Real Rosetta scores (or good proxy)
- [ ] Discordance validated on known mutations
- [ ] Amino acid enrichment confirmed

### C4: Mutation Effect Predictor
- [ ] ProTherm training complete
- [ ] Pearson r > 0.6 achieved
- [ ] P-adic features contributing

### H6: TDR Screening
- [ ] Stanford API integrated
- [ ] Reports match Stanford output
- [ ] All WHO mutations covered

### H7: LA Injectable Selection
- [ ] Eligibility criteria validated
- [ ] Success probability calibrated
- [ ] Clinical workflow ready

---

*Implementation Priority Guide*
*Ternary VAE Bioinformatics Partnership*
