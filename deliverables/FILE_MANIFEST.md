# Deliverables File Manifest

This document lists all files to include in each partner's deliverable package.

---

## Carlos Brizuela - Antimicrobial Peptides

### To Copy

````text

deliverables/carlos_brizuela/
├── README.md                          ← Partner-specific guide (created)
├── scripts/
│   ├── __init__.py                    ← From scripts/optimization/__init__.py
│   └── latent_nsga2.py                ← From scripts/optimization/latent_nsga2.py
├── notebooks/
│   └── brizuela_amp_navigator.ipynb   ← From notebooks/partners/brizuela_amp_navigator.ipynb
├── results/
│   └── pareto_peptides.csv            ← From results/partners/brizuela/pareto_peptides.csv
├── data/
│   └── demo_amp_embeddings.csv        ← From data/processed/demo_amp_embeddings.csv
└── docs/
    ├── TECHNICAL_BRIEF.md             ← From DOCUMENTATION/04_PARTNERSHIP_PROJECTS/01_BRIZUELA_AMPS/TECHNICAL_BRIEF.md
    └── IMPLEMENTATION_GUIDE.md        ← From DOCUMENTATION/04_PARTNERSHIP_PROJECTS/01_BRIZUELA_AMPS/IMPLEMENTATION_GUIDE.md
```text


### Copy Commands (PowerShell)


```text
powershell
$src = "C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics"
$dst = "$src\deliverables\carlos_brizuela"

# Create directories
New-Item -ItemType Directory -Force -Path "$dst\scripts"
New-Item -ItemType Directory -Force -Path "$dst\notebooks"
New-Item -ItemType Directory -Force -Path "$dst\results"
New-Item -ItemType Directory -Force -Path "$dst\data"
New-Item -ItemType Directory -Force -Path "$dst\docs"

# Copy files
Copy-Item "$src\scripts\optimization\__init__.py" "$dst\scripts\"
Copy-Item "$src\scripts\optimization\latent_nsga2.py" "$dst\scripts\"
Copy-Item "$src\notebooks\partners\brizuela_amp_navigator.ipynb" "$dst\notebooks\"
Copy-Item "$src\results\partners\brizuela\pareto_peptides.csv" "$dst\results\"
Copy-Item "$src\data\processed\demo_amp_embeddings.csv" "$dst\data\"
Copy-Item "$src\DOCUMENTATION\04_PARTNERSHIP_PROJECTS\01_BRIZUELA_AMPS\TECHNICAL_BRIEF.md" "$dst\docs\"
Copy-Item "$src\DOCUMENTATION\04_PARTNERSHIP_PROJECTS\01_BRIZUELA_AMPS\IMPLEMENTATION_GUIDE.md" "$dst\docs\"
```text


---

## Dr. José Colbes - Protein Optimization

### To Copy:

```text

deliverables/jose_colbes/
├── README.md                          ← Partner-specific guide (created)
├── scripts/
│   ├── ingest_pdb_rotamers.py         ← From scripts/ingest/ingest_pdb_rotamers.py
│   └── rotamer_stability.py           ← From scripts/analysis/rotamer_stability.py
├── notebooks/
│   └── colbes_scoring_function.ipynb  ← From notebooks/partners/colbes_scoring_function.ipynb
├── results/
│   └── rotamer_stability.json         ← From results/partners/colbes/rotamer_stability.json
├── data/
│   └── demo_rotamers.pt               ← From data/processed/demo_rotamers.pt
└── docs/
    ├── TECHNICAL_PROPOSAL.md          ← From DOCUMENTATION/04_PARTNERSHIP_PROJECTS/02_COLBES_OPTIMIZATION/TECHNICAL_PROPOSAL.md
    └── IMPLEMENTATION_GUIDE.md        ← From DOCUMENTATION/04_PARTNERSHIP_PROJECTS/02_COLBES_OPTIMIZATION/IMPLEMENTATION_GUIDE.md
```text


### Copy Commands (PowerShell):

```text
powershell
$src = "C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics"
$dst = "$src\deliverables\jose_colbes"

# Create directories
New-Item -ItemType Directory -Force -Path "$dst\scripts"
New-Item -ItemType Directory -Force -Path "$dst\notebooks"
New-Item -ItemType Directory -Force -Path "$dst\results"
New-Item -ItemType Directory -Force -Path "$dst\data"
New-Item -ItemType Directory -Force -Path "$dst\docs"

# Copy files
Copy-Item "$src\scripts\ingest\ingest_pdb_rotamers.py" "$dst\scripts\"
Copy-Item "$src\scripts\analysis\rotamer_stability.py" "$dst\scripts\"
Copy-Item "$src\notebooks\partners\colbes_scoring_function.ipynb" "$dst\notebooks\"
Copy-Item "$src\results\partners\colbes\rotamer_stability.json" "$dst\results\"
Copy-Item "$src\data\processed\demo_rotamers.pt" "$dst\data\"
Copy-Item "$src\DOCUMENTATION\04_PARTNERSHIP_PROJECTS\02_COLBES_OPTIMIZATION\TECHNICAL_PROPOSAL.md" "$dst\docs\"
Copy-Item "$src\DOCUMENTATION\04_PARTNERSHIP_PROJECTS\02_COLBES_OPTIMIZATION\IMPLEMENTATION_GUIDE.md" "$dst\docs\"
```text


---

## Alejandra Rojas - Arbovirus Surveillance

### To Copy:

```text

deliverables/alejandra_rojas/
├── README.md                              ← Partner-specific guide (created)
├── scripts/
│   ├── ingest_arboviruses.py              ← From scripts/ingest/ingest_arboviruses.py
│   ├── arbovirus_hyperbolic_trajectory.py ← From scripts/analysis/arbovirus_hyperbolic_trajectory.py
│   └── primer_stability_scanner.py        ← From scripts/analysis/primer_stability_scanner.py
├── notebooks/
│   └── rojas_serotype_forecast.ipynb      ← From notebooks/partners/rojas_serotype_forecast.ipynb
├── results/
│   ├── dengue_forecast.json               ← From results/partners/rojas/dengue_forecast.json
│   └── primer_candidates.csv              ← From results/partners/rojas/primer_candidates.csv
├── data/
│   └── dengue_paraguay.fasta              ← From data/raw/dengue_paraguay.fasta
└── docs/
    └── IMPLEMENTATION_GUIDE.md            ← From DOCUMENTATION/04_PARTNERSHIP_PROJECTS/03_ROJAS_ARBOVIRUSES/IMPLEMENTATION_GUIDE.md
```text


### Copy Commands (PowerShell):

```text
powershell
$src = "C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics"
$dst = "$src\deliverables\alejandra_rojas"

# Create directories
New-Item -ItemType Directory -Force -Path "$dst\scripts"
New-Item -ItemType Directory -Force -Path "$dst\notebooks"
New-Item -ItemType Directory -Force -Path "$dst\results"
New-Item -ItemType Directory -Force -Path "$dst\data"
New-Item -ItemType Directory -Force -Path "$dst\docs"

# Copy files
Copy-Item "$src\scripts\ingest\ingest_arboviruses.py" "$dst\scripts\"
Copy-Item "$src\scripts\analysis\arbovirus_hyperbolic_trajectory.py" "$dst\scripts\"
Copy-Item "$src\scripts\analysis\primer_stability_scanner.py" "$dst\scripts\"
Copy-Item "$src\notebooks\partners\rojas_serotype_forecast.ipynb" "$dst\notebooks\"
Copy-Item "$src\results\partners\rojas\dengue_forecast.json" "$dst\results\"
Copy-Item "$src\results\partners\rojas\primer_candidates.csv" "$dst\results\"
Copy-Item "$src\data\raw\dengue_paraguay.fasta" "$dst\data\"
Copy-Item "$src\DOCUMENTATION\04_PARTNERSHIP_PROJECTS\03_ROJAS_ARBOVIRUSES\IMPLEMENTATION_GUIDE.md" "$dst\docs\"
```text


---

## Quick Copy All (Single Script)

Save as `copy_deliverables.ps1`:

```text
powershell
$src = "C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics"

# Carlos Brizuela
$dst = "$src\deliverables\carlos_brizuela"
New-Item -ItemType Directory -Force -Path "$dst\scripts","$dst\notebooks","$dst\results","$dst\data","$dst\docs" | Out-Null
Copy-Item "$src\scripts\optimization\__init__.py" "$dst\scripts\"
Copy-Item "$src\scripts\optimization\latent_nsga2.py" "$dst\scripts\"
Copy-Item "$src\notebooks\partners\brizuela_amp_navigator.ipynb" "$dst\notebooks\"
Copy-Item "$src\results\partners\brizuela\pareto_peptides.csv" "$dst\results\"
Copy-Item "$src\data\processed\demo_amp_embeddings.csv" "$dst\data\"
Copy-Item "$src\DOCUMENTATION\04_PARTNERSHIP_PROJECTS\01_BRIZUELA_AMPS\TECHNICAL_BRIEF.md" "$dst\docs\" -ErrorAction SilentlyContinue
Copy-Item "$src\DOCUMENTATION\04_PARTNERSHIP_PROJECTS\01_BRIZUELA_AMPS\IMPLEMENTATION_GUIDE.md" "$dst\docs\" -ErrorAction SilentlyContinue

# José Colbes
$dst = "$src\deliverables\jose_colbes"
New-Item -ItemType Directory -Force -Path "$dst\scripts","$dst\notebooks","$dst\results","$dst\data","$dst\docs" | Out-Null
Copy-Item "$src\scripts\ingest\ingest_pdb_rotamers.py" "$dst\scripts\"
Copy-Item "$src\scripts\analysis\rotamer_stability.py" "$dst\scripts\"
Copy-Item "$src\notebooks\partners\colbes_scoring_function.ipynb" "$dst\notebooks\"
Copy-Item "$src\results\partners\colbes\rotamer_stability.json" "$dst\results\"
Copy-Item "$src\data\processed\demo_rotamers.pt" "$dst\data\"
Copy-Item "$src\DOCUMENTATION\04_PARTNERSHIP_PROJECTS\02_COLBES_OPTIMIZATION\TECHNICAL_PROPOSAL.md" "$dst\docs\" -ErrorAction SilentlyContinue
Copy-Item "$src\DOCUMENTATION\04_PARTNERSHIP_PROJECTS\02_COLBES_OPTIMIZATION\IMPLEMENTATION_GUIDE.md" "$dst\docs\" -ErrorAction SilentlyContinue

# Alejandra Rojas
$dst = "$src\deliverables\alejandra_rojas"
New-Item -ItemType Directory -Force -Path "$dst\scripts","$dst\notebooks","$dst\results","$dst\data","$dst\docs" | Out-Null
Copy-Item "$src\scripts\ingest\ingest_arboviruses.py" "$dst\scripts\"
Copy-Item "$src\scripts\analysis\arbovirus_hyperbolic_trajectory.py" "$dst\scripts\"
Copy-Item "$src\scripts\analysis\primer_stability_scanner.py" "$dst\scripts\"
Copy-Item "$src\notebooks\partners\rojas_serotype_forecast.ipynb" "$dst\notebooks\"
Copy-Item "$src\results\partners\rojas\dengue_forecast.json" "$dst\results\"
Copy-Item "$src\results\partners\rojas\primer_candidates.csv" "$dst\results\"
Copy-Item "$src\data\raw\dengue_paraguay.fasta" "$dst\data\"
Copy-Item "$src\DOCUMENTATION\04_PARTNERSHIP_PROJECTS\03_ROJAS_ARBOVIRUSES\IMPLEMENTATION_GUIDE.md" "$dst\docs\" -ErrorAction SilentlyContinue

Write-Host "Deliverables copied successfully!"
```text


---

## Creating ZIP Archives

After copying, create individual ZIP files for sending:

```text
powershell
$deliverables = "C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics\deliverables"

Compress-Archive -Path "$deliverables\carlos_brizuela" -DestinationPath "$deliverables\carlos_brizuela_AMP_optimizer.zip" -Force
Compress-Archive -Path "$deliverables\jose_colbes" -DestinationPath "$deliverables\jose_colbes_rotamer_scoring.zip" -Force
Compress-Archive -Path "$deliverables\alejandra_rojas" -DestinationPath "$deliverables\alejandra_rojas_arbovirus_surveillance.zip" -Force

Write-Host "ZIP archives created!"
```text


---

_This manifest ensures each partner receives a complete, self-contained package._

---

## HIV Research Package

### To Copy:

```text

deliverables/hiv_research_package/
├── README.md                          ← Package guide
├── scripts/
│   ├── run_complete_analysis.py       ← Main pipeline
│   ├── 02_hiv_drug_resistance.py      ← Drug resistance logic
│   ├── 07_validate_all_conjectures.py ← Validation logic
│   └── analyze_stanford_resistance.py ← Data analysis
└── docs/
    └── COMPLETE_PLATFORM_ANALYSIS.md  ← Full technical report
````

### Copy Commands (PowerShell):

```powershell
$src = "C:\Users\Alejandro\Documents\Ivan\Work\ternary-vaes-bioinformatics"
$dst = "$src\deliverables\hiv_research_package"

# Create directories
New-Item -ItemType Directory -Force -Path "$dst\scripts"
New-Item -ItemType Directory -Force -Path "$dst\docs"

# Copy files
Copy-Item "$src\src\research\bioinformatics\codon_encoder_research\hiv\scripts\run_complete_analysis.py" "$dst\scripts\"
Copy-Item "$src\src\research\bioinformatics\codon_encoder_research\hiv\scripts\02_hiv_drug_resistance.py" "$dst\scripts\"
Copy-Item "$src\src\research\bioinformatics\codon_encoder_research\hiv\scripts\07_validate_all_conjectures.py" "$dst\scripts\"
Copy-Item "$src\src\research\bioinformatics\codon_encoder_research\hiv\scripts\analyze_stanford_resistance.py" "$dst\scripts\"
Copy-Item "$src\docs\content\research\hiv\complete_platform_analysis.md" "$dst\docs\COMPLETE_PLATFORM_ANALYSIS.md"
```
