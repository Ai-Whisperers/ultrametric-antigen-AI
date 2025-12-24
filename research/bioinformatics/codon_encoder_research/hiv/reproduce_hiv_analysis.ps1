# HIV Analysis Reproduction Script
# Automates the execution of HIV CTL Escape and Glycan Sentinel analysis.

$ErrorActionPreference = "Stop"

function Assert-Dependency {
    param($Name)
    try {
        python -c "import $Name"
    }
    catch {
        Write-Error "Missing Python dependency: $Name. Please install it via pip."
        exit 1
    }
}

Write-Host "=================================================="
Write-Host "HIV Analysis Reproduction Pipeline"
Write-Host "=================================================="

# 1. Check Environment
Write-Host "[1/3] Checking Environment..."
Assert-Dependency "torch"
Assert-Dependency "numpy"
Write-Host "Dependencies OK."

# 2. Run CTL Escape Analysis
Write-Host "`n[2/3] Running CTL Escape Analysis..."
# Note: Data for CTL epitopes is currently embedded in the script.
$CtlScript = Join-Path $PSScriptRoot "src\01_hiv_escape_analysis.py"
python $CtlScript
if ($LASTEXITCODE -ne 0) { throw "CTL Analysis Failed" }

# 3. Run Drug Resistance Analysis
Write-Host "`n[3/3] Running Drug Resistance Analysis..."
$DrugScript = Join-Path $PSScriptRoot "src\02_hiv_drug_resistance.py"
python $DrugScript
if ($LASTEXITCODE -ne 0) { throw "Drug Resistance Analysis Failed" }

# 4. Run Glycan Sentinel Analysis
Write-Host "`n[4/4] Running Glycan Sentinel Analysis..."
# Note: BG505 sequence is embedded in the script.
$GlycanScript = Join-Path $PSScriptRoot "src\glycan_shield\01_glycan_sentinel_analysis.py"
python $GlycanScript
if ($LASTEXITCODE -ne 0) { throw "Glycan Analysis Failed" }

Write-Host "`n=================================================="
Write-Host "Success! Analysis complete."
Write-Host "Results generated in: data/metrics/"
Write-Host "=================================================="
