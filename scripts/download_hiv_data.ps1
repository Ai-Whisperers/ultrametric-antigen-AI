# HIV Datasets Download Script (PowerShell)
# Run with: powershell -ExecutionPolicy Bypass -File scripts/download_hiv_data.ps1

$ErrorActionPreference = "Continue"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data\external"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "HIV DATASET DOWNLOADER" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Output directory: $DataDir"

# Create directories
$dirs = @("github", "kaggle", "huggingface", "zenodo", "csv", "lanl")
foreach ($dir in $dirs) {
    $path = Join-Path $DataDir $dir
    if (-not (Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

# =============================================================================
# 1. GitHub Repositories
# =============================================================================
Write-Host "`n[1/5] DOWNLOADING GITHUB REPOSITORIES" -ForegroundColor Yellow

$githubDir = Join-Path $DataDir "github"
$repos = @(
    @{Name="HIV-data"; Url="https://github.com/malabz/HIV-data.git"; Desc="HIV sequences by length"},
    @{Name="HIV-DRM-machine-learning"; Url="https://github.com/lucblassel/HIV-DRM-machine-learning.git"; Desc="Drug resistance ML data"},
    @{Name="HIV-1_Paper"; Url="https://github.com/pauloluniyi/HIV-1_Paper.git"; Desc="Nigeria drug resistance"}
)

foreach ($repo in $repos) {
    $repoPath = Join-Path $githubDir $repo.Name
    if (Test-Path $repoPath) {
        Write-Host "  [SKIP] $($repo.Name) already exists" -ForegroundColor Gray
    } else {
        Write-Host "  Cloning: $($repo.Name) - $($repo.Desc)"
        try {
            git clone --depth 1 $repo.Url $repoPath 2>&1 | Out-Null
            Write-Host "    -> Success" -ForegroundColor Green
        } catch {
            Write-Host "    -> Failed: $_" -ForegroundColor Red
        }
    }
}

# =============================================================================
# 2. Direct CSV Downloads
# =============================================================================
Write-Host "`n[2/5] DOWNLOADING CSV FILES" -ForegroundColor Yellow

$csvDir = Join-Path $DataDir "csv"
$csvFiles = @(
    @{Name="corgis_aids.csv"; Url="https://corgis-edu.github.io/corgis/datasets/csv/aids/aids.csv"; Desc="UNAIDS statistics"}
)

foreach ($csv in $csvFiles) {
    $csvPath = Join-Path $csvDir $csv.Name
    if (Test-Path $csvPath) {
        Write-Host "  [SKIP] $($csv.Name) already exists" -ForegroundColor Gray
    } else {
        Write-Host "  Downloading: $($csv.Name) - $($csv.Desc)"
        try {
            Invoke-WebRequest -Uri $csv.Url -OutFile $csvPath -UseBasicParsing
            $size = (Get-Item $csvPath).Length / 1KB
            Write-Host "    -> Success ($([math]::Round($size, 1)) KB)" -ForegroundColor Green
        } catch {
            Write-Host "    -> Failed: $_" -ForegroundColor Red
        }
    }
}

# =============================================================================
# 3. Zenodo Datasets
# =============================================================================
Write-Host "`n[3/5] DOWNLOADING ZENODO DATASETS" -ForegroundColor Yellow

$zenodoDir = Join-Path $DataDir "zenodo"

# Zenodo record 6475667 - CView gp120 sequences
$cviewDir = Join-Path $zenodoDir "cview_gp120"
if (-not (Test-Path $cviewDir)) {
    New-Item -ItemType Directory -Path $cviewDir -Force | Out-Null
}

Write-Host "  Fetching Zenodo record 6475667 metadata..."
try {
    $response = Invoke-RestMethod -Uri "https://zenodo.org/api/records/6475667" -UseBasicParsing
    foreach ($file in $response.files) {
        $filePath = Join-Path $cviewDir $file.key
        if (Test-Path $filePath) {
            Write-Host "    [SKIP] $($file.key)" -ForegroundColor Gray
        } else {
            $sizeMB = $file.size / 1MB
            if ($sizeMB -lt 50) {
                Write-Host "    Downloading: $($file.key) ($([math]::Round($sizeMB, 2)) MB)"
                Invoke-WebRequest -Uri $file.links.self -OutFile $filePath -UseBasicParsing
                Write-Host "      -> Success" -ForegroundColor Green
            } else {
                Write-Host "    [SKIP] $($file.key) - too large ($([math]::Round($sizeMB, 2)) MB)" -ForegroundColor Gray
            }
        }
    }
} catch {
    Write-Host "    -> Failed: $_" -ForegroundColor Red
}

# =============================================================================
# 4. Hugging Face Datasets
# =============================================================================
Write-Host "`n[4/5] DOWNLOADING HUGGING FACE DATASETS" -ForegroundColor Yellow

$hfDir = Join-Path $DataDir "huggingface"
$hfDatasets = @(
    @{Name="human_hiv_ppi"; Repo="damlab/human_hiv_ppi"},
    @{Name="HIV_V3_coreceptor"; Repo="damlab/HIV_V3_coreceptor"},
    @{Name="Protease_Hiv_drug"; Repo="rebe121314/Protease_Hiv_drug"}
)

# Try using huggingface_hub if available
$pythonScript = @"
import sys
try:
    from huggingface_hub import snapshot_download
    import os

    datasets = [
        ('damlab/human_hiv_ppi', 'human_hiv_ppi'),
        ('damlab/HIV_V3_coreceptor', 'HIV_V3_coreceptor'),
        ('rebe121314/Protease_Hiv_drug', 'Protease_Hiv_drug'),
    ]

    base_dir = sys.argv[1]
    for repo, name in datasets:
        local_dir = os.path.join(base_dir, name)
        if os.path.exists(local_dir) and os.listdir(local_dir):
            print(f'  [SKIP] {name} already exists')
            continue
        print(f'  Downloading: {name}')
        try:
            snapshot_download(repo, repo_type='dataset', local_dir=local_dir)
            print(f'    -> Success')
        except Exception as e:
            print(f'    -> Failed: {e}')
except ImportError:
    print('  huggingface_hub not installed. Run: pip install huggingface_hub')
    print('  Manual URLs:')
    print('    - https://huggingface.co/datasets/damlab/human_hiv_ppi')
    print('    - https://huggingface.co/datasets/damlab/HIV_V3_coreceptor')
    print('    - https://huggingface.co/datasets/rebe121314/Protease_Hiv_drug')
"@

$pythonScript | python - $hfDir

# =============================================================================
# 5. Kaggle Datasets
# =============================================================================
Write-Host "`n[5/5] DOWNLOADING KAGGLE DATASETS" -ForegroundColor Yellow

$kaggleDir = Join-Path $DataDir "kaggle"

# Check if kaggle is available
$kaggleAvailable = $false
try {
    kaggle --version 2>&1 | Out-Null
    $kaggleAvailable = $true
} catch {
    $kaggleAvailable = $false
}

if ($kaggleAvailable) {
    $kaggleDatasets = @(
        @{Slug="protobioengineering/hiv-1-and-hiv-2-rna-sequences"; Name="hiv_sequences"},
        @{Slug="imdevskp/hiv-aids-dataset"; Name="hiv_aids_stats"}
    )

    foreach ($ds in $kaggleDatasets) {
        $dsPath = Join-Path $kaggleDir $ds.Name
        if ((Test-Path $dsPath) -and (Get-ChildItem $dsPath).Count -gt 0) {
            Write-Host "  [SKIP] $($ds.Name) already exists" -ForegroundColor Gray
        } else {
            Write-Host "  Downloading: $($ds.Name)"
            New-Item -ItemType Directory -Path $dsPath -Force | Out-Null
            try {
                kaggle datasets download -d $ds.Slug -p $dsPath --unzip 2>&1 | Out-Null
                Write-Host "    -> Success" -ForegroundColor Green
            } catch {
                Write-Host "    -> Failed: $_" -ForegroundColor Red
            }
        }
    }
} else {
    Write-Host "  [WARN] Kaggle CLI not available" -ForegroundColor Yellow
    Write-Host "  Install: pip install kaggle" -ForegroundColor Yellow
    Write-Host "  Configure: https://www.kaggle.com/docs/api" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Manual download URLs:" -ForegroundColor Yellow
    Write-Host "    - https://www.kaggle.com/datasets/protobioengineering/hiv-1-and-hiv-2-rna-sequences"
    Write-Host "    - https://www.kaggle.com/datasets/imdevskp/hiv-aids-dataset"
}

# =============================================================================
# Summary
# =============================================================================
Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "DOWNLOAD COMPLETE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

Write-Host "`nDownloaded datasets:"
Get-ChildItem $DataDir -Directory | ForEach-Object {
    $subItems = Get-ChildItem $_.FullName -ErrorAction SilentlyContinue
    if ($subItems.Count -gt 0) {
        Write-Host "  [$($_.Name)]" -ForegroundColor Green
        foreach ($item in $subItems) {
            if ($item.PSIsContainer) {
                $fileCount = (Get-ChildItem $item.FullName -Recurse -File).Count
                Write-Host "    - $($item.Name): $fileCount files"
            } else {
                $sizeMB = $item.Length / 1MB
                Write-Host "    - $($item.Name): $([math]::Round($sizeMB, 2)) MB"
            }
        }
    }
}

Write-Host "`nFor manual downloads, see: data/external/HIV_DATASETS_DOWNLOAD_GUIDE.md"
