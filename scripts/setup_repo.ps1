<#
.SYNOPSIS
    Sets up the repository environment, ensuring Git LFS is installed and submodules are updated.
    Run this script after cloning the repository.

.DESCRIPTION
    This script performs the following:
    1. Checks if Git LFS is available.
    2. If not, automatically downloads a portable version to a local 'bin/' directory.
    3. Temporarily adds Git LFS to the session PATH (so it works immediately).
    4. Initializes Git LFS.
    5. Updates and initializes all git submodules (fetching the external data).
    
    This ensures that anyone on the team can get up and running without needing admin rights
    or manual software installation sequences.
#>

$ErrorActionPreference = "Stop"
$originalLocation = Get-Location

try {
    # Ensure we are in the repo root
    $scriptPath = $MyInvocation.MyCommand.Path
    $repoRoot = Split-Path (Split-Path $scriptPath)
    Set-Location $repoRoot

    Write-Host "🚀 Starting Repository Setup..." -ForegroundColor Cyan

    # --- Step 1: Git LFS Check & Install ---
    if (Get-Command "git-lfs" -ErrorAction SilentlyContinue) {
        Write-Host "✅ Git LFS is detected in your system PATH." -ForegroundColor Green
    }
    else {
        Write-Host "⚠️ Git LFS not found globally. Checking local bin..." -ForegroundColor Yellow
        $binDir = Join-Path $repoRoot "bin"
        $lfsVersion = "v3.4.1"
        $lfsDir = Join-Path $binDir "git-lfs-$lfsVersion"
        $lfsExe = Join-Path $lfsDir "git-lfs.exe"
        $zipUrl = "https://github.com/git-lfs/git-lfs/releases/download/$lfsVersion/git-lfs-windows-amd64-$lfsVersion.zip"
        $zipPath = Join-Path $binDir "git-lfs.zip"

        if (-not (Test-Path $lfsExe)) {
            Write-Host "   Downloading Git LFS ($lfsVersion)..."
            New-Item -ItemType Directory -Force -Path $binDir | Out-Null
            
            # Download
            Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath
            
            # Extract
            Write-Host "   Extracting..."
            Expand-Archive -Path $zipPath -DestinationPath $lfsDir -Force
            
            # Cleanup Zip
            Remove-Item $zipPath -Force
            Write-Host "✅ Portable Git LFS installed to $lfsDir" -ForegroundColor Green
        } else {
             Write-Host "✅ Portable Git LFS found in $lfsDir" -ForegroundColor Green
        }

        # Add to Session PATH
        $env:PATH = "$lfsDir;$env:PATH"
        Write-Host "   Temporarily added to session PATH." -ForegroundColor Gray
        
        # Optional: Offer to add to User PATH permanently
        $addToPath = Read-Host "   Do you want to add this to your User PATH permanently (Recommended)? (y/n)"
        if ($addToPath -eq 'y' -or $addToPath -eq 'Y') {
            $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
            if (-not ($userPath -split ';').Contains($lfsDir)) {
                [Environment]::SetEnvironmentVariable("Path", $userPath + ";" + $lfsDir, "User")
                Write-Host "   ✅ Added to persistent User PATH." -ForegroundColor Green
            } else {
                 Write-Host "   info: Already in User PATH." -ForegroundColor Gray
            }
        }
    }

    # --- Step 2: Initialize LFS ---
    Write-Host "`n📦 Initializing Git LFS..."
    git lfs install

    # --- Step 3: Update Submodules ---
    Write-Host "`n🔗 Updating Git Submodules (Downloading External Data)..."
    git submodule update --init --recursive
    
    if ($LASTEXITCODE -eq 0) {
         Write-Host "`n✨ Success! Repository is fully set up." -ForegroundColor Cyan
         Write-Host "You can now run your analysis scripts." -ForegroundColor Green
    } else {
        Write-Host "`n❌ Error updating submodules." -ForegroundColor Red
    }

}
catch {
    Write-Host "`n❌ Setup Failed: $_" -ForegroundColor Red
}
finally {
    Set-Location $originalLocation
}
