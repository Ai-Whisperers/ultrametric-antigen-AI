#!/bin/bash
# ROCm Setup Script for AMD RX 6500M
# Ternary VAE Bioinformatics Project
# Created: 2026-01-30

set -e

echo "=========================================="
echo "ROCm Setup for AMD Radeon RX 6500M"
echo "=========================================="

# Check if running as root for system packages
if [ "$EUID" -eq 0 ]; then
    echo "Please run without sudo - will prompt when needed"
    exit 1
fi

# Step 1: Check current state
echo ""
echo "[1/6] Checking current system state..."
echo "GPU detected:"
lspci | grep -i "vga\|display" | grep -i amd

echo ""
echo "Current kernel: $(uname -r)"
echo "Ubuntu version: $(lsb_release -ds)"

# Step 2: Download ROCm installer
echo ""
echo "[2/6] Downloading ROCm installer..."
if [ ! -f "/tmp/amdgpu-install.deb" ]; then
    wget -q --show-progress -O /tmp/amdgpu-install.deb \
        https://repo.radeon.com/amdgpu-install/6.4/ubuntu/noble/amdgpu-install_6.4.60400-1_all.deb
else
    echo "ROCm installer already downloaded"
fi

# Step 3: Install ROCm
echo ""
echo "[3/6] Installing ROCm (requires sudo)..."
sudo apt install -y /tmp/amdgpu-install.deb
sudo amdgpu-install --usecase=rocm,hip --no-dkms -y

# Step 4: Add user to groups
echo ""
echo "[4/6] Adding user to video/render groups..."
sudo usermod -aG video,render $USER

# Step 5: Set up environment variables
echo ""
echo "[5/6] Setting up environment variables..."
ROCM_ENV="
# ROCm Environment for RX 6500M (gfx1032)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
"

if ! grep -q "HSA_OVERRIDE_GFX_VERSION" ~/.bashrc; then
    echo "$ROCM_ENV" >> ~/.bashrc
    echo "Added ROCm environment to ~/.bashrc"
else
    echo "ROCm environment already in ~/.bashrc"
fi

# Step 6: Create Python virtual environment
echo ""
echo "[6/6] Setting up Python environment..."
cd "$(dirname "$0")/../.."  # Go to project root

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created virtual environment"
fi

source .venv/bin/activate

# Install PyTorch with ROCm
echo "Installing PyTorch with ROCm 6.2 (compatible with ROCm 6.4)..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Install project dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: You must reboot for ROCm driver changes to take effect."
echo ""
echo "After reboot, activate the environment and test:"
echo "  source .venv/bin/activate"
echo "  python -c \"import torch; print(torch.cuda.is_available())\""
echo ""
echo "To run a training test:"
echo "  python src/scripts/training/train_v5_12.py --epochs 5"
echo ""
read -p "Reboot now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo reboot
fi
