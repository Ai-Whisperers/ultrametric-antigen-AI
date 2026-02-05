#!/bin/bash
# ROCm Installation Script - Execute All Steps
# Run this script directly: bash docs/amd/install_rocm_now.sh

set -e

echo "=============================================="
echo "ROCm Installation for AMD RX 6500M"
echo "=============================================="
echo ""
echo "This script will:"
echo "  1. Install amdgpu-install package"
echo "  2. Install ROCm + HIP drivers"
echo "  3. Add user to video/render groups"
echo "  4. Configure environment variables"
echo "  5. Reinstall PyTorch with ROCm support"
echo ""
echo "You will be prompted for your sudo password."
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Step 1
echo ""
echo "[1/5] Installing amdgpu-install package..."
sudo apt install -y /tmp/amdgpu-install.deb

# Step 2
echo ""
echo "[2/5] Installing ROCm + HIP (this takes several minutes)..."
sudo amdgpu-install --usecase=rocm,hip --no-dkms -y

# Step 3
echo ""
echo "[3/5] Adding user to video/render groups..."
sudo usermod -aG video,render $USER

# Step 4
echo ""
echo "[4/5] Configuring environment variables..."
if ! grep -q "HSA_OVERRIDE_GFX_VERSION" ~/.bashrc; then
    cat >> ~/.bashrc << 'ROCM_ENV'

# ROCm Environment for RX 6500M (gfx1032)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
ROCM_ENV
    echo "  Added ROCm environment to ~/.bashrc"
else
    echo "  ROCm environment already configured"
fi

# Source for current session
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

# Step 5
echo ""
echo "[5/5] Reinstalling PyTorch with ROCm support..."
cd /home/ai-whisperers/Desktop/dev/ternary-vaes-bioinformatics
source .venv/bin/activate
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

echo ""
echo "=============================================="
echo "Installation Complete!"
echo "=============================================="
echo ""
echo "IMPORTANT: You must REBOOT for the GPU drivers to activate."
echo ""
echo "After reboot, run:"
echo "  cd /home/ai-whisperers/Desktop/dev/ternary-vaes-bioinformatics"
echo "  source .venv/bin/activate"
echo "  python -c \"import torch; print(torch.cuda.is_available())\""
echo ""
read -p "Reboot now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo reboot
else
    echo "Remember to reboot before using the GPU!"
fi
