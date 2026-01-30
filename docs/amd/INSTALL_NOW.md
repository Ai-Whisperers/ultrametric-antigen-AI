# ROCm Installation Commands - Ready to Execute

**Run these commands in order. The installer has been downloaded to /tmp/amdgpu-install.deb**

## Step 1: Install amdgpu-install package
```bash
sudo apt install -y /tmp/amdgpu-install.deb
```

## Step 2: Install ROCm with HIP (this takes several minutes)
```bash
sudo amdgpu-install --usecase=rocm,hip --no-dkms -y
```

## Step 3: Add user to required groups
```bash
sudo usermod -aG video,render $USER
```

## Step 4: Set up environment variables
```bash
cat >> ~/.bashrc << 'ROCM_ENV'

# ROCm Environment for RX 6500M (gfx1032)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:128
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
ROCM_ENV
```

## Step 5: Source the new environment (for current session)
```bash
source ~/.bashrc
```

## Step 6: Reinstall PyTorch with ROCm support
```bash
cd /home/ai-whisperers/Desktop/dev/ternary-vaes-bioinformatics
source .venv/bin/activate
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

## Step 7: REBOOT (do this last!)
```bash
sudo reboot
```

---

## After Reboot - Verification

```bash
# Verify ROCm installation
rocm-smi --showproductname

# Verify PyTorch sees the GPU
cd /home/ai-whisperers/Desktop/dev/ternary-vaes-bioinformatics
source .venv/bin/activate
python -c "import torch; print(f'ROCm available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Run training test
python src/scripts/training/train_v5_12.py --config src/configs/v5_12_4.yaml --epochs 2
```
