#!/bin/bash
# Quick fix for CUDA kernel error - installs correct PyTorch version

echo "=================================================="
echo "PyTorch CUDA Compatibility Fix"
echo "=================================================="
echo ""

# Load Python module
module load Python/3.12.8-GCCcore-13.3.0

# Check CUDA version
echo "Checking CUDA version on system:"
nvidia-smi | grep "CUDA Version"
echo ""

# Uninstall existing PyTorch
echo "Removing existing PyTorch installation..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
echo ""

# Install PyTorch with CUDA 12.1 (most compatible)
echo "Installing PyTorch with CUDA 12.1 support..."
pip install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo ""

# Verify installation
echo "Verifying PyTorch installation..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo ""

echo "=================================================="
echo "Fix complete! You can now run the model script."
echo "=================================================="