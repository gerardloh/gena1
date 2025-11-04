#!/bin/bash
# Setup script to install compatible versions of all packages

echo "=========================================="
echo "Setting up Python environment for VLM"
echo "=========================================="

# Remove existing installations
echo "Cleaning up existing packages..."
pip uninstall -y torch torchvision transformers peft numpy

# Install compatible versions
echo ""
echo "Installing NumPy 1.x..."
pip install "numpy<2.0"

echo ""
echo "Installing PyTorch 2.1.0 with CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Installing transformers and other dependencies..."
pip install transformers peft pillow requests accelerate

echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python3 << 'PYEOF'
import sys
print("\nPython version:", sys.version)

try:
    import numpy
    print("✓ NumPy:", numpy.__version__)
except:
    print("✗ NumPy failed")

try:
    import torch
    print("✓ PyTorch:", torch.__version__)
    print("  CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  CUDA version:", torch.version.cuda)
        print("  GPU:", torch.cuda.get_device_name(0))
except:
    print("✗ PyTorch failed")

try:
    import transformers
    print("✓ Transformers:", transformers.__version__)
except:
    print("✗ Transformers failed")

try:
    import peft
    print("✓ PEFT:", peft.__version__)
except:
    print("✗ PEFT failed")

try:
    from PIL import Image
    print("✓ Pillow: OK")
except:
    print("✗ Pillow failed")
PYEOF

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "You can now run: python gpu_cluster_backend.py"