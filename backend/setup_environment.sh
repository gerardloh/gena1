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
echo "Installing PyTorch 2.1.2 with CUDA 11.8..."
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Installing transformers 4.37.2 (compatible with PyTorch 2.1.2)..."
pip install transformers==4.37.2

echo ""
echo "Installing other dependencies..."
pip install peft pillow requests accelerate sentencepiece

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
except Exception as e:
    print("✗ NumPy failed:", e)

try:
    import torch
    print("✓ PyTorch:", torch.__version__)
    print("  CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  CUDA version:", torch.version.cuda)
        print("  GPU:", torch.cuda.get_device_name(0))
except Exception as e:
    print("✗ PyTorch failed:", e)

try:
    import transformers
    print("✓ Transformers:", transformers.__version__)
except Exception as e:
    print("✗ Transformers failed:", e)

try:
    import peft
    print("✓ PEFT:", peft.__version__)
except Exception as e:
    print("✗ PEFT failed:", e)

try:
    from PIL import Image
    print("✓ Pillow: OK")
except Exception as e:
    print("✗ Pillow failed:", e)

print("\n========================================")
print("Testing imports from script...")
print("========================================")

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    print("✓ AutoModelForVision2Seq: OK")
except Exception as e:
    print("✗ AutoModelForVision2Seq failed:", e)
PYEOF

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo "You can now run: python gpu_cluster_backend.py"