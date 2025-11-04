#!/bin/bash
# One-shot clean install of all dependencies

echo "=========================================="
echo "Clean install - removing everything first"
echo "=========================================="

pip3 uninstall -y torch torchvision transformers peft numpy accelerate pillow requests sentencepiece

echo ""
echo "=========================================="
echo "Installing everything fresh"
echo "=========================================="

# Install in one go with compatible versions
pip3 install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers==4.45.0 peft==0.13.0 accelerate pillow requests sentencepiece "numpy<2.0"

echo ""
echo "=========================================="
echo "Testing..."
echo "=========================================="

python3 -c "
import torch
print('✓ PyTorch:', torch.__version__)
print('✓ CUDA:', torch.cuda.is_available())
from transformers import AutoModelForVision2Seq
from peft import PeftModel
print('✓ All imports successful!')
"

echo ""
echo "Done! Run: python3 gpu_cluster_backend.py"