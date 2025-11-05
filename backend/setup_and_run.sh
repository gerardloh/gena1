#!/bin/bash
# Simple setup and run script for local testing (not SLURM)

echo "=================================================="
echo "Qwen2.5-VL LoRA Model Test - Setup and Run"
echo "=================================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "Python version:"
python3 --version
echo ""

# Check if pip is available
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "Error: pip is not installed"
    exit 1
fi

# Install requirements
echo "Installing required packages..."
echo "This may take a few minutes..."
pip3 install --break-system-packages -r requirements.txt

echo ""
echo "=================================================="
echo "Checking GPU availability..."
echo "=================================================="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo ""
echo "=================================================="
echo "Running model test..."
echo "=================================================="
echo ""

# Create logs directory
mkdir -p logs

# Run the script
python3 load_and_test_model.py 2>&1 | tee logs/run_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=================================================="
echo "Test completed!"
echo "Check the output above for results."
echo "=================================================="