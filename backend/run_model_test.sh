#!/bin/bash
#SBATCH --job-name=qwen_lora_test
#SBATCH --account=is469           # Your account
#SBATCH --partition=student       # Student partition
#SBATCH --qos=studentqos          # Student QOS
#SBATCH --gres=gpu:1              # Request 1 GPU (max allowed)
#SBATCH --cpus-per-task=4         # Number of CPU cores (max allowed)
#SBATCH --mem=32G                 # Memory (max allowed is 32GB)
#SBATCH --time=23:59:00           # Time limit (max is 1 day)
#SBATCH --output=logs/job_%j.out  # Standard output log
#SBATCH --error=logs/job_%j.err   # Standard error log

# Print job information
echo "=================================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"
echo "QOS: $SLURM_JOB_QOS"
echo "=================================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Load Python module
echo "Loading Python module..."
module load Python/3.12.8-GCCcore-13.3.0
echo "Python version:"
python3 --version
echo ""

# Show GPU and CUDA information
echo "GPU Information:"
nvidia-smi
echo ""
echo "CUDA Version on system:"
nvcc --version || echo "nvcc not in PATH"
echo ""

# Check current PyTorch CUDA version
echo "Checking current PyTorch installation..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "PyTorch not installed yet"
echo ""

# Uninstall existing torch to avoid conflicts
echo "Removing existing PyTorch installation..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
echo ""

# Install PyTorch with CUDA 12.1 (compatible with most recent GPUs)
echo "Installing PyTorch with CUDA 12.1 support..."
pip install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo ""

# Install other required packages
echo "Installing other required packages..."
pip install --break-system-packages transformers accelerate peft bitsandbytes pillow safetensors

echo ""
echo "=================================================="
echo "Running model loading and testing script..."
echo "=================================================="
echo ""

# Run the Python script
python3 load_and_test_model.py

# Print job completion info
echo ""
echo "=================================================="
echo "Job completed at: $(date)"
echo "=================================================="