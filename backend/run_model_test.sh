#!/bin/bash
#SBATCH --job-name=qwen_lora_test
#SBATCH --partition=gpu           # GPU partition
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --cpus-per-task=4         # Number of CPU cores
#SBATCH --mem=32G                 # Memory (32GB should be sufficient for 4-bit model)
#SBATCH --time=01:00:00           # Time limit (1 hour)
#SBATCH --output=logs/job_%j.out  # Standard output log
#SBATCH --error=logs/job_%j.err   # Standard error log

# Print job information
echo "=================================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=================================================="
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required modules (adjust based on your cluster's available modules)
# Uncomment and modify as needed:
# module load python/3.10
# module load cuda/12.1
# module load cudnn/8.9

# Show GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# Activate your Python environment (if using virtual environment or conda)
# Uncomment and modify as needed:
# source ~/venv/bin/activate
# conda activate myenv

# Install required packages if not already installed
echo "Installing/Checking required packages..."
pip install --break-system-packages transformers accelerate peft bitsandbytes torch pillow

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