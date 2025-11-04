#!/bin/bash
#SBATCH --job-name=vlm_inference
#SBATCH --partition=student
#SBATCH --qos=studentqos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=vlm_output_%j.log
#SBATCH --error=vlm_error_%j.log

# Vision-Language Model Inference Job
# This script runs your model on the GPU cluster

echo "=========================================="
echo "Job Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

# Load required modules (check what's available with: module avail)
# Uncomment if these modules exist on your cluster:
# module load cuda/11.8
# module load python/3.10

# Create and activate virtual environment if it doesn't exist
VENV_PATH="/common/scratch/users/g/gerard.loh.2022/vlm_env"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_PATH
fi

echo "Activating virtual environment..."
source $VENV_PATH/bin/activate

# Check GPU availability
echo "Checking GPU..."
nvidia-smi
echo ""

# Install/upgrade required packages
echo "Installing/checking dependencies..."
pip install --upgrade pip
pip install torch torchvision transformers peft pillow requests accelerate

echo ""
echo "=========================================="
echo "Running Model Inference"
echo "=========================================="
echo ""

# Run your Python script
python gpu_cluster_backend.py

echo ""
echo "=========================================="
echo "Job Completed: $(date)"
echo "=========================================="