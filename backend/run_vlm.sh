#!/bin/bash
#SBATCH --job-name=vlm_inference
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
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

# Load required modules (adjust based on your cluster)
# module load cuda/11.8
# module load python/3.10

# Activate your virtual environment
# Uncomment and adjust path as needed:
# source /path/to/your/venv/bin/activate

# Check GPU availability
echo "Checking GPU..."
nvidia-smi
echo ""

# Install required packages (if not already in your environment)
echo "Installing/checking dependencies..."
pip install -q torch transformers peft pillow requests accelerate

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