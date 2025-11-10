#!/bin/bash
#SBATCH --job-name=chatbot_api
#SBATCH --account=is469
#SBATCH --partition=student
#SBATCH --qos=studentqos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:59:00
#SBATCH --output=logs/api_%j.out
#SBATCH --error=logs/api_%j.err

echo "=================================================="
echo "Flask API Server Starting"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "=================================================="
echo ""

# Create logs directory
mkdir -p logs

# Load Python module
module load Python/3.12.8-GCCcore-13.3.0

# Show GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Install required packages if not already installed
echo "Installing required packages..."
pip install --break-system-packages flask flask-cors transformers accelerate peft bitsandbytes pillow safetensors chromadb sentence-transformers scikit-learn webdataset huggingface-hub datasets numpy

# Install correct PyTorch version
echo "Installing PyTorch with CUDA support..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install --break-system-packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


DB_PATH="./chroma_fashion_db_hybrid"
IMAGE_STORE="fashion_image_store_hybrid.pkl"

echo "🔍 Checking RAG database and image store..."

if [ -d "$DB_PATH" ] && [ -f "$IMAGE_STORE" ]; then
    echo "✅ RAG database already exists. Skipping build."
else
    echo "⚙️  RAG database not found. Building now..."
    python3 build_fashion_rag.py
fi

# Check if port 5000 is already in use
echo ""
echo "🔍 Checking if port 5000 is available..."
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Port 5000 is in use. Killing existing process..."
    kill -9 $(lsof -t -i:5000) 2>/dev/null || true
    sleep 2
fi
echo "✅ Port 5000 is available"

echo ""
echo "=================================================="
echo "Starting Flask API server on port 5000..."
echo "=================================================="
echo ""

# Get the node's IP address
NODE_IP=$(hostname -I | awk '{print $1}')
echo "Server will be accessible at: http://$NODE_IP:5000"
echo "API endpoint: http://$NODE_IP:5000/chat"
echo ""

# Run the Flask server
python3 backend_api.py

echo ""
echo "=================================================="
echo "Server stopped at: $(date)"
echo "=================================================="