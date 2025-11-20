#!/bin/bash
# Start host services using Docker Compose
# This runs vLLM and embedding services on the host with GPU access

set -e

GREEN='\033[0;32m'
NC='\033[0m'

echo_info() {
    echo -e "${GREEN}[Aeon::Host]${NC} $1"
}

echo_info "=== Starting Aeon Host Services ==="
echo ""

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU may not be available."
    echo "Press Ctrl+C to cancel, or wait 5 seconds to continue anyway..."
    sleep 5
else
    echo_info "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Start embedding service with Docker Compose
cd ../inference

echo_info "Starting embedding service..."
docker-compose up -d embeddings

echo_info "Embedding service started on http://localhost:8001"
echo ""

echo_info "To start vLLM server (run in separate terminal):"
echo "  cd inference"
echo "  ./start_vllm.sh"
echo ""

echo_info "Or install dependencies and run directly:"
echo "  uv pip install -r requirements.txt"
echo "  ./start_vllm.sh"
echo ""

echo_info "Check service status:"
echo "  docker-compose ps"
echo "  docker-compose logs -f embeddings"
