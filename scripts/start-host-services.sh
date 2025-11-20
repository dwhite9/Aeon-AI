#!/bin/bash
# Start host services using Podman Compose or Docker Compose
# This runs vLLM and embedding services on the host with GPU access

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo_info() {
    echo -e "${GREEN}[Aeon::Host]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[Aeon::Host]${NC} $1"
}

echo_error() {
    echo -e "${RED}[Aeon::Host]${NC} $1"
}

echo_info "=== Starting Aeon Host Services ==="
echo ""

# Set TMPDIR to use /zdata for builds (more space)
export TMPDIR=/zdata/tmp
mkdir -p "$TMPDIR"

# Detect container runtime and compose tool
if command -v podman-compose &> /dev/null; then
    COMPOSE_CMD="podman-compose"
    RUNTIME="Podman"
    echo_info "Using Podman Compose"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    RUNTIME="Docker"
    echo_warn "Using Docker Compose. Consider using Podman for better isolation."
elif command -v podman &> /dev/null && command -v docker &> /dev/null; then
    # Podman is installed but podman-compose is not
    echo_error "Podman is installed but podman-compose not found."
    echo_error "Install it with: scripts/install-podman.sh"
    exit 1
else
    echo_error "Neither podman-compose nor docker-compose found."
    echo_error "Install Podman with: scripts/install-podman.sh"
    exit 1
fi

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo_warn "WARNING: nvidia-smi not found. GPU may not be available."
    echo_warn "Press Ctrl+C to cancel, or wait 5 seconds to continue anyway..."
    sleep 5
else
    echo_info "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Start embedding service with compose
cd ../inference

echo_info "Starting embedding service with $RUNTIME..."
$COMPOSE_CMD up -d embeddings

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
echo "  $COMPOSE_CMD ps"
echo "  $COMPOSE_CMD logs -f embeddings"
