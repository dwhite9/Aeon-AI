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

# Load environment variables from .env if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then
    echo_info "Loading environment from .env"
    source "$REPO_ROOT/.env"
fi

echo_info "=== Starting Aeon Host Services ==="
echo ""

# Set TMPDIR from environment or use reasonable default
# Prefer .env setting, fallback to /tmp (system default)
export TMPDIR="${TMPDIR:-/tmp}"
mkdir -p "$TMPDIR"
echo_info "Using TMPDIR: $TMPDIR"

# Configure Buildah/Podman to use TMPDIR for build operations
# This prevents "no space left on device" errors during image builds
export BUILDAH_TMPDIR="$TMPDIR"
export BUILDAH_ISOLATION=rootless

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

# Navigate to inference directory (relative to repo root)
cd "$REPO_ROOT/inference"

# Build embedding service
echo_info "Building embedding service with $RUNTIME..."
if [ "$RUNTIME" = "Podman" ]; then
    # Build with Podman - BUILDAH_TMPDIR already set above for build storage
    # The Dockerfile sets ENV TMPDIR=/var/tmp for the build process
    podman build \
        -t aeon-embeddings:latest \
        -f Dockerfile.embeddings \
        .

    if [ $? -ne 0 ]; then
        echo_error "Build failed. Check the error messages above."
        exit 1
    fi

    echo_info "Starting embedding service..."
    $COMPOSE_CMD up -d embeddings
else
    # Use docker-compose build as normal
    $COMPOSE_CMD up -d --build embeddings
fi

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
