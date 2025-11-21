#!/bin/bash
# Start vLLM server with Mistral 7B Instruct (8-bit quantized)
# This script runs vLLM on the host machine with GPU access

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo_info() {
    echo -e "${GREEN}[Aeon::vLLM]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[Aeon::vLLM]${NC} $1"
}

echo_error() {
    echo -e "${RED}[Aeon::vLLM]${NC} $1"
}

# Load environment variables from .env if it exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then
    echo_info "Loading environment from .env"
    source "$REPO_ROOT/.env"
fi

# Trap handler for cleanup on exit
cleanup() {
    echo ""
    echo_warn "Caught signal, shutting down vLLM..."
    if [ -n "$VLLM_PID" ]; then
        kill -15 "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    echo_info "vLLM stopped"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

echo_info "=== Starting vLLM Inference Server ==="
echo ""

# Check for existing vLLM processes
EXISTING=$(pgrep -f "vllm serve" || true)
if [ -n "$EXISTING" ]; then
    echo_error "vLLM is already running (PID: $EXISTING)"
    echo_warn "Stop it first with: ./stop_vllm.sh"
    echo_warn "Or force kill with: pkill -9 -f 'vllm serve'"
    exit 1
fi

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo_warn "nvidia-smi not found. GPU may not be available."
fi

# Display GPU info
echo_info "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader || echo "Unable to query GPU"
echo ""

# Check GPU memory usage
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}' || echo "0")
if [ "$GPU_USED" -gt 1000 ]; then
    echo_warn "GPU memory in use: ${GPU_USED}MB"
    echo_warn "This may cause OOM errors. Consider freeing GPU memory first."
    echo ""
fi

# Use environment variables with fallback defaults
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.6}"
MAX_LEN="${MAX_MODEL_LENGTH:-8192}"

echo_info "Configuration:"
echo "  GPU Memory Utilization: $GPU_MEM_UTIL"
echo "  Max Model Length: $MAX_LEN"
echo ""

# Start vLLM with 8-bit quantization
echo_info "Starting vLLM server..."
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype float16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code &

# Capture PID for cleanup handler
VLLM_PID=$!

echo ""
echo_info "vLLM server started (PID: $VLLM_PID)"
echo_info "Server available at: http://0.0.0.0:8000"
echo_info "Press Ctrl+C to stop"
echo ""

# Wait for vLLM process
wait "$VLLM_PID"
