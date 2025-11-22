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
MODEL="${MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}"
QUANT="${QUANTIZATION:-bitsandbytes}"
GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.6}"
MAX_LEN="${MAX_MODEL_LENGTH:-8192}"
HF_TOKEN="${HF_TOKEN:-}"

echo_info "Configuration:"
echo "  Model: $MODEL"
echo "  Quantization: $QUANT"
echo "  GPU Memory Utilization: $GPU_MEM_UTIL"
echo "  Max Model Length: $MAX_LEN"
if [ -n "$HF_TOKEN" ]; then
  echo "  HuggingFace Token: ****${HF_TOKEN: -4}"
else
  echo "  HuggingFace Token: Not set"
fi
echo ""

# Build vLLM command based on quantization type
VLLM_ARGS=(
  "$MODEL"
  --quantization "$QUANT"
  --dtype float16
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --host 0.0.0.0
  --port 8000
  --trust-remote-code
)

# Add load-format for bitsandbytes quantization
if [ "$QUANT" = "bitsandbytes" ]; then
  VLLM_ARGS+=(--load-format bitsandbytes)
fi

# Start vLLM in tmux session with logging
SESSION_NAME="aeon-vllm"
LOG_FILE="$SCRIPT_DIR/vllm.log"

echo_info "Starting vLLM server in tmux session..."

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo_error "tmux session '$SESSION_NAME' already exists!"
    echo_warn "Attach to it with: tmux attach -t $SESSION_NAME"
    echo_warn "Or kill it with: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create tmux session and run vLLM with logging
# Export HF_TOKEN if set (for private/gated HuggingFace models)
if [ -n "$HF_TOKEN" ]; then
  tmux new-session -d -s "$SESSION_NAME" "export HUGGING_FACE_HUB_TOKEN='$HF_TOKEN' && vllm serve ${VLLM_ARGS[*]} 2>&1 | tee '$LOG_FILE'"
else
  tmux new-session -d -s "$SESSION_NAME" "vllm serve ${VLLM_ARGS[*]} 2>&1 | tee '$LOG_FILE'"
fi

echo ""
echo_info "vLLM server started in tmux session: $SESSION_NAME"
echo_info "Server will be available at: http://0.0.0.0:8000"
echo ""
echo_info "Monitor logs:"
echo "  tail -f $LOG_FILE"
echo ""
echo_info "Attach to tmux session:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo_info "Stop vLLM:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
