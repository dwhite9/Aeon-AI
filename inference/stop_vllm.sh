#!/bin/bash
# Stop vLLM server and clean up GPU memory
# This script gracefully terminates vLLM processes

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

echo_info "=== Stopping vLLM Server ==="
echo ""

# Find vLLM processes
VLLM_PIDS=$(pgrep -f "vllm serve" || true)

if [ -z "$VLLM_PIDS" ]; then
    echo_info "No vLLM processes found"
    exit 0
fi

echo_info "Found vLLM processes: $VLLM_PIDS"

# Try graceful shutdown first (SIGTERM)
echo_info "Sending SIGTERM for graceful shutdown..."
for pid in $VLLM_PIDS; do
    kill -15 "$pid" 2>/dev/null || true
done

# Wait up to 10 seconds for graceful shutdown
echo_info "Waiting for processes to terminate..."
for i in {1..10}; do
    if ! pgrep -f "vllm serve" > /dev/null; then
        echo_info "vLLM stopped gracefully"
        break
    fi
    sleep 1
done

# If still running, force kill (SIGKILL)
REMAINING=$(pgrep -f "vllm serve" || true)
if [ -n "$REMAINING" ]; then
    echo_warn "Processes still running, forcing shutdown..."
    for pid in $REMAINING; do
        kill -9 "$pid" 2>/dev/null || true
    done
    sleep 2
fi

# Verify all processes are stopped
if pgrep -f "vllm serve" > /dev/null; then
    echo_error "Failed to stop all vLLM processes"
    echo_error "Remaining processes:"
    ps aux | grep "vllm serve" | grep -v grep
    exit 1
fi

echo_info "All vLLM processes stopped"
echo ""

# Check GPU memory status
if command -v nvidia-smi &> /dev/null; then
    echo_info "GPU Memory Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    echo ""

    # Check for any remaining GPU processes
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)
    if [ -n "$GPU_PROCS" ]; then
        echo_warn "Other processes still using GPU:"
        echo "$GPU_PROCS"
    else
        echo_info "GPU memory cleared"
    fi
fi

echo ""
echo_info "vLLM server stopped successfully"
