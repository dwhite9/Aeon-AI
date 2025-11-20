#!/bin/bash
# Start vLLM server with Mistral 7B Instruct (8-bit quantized)
# This script runs vLLM on the host machine with GPU access

set -e

echo "[Aeon::vLLM] Starting vLLM inference server..."

# Check for CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "[Aeon::vLLM] WARNING: nvidia-smi not found. GPU may not be available."
fi

# Display GPU info
echo "[Aeon::vLLM] GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "Unable to query GPU"

# Start vLLM with 8-bit quantization
vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --dtype float16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.6 \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code

echo "[Aeon::vLLM] Server started on http://0.0.0.0:8000"
