# Aeon Inference Services

This directory contains GPU-intensive services that run on the host machine:

- **vLLM Server**: Serves Mistral 7B Instruct with 8-bit quantization
- **Embedding Server**: Generates text embeddings using sentence-transformers

## Requirements

- NVIDIA GPU with 8GB+ VRAM
- CUDA 12.1+
- Docker with NVIDIA Container Toolkit (for embedding server)

## Quick Start

### Option 1: Docker Compose (Embedding Server Only)

```bash
cd inference
docker-compose up -d
```

### Option 2: Direct Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start embedding server
python3 embedding_server.py

# Start vLLM server (in separate terminal)
chmod +x start_vllm.sh
./start_vllm.sh
```

## Services

### vLLM Server

- **Endpoint**: `http://localhost:8000`
- **Model**: Mistral 7B Instruct v0.3
- **Quantization**: 8-bit (bitsandbytes)
- **Context Length**: 8192 tokens
- **API**: OpenAI-compatible

**Test:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Embedding Server

- **Endpoint**: `http://localhost:8001`
- **Model**: all-MiniLM-L6-v2
- **Dimension**: 384
- **API**: FastAPI

**Test:**
```bash
curl http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "Test embedding"]
  }'
```

## Configuration

Edit the following variables in the scripts:

- `MODEL_NAME`: Change embedding model
- `--max-model-len`: Adjust vLLM context window
- `--gpu-memory-utilization`: Adjust VRAM usage (default: 0.6)

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Out of Memory

Reduce `--gpu-memory-utilization` in [start_vllm.sh](start_vllm.sh):

```bash
--gpu-memory-utilization 0.5  # Lower from 0.6 to 0.5
```

### Connection Refused

Ensure services are bound to `0.0.0.0`, not `localhost`:

```bash
--host 0.0.0.0  # Accessible from K8s pods
```
