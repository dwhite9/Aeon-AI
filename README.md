# Aeon

> Self-hosted AI platform for home lab environments with intelligent agent capabilities

## Overview

Aeon is a privacy-focused, self-hosted AI platform designed to run on Kubernetes (K3s) in home lab environments. It features **Cipher**, an intelligent AI agent with RAG (Retrieval-Augmented Generation) capabilities, web search integration, code execution, and self-tuning optimization.

## Key Features

- **Local LLM**: Mistral 7B Instruct (8-bit quantized) via vLLM
- **RAG Pipeline**: Semantic document chunking with Qdrant vector database
- **Intelligent Agent**: Multi-tool agent using LangGraph (web search, document retrieval, code execution)
- **Two-Tier Caching**: Redis exact cache + Qdrant semantic cache for fast responses
- **Code Execution**: Sandboxed Python execution via Kubernetes Jobs
- **Self-Tuning**: Automated nightly optimization and weekly embedding fine-tuning
- **Privacy-First**: All processing happens locally on your infrastructure

## Tech Stack

- **Backend**: FastAPI (async/await)
- **Frontend**: React + TypeScript
- **LLM Serving**: vLLM with Mistral 7B Instruct
- **Vector Database**: Qdrant
- **Cache & Sessions**: Redis
- **Analytics DB**: PostgreSQL
- **Orchestration**: K3s (lightweight Kubernetes)
- **Monitoring**: Prometheus + Grafana

## Architecture

Aeon uses a hybrid deployment model with GPU-intensive services on the host machine and application services in Kubernetes:

- **Host Services** (GPU): vLLM inference server, embedding generation, model fine-tuning
- **K3s Services**: React UI, FastAPI backend, LangGraph agent, databases, monitoring

View detailed architecture diagrams in the [diagrams/](diagrams/) directory:
- [System Context](diagrams/structurizr-SystemContext.mmd) - High-level overview
- [Container Architecture](diagrams/structurizr-Containers.mmd) - Complete system structure
- [Host Services](diagrams/structurizr-HostServices.mmd) - GPU workloads
- [K3s Services](diagrams/structurizr-K3sServices.mmd) - Kubernetes deployments
- [API Components](diagrams/structurizr-APIComponents.mmd) - Backend internals
- [Agent Components](diagrams/structurizr-AgentComponents.mmd) - LangGraph agent structure
- [Deployment](diagrams/structurizr-Deployment.mmd) - Physical infrastructure

## Resource Requirements

**Recommended Single VM Configuration:**
- CPU: 16 cores (14 allocated, 2 for host)
- RAM: 56GB (48GB allocated, 8GB for host)
- GPU: NVIDIA GPU with 8GB+ VRAM
- Storage: 100GB SSD (thin provisioned)

## Project Status

✅ **Phase 1: Foundation - COMPLETE**
- ✅ Repository structure and directory layout
- ✅ Host services (vLLM and embedding server)
- ✅ FastAPI backend with chat endpoints and session management
- ✅ React frontend with TypeScript and Vite
- ✅ Kubernetes manifests for infrastructure and application
- ✅ Deployment scripts and automation

🚧 **Phase 2: RAG Pipeline - TODO**
- ⏳ Document processing and semantic chunking
- ⏳ Qdrant integration for vector storage
- ⏳ Retrieval with two-tier caching
- ⏳ PostgreSQL analytics integration

🚧 **Phase 3: Advanced Agent System - TODO**
- ⏳ LangGraph multi-tool agent implementation
- ⏳ Web search integration (SearXNG)
- ⏳ Tool orchestration and routing

🚧 **Phase 4: Code Execution - TODO**
- ⏳ Kubernetes Job-based code execution
- ⏳ Security sandboxing and resource limits

🚧 **Phase 5: Self-Tuning - TODO**
- ⏳ Query analytics pipeline
- ⏳ Nightly optimization jobs
- ⏳ Embedding fine-tuning

## Quick Start

### Prerequisites

- **Operating System**: Ubuntu 20.04+ or similar Linux distribution
- **Hardware**:
  - 16+ CPU cores
  - 56GB+ RAM
  - NVIDIA GPU with 8GB+ VRAM and drivers installed
  - 100GB+ free disk space
- **Software**:
  - Docker installed and running
  - Root/sudo access
  - Git

### Option 1: Automated Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/aeon.git
cd aeon

# Run complete setup (requires sudo)
cd scripts
sudo ./setup.sh
```

The setup script will:
1. Install K3s Kubernetes cluster
2. Deploy all infrastructure services
3. Build and deploy Aeon application
4. Configure monitoring and ingress

### Option 2: Development Setup

For local development without Kubernetes:

```bash
# Setup development environment
cd scripts
./dev-setup.sh

# Start services in separate terminals

# Terminal 1: Backend
cd services && source .venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8080

# Terminal 2: Frontend
cd ui
npm run dev

# Terminal 3: Embedding Server
cd inference && source .venv/bin/activate
python embedding_server.py

# Terminal 4: vLLM (requires GPU)
cd inference
./start_vllm.sh
```

Access the application at: http://localhost:3000

### Post-Installation

After running the setup script:

1. **Start Host Services** (GPU-dependent services run on host):
   ```bash
   cd inference

   # Start vLLM in one terminal
   ./start_vllm.sh

   # Or start embedding service via Docker
   cd inference
   docker-compose up -d
   ```

2. **Access the Application**:

   Add to `/etc/hosts`:
   ```
   127.0.0.1 aeon.local
   ```

   Then open in browser:
   - **Web UI**: http://aeon.local
   - **Grafana**: http://localhost:30001 (admin/prom-operator)
   - **API Docs**: http://aeon.local/api/docs

3. **Verify Services**:
   ```bash
   # Check all pods are running
   kubectl get pods -A

   # Check application logs
   kubectl logs -f deployment/api-backend
   kubectl logs -f deployment/ui-frontend

   # Check resource usage
   kubectl top nodes
   kubectl top pods
   ```

## Repository Structure

```
aeon/
├── agent/              # Cipher agent logic (LangGraph workflows) - TODO
├── services/           # FastAPI backend services
│   ├── api/           # Main API endpoints ✅
│   ├── rag/           # RAG retrieval and chunking - TODO
│   ├── code_exec/     # Code execution via K8s Jobs - TODO
│   └── analytics/     # Query logging and optimization - TODO
├── inference/          # vLLM and embedding server configs ✅
├── k8s/               # Kubernetes manifests ✅
│   ├── base/          # Core infrastructure ✅
│   ├── app/           # Application deployments ✅
│   └── jobs/          # CronJobs for optimization - TODO
├── ui/                # React frontend ✅
├── scripts/           # Deployment and setup scripts ✅
└── diagrams/          # Architecture diagrams
```

## Development

### Package Management

This project uses [UV](https://github.com/astral-sh/uv) for Python package management:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd services
uv pip install -r requirements.txt

cd inference
uv pip install -r requirements.txt
```

### Building Images

```bash
cd scripts
./build.sh

# Custom registry and tag
REGISTRY=my-registry.com TAG=v1.0.0 ./build.sh
```

### Testing

```bash
# Backend tests
cd services
pytest

# Frontend tests
cd ui
npm test
```

## Configuration

### Update Host IP

The backend needs to connect to vLLM and embedding services on the host. Update the IP in:

`k8s/app/api-backend.yaml`:
```yaml
data:
  VLLM_ENDPOINT: "http://YOUR_HOST_IP:8000/v1"
  EMBEDDING_ENDPOINT: "http://YOUR_HOST_IP:8001"
```

### Environment Variables

See individual component READMEs:
- [Backend Configuration](services/README.md)
- [Frontend Configuration](ui/README.md)
- [Inference Services](inference/README.md)
- [Kubernetes Deployment](k8s/README.md)

## Monitoring

Access Grafana at http://localhost:30001

Default credentials: `admin` / `prom-operator`

Key metrics:
- LLM requests/sec and latency
- Cache hit rates
- GPU utilization
- Pod resource usage

## Troubleshooting

### Services Not Starting

```bash
# Check pod status
kubectl get pods -A

# Check logs
kubectl logs -f deployment/api-backend
kubectl describe pod POD_NAME

# Check events
kubectl get events --sort-by='.lastTimestamp'
```

### GPU Not Detected

```bash
# Verify NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Connection Issues

```bash
# Test backend health
kubectl port-forward svc/api-backend 8080:8080
curl http://localhost:8080/health

# Test from within cluster
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://api-backend:8080/health
```

For more troubleshooting tips, see [scripts/README.md](scripts/README.md)

## Cleanup

```bash
# Remove all deployments
cd scripts
./cleanup.sh

# Completely remove K3s
sudo /usr/local/bin/k3s-uninstall.sh
```

## License

MIT

## Contributing

This is currently a personal research project to experiement and learn more about AI Systems Archetecture. Contributions welcome once the initial implementation is complete.
