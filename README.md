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

✅ **Phase 2: RAG Pipeline - COMPLETE**
- ✅ Document processing and semantic chunking
- ✅ Qdrant integration for vector storage
- ✅ Retrieval with two-tier caching (Redis + Qdrant)
- ✅ PostgreSQL analytics integration

✅ **Phase 3: Advanced Agent System - COMPLETE**
- ✅ LangGraph multi-tool agent implementation (Cipher)
- ✅ Web search integration (SearXNG)
- ✅ Tool orchestration and intelligent routing
- ✅ Agent API endpoints and status monitoring

✅ **Phase 4: Code Execution - COMPLETE**
- ✅ Kubernetes Job-based Python code execution
- ✅ Security sandboxing and resource limits
- ✅ Code validation and safety checks
- ✅ RBAC permissions and isolation

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
  - **Podman** (recommended) or Docker installed and running
  - Root/sudo access
  - Git

**Note on Container Runtime**: Aeon now supports both Docker and Podman. **Podman is recommended** for better system isolation and security. See [Installing Podman](#installing-podman) section below.

## Installing Podman

Podman provides better system isolation than Docker by running containers rootless and daemonless. It's fully compatible with Docker commands.

### Quick Installation

```bash
# Install Podman and configure GPU support
cd scripts
sudo ./install-podman.sh
```

The installation script will:
- Install Podman and podman-compose
- Configure NVIDIA GPU support (if available)
- Set up rootless containers
- Optionally create a `docker` alias for compatibility

### Manual Installation

```bash
# Ubuntu 22.04+
sudo apt-get update
sudo apt-get install -y podman

# Install podman-compose
pip3 install podman-compose

# For GPU support
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
```

### Verify Installation

```bash
# Check Podman version
podman --version

# Test GPU access
podman run --rm --device nvidia.com/gpu=all \
  docker.io/nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Deployment Options

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
1. Detect and use Podman or Docker (prefers Podman)
2. Install K3s Kubernetes cluster
3. Deploy all infrastructure services
4. Build and deploy Aeon application
5. Configure monitoring and ingress

**Container Runtime**: The setup script automatically detects Podman (preferred) or Docker. To specify:
```bash
# Force use of Podman
CONTAINER_RUNTIME=podman sudo -E ./setup.sh

# Or use Docker
CONTAINER_RUNTIME=docker sudo -E ./setup.sh
```

### Option 2: Podman Development Environment (Recommended)

For local development with full isolation using Podman:

```bash
# Copy environment configuration
cp .env.example .env
# Edit .env with your settings (optional)

# Start full development environment
cd scripts
./podman-services.sh start-dev
```

This starts all services in isolated Podman containers:
- vLLM Server (GPU-accelerated): http://localhost:8000
- Embedding Server (GPU-accelerated): http://localhost:8001
- Redis Cache: localhost:6379
- PostgreSQL: localhost:5432
- Qdrant Vector DB: http://localhost:6333
- Backend API: http://localhost:8080
- Frontend UI: http://localhost:3000

**Managing Services**:
```bash
# View all commands
./podman-services.sh help

# Check service status
./podman-services.sh status

# View logs
./podman-services.sh logs          # All services
./podman-services.sh logs vllm-server  # Specific service

# Restart services
./podman-services.sh restart

# Stop all services
./podman-services.sh stop

# Health check
./podman-services.sh health
```

### Option 3: Traditional Development Setup

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

1. **Start Inference Services** (choose one option):

   **Option A: Using Podman (Recommended for isolation)**:
   ```bash
   cd scripts
   ./podman-services.sh start-prod
   ```
   This starts vLLM, embeddings, and registry in isolated Podman containers.

   **Option B: Traditional Host Services**:
   ```bash
   cd inference

   # Start vLLM in one terminal
   ./start_vllm.sh

   # Start embedding service in another terminal
   python3 embedding_server.py

   # Or use Docker Compose
   docker-compose up -d
   # Or use Podman Compose
   podman-compose up -d
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
   # Check Kubernetes pods
   kubectl get pods -A
   kubectl logs -f deployment/api-backend
   kubectl logs -f deployment/ui-frontend

   # Check Podman services
   cd scripts
   ./podman-services.sh status
   ./podman-services.sh health

   # Check resource usage
   kubectl top nodes
   kubectl top pods
   podman stats  # Podman container stats
   ```

## Repository Structure

```
aeon/
├── services/           # FastAPI backend services
│   ├── api/           # Main API endpoints ✅
│   ├── rag/           # RAG retrieval and chunking ✅
│   ├── agent/         # Cipher agent (LangGraph workflows) ✅
│   ├── code_exec/     # Code execution via K8s Jobs ✅
│   └── analytics/     # Query logging and optimization - TODO
├── inference/          # vLLM and embedding server configs ✅
├── k8s/               # Kubernetes manifests ✅
│   ├── base/          # Core infrastructure (incl. SearXNG) ✅
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

The build script supports both Docker and Podman:

```bash
cd scripts
./build.sh

# Force use of Podman
CONTAINER_RUNTIME=podman ./build.sh

# Custom registry and tag
REGISTRY=my-registry.com TAG=v1.0.0 ./build.sh

# Use Podman with custom registry
CONTAINER_RUNTIME=podman REGISTRY=my-registry.com TAG=v1.0.0 ./build.sh
```

The script will:
1. Auto-detect Podman (preferred) or Docker
2. Start local container registry if needed
3. Build all images (backend, frontend, embeddings)
4. Push to specified registry

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

### Environment Variables

Copy and customize the environment configuration:

```bash
cp .env.example .env
```

Key settings in `.env`:
- `HOST_IP`: Your machine's IP address (auto-detected during setup)
- `CONTAINER_RUNTIME`: Choose `podman` (recommended) or `docker`
- `VLLM_ENDPOINT`: vLLM service endpoint
- `EMBEDDING_ENDPOINT`: Embedding service endpoint
- `POSTGRES_PASSWORD`: Database password (change in production!)

### Podman Network Architecture

When using Podman, services run in an isolated `aeon-network` with static IPs:

| Service | Internal IP | Host Port | Purpose |
|---------|-------------|-----------|---------|
| vLLM | 172.20.0.10 | 8000 | LLM inference |
| Embeddings | 172.20.0.11 | 8001 | Text embeddings |
| Registry | 172.20.0.20 | 5000 | Container images |
| Redis (dev) | 172.20.0.30 | 6379 | Cache |
| PostgreSQL (dev) | 172.20.0.31 | 5432 | Database |
| Qdrant (dev) | 172.20.0.32 | 6333 | Vector DB |
| Backend (dev) | 172.20.0.40 | 8080 | API |
| Frontend (dev) | 172.20.0.41 | 3000 | Web UI |

**For K8s deployment**, pods connect to Podman services using HOST_IP:
- vLLM: `http://HOST_IP:8000/v1`
- Embeddings: `http://HOST_IP:8001`

### Update Host IP

The backend needs to connect to inference services. This is auto-configured during setup, but you can manually update in `k8s/app/api-backend.yaml`:

```yaml
data:
  VLLM_ENDPOINT: "http://YOUR_HOST_IP:8000/v1"
  EMBEDDING_ENDPOINT: "http://YOUR_HOST_IP:8001"
```

### Component Configuration

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

### Remove Podman Services

```bash
# Stop and remove all Podman services
cd scripts
./podman-services.sh cleanup

# Or manually
podman-compose --profile dev down -v
```

### Remove Kubernetes Deployment

```bash
# Remove all K8s deployments
cd scripts
./cleanup.sh

# Completely remove K3s
sudo /usr/local/bin/k3s-uninstall.sh
```

### Complete Cleanup

```bash
# Remove everything (K8s + Podman)
cd scripts
./cleanup.sh                    # Remove K8s
./podman-services.sh cleanup    # Remove Podman services
sudo /usr/local/bin/k3s-uninstall.sh  # Uninstall K3s
```

## Migration from Docker

If you're currently using Docker and want to migrate to Podman for better isolation, see the comprehensive [Podman Migration Guide](docs/PODMAN_MIGRATION.md).

The guide covers:
- Step-by-step migration process
- Troubleshooting common issues
- Performance comparisons
- Rollback procedures

## License

MIT

## Contributing

This is currently a personal research project to experiement and learn more about AI Systems Archetecture. Contributions welcome once the initial implementation is complete.
