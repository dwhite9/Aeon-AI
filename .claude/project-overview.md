# Aeon AI Platform - Project Overview

## Table of Contents
- [Project Summary](#project-summary)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Development Setup](#development-setup)
- [Deployment Guide](#deployment-guide)
- [Environment Variables](#environment-variables)
- [Common Tasks](#common-tasks)
- [Troubleshooting](#troubleshooting)

---

## Project Summary

**Aeon** is a self-hosted AI platform designed for home lab environments running on K3s (lightweight Kubernetes). The platform features **Cipher**, an intelligent AI agent with RAG (Retrieval-Augmented Generation) capabilities, web search, code execution, and self-tuning optimization.

### Key Features
- **Multi-modal AI Agent (Cipher)**: LangGraph-based agent with tool orchestration
- **RAG Pipeline**: Semantic chunking, two-tier caching (exact + semantic)
- **GPU-Accelerated Inference**: vLLM serving Mistral 7B (8-bit quantized)
- **Code Execution**: Sandboxed Python execution via K8s Jobs
- **Self-Tuning**: Nightly optimization and weekly fine-tuning
- **Full Stack**: FastAPI backend + React TypeScript frontend

### Current Status
**Phase 3 Complete**: Advanced agent system with LangGraph orchestration, multi-tool support, and production-ready deployment scripts.

---

## Architecture

### Deployment Model

```
┌─────────────────────────────────────┐
│  Host Machine (GPU)                 │
│  - vLLM Server (port 8000)         │
│  - Embedding Server (port 8001)    │
│  - Podman/Docker Runtime           │
└─────────────┬───────────────────────┘
              │ Network
┌─────────────▼───────────────────────┐
│  K3s Cluster (VM or Host)           │
│  ┌───────────────────────────────┐ │
│  │ Application Layer             │ │
│  │ - FastAPI Backend             │ │
│  │ - React Frontend              │ │
│  │ - Cipher Agent (LangGraph)    │ │
│  └───────────────────────────────┘ │
│  ┌───────────────────────────────┐ │
│  │ Data Layer                    │ │
│  │ - Qdrant (vectors)            │ │
│  │ - PostgreSQL (analytics)      │ │
│  │ - Redis (cache)               │ │
│  └───────────────────────────────┘ │
│  ┌───────────────────────────────┐ │
│  │ Infrastructure                │ │
│  │ - NGINX Ingress               │ │
│  │ - Prometheus + Grafana        │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
```

### Why Split Architecture?
- **GPU services on host**: Direct GPU access, simpler than GPU passthrough
- **Everything else in K8s**: Orchestration, scaling, resource management
- **Network communication**: K8s pods reach host services via `HOST_IP`

---

## Technology Stack

### Core Services
- **LLM**: vLLM + Mistral 7B Instruct (8-bit quantized, ~7GB VRAM)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- **Backend**: FastAPI with async/await
- **Frontend**: React + TypeScript
- **Agent Framework**: LangGraph for state-based workflows

### Data Layer
- **Vector Database**: Qdrant (HNSW indexing, filtered search)
- **Cache**: Redis (session + semantic caching)
- **Database**: PostgreSQL (analytics, query logs)

### Infrastructure
- **Container Runtime**: Podman (preferred) or Docker
- **Orchestration**: K3s (single-node Kubernetes)
- **Monitoring**: Prometheus + Grafana
- **Package Manager**: UV (fast Python package management)

### Resource Allocation
```
CPU: 14 cores (16 total, 2 reserved for host)
RAM: 48GB (56GB total, 8GB reserved)
GPU: NVIDIA GPU shared for vLLM + embeddings
Storage: Expandable, uses /zdata for large files
```

---

## Repository Structure

```
aeon/
├── .claude/                    # Claude Code project documentation
│   └── project-overview.md     # This file
├── .env.template              # Environment variable template
├── .env                       # Local config (gitignored)
├── CLAUDE.md                  # Legacy project instructions
│
├── agent/                     # Cipher agent logic (LangGraph)
│   ├── workflows/            # Agent workflow definitions
│   ├── tools/                # Agent tool implementations
│   └── state.py              # Agent state management
│
├── services/                  # FastAPI backend services
│   ├── api/                  # REST API endpoints
│   ├── rag/                  # RAG retrieval and chunking
│   ├── code_exec/            # Code execution service
│   └── analytics/            # Query logging and optimization
│
├── inference/                 # Inference service configs
│   ├── Dockerfile.embeddings # Embedding server container
│   ├── docker-compose.yml    # Local dev compose
│   └── embedding_server.py   # Embedding API server
│
├── k8s/                       # Kubernetes manifests
│   ├── base/                 # Infrastructure (Qdrant, Redis, Postgres)
│   ├── app/                  # Application deployments
│   └── jobs/                 # CronJobs for optimization
│
├── ui/                        # React frontend
│   ├── src/
│   └── package.json
│
└── scripts/                   # Deployment and setup scripts
    ├── install-podman.sh     # Podman installation + GPU setup
    ├── start-host-services.sh # Start inference services
    ├── setup.sh              # Full K3s deployment
    └── podman-services.sh    # Podman service management
```

---

## Development Setup

### Prerequisites
- Ubuntu 24.04 LTS (or 20.04/22.04)
- NVIDIA GPU with drivers installed
- 16GB+ RAM, 100GB+ storage
- Python 3.11+

### Initial Setup

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd Aeon
   ```

2. **Configure Environment**
   ```bash
   # Copy template and customize
   cp .env.template .env

   # Edit .env with your settings
   nano .env

   # Key settings to update:
   # - TMPDIR: Use a partition with plenty of space (e.g., /zdata/tmp)
   # - HOST_IP: Your host machine's IP address
   # - POSTGRES_PASSWORD: Change from default
   ```

3. **Install Podman (Recommended)**
   ```bash
   ./scripts/install-podman.sh
   ```

   This script:
   - Installs Podman 4.9+ from Ubuntu repos (24.04) or Kubic (20.04/22.04)
   - Installs podman-compose via UV in a venv
   - Configures NVIDIA Container Toolkit for GPU access
   - Sets up Docker Hub for short image names
   - Configures rootless Podman storage

4. **Start Inference Services**
   ```bash
   # Source environment variables
   source .env

   # Start vLLM and embedding services
   ./scripts/start-host-services.sh
   ```

5. **Deploy K3s Cluster (Optional for full stack)**
   ```bash
   sudo ./scripts/setup.sh
   ```

### Development Workflow

**Backend Development**
```bash
cd services
uv pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8080
```

**Frontend Development**
```bash
cd ui
npm install
npm run dev
```

**Agent Development**
```bash
cd agent
uv pip install -r requirements.txt
python -m workflows.test_agent
```

---

## Deployment Guide

### Deployment Options

1. **Development Mode** (Podman Compose only)
   ```bash
   ./scripts/podman-services.sh start-dev
   ```
   All services in containers, no K8s needed.

2. **Production Mode** (K3s + Host Inference)
   ```bash
   # 1. Start inference services on host
   ./scripts/start-host-services.sh

   # 2. Deploy K8s cluster
   sudo ./scripts/setup.sh
   ```

3. **Custom Deployment**
   - Inference on separate GPU server
   - K3s cluster on VM
   - Multi-node K3s (future)

### Post-Deployment

**Verify Services**
```bash
# Check Podman services
podman-compose ps
podman-compose logs -f embeddings

# Check K8s pods
kubectl get pods -A
kubectl logs -f deployment/api-backend

# Check GPU access
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Access Web UI**
```bash
# Add to /etc/hosts
echo "127.0.0.1 aeon.local" | sudo tee -a /etc/hosts

# Access at http://aeon.local
```

**Monitor Performance**
- Grafana: `http://localhost:30001` (admin/prom-operator)
- Metrics: Request latency, cache hit rate, GPU utilization

---

## Environment Variables

See [`.env.template`](../.env.template) for comprehensive documentation.

### Critical Variables

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `TMPDIR` | Container build temp dir | `/tmp` | Use partition with >50GB free |
| `PODMAN_RUNROOT` | Podman runtime dir | `/tmp/podman/runroot` | Should match TMPDIR partition |
| `HOST_IP` | Host machine IP | `192.168.1.100` | K8s pods use this to reach host |
| `POSTGRES_PASSWORD` | DB password | `changeme_to_secure_password` | **Change in production!** |

### Variable Loading

Scripts automatically source `.env` if present:
```bash
# In scripts
if [ -f .env ]; then
    source .env
fi
```

---

## Common Tasks

### Managing Podman Services

```bash
# Start production services (vLLM, embeddings, registry)
./scripts/podman-services.sh start-prod

# Start full dev environment
./scripts/podman-services.sh start-dev

# Check status
./scripts/podman-services.sh status

# View logs
./scripts/podman-services.sh logs vllm-server

# Restart service
./scripts/podman-services.sh restart embedding-server

# Health check
./scripts/podman-services.sh health

# Stop all
./scripts/podman-services.sh stop
```

### Managing vLLM Server

```bash
# Start vLLM server (runs in foreground with trap handler)
cd inference
./start_vllm.sh

# Stop vLLM server (graceful shutdown)
./stop_vllm.sh

# Force kill if needed
pkill -9 -f 'vllm serve'

# Check GPU memory
nvidia-smi
```

### Managing K8s Deployments

```bash
# Deploy infrastructure
kubectl apply -f k8s/base/

# Deploy applications
kubectl apply -f k8s/app/

# Scale backend
kubectl scale deployment/api-backend --replicas=3

# Update configuration
kubectl edit configmap api-backend-config

# Restart deployment
kubectl rollout restart deployment/api-backend
```

### Database Operations

```bash
# Connect to PostgreSQL
kubectl exec -it postgres-postgresql-0 -- psql -U aiuser -d aiplatform

# Initialize analytics tables
kubectl exec -i postgres-postgresql-0 -- psql -U aiuser -d aiplatform < services/analytics/schema.sql

# Backup database
kubectl exec postgres-postgresql-0 -- pg_dump -U aiuser aiplatform > backup.sql
```

### Building Images

```bash
# Build backend
cd services
podman build -t localhost:5000/aeon-api:latest .
podman push localhost:5000/aeon-api:latest

# Build frontend
cd ui
podman build -t localhost:5000/aeon-ui:latest .
podman push localhost:5000/aeon-ui:latest
```

---

## Troubleshooting

### Podman Issues

**Short image names not resolving**
```bash
# Fix: Configure Docker Hub registry
sudo ./scripts/configure-podman-registries.sh

# Verify
grep "^unqualified-search-registries" /etc/containers/registries.conf
```

**No space left on device during build**
```bash
# Check disk space
df -h /var/tmp
df -h $TMPDIR

# Solution 1: Clean up Podman cache
podman system prune -a -f

# Solution 2: Configure TMPDIR in .env
echo "TMPDIR=/zdata/tmp" >> .env
echo "PODMAN_RUNROOT=/zdata/tmp/podman/runroot" >> .env

# Update storage config
nano ~/.config/containers/storage.conf
# Add: runroot = "/zdata/tmp/podman/runroot"
```

**Permission denied: bad interpreter**
```bash
# Fix venv permissions (if podman-compose installed with sudo)
sudo chmod -R a+rX /opt/aeon/venv/podman-compose

# Or reinstall in user venv
source .venv/bin/activate
uv pip install podman-compose
```

### GPU Issues

**GPU not accessible in container**
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify CDI config
ls -la /etc/cdi/nvidia.yaml

# Regenerate CDI
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Test GPU access
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**GPU memory not cleaned up after stopping vLLM**
```bash
# Check what's using GPU
nvidia-smi

# Find vLLM processes
ps aux | grep vllm

# Use stop script for graceful cleanup
cd inference
./stop_vllm.sh

# If stop script fails, force kill
pkill -9 -f 'vllm serve'

# Find all processes using GPU
sudo fuser -v /dev/nvidia*

# Last resort: reset GPU (requires sudo, kills all GPU processes)
sudo nvidia-smi --gpu-reset
```

**vLLM fails to start - already running**
```bash
# Check for existing processes
pgrep -f 'vllm serve'

# Stop existing instance
cd inference
./stop_vllm.sh

# Verify GPU memory is clear
nvidia-smi
```

### K8s Issues

**Pods can't reach host services**
```bash
# Verify HOST_IP is correct
echo $HOST_IP

# Test connectivity from pod
kubectl run -it test --image=busybox --rm -- wget -O- http://$HOST_IP:8000/health

# Check firewall
sudo ufw status
sudo ufw allow from 10.42.0.0/16  # K8s pod network
```

**Image pull failures**
```bash
# Check local registry
curl http://localhost:5000/v2/_catalog

# Rebuild and push
podman build -t localhost:5000/aeon-api:latest services/
podman push localhost:5000/aeon-api:latest

# Force pod to re-pull
kubectl delete pod -l app=api-backend
```

### vLLM Issues

**Out of memory errors**
```bash
# Reduce GPU memory utilization in .env
GPU_MEMORY_UTILIZATION=0.5
MAX_MODEL_LENGTH=4096

# Restart vLLM
podman-compose restart vllm-server
```

**Slow inference**
```bash
# Check GPU usage
nvidia-smi -l 1

# Check vLLM logs
podman-compose logs -f vllm-server

# Consider using smaller batch size
# Edit docker-compose.yml vLLM args
```

---

## Additional Resources

- **CLAUDE.md**: Legacy project overview (still valid)
- **K8s Manifests**: See `k8s/` for deployment configs
- **API Documentation**: FastAPI auto-docs at `http://localhost:8080/docs`
- **Grafana Dashboards**: Pre-configured monitoring at port 30001

---

## Project Conventions

### Logging Format
```python
logger.info("[Aeon::Component] Message")
# Examples:
logger.info("[Aeon::RAG] Retrieved 3 documents")
logger.info("[Aeon::Cipher] Processing query with tools")
```

### Docker Image Naming
```
aeon/<service-name>:<version>
# Examples:
aeon/cipher-api:latest
aeon/cipher-ui:latest
aeon/embeddings:latest
```

### Branding
- **Project Name**: Aeon
- **Agent Name**: Cipher
- **Log Prefix**: `[Aeon::Component]`

---

**Last Updated**: 2025-11-20
**Project Phase**: Phase 3 Complete - Advanced Agent System
