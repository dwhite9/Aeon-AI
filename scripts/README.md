# Aeon Deployment Scripts

This directory contains automation scripts for deploying and managing the Aeon platform.

## Scripts Overview

### Production Deployment

#### `setup.sh`
Complete platform deployment script. Installs and configures everything needed to run Aeon.

**Requirements:**
- Root/sudo access
- Ubuntu/Debian-based system
- NVIDIA GPU with drivers installed
- Docker installed

**Usage:**
```bash
sudo ./setup.sh
```

**What it does:**
1. Installs K3s Kubernetes cluster
2. Installs NGINX Ingress Controller
3. Adds Helm repositories
4. Deploys infrastructure (Qdrant, Redis, PostgreSQL, Prometheus/Grafana)
5. Sets up local Docker registry
6. Builds and pushes application images
7. Deploys Aeon application

### Development

#### `dev-setup.sh`
Sets up local development environment without Kubernetes.

**Usage:**
```bash
./dev-setup.sh
```

**What it does:**
- Installs UV package manager
- Creates Python virtual environments
- Installs backend dependencies
- Installs frontend dependencies

#### `start-host-services.sh`
Starts GPU-dependent services on the host machine using Docker Compose.

**Usage:**
```bash
./start-host-services.sh
```

**Services started:**
- Embedding server (port 8001)
- Instructions for starting vLLM manually

### Build & Deploy

#### `build.sh`
Builds and pushes all Docker images.

**Usage:**
```bash
./build.sh

# Custom registry and tag
REGISTRY=my-registry.com TAG=v1.0.0 ./build.sh
```

**Images built:**
- `aeon-api` - FastAPI backend
- `aeon-ui` - React frontend
- `aeon-embeddings` - Embedding server

### Maintenance

#### `cleanup.sh`
Removes all Aeon deployments and optionally data.

**Usage:**
```bash
./cleanup.sh
```

**⚠️ WARNING:** This will delete all data! Backup important data first.

## Quick Start Workflows

### First Time Setup (Production)

```bash
# 1. Run complete setup
sudo ./setup.sh

# 2. Start host services (in separate terminals)
cd ../inference
./start_vllm.sh              # Terminal 1
python3 embedding_server.py  # Terminal 2

# 3. Access the application
# Add to /etc/hosts: 127.0.0.1 aeon.local
# Open browser: http://aeon.local
```

### Development Workflow

```bash
# 1. Setup development environment
./dev-setup.sh

# 2. Start services in separate terminals
cd ../services && source .venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8080

cd ../ui
npm run dev

cd ../inference && source .venv/bin/activate
python embedding_server.py

# 3. Access at http://localhost:3000
```

### Update and Redeploy

```bash
# 1. Rebuild images
./build.sh

# 2. Restart deployments
kubectl rollout restart deployment/api-backend
kubectl rollout restart deployment/ui-frontend

# 3. Check status
kubectl get pods
kubectl logs -f deployment/api-backend
```

## Environment Variables

### `setup.sh`
- No environment variables needed (interactive prompts)

### `build.sh`
- `REGISTRY` - Docker registry URL (default: `localhost:5000`)
- `TAG` - Image tag (default: `latest`)

## Troubleshooting

### K3s Installation Fails

```bash
# Check if K3s is already installed
sudo k3s kubectl get nodes

# Uninstall and retry
sudo /usr/local/bin/k3s-uninstall.sh
sudo ./setup.sh
```

### Docker Registry Connection Issues

```bash
# Check if registry is running
docker ps | grep registry

# Restart registry
docker restart registry

# Or start new registry
docker run -d -p 5000:5000 --restart=always --name registry registry:2
```

### Image Pull Errors

```bash
# Re-push images
./build.sh

# Check if images exist
curl http://localhost:5000/v2/_catalog
```

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Install NVIDIA Container Toolkit if needed
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Script Dependencies

All scripts assume they're run from the `scripts/` directory and that the repository structure is intact:

```
Aeon/
├── scripts/      # You are here
├── k8s/          # Kubernetes manifests
├── services/     # Backend code
├── ui/           # Frontend code
└── inference/    # Host services
```

## Security Notes

- **Default passwords** are used in these scripts (PostgreSQL, Grafana)
- **Change all default passwords** in production deployments
- The setup uses a **local Docker registry** - configure proper registry authentication for production
- **No TLS/SSL** is configured by default - add certificates for production use

## Support

For issues or questions:
1. Check logs: `kubectl logs -f deployment/api-backend`
2. Check pod status: `kubectl get pods -A`
3. Review K8s events: `kubectl get events --sort-by='.lastTimestamp'`
