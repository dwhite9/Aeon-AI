# Migrating from Docker to Podman

This guide explains how to migrate your Aeon AI Platform from Docker to Podman for better system isolation and security.

## Table of Contents

- [Why Podman?](#why-podman)
- [Prerequisites](#prerequisites)
- [Migration Steps](#migration-steps)
- [Podman vs Docker Differences](#podman-vs-docker-differences)
- [Troubleshooting](#troubleshooting)
- [Rollback to Docker](#rollback-to-docker)

## Why Podman?

Podman offers several advantages over Docker:

1. **Rootless Containers**: Run containers without root privileges for better security
2. **Daemonless Architecture**: No background daemon required, reducing attack surface
3. **Drop-in Replacement**: Compatible with Docker CLI and Compose files
4. **Pod Support**: Native pod support like Kubernetes
5. **Better Isolation**: Each container runs in its own user namespace
6. **No Single Point of Failure**: No daemon means no single point of failure

## Prerequisites

Before migrating, ensure you have:

- Ubuntu 20.04+ or compatible Linux distribution
- Existing Aeon installation running with Docker
- Root/sudo access
- At least 10GB free disk space (for pulling new images)
- Active backup of important data (optional but recommended)

## Migration Steps

### 1. Backup Current State

```bash
# Backup Docker volumes (if any important data)
cd /var/lib/docker/volumes
sudo tar -czf ~/docker-volumes-backup.tar.gz .

# Export Docker images to local files (optional)
docker save localhost:5000/aeon-api:latest -o ~/aeon-api.tar
docker save localhost:5000/aeon-ui:latest -o ~/aeon-ui.tar
docker save localhost:5000/aeon-embeddings:latest -o ~/aeon-embeddings.tar
```

### 2. Stop Docker Services

```bash
# Stop inference services
cd ~/Aeon-AI/inference
docker-compose down

# Stop local registry
docker stop registry
docker rm registry

# Or stop all Aeon-related containers
docker ps -a | grep aeon | awk '{print $1}' | xargs docker stop
docker ps -a | grep aeon | awk '{print $1}' | xargs docker rm
```

### 3. Install Podman

```bash
cd ~/Aeon-AI/scripts
sudo ./install-podman.sh
```

This will:
- Install Podman and podman-compose
- Configure NVIDIA GPU support
- Set up rootless containers
- Optionally create a `docker` alias

### 4. Verify Podman Installation

```bash
# Check Podman version
podman --version

# Test GPU access
podman run --rm --device nvidia.com/gpu=all \
  docker.io/nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Should show your GPU information
```

### 5. Import Images to Podman (Optional)

If you saved Docker images in step 1:

```bash
# Load images into Podman
podman load -i ~/aeon-api.tar
podman load -i ~/aeon-ui.tar
podman load -i ~/aeon-embeddings.tar
```

Or rebuild images with Podman:

```bash
cd ~/Aeon-AI/scripts
CONTAINER_RUNTIME=podman ./build.sh
```

### 6. Start Podman Services

#### Option A: Production Inference Services

```bash
cd ~/Aeon-AI/scripts
./podman-services.sh start-prod
```

This starts:
- vLLM Server
- Embedding Server
- Container Registry

#### Option B: Full Development Environment

```bash
cd ~/Aeon-AI/scripts
./podman-services.sh start-dev
```

This starts all services including databases and backend/frontend.

### 7. Verify Services

```bash
# Check service status
./podman-services.sh status

# Check health
./podman-services.sh health

# View logs
./podman-services.sh logs
```

### 8. Update K8s Configuration (If Using K8s)

If you're using Kubernetes, the services should already be configured correctly. Verify:

```bash
# Check if backend can reach Podman services
kubectl logs -f deployment/api-backend

# Should see successful connections to vLLM and embedding services
```

### 9. Test the Application

```bash
# Test vLLM endpoint
curl http://localhost:8000/v1/models

# Test embedding endpoint
curl http://localhost:8001/health

# Test backend (if running dev environment)
curl http://localhost:8080/health

# Test frontend (if running dev environment)
curl http://localhost:3000/
```

### 10. Clean Up Docker (Optional)

Once you've verified everything works with Podman:

```bash
# Remove Docker images (frees up space)
docker image prune -a

# Optionally uninstall Docker completely
sudo apt-get remove docker-ce docker-ce-cli containerd.io
sudo rm -rf /var/lib/docker

# Keep Docker installed if you want it as a fallback
```

## Podman vs Docker Differences

### Command Compatibility

Podman is designed to be a drop-in replacement:

| Docker Command | Podman Equivalent | Notes |
|----------------|-------------------|-------|
| `docker run` | `podman run` | Identical |
| `docker build` | `podman build` | Identical |
| `docker ps` | `podman ps` | Identical |
| `docker-compose` | `podman-compose` | Requires podman-compose package |
| `docker exec` | `podman exec` | Identical |
| `docker logs` | `podman logs` | Identical |

### Key Differences

1. **No Daemon**: Podman runs containers directly, no background daemon
2. **Rootless by Default**: Containers run as your user, not root
3. **Pod Support**: `podman pod` commands for managing groups of containers
4. **Socket Location**: Podman socket is at `/run/user/$UID/podman/podman.sock`

### Using Docker Alias

If you want to keep using `docker` commands:

```bash
# Add to ~/.bashrc
alias docker=podman

# Reload
source ~/.bashrc

# Now docker commands work
docker ps
docker run ...
```

## Troubleshooting

### Issue: GPU Not Detected

**Symptom**: Container can't access GPU

**Solution**:
```bash
# Verify NVIDIA Container Toolkit is installed
nvidia-ctk --version

# Regenerate CDI configuration
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Test GPU access
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Issue: Permission Denied

**Symptom**: "Permission denied" when running Podman commands

**Solution**:
```bash
# Run as current user (rootless)
podman ps

# If you need root Podman
sudo podman ps

# Enable user namespaces
sudo sysctl -w user.max_user_namespaces=15000
echo "user.max_user_namespaces=15000" | sudo tee -a /etc/sysctl.conf
```

### Issue: Network Connectivity Problems

**Symptom**: Containers can't communicate with each other or host

**Solution**:
```bash
# Check network
podman network ls
podman network inspect aeon-network

# Recreate network
podman network rm aeon-network
cd ~/Aeon-AI
podman-compose down
podman-compose up -d

# Check firewall rules
sudo iptables -L
```

### Issue: Port Already in Use

**Symptom**: "bind: address already in use"

**Solution**:
```bash
# Find what's using the port
sudo lsof -i :8000  # Replace with your port

# If Docker container is still running
docker ps
docker stop <container-id>

# If Podman container from previous run
podman ps -a
podman stop <container-id>
podman rm <container-id>
```

### Issue: Images Not Found

**Symptom**: "Error: image not found in local storage"

**Solution**:
```bash
# Pull images from Docker Hub with full path
podman pull docker.io/library/registry:2
podman pull docker.io/nvidia/cuda:12.1.0-base-ubuntu22.04

# Or rebuild images
cd ~/Aeon-AI/scripts
CONTAINER_RUNTIME=podman ./build.sh
```

### Issue: Slow Container Startup

**Symptom**: Containers take long time to start

**Solution**:
```bash
# Check if using overlay2 storage driver
podman info | grep -A5 "graphDriverName"

# Should show "overlay"
# If not, configure it:
mkdir -p ~/.config/containers
cat > ~/.config/containers/storage.conf << 'EOF'
[storage]
driver = "overlay"

[storage.options]
mount_program = "/usr/bin/fuse-overlayfs"
EOF

# Restart Podman
systemctl --user restart podman
```

### Issue: K8s Pods Can't Reach Podman Services

**Symptom**: Backend logs show connection errors to vLLM/embeddings

**Solution**:
```bash
# Verify HOST_IP is correct
ip route get 1.1.1.1 | grep -oP 'src \K[^ ]+'

# Update K8s ConfigMap
kubectl edit configmap api-backend-config

# Update VLLM_ENDPOINT and EMBEDDING_ENDPOINT with correct HOST_IP
# Restart backend pods
kubectl rollout restart deployment/api-backend
```

## Rollback to Docker

If you encounter issues and need to rollback:

### 1. Stop Podman Services

```bash
cd ~/Aeon-AI/scripts
./podman-services.sh stop
```

### 2. Restore Docker Images

If you backed up images:

```bash
docker load -i ~/aeon-api.tar
docker load -i ~/aeon-ui.tar
docker load -i ~/aeon-embeddings.tar
```

Or rebuild:

```bash
cd ~/Aeon-AI/scripts
CONTAINER_RUNTIME=docker ./build.sh
```

### 3. Start Docker Services

```bash
# Start registry
docker run -d -p 5000:5000 --restart=always --name registry registry:2

# Start inference services
cd ~/Aeon-AI/inference
docker-compose up -d

# Or use the old method
./start_vllm.sh
python3 embedding_server.py
```

### 4. Verify Docker Services

```bash
docker ps
curl http://localhost:8000/v1/models
curl http://localhost:8001/health
```

## Best Practices

### 1. Use Podman Compose for Multi-Service Deployments

```bash
# Instead of managing containers individually
cd ~/Aeon-AI
podman-compose up -d
```

### 2. Monitor Resource Usage

```bash
# Real-time stats
podman stats

# Specific container
podman stats aeon-vllm
```

### 3. Regular Cleanup

```bash
# Remove unused images
podman image prune -a

# Remove stopped containers
podman container prune

# Remove unused volumes
podman volume prune
```

### 4. Use Systemd for Auto-Start (Optional)

```bash
# Generate systemd unit file
podman generate systemd --name aeon-vllm > ~/.config/systemd/user/aeon-vllm.service

# Enable auto-start
systemctl --user enable aeon-vllm.service
systemctl --user start aeon-vllm.service
```

### 5. Keep Podman Updated

```bash
# Update Podman
sudo apt-get update
sudo apt-get upgrade podman

# Update podman-compose
pip3 install --upgrade podman-compose
```

## Performance Comparison

| Metric | Docker | Podman | Notes |
|--------|--------|--------|-------|
| Container Startup | ~2-3s | ~2-3s | Similar |
| Memory Overhead | ~100MB (daemon) | ~0MB (no daemon) | Podman wins |
| GPU Performance | Excellent | Excellent | Identical |
| Build Time | Baseline | Similar | Negligible difference |
| Security | Good | Excellent | Rootless by default |

## Additional Resources

- [Podman Official Documentation](https://docs.podman.io/)
- [Podman vs Docker](https://docs.podman.io/en/latest/Introduction.html#podman-vs-docker)
- [Podman Compose GitHub](https://github.com/containers/podman-compose)
- [Rootless Containers](https://rootlesscontaine.rs/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

## Support

If you encounter issues during migration:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Review Podman logs: `journalctl --user -u podman`
3. Check Aeon service logs: `./podman-services.sh logs`
4. Open an issue on the Aeon repository with:
   - Podman version: `podman --version`
   - OS version: `cat /etc/os-release`
   - Error messages and logs
   - Steps to reproduce
