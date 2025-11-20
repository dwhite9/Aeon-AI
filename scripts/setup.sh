#!/bin/bash
# Aeon Platform Complete Setup Script
# This script deploys the entire Aeon AI platform

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[Aeon::Setup]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[Aeon::Setup]${NC} $1"
}

echo_error() {
    echo -e "${RED}[Aeon::Setup]${NC} $1"
}

# Check if running as root (needed for K3s installation)
if [ "$EUID" -ne 0 ]; then
    echo_error "Please run as root (use sudo)"
    exit 1
fi

echo_info "=== Aeon AI Platform Setup ==="
echo ""

# Determine container runtime
echo_info "Detecting container runtime..."
if command -v podman &> /dev/null; then
    RUNTIME="podman"
    echo_info "Using Podman (recommended for better isolation)"
elif command -v docker &> /dev/null; then
    RUNTIME="docker"
    echo_warn "Using Docker. Consider installing Podman for better isolation:"
    echo_warn "  Run: scripts/install-podman.sh"
else
    echo_error "Neither Docker nor Podman found. Please install one of them."
    exit 1
fi
echo ""

# Get host IP address
echo_warn "Please enter your host machine's IP address (where inference services will run):"
echo_warn "Or press Enter to auto-detect..."
read -p "Host IP: " HOST_IP

if [ -z "$HOST_IP" ]; then
    # Auto-detect host IP
    HOST_IP=$(ip route get 1.1.1.1 | grep -oP 'src \K[^ ]+' 2>/dev/null || hostname -I | awk '{print $1}')
    echo_info "Auto-detected host IP: $HOST_IP"
fi

if [ -z "$HOST_IP" ]; then
    echo_error "Host IP is required"
    exit 1
fi

echo_info "Using host IP: $HOST_IP"
echo ""

# Step 1: Install K3s
echo_info "Step 1/7: Installing K3s..."
if command -v k3s &> /dev/null; then
    echo_warn "K3s already installed, skipping..."
else
    curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable traefik" sh -
    export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
    echo_info "K3s installed successfully"
fi

# Wait for K3s to be ready
echo_info "Waiting for K3s to be ready..."
sleep 10
kubectl wait --for=condition=ready node --all --timeout=120s

# Step 2: Install NGINX Ingress Controller
echo_info "Step 2/7: Installing NGINX Ingress Controller..."
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/baremetal/deploy.yaml
echo_info "Waiting for Ingress Controller to be ready..."
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s

# Step 3: Add Helm repositories
echo_info "Step 3/7: Adding Helm repositories..."
if ! command -v helm &> /dev/null; then
    echo_info "Installing Helm..."
    curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
fi

helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
echo_info "Helm repositories added"

# Step 4: Deploy infrastructure services
echo_info "Step 4/7: Deploying infrastructure services..."

# Deploy Qdrant
echo_info "Deploying Qdrant vector database..."
kubectl apply -f ../k8s/base/qdrant.yaml

# Deploy Redis
echo_info "Deploying Redis..."
helm install redis bitnami/redis \
  --set auth.enabled=false \
  --set master.persistence.enabled=false \
  --set master.resources.requests.memory=4Gi \
  --set master.resources.requests.cpu=1 \
  --wait --timeout=5m || echo_warn "Redis deployment may need more time"

# Deploy PostgreSQL
echo_info "Deploying PostgreSQL..."
helm install postgres bitnami/postgresql \
  --set primary.persistence.size=20Gi \
  --set primary.persistence.storageClass=local-path \
  --set primary.resources.requests.memory=8Gi \
  --set primary.resources.requests.cpu=2 \
  --set auth.database=aiplatform \
  --set auth.username=aiuser \
  --set auth.password=changeme \
  --wait --timeout=5m || echo_warn "PostgreSQL deployment may need more time"

# Deploy Monitoring
echo_info "Deploying Prometheus + Grafana..."
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=20Gi \
  --set prometheus.prometheusSpec.resources.requests.memory=4Gi \
  --set grafana.service.type=NodePort \
  --set grafana.service.nodePort=30001 \
  --wait --timeout=5m || echo_warn "Monitoring stack deployment may need more time"

echo_info "Infrastructure services deployed"

# Step 5: Start local container registry
echo_info "Step 5/7: Setting up local container registry..."
if [ "$($RUNTIME ps -q -f name=registry 2>/dev/null || $RUNTIME ps -q -f name=aeon-registry 2>/dev/null)" ]; then
    echo_warn "Container registry already running"
else
    $RUNTIME run -d -p 5000:5000 --restart=always --name registry registry:2
    echo_info "Container registry started on port 5000 using $RUNTIME"
fi

# Step 6: Build and push container images
echo_info "Step 6/7: Building and pushing container images with $RUNTIME..."

# Update API backend config with host IP
echo_info "Updating API backend configuration with host IP..."
sed -i "s/192.168.1.100/$HOST_IP/g" ../k8s/app/api-backend.yaml

# Build and push backend
echo_info "Building backend image..."
cd ../services
$RUNTIME build -t localhost:5000/aeon-api:latest .
$RUNTIME push localhost:5000/aeon-api:latest
echo_info "Backend image pushed"

# Build and push frontend
echo_info "Building frontend image..."
cd ../ui
$RUNTIME build -t localhost:5000/aeon-ui:latest .
$RUNTIME push localhost:5000/aeon-ui:latest
echo_info "Frontend image pushed"

cd ../scripts

# Step 7: Deploy application
echo_info "Step 7/7: Deploying Aeon application..."

kubectl apply -f ../k8s/app/api-backend.yaml
kubectl apply -f ../k8s/app/ui-frontend.yaml
kubectl apply -f ../k8s/jobs/nightly-optimize.yaml

echo_info "Waiting for application pods to be ready..."
kubectl wait --for=condition=ready pod -l app=api-backend --timeout=300s || echo_warn "Backend pods may need more time"
kubectl wait --for=condition=ready pod -l app=ui-frontend --timeout=300s || echo_warn "Frontend pods may need more time"

echo ""
echo_info "=== Setup Complete! ==="
echo ""
echo_info "Container Runtime: $RUNTIME"
echo ""
echo_info "Next steps:"
echo ""
echo "1. Start inference services with Podman (recommended for isolation):"
echo "   cd ../scripts"
echo "   ./podman-services.sh start-prod"
echo ""
echo "   OR start traditionally on host:"
echo "   cd ../inference"
echo "   ./start_vllm.sh  # In one terminal"
echo "   python3 embedding_server.py  # In another terminal"
echo ""
echo "2. Access the application:"
echo "   - Web UI: http://aeon.local (add '127.0.0.1 aeon.local' to /etc/hosts)"
echo "   - Grafana: http://localhost:30001 (admin/prom-operator)"
echo ""
echo "3. Check service status:"
echo "   kubectl get pods -A"
echo "   kubectl logs -f deployment/api-backend"
echo "   ./podman-services.sh status  # Check Podman services"
echo ""
echo_info "For development environment, run: ./podman-services.sh start-dev"
echo_info "Happy chatting with Cipher!"
