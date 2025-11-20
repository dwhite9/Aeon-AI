#!/bin/bash
# Cleanup script - removes all Aeon deployments

set -e

RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo_warn() {
    echo -e "${YELLOW}[Aeon::Cleanup]${NC} $1"
}

echo_error() {
    echo -e "${RED}[Aeon::Cleanup]${NC} $1"
}

echo_warn "=== Aeon Platform Cleanup ==="
echo ""
echo_error "WARNING: This will remove all Aeon deployments and data!"
echo_warn "This will delete:"
echo "  - All K8s deployments and services"
echo "  - All Helm releases (Redis, PostgreSQL, Prometheus)"
echo "  - All persistent volumes (data will be lost)"
echo "  - Local Docker registry container"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cleanup cancelled"
    exit 0
fi

echo ""
echo_warn "Starting cleanup..."

# Delete application deployments
echo_warn "Removing application deployments..."
kubectl delete -f ../k8s/app/ --ignore-not-found=true
kubectl delete -f ../k8s/jobs/ --ignore-not-found=true

# Delete infrastructure
echo_warn "Removing infrastructure services..."
kubectl delete -f ../k8s/base/ --ignore-not-found=true

# Uninstall Helm releases
echo_warn "Uninstalling Helm releases..."
helm uninstall redis --ignore-not-found 2>/dev/null || true
helm uninstall postgres --ignore-not-found 2>/dev/null || true
helm uninstall kube-prometheus-stack --ignore-not-found 2>/dev/null || true

# Delete PVCs
echo_warn "Deleting persistent volume claims..."
kubectl delete pvc --all --ignore-not-found=true

# Stop local Docker registry
echo_warn "Stopping local Docker registry..."
docker stop registry 2>/dev/null || true
docker rm registry 2>/dev/null || true

# Stop host services
echo_warn "Stopping host services..."
cd ../inference
docker-compose down 2>/dev/null || true
cd ../scripts

echo ""
echo_warn "=== Cleanup Complete ==="
echo ""
echo_warn "To completely remove K3s, run:"
echo "  sudo /usr/local/bin/k3s-uninstall.sh"
echo ""
echo_warn "To remove Docker images, run:"
echo "  docker rmi localhost:5000/aeon-api:latest"
echo "  docker rmi localhost:5000/aeon-ui:latest"
echo "  docker rmi localhost:5000/aeon-embeddings:latest"
