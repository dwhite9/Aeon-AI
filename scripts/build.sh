#!/bin/bash
# Build and push container images for Aeon services
# Supports both Docker and Podman

set -e

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo_info() {
    echo -e "${GREEN}[Aeon::Build]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[Aeon::Build]${NC} $1"
}

# Determine container runtime
if [ -n "$CONTAINER_RUNTIME" ]; then
    RUNTIME="$CONTAINER_RUNTIME"
elif command -v podman &> /dev/null; then
    RUNTIME="podman"
    echo_info "Using Podman (recommended for better isolation)"
elif command -v docker &> /dev/null; then
    RUNTIME="docker"
    echo_warn "Using Docker. Consider switching to Podman for better isolation."
else
    echo_info "Neither Docker nor Podman found. Please install one of them."
    exit 1
fi

REGISTRY="${REGISTRY:-localhost:5000}"
TAG="${TAG:-latest}"

echo_info "=== Building Aeon Container Images ==="
echo_info "Runtime: $RUNTIME"
echo_info "Registry: $REGISTRY"
echo_info "Tag: $TAG"
echo ""

# Check if local registry is running
if [ "$REGISTRY" = "localhost:5000" ]; then
    if ! $RUNTIME ps | grep -q "registry:2\|aeon-registry"; then
        echo_info "Starting local container registry..."
        $RUNTIME run -d -p 5000:5000 --restart=always --name registry registry:2
    fi
fi

# Build backend
echo_info "Building backend image..."
cd ../services
$RUNTIME build -t $REGISTRY/aeon-api:$TAG .
$RUNTIME push $REGISTRY/aeon-api:$TAG
echo_info "Backend image: $REGISTRY/aeon-api:$TAG"

# Build frontend
echo_info "Building frontend image..."
cd ../ui
$RUNTIME build -t $REGISTRY/aeon-ui:$TAG .
$RUNTIME push $REGISTRY/aeon-ui:$TAG
echo_info "Frontend image: $REGISTRY/aeon-ui:$TAG"

# Build inference (embeddings)
echo_info "Building inference image..."
cd ../inference
$RUNTIME build -f Dockerfile.embeddings -t $REGISTRY/aeon-embeddings:$TAG .
$RUNTIME push $REGISTRY/aeon-embeddings:$TAG
echo_info "Embeddings image: $REGISTRY/aeon-embeddings:$TAG"

cd ../scripts

echo ""
echo_info "=== Build Complete ==="
echo_info "Images pushed to $REGISTRY with $RUNTIME"
echo ""
echo_info "To deploy, run:"
echo "  kubectl apply -f ../k8s/app/"
echo ""
echo_info "To restart deployments with new images:"
echo "  kubectl rollout restart deployment/api-backend"
echo "  kubectl rollout restart deployment/ui-frontend"
