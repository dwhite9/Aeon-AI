#!/bin/bash
# Build and push Docker images for Aeon services

set -e

# Color output
GREEN='\033[0;32m'
NC='\033[0m'

echo_info() {
    echo -e "${GREEN}[Aeon::Build]${NC} $1"
}

REGISTRY="${REGISTRY:-localhost:5000}"
TAG="${TAG:-latest}"

echo_info "=== Building Aeon Docker Images ==="
echo_info "Registry: $REGISTRY"
echo_info "Tag: $TAG"
echo ""

# Check if local registry is running
if [ "$REGISTRY" = "localhost:5000" ]; then
    if ! docker ps | grep -q "registry:2"; then
        echo_info "Starting local Docker registry..."
        docker run -d -p 5000:5000 --restart=always --name registry registry:2
    fi
fi

# Build backend
echo_info "Building backend image..."
cd ../services
docker build -t $REGISTRY/aeon-api:$TAG .
docker push $REGISTRY/aeon-api:$TAG
echo_info "Backend image: $REGISTRY/aeon-api:$TAG"

# Build frontend
echo_info "Building frontend image..."
cd ../ui
docker build -t $REGISTRY/aeon-ui:$TAG .
docker push $REGISTRY/aeon-ui:$TAG
echo_info "Frontend image: $REGISTRY/aeon-ui:$TAG"

# Build inference (embeddings)
echo_info "Building inference image..."
cd ../inference
docker build -f Dockerfile.embeddings -t $REGISTRY/aeon-embeddings:$TAG .
docker push $REGISTRY/aeon-embeddings:$TAG
echo_info "Embeddings image: $REGISTRY/aeon-embeddings:$TAG"

cd ../scripts

echo ""
echo_info "=== Build Complete ==="
echo_info "Images pushed to $REGISTRY"
echo ""
echo_info "To deploy, run:"
echo "  kubectl apply -f ../k8s/app/"
echo ""
echo_info "To restart deployments with new images:"
echo "  kubectl rollout restart deployment/api-backend"
echo "  kubectl rollout restart deployment/ui-frontend"
