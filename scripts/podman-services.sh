#!/bin/bash
# Aeon Platform - Podman Services Management Script
# This script manages Podman-based services for better isolation

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${GREEN}[Aeon::Podman]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[Aeon::Podman]${NC} $1"
}

echo_error() {
    echo -e "${RED}[Aeon::Podman]${NC} $1"
}

echo_step() {
    echo -e "${BLUE}[Aeon::Podman]${NC} $1"
}

# Check for podman-compose
check_podman_compose() {
    if ! command -v podman-compose &> /dev/null; then
        echo_error "podman-compose not found. Installing..."
        pip3 install podman-compose
    fi
}

# Get host IP address (needed for K8s pods to reach Podman services)
get_host_ip() {
    # Try to get the primary interface IP
    HOST_IP=$(ip route get 1.1.1.1 | grep -oP 'src \K[^ ]+' 2>/dev/null || echo "")

    if [ -z "$HOST_IP" ]; then
        # Fallback: use hostname -I
        HOST_IP=$(hostname -I | awk '{print $1}')
    fi

    echo "$HOST_IP"
}

# Start production inference services (vLLM + Embeddings + Registry)
start_prod() {
    echo_info "Starting production inference services..."
    check_podman_compose

    cd "$(dirname "$0")/.."

    echo_step "Starting vLLM, Embeddings, and Registry..."
    podman-compose up -d vllm-server embedding-server registry

    echo ""
    echo_info "Production services started!"
    echo ""
    echo "Services:"
    echo "  - vLLM Server: http://localhost:8000 (172.20.0.10:8000)"
    echo "  - Embedding Server: http://localhost:8001 (172.20.0.11:8001)"
    echo "  - Container Registry: http://localhost:5000 (172.20.0.20:5000)"
    echo ""
    echo "Access from K8s pods using: http://$(get_host_ip):PORT"
    echo ""
}

# Start development environment (all services)
start_dev() {
    echo_info "Starting development environment..."
    check_podman_compose

    cd "$(dirname "$0")/.."

    echo_step "Starting all development services..."
    podman-compose --profile dev up -d

    echo ""
    echo_info "Development environment started!"
    echo ""
    echo "Services:"
    echo "  - vLLM Server: http://localhost:8000"
    echo "  - Embedding Server: http://localhost:8001"
    echo "  - Redis: localhost:6379"
    echo "  - PostgreSQL: localhost:5432"
    echo "  - Qdrant: http://localhost:6333"
    echo "  - Backend API: http://localhost:8080"
    echo "  - Frontend UI: http://localhost:3000"
    echo "  - Container Registry: http://localhost:5000"
    echo ""
}

# Stop all services
stop() {
    echo_info "Stopping all Aeon services..."
    cd "$(dirname "$0")/.."

    podman-compose --profile dev down

    echo_info "All services stopped"
}

# Show service status
status() {
    echo_info "Aeon Service Status"
    echo ""

    cd "$(dirname "$0")/.."
    podman-compose ps

    echo ""
    echo_info "Network Information:"
    podman network ls | grep aeon || echo "No Aeon networks found"

    echo ""
    echo_info "Volume Information:"
    podman volume ls | grep aeon || echo "No Aeon volumes found"
}

# Show logs
logs() {
    local service=$1
    cd "$(dirname "$0")/.."

    if [ -z "$service" ]; then
        echo_info "Showing logs for all services..."
        podman-compose logs -f
    else
        echo_info "Showing logs for $service..."
        podman-compose logs -f "$service"
    fi
}

# Restart services
restart() {
    local service=$1
    cd "$(dirname "$0")/.."

    if [ -z "$service" ]; then
        echo_info "Restarting all services..."
        podman-compose restart
    else
        echo_info "Restarting $service..."
        podman-compose restart "$service"
    fi
}

# Build images
build() {
    echo_info "Building Aeon images with Podman..."
    cd "$(dirname "$0")/.."

    echo_step "Building embedding server..."
    podman-compose build embedding-server

    echo_step "Building backend API..."
    podman-compose build backend-dev

    echo_step "Building frontend UI..."
    podman-compose build frontend-dev

    echo_info "All images built successfully"
}

# Push images to local registry
push() {
    echo_info "Pushing images to local registry..."

    # Wait for registry to be ready
    echo_step "Waiting for registry to be ready..."
    timeout 30 bash -c 'until curl -f http://localhost:5000/v2/ &>/dev/null; do sleep 1; done' || {
        echo_error "Registry not available"
        exit 1
    }

    echo_step "Pushing images..."
    podman push localhost:5000/aeon-embeddings:latest
    podman push localhost:5000/aeon-api:latest
    podman push localhost:5000/aeon-ui:latest

    echo_info "Images pushed successfully"
}

# Clean up everything
cleanup() {
    echo_warn "This will remove all Aeon containers, networks, and volumes!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$(dirname "$0")/.."

        echo_step "Stopping and removing containers..."
        podman-compose --profile dev down -v

        echo_step "Removing networks..."
        podman network rm aeon-network 2>/dev/null || true

        echo_step "Removing volumes..."
        podman volume rm aeon_vllm-cache aeon_embedding-cache aeon_registry-data 2>/dev/null || true

        echo_info "Cleanup complete"
    else
        echo_info "Cleanup cancelled"
    fi
}

# Health check
health() {
    echo_info "Checking service health..."
    echo ""

    # Check vLLM
    echo_step "vLLM Server:"
    curl -sf http://localhost:8000/health &>/dev/null && echo "  ✓ Healthy" || echo "  ✗ Unhealthy"

    # Check Embeddings
    echo_step "Embedding Server:"
    curl -sf http://localhost:8001/health &>/dev/null && echo "  ✓ Healthy" || echo "  ✗ Unhealthy"

    # Check Registry
    echo_step "Container Registry:"
    curl -sf http://localhost:5000/v2/ &>/dev/null && echo "  ✓ Healthy" || echo "  ✗ Unhealthy"

    # Check dev services if running
    if podman ps --filter "label=aeon.env=development" --format "{{.Names}}" | grep -q aeon-redis-dev; then
        echo_step "Redis (dev):"
        podman exec aeon-redis-dev redis-cli ping &>/dev/null && echo "  ✓ Healthy" || echo "  ✗ Unhealthy"

        echo_step "PostgreSQL (dev):"
        podman exec aeon-postgres-dev pg_isready &>/dev/null && echo "  ✓ Healthy" || echo "  ✗ Unhealthy"

        echo_step "Qdrant (dev):"
        curl -sf http://localhost:6333/healthz &>/dev/null && echo "  ✓ Healthy" || echo "  ✗ Unhealthy"

        echo_step "Backend API (dev):"
        curl -sf http://localhost:8080/health &>/dev/null && echo "  ✓ Healthy" || echo "  ✗ Unhealthy"

        echo_step "Frontend UI (dev):"
        curl -sf http://localhost:3000/ &>/dev/null && echo "  ✓ Healthy" || echo "  ✗ Unhealthy"
    fi

    echo ""
}

# Show usage
usage() {
    cat << EOF
Aeon Platform - Podman Services Management

Usage: $0 <command> [options]

Commands:
  start-prod        Start production inference services (vLLM, Embeddings, Registry)
  start-dev         Start full development environment (all services)
  stop              Stop all services
  restart [service] Restart all services or specific service
  status            Show service status
  logs [service]    Show logs for all services or specific service
  build             Build all container images
  push              Push images to local registry
  health            Check health of all services
  cleanup           Remove all containers, networks, and volumes
  help              Show this help message

Examples:
  $0 start-prod           # Start production services only
  $0 start-dev            # Start full dev environment
  $0 logs vllm-server     # Show vLLM logs
  $0 restart backend-dev  # Restart backend service
  $0 health               # Check all service health

For K8s deployment, use: scripts/setup.sh (with Podman support)

EOF
}

# Main command dispatcher
case "${1:-}" in
    start-prod)
        start_prod
        ;;
    start-dev)
        start_dev
        ;;
    stop)
        stop
        ;;
    restart)
        restart "$2"
        ;;
    status)
        status
        ;;
    logs)
        logs "$2"
        ;;
    build)
        build
        ;;
    push)
        push
        ;;
    health)
        health
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        echo_error "Unknown command: ${1:-}"
        echo ""
        usage
        exit 1
        ;;
esac
