#!/bin/bash
# Development environment setup
# Sets up local development environment without K8s

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo_info() {
    echo -e "${GREEN}[Aeon::Dev]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[Aeon::Dev]${NC} $1"
}

echo_info "=== Aeon Development Environment Setup ==="
echo ""

# Check for UV
if ! command -v uv &> /dev/null; then
    echo_info "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Setup backend
echo_info "Setting up backend environment..."
cd ../services
if [ ! -d ".venv" ]; then
    uv venv
fi
source .venv/bin/activate
uv pip install -r requirements.txt
echo_info "Backend dependencies installed"

# Setup inference
echo_info "Setting up inference environment..."
cd ../inference
if [ ! -d ".venv" ]; then
    uv venv
fi
source .venv/bin/activate
uv pip install -r requirements.txt
echo_info "Inference dependencies installed"

# Setup frontend
echo_info "Setting up frontend environment..."
cd ../ui
if [ ! -d "node_modules" ]; then
    npm install
fi
echo_info "Frontend dependencies installed"

cd ../scripts

echo ""
echo_info "=== Development Setup Complete ==="
echo ""
echo_info "Start development services:"
echo ""
echo "1. Backend (FastAPI):"
echo "   cd services && source .venv/bin/activate"
echo "   uvicorn api.main:app --reload --host 0.0.0.0 --port 8080"
echo ""
echo "2. Frontend (React + Vite):"
echo "   cd ui"
echo "   npm run dev"
echo ""
echo "3. Embedding Server:"
echo "   cd inference && source .venv/bin/activate"
echo "   python embedding_server.py"
echo ""
echo "4. vLLM (requires GPU):"
echo "   cd inference"
echo "   ./start_vllm.sh"
echo ""
echo_warn "Note: You'll need Redis and PostgreSQL running for full functionality."
echo_warn "Use Docker Compose or run via K8s for complete local testing."
