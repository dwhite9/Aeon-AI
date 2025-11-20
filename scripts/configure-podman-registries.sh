#!/bin/bash
# Configure Podman to use Docker Hub for unqualified image names

set -e

GREEN='\033[0;32m'
NC='\033[0m'

echo_info() {
    echo -e "${GREEN}[Aeon::Podman]${NC} $1"
}

echo_info "Configuring Podman registries for Docker Hub compatibility..."

# Check if already configured (ignore commented lines)
if grep -q "^unqualified-search-registries" /etc/containers/registries.conf 2>/dev/null; then
    echo_info "Registries already configured"
    exit 0
fi

# Add Docker Hub to unqualified-search registries
cat >> /etc/containers/registries.conf << 'EOF'

# Added by Aeon installation - Enable Docker Hub short names
unqualified-search-registries = ["docker.io"]
EOF

echo_info "Docker Hub added to unqualified-search registries"
echo_info "You can now use short image names like 'nvidia/cuda:12.1.0-runtime-ubuntu22.04'"
