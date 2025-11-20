#!/bin/bash
# Install Podman and podman-compose on Ubuntu/Debian-based systems
# This provides better container isolation than Docker

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

# Check if running on Ubuntu/Debian
if [ ! -f /etc/os-release ]; then
    echo_error "Cannot detect OS. This script is for Ubuntu/Debian systems."
    exit 1
fi

. /etc/os-release
if [[ "$ID" != "ubuntu" && "$ID" != "debian" ]]; then
    echo_warn "This script is designed for Ubuntu/Debian. You're running: $ID"
    echo_warn "Installation may not work correctly."
fi

echo_info "=== Podman Installation for Aeon AI Platform ==="
echo ""

# Check if Podman is already installed
if command -v podman &> /dev/null; then
    PODMAN_VERSION=$(podman --version | awk '{print $3}')
    REQUIRED_VERSION="4.9"

    # Check if version meets minimum requirement
    if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PODMAN_VERSION" | sort -V | head -n1)" == "$REQUIRED_VERSION" ]]; then
        echo_info "Podman $PODMAN_VERSION is already installed (>= $REQUIRED_VERSION required)"
        echo_info "Skipping Podman installation. Continuing with remaining setup..."
        SKIP_PODMAN_INSTALL=true
    else
        echo_warn "Podman $PODMAN_VERSION is installed but version $REQUIRED_VERSION+ is required"
        echo_info "Upgrading Podman..."
        SKIP_PODMAN_INSTALL=false
    fi
else
    SKIP_PODMAN_INSTALL=false
fi

if [ "$SKIP_PODMAN_INSTALL" = false ]; then
    # Update package list
    echo_step "Step 1/5: Updating package list..."

    # Clean up broken NVIDIA repo if it exists
    if [ -f /etc/apt/sources.list.d/nvidia-container-toolkit.list ]; then
        echo_info "Cleaning up broken NVIDIA repository..."
        sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list
    fi

    sudo apt-get update

    # Install prerequisites
    echo_step "Step 2/5: Installing prerequisites..."
    sudo apt-get install -y \
        curl \
        wget \
        ca-certificates \
        gnupg \
        lsb-release \
        software-properties-common

    # Add Podman repository (for newer versions)
    echo_step "Step 3/5: Adding Podman repository..."

    # For Ubuntu 24.04+, use default repository (includes Podman 4.9+)
    if [[ "$VERSION_ID" == "24.04" ]]; then
        echo_info "Ubuntu 24.04 detected. Using default repository (Podman 4.9+)..."
        sudo apt-get update
    # For older Ubuntu versions, use Kubic repository
    elif [[ "$VERSION_ID" == "20.04" || "$VERSION_ID" == "22.04" ]]; then
        echo_info "Adding Kubic Podman repository for Ubuntu $VERSION_ID..."
        echo "deb https://download.opensuse.org/repositories/devel:/kubic:/libcontainers:/stable/xUbuntu_${VERSION_ID}/ /" | \
            sudo tee /etc/apt/sources.list.d/devel:kubic:libcontainers:stable.list
        curl -fsSL https://download.opensuse.org/repositories/devel:kubic:libcontainers:stable/xUbuntu_${VERSION_ID}/Release.key | \
            gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/kubic-libcontainers.gpg > /dev/null
        sudo apt-get update
    else
        echo_warn "Ubuntu version $VERSION_ID detected. Using default repository."
        sudo apt-get update
    fi

    # Install Podman
    echo_step "Step 4/5: Installing Podman..."
    sudo apt-get install -y podman

    # Verify installation
    PODMAN_VERSION=$(podman --version | awk '{print $3}')
    echo_info "Podman installed successfully: version $PODMAN_VERSION"
fi

# Install podman-compose
echo_step "Step 5/5: Installing podman-compose..."

# Install using UV in a virtual environment
echo_info "Installing podman-compose via UV in a venv..."

# Ensure UV is installed
if ! command -v uv &> /dev/null; then
    echo_info "UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Check if we're already in a venv
if [ -n "$VIRTUAL_ENV" ]; then
    echo_info "Using existing virtual environment: $VIRTUAL_ENV"
    uv pip install podman-compose
else
    # Create venv and install podman-compose
    VENV_PATH="/opt/aeon/venv/podman-compose"
    echo_info "Creating venv at $VENV_PATH..."
    sudo mkdir -p /opt/aeon/venv
    sudo uv venv "$VENV_PATH"
    sudo uv pip install --python "$VENV_PATH" podman-compose

    # Create symlink to make podman-compose available system-wide
    echo_info "Creating symlink for podman-compose..."
    sudo ln -sf "$VENV_PATH/bin/podman-compose" /usr/local/bin/podman-compose
fi

# Verify podman-compose installation
if command -v podman-compose &> /dev/null; then
    COMPOSE_VERSION=$(podman-compose --version 2>/dev/null || echo "installed")
    echo_info "podman-compose installed successfully: $COMPOSE_VERSION"
else
    echo_error "podman-compose installation failed"
    exit 1
fi

# Configure Podman for NVIDIA GPU support (if NVIDIA GPU is present)
echo_step "Configuring GPU support..."
if command -v nvidia-smi &> /dev/null; then
    echo_info "NVIDIA GPU detected. Setting up GPU support..."

    # Install nvidia-container-toolkit for Podman
    if ! command -v nvidia-ctk &> /dev/null; then
        echo_step "Installing NVIDIA Container Toolkit..."

        # Add NVIDIA repository using the new generic deb repository
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit

        # Configure Podman to use NVIDIA runtime
        sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

        echo_info "NVIDIA Container Toolkit installed and configured"
    else
        echo_info "NVIDIA Container Toolkit already installed"
    fi

    # Test GPU access
    echo_step "Testing GPU access with Podman..."
    if podman run --rm --device nvidia.com/gpu=all docker.io/nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        echo_info "✓ GPU access working correctly!"
    else
        echo_warn "⚠ GPU test failed. You may need to configure GPU support manually."
    fi
else
    echo_warn "No NVIDIA GPU detected. Skipping GPU configuration."
fi

# Configure rootless Podman (optional but recommended)
echo_step "Configuring rootless Podman..."
if [ ! -f "$HOME/.config/containers/storage.conf" ]; then
    mkdir -p "$HOME/.config/containers"
    cat > "$HOME/.config/containers/storage.conf" << 'EOF'
[storage]
driver = "overlay"

[storage.options]
mount_program = "/usr/bin/fuse-overlayfs"
EOF
    echo_info "Rootless Podman storage configured"
fi

# Enable Podman socket for Docker compatibility (optional)
echo_step "Setting up Docker compatibility..."
systemctl --user enable --now podman.socket 2>/dev/null || echo_warn "Unable to enable Podman socket (may require reboot)"

# Create Docker alias (optional)
read -p "Create 'docker' alias for Podman? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if ! grep -q "alias docker=podman" "$HOME/.bashrc"; then
        echo "alias docker=podman" >> "$HOME/.bashrc"
        echo_info "Added 'docker' alias to ~/.bashrc"
        echo_warn "Run 'source ~/.bashrc' or restart your terminal to use the alias"
    else
        echo_info "Docker alias already exists"
    fi
fi

echo ""
echo_info "=== Podman Installation Complete! ==="
echo ""
echo_info "Podman Version: $PODMAN_VERSION"
echo_info "podman-compose: Installed"
if command -v nvidia-smi &> /dev/null; then
    echo_info "GPU Support: Configured"
fi
echo ""
echo_info "Next steps:"
echo "  1. Verify installation: podman --version"
echo "  2. Test GPU access: podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi"
echo "  3. Start Aeon services: cd scripts && ./podman-services.sh start-prod"
echo ""
echo_info "For full Aeon setup, run: cd scripts && sudo ./setup.sh"
echo ""
