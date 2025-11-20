# Aeon Architecture Diagrams

This directory contains Mermaid diagrams auto-generated from the Structurizr DSL model ([workspace.dsl](../workspace.dsl)).

## Available Diagrams

### System Context
**File**: [structurizr-SystemContext.mmd](structurizr-SystemContext.mmd)

Shows the high-level system context - how users interact with the Aeon platform and its relationship with external systems.

```mermaid
graph TB
    linkStyle default fill:#ffffff
```

### Container Architecture
**File**: [structurizr-Containers.mmd](structurizr-Containers.mmd)

Complete container-level view showing all components of the Aeon platform including:
- Host machine services (vLLM, embeddings, trainer)
- K3s application layer (UI, API, Agent)
- Data layer (Qdrant, PostgreSQL, Redis)
- Supporting services (SearXNG, monitoring, background jobs)

### Host Services
**File**: [structurizr-HostServices.mmd](structurizr-HostServices.mmd)

Focused view of GPU-accelerated services running on the host machine:
- vLLM Server (Mistral 7B inference)
- Embedding Server (all-MiniLM-L6-v2)
- Training Pipeline (embedding fine-tuning)

### K3s Services
**File**: [structurizr-K3sServices.mmd](structurizr-K3sServices.mmd)

Kubernetes-orchestrated application services including the React UI, FastAPI backend, LangGraph agent, databases, and supporting services.

### API Components
**File**: [structurizr-APIComponents.mmd](structurizr-APIComponents.mmd)

Internal component structure of the FastAPI backend:
- Chat API endpoint
- RAG retrieval engine
- Semantic cache layer
- Tool manager
- Analytics logger

### Agent Components
**File**: [structurizr-AgentComponents.mmd](structurizr-AgentComponents.mmd)

LangGraph agent (Cipher) internal components:
- State graph workflow
- Tool executor
- Web search, RAG, and code execution tools

### Deployment Architecture
**File**: [structurizr-Deployment.mmd](structurizr-Deployment.mmd)

Physical deployment showing how containers are distributed across:
- Host machine (Docker containers for GPU workloads)
- K3s cluster pods (application, data, service, and job pods)

## Viewing Diagrams

### In GitHub
GitHub automatically renders `.mmd` files. Click any diagram file above to view it.

### In VS Code
Install the [Mermaid Preview](https://marketplace.visualstudio.com/items?itemName=vstirbu.vscode-mermaid-preview) extension.

### Online
Copy the diagram content and paste it into [Mermaid Live Editor](https://mermaid.live/).

## Regenerating Diagrams

To regenerate these diagrams after modifying `workspace.dsl`:

```bash
# Using Podman
podman run --rm -v "$(pwd):/workspace:Z" docker.io/structurizr/cli export -workspace /workspace/workspace.dsl -format mermaid -output /workspace/diagrams

# Using Docker
docker run --rm -v "$(pwd):/workspace" docker.io/structurizr/cli export -workspace /workspace/workspace.dsl -format mermaid -output /workspace/diagrams
```

## Architecture Documentation

For detailed architecture documentation, see:
- [workspace.dsl](../workspace.dsl) - Complete Structurizr DSL model
- [README.md](../README.md) - Project overview
