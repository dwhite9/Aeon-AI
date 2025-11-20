# Aeon

> Self-hosted AI platform for home lab environments with intelligent agent capabilities

## Overview

Aeon is a privacy-focused, self-hosted AI platform designed to run on Kubernetes (K3s) in home lab environments. It features **Cipher**, an intelligent AI agent with RAG (Retrieval-Augmented Generation) capabilities, web search integration, code execution, and self-tuning optimization.

## Key Features

- **Local LLM**: Mistral 7B Instruct (8-bit quantized) via vLLM
- **RAG Pipeline**: Semantic document chunking with Qdrant vector database
- **Intelligent Agent**: Multi-tool agent using LangGraph (web search, document retrieval, code execution)
- **Two-Tier Caching**: Redis exact cache + Qdrant semantic cache for fast responses
- **Code Execution**: Sandboxed Python execution via Kubernetes Jobs
- **Self-Tuning**: Automated nightly optimization and weekly embedding fine-tuning
- **Privacy-First**: All processing happens locally on your infrastructure

## Tech Stack

- **Backend**: FastAPI (async/await)
- **Frontend**: React + TypeScript
- **LLM Serving**: vLLM with Mistral 7B Instruct
- **Vector Database**: Qdrant
- **Cache & Sessions**: Redis
- **Analytics DB**: PostgreSQL
- **Orchestration**: K3s (lightweight Kubernetes)
- **Monitoring**: Prometheus + Grafana

## Architecture

Aeon uses a hybrid deployment model with GPU-intensive services on the host machine and application services in Kubernetes:

- **Host Services** (GPU): vLLM inference server, embedding generation, model fine-tuning
- **K3s Services**: React UI, FastAPI backend, LangGraph agent, databases, monitoring

View detailed architecture diagrams in the [diagrams/](diagrams/) directory:
- [System Context](diagrams/structurizr-SystemContext.mmd) - High-level overview
- [Container Architecture](diagrams/structurizr-Containers.mmd) - Complete system structure
- [Host Services](diagrams/structurizr-HostServices.mmd) - GPU workloads
- [K3s Services](diagrams/structurizr-K3sServices.mmd) - Kubernetes deployments
- [API Components](diagrams/structurizr-APIComponents.mmd) - Backend internals
- [Agent Components](diagrams/structurizr-AgentComponents.mmd) - LangGraph agent structure
- [Deployment](diagrams/structurizr-Deployment.mmd) - Physical infrastructure

## Resource Requirements

**Recommended Single VM Configuration:**
- CPU: 16 cores (14 allocated, 2 for host)
- RAM: 56GB (48GB allocated, 8GB for host)
- GPU: NVIDIA GPU with 8GB+ VRAM
- Storage: 100GB SSD (thin provisioned)

## Project Status

🚧 **Planning Phase** - Repository structure and initial implementation in progress.

## Quick Start

Coming Soon

## License

MIT

## Contributing

This is currently a personal research project to experiement and learn more about AI Systems Archetecture. Contributions welcome once the initial implementation is complete.
