workspace "Aeon Project" "Personal AI Research Platform with self-tuning capabilities" {

    model {
        user = person "Researcher" "Developer using Aeon for AI research and coding assistance"

        aeon = softwareSystem "Aeon Platform" "Kubernetes-native AI agentic platform with RAG, caching, and self-tuning" {
            # Host Machine Components (GPU)
            vllm = container "vLLM Server" "Mistral 7B Instruct (8-bit)" "Docker" "Serves LLM inference with GPU acceleration" {
                tags "Host" "GPU"
            }

            embedding = container "Embedding Server" "all-MiniLM-L6-v2" "FastAPI + Docker" "Generates text embeddings for semantic search" {
                tags "Host" "GPU"
            }

            trainer = container "Training Pipeline" "PyTorch" "Docker" "Fine-tunes embeddings based on feedback" {
                tags "Host" "GPU" "Batch"
            }

            # K3s Application Layer
            ui = container "React UI" "Chat interface, code editor, analytics" "React + TypeScript" "Web-based user interface for Cipher" {
                tags "K3s" "Frontend"
            }

            api = container "FastAPI Backend" "REST + WebSocket API" "Python + FastAPI" "Main application API and orchestration" {
                chatEndpoint = component "Chat API" "Handles chat requests" "FastAPI Router"
                retriever = component "RAG Retrieval" "Retrieves relevant context" "Python"
                cacheLayer = component "Semantic Cache" "Caches queries and responses" "Python"
                toolManager = component "Tool Manager" "Manages tool execution" "Python"
                analytics = component "Analytics Logger" "Logs queries and feedback" "Python"

                tags "K3s" "Backend"
            }

            agent = container "LangGraph Agent (Cipher)" "Multi-tool agent orchestration" "LangGraph + LangChain" "Decision-making and workflow execution" {
                stateGraph = component "State Graph" "Workflow state machine" "LangGraph"
                toolExecutor = component "Tool Executor" "Executes selected tools" "LangChain"
                webSearchTool = component "Web Search Tool" "Searches internet via SearXNG" "Python"
                ragTool = component "RAG Search Tool" "Searches local documents" "Python"
                codeTool = component "Code Execution Tool" "Executes code in K8s Jobs" "Python"

                tags "K3s" "Agent"
            }

            # K3s Data Layer
            qdrant = container "Qdrant" "Vector database" "Qdrant" "Stores document embeddings and semantic cache" {
                tags "K3s" "Database"
            }

            postgres = container "PostgreSQL" "Relational database" "PostgreSQL" "Stores metadata, analytics, and feedback" {
                tags "K3s" "Database"
            }

            redis = container "Redis" "In-memory cache" "Redis" "Session state and response cache" {
                tags "K3s" "Cache"
            }

            # K3s Supporting Services
            searxng = container "SearXNG" "Web search engine" "SearXNG" "Privacy-focused web search aggregator" {
                tags "K3s" "Tool"
            }

            docProcessor = container "Document Processor" "Indexing pipeline" "Python CronJob" "Extracts, chunks, and indexes documents" {
                tags "K3s" "Batch"
            }

            codeExecutor = container "Code Executor" "Sandboxed execution" "Kubernetes Jobs" "Executes code in isolated pods" {
                tags "K3s" "Tool"
            }

            optimizer = container "Optimizer" "Self-tuning system" "Python CronJob" "Analyzes usage and optimizes RAG pipeline" {
                tags "K3s" "Batch"
            }

            monitoring = container "Prometheus + Grafana" "Metrics and monitoring" "Prometheus Stack" "System observability and dashboards" {
                tags "K3s" "Observability"
            }
        }

        internet = softwareSystem "Internet" "External web resources" {
            tags "External"
        }

        # Relationships - User
        user -> ui "Interacts with Cipher via"
        user -> monitoring "Views metrics and analytics"

        # Relationships - UI
        ui -> api "Makes API calls to" "HTTPS/WSS"

        # Relationships - API Internal
        api -> agent "Delegates requests to"
        api -> redis "Reads/writes session state"
        api -> postgres "Logs analytics"
        chatEndpoint -> retriever "Requests context from"
        chatEndpoint -> cacheLayer "Checks cache"
        retriever -> qdrant "Searches vectors in"
        cacheLayer -> redis "Stores responses in"
        cacheLayer -> qdrant "Stores semantic cache in"
        analytics -> postgres "Writes logs to"

        # Relationships - Agent
        agent -> vllm "Calls for completions" "HTTP"
        agent -> searxng "Searches web via"
        agent -> qdrant "Retrieves documents from"
        agent -> codeExecutor "Executes code via"
        stateGraph -> toolExecutor "Routes to"
        toolExecutor -> webSearchTool "Executes"
        toolExecutor -> ragTool "Executes"
        toolExecutor -> codeTool "Executes"
        webSearchTool -> searxng "Uses"
        ragTool -> qdrant "Queries"
        codeTool -> codeExecutor "Triggers"

        # Relationships - Embeddings
        api -> embedding "Generates embeddings" "HTTP"
        docProcessor -> embedding "Generates embeddings" "HTTP"
        optimizer -> embedding "Generates embeddings" "HTTP"

        # Relationships - Document Processing
        docProcessor -> qdrant "Indexes chunks to"
        docProcessor -> postgres "Stores metadata in"

        # Relationships - Code Execution
        codeExecutor -> postgres "Stores execution logs" "Optional"

        # Relationships - Optimization
        optimizer -> postgres "Reads analytics from"
        optimizer -> qdrant "Optimizes indices in"
        optimizer -> trainer "Triggers fine-tuning"
        trainer -> embedding "Updates model"

        # Relationships - External
        searxng -> internet "Searches"

        # Relationships - Monitoring
        monitoring -> api "Scrapes metrics from"
        monitoring -> qdrant "Scrapes metrics from"
        monitoring -> postgres "Scrapes metrics from"

        # Deployment
        deploymentEnvironment "HomeLab" {
            deploymentNode "Host Machine" "AMD Ryzen 9 7950X3D, RTX 4070 Ti (12GB VRAM), 56GB RAM" "Physical Hardware" {
                deploymentNode "Docker" "Container Runtime" "Docker" {
                    vllmInstance = containerInstance vllm
                    embeddingInstance = containerInstance embedding
                    trainerInstance = containerInstance trainer
                }
            }

            deploymentNode "K3s Cluster" "16 cores, 48GB RAM allocated" "Single-node K3s" {
                deploymentNode "Application Pods" {
                    uiInstance = containerInstance ui
                    apiInstance = containerInstance api
                    agentInstance = containerInstance agent
                }

                deploymentNode "Data Pods" {
                    qdrantInstance = containerInstance qdrant
                    postgresInstance = containerInstance postgres
                    redisInstance = containerInstance redis
                }

                deploymentNode "Service Pods" {
                    searxngInstance = containerInstance searxng
                    monitoringInstance = containerInstance monitoring
                }

                deploymentNode "Job Pods" {
                    docProcessorInstance = containerInstance docProcessor
                    codeExecutorInstance = containerInstance codeExecutor
                    optimizerInstance = containerInstance optimizer
                }
            }
        }
    }

    views {
        systemContext aeon "SystemContext" {
            include *
            autoLayout
        }

        container aeon "Containers" {
            include *
            autoLayout
        }

        container aeon "HostServices" {
            include element.tag==Host
            include user
            autoLayout
            description "GPU-accelerated services on host machine"
        }

        container aeon "K3sServices" {
            include element.tag==K3s
            include user
            autoLayout
            description "Kubernetes-orchestrated application services"
        }

        component api "APIComponents" {
            include *
            autoLayout
        }

        component agent "AgentComponents" {
            include *
            autoLayout
        }

        deployment aeon "HomeLab" "Deployment" {
            include *
            autoLayout
        }

        styles {
            element "Software System" {
                background #1168bd
                color #ffffff
            }
            element "Person" {
                shape person
                background #08427b
                color #ffffff
            }
            element "Container" {
                background #438dd5
                color #ffffff
            }
            element "Component" {
                background #85bbf0
                color #000000
            }
            element "External" {
                background #999999
                color #ffffff
            }
            element "Host" {
                background #ff6b35
                color #ffffff
            }
            element "GPU" {
                background #d32f2f
                color #ffffff
            }
            element "K3s" {
                background #326ce5
                color #ffffff
            }
            element "Database" {
                shape cylinder
            }
            element "Cache" {
                shape pipe
            }
            element "Batch" {
                shape component
            }
            element "Frontend" {
                shape WebBrowser
            }
        }

        themes default
    }

}
