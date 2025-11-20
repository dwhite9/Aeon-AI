"""
Aeon API Backend
Main FastAPI application with chat endpoints, RAG, and session management
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import redis.asyncio as redis
import json
import logging
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from pathlib import Path
import tempfile
import shutil
import time
import hashlib

# Import RAG components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from rag import RAGPipeline, QdrantVectorStore, RAGAnalytics, process_and_chunk_document

# Import Agent components
from agent import CipherAgent

# Import Code Execution components
from code_exec import CodeExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[Aeon::API] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://192.168.1.100:8000/v1")
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "http://192.168.1.100:8001")
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://qdrant.vector-db:6333")
REDIS_HOST = os.getenv("REDIS_HOST", "redis-master")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
SEARXNG_ENDPOINT = os.getenv("SEARXNG_ENDPOINT", "http://searxng.search-engine:8080")
SESSION_TTL = 3600  # 1 hour
ENABLE_RAG = os.getenv("ENABLE_RAG", "true").lower() == "true"
ENABLE_AGENT = os.getenv("ENABLE_AGENT", "true").lower() == "true"
ENABLE_CODE_EXEC = os.getenv("ENABLE_CODE_EXEC", "true").lower() == "true"

# Initialize FastAPI app
app = FastAPI(
    title="Aeon AI Platform API",
    description="API for Cipher AI agent with RAG capabilities, web search, code execution, and intelligent routing",
    version="0.4.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global clients (initialized on startup)
redis_client: Optional[redis.Redis] = None
rag_pipeline: Optional[RAGPipeline] = None
rag_analytics: Optional[RAGAnalytics] = None
cipher_agent: Optional[CipherAgent] = None
code_executor: Optional[CodeExecutor] = None


# Pydantic models
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024


class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str


class SessionInfo(BaseModel):
    session_id: str
    message_count: int
    created_at: str
    last_activity: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    status: str


class RAGQueryRequest(BaseModel):
    query: str
    limit: int = 5
    score_threshold: float = 0.7
    use_cache: bool = True


class RAGQueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    count: int
    source: str  # "cache" or "vector_db"
    timestamp: str


class RAGChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_rag: bool = True
    rag_limit: int = 3
    temperature: float = 0.7
    max_tokens: int = 1024


class RAGChatResponse(BaseModel):
    response: str
    session_id: str
    context_used: List[Dict[str, Any]]
    timestamp: str


class AgentRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class AgentResponse(BaseModel):
    response: str
    session_id: str
    tool_used: str
    metadata: Dict[str, Any]
    timestamp: str


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize Redis, RAG pipeline, Code Executor, and Cipher agent on startup"""
    global redis_client, rag_pipeline, rag_analytics, cipher_agent, code_executor

    try:
        # Initialize Redis
        redis_client = await redis.from_url(
            f"redis://{REDIS_HOST}:{REDIS_PORT}",
            encoding="utf-8",
            decode_responses=True
        )
        await redis_client.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")

        # Initialize RAG pipeline if enabled
        if ENABLE_RAG:
            logger.info("Initializing RAG pipeline...")

            # Initialize vector store
            vector_store = QdrantVectorStore(
                collection_name="aeon_documents",
                qdrant_host=QDRANT_HOST,
                embedding_endpoint=EMBEDDING_ENDPOINT
            )

            # Initialize RAG pipeline
            rag_pipeline = RAGPipeline(
                redis_client=redis_client,
                vector_store=vector_store,
                cache_ttl=SESSION_TTL
            )

            # Initialize analytics
            rag_analytics = RAGAnalytics()

            # Health check
            health = await rag_pipeline.health_check()
            if health["status"] == "healthy":
                logger.info("RAG pipeline initialized successfully")
            else:
                logger.warning(f"RAG pipeline initialized with warnings: {health}")
        else:
            logger.info("RAG pipeline disabled (set ENABLE_RAG=true to enable)")

        # Initialize Code Executor if enabled
        if ENABLE_CODE_EXEC:
            logger.info("Initializing Code Executor...")

            code_executor = CodeExecutor(
                namespace="default",
                executor_image="python:3.11-slim",
                cpu_limit="500m",
                memory_limit="512Mi",
                timeout_seconds=30,
                max_output_lines=1000
            )

            logger.info("Code Executor initialized successfully")
        else:
            logger.info("Code Executor disabled (set ENABLE_CODE_EXEC=true to enable)")

        # Initialize Cipher agent if enabled
        if ENABLE_AGENT:
            logger.info("Initializing Cipher agent...")

            cipher_agent = CipherAgent(
                rag_pipeline=rag_pipeline if ENABLE_RAG else None,
                vllm_endpoint=VLLM_ENDPOINT,
                searxng_endpoint=SEARXNG_ENDPOINT,
                model="mistralai/Mistral-7B-Instruct-v0.2",
                code_executor=code_executor if ENABLE_CODE_EXEC else None
            )

            logger.info("Cipher agent initialized successfully")
            logger.info(f"Agent tools: {[tool.name for tool in cipher_agent.tools]}")
        else:
            logger.info("Cipher agent disabled (set ENABLE_AGENT=true to enable)")

    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Close Redis connection on shutdown"""
    global redis_client
    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")


# Helper functions
async def get_session_history(session_id: str) -> List[dict]:
    """Retrieve conversation history for a session"""
    try:
        history_json = await redis_client.get(f"session:{session_id}")
        if history_json:
            return json.loads(history_json)
        return []
    except Exception as e:
        logger.error(f"Error retrieving session history: {e}")
        return []


async def save_session_history(session_id: str, history: List[dict]):
    """Save conversation history for a session"""
    try:
        # Keep only last 10 messages to manage context size
        history = history[-10:]
        await redis_client.setex(
            f"session:{session_id}",
            SESSION_TTL,
            json.dumps(history)
        )
    except Exception as e:
        logger.error(f"Error saving session history: {e}")


async def call_llm(messages: List[dict], temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """Call vLLM inference server"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{VLLM_ENDPOINT}/chat/completions",
                json={
                    "model": "mistralai/Mistral-7B-Instruct-v0.3",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except httpx.HTTPError as e:
        logger.error(f"LLM API error: {e}")
        raise HTTPException(status_code=503, detail=f"LLM service unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error calling LLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "service": "Aeon AI Platform",
        "version": "0.4.0",
        "status": "operational",
        "agent": "Cipher",
        "features": {
            "rag": ENABLE_RAG,
            "chat": True,
            "websocket": True,
            "code_execution": ENABLE_CODE_EXEC
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

    try:
        # Check Redis connectivity
        await redis_client.ping()
        health_status["redis"] = "healthy"
    except Exception as e:
        health_status["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    # Check RAG pipeline if enabled
    if ENABLE_RAG and rag_pipeline:
        try:
            rag_health = await rag_pipeline.health_check()
            health_status["rag"] = rag_health["status"]
            if rag_health["status"] != "healthy":
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["rag"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
    else:
        health_status["rag"] = "disabled"

    # Check Cipher agent if enabled
    if ENABLE_AGENT and cipher_agent:
        health_status["agent"] = "healthy"
        health_status["agent_tools"] = len(cipher_agent.tools)
    elif ENABLE_AGENT:
        health_status["agent"] = "unhealthy: not initialized"
        health_status["status"] = "degraded"
    else:
        health_status["agent"] = "disabled"

    return health_status


# ===== RAG Endpoints =====

@app.post("/api/rag/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document for RAG

    Supports: PDF, DOCX, TXT, MD, HTML
    """
    if not ENABLE_RAG:
        raise HTTPException(status_code=503, detail="RAG is not enabled")

    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.markdown', '.html', '.htm'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
            )

        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = Path(tmp_file.name)

        try:
            # Process and chunk document
            chunks = process_and_chunk_document(
                tmp_path,
                chunk_size=512,
                chunk_overlap=128,
                metadata={
                    "upload_timestamp": datetime.utcnow().isoformat(),
                    "original_filename": file.filename
                }
            )

            if not chunks:
                raise HTTPException(status_code=400, detail="No content extracted from document")

            # Add to RAG system
            document_id = chunks[0].document_id
            count = await rag_pipeline.add_documents(chunks)

            # Track in analytics
            if rag_analytics:
                rag_analytics.track_document(
                    document_id=document_id,
                    filename=file.filename,
                    file_type=file_ext,
                    file_size=tmp_path.stat().st_size,
                    chunk_count=len(chunks)
                )

            logger.info(f"Uploaded document {file.filename}: {count} chunks created")

            return DocumentUploadResponse(
                document_id=document_id,
                filename=file.filename,
                chunks_created=count,
                status="success"
            )

        finally:
            # Clean up temporary file
            tmp_path.unlink()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Query the RAG system for relevant context

    Uses two-tier caching for performance
    """
    if not ENABLE_RAG:
        raise HTTPException(status_code=503, detail="RAG is not enabled")

    try:
        start_time = time.time()

        # Query RAG pipeline
        result = await rag_pipeline.query(
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold,
            use_cache=request.use_cache
        )

        latency_ms = (time.time() - start_time) * 1000

        # Log query for analytics
        if rag_analytics and result["results"]:
            query_hash = hashlib.sha256(request.query.encode()).hexdigest()[:16]
            rag_analytics.log_query(
                query_text=request.query,
                query_hash=query_hash,
                session_id="api",
                result_count=result["count"],
                top_score=result["results"][0]["score"] if result["results"] else None,
                cache_hit=(result["source"] == "cache"),
                latency_ms=latency_ms,
                score_threshold=request.score_threshold
            )

            # Update document retrieval stats
            for res in result["results"]:
                rag_analytics.update_document_retrieval(
                    document_id=res["document_id"],
                    relevance_score=res["score"]
                )

        return RAGQueryResponse(**result)

    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rag/chat", response_model=RAGChatResponse)
async def rag_chat(request: RAGChatRequest):
    """
    RAG-enhanced chat endpoint

    Retrieves relevant context and includes it in the LLM prompt
    """
    if not ENABLE_RAG:
        raise HTTPException(status_code=503, detail="RAG is not enabled")

    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(f"RAG chat request for session {session_id}: {request.message[:50]}...")

    try:
        # Retrieve conversation history
        history = await get_session_history(session_id)

        # Retrieve relevant context if RAG is enabled
        context_results = []
        if request.use_rag:
            rag_result = await rag_pipeline.query(
                query=request.message,
                limit=request.rag_limit,
                score_threshold=0.7,
                use_cache=True
            )
            context_results = rag_result["results"]

        # Build system message with context
        system_message = {
            "role": "system",
            "content": "You are Cipher, an intelligent AI assistant with access to a knowledge base. "
                      "Use the provided context to answer questions accurately."
        }

        if context_results:
            context_text = "\n\n---\n\n".join([
                f"Context {i + 1} (relevance: {r['score']:.2f}):\n{r['content']}"
                for i, r in enumerate(context_results)
            ])
            system_message["content"] += f"\n\nRelevant context:\n{context_text}"

        # Build messages for LLM
        messages_for_llm = [system_message] + history + [
            {"role": "user", "content": request.message}
        ]

        # Call LLM
        assistant_response = await call_llm(
            messages_for_llm,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        # Update history (without system message)
        history.append({"role": "user", "content": request.message})
        history.append({"role": "assistant", "content": assistant_response})
        await save_session_history(session_id, history)

        logger.info(f"RAG chat response for session {session_id}: {assistant_response[:50]}...")

        return RAGChatResponse(
            response=assistant_response,
            session_id=session_id,
            context_used=context_results,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Error in RAG chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/rag/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the RAG system"""
    if not ENABLE_RAG:
        raise HTTPException(status_code=503, detail="RAG is not enabled")

    try:
        await rag_pipeline.delete_document(document_id)
        return {"message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rag/stats")
async def rag_stats():
    """Get RAG pipeline statistics"""
    if not ENABLE_RAG:
        raise HTTPException(status_code=503, detail="RAG is not enabled")

    try:
        stats = await rag_pipeline.get_stats()

        # Add analytics if available
        if rag_analytics:
            query_stats = rag_analytics.get_query_stats(hours=24)
            top_docs = rag_analytics.get_top_documents(limit=10)
            stats["analytics"] = {
                "query_stats_24h": query_stats,
                "top_documents": top_docs
            }

        return stats
    except Exception as e:
        logger.error(f"Error getting RAG stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===== Agent Endpoints (Cipher) =====

@app.post("/api/agent/query", response_model=AgentResponse)
async def agent_query(request: AgentRequest):
    """
    Query the Cipher agent with intelligent routing

    The agent automatically routes to:
    - RAG retrieval for knowledge base queries
    - Web search for current information
    - Direct chat for general conversation
    """
    if not ENABLE_AGENT:
        raise HTTPException(status_code=503, detail="Cipher agent is not enabled")

    if not cipher_agent:
        raise HTTPException(status_code=503, detail="Cipher agent not initialized")

    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())

    try:
        logger.info(f"Agent query from session {session_id}: {request.query[:100]}")

        # Run agent
        result = await cipher_agent.run(query=request.query)

        # Save to session history if Redis is available
        if redis_client:
            try:
                history = await get_session_history(session_id)
                history.append({
                    "role": "user",
                    "content": request.query,
                    "timestamp": datetime.utcnow().isoformat()
                })
                history.append({
                    "role": "assistant",
                    "content": result["response"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "tool_used": result["selected_tool"],
                    "metadata": result["metadata"]
                })

                # Keep last 20 messages
                history = history[-20:]

                await redis_client.setex(
                    f"session:{session_id}",
                    SESSION_TTL,
                    json.dumps(history)
                )
            except Exception as e:
                logger.warning(f"Failed to save session history: {e}")

        return AgentResponse(
            response=result["response"],
            session_id=session_id,
            tool_used=result["selected_tool"],
            metadata=result["metadata"],
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Agent query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/api/agent/status")
async def agent_status():
    """Get Cipher agent status and available tools"""
    if not ENABLE_AGENT:
        return {
            "enabled": False,
            "message": "Cipher agent is not enabled"
        }

    if not cipher_agent:
        return {
            "enabled": True,
            "status": "not_initialized",
            "message": "Cipher agent failed to initialize"
        }

    return {
        "enabled": True,
        "status": "ready",
        "tools": [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in cipher_agent.tools
        ],
        "endpoints": {
            "vllm": VLLM_ENDPOINT,
            "searxng": SEARXNG_ENDPOINT,
            "rag_enabled": ENABLE_RAG
        }
    }


# ===== Chat Endpoints (Original) =====

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint with session management

    Maintains conversation history in Redis and calls vLLM for responses
    """
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(f"Chat request for session {session_id}: {request.message[:50]}...")

    # Retrieve conversation history
    history = await get_session_history(session_id)

    # Add user message
    user_message = {
        "role": "user",
        "content": request.message,
        "timestamp": datetime.utcnow().isoformat()
    }
    history.append({"role": "user", "content": request.message})

    # Call LLM
    messages_for_llm = history.copy()
    assistant_response = await call_llm(
        messages_for_llm,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )

    # Add assistant message to history
    history.append({
        "role": "assistant",
        "content": assistant_response
    })

    # Save updated history
    await save_session_history(session_id, history)

    logger.info(f"Chat response for session {session_id}: {assistant_response[:50]}...")

    return ChatResponse(
        response=assistant_response,
        session_id=session_id,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/api/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get information about a specific session"""
    history = await get_session_history(session_id)

    if not history:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionInfo(
        session_id=session_id,
        message_count=len(history),
        created_at=history[0].get("timestamp", "unknown") if history else "unknown",
        last_activity=history[-1].get("timestamp", "unknown") if history else "unknown"
    )


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its history"""
    try:
        deleted = await redis_client.delete(f"session:{session_id}")
        if deleted:
            logger.info(f"Session {session_id} deleted")
            return {"message": "Session deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming chat responses

    Provides real-time streaming of LLM responses
    """
    await websocket.accept()
    logger.info(f"WebSocket connection established for session {session_id}")

    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            logger.info(f"WebSocket message from {session_id}: {message[:50]}...")

            # Get conversation history
            history = await get_session_history(session_id)
            history.append({"role": "user", "content": message})

            # Stream response from vLLM
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    async with client.stream(
                        "POST",
                        f"{VLLM_ENDPOINT}/chat/completions",
                        json={
                            "model": "mistralai/Mistral-7B-Instruct-v0.3",
                            "messages": history,
                            "temperature": 0.7,
                            "max_tokens": 1024,
                            "stream": True
                        }
                    ) as response:
                        full_response = ""
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data = line[6:]
                                if data.strip() != "[DONE]":
                                    await websocket.send_text(data)
                                    try:
                                        chunk = json.loads(data)
                                        if "choices" in chunk and len(chunk["choices"]) > 0:
                                            delta = chunk["choices"][0].get("delta", {})
                                            if "content" in delta:
                                                full_response += delta["content"]
                                    except json.JSONDecodeError:
                                        pass

                        # Save complete response to history
                        history.append({"role": "assistant", "content": full_response})
                        await save_session_history(session_id, history)

            except Exception as e:
                logger.error(f"Error in streaming response: {e}")
                await websocket.send_text(json.dumps({"error": str(e)}))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
