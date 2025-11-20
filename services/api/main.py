"""
Aeon API Backend
Main FastAPI application with chat endpoints and session management
"""
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import redis.asyncio as redis
import json
import logging
import os
from typing import List, Optional
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[Aeon::API] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://192.168.1.100:8000/v1")
REDIS_HOST = os.getenv("REDIS_HOST", "redis-master")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
SESSION_TTL = 3600  # 1 hour

# Initialize FastAPI app
app = FastAPI(
    title="Aeon AI Platform API",
    description="API for Cipher AI agent with RAG capabilities",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis client (initialized on startup)
redis_client: Optional[redis.Redis] = None


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


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize Redis connection on startup"""
    global redis_client
    try:
        redis_client = await redis.from_url(
            f"redis://{REDIS_HOST}:{REDIS_PORT}",
            encoding="utf-8",
            decode_responses=True
        )
        await redis_client.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
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
        "version": "0.1.0",
        "status": "operational",
        "agent": "Cipher"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connectivity
        await redis_client.ping()
        redis_status = "healthy"
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"

    return {
        "status": "healthy" if redis_status == "healthy" else "degraded",
        "redis": redis_status,
        "timestamp": datetime.utcnow().isoformat()
    }


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
