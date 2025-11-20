"""
Aeon Embedding Server
Provides text embedding generation using sentence-transformers on GPU
"""
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[Aeon::Embeddings] %(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Aeon Embedding Service")

# Model configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading embedding model: {MODEL_NAME}")
logger.info(f"Using device: {device}")

# Load model
model = SentenceTransformer(MODEL_NAME, device=device)
logger.info("Embedding model loaded successfully")


class EmbedRequest(BaseModel):
    texts: List[str]
    normalize: bool = True


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int


@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """
    Generate embeddings for a list of texts

    Args:
        texts: List of text strings to embed
        normalize: Whether to normalize embeddings (default: True)

    Returns:
        Embeddings as list of float vectors
    """
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")

        logger.info(f"Embedding {len(request.texts)} texts")

        # Generate embeddings
        embeddings = model.encode(
            request.texts,
            convert_to_numpy=True,
            normalize_embeddings=request.normalize,
            show_progress_bar=False
        )

        return EmbedResponse(
            embeddings=embeddings.tolist(),
            model=MODEL_NAME,
            dimension=EMBEDDING_DIM
        )

    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "dimension": EMBEDDING_DIM,
        "device": device
    }


@app.get("/info")
async def model_info():
    """Return model information"""
    return {
        "model_name": MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIM,
        "max_sequence_length": model.max_seq_length,
        "device": device,
        "cuda_available": torch.cuda.is_available()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
