from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from httpx import HTTPError
from ollama import ResponseError
from pydantic import BaseModel, Field

from app.embeddings import MODEL_NAME, load_embedding_model
from app.vector_store import load_faiss_index
from app.llm_client import OllamaLLMClient
from app.rag_pipeline import generate_rag_answer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "data" / "index"

OLLAMA_MODEL = "llama3.2:3b"


app = FastAPI(
    title="Industrial Knowledge-Base RAG Assistant",
    description="A local RAG assistant for industrial documentation.",
    version="0.2.0",
)


# -------------------------------
# Request schema
# -------------------------------

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


# -------------------------------
# Load and cache RAG components
# -------------------------------

@lru_cache(maxsize=1)
def get_rag_components():
    """
    Load the index and embedding model once.

    Subsequent requests reuse the cached objects.
    """

    try:
        index, chunks = load_faiss_index(
            directory=INDEX_DIR,
            expected_model_name=MODEL_NAME,
        )

        embedding_model = load_embedding_model()

    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "RAG resources are unavailable. "
                "Check that the FAISS index has been built."
            ),
        ) from exc

    return embedding_model, index, chunks


# -------------------------------
# Health endpoint
# -------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "industrial-knowledge-rag",
    }


# -------------------------------
# Ask endpoint
# -------------------------------

@app.post("/ask")
def ask_question(
    request: AskRequest,
    components=Depends(get_rag_components),
):
    """
    Retrieve relevant passages, generate a local answer,
    and return citation-validation results.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    embedding_model, index, chunks = components

    try:
        llm = OllamaLLMClient(
            model=OLLAMA_MODEL,
        )

        result = generate_rag_answer(
            question=question,
            model=embedding_model,
            index=index,
            chunks=chunks,
            llm=llm,
            top_k=request.top_k,
        )

    except (HTTPError, ResponseError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is unavailable or the local model "
                "could not be loaded. Check that Ollama is "
                "running and llama3.2:3b is installed."
            ),
        ) from exc

    return result