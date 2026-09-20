import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model() -> SentenceTransformer:
    """Load the pretrained embedding model."""

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    return model


def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer
) -> np.ndarray:
    """
    Convert text chunks into normalized embeddings.
    """

    if not chunks:
        raise ValueError("No chunks provided.")

    # Extract text from each chunk.
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Generate embeddings.
    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > 32
    )

    return np.asarray(embeddings, dtype=np.float32)