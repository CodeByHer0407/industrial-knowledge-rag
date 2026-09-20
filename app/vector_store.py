import faiss
import numpy as np
import json
from pathlib import Path

def build_faiss_index(
    embeddings: np.ndarray,
    chunks: list[dict],
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Build a FAISS index and preserve chunk metadata.
    """

    # Convert embeddings to the format FAISS expects.
    vectors = np.asarray(embeddings, dtype=np.float32).copy()

    # Validate dimensions and metadata.
    if vectors.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")

    if len(vectors) == 0 or vectors.shape[1] == 0:
        raise ValueError("No valid embeddings provided.")

    if len(vectors) != len(chunks):
        raise ValueError(
            "Number of embeddings must match number of chunks."
        )

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain invalid values.")

    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Embeddings cannot contain zero vectors.")

    # Normalize vectors to unit length.
    faiss.normalize_L2(vectors)

    # Number of dimensions in each embedding.
    dimension = vectors.shape[1]

    # Create an exact inner-product search index.
    index = faiss.IndexFlatIP(dimension)

    # Store all document embeddings.
    index.add(vectors)

    return index, list(chunks)

def search_faiss_index(
    query: str,
    model,
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Search for relevant document chunks using a user question.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if index.ntotal != len(chunks):
        raise ValueError("Index and chunk metadata are misaligned.")

    if index.ntotal == 0:
        return []

    # Convert the question into a numerical embedding.
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_vector = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).copy()

    if query_vector.shape != (1, index.d):
        raise ValueError("Query embedding has incorrect dimensions.")

    if not np.isfinite(query_vector).all():
        raise ValueError("Query embedding contains invalid values.")

    if np.linalg.norm(query_vector) == 0:
        raise ValueError("Query embedding cannot be zero.")

    # Ensure unit-length query vector.
    faiss.normalize_L2(query_vector)

    # Search the index.
    k = min(top_k, index.ntotal)

    scores, indices = index.search(query_vector, k)

    # Convert FAISS results back to readable documents.
    results = []

    for score, chunk_position in zip(scores[0], indices[0]):

        chunk = chunks[int(chunk_position)]

        results.append({
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": float(score),
        })

    return results

def save_faiss_index(
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    directory: str | Path,
    model_name: str,
) -> None:
    """Save the FAISS index and associated metadata."""

    # Validate the relationship between vectors and chunks.
    if index.ntotal != len(chunks):
        raise ValueError(
            "Index and chunk metadata are misaligned."
        )

    if not model_name.strip():
        raise ValueError("Model name cannot be empty.")

    # Create the directory if necessary.
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Save numerical vectors and FAISS index.
    index_path = directory / "index.faiss"
    faiss.write_index(index, str(index_path))

    # Prepare metadata.
    metadata = {
        "model_name": model_name,
        "dimension": index.d,
        "chunks": chunks,
    }

    # Save metadata as JSON.
    metadata_path = directory / "metadata.json"

    with metadata_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

def load_faiss_index(
    directory: str | Path,
    expected_model_name: str,
) -> tuple[faiss.Index, list[dict]]:
    """Load a trusted local FAISS index and its metadata."""

    directory = Path(directory)

    index_path = directory / "index.faiss"
    metadata_path = directory / "metadata.json"

    # Verify both files exist.
    if not index_path.is_file():
        raise FileNotFoundError(index_path)

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    # Read metadata first.
    with metadata_path.open(
        "r", encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise ValueError("Invalid metadata format.")

    # Check embedding model compatibility.
    if metadata.get("model_name") != expected_model_name:
        raise ValueError("Embedding model mismatch.")

    dimension = metadata.get("dimension")
    chunks = metadata.get("chunks")

    if (
        not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension <= 0
        or not isinstance(chunks, list)
    ):
        raise ValueError("Invalid index metadata.")

    # Only deserialize a trusted, locally generated index.
    index = faiss.read_index(str(index_path))

    # Verify consistency.
    if index.d != dimension:
        raise ValueError("Index dimension mismatch.")

    if index.ntotal != len(chunks):
        raise ValueError(
            "Index and chunk metadata are misaligned."
        )

    return index, chunks