import faiss
import numpy as np


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