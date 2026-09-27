from app.vector_store import search_faiss_index


def reciprocal_rank_fusion(
    faiss_results: list[dict],
    bm25_results: list[dict],
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict]:
    """Fuse two ranked result lists using Reciprocal Rank Fusion."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")

    scores = {}
    documents = {}
    first_seen = {}

    for results in (faiss_results, bm25_results):
        seen_in_list = set()

        for rank, document in enumerate(results, start=1):
            chunk_id = document["chunk_id"]

            if chunk_id in seen_in_list:
                raise ValueError(
                    f"Duplicate chunk ID in one result list: {chunk_id}"
                )

            seen_in_list.add(chunk_id)

            if chunk_id not in documents:
                documents[chunk_id] = document
                first_seen[chunk_id] = len(first_seen)
                scores[chunk_id] = 0.0

            scores[chunk_id] += 1.0 / (rrf_k + rank)

    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            first_seen[chunk_id],
        ),
    )

    fused = []

    for chunk_id in ranked_ids[:top_k]:
        document = documents[chunk_id]

        fused.append({
            "chunk_id": document["chunk_id"],
            "source": document["source"],
            "page": document["page"],
            "text": document["text"],
            "score": scores[chunk_id],
        })

    return fused


def search_hybrid_index(
    query: str,
    model,
    index,
    chunks: list[dict],
    bm25_retriever,
    top_k: int = 5,
    candidate_k: int = 20,
    rrf_k: int = 60,
) -> list[dict]:
    """Retrieve candidates from FAISS and BM25, then fuse them."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if candidate_k < top_k:
        raise ValueError(
            "candidate_k must be greater than or equal to top_k."
        )

    faiss_results = search_faiss_index(
        query=query,
        model=model,
        index=index,
        chunks=chunks,
        top_k=candidate_k,
    )

    bm25_results = bm25_retriever.search(
        query=query,
        top_k=candidate_k,
    )

    return reciprocal_rank_fusion(
        faiss_results=faiss_results,
        bm25_results=bm25_results,
        top_k=top_k,
        rrf_k=rrf_k,
    )