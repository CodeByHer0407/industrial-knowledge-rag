from app.prompt_builder import build_rag_prompt
from app.vector_store import search_faiss_index
from app.hybrid_retriever import search_hybrid_index
from app.context_expansion import expand_adjacent_chunks
from app.citation_validator import validate_citations


ABSTENTION_MESSAGE = (
    "The available documentation does not provide enough "
    "information to answer this question."
)


def prepare_rag_context(
    question: str,
    model,
    index,
    chunks: list[dict],
    top_k: int = 5,
    retrieval_mode: str = "faiss",
    bm25_retriever=None,
    candidate_k: int = 20,
    max_context_chunks: int = 10,
) -> dict:
    """
    Retrieve passages and prepare the LLM prompt.

    faiss:
        Use the original FAISS-only retrieval.

    hybrid:
        Fuse FAISS and BM25 rankings, then append
        adjacent chunks to the context.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if retrieval_mode == "faiss":
        retrieved_chunks = search_faiss_index(
            query=question,
            model=model,
            index=index,
            chunks=chunks,
            top_k=top_k,
        )

        context_chunks = retrieved_chunks

    elif retrieval_mode == "faiss_expanded":
        if max_context_chunks < top_k:
            raise ValueError(
                "max_context_chunks must be at least top_k."
            )

        retrieved_chunks = search_faiss_index(
            query=question,
            model=model,
            index=index,
            chunks=chunks,
            top_k=top_k,
        )

        context_chunks = expand_adjacent_chunks(
            retrieved_chunks=retrieved_chunks,
            all_chunks=chunks,
            max_total_chunks=max_context_chunks,
        )

    elif retrieval_mode == "hybrid":
        if bm25_retriever is None:
            raise ValueError(
                "Hybrid mode requires a BM25 retriever."
            )

        if max_context_chunks < top_k:
            raise ValueError(
                "max_context_chunks must be at least top_k."
            )

        retrieved_chunks = search_hybrid_index(
            query=question,
            model=model,
            index=index,
            chunks=chunks,
            bm25_retriever=bm25_retriever,
            top_k=top_k,
            candidate_k=candidate_k,
        )

        context_chunks = expand_adjacent_chunks(
            retrieved_chunks=retrieved_chunks,
            all_chunks=chunks,
            max_total_chunks=max_context_chunks,
        )

    else:
        raise ValueError(
            "retrieval_mode must be 'faiss', "
            "'faiss_expanded' or 'hybrid'."
        )

    prompt = (
        build_rag_prompt(
            question=question,
            retrieved_chunks=context_chunks,
        )
        if context_chunks
        else None
    )

    result = {
        "question": question,
        "prompt": prompt,
        "sources": context_chunks,
    }

    # Keep retrieval rankings separate from expanded context.
    # Preserve the original return structure for FAISS mode.
    if retrieval_mode in ("hybrid", "faiss_expanded"):
        result["retrieved_sources"] = retrieved_chunks

    return result


def generate_rag_answer(
    question: str,
    model,
    index,
    chunks: list[dict],
    llm,
    top_k: int = 5,
    retrieval_mode: str = "faiss",
    bm25_retriever=None,
    candidate_k: int = 20,
    max_context_chunks: int = 10,
) -> dict:
    """Retrieve evidence, generate an answer and validate citations."""

    context_kwargs = {
        "question": question,
        "model": model,
        "index": index,
        "chunks": chunks,
        "top_k": top_k,
    }

    # Keep the original FAISS invocation unchanged.
    if retrieval_mode != "faiss":
        context_kwargs.update({
            "retrieval_mode": retrieval_mode,
            "bm25_retriever": bm25_retriever,
            "candidate_k": candidate_k,
            "max_context_chunks": max_context_chunks,
        })

    context = prepare_rag_context(**context_kwargs)

    if context["prompt"] is None:
        result = {
            "question": question,
            "answer": (
                "No document passages were retrieved "
                "for this question."
            ),
            "sources": [],
            "answer_status": "no_context",
            "citation_validation": None,
        }

    else:
        answer = llm.generate(context["prompt"])

        is_abstention = (
            answer.strip() == ABSTENTION_MESSAGE
        )

        citation_report = (
            None
            if is_abstention
            else validate_citations(
                answer=answer,
                retrieved_sources=context["sources"],
            )
        )

        result = {
            "question": question,
            "answer": answer,
            "sources": context["sources"],
            "answer_status": (
                "abstained" if is_abstention else "answered"
            ),
            "citation_validation": citation_report,
        }

    if "retrieved_sources" in context:
        result["retrieved_sources"] = context[
            "retrieved_sources"
        ]

    return result