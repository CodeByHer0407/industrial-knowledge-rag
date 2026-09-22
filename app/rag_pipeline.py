from app.prompt_builder import build_rag_prompt
from app.vector_store import search_faiss_index
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
) -> dict:
    """
    Retrieve relevant document chunks and build an LLM prompt.

    The embedding model and FAISS index are passed in so
    they can be loaded once and reused across questions.
    """

    # Step 1: Validate the question.
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    # Step 2: Retrieve relevant document chunks.
    retrieved_chunks = search_faiss_index(
        query=question,
        model=model,
        index=index,
        chunks=chunks,
        top_k=top_k,
    )

    # Step 3: Handle cases where no documents are retrieved.
    if not retrieved_chunks:
        return {
            "question": question,
            "prompt": None,
            "sources": [],
        }

    # Step 4: Build a document-grounded prompt.
    prompt = build_rag_prompt(
        question=question,
        retrieved_chunks=retrieved_chunks,
    )

    # Step 5: Return the prompt and its supporting sources.
    return {
        "question": question,
        "prompt": prompt,
        "sources": retrieved_chunks,
    }

def generate_rag_answer(
    question: str,
    model,
    index,
    chunks: list[dict],
    llm,
    top_k: int = 5,
) -> dict:
    """
    Run retrieval, prompt construction, and LLM generation.

    Returns the generated answer and retrieved source metadata.
    """

    # Step 1: Retrieve evidence and prepare the prompt.
    context = prepare_rag_context(
        question=question,
        model=model,
        index=index,
        chunks=chunks,
        top_k=top_k,
    )

    # Step 2: Handle empty retrieval without calling the LLM.
    if context["prompt"] is None:
        return {
            "question": question,
            "answer": (
                "No document passages were retrieved "
                "for this question."
            ),
            "sources": [],
            "answer_status": "no_context",
            "citation_validation": None,
        }

    # Step 3: Generate the answer.
    answer = llm.generate(context["prompt"])

    # Recognize the explicit insufficient-evidence response.
    is_abstention = answer.strip() == ABSTENTION_MESSAGE

    if is_abstention:
        citation_report = None
    else:
        citation_report = validate_citations(
        answer=answer,
        retrieved_sources=context["sources"],
    )

    # Step 5: Return the answer, evidence, and validation report.
    return {
        "question": question,
        "answer": answer,
        "sources": context["sources"],
        "answer_status": (
            "abstained" if is_abstention else "answered"
        ),
        "citation_validation": citation_report,
}