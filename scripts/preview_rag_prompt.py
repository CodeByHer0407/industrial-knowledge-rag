import argparse
from pathlib import Path

from app.embeddings import MODEL_NAME, load_embedding_model
from app.vector_store import load_faiss_index
from app.rag_pipeline import prepare_rag_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "data" / "index"


def main():
    # Step 1: Accept a question from the terminal.
    parser = argparse.ArgumentParser(
        description="Preview the RAG prompt using real FAISS retrieval."
    )

    parser.add_argument(
        "question",
        help="Question to ask about the industrial documentation",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of document chunks to retrieve (default: 5)",
    )

    args = parser.parse_args()

    # Step 2: Load the saved index.
    print("Step 1: Loading saved FAISS index...")

    index, chunks = load_faiss_index(
        directory=INDEX_DIR,
        expected_model_name=MODEL_NAME,
    )

    print(f"Loaded vectors: {index.ntotal}")

    # Step 3: Load the embedding model.
    print("\nStep 2: Loading embedding model...")

    model = load_embedding_model()

    # Step 4: Retrieve evidence and build the prompt.
    print("\nStep 3: Preparing RAG context...")

    result = prepare_rag_context(
        question=args.question,
        model=model,
        index=index,
        chunks=chunks,
        top_k=args.top_k,
    )

    # Step 5: Handle empty retrieval results.
    if result["prompt"] is None:
        print("No document passages were retrieved.")
        return

    # Step 6: Show the retrieved sources.
    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for rank, source in enumerate(result["sources"], start=1):
        print(
            f"{rank}. {source['source']} "
            f"| Page: {source['page']} "
            f"| Score: {source['score']:.4f}"
        )

    # Step 7: Display the generated prompt.
    print("\n" + "=" * 70)
    print("GENERATED RAG PROMPT")
    print("=" * 70)

    print(result["prompt"])

    print("\n" + "=" * 70)
    print("Prompt preparation completed successfully.")
    print("No LLM was called.")
    print("=" * 70)


if __name__ == "__main__":
    main()