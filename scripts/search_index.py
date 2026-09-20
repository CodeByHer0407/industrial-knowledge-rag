import argparse
from pathlib import Path

from app.embeddings import MODEL_NAME, load_embedding_model
from app.vector_store import load_faiss_index, search_faiss_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_DIR = PROJECT_ROOT / "data" / "index"


def main():

    # Accept a question from the terminal.
    parser = argparse.ArgumentParser(
        description="Search industrial documentation."
    )

    parser.add_argument(
        "question",
        help="Question to search for"
    )

    args = parser.parse_args()

    # Step 1: Load the saved FAISS index.
    print("Loading saved FAISS index...")

    index, metadata = load_faiss_index(
        directory=INDEX_DIR,
        expected_model_name=MODEL_NAME
    )

    print(f"Loaded vectors: {index.ntotal}")
    print(f"Vector dimensions: {index.d}")

    # Step 2: Load the embedding model.
    print("Loading embedding model...")

    model = load_embedding_model()

    # Step 3: Search the documentation.
    print("Searching documentation...")

    results = search_faiss_index(
        query=args.question,
        model=model,
        index=index,
        chunks=metadata,
        top_k=5
    )

    # Step 4: Display search results.
    for rank, result in enumerate(results, start=1):

        print("\n" + "=" * 60)

        print(f"Result: {rank}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Text:\n{result['text'][:500]}")


if __name__ == "__main__":
    main()