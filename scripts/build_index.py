from pathlib import Path
from app.preprocessing import filter_header_only_chunks

from app.ingestion import extract_pdf_pages
from app.chunking import chunk_documents
from app.embeddings import (
    MODEL_NAME,
    load_embedding_model,
    embed_chunks,
)
from app.vector_store import (
    build_faiss_index,
    save_faiss_index,
)


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_PATH = PROJECT_ROOT / "data" / "raw" / "motor_manual.pdf"

INDEX_DIR = PROJECT_ROOT / "data" / "index"


def main():

    print("Step 1: Extracting PDF pages...")

    pages = extract_pdf_pages(PDF_PATH)

    print(f"Extracted pages: {len(pages)}")

    print("Step 2: Generating text chunks...")

    chunks = chunk_documents(pages)

    print(f"Generated chunks before filtering: {len(chunks)}")

    # Remove standalone document headers.
    original_count = len(chunks)

    chunks = filter_header_only_chunks(chunks)

    removed_count = original_count - len(chunks)

    print(f"Header-only chunks removed: {removed_count}")
    print(f"Chunks after filtering: {len(chunks)}")

    if not chunks:
        raise ValueError("No usable chunks remain after preprocessing.")

    print("Step 3: Loading embedding model...")

    model = load_embedding_model()

    print("Step 4: Generating embeddings...")

    embeddings = embed_chunks(chunks, model)

    print(f"Embedding shape: {embeddings.shape}")

    print("Step 5: Building FAISS index...")

    index, metadata = build_faiss_index(
            embeddings,
            chunks,
    )

    print(f"Indexed vectors: {index.ntotal}")

    print("Step 6: Saving FAISS index...")

    save_faiss_index(
        index=index,
        chunks=metadata,
        directory=INDEX_DIR,
        model_name=MODEL_NAME,
    )

    print("\nIndex saved successfully!")
    print(f"Location: {INDEX_DIR}")


if __name__ == "__main__":
    main()