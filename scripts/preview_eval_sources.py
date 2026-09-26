from pathlib import Path

from app.embeddings import MODEL_NAME
from app.vector_store import load_faiss_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "data" / "index"

# Pages not used as primary evidence by our existing questions.
PAGES_TO_REVIEW = [18, 30, 41, 45, 54, 60]


def main():
    _, chunks = load_faiss_index(
        directory=INDEX_DIR,
        expected_model_name=MODEL_NAME,
    )

    sections = []

    for page in PAGES_TO_REVIEW:
        candidates = [
            chunk
            for chunk in chunks
            if chunk["page"] == page
            and len(chunk.get("text", "").strip()) > 250
        ]

        for chunk in candidates[:2]:
            sections.append(
                f"\n{'=' * 70}\n"
                f"Chunk: {chunk['chunk_id']}\n"
                f"Page: {chunk['page']}\n"
                f"{'=' * 70}\n"
                f"{chunk['text']}\n"
            )

    output = INDEX_DIR / "curation_candidates.txt"
    output.write_text(
        "\n".join(sections),
        encoding="utf-8",
    )

    print(f"Created: {output}")
    print(f"Candidate passages: {len(sections)}")


if __name__ == "__main__":
    main()