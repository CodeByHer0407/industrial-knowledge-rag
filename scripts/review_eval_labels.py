import json
from pathlib import Path

from app.embeddings import MODEL_NAME
from app.vector_store import load_faiss_index


ROOT = Path(__file__).resolve().parents[1]

REVIEW_CANDIDATES = {
    "Q013": [
        "motor_manual.pdf_p47_c3",
        "motor_manual.pdf_p49_c1",
        "motor_manual.pdf_p47_c4",
    ],
    "Q015": [
        "motor_manual.pdf_p36_c6",
        "motor_manual.pdf_p36_c5",
        "motor_manual.pdf_p36_c4",
    ],
    "Q016": [
        "motor_manual.pdf_p65_c0",
        "motor_manual.pdf_p36_c0",
        "motor_manual.pdf_p36_c2",
        "motor_manual.pdf_p36_c3",
        "motor_manual.pdf_p64_c5",
    ],
}


def main():
    questions = json.loads(
        (ROOT / "eval" / "dev_questions.json").read_text(
            encoding="utf-8"
        )
    )

    _, chunks = load_faiss_index(
        directory=ROOT / "data" / "index",
        expected_model_name=MODEL_NAME,
    )

    chunk_lookup = {
        chunk["chunk_id"]: chunk for chunk in chunks
    }

    sections = []

    for question in questions:
        question_id = question["question_id"]

        if question_id not in REVIEW_CANDIDATES:
            continue

        sections.append(
            f"\n{'=' * 70}\n"
            f"{question_id}: {question['question']}\n"
            f"{'=' * 70}"
        )

        existing = question["relevant_chunk_ids"]
        candidates = REVIEW_CANDIDATES[question_id]

        # Review existing labels alongside new candidates.
        chunk_ids = list(dict.fromkeys(existing + candidates))

        for chunk_id in chunk_ids:
            chunk = chunk_lookup.get(chunk_id)

            if chunk is None:
                sections.append(
                    f"\nNOT FOUND: {chunk_id}\n"
                )
                continue

            status = (
                "CURRENTLY LABELED"
                if chunk_id in existing
                else "CANDIDATE — NEEDS REVIEW"
            )

            sections.append(
                f"\n[{status}]\n"
                f"Chunk: {chunk_id}\n"
                f"Source: {chunk['source']}\n"
                f"Page: {chunk['page']}\n"
                f"Text:\n{chunk['text']}\n"
            )

    output = ROOT / "data" / "index" / "label_review.txt"

    output.write_text(
        "\n".join(sections),
        encoding="utf-8",
    )

    print(f"Review file created: {output}")
    print(f"Questions flagged: {len(REVIEW_CANDIDATES)}")


if __name__ == "__main__":
    main()