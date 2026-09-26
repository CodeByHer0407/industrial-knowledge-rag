import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.embeddings import MODEL_NAME, load_embedding_model
from app.eval_dataset import load_evaluation_dataset
from app.llm_client import OllamaLLMClient
from app.rag_pipeline import generate_rag_answer
from app.vector_store import load_faiss_index


ROOT = Path(__file__).resolve().parents[1]
DEV_PATH = ROOT / "eval" / "dev_questions.json"
INDEX_DIR = ROOT / "data" / "index"
DEFAULT_OUTPUT = ROOT / "eval" / "runs" / "dev_ollama_pilot.json"

PILOT_IDS = ["Q012", "Q013", "Q040"]


def load_chunk_texts():
    """Load indexed passages so evaluation results retain evidence."""
    metadata_path = INDEX_DIR / "metadata.json"

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    raw_chunks = metadata["chunks"]
    lookup = {}

    if isinstance(raw_chunks, list):
        items = [
            (None, chunk)
            for chunk in raw_chunks
        ]
    elif isinstance(raw_chunks, dict):
        items = list(raw_chunks.items())
    else:
        raise ValueError("Unexpected metadata chunks format")

    for fallback_id, chunk in items:
        if isinstance(chunk, str):
            if fallback_id:
                lookup[fallback_id] = chunk
            continue

        if not isinstance(chunk, dict):
            continue

        chunk_id = (
            chunk.get("chunk_id")
            or chunk.get("id")
            or fallback_id
        )

        text = (
            chunk.get("text")
            or chunk.get("content")
            or chunk.get("chunk_text")
        )

        if chunk_id and text:
            lookup[chunk_id] = text

    return lookup


def save_results(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate Ollama answers for development evaluation."
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all development questions instead of the pilot.",
    )

    parser.add_argument(
        "--ids",
        nargs="+",
        default=PILOT_IDS,
        help="Specific development question IDs.",
    )

    parser.add_argument(
        "--model",
        default="llama3.2:3b",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    # Deliberately load only the development set.
    # The held-out test set must remain unused during development.
    questions = load_evaluation_dataset(DEV_PATH)

    questions_by_id = {
        question["question_id"]: question
        for question in questions
    }

    if args.all:
        selected = questions
    else:
        unknown = set(args.ids) - set(questions_by_id)

        if unknown:
            parser.error(
                f"Unknown development question IDs: {sorted(unknown)}"
            )

        selected = [
            questions_by_id[question_id]
            for question_id in dict.fromkeys(args.ids)
        ]

    print(f"Selected questions: {len(selected)}")

    print("Loading FAISS index...")

    index, chunks = load_faiss_index(
        directory=INDEX_DIR,
        expected_model_name=MODEL_NAME,
    )

    print(f"Loaded vectors: {index.ntotal}")

    print("Loading embedding model...")
    embedding_model = load_embedding_model()

    print(f"Initializing Ollama: {args.model}")
    llm = OllamaLLMClient(model=args.model)

    chunk_texts = load_chunk_texts()

    report = {
        "dataset": "dev",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": MODEL_NAME,
        "generation_model": args.model,
        "top_k": args.top_k,
        "index_vectors": index.ntotal,
        "question_count": len(selected),
        "results": [],
    }

    for number, question in enumerate(selected, start=1):
        question_id = question["question_id"]

        print(
            f"\n[{number}/{len(selected)}] "
            f"{question_id}: {question['question']}"
        )

        try:
            result = generate_rag_answer(
                question=question["question"],
                model=embedding_model,
                index=index,
                chunks=chunks,
                llm=llm,
                top_k=args.top_k,
            )

        except Exception:
            # Preserve already generated results if Ollama fails.
            save_results(args.output, report)
            print(
                f"Stopped at {question_id}. "
                f"Completed results saved to {args.output}"
            )
            raise

        retrieved_sources = []

        for rank, source in enumerate(
            result["sources"],
            start=1,
        ):
            chunk_id = source["chunk_id"]

            text = (
                source.get("text")
                or source.get("content")
                or chunk_texts.get(chunk_id)
            )

            retrieved_sources.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "source": source["source"],
                "page": source["page"],
                "score": float(source["score"]),
                "text": text,
            })

        record = {
            "question_id": question_id,
            "question": question["question"],
            "answerable": question["answerable"],
            "reference_answer": question.get("reference_answer"),
            "expected_pages": question["expected_pages"],
            "relevant_chunk_ids": question["relevant_chunk_ids"],
            "generated_answer": result["answer"],
            "answer_status": result["answer_status"],
            "citation_validation": result["citation_validation"],
            "retrieved_sources": retrieved_sources,
        }

        report["results"].append(record)

        # Save after every question to avoid losing a long run.
        save_results(args.output, report)

        print(f"Answer status: {record['answer_status']}")

        validation = record["citation_validation"]

        if validation is not None:
            print(
                "Citation-reference validation:",
                validation["is_valid"],
            )

        missing_text = sum(
            source["text"] is None
            for source in retrieved_sources
        )

        if missing_text:
            print(
                f"Warning: {missing_text} retrieved passages "
                "have no saved text."
            )

    print("\nBatch completed.")
    print(f"Generated answers: {len(report['results'])}")
    print(f"Saved results: {args.output}")


if __name__ == "__main__":
    main()