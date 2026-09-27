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
TEST_PATH = ROOT / "eval" / "test_questions.json"
INDEX_DIR = ROOT / "data" / "index"

DEFAULT_OUTPUT = (
    ROOT / "eval" / "runs" / "dev_ollama_pilot.json"
)

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
        raise ValueError(
            "Unexpected metadata chunks format"
        )

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
    """Save evaluation results as formatted JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    # --------------------------------------------------
    # 1. Command-line arguments
    # --------------------------------------------------

    parser = argparse.ArgumentParser(
        description=(
            "Generate Ollama answers for development "
            "evaluation."
        )
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Process all development questions "
            "instead of the pilot."
        ),
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

    # Choose between the original FAISS retrieval
    # and FAISS with adjacent-chunk expansion.
    parser.add_argument(
        "--retrieval-mode",
        choices=(
            "faiss",
            "faiss_expanded",
        ),
        default="faiss",
    )

    # Maximum number of passages supplied to the LLM
    # after adjacent-chunk expansion.
    parser.add_argument(
        "--max-context-chunks",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--dataset",
        choices=("dev", "test"),
        default="dev",
    )

    parser.add_argument(
        "--confirm-heldout",
        action="store_true",
        help="Explicitly authorize the held-out test run.",
    )

    args = parser.parse_args()

    if args.dataset == "test":
        if not args.confirm_heldout:
            parser.error(
                "Use --confirm-heldout to run the test dataset."
            )

        if not args.all:
            parser.error(
                "Use --all when evaluating the test dataset."
            )

        if args.output.resolve() == DEFAULT_OUTPUT.resolve():
            parser.error(
                "Specify a separate output file for the test run."
            )

        if args.output.exists():
            parser.error(
                "The test output already exists. "
                "Refusing to overwrite a previous run."
            )
    # --------------------------------------------------
    # 2. Validate configuration
    # --------------------------------------------------

    if args.top_k <= 0:
        parser.error(
            "--top-k must be positive"
        )

    if args.max_context_chunks <= 0:
        parser.error(
            "--max-context-chunks must be positive"
        )

    if (
        args.retrieval_mode == "faiss_expanded"
        and args.max_context_chunks < args.top_k
    ):
        parser.error(
            "--max-context-chunks must be at least --top-k"
        )

    # --------------------------------------------------
    # 3. Load development questions
    # --------------------------------------------------

    # Deliberately load only the development dataset.
    # The held-out test set remains untouched.
    dataset_path = (
        TEST_PATH if args.dataset == "test" else DEV_PATH
    )
    questions = load_evaluation_dataset(dataset_path)

    questions_by_id = {
        question["question_id"]: question
        for question in questions
    }

    if args.all:
        selected = questions

    else:
        unknown = (
            set(args.ids)
            - set(questions_by_id)
        )

        if unknown:
            parser.error(
                "Unknown development question IDs: "
                f"{sorted(unknown)}"
            )

        # Preserve the order supplied by the user
        # while avoiding duplicate question IDs.
        selected = [
            questions_by_id[question_id]
            for question_id in dict.fromkeys(args.ids)
        ]

    print(
        f"Selected questions: {len(selected)}"
    )

    print(
        f"Retrieval mode: {args.retrieval_mode}"
    )

    print(
        f"Top-k: {args.top_k}"
    )

    if args.retrieval_mode == "faiss_expanded":
        print(
            "Maximum context passages:",
            args.max_context_chunks,
        )

    # --------------------------------------------------
    # 4. Load FAISS index and embedding model
    # --------------------------------------------------

    print("\nLoading FAISS index...")

    index, chunks = load_faiss_index(
        directory=INDEX_DIR,
        expected_model_name=MODEL_NAME,
    )

    print(
        f"Loaded vectors: {index.ntotal}"
    )

    print(
        "Loading embedding model..."
    )

    embedding_model = load_embedding_model()

    # --------------------------------------------------
    # 5. Initialize local Ollama
    # --------------------------------------------------

    print(
        f"Initializing Ollama: {args.model}"
    )

    llm = OllamaLLMClient(
        model=args.model
    )

    # Fallback text lookup, if a retrieval result
    # does not directly contain its chunk text.
    chunk_texts = load_chunk_texts()

    # --------------------------------------------------
    # 6. Initialize the evaluation report
    # --------------------------------------------------

    report = {
        "dataset": args.dataset,
        "created_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "embedding_model": MODEL_NAME,
        "generation_model": args.model,
        "retrieval_mode": args.retrieval_mode,
        "top_k": args.top_k,
        "max_context_chunks": args.max_context_chunks,
        "index_vectors": index.ntotal,
        "question_count": len(selected),
        "results": [],
    }

    # --------------------------------------------------
    # 7. Generate answers
    # --------------------------------------------------

    for number, question in enumerate(
        selected,
        start=1,
    ):

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
                retrieval_mode=args.retrieval_mode,
                max_context_chunks=args.max_context_chunks,
            )

        except Exception:

            # Preserve previously generated answers
            # if an Ollama call or retrieval fails.
            save_results(
                args.output,
                report,
            )

            print(
                f"Stopped at {question_id}. "
                "Completed results saved to "
                f"{args.output}"
            )

            raise

        # --------------------------------------------------
        # 8. Preserve original retrieval rankings
        # --------------------------------------------------

        # In faiss_expanded mode, retrieved_sources
        # contains only the original ranked passages.
        #
        # In normal FAISS mode, result["sources"]
        # already contains the original top-k passages.
        ranked_sources = result.get(
            "retrieved_sources",
            result["sources"],
        )

        ranked_chunk_ids = [
            source["chunk_id"]
            for source in ranked_sources
        ]

        # --------------------------------------------------
        # 9. Preserve the full context supplied to Ollama
        # --------------------------------------------------

        # For FAISS mode, this contains the top-k
        # retrieved passages.
        #
        # For FAISS + expansion, it also includes
        # adjacent passages.
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

            score = source.get("score")

            retrieved_sources.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "source": source["source"],
                "page": source["page"],
                "score": (
                    float(score)
                    if score is not None
                    else None
                ),
                "text": text,
                "retrieval_origin": source.get(
                    "retrieval_origin",
                    "ranked_retrieval",
                ),
            })

        # --------------------------------------------------
        # 10. Save the answer and its supporting evidence
        # --------------------------------------------------

        record = {
            "question_id": question_id,
            "question": question["question"],
            "answerable": question["answerable"],
            "reference_answer": question.get(
                "reference_answer"
            ),
            "expected_pages": question["expected_pages"],
            "relevant_chunk_ids": (
                question["relevant_chunk_ids"]
            ),
            "retrieval_mode": args.retrieval_mode,
            "ranked_chunk_ids": ranked_chunk_ids,
            "generated_answer": result["answer"],
            "answer_status": result["answer_status"],
            "citation_validation": (
                result["citation_validation"]
            ),
            "retrieved_sources": retrieved_sources,
        }

        report["results"].append(
            record
        )

        # Save after every question.
        # This avoids losing all progress if a
        # later generation call fails.
        save_results(
            args.output,
            report,
        )

        # --------------------------------------------------
        # 11. Print progress
        # --------------------------------------------------

        print(
            "Ranked passages:",
            len(ranked_sources),
        )

        print(
            "Context passages:",
            len(retrieved_sources),
        )

        print(
            "Answer status:",
            record["answer_status"],
        )

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
                f"Warning: {missing_text} "
                "context passages have no saved text."
            )

    # --------------------------------------------------
    # 12. Final summary
    # --------------------------------------------------

    print(
        "\nBatch completed."
    )

    print(
        "Retrieval mode:",
        args.retrieval_mode,
    )

    print(
        "Generated answers:",
        len(report["results"]),
    )

    print(
        "Saved results:",
        args.output,
    )


if __name__ == "__main__":
    main()