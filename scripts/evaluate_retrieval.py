import argparse
import json
from pathlib import Path

from app.evaluation import summarize_evaluation
from app.eval_dataset import load_evaluation_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "data" / "index"
EVAL_DIR = PROJECT_ROOT / "eval"

DATASET_FILES = {
    "legacy": "questions.json",
    "dev": "dev_questions.json",
    "test": "test_questions.json",
}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate industrial RAG retrieval."
    )

    parser.add_argument(
        "--dataset",
        choices=DATASET_FILES,
        default="dev",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--confirm-final",
        action="store_true",
        help="Confirm that configuration is frozen "
             "before evaluating the final test set.",
    )

    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive.")

    if args.dataset == "test" and not args.confirm_final:
        parser.error(
            "Freeze the retrieval configuration first, "
            "then use --confirm-final to evaluate "
            "the untouched test set."
        )

    dataset_path = EVAL_DIR / DATASET_FILES[args.dataset]

    print(f"Dataset: {args.dataset}")
    print(f"Dataset path: {dataset_path}")

    try:
        if args.dataset == "legacy":
            questions = json.loads(
                dataset_path.read_text(encoding="utf-8")
            )
        else:
            questions = load_evaluation_dataset(dataset_path)

    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    # Unanswerable questions have no known relevant chunks.
    # Do not include them in positive retrieval metrics.
    answerable_questions = [
        item for item in questions
        if item.get("answerable", True)
    ]

    unanswerable_count = (
        len(questions) - len(answerable_questions)
    )

    print(f"Total questions: {len(questions)}")
    print(f"Answerable: {len(answerable_questions)}")
    print(f"Unanswerable: {unanswerable_count}")

    if not answerable_questions:
        parser.error(
            "No answerable questions available "
            "for positive retrieval evaluation."
        )

        # Import ML dependencies only after dataset validation.
    from app.embeddings import MODEL_NAME, load_embedding_model
    from app.vector_store import (
        load_faiss_index,
        search_faiss_index,
    )

    print("\nLoading saved FAISS index...")

    index, metadata = load_faiss_index(
        directory=INDEX_DIR,
        expected_model_name=MODEL_NAME,
    )

    print(f"Loaded vectors: {index.ntotal}")

    print("\nLoading embedding model...")
    model = load_embedding_model()

    evaluation_results = []

    for item in answerable_questions:
        results = search_faiss_index(
            query=item["question"],
            model=model,
            index=index,
            chunks=metadata,
            top_k=args.top_k,
        )

        retrieved_ids = [
            result["chunk_id"] for result in results
        ]

        relevant_ids = item["relevant_chunk_ids"]
        relevant_set = set(relevant_ids)

        evaluation_results.append({
            "question_id": item["question_id"],
            "retrieved_chunk_ids": retrieved_ids,
            "relevant_chunk_ids": relevant_ids,
        })

        print("\n" + "=" * 60)
        print(f"{item['question_id']}: {item['question']}")

        for rank, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]

            status = (
                "MATCH"
                if chunk_id in relevant_set
                else "NOT LABELED"
            )

            print(
                f"{rank}. {chunk_id} | "
                f"Score: {result['score']:.4f} | "
                f"{status}"
            )

    summary = summarize_evaluation(
        evaluation_results=evaluation_results,
        k=args.top_k,
    )

    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Dataset: {args.dataset}")
    print(f"Questions scored: {summary['question_count']}")
    print(f"Unanswerable questions excluded: {unanswerable_count}")

    print(
        f"Hit Rate@{args.top_k}: "
        f"{summary['hit_rate_at_k']:.4f}"
    )

    print(
        f"Labeled Recall@{args.top_k}: "
        f"{summary['labeled_recall_at_k']:.4f}"
    )

    print(
        f"MRR@{args.top_k}: "
        f"{summary['mrr_at_k']:.4f}"
    )

    print("\nPER-QUESTION METRICS")

    for item in summary["per_question"]:
        print(
            f"{item['question_id']} | "
            f"Hit: {item['hit_rate']:.2f} | "
            f"Recall: {item['labeled_recall']:.2f} | "
            f"RR: {item['reciprocal_rank']:.2f}"
        )


if __name__ == "__main__":
    main()