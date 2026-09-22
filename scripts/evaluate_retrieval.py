import json
from pathlib import Path

from app.embeddings import MODEL_NAME, load_embedding_model
from app.vector_store import (
    load_faiss_index,
    search_faiss_index,
)
from app.evaluation import summarize_evaluation


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_DIR = PROJECT_ROOT / "data" / "index"

QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"

TOP_K = 5


def main():

    # Step 1: Load evaluation questions
    print("Step 1: Loading evaluation dataset...")

    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        questions = json.load(file)

    print(f"Evaluation questions: {len(questions)}")

    # Step 2: Load saved FAISS index
    print("\nStep 2: Loading saved FAISS index...")

    index, metadata = load_faiss_index(
        directory=INDEX_DIR,
        expected_model_name=MODEL_NAME,
    )

    print(f"Loaded vectors: {index.ntotal}")

    # Step 3: Load embedding model once
    print("\nStep 3: Loading embedding model...")

    model = load_embedding_model()

    # Store retrieval results for all questions
    evaluation_results = []

    # Step 4: Evaluate every question
    print("\nStep 4: Running retrieval evaluation...")

    for item in questions:

        question_id = item["question_id"]
        question = item["question"]
        relevant_ids = item["relevant_chunk_ids"]

        # Retrieve top-k chunks
        results = search_faiss_index(
            query=question,
            model=model,
            index=index,
            chunks=metadata,
            top_k=TOP_K,
        )

        retrieved_ids = [
            result["chunk_id"]
            for result in results
        ]

        evaluation_results.append({
            "question_id": question_id,
            "retrieved_chunk_ids": retrieved_ids,
            "relevant_chunk_ids": relevant_ids,
        })

        # Display results for manual inspection
        print("\n" + "=" * 65)
        print(f"{question_id}: {question}")

        print("\nKnown relevant chunks:")
        for chunk_id in relevant_ids:
            print(f"  {chunk_id}")

        print("\nRetrieved chunks:")

        relevant_set = set(relevant_ids)

        for rank, result in enumerate(results, start=1):

            chunk_id = result["chunk_id"]

            is_relevant = chunk_id in relevant_set

            status = "MATCH" if is_relevant else "NOT LABELED"

            print(
                f"  Rank {rank}: {chunk_id} "
                f"| Score: {result['score']:.4f} "
                f"| {status}"
            )

    # Step 5: Calculate aggregate evaluation metrics
    print("\n" + "=" * 65)
    print("EVALUATION SUMMARY")
    print("=" * 65)

    summary = summarize_evaluation(
        evaluation_results=evaluation_results,
        k=TOP_K,
    )

    print(f"Questions evaluated: {summary['question_count']}")
    print(f"Top-k: {summary['k']}")

    print(
        f"Hit Rate@{TOP_K}: "
        f"{summary['hit_rate_at_k']:.4f}"
    )

    print(
        f"Labeled Recall@{TOP_K}: "
        f"{summary['labeled_recall_at_k']:.4f}"
    )

    print(
        f"MRR@{TOP_K}: "
        f"{summary['mrr_at_k']:.4f}"
    )

    # Step 6: Show per-question metrics
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