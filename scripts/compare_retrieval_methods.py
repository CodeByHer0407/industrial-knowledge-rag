import json
from pathlib import Path

from app.bm25_retriever import BM25Retriever
from app.context_expansion import expand_adjacent_chunks
from app.embeddings import MODEL_NAME, load_embedding_model
from app.eval_dataset import load_evaluation_dataset
from app.hybrid_retriever import reciprocal_rank_fusion
from app.vector_store import (
    load_faiss_index,
    search_faiss_index,
)


ROOT = Path(__file__).resolve().parents[1]

DEV_PATH = ROOT / "eval" / "dev_questions.json"
INDEX_DIR = ROOT / "data" / "index"
OUTPUT = ROOT / "eval" / "reports" / "dev_retrieval_comparison.json"

TOP_K = 5
CANDIDATE_K = 20
MAX_CONTEXT_CHUNKS = 10


def calculate_metrics(records, method):
    """Calculate labeled retrieval metrics on answerable questions."""

    count = len(records)
    hits = 0
    recall_total = 0.0
    reciprocal_rank_total = 0.0

    for record in records:
        gold_ids = set(record["relevant_chunk_ids"])
        result_ids = record[method]

        matches = gold_ids.intersection(result_ids)

        hits += bool(matches)
        recall_total += len(matches) / len(gold_ids)

        first_match_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(result_ids, start=1)
                if chunk_id in gold_ids
            ),
            None,
        )

        if first_match_rank is not None:
            reciprocal_rank_total += 1 / first_match_rank

    return {
        "questions": count,
        "hits": hits,
        "hit_at_5": round(hits / count, 4),
        "labeled_recall_at_5": round(recall_total / count, 4),
        "mrr_at_5": round(reciprocal_rank_total / count, 4),
    }


def main():
    print("Loading development dataset...")
    questions = load_evaluation_dataset(DEV_PATH)

    # Unanswerable questions have no labeled relevant passages.
    # Include only answerable questions in relevance metrics.
    answerable = [
        question
        for question in questions
        if question["answerable"]
        and question["relevant_chunk_ids"]
    ]

    print(f"Development questions: {len(questions)}")
    print(f"Questions with labeled evidence: {len(answerable)}")

    print("\nLoading FAISS index...")
    index, chunks = load_faiss_index(
        directory=INDEX_DIR,
        expected_model_name=MODEL_NAME,
    )

    print(f"Indexed passages: {index.ntotal}")

    print("Loading embedding model...")
    model = load_embedding_model()

    print("Building BM25 retriever...")
    bm25 = BM25Retriever(chunks)

    records = []

    for number, question in enumerate(answerable, start=1):
        question_id = question["question_id"]
        query = question["question"]

        print(
            f"[{number}/{len(answerable)}] {question_id}"
        )

        # Fetch up to 20 candidates from each method.
        faiss_candidates = search_faiss_index(
            query=query,
            model=model,
            index=index,
            chunks=chunks,
            top_k=CANDIDATE_K,
        )

        bm25_candidates = bm25.search(
            query=query,
            top_k=CANDIDATE_K,
        )

        # Fuse the same candidate lists used by our
        # production hybrid retriever.
        hybrid_top5 = reciprocal_rank_fusion(
            faiss_results=faiss_candidates,
            bm25_results=bm25_candidates,
            top_k=TOP_K,
            rrf_k=60,
        )

        expanded = expand_adjacent_chunks(
            retrieved_chunks=hybrid_top5,
            all_chunks=chunks,
            max_total_chunks=MAX_CONTEXT_CHUNKS,
        )

        faiss_expanded = expand_adjacent_chunks(
        retrieved_chunks=faiss_candidates[:TOP_K],
        all_chunks=chunks,
        max_total_chunks=MAX_CONTEXT_CHUNKS,
        )
        
        records.append({
            "question_id": question_id,
            "relevant_chunk_ids": question["relevant_chunk_ids"],
            "faiss": [
                item["chunk_id"]
                for item in faiss_candidates[:TOP_K]
            ],
            "bm25": [
                item["chunk_id"]
                for item in bm25_candidates[:TOP_K]
            ],
            "hybrid": [
                item["chunk_id"]
                for item in hybrid_top5
            ],
            "expanded_context": [
                item["chunk_id"]
                for item in expanded
            ],
            "faiss_expanded_context": [
                item["chunk_id"] for item in faiss_expanded
            ],
        })

    metrics = {
        method: calculate_metrics(records, method)
        for method in ("faiss", "bm25", "hybrid")
    }

    # Expanded context can contain up to 10 passages,
    # so report its coverage separately from Hit@5.
    context_hits = sum(
        bool(
            set(record["relevant_chunk_ids"])
            & set(record["expanded_context"])
        )
        for record in records
    )

    context_coverage = {
        "questions": len(records),
        "questions_with_labeled_evidence": context_hits,
        "coverage_rate": round(
            context_hits / len(records), 4
        ),
    }

    improved = []
    regressed = []

    for record in records:
        gold = set(record["relevant_chunk_ids"])

        faiss_hit = bool(gold & set(record["faiss"]))
        hybrid_hit = bool(gold & set(record["hybrid"]))

        if hybrid_hit and not faiss_hit:
            improved.append(record["question_id"])

        elif faiss_hit and not hybrid_hit:
            regressed.append(record["question_id"])

    faiss_expansion_hits = sum(
    bool(
        set(record["relevant_chunk_ids"])
        & set(record["faiss_expanded_context"])
    )
    for record in records
)

    print(
        "\nFAISS + EXPANSION:",
        f"{faiss_expansion_hits}/{len(records)}"
    )
    print(
        "HYBRID + EXPANSION:",
        f"{context_hits}/{len(records)}"
    )

    for question_id in ("Q003", "Q013", "Q024", "Q030"):
        record = next(
            item for item in records
            if item["question_id"] == question_id
        )
        gold = set(record["relevant_chunk_ids"])

        print(
            question_id,
            "FAISS + expansion:",
            bool(gold & set(record["faiss_expanded_context"])),
            "Hybrid + expansion:",
            bool(gold & set(record["expanded_context"])),
        )
    report = {
        "dataset": "dev",
        "total_dev_questions": len(questions),
        "evaluated_answerable_questions": len(records),
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "max_context_chunks": MAX_CONTEXT_CHUNKS,
        "metrics": metrics,
        "expanded_context_coverage": context_coverage,
        "faiss_expanded_context_coverage": {
        "questions": len(records),
        "questions_with_labeled_evidence": faiss_expansion_hits,
        "coverage_rate": round(
            faiss_expansion_hits / len(records), 4
        ),
    },
        "improved_vs_faiss": improved,
        "regressed_vs_faiss": regressed,
        "results": records,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("RETRIEVAL COMPARISON")
    print("=" * 65)

    for method, result in metrics.items():
        print(f"\n{method.upper()}")
        print(f"Hit@5: {result['hit_at_5']:.4f}")
        print(
            "Labeled Recall@5:",
            f"{result['labeled_recall_at_5']:.4f}",
        )
        print(f"MRR@5: {result['mrr_at_5']:.4f}")

    print("\nHYBRID + EXPANSION")
    print(
        "Labeled context coverage:",
        context_coverage["questions_with_labeled_evidence"],
        "/",
        context_coverage["questions"],
    )

    print("\nImproved vs FAISS:", improved)
    print("Regressed vs FAISS:", regressed)

    for question_id in ("Q013", "Q024"):
        record = next(
            item
            for item in records
            if item["question_id"] == question_id
        )

        gold = set(record["relevant_chunk_ids"])

        print(f"\n{question_id}")
        for method in (
            "faiss", "bm25", "hybrid", "expanded_context"
        ):
            found = bool(gold & set(record[method]))
            print(f"  {method}: {found}")

    print(f"\nSaved report: {OUTPUT}")


if __name__ == "__main__":
    main()