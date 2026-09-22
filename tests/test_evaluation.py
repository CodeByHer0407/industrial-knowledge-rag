import pytest

from app.evaluation import (
    score_retrieval,
    summarize_evaluation,
)


def test_relevant_chunk_at_first_rank():
    """All metrics should equal 1 for a perfect first result."""

    result = score_retrieval(
        retrieved_ids=["A", "X", "Y"],
        relevant_ids=["A"],
        k=3,
    )

    assert result["hit_rate"] == 1.0
    assert result["labeled_recall"] == 1.0
    assert result["reciprocal_rank"] == 1.0


def test_partial_recall_and_second_rank():
    """Only one of two relevant chunks appears at rank 2."""

    result = score_retrieval(
        retrieved_ids=["X", "A", "Y"],
        relevant_ids=["A", "B"],
        k=3,
    )

    assert result["hit_rate"] == 1.0
    assert result["labeled_recall"] == 0.5
    assert result["reciprocal_rank"] == 0.5


def test_no_relevant_chunks_retrieved():
    """All metrics should be zero when nothing relevant is found."""

    result = score_retrieval(
        retrieved_ids=["X", "Y"],
        relevant_ids=["A"],
        k=2,
    )

    assert result["hit_rate"] == 0.0
    assert result["labeled_recall"] == 0.0
    assert result["reciprocal_rank"] == 0.0


def test_top_k_cutoff():
    """A relevant chunk after rank k should not count."""

    result = score_retrieval(
        retrieved_ids=["X", "A"],
        relevant_ids=["A"],
        k=1,
    )

    assert result["hit_rate"] == 0.0
    assert result["labeled_recall"] == 0.0
    assert result["reciprocal_rank"] == 0.0


def test_dataset_summary():
    """Verify averages across two questions."""

    evaluation_results = [
        {
            "question_id": "Q001",
            "retrieved_chunk_ids": ["X", "A"],
            "relevant_chunk_ids": ["A", "B"],
        },
        {
            "question_id": "Q002",
            "retrieved_chunk_ids": ["X", "Y"],
            "relevant_chunk_ids": ["C"],
        },
    ]

    summary = summarize_evaluation(
        evaluation_results,
        k=2,
    )

    assert summary["question_count"] == 2

    assert summary["hit_rate_at_k"] == pytest.approx(0.5)

    assert summary["labeled_recall_at_k"] == pytest.approx(0.25)

    assert summary["mrr_at_k"] == pytest.approx(0.25)

    assert len(summary["per_question"]) == 2


def test_invalid_inputs():
    """Check invalid evaluation configurations."""

    with pytest.raises(ValueError):
        score_retrieval(["A"], ["A"], k=0)

    with pytest.raises(ValueError):
        score_retrieval(["A"], [], k=5)

    with pytest.raises(ValueError):
        score_retrieval(["A"], ["A", "A"], k=5)

    with pytest.raises(ValueError):
        summarize_evaluation([], k=5)