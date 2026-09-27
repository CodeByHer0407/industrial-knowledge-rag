import pytest

from app.hybrid_retriever import reciprocal_rank_fusion


def make_document(chunk_id):
    return {
        "chunk_id": chunk_id,
        "source": "motor_manual.pdf",
        "page": 1,
        "text": f"Document {chunk_id}",
        "score": 0.5,
    }


def test_promotes_document_found_by_both_retrievers():
    faiss = [
        make_document("A"),
        make_document("B"),
    ]

    bm25 = [
        make_document("B"),
        make_document("C"),
    ]

    results = reciprocal_rank_fusion(
        faiss,
        bm25,
        top_k=3,
    )

    assert results[0]["chunk_id"] == "B"
    assert len(results) == 3


def test_empty_results():
    assert reciprocal_rank_fusion([], []) == []


def test_preserves_standard_result_fields():
    results = reciprocal_rank_fusion(
        [make_document("A")],
        [make_document("A")],
    )

    assert len(results) == 1
    assert set(results[0]) == {
        "chunk_id",
        "source",
        "page",
        "text",
        "score",
    }

    assert results[0]["score"] > 0


def test_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion([], [], top_k=0)

    with pytest.raises(ValueError, match="rrf_k"):
        reciprocal_rank_fusion([], [], rrf_k=0)