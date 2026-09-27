import pytest

from app.context_expansion import expand_adjacent_chunks


def make_chunk(page, number, source="motor_manual.pdf"):
    return {
        "chunk_id": f"{source}_p{page}_c{number}",
        "source": source,
        "page": page,
        "text": f"Content for page {page}, chunk {number}.",
    }


def make_result(chunk):
    return {**chunk, "score": 0.9}


def test_adds_adjacent_maintenance_chunk():
    c0 = make_chunk(39, 0)
    c1 = make_chunk(39, 1)
    c2 = make_chunk(39, 2)

    result = expand_adjacent_chunks(
        retrieved_chunks=[make_result(c0)],
        all_chunks=[c0, c1, c2],
    )

    assert [item["chunk_id"] for item in result] == [
        c0["chunk_id"],
        c1["chunk_id"],
    ]

    assert result[1]["retrieval_origin"] == "adjacent_chunk"
    assert result[1]["score"] is None


def test_preserves_ranked_results_before_neighbors():
    first = make_chunk(5, 10)
    second = make_chunk(39, 0)
    neighbor = make_chunk(39, 1)

    result = expand_adjacent_chunks(
        retrieved_chunks=[
            make_result(first),
            make_result(second),
        ],
        all_chunks=[first, second, neighbor],
    )

    assert [item["chunk_id"] for item in result[:2]] == [
        first["chunk_id"],
        second["chunk_id"],
    ]

    assert result[2]["chunk_id"] == neighbor["chunk_id"]


def test_does_not_duplicate_existing_results():
    c0 = make_chunk(39, 0)
    c1 = make_chunk(39, 1)

    result = expand_adjacent_chunks(
        retrieved_chunks=[
            make_result(c0),
            make_result(c1),
        ],
        all_chunks=[c0, c1],
    )

    assert len(result) == 2


def test_does_not_cross_page_boundaries():
    last_chunk = make_chunk(39, 1)
    next_page = make_chunk(40, 2)

    result = expand_adjacent_chunks(
        retrieved_chunks=[make_result(last_chunk)],
        all_chunks=[last_chunk, next_page],
    )

    assert len(result) == 1


def test_respects_context_limit():
    c0 = make_chunk(39, 0)
    c1 = make_chunk(39, 1)
    c2 = make_chunk(39, 2)

    result = expand_adjacent_chunks(
        retrieved_chunks=[make_result(c1)],
        all_chunks=[c0, c1, c2],
        max_total_chunks=2,
    )

    assert len(result) == 2
    assert result[0]["chunk_id"] == c1["chunk_id"]


def test_rejects_limit_smaller_than_retrieved_count():
    c0 = make_chunk(39, 0)
    c1 = make_chunk(39, 1)

    with pytest.raises(ValueError, match="smaller"):
        expand_adjacent_chunks(
            retrieved_chunks=[
                make_result(c0),
                make_result(c1),
            ],
            all_chunks=[c0, c1],
            max_total_chunks=1,
        )