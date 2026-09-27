import pytest

from app.bm25_retriever import BM25Retriever


@pytest.fixture
def chunks():
    return [
        {
            "chunk_id": "manual_p39_c1",
            "source": "motor_manual.pdf",
            "page": 39,
            "text": (
                "Annual inspection includes regreasing "
                "antifriction bearings and checking clearances."
            ),
        },
        {
            "chunk_id": "manual_p19_c1",
            "source": "motor_manual.pdf",
            "page": 19,
            "text": (
                "Voltage sags and surges can cause "
                "motor operating problems."
            ),
        },
        {
            "chunk_id": "manual_p49_c2",
            "source": "motor_manual.pdf",
            "page": 49,
            "text": (
                "High static head can limit pump "
                "speed reduction with a VFD."
            ),
        },
    ]


def test_finds_relevant_maintenance_chunk(chunks):
    retriever = BM25Retriever(chunks)

    results = retriever.search(
        "annual inspection regreasing"
    )

    assert results[0]["chunk_id"] == "manual_p39_c1"


def test_search_is_case_insensitive(chunks):
    retriever = BM25Retriever(chunks)

    results = retriever.search(
        "VOLTAGE, SURGES!"
    )

    assert results[0]["chunk_id"] == "manual_p19_c1"


def test_returns_empty_for_unmatched_query(chunks):
    retriever = BM25Retriever(chunks)

    assert retriever.search("hydroelectric") == []


def test_returns_standard_retrieval_fields(chunks):
    retriever = BM25Retriever(chunks)

    result = retriever.search(
        "high static head",
        top_k=1,
    )[0]

    assert set(result) == {
        "chunk_id",
        "source",
        "page",
        "text",
        "score",
    }

    assert result["chunk_id"] == "manual_p49_c2"
    assert result["score"] > 0


def test_rejects_duplicate_chunk_ids(chunks):
    duplicate = dict(chunks[0])

    with pytest.raises(ValueError, match="Duplicate"):
        BM25Retriever(chunks + [duplicate])


@pytest.mark.parametrize("query", ["", "   "])
def test_rejects_empty_queries(chunks, query):
    retriever = BM25Retriever(chunks)

    with pytest.raises(ValueError, match="empty"):
        retriever.search(query)


@pytest.mark.parametrize("top_k", [0, -1])
def test_rejects_invalid_top_k(chunks, top_k):
    retriever = BM25Retriever(chunks)

    with pytest.raises(ValueError, match="positive"):
        retriever.search("motor", top_k=top_k)


def test_empty_index_returns_no_results():
    retriever = BM25Retriever([])

    assert retriever.search("motor") == []