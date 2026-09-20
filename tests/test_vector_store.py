import numpy as np
import pytest

from app.vector_store import (
    build_faiss_index,
    search_faiss_index,
)


# Fake embedding model for testing.
class FakeEmbeddingModel:

    def encode(self, texts, **kwargs):
        return np.array(
            [[1.0, 0.0]],
            dtype=np.float32
        )


@pytest.fixture
def sample_data():

    chunks = [
        {
            "chunk_id": "motor_p1_c0",
            "source": "motor.pdf",
            "page": 1,
            "text": "Motor efficiency information."
        },
        {
            "chunk_id": "motor_p2_c0",
            "source": "motor.pdf",
            "page": 2,
            "text": "Bearing maintenance information."
        }
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0]
        ],
        dtype=np.float32
    )

    return embeddings, chunks


def test_build_faiss_index(sample_data):

    embeddings, chunks = sample_data

    index, metadata = build_faiss_index(
        embeddings,
        chunks
    )

    assert index.ntotal == 2
    assert index.d == 2
    assert len(metadata) == 2


def test_semantic_search(sample_data):

    embeddings, chunks = sample_data

    index, metadata = build_faiss_index(
        embeddings,
        chunks
    )

    model = FakeEmbeddingModel()

    results = search_faiss_index(
        query="Motor efficiency",
        model=model,
        index=index,
        chunks=metadata,
        top_k=1
    )

    assert len(results) == 1

    assert results[0]["chunk_id"] == "motor_p1_c0"
    assert results[0]["source"] == "motor.pdf"
    assert results[0]["page"] == 1

    assert results[0]["score"] == pytest.approx(1.0)


def test_mismatched_embeddings_and_chunks(sample_data):

    embeddings, chunks = sample_data

    with pytest.raises(ValueError):
        build_faiss_index(
            embeddings,
            chunks[:1]
        )


def test_empty_query(sample_data):

    embeddings, chunks = sample_data

    index, metadata = build_faiss_index(
        embeddings,
        chunks
    )

    with pytest.raises(ValueError):
        search_faiss_index(
            query="  ",
            model=FakeEmbeddingModel(),
            index=index,
            chunks=metadata
        )


def test_invalid_top_k(sample_data):

    embeddings, chunks = sample_data

    index, metadata = build_faiss_index(
        embeddings,
        chunks
    )

    with pytest.raises(ValueError):
        search_faiss_index(
            query="Motor efficiency",
            model=FakeEmbeddingModel(),
            index=index,
            chunks=metadata,
            top_k=0
        )