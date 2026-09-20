import numpy as np
import pytest

from app.vector_store import (
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
)


def test_save_and_load_index(tmp_path):

    # Create two sample document chunks.
    chunks = [
        {
            "chunk_id": "motor_p1_c0",
            "source": "motor.pdf",
            "page": 1,
            "text": "Motor efficiency",
        },
        {
            "chunk_id": "motor_p2_c0",
            "source": "motor.pdf",
            "page": 2,
            "text": "Bearing maintenance",
        },
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # Build index.
    index, metadata = build_faiss_index(
        embeddings, chunks
    )

    # Save index.
    save_faiss_index(
        index,
        metadata,
        tmp_path,
        model_name="test-model",
    )

    # Verify files exist.
    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "metadata.json").exists()

    # Load saved index.
    loaded_index, loaded_metadata = load_faiss_index(
        tmp_path,
        expected_model_name="test-model",
    )

    assert loaded_index.ntotal == 2
    assert loaded_index.d == 2
    assert loaded_metadata == chunks

    # Verify that loaded vectors still work.
    query = np.array([[1.0, 0.0]], dtype=np.float32)

    scores, indices = loaded_index.search(query, 1)

    assert indices[0][0] == 0
    assert scores[0][0] == pytest.approx(1.0)


def test_model_mismatch(tmp_path):

    chunks = [{
        "chunk_id": "c0",
        "source": "test.pdf",
        "page": 1,
        "text": "Example",
    }]

    embeddings = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    index, metadata = build_faiss_index(
        embeddings, chunks
    )

    save_faiss_index(
        index,
        metadata,
        tmp_path,
        model_name="model-A",
    )

    with pytest.raises(ValueError, match="model mismatch"):
        load_faiss_index(
            tmp_path,
            expected_model_name="model-B",
        )