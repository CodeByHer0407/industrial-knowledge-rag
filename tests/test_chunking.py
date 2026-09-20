import pytest

from app.chunking import chunk_documents


def test_chunk_size_and_overlap():
    """Verify chunk size and overlapping words."""

    pages = [{
        "source": "motor.pdf",
        "page": 1,
        "text": "one two three four five six seven eight nine ten"
    }]

    chunks = chunk_documents(
        pages,
        chunk_size=6,
        chunk_overlap=2
    )

    assert len(chunks) == 2

    assert chunks[0]["text"] == (
        "one two three four five six"
    )

    assert chunks[1]["text"] == (
        "five six seven eight nine ten"
    )

    # Verify maximum chunk size.
    assert all(
        len(chunk["text"].split()) <= 6
        for chunk in chunks
    )


def test_metadata_preservation():
    """Verify source, page number, and chunk IDs."""

    pages = [
        {
            "source": "motor.pdf",
            "page": 1,
            "text": "one two three four"
        },
        {
            "source": "motor.pdf",
            "page": 2,
            "text": "five six seven eight"
        }
    ]

    chunks = chunk_documents(
        pages,
        chunk_size=3,
        chunk_overlap=1
    )

    assert len(chunks) == 4

    assert [chunk["page"] for chunk in chunks] == [
        1, 1, 2, 2
    ]

    assert all(
        chunk["source"] == "motor.pdf"
        for chunk in chunks
    )

    assert chunks[0]["chunk_id"] == "motor.pdf_p1_c0"
    assert chunks[1]["chunk_id"] == "motor.pdf_p1_c1"
    assert chunks[2]["chunk_id"] == "motor.pdf_p2_c0"


def test_short_and_empty_pages():
    """Verify short pages and empty pages."""

    pages = [
        {
            "source": "motor.pdf",
            "page": 1,
            "text": ""
        },
        {
            "source": "motor.pdf",
            "page": 2,
            "text": "Motor maintenance instructions"
        }
    ]

    chunks = chunk_documents(
        pages,
        chunk_size=120,
        chunk_overlap=25
    )

    assert len(chunks) == 1

    assert chunks[0]["page"] == 2

    assert chunks[0]["text"] == (
        "Motor maintenance instructions"
    )


@pytest.mark.parametrize(
    "chunk_size, chunk_overlap",
    [
        (0, 0),
        (4, -1),
        (4, 4),
    ]
)
def test_invalid_chunk_configuration(
    chunk_size,
    chunk_overlap
):
    """Verify invalid configurations raise errors."""

    with pytest.raises(ValueError):
        chunk_documents(
            [],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )