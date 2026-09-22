from app.preprocessing import (
    filter_header_only_chunks,
    is_header_only_chunk,
)


def test_detects_standalone_page_header():

    text = (
        "20 | IMPROVING MOTOR AND DRIVE SYSTEM "
        "PERFORMANCE: A SOURCEBOOK FOR INDUSTRY"
    )

    assert is_header_only_chunk(text) is True


def test_detects_appendix_header():

    text = (
        "APPENDICES 72 | IMPROVING MOTOR AND DRIVE "
        "SYSTEM PERFORMANCE: A SOURCEBOOK FOR INDUSTRY"
    )

    assert is_header_only_chunk(text) is True


def test_preserves_substantive_content():

    text = (
        "20 | IMPROVING MOTOR AND DRIVE SYSTEM "
        "PERFORMANCE: A SOURCEBOOK FOR INDUSTRY "
        "Alignment problems can result from poor installation."
    )

    assert is_header_only_chunk(text) is False


def test_preserves_short_technical_content():

    text = "Motor efficiency decreases significantly at very low loads."

    assert is_header_only_chunk(text) is False


def test_filters_headers_without_changing_chunk_ids():

    chunks = [
        {
            "chunk_id": "manual_p30_c0",
            "text": (
                "20 | IMPROVING MOTOR AND DRIVE SYSTEM "
                "PERFORMANCE: A SOURCEBOOK FOR INDUSTRY"
            ),
        },
        {
            "chunk_id": "manual_p30_c1",
            "text": "Useful technical information about motors.",
        },
    ]

    filtered = filter_header_only_chunks(chunks)

    assert len(filtered) == 1

    assert filtered[0]["chunk_id"] == "manual_p30_c1"

    assert filtered[0]["text"] == (
        "Useful technical information about motors."
    )