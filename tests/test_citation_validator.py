import pytest

from app.citation_validator import validate_citations


SOURCES = [
    {
        "source": "motor_manual.pdf",
        "page": 29,
    },
    {
        "source": "motor_manual.pdf",
        "page": 43,
    },
]


def test_valid_citation():

    result = validate_citations(
        answer="Proper maintenance helps. [motor_manual.pdf, p. 29]",
        retrieved_sources=SOURCES,
    )

    assert result["is_valid"] is True
    assert result["has_citations"] is True
    assert result["valid_citations"] == [
        "[motor_manual.pdf, p. 29]"
    ]
    assert result["invalid_citations"] == []


def test_rejects_unretrieved_page():

    result = validate_citations(
        answer="Example claim. [motor_manual.pdf, p. 999]",
        retrieved_sources=SOURCES,
    )

    assert result["is_valid"] is False
    assert result["invalid_citations"] == [
        "[motor_manual.pdf, p. 999]"
    ]


def test_detects_missing_citation():

    result = validate_citations(
        answer="Proper maintenance helps motor performance.",
        retrieved_sources=SOURCES,
    )

    assert result["has_citations"] is False
    assert result["is_valid"] is False


def test_rejects_incorrect_format():

    result = validate_citations(
        answer="According to [Document 1, p. 29], maintenance helps.",
        retrieved_sources=SOURCES,
    )

    assert result["is_valid"] is False
    assert result["invalid_citations"] == [
        "[Document 1, p. 29]"
    ]


def test_rejects_mixed_valid_and_invalid_citations():

    result = validate_citations(
        answer=(
            "Claim one. [motor_manual.pdf, p. 29] "
            "Claim two. [motor_manual.pdf, p. 999]"
        ),
        retrieved_sources=SOURCES,
    )

    assert result["is_valid"] is False

    assert result["valid_citations"] == [
        "[motor_manual.pdf, p. 29]"
    ]

    assert result["invalid_citations"] == [
        "[motor_manual.pdf, p. 999]"
    ]


def test_rejects_empty_answer():

    with pytest.raises(ValueError):
        validate_citations(
            answer="",
            retrieved_sources=SOURCES,
        )