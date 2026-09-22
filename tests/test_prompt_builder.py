import pytest

from app.prompt_builder import build_rag_prompt


def test_prompt_contains_question_and_evidence():

    chunks = [
        {
            "source": "motor_manual.pdf",
            "page": 29,
            "text": "Poorly maintained motors can reduce efficiency.",
            "chunk_id": "motor_manual.pdf_p29_c2",
            "score": 0.7,
        }
    ]

    prompt = build_rag_prompt(
        question="What affects motor efficiency?",
        retrieved_chunks=chunks,
    )

    assert "What affects motor efficiency?" in prompt
    assert "Poorly maintained motors" in prompt
    assert "motor_manual.pdf" in prompt
    assert "Page: 29" in prompt
    assert "Do not invent page numbers, source names, or technical details." in prompt
    assert "Cite supporting evidence immediately" in prompt
    assert "Do not cite document numbers" in prompt
    assert "Citation: [motor_manual.pdf, p. 29]" in prompt
    assert "[Document 1]" not in prompt


def test_empty_question_raises_error():

    with pytest.raises(ValueError):
        build_rag_prompt(
            question="",
            retrieved_chunks=[{"text": "Example"}],
        )


def test_empty_chunks_raise_error():

    with pytest.raises(ValueError):
        build_rag_prompt(
            question="What affects efficiency?",
            retrieved_chunks=[],
        )