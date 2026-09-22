import pytest
from unittest.mock import Mock

from app import rag_pipeline


def test_prepare_rag_context_builds_prompt(monkeypatch):

    fake_results = [
        {
            "chunk_id": "motor_manual.pdf_p29_c2",
            "source": "motor_manual.pdf",
            "page": 29,
            "text": "Poor maintenance reduces motor performance.",
            "score": 0.75,
        }
    ]

    # Replace real FAISS search with a fake implementation.
    def fake_search(query, model, index, chunks, top_k):

        assert query == "What affects motor performance?"
        assert top_k == 5

        return fake_results

    monkeypatch.setattr(
        rag_pipeline,
        "search_faiss_index",
        fake_search,
    )

    result = rag_pipeline.prepare_rag_context(
        question="What affects motor performance?",
        model=object(),
        index=object(),
        chunks=[],
    )

    assert result["question"] == "What affects motor performance?"
    assert result["sources"] == fake_results

    assert "Poor maintenance reduces motor performance." in result["prompt"]
    assert "motor_manual.pdf" in result["prompt"]
    assert "Page: 29" in result["prompt"]


def test_prepare_rag_context_handles_no_results(monkeypatch):

    monkeypatch.setattr(
        rag_pipeline,
        "search_faiss_index",
        lambda **kwargs: [],
    )

    result = rag_pipeline.prepare_rag_context(
        question="What is the maintenance procedure?",
        model=object(),
        index=object(),
        chunks=[],
    )

    assert result["prompt"] is None
    assert result["sources"] == []


def test_prepare_rag_context_rejects_empty_question():

    with pytest.raises(ValueError):
        rag_pipeline.prepare_rag_context(
            question="   ",
            model=object(),
            index=object(),
            chunks=[],
        )


def test_prepare_rag_context_rejects_invalid_top_k():

    with pytest.raises(ValueError):
        rag_pipeline.prepare_rag_context(
            question="What affects efficiency?",
            model=object(),
            index=object(),
            chunks=[],
            top_k=0,
        )


def test_generate_rag_answer_success(monkeypatch):
    """Retrieval, prompt building, and generation work together."""

    fake_results = [
        {
            "chunk_id": "motor_manual.pdf_p29_c2",
            "source": "motor_manual.pdf",
            "page": 29,
            "text": "Proper maintenance can improve motor performance.",
            "score": 0.75,
        }
    ]

    # Replace actual FAISS retrieval.
    monkeypatch.setattr(
        rag_pipeline,
        "search_faiss_index",
        lambda **kwargs: fake_results,
    )

    # Replace the real LLM with a mock.
    fake_llm = Mock()

    fake_llm.generate.return_value = (
        "Proper maintenance can improve motor performance."
    )

    result = rag_pipeline.generate_rag_answer(
        question="How can motor performance be improved?",
        model=object(),
        index=object(),
        chunks=[],
        llm=fake_llm,
    )

    assert result["question"] == (
        "How can motor performance be improved?"
    )

    assert result["answer"] == (
        "Proper maintenance can improve motor performance."
    )

    assert result["sources"] == fake_results

    # Confirm the LLM was called exactly once.
    fake_llm.generate.assert_called_once()

    # Confirm it received a prompt containing the evidence.
    prompt = fake_llm.generate.call_args.args[0]

    assert "Proper maintenance" in prompt
    assert "motor_manual.pdf" in prompt
    assert "Page: 29" in prompt


def test_generate_rag_answer_without_results(monkeypatch):
    """The LLM should not be called if retrieval returns nothing."""

    monkeypatch.setattr(
        rag_pipeline,
        "search_faiss_index",
        lambda **kwargs: [],
    )

    fake_llm = Mock()

    result = rag_pipeline.generate_rag_answer(
        question="What is the maintenance procedure?",
        model=object(),
        index=object(),
        chunks=[],
        llm=fake_llm,
    )

    assert result["sources"] == []

    assert result["answer"] == (
        "No document passages were retrieved "
        "for this question."
    )

    fake_llm.generate.assert_not_called()


def test_generate_rag_answer_rejects_empty_question():
    """Invalid questions should be rejected before LLM generation."""

    fake_llm = Mock()

    with pytest.raises(ValueError):
        rag_pipeline.generate_rag_answer(
            question="   ",
            model=object(),
            index=object(),
            chunks=[],
            llm=fake_llm,
        )

    fake_llm.generate.assert_not_called()

def test_generate_rag_answer_validates_correct_citation(
    monkeypatch,
):
    """A citation to a retrieved source should be accepted."""

    fake_sources = [
        {
            "chunk_id": "motor_manual.pdf_p29_c3",
            "source": "motor_manual.pdf",
            "page": 29,
            "text": "Proper maintenance improves motor performance.",
            "score": 0.75,
        }
    ]

    monkeypatch.setattr(
        rag_pipeline,
        "search_faiss_index",
        lambda **kwargs: fake_sources,
    )

    fake_llm = Mock()

    fake_llm.generate.return_value = (
        "Proper maintenance helps improve performance. "
        "[motor_manual.pdf, p. 29]"
    )

    result = rag_pipeline.generate_rag_answer(
        question="How can motor performance be improved?",
        model=object(),
        index=object(),
        chunks=[],
        llm=fake_llm,
    )

    report = result["citation_validation"]

    assert report["is_valid"] is True

    assert report["valid_citations"] == [
        "[motor_manual.pdf, p. 29]"
    ]

    assert report["invalid_citations"] == []


def test_generate_rag_answer_detects_invalid_citation(
    monkeypatch,
):
    """An invented page number should be flagged."""

    fake_sources = [
        {
            "chunk_id": "motor_manual.pdf_p29_c3",
            "source": "motor_manual.pdf",
            "page": 29,
            "text": "Proper maintenance improves motor performance.",
            "score": 0.75,
        }
    ]

    monkeypatch.setattr(
        rag_pipeline,
        "search_faiss_index",
        lambda **kwargs: fake_sources,
    )

    fake_llm = Mock()

    fake_llm.generate.return_value = (
        "Maintenance improves performance. "
        "[motor_manual.pdf, p. 999]"
    )

    result = rag_pipeline.generate_rag_answer(
        question="How can motor performance be improved?",
        model=object(),
        index=object(),
        chunks=[],
        llm=fake_llm,
    )

    report = result["citation_validation"]

    assert report["is_valid"] is False

    assert report["invalid_citations"] == [
        "[motor_manual.pdf, p. 999]"
    ]

def test_generate_rag_answer_handles_abstention(monkeypatch):

    fake_sources = [
        {
            "chunk_id": "motor_manual.pdf_p4_c23",
            "source": "motor_manual.pdf",
            "page": 4,
            "text": "Information about motor systems.",
            "score": 0.29,
        }
    ]

    monkeypatch.setattr(
        rag_pipeline,
        "search_faiss_index",
        lambda **kwargs: fake_sources,
    )

    fake_llm = Mock()

    fake_llm.generate.return_value = (
        rag_pipeline.ABSTENTION_MESSAGE
    )

    result = rag_pipeline.generate_rag_answer(
        question="What is the Wi-Fi password?",
        model=object(),
        index=object(),
        chunks=[],
        llm=fake_llm,
    )

    assert result["answer_status"] == "abstained"
    assert result["citation_validation"] is None
    assert result["answer"] == (
        rag_pipeline.ABSTENTION_MESSAGE
    )

    fake_llm.generate.assert_called_once()