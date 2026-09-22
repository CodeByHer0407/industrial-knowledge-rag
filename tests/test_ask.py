import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import ask


def test_dry_run_does_not_call_llm(monkeypatch, capsys):
    """Dry-run mode must retrieve evidence without calling an LLM."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["ask", "How can motor efficiency be improved?"],
    )

    fake_index = SimpleNamespace(ntotal=524)

    monkeypatch.setattr(
        ask,
        "load_faiss_index",
        Mock(return_value=(fake_index, [])),
    )

    monkeypatch.setattr(
        ask,
        "load_embedding_model",
        Mock(return_value=object()),
    )

    fake_source = {
        "chunk_id": "motor_manual.pdf_p29_c3",
        "source": "motor_manual.pdf",
        "page": 29,
        "score": 0.6980,
        "text": "Proper maintenance improves performance.",
    }

    fake_prepare = Mock(
        return_value={
            "question": "How can motor efficiency be improved?",
            "prompt": "Generated test prompt",
            "sources": [fake_source],
        }
    )

    monkeypatch.setattr(
        ask,
        "prepare_rag_context",
        fake_prepare,
    )

    fake_llm_class = Mock()
    fake_generation = Mock()

    monkeypatch.setattr(
        ask,
        "OpenAILLMClient",
        fake_llm_class,
    )

    monkeypatch.setattr(
        ask,
        "generate_rag_answer",
        fake_generation,
    )

    ask.main()

    output = capsys.readouterr().out

    assert "DRY RUN" in output
    assert "Prompt ready: True" in output

    fake_prepare.assert_called_once()
    fake_llm_class.assert_not_called()
    fake_generation.assert_not_called()


def test_live_mode_requires_api_key(monkeypatch):
    """Live mode should stop before loading models if the key is missing."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    monkeypatch.setattr(
        sys,
        "argv",
        ["ask", "Explain motor efficiency.", "--live"],
    )

    fake_index_loader = Mock()

    monkeypatch.setattr(
        ask,
        "load_faiss_index",
        fake_index_loader,
    )

    with pytest.raises(SystemExit) as error:
        ask.main()

    assert error.value.code == 2

    fake_index_loader.assert_not_called()


def test_rejects_invalid_top_k(monkeypatch):
    """The CLI must reject zero or negative retrieval limits."""

    monkeypatch.setattr(
        sys,
        "argv",
        ["ask", "Explain motor efficiency.", "--top-k", "0"],
    )

    fake_index_loader = Mock()

    monkeypatch.setattr(
        ask,
        "load_faiss_index",
        fake_index_loader,
    )

    with pytest.raises(SystemExit) as error:
        ask.main()

    assert error.value.code == 2

    fake_index_loader.assert_not_called()

def test_local_mode_does_not_require_openai_key(monkeypatch):

    # Remove the cloud API key.
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ask",
            "How can motor efficiency be improved?",
            "--provider",
            "ollama",
            "--live",
        ],
    )

    fake_index = SimpleNamespace(ntotal=524)

    monkeypatch.setattr(
        ask,
        "load_faiss_index",
        Mock(return_value=(fake_index, [])),
    )

    monkeypatch.setattr(
        ask,
        "load_embedding_model",
        Mock(return_value=object()),
    )

    fake_ollama_class = Mock()

    monkeypatch.setattr(
        ask,
        "OllamaLLMClient",
        fake_ollama_class,
    )

    fake_generation = Mock(
        return_value={
            "answer": "Test answer from local Ollama.",
            "sources": [],
            "answer_status": "answered",
            "citation_validation": {
                "has_citations": False,
                "valid_citations": [],
                "invalid_citations": [],
                "is_valid": False,
            },
        }
    )

    monkeypatch.setattr(
        ask,
        "generate_rag_answer",
        fake_generation,
    )

    fake_openai_class = Mock()

    monkeypatch.setattr(
        ask,
        "OpenAILLMClient",
        fake_openai_class,
    )

    ask.main()

    fake_ollama_class.assert_called_once_with(
        model="llama3.2:3b",
    )

    fake_generation.assert_called_once()
    fake_openai_class.assert_not_called()


def test_ollama_dry_run_does_not_initialize_llm(monkeypatch):

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ask",
            "Explain motor efficiency.",
            "--provider",
            "ollama",
        ],
    )

    fake_index = SimpleNamespace(ntotal=524)

    monkeypatch.setattr(
        ask,
        "load_faiss_index",
        Mock(return_value=(fake_index, [])),
    )

    monkeypatch.setattr(
        ask,
        "load_embedding_model",
        Mock(return_value=object()),
    )

    monkeypatch.setattr(
        ask,
        "prepare_rag_context",
        Mock(
            return_value={
                "prompt": "Test prompt",
                "sources": [],
            }
        ),
    )

    fake_ollama_class = Mock()

    monkeypatch.setattr(
        ask,
        "OllamaLLMClient",
        fake_ollama_class,
    )

    ask.main()

    fake_ollama_class.assert_not_called()