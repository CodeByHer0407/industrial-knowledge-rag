from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from httpx import ConnectError

import app.main as main


@pytest.fixture
def client(monkeypatch):
    """
    Replace real RAG resources with lightweight test objects.
    """

    fake_components = (
        object(),
        object(),
        [],
    )

    main.app.dependency_overrides[
        main.get_rag_components
    ] = lambda: fake_components

    monkeypatch.setattr(
        main,
        "OllamaLLMClient",
        Mock(return_value=object()),
    )

    with TestClient(main.app) as test_client:
        yield test_client

    main.app.dependency_overrides.clear()


def test_ask_returns_rag_answer(client, monkeypatch):

    fake_result = {
        "question": "How can motor efficiency be improved?",
        "answer": (
            "Proper maintenance can help. "
            "[motor_manual.pdf, p. 29]"
        ),
        "sources": [
            {
                "source": "motor_manual.pdf",
                "page": 29,
                "chunk_id": "motor_manual.pdf_p29_c3",
                "text": "Proper maintenance can help.",
                "score": 0.75,
            }
        ],
        "answer_status": "answered",
        "citation_validation": {
            "has_citations": True,
            "valid_citations": [
                "[motor_manual.pdf, p. 29]"
            ],
            "invalid_citations": [],
            "is_valid": True,
        },
    }

    fake_pipeline = Mock(return_value=fake_result)

    monkeypatch.setattr(
        main,
        "generate_rag_answer",
        fake_pipeline,
    )

    response = client.post(
        "/ask",
        json={
            "question": "How can motor efficiency be improved?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    assert response.json() == fake_result

    fake_pipeline.assert_called_once()

    assert fake_pipeline.call_args.kwargs["top_k"] == 5


def test_ask_rejects_empty_question(client):

    response = client.post(
        "/ask",
        json={
            "question": "   ",
        },
    )

    assert response.status_code == 422


def test_ask_rejects_invalid_top_k(client):

    response = client.post(
        "/ask",
        json={
            "question": "Explain motor efficiency.",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_ask_handles_ollama_connection_error(
    client,
    monkeypatch,
):

    fake_pipeline = Mock(
        side_effect=ConnectError("Connection refused")
    )

    monkeypatch.setattr(
        main,
        "generate_rag_answer",
        fake_pipeline,
    )

    response = client.post(
        "/ask",
        json={
            "question": "Explain motor efficiency.",
        },
    )

    assert response.status_code == 503
    assert "Ollama" in response.json()["detail"]