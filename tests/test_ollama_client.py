from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.llm_client import OllamaLLMClient


def test_ollama_generates_answer():

    fake_client = Mock()

    fake_client.generate.return_value = SimpleNamespace(
        response="Regular maintenance improves motor performance."
    )

    llm = OllamaLLMClient(
        model="llama3.2:3b",
        client=fake_client,
    )

    answer = llm.generate("Explain motor maintenance.")

    assert answer == (
        "Regular maintenance improves motor performance."
    )

    fake_client.generate.assert_called_once_with(
        model="llama3.2:3b",
        prompt="Explain motor maintenance.",
        stream=False,
        options={"temperature": 0},
    )


def test_ollama_rejects_empty_prompt():

    fake_client = Mock()

    llm = OllamaLLMClient(
        client=fake_client,
    )

    with pytest.raises(ValueError):
        llm.generate("   ")

    fake_client.generate.assert_not_called()


def test_ollama_rejects_empty_response():

    fake_client = Mock()

    fake_client.generate.return_value = SimpleNamespace(
        response=""
    )

    llm = OllamaLLMClient(
        client=fake_client,
    )

    with pytest.raises(RuntimeError):
        llm.generate("Explain motor efficiency.")


def test_ollama_rejects_empty_model_name():

    with pytest.raises(ValueError):
        OllamaLLMClient(
            model="",
            client=Mock(),
        )