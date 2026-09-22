from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.llm_client import OpenAILLMClient


def make_fake_client(answer: str):

    fake_response = SimpleNamespace(
        output_text=answer
    )

    fake_create = Mock(
        return_value=fake_response
    )

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=fake_create
        )
    )

    return fake_client, fake_create


def test_generate_returns_answer():

    fake_client, fake_create = make_fake_client(
        "Regular maintenance can improve motor performance."
    )

    llm = OpenAILLMClient(
        model="test-model",
        client=fake_client,
    )

    answer = llm.generate(
        "How can motor performance be improved?"
    )

    assert answer == (
        "Regular maintenance can improve motor performance."
    )

    fake_create.assert_called_once_with(
        model="test-model",
        input="How can motor performance be improved?",
        store=False,
    )


def test_generate_rejects_empty_prompt():

    fake_client, fake_create = make_fake_client(
        "Example answer"
    )

    llm = OpenAILLMClient(
        model="test-model",
        client=fake_client,
    )

    with pytest.raises(ValueError):
        llm.generate("   ")

    fake_create.assert_not_called()


def test_generate_rejects_empty_response():

    fake_client, _ = make_fake_client("")

    llm = OpenAILLMClient(
        model="test-model",
        client=fake_client,
    )

    with pytest.raises(RuntimeError):
        llm.generate("Explain motor efficiency.")


def test_rejects_empty_model_name():

    fake_client, _ = make_fake_client(
        "Example answer"
    )

    with pytest.raises(ValueError):
        OpenAILLMClient(
            model="",
            client=fake_client,
        )