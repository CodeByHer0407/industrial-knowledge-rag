from typing import Protocol


class LLMClient(Protocol):
    """
    Common interface for text-generation providers.
    """

    def generate(self, prompt: str) -> str:
        ...


class OpenAILLMClient:
    """
    OpenAI implementation of the LLM interface.

    An API client can be injected to enable testing
    without making real API requests.
    """

    def __init__(self, model: str, client=None):

        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model name cannot be empty.")

        self.model = model

        if client is None:
            from openai import OpenAI

            # Reads OPENAI_API_KEY from the environment.
            client = OpenAI()

        self.client = client

    def generate(self, prompt: str) -> str:

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            store=False,
        )

        answer = response.output_text

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(
                "The LLM did not return a text answer."
            )

        return answer.strip()

class OllamaLLMClient:
    """
    Generate answers using a locally running Ollama model.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        client=None,
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model name cannot be empty.")

        self.model = model

        if client is None:
            from ollama import Client

            client = Client(
                host="http://127.0.0.1:11434",
                timeout=120,
            )

        self.client = client

    def generate(self, prompt: str) -> str:

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
            options={"temperature": 0},
        )

        answer = response.response

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(
                "Ollama did not return a text answer."
            )

        return answer.strip()