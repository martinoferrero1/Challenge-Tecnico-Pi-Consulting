import asyncio
from types import SimpleNamespace

from app.infrastructure.llms.cohere_llm import CohereLLM
from app.infrastructure.llms.gemini_llm import GeminiLLM
from app.infrastructure.llms.openai_llm import OpenAILLM


class FakeOpenAIResponses:
    def __init__(self) -> None:
        self.payload: dict[str, str] | None = None

    async def create(self, model: str, input: str) -> SimpleNamespace:
        self.payload = {"model": model, "input": input}
        return SimpleNamespace(output_text=" OpenAI answer ")


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = FakeOpenAIResponses()


class FakeCohereClient:
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None

    def chat(self, model: str, messages: list[dict[str, str]]) -> SimpleNamespace:
        self.payload = {"model": model, "messages": messages}
        return SimpleNamespace(
            message=SimpleNamespace(
                content=[SimpleNamespace(text=" Cohere answer ")]
            )
        )


class FakeGeminiModels:
    def __init__(self) -> None:
        self.payload: dict[str, str] | None = None

    def generate_content(self, model: str, contents: str) -> SimpleNamespace:
        self.payload = {"model": model, "contents": contents}
        return SimpleNamespace(text=" Gemini answer ")


class FakeGeminiClient:
    def __init__(self) -> None:
        self.models = FakeGeminiModels()


def test_openai_llm_generates_text() -> None:
    client = FakeOpenAIClient()
    llm = OpenAILLM(
        api_key="openai-key",
        model="gpt-5.5",
        client=client,
    )

    answer = asyncio.run(llm.generate("Prompt"))

    assert client.responses.payload == {"model": "gpt-5.5", "input": "Prompt"}
    assert answer == "OpenAI answer"


def test_cohere_llm_generates_text() -> None:
    client = FakeCohereClient()
    llm = CohereLLM(
        api_key="cohere-key",
        model="command-a-plus-05-2026",
        client=client,
    )

    answer = asyncio.run(llm.generate("Prompt"))

    assert client.payload == {
        "model": "command-a-plus-05-2026",
        "messages": [{"role": "user", "content": "Prompt"}],
    }
    assert answer == "Cohere answer"


def test_gemini_llm_generates_text() -> None:
    client = FakeGeminiClient()
    llm = GeminiLLM(
        api_key="gemini-key",
        model="gemini-3.5-flash",
        client=client,
    )

    answer = asyncio.run(llm.generate("Prompt"))

    assert client.models.payload == {
        "model": "gemini-3.5-flash",
        "contents": "Prompt",
    }
    assert answer == "Gemini answer"
