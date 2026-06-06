import asyncio
from types import SimpleNamespace

from app.infrastructure.llms.cohere_llm import CohereLLM
from app.infrastructure.llms.gemini_llm import GeminiLLM
from app.infrastructure.llms.openai_llm import OpenAILLM


class FakeOpenAIResponses:
    def __init__(self) -> None:
        self.prompt: str | None = None

    async def ainvoke(self, prompt: str) -> SimpleNamespace:
        self.prompt = prompt
        return SimpleNamespace(content=" OpenAI answer ")


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.delegate = FakeOpenAIResponses()

    async def ainvoke(self, prompt: str) -> SimpleNamespace:
        return await self.delegate.ainvoke(prompt)

class FakeChatModel:
    def __init__(self) -> None:
        self.prompt: str | None = None

    async def ainvoke(self, prompt: str) -> SimpleNamespace:
        self.prompt = prompt
        return SimpleNamespace(content=f" {prompt} answer ")


def test_openai_llm_generates_text() -> None:
    client = FakeOpenAIClient()
    llm = OpenAILLM(
        api_key="openai-key",
        model="gpt-5.5",
        client=client,
    )

    answer = asyncio.run(llm.generate("Prompt"))

    assert client.delegate.prompt == "Prompt"
    assert answer == "OpenAI answer"


def test_cohere_llm_generates_text() -> None:
    client = FakeChatModel()
    llm = CohereLLM(
        api_key="cohere-key",
        model="command-a-plus-05-2026",
        client=client,
    )

    answer = asyncio.run(llm.generate("Prompt"))

    assert client.prompt == "Prompt"
    assert answer == "Prompt answer"


def test_gemini_llm_generates_text() -> None:
    client = FakeChatModel()
    llm = GeminiLLM(
        api_key="gemini-key",
        model="gemini-3.5-flash",
        client=client,
    )

    answer = asyncio.run(llm.generate("Prompt"))

    assert client.prompt == "Prompt"
    assert answer == "Prompt answer"
