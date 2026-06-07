import asyncio
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel

from app.infrastructure.llms.cohere_llm import CohereLLM
from app.infrastructure.llms.gemini_llm import GeminiLLM
from app.infrastructure.llms.openai_llm import OpenAILLM


class DecisionOutput(BaseModel):
    decision: Literal["same", "different"]


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


class FakeStructuredChatModel:
    def __init__(self, payload: dict[str, str]) -> None:
        self.payload = payload
        self.prompt: str | None = None
        self.output_schema: type[BaseModel] | None = None

    def with_structured_output(
        self,
        output_schema: type[BaseModel],
    ) -> "FakeStructuredChatModel":
        self.output_schema = output_schema
        return self

    async def ainvoke(self, prompt: str) -> BaseModel:
        if self.output_schema is None:
            raise AssertionError("Structured output schema was not configured")

        self.prompt = prompt
        return self.output_schema.model_validate(self.payload)


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


def test_openai_llm_generates_structured_output() -> None:
    client = FakeStructuredChatModel({"decision": "same"})
    llm = OpenAILLM(
        api_key="openai-key",
        model="gpt-5.5",
        client=client,
    )

    output = asyncio.run(llm.generate_structured("Prompt", DecisionOutput))

    assert client.prompt == "Prompt"
    assert output == DecisionOutput(decision="same")


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


def test_cohere_llm_generates_structured_output() -> None:
    client = FakeStructuredChatModel({"decision": "different"})
    llm = CohereLLM(
        api_key="cohere-key",
        model="command-a-plus-05-2026",
        client=client,
    )

    output = asyncio.run(llm.generate_structured("Prompt", DecisionOutput))

    assert client.prompt == "Prompt"
    assert output == DecisionOutput(decision="different")


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


def test_gemini_llm_generates_structured_output() -> None:
    client = FakeStructuredChatModel({"decision": "same"})
    llm = GeminiLLM(
        api_key="gemini-key",
        model="gemini-3.5-flash",
        client=client,
    )

    output = asyncio.run(llm.generate_structured("Prompt", DecisionOutput))

    assert client.prompt == "Prompt"
    assert output == DecisionOutput(decision="same")
