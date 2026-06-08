from types import SimpleNamespace
from typing import TypeVar

from pydantic import BaseModel

from app.infrastructure.pipelines import question_answering_pipeline


StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


class FakeLLM:
    async def generate(self, prompt: str) -> str:
        return "respuesta"

    async def generate_structured(
        self,
        prompt: str,
        output_schema: type[StructuredOutput],
    ) -> StructuredOutput:
        return output_schema.model_validate({"decision": "same"})


class FakeEmbeddingModel:
    pass


class FakeVectorStore:
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name


class FakeLanguageDetector:
    pass


def test_question_answering_pipeline_builds_separate_judge_llm(
    monkeypatch,
) -> None:
    main_llm = FakeLLM()
    judge_llm = FakeLLM()
    calls: list[dict[str, str | None]] = []

    def fake_create_llm(
        settings,
        *,
        provider_override: str | None = None,
        model_override: str | None = None,
    ) -> FakeLLM:
        calls.append(
            {
                "provider_override": provider_override,
                "model_override": model_override,
            }
        )
        return main_llm if len(calls) == 1 else judge_llm

    monkeypatch.setattr(
        question_answering_pipeline,
        "create_embedding_model",
        lambda settings: FakeEmbeddingModel(),
    )
    monkeypatch.setattr(
        question_answering_pipeline,
        "ChromaVectorStore",
        FakeVectorStore,
    )
    monkeypatch.setattr(
        question_answering_pipeline,
        "LinguaLanguageDetector",
        FakeLanguageDetector,
    )
    monkeypatch.setattr(question_answering_pipeline, "create_llm", fake_create_llm)

    use_case = question_answering_pipeline.create_answer_question_use_case(
        build_settings(
            judge_llm_provider="cohere",
            judge_llm_model="command-r-judge",
        )
    )

    assert use_case.llm is main_llm
    assert use_case.cache_judge_llm is judge_llm
    assert calls == [
        {
            "provider_override": None,
            "model_override": None,
        },
        {
            "provider_override": "cohere",
            "model_override": "command-r-judge",
        },
    ]


def build_settings(**overrides: object) -> SimpleNamespace:
    defaults = {
        "rag_retrieval_limit": 3,
        "conversation_context_mode": "disabled",
        "answer_cache_mode": "document_context",
        "conversation_history_limit": 10,
        "language_confidence_threshold": 0.5,
        "answer_validation_retries": 1,
        "llm_provider": "openai",
        "llm_temperature": 0.0,
        "judge_llm_provider": None,
        "judge_llm_model": None,
        "embedding_provider": "openai",
        "openai_api_key": "openai-key",
        "openai_llm_model": "gpt-5.5",
        "openai_embedding_model": "text-embedding-3-small",
        "cohere_api_key": "cohere-key",
        "cohere_llm_model": "command-a-03-2025",
        "cohere_embedding_model": "embed-v4.0",
        "cohere_embedding_input_type": "search_document",
        "gemini_api_key": "gemini-key",
        "gemini_llm_model": "gemini-3.5-flash",
        "gemini_embedding_model": "gemini-embedding-2",
        "chroma_persist_dir": ".chroma",
        "chroma_collection_name": "documents",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)
