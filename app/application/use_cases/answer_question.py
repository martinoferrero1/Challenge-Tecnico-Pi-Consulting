from dataclasses import dataclass
from hashlib import sha256

from app.application.ports.answer_cache import AnswerCachePort
from app.application.ports.embedding_model import EmbeddingModelPort
from app.application.ports.llm import LLMPort
from app.application.ports.vector_store import VectorStorePort
from app.domain.entities.answer import Answer
from app.domain.entities.cache_key import AnswerCacheKey
from app.domain.entities.question import UserQuestion
from app.domain.entities.retrieval import RetrievedChunk


@dataclass(frozen=True)
class AnswerQuestionConfig:
    retrieval_limit: int = 4
    fallback_answer: str = (
        "No encontre informacion suficiente en el documento para responder esa pregunta."
    )


class AnswerQuestionUseCase:
    def __init__(
        self,
        embedding_model: EmbeddingModelPort,
        vector_store: VectorStorePort,
        llm: LLMPort,
        answer_cache: AnswerCachePort,
        config: AnswerQuestionConfig | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.llm = llm
        self.answer_cache = answer_cache
        self.config = config or AnswerQuestionConfig()

        if self.config.retrieval_limit <= 0:
            raise ValueError("Retrieval limit must be greater than zero")

    async def execute(self, question: UserQuestion) -> Answer:
        query_embedding = await self.embedding_model.embed_text(question.content)
        retrieved_chunks = tuple(
            await self.vector_store.search(
                query_embedding=query_embedding,
                limit=self.config.retrieval_limit,
            )
        )
        cache_key = self._build_cache_key(question, retrieved_chunks)
        cached_answer = await self.answer_cache.get(cache_key)

        if cached_answer is not None:
            print("Answer retrieved from cache")
            return Answer(
                question=question,
                content=cached_answer.content,
                context=cached_answer.context,
            )

        if not retrieved_chunks:
            answer = Answer(
                question=question,
                content=self.config.fallback_answer,
                context=retrieved_chunks,
            )
            await self.answer_cache.set(cache_key, answer)
            return answer

        prompt = self._build_prompt(question, retrieved_chunks)
        generated_answer = (await self.llm.generate(prompt)).strip()
        answer = Answer(
            question=question,
            content=generated_answer,
            context=retrieved_chunks,
        )
        await self.answer_cache.set(cache_key, answer)

        return answer

    def _build_cache_key(
        self,
        question: UserQuestion,
        retrieved_chunks: tuple[RetrievedChunk, ...],
    ) -> AnswerCacheKey:
        return AnswerCacheKey(
            question=question.normalized_content,
            context_hash=self._build_context_hash(retrieved_chunks),
        )

    def _build_context_hash(self, retrieved_chunks: tuple[RetrievedChunk, ...]) -> str:
        digest = sha256()

        for retrieved_chunk in retrieved_chunks:
            digest.update(retrieved_chunk.chunk.id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(retrieved_chunk.chunk.content.encode("utf-8"))
            digest.update(b"\0")

        return digest.hexdigest()

    def _build_prompt(
        self,
        question: UserQuestion,
        retrieved_chunks: tuple[RetrievedChunk, ...],
    ) -> str:
        context = "\n\n".join(
            f"[{index}] {retrieved_chunk.chunk.content}"
            for index, retrieved_chunk in enumerate(retrieved_chunks, start=1)
        )

        return (
            "Answer the user question using only the provided context.\n"
            "If the context is not enough, say that there is not enough information.\n"
            "Keep the answer concise and in the same language as the question.\n\n"
            f"User: {question.user_name}\n"
            f"Question: {question.content}\n\n"
            f"Context:\n{context}"
        )
