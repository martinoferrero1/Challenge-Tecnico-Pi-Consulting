import asyncio
from typing import Any, Sequence


class GeminiEmbeddingModel:
    def __init__(
        self,
        api_key: str,
        model: str,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.client = client

    async def embed_text(self, text: str) -> list[float]:
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        response = await asyncio.to_thread(
            self._get_client().models.embed_content,
            model=self.model,
            contents=list(texts),
        )

        return [self._embedding_values(embedding) for embedding in response.embeddings]

    def _get_client(self) -> Any:
        if self.client is None:
            from google import genai

            self.client = genai.Client(api_key=self.api_key)

        return self.client

    def _embedding_values(self, embedding: Any) -> list[float]:
        values = getattr(embedding, "values", embedding)
        return list(values)
