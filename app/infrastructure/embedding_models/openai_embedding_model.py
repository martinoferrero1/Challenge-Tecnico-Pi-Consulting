from typing import Any, Sequence


class OpenAIEmbeddingModel:
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
        response = await self._get_client().embeddings.create(
            model=self.model,
            input=list(texts),
        )

        data = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in data]

    def _get_client(self) -> Any:
        if self.client is None:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(api_key=self.api_key)

        return self.client
