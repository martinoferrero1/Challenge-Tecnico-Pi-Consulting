import asyncio
from typing import Any, Sequence


class CohereEmbeddingModel:
    def __init__(
        self,
        api_key: str,
        model: str,
        input_type: str,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.input_type = input_type
        self.client = client

    async def embed_text(self, text: str) -> list[float]:
        embeddings = await self.embed_batch([text])
        return embeddings[0]

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        response = await asyncio.to_thread(
            self._get_client().embed,
            inputs=[
                {"content": [{"type": "text", "text": text}]}
                for text in texts
            ],
            model=self.model,
            input_type=self.input_type,
            embedding_types=["float"],
        )

        return self._extract_embeddings(response)

    def _get_client(self) -> Any:
        if self.client is None:
            import cohere

            self.client = cohere.ClientV2(api_key=self.api_key)

        return self.client

    def _extract_embeddings(self, response: Any) -> list[list[float]]:
        embeddings = getattr(response, "embeddings", None)
        values = getattr(embeddings, "float", None)

        if values is None and isinstance(embeddings, dict):
            values = embeddings.get("float")
        if values is None:
            raise ValueError("Cohere response does not contain float embeddings")

        return [list(embedding) for embedding in values]
