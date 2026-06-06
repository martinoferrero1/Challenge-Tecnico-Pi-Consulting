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
        client = self._get_client()
        if hasattr(client, "aembed_query"):
            return list(await client.aembed_query(text))

        return list(await asyncio.to_thread(client.embed_query, text))

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        client = self._get_client()
        text_list = list(texts)

        if hasattr(client, "aembed_documents"):
            embeddings = await client.aembed_documents(text_list)
        else:
            embeddings = await asyncio.to_thread(client.embed_documents, text_list)

        return [list(embedding) for embedding in embeddings]

    def _get_client(self) -> Any:
        if self.client is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            self.client = GoogleGenerativeAIEmbeddings(
                google_api_key=self.api_key,
                model=self.model,
            )

        return self.client
