import asyncio
from typing import Any, Sequence


class OpenAIEmbeddingModel:
    """Adaptador de embeddings de OpenAI."""

    def __init__(
        self,
        api_key: str,
        model: str,
        client: Any | None = None,
    ) -> None:
        """Guarda configuración y permite inyectar cliente para tests.

        Args:
            api_key: API key de OpenAI.
            model: Nombre del modelo de embeddings.
            client: Cliente compatible con LangChain, opcional para tests.
        """
        self.api_key = api_key
        self.model = model
        self.client = client

    async def embed_text(self, text: str) -> list[float]:
        """Genera un embedding para una consulta individual.

        Args:
            text: Texto de query a vectorizar.

        Returns:
            Embedding generado para ``text``.
        """
        client = self._get_client()
        if hasattr(client, "aembed_query"):
            return list(await client.aembed_query(text))

        return list(await asyncio.to_thread(client.embed_query, text))

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Genera embeddings para documentos o chunks.

        Args:
            texts: Textos de documentos o chunks a vectorizar.

        Returns:
            Lista de embeddings en el mismo orden que ``texts``.
        """
        client = self._get_client()
        text_list = list(texts)

        if hasattr(client, "aembed_documents"):
            embeddings = await client.aembed_documents(text_list)
        else:
            embeddings = await asyncio.to_thread(client.embed_documents, text_list)

        return [list(embedding) for embedding in embeddings]

    def _get_client(self) -> Any:
        """Construye perezosamente el cliente de LangChain OpenAI.

        Returns:
            Cliente de embeddings cacheado en ``self.client``.
        """
        if self.client is None:
            from langchain_openai import OpenAIEmbeddings

            self.client = OpenAIEmbeddings(
                api_key=self.api_key,
                model=self.model,
            )

        return self.client
