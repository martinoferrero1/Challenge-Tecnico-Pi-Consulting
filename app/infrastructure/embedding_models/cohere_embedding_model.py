import asyncio
from typing import Any, Sequence


class CohereEmbeddingModel:
    """Adaptador de embeddings de Cohere."""

    def __init__(
        self,
        api_key: str,
        model: str,
        input_type: str,
        client: Any | None = None,
    ) -> None:
        """Guarda configuración y permite inyectar cliente para tests.

        Args:
            api_key: API key de Cohere.
            model: Nombre del modelo de embeddings.
            input_type: Tipo de input usado por Cohere para documentos o
                queries.
            client: Cliente compatible con LangChain, opcional para tests.
        """
        self.api_key = api_key
        self.model = model
        self.input_type = input_type
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
        if hasattr(client, "embed_query"):
            return list(await asyncio.to_thread(client.embed_query, text))

        embeddings = await self.embed_batch([text])
        return embeddings[0]

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
        elif hasattr(client, "embed"):
            embeddings = await asyncio.to_thread(
                client.embed,
                text_list,
                input_type=self.input_type,
            )
        else:
            embeddings = await asyncio.to_thread(client.embed_documents, text_list)

        return [list(embedding) for embedding in embeddings]

    def _get_client(self) -> Any:
        """Construye perezosamente el cliente de LangChain Cohere.

        Returns:
            Cliente de embeddings cacheado en ``self.client``.
        """
        if self.client is None:
            from langchain_cohere import CohereEmbeddings

            self.client = CohereEmbeddings(
                cohere_api_key=self.api_key,
                model=self.model,
            )

        return self.client
