from typing import Protocol, Sequence


class EmbeddingModelPort(Protocol):
    async def embed_text(self, text: str) -> list[float]:
        raise NotImplementedError

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError
