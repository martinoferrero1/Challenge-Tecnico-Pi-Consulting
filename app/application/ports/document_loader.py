from typing import Protocol

from app.domain.entities.document import Document


class DocumentLoaderPort(Protocol):
    async def load(self, source: str) -> Document:
        raise NotImplementedError
