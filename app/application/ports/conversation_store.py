from typing import Protocol

from app.domain.entities.question import ConversationMessage


class ConversationStorePort(Protocol):
    async def get_recent(
        self,
        conversation_key: str,
        limit: int,
    ) -> tuple[ConversationMessage, ...]:
        raise NotImplementedError

    async def append(
        self,
        conversation_key: str,
        messages: tuple[ConversationMessage, ...],
    ) -> None:
        raise NotImplementedError
