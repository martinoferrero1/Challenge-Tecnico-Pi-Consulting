import asyncio

from app.domain.entities.question import ConversationMessage
from app.infrastructure.conversation_stores.in_memory_conversation_store import (
    InMemoryConversationStore,
)


def test_in_memory_conversation_store_returns_recent_messages() -> None:
    store = InMemoryConversationStore()
    messages = (
        ConversationMessage(role="user", content="Primera pregunta"),
        ConversationMessage(role="assistant", content="Primera respuesta"),
        ConversationMessage(role="user", content="Segunda pregunta"),
    )

    asyncio.run(store.append("ana", messages))

    assert asyncio.run(store.get_recent("ana", limit=2)) == messages[-2:]
    assert asyncio.run(store.get_recent("luis", limit=2)) == ()
    assert asyncio.run(store.get_recent("ana", limit=0)) == ()
