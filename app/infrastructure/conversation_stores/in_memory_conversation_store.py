from app.domain.entities.question import ConversationMessage


class InMemoryConversationStore: # lo hice asi para simplificar y no introducir otra db para el historial, entonces se reinicia por cada vez que se levanta la api
    def __init__(self) -> None:
        self._messages_by_key: dict[str, list[ConversationMessage]] = {}

    async def get_recent(
        self,
        conversation_key: str,
        limit: int,
    ) -> tuple[ConversationMessage, ...]:
        if limit <= 0:
            return ()

        messages = self._messages_by_key.get(conversation_key, [])
        return tuple(messages[-limit:])

    async def append(
        self,
        conversation_key: str,
        messages: tuple[ConversationMessage, ...],
    ) -> None:
        if not messages:
            return

        self._messages_by_key.setdefault(conversation_key, []).extend(messages)

    def clear(self) -> None:
        self._messages_by_key.clear()
