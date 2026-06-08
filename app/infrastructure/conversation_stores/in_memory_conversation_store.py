from app.domain.entities.question import ConversationMessage


class InMemoryConversationStore: # lo hice asi para simplificar y no introducir otra db para el historial, entonces se reinicia por cada vez que se levanta la api
    """Store de conversación en memoria mientras vive la API."""

    def __init__(self) -> None:
        """Inicializa ``_messages_by_key`` agrupado por clave de conversación."""
        self._messages_by_key: dict[str, list[ConversationMessage]] = {}

    async def get_recent(
        self,
        conversation_key: str,
        limit: int,
    ) -> tuple[ConversationMessage, ...]:
        """Devuelve los últimos mensajes de una conversación.

        Args:
            conversation_key: Clave normalizada del usuario/conversación.
            limit: Cantidad máxima de mensajes recientes a devolver.

        Returns:
            Tupla de mensajes recientes en orden cronológico.
        """
        if limit <= 0:
            return ()

        messages = self._messages_by_key.get(conversation_key, [])
        return tuple(messages[-limit:])

    async def append(
        self,
        conversation_key: str,
        messages: tuple[ConversationMessage, ...],
    ) -> None:
        """Agrega mensajes al historial de una conversación.

        Args:
            conversation_key: Clave normalizada del usuario/conversación.
            messages: Mensajes que se anexan al historial existente.
        """
        if not messages:
            return

        self._messages_by_key.setdefault(conversation_key, []).extend(messages)

    def clear(self) -> None:
        """Elimina todo el historial conversacional."""
        self._messages_by_key.clear()
