from typing import Protocol

from app.domain.entities.question import ConversationMessage


class ConversationStorePort(Protocol):
    """Contrato para persistir historial conversacional por usuario."""

    async def get_recent(
        self,
        conversation_key: str,
        limit: int,
    ) -> tuple[ConversationMessage, ...]:
        """Devuelve los mensajes recientes de una conversación.

        Args:
            conversation_key: Clave normalizada que identifica la conversación.
            limit: Cantidad máxima de mensajes a devolver.

        Returns:
            Tupla con los últimos mensajes en orden cronológico.
        """
        raise NotImplementedError

    async def append(
        self,
        conversation_key: str,
        messages: tuple[ConversationMessage, ...],
    ) -> None:
        """Agrega mensajes al historial de una conversación.

        Args:
            conversation_key: Clave normalizada que identifica la conversación.
            messages: Mensajes a anexar al historial.
        """
        raise NotImplementedError
