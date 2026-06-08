from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationMessage:
    """Mensaje previo de una conversación persistida por usuario.

    Atributos:
        role: Rol del mensaje dentro de la conversación. Solo acepta ``user``
            o ``assistant``.
        content: Texto del mensaje ya normalizado y no vacío.
    """

    role: str
    content: str

    def __post_init__(self) -> None:
        """Normaliza y valida ``role`` y ``content``.

        Raises:
            ValueError: Si ``role`` no es ``user`` ni ``assistant``, o si
                ``content`` queda vacío luego de hacer ``strip``.
        """
        role = self.role.strip().casefold()
        content = self.content.strip()

        if role not in {"user", "assistant"}:
            raise ValueError("The conversation message role must be user or assistant")
        if not content:
            raise ValueError("The conversation message content cannot be empty")

        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", content)


@dataclass(frozen=True)
class UserQuestion:
    """Pregunta del usuario con historial conversacional opcional.

    Atributos:
        user_name: Identificador del usuario usado también como clave de
            conversación en memoria.
        content: Pregunta actual enviada por el usuario.
        conversation_history: Mensajes previos ya asociados a la pregunta
            cuando el flujo usa contexto conversacional.
    """

    user_name: str
    content: str
    conversation_history: tuple[ConversationMessage, ...] = ()

    def __post_init__(self) -> None:
        """Normaliza ``user_name``, ``content`` y ``conversation_history``.

        Raises:
            ValueError: Si ``user_name`` o ``content`` quedan vacíos luego de
                hacer ``strip``.
        """
        user_name = self.user_name.strip()
        content = self.content.strip()
        conversation_history = tuple(self.conversation_history)

        if not user_name:
            raise ValueError("The user name cannot be empty")
        if not content:
            raise ValueError("The question content cannot be empty")

        object.__setattr__(self, "user_name", user_name)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "conversation_history", conversation_history)

    @property
    def normalized_content(self) -> str:
        """Devuelve ``content`` normalizado para cache y comparaciones.

        Returns:
            Texto de ``content`` sin espacios repetidos y en minúsculas
            case-insensitive.
        """
        return " ".join(self.content.strip().split()).casefold()
