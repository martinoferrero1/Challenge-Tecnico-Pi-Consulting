from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
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
    user_name: str
    content: str
    conversation_history: tuple[ConversationMessage, ...] = ()

    def __post_init__(self) -> None:
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
        return " ".join(self.content.strip().split()).casefold()
