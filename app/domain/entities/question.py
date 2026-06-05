from dataclasses import dataclass


@dataclass(frozen=True)
class UserQuestion:
    user_name: str
    content: str

    def __post_init__(self) -> None:
        user_name = self.user_name.strip()
        content = self.content.strip()

        if not user_name:
            raise ValueError("The user name cannot be empty")
        if not content:
            raise ValueError("The question content cannot be empty")

        object.__setattr__(self, "user_name", user_name)
        object.__setattr__(self, "content", content)

    @property
    def normalized_content(self) -> str:
        return " ".join(self.content.strip().split()).casefold()
