from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    content: str
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("The id of the document chunk cannot be empty")
        if not self.content.strip():
            raise ValueError("The content of the document chunk cannot be empty")
