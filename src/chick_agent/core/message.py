from datetime import datetime
from pydantic import BaseModel
from typing import Literal, override

MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    content: str
    role: MessageRole
    timestamp: datetime | None = None
    metadata: dict[str, object] | None = None

    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp", datetime.now()),
            metadata=kwargs.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "content": self.content,
        }

    @override
    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"
