import os
from pydantic import BaseModel


class Config(BaseModel):
    default_model: str = "deepseek-chat"
    default_provider: str = "deepseek"
    temperature: float = 0.7
    max_tokens: int | None = None
    debug: bool = False
    log_level: str = "INFO"
    max_history_length: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS", 4096))
            if os.getenv("MAX_TOKENS")
            else None,
        )

    def to_dict(self) -> dict[str, object]:
        return self.model_dump()
