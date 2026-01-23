from abc import ABC, abstractmethod

from pydantic import BaseModel


class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: object = None


class Tool(ABC):
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: dict[str, object]) -> str:
        pass

    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        pass

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [param.model_dump() for param in self.get_parameters()],
        }
