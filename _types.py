import typing as t
from enum import Enum

from pydantic import BaseModel


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class Feedback(BaseModel):
    prompt: str
    improvement: str


class TreeNode(BaseModel):
    children: t.List["TreeNode"]
    conversation: t.List[Message]
    feedback: t.Optional[Feedback]
    # Multiple random document orders
    responses: t.Optional[t.List[str]] 
    on_topic: t.Optional[bool]
    score: t.Optional[float]


class Parameters(BaseModel):
    model: str
    temperature: float = 1.0
    max_tokens: int = 512
    top_p: float = 0.9


ChatFunction = t.Callable[[t.List[Message]], Message]
Conversation = t.List[Message]


class Product(BaseModel):
    category: str
    brand: str
    model: str
    
    def __hash__(self):
        return hash((self.category, self.brand, self.model))
    
    def __eq__(self, other):
        return (
            (self.category, self.brand, self.model) ==
            (other.category, other.brand, other.model)
        )