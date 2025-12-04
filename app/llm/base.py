from dataclasses import dataclass
from typing import List


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


class LLMClient:
    async def chat(self, messages: List[ChatMessage]) -> str:
        raise NotImplementedError("chat() must be implemented by subclasses")
