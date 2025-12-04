import os
from typing import List, Optional

from .base import LLMClient, ChatMessage
from ..utils.logger import logger

try:
    import anthropic
except ImportError:
    anthropic = None


class AnthropicClient(LLMClient):
    def __init__(self, model: str, api_key: Optional[str] = None):
        if anthropic is None:
            logger.warning("anthropic package not installed; Claude calls will be mocked")
            self.client = None
        else:
            self.client = anthropic.AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    async def chat(self, messages: List[ChatMessage]) -> str:
        if self.client is None:
            joined = "\n".join(f"{m.role}: {m.content}" for m in messages)
            return f"[MOCK CLAUDE {self.model}] Generated TS/FS based on:\n{joined[:1000]}"

        # For Claude, separate system + user/assistant blocks
        system_msg = "\n".join(m.content for m in messages if m.role == "system") or None
        user_blocks = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        resp = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_msg,
            messages=user_blocks,
        )
        return resp.content[0].text
