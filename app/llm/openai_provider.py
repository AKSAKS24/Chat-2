import os
from typing import List, Optional

from .base import LLMClient, ChatMessage
from ..utils.logger import logger

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


class OpenAIClient(LLMClient):
    def __init__(self, model: str, api_key: Optional[str] = None):
        if AsyncOpenAI is None:
            logger.warning("openai package not installed; OpenAI calls will be mocked")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    async def chat(self, messages: List[ChatMessage]) -> str:
        if self.client is None:
            # Fallback mock for development without keys
            joined = "\n".join(f"{m.role}: {m.content}" for m in messages)
            return f"[MOCK OPENAI {self.model}] Generated TS/FS based on:\n{joined[:1000]}"

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=0.1,
        )
        return resp.choices[0].message.content
