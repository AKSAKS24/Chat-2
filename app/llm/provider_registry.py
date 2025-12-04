from typing import Dict, List
from .base import LLMClient
from .openai_provider import OpenAIClient
from .anthropic_provider import AnthropicClient
from ..config import settings


def list_providers_and_models() -> Dict[str, List[str]]:
    return settings.DEFAULT_MODELS


def create_llm_client(provider: str, model: str) -> LLMClient:
    provider = provider.lower()
    if provider == "openai":
        return OpenAIClient(model=model, api_key=settings.OPENAI_API_KEY)
    if provider in ("claude", "anthropic"):
        return AnthropicClient(model=model, api_key=settings.ANTHROPIC_API_KEY)
    raise ValueError(f"Unknown provider: {provider}")
