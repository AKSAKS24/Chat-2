from typing import Dict, List
from .base import LLMClient
from .openai_provider import OpenAIClient
from .anthropic_provider import AnthropicClient
from ..config import settings


# ---------------------------------------------------------
# Return available providers and their models
# ---------------------------------------------------------
def list_providers_and_models() -> Dict[str, List[str]]:
    """
    Returns configured providers and supported models.
    Comes from settings.DEFAULT_MODELS.
    """
    return settings.DEFAULT_MODELS


# ---------------------------------------------------------
# Factory to create proper LLM client
# ---------------------------------------------------------
def create_llm_client(provider: str, model: str) -> LLMClient:
    """
    Creates an appropriate LLM client based on provider name.
    Ensures:
      - provider case-insensitive
      - standardized LLMClient API
      - normal chat mode compatibility
    """
    if not provider:
        raise ValueError("LLM provider is required.")
    if not model:
        raise ValueError("LLM model is required.")

    provider = provider.strip().lower()

    # ---- OpenAI ----
    if provider in ("openai", "gpt", "chatgpt"):
        return OpenAIClient(
            model=model,
            api_key=settings.OPENAI_API_KEY
        )

    # ---- Anthropic Claude ----
    if provider in ("anthropic", "claude"):
        return AnthropicClient(
            model=model,
            api_key=settings.ANTHROPIC_API_KEY
        )

    # ---- Unknown provider ----
    raise ValueError(
        f"Unknown provider: {provider}. "
        f"Supported: openai, anthropic"
    )
