import os
from typing import Callable, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

PROVIDER_ENV_VARS: Dict[str, str] = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

LOCAL_PROVIDERS = {"ollama"}


def _extract_text(content) -> str:
    """Normalizes langchain response.content across providers.

    Gemini returns a list of content-block dicts (e.g. [{'type': 'text', 'text': ...}]);
    OpenAI/Anthropic/Ollama typically return a plain string. Handling this here keeps
    provider-specific response shapes out of GTMBrain's prompt logic.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content)


def _build_gemini(api_key: Optional[str], **kwargs) -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=kwargs.get("model", "gemini-3-flash-preview"),
        api_key=api_key,
        temperature=kwargs.get("temperature", 0.1),
    )


def _build_openai(api_key: Optional[str], **kwargs) -> BaseChatModel:
    return ChatOpenAI(
        model=kwargs.get("model", "gpt-4o-mini"),
        api_key=api_key,
        temperature=kwargs.get("temperature", 0.1),
    )


def _build_anthropic(api_key: Optional[str], **kwargs) -> BaseChatModel:
    return ChatAnthropic(
        model=kwargs.get("model", "claude-sonnet-4-5"),
        api_key=api_key,
        temperature=kwargs.get("temperature", 0.1),
    )


def _build_ollama(api_key: Optional[str], **kwargs) -> BaseChatModel:
    return ChatOllama(
        model=kwargs.get("model", "qwen3.5:9b-q4_K_M"),
        temperature=kwargs.get("temperature", 0.4),
    )


_PROVIDER_BUILDERS: Dict[str, Callable[..., BaseChatModel]] = {
    "gemini": _build_gemini,
    "openai": _build_openai,
    "anthropic": _build_anthropic,
    "ollama": _build_ollama,
}


class LLMProvider:
    """Wraps a langchain chat model behind a single invoke(template, variables) call."""

    def __init__(self, name: str, model: BaseChatModel):
        self.name = name
        self.model = model

    def invoke(self, template: str, variables: Dict) -> str:
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.model
        response = chain.invoke(variables)
        return _extract_text(response.content)


def detect_available_providers() -> List[str]:
    """Cloud providers whose API key env var is set, plus local providers (no key needed)."""
    available = [name for name, env_var in PROVIDER_ENV_VARS.items() if os.environ.get(env_var)]
    available.extend(LOCAL_PROVIDERS)
    return available


def create_provider(name: str, api_key: Optional[str] = None, **kwargs) -> LLMProvider:
    try:
        builder = _PROVIDER_BUILDERS[name]
    except KeyError:
        raise ValueError(f"Unknown provider '{name}'. Available: {sorted(_PROVIDER_BUILDERS)}") from None
    return LLMProvider(name=name, model=builder(api_key, **kwargs))
