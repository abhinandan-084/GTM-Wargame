import os
from typing import Callable, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

PROVIDER_ENV_VARS: Dict[str, str] = {
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

LOCAL_PROVIDERS = {"llamacpp"}

LLAMACPP_MODEL_PATH_ENV_VAR = "LLAMACPP_MODEL_PATH"
DEFAULT_LLAMACPP_MODEL_PATH = "~/llama.cpp/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"


def _extract_text(content) -> str:
    """Normalizes langchain response.content across providers.

    Gemini returns a list of content-block dicts (e.g. [{'type': 'text', 'text': ...}]);
    OpenAI/Anthropic/llama.cpp typically return a plain string. Handling this here keeps
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


def _key_kwarg(api_key: Optional[str]) -> Dict[str, str]:
    # Passing api_key=None explicitly suppresses the langchain classes' own
    # env-var fallback (observed with ChatGoogleGenerativeAI), so the kwarg
    # must be omitted entirely when no key is given.
    return {} if api_key is None else {"api_key": api_key}


def _build_gemini(api_key: Optional[str], **kwargs) -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=kwargs.get("model", "gemini-3-flash-preview"),
        temperature=kwargs.get("temperature", 0.1),
        **_key_kwarg(api_key),
    )


def _build_openai(api_key: Optional[str], **kwargs) -> BaseChatModel:
    return ChatOpenAI(
        model=kwargs.get("model", "gpt-4o-mini"),
        temperature=kwargs.get("temperature", 0.1),
        **_key_kwarg(api_key),
    )


def _build_anthropic(api_key: Optional[str], **kwargs) -> BaseChatModel:
    return ChatAnthropic(
        model=kwargs.get("model", "claude-sonnet-4-5"),
        temperature=kwargs.get("temperature", 0.1),
        **_key_kwarg(api_key),
    )


def _build_llamacpp(api_key: Optional[str], **kwargs) -> BaseChatModel:
    model_path = (
        kwargs.get("model_path")
        or os.environ.get(LLAMACPP_MODEL_PATH_ENV_VAR)
        or DEFAULT_LLAMACPP_MODEL_PATH
    )
    return ChatLlamaCpp(
        model_path=os.path.expanduser(model_path),
        temperature=kwargs.get("temperature", 0.4),
        n_ctx=kwargs.get("n_ctx", 8192),
        max_tokens=kwargs.get("max_tokens", 1024),
        n_gpu_layers=kwargs.get("n_gpu_layers", 0),
        verbose=kwargs.get("verbose", False),
    )


_PROVIDER_BUILDERS: Dict[str, Callable[..., BaseChatModel]] = {
    "gemini": _build_gemini,
    "openai": _build_openai,
    "anthropic": _build_anthropic,
    "llamacpp": _build_llamacpp,
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
