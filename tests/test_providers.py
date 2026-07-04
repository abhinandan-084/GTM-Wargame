import pytest

import gtm_boardroom.agents.providers as providers
from gtm_boardroom.agents.providers import (
    PROVIDER_ENV_VARS,
    LOCAL_PROVIDERS,
    LLAMACPP_MODEL_PATH_ENV_VAR,
    _extract_text,
    create_provider,
    detect_available_providers,
)


def test_extract_text_from_plain_string():
    assert _extract_text("hello world") == "hello world"


def test_extract_text_from_gemini_style_content_blocks():
    content = [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]
    assert _extract_text(content) == "hello world"


def test_extract_text_from_list_of_strings():
    assert _extract_text(["hello ", "world"]) == "hello world"


def test_extract_text_ignores_non_text_blocks():
    content = [{"type": "tool_use", "input": {}}, {"type": "text", "text": "answer"}]
    assert _extract_text(content) == "answer"


def test_extract_text_falls_back_to_str_for_unknown_shape():
    assert _extract_text(42) == "42"


def test_create_provider_raises_on_unknown_name():
    with pytest.raises(ValueError):
        create_provider("not-a-real-provider")


def test_detect_available_providers_only_env_backed_when_no_keys_set(monkeypatch):
    for env_var in PROVIDER_ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)

    available = detect_available_providers()

    assert set(available) == LOCAL_PROVIDERS


def test_detect_available_providers_includes_provider_with_key_set(monkeypatch):
    for env_var in PROVIDER_ENV_VARS.values():
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    available = detect_available_providers()

    assert "anthropic" in available
    assert "gemini" not in available
    assert "openai" not in available
    assert LOCAL_PROVIDERS.issubset(set(available))


def test_create_provider_builds_model_without_network_call():
    provider = create_provider("gemini", api_key="dummy-key-not-real")
    assert provider.name == "gemini"
    assert provider.model is not None


def test_create_provider_llamacpp_raises_for_missing_model_file():
    with pytest.raises(Exception):
        create_provider("llamacpp", model_path="/nonexistent/path/model.gguf")


def test_llamacpp_model_path_precedence_kwarg_over_env_over_default(monkeypatch):
    captured_kwargs = {}

    class FakeChatLlamaCpp:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setattr(providers, "ChatLlamaCpp", FakeChatLlamaCpp)

    # Default, when nothing else is set
    monkeypatch.delenv(LLAMACPP_MODEL_PATH_ENV_VAR, raising=False)
    create_provider("llamacpp")
    assert captured_kwargs["model_path"].endswith("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

    # Env var overrides the default
    monkeypatch.setenv(LLAMACPP_MODEL_PATH_ENV_VAR, "/env/model.gguf")
    create_provider("llamacpp")
    assert captured_kwargs["model_path"] == "/env/model.gguf"

    # Explicit kwarg overrides both
    create_provider("llamacpp", model_path="/explicit/model.gguf")
    assert captured_kwargs["model_path"] == "/explicit/model.gguf"
