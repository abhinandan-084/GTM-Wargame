import pytest

from gtm_boardroom.agents.providers import (
    PROVIDER_ENV_VARS,
    LOCAL_PROVIDERS,
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
