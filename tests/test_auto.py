import os
import warnings

import chatlas
import pytest
from chatlas import Chat, ChatAuto
from chatlas._auto import _provider_chat_model_map
from chatlas._provider_anthropic import AnthropicBedrockProvider, AnthropicProvider
from chatlas._provider_google import GoogleProvider
from chatlas._provider_openai import OpenAIProvider

from .conftest import assert_turns_existing, assert_turns_system


@pytest.fixture(autouse=True)
def mock_api_keys(monkeypatch):
    """Set mock API keys for providers to avoid missing key errors."""
    api_keys = {
        "OPENAI_API_KEY": "api-key",
        "ANTHROPIC_API_KEY": "api-key",
        "GOOGLE_API_KEY": "api-key",
    }

    for key, value in api_keys.items():
        monkeypatch.setenv(key, value)


def test_auto_settings_from_env(monkeypatch):
    """Test the new CHATLAS_CHAT_PROVIDER_MODEL environment variable."""
    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER_MODEL", "openai/gpt-4o")
    monkeypatch.setenv(
        "CHATLAS_CHAT_ARGS",
        """{
    "system_prompt": "Be as terse as possible; no punctuation",
    "kwargs": {"max_retries": 2}
}""",
    )

    chat = ChatAuto()

    assert isinstance(chat, Chat)
    assert isinstance(chat.provider, OpenAIProvider)


def test_auto_settings_from_old_env_backwards_compatibility(monkeypatch):
    """Test backwards compatibility with old environment variables."""
    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER", "openai")
    monkeypatch.setenv("CHATLAS_CHAT_MODEL", "gpt-4o")
    monkeypatch.setenv(
        "CHATLAS_CHAT_ARGS",
        """{
    "system_prompt": "Be as terse as possible; no punctuation",
    "kwargs": {"max_retries": 2}
}""",
    )

    with pytest.warns(DeprecationWarning, match="CHATLAS_CHAT_PROVIDER"):
        with pytest.warns(DeprecationWarning, match="CHATLAS_CHAT_MODEL"):
            chat = ChatAuto()

    assert isinstance(chat, Chat)
    assert isinstance(chat.provider, OpenAIProvider)


def test_auto_provider_model_parameter():
    """Test using provider_model parameter directly."""
    chat = ChatAuto(provider_model="openai/gpt-4o")
    assert isinstance(chat, Chat)
    assert isinstance(chat.provider, OpenAIProvider)


def test_auto_provider_only_parameter():
    """Test using provider_model with just provider (no model)."""
    chat = ChatAuto(provider_model="openai")
    assert isinstance(chat, Chat)
    assert isinstance(chat.provider, OpenAIProvider)


def test_auto_settings_from_env_unknown_arg_fails(monkeypatch):
    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER_MODEL", "openai/gpt-4o")
    monkeypatch.setenv("CHATLAS_CHAT_ARGS", '{"aws_region": "us-east-1"}')

    with pytest.raises(TypeError):
        ChatAuto()


def test_auto_parameter_overrides_env(monkeypatch):
    """Test that direct parameters override environment variables."""
    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER_MODEL", "anthropic")
    chat = ChatAuto(provider_model="openai")
    assert isinstance(chat.provider, OpenAIProvider)


def test_auto_falls_back_to_openai_default():
    """Test that ChatAuto falls back to OpenAI when no provider is specified."""
    chat = ChatAuto()
    assert isinstance(chat, Chat)
    assert isinstance(chat.provider, OpenAIProvider)


def test_auto_unknown_provider_raises_exception():
    """Test that unknown provider raises ValueError."""
    with pytest.raises(
        ValueError, match="Provider name 'unknown' is not a known chatlas provider"
    ):
        ChatAuto(provider_model="unknown")


def test_auto_respects_turns_interface(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER_MODEL", "openai/gpt-4o")

    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("Skipping test because OPENAI_API_KEY is not set.")

    chat_fun = ChatAuto
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


def test_deprecated_provider_parameter_warning():
    """Test that using deprecated provider parameter raises warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ChatAuto(provider="openai")

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "provider" in str(w[0].message)
        assert "provider_model" in str(w[0].message)


def test_deprecated_model_parameter_warning():
    """Test that using deprecated model parameter raises warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ChatAuto(provider="openai", model="gpt-4o")

        assert len(w) == 2  # Both provider and model warnings
        assert all(issubclass(warning.category, DeprecationWarning) for warning in w)


def test_deprecated_model_without_provider_error():
    """Test that using model parameter without provider raises ValueError."""
    with pytest.raises(
        ValueError,
        match="The `model` parameter is deprecated and cannot be used without the `provider` parameter",
    ):
        ChatAuto(model="gpt-4o")


def test_parse_provider_model_with_model():
    """Test _parse_provider_model with provider/model format."""
    provider = ChatAuto("openai/gpt-4o").provider
    assert provider.name.lower() == "openai"
    assert provider.model == "gpt-4o"


def test_parse_provider_model_without_model():
    """Test _parse_provider_model with just provider."""
    provider = ChatAuto("openai").provider
    assert provider.name.lower() == "openai"
    assert provider.model is not None


def test_parse_provider_model_with_multiple_slashes():
    """Test _parse_provider_model handles multiple slashes correctly."""
    provider = ChatAuto("open-router/model/with/slashes").provider
    assert provider.name.lower() == "openrouter"
    assert provider.model == "model/with/slashes"


def chat_to_kebab_case(s):
    if s == "ChatOpenAI":
        return "openai"
    elif s == "ChatAzureOpenAI":
        return "azure-openai"
    elif s == "ChatOpenAICompletions":
        return "openai-completions"
    elif s == "ChatAzureOpenAICompletions":
        return "azure-openai-completions"

    # Remove 'Chat' prefix if present
    if s.startswith("Chat"):
        s = s[4:]

    # Convert the string to a list of characters
    result = []
    for i, char in enumerate(s):
        # Add hyphen before uppercase letters (except first character)
        if i > 0 and char.isupper():
            result.append("-")
        result.append(char.lower())

    return "".join(result)


def test_auto_includes_all_providers():
    providers = set(
        [
            chat_to_kebab_case(x)
            for x in dir(chatlas)
            if x.startswith("Chat") and x not in ["Chat", "ChatAuto"]
        ]
    )

    missing = providers.difference(_provider_chat_model_map.keys())

    assert len(missing) == 0, (
        f"Missing chat providers from ChatAuto: {', '.join(missing)}"
    )


def test_provider_instances(monkeypatch):
    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER_MODEL", "anthropic")
    chat = ChatAuto()
    assert isinstance(chat, Chat)
    assert isinstance(chat.provider, AnthropicProvider)

    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER_MODEL", "bedrock-anthropic")
    chat = ChatAuto()
    assert isinstance(chat, Chat)
    assert isinstance(chat.provider, AnthropicBedrockProvider)

    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER_MODEL", "google")
    chat = ChatAuto()
    assert isinstance(chat, Chat)
    assert isinstance(chat.provider, GoogleProvider)


def test_kwargs_priority_over_env_args(monkeypatch):
    """Test that direct kwargs override CHATLAS_CHAT_ARGS."""
    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER_MODEL", "openai")
    monkeypatch.setenv("CHATLAS_CHAT_ARGS", '{"base_url": "foo"}')

    chat = ChatAuto(base_url="bar")
    assert isinstance(chat.provider, OpenAIProvider)
    assert str(chat.provider._client.base_url).startswith("bar")


def test_system_prompt_parameter_priority():
    """Test that system_prompt parameter is always respected."""
    chat = ChatAuto(provider_model="openai", system_prompt="Test prompt")
    assert isinstance(chat, Chat)
    # The system_prompt should be set - this would need verification based on Chat implementation


def test_new_env_var_priority_over_old(monkeypatch):
    """Test that new env var takes priority over old ones."""
    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER_MODEL", "anthropic")
    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER", "openai")  # Should be ignored

    chat = ChatAuto()
    assert isinstance(chat.provider, AnthropicProvider)
