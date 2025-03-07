import pytest

import chatlas
from chatlas import Chat, ChatAuto
from chatlas._anthropic import AnthropicBedrockProvider, AnthropicProvider
from chatlas._auto import _provider_chat_model_map
from chatlas._google import GoogleProvider
from chatlas._openai import OpenAIProvider

from .conftest import assert_turns_existing, assert_turns_system


def test_auto_settings_from_env(monkeypatch):
    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER", "openai")
    monkeypatch.setenv(
        "CHATLAS_CHAT_ARGS",
        """{
    "model": "gpt-4o",
    "system_prompt": "Be as terse as possible; no punctuation",
    "kwargs": {"max_retries": 2}
}""",
    )

    chat = ChatAuto()

    assert isinstance(chat, Chat)
    assert isinstance(chat.provider, OpenAIProvider)


def test_auto_settings_from_env_unknown_arg_fails(monkeypatch):
    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER", "openai")
    monkeypatch.setenv(
        "CHATLAS_CHAT_ARGS", '{"model": "gpt-4o", "aws_region": "us-east-1"}'
    )

    with pytest.raises(TypeError):
        ChatAuto()


def test_auto_override_provider_with_env(monkeypatch):
    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER", "openai")
    chat = ChatAuto(provider="anthropic")
    assert isinstance(chat.provider, OpenAIProvider)


def test_auto_missing_provider_raises_exception():
    with pytest.raises(ValueError):
        ChatAuto()


def test_auto_respects_turns_interface(monkeypatch):
    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER", "openai")
    monkeypatch.setenv("CHATLAS_CHAT_ARGS", '{"model": "gpt-4o"}')
    chat_fun = ChatAuto
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


def chat_to_kebab_case(s):
    if s == "ChatOpenAI":
        return "openai"
    elif s == "ChatAzureOpenAI":
        return "azure-openai"

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
    providers = [
        chat_to_kebab_case(x)
        for x in dir(chatlas)
        if x.startswith("Chat") and x != "Chat"
    ]
    providers = set(providers)

    missing = set(_provider_chat_model_map.keys()).difference(providers)

    assert len(missing) == 0, (
        f"Missing chat providers from ChatAuto: {', '.join(missing)}"
    )


def test_provider_instances(monkeypatch):
    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER", "anthropic")
    chat = ChatAuto()
    assert isinstance(chat, Chat)
    assert isinstance(chat.provider, AnthropicProvider)

    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER", "bedrock-anthropic")
    chat = ChatAuto()
    assert isinstance(chat, Chat)
    assert isinstance(chat.provider, AnthropicBedrockProvider)

    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER", "google")
    chat = ChatAuto()
    assert isinstance(chat, Chat)
    assert isinstance(chat.provider, GoogleProvider)
