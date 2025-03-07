import chatlas
import pytest
from chatlas import Chat, ChatAuto
from chatlas._auto import _provider_chat_model_map
from chatlas._openai import OpenAIProvider

from .conftest import (
    assert_turns_existing,
    assert_turns_system,
)


@pytest.mark.parametrize(
    "provider, model, args",
    [
        # ("bedrock:anthropic", {"model": "anthropic.claude-3-5-sonnet-20240620-v1:0", "aws_region": "us-east-1", "kwargs": {"max_retries": 2}}),
        (
            "openai",
            "gpt-4",
            {"kwargs": {"max_retries": 2}},
        ),
        ("anthropic", None, {"kwargs": {"max_retries": 2}}),
        ("google", None, {}),
        (
            "azure-openai",
            None,
            {"endpoint": "", "deployment_id": "1", "api_version": 1},
        ),
    ],
)
def test_auto_simple_request(provider, model, args):
    chat = ChatAuto(
        provider=provider,
        model=model,
        system_prompt="Be as terse as possible; no punctuation",
        **args,
    )
    assert chat.provider._client.max_retries == 2
    response = chat.chat("What is 1 + 1?")
    assert str(response) == "2"
    turn = chat.get_last_turn()
    assert turn is not None


@pytest.mark.parametrize(
    "provider, args",
    [
        # ("bedrock:anthropic", {"model": "anthropic.claude-3-5-sonnet-20240620-v1:0", "aws_region": "us-east-1", "kwargs": {"max_retries": 2}}),
        ("openai", {"kwargs": {"max_retries": 2}}),
        ("anthropic", {"kwargs": {"max_retries": 2}}),
        ("google", {"kwargs": {"max_retries": None}}),
        (
            "azure-openai",
            {"endpoint": "", "deployment_id": "1", "api_version": 1},
        ),
    ],
)
@pytest.mark.asyncio
async def test_auto_simple_streaming_request(provider, args):
    chat = ChatAuto(
        provider=provider,
        system_prompt="Be as terse as possible; no punctuation",
        **args,
    )
    assert isinstance(chat, Chat)

    assert chat.provider._client.max_retries == 2
    res = []
    foo = await chat.stream_async("What is 1 + 1?")
    async for x in foo:
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None


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

    assert chat.provider._client.max_retries == 2
    response = chat.chat("What is 1 + 1?")
    assert str(response) == "2"
    turn = chat.get_last_turn()
    assert turn is not None


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


def convert_to_kebab_case(arr):
    def process_string(s):
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

    return [process_string(s) for s in arr]


def test_auto_includes_all_providers():
    providers = [x for x in dir(chatlas) if x.startswith("Chat") and x != "Chat"]
    providers = set(convert_to_kebab_case(providers))

    missing = set(_provider_chat_model_map.keys()).difference(providers)

    assert len(missing) == 0, (
        f"Missing chat providers from ChatAuto: {', '.join(missing)}"
    )
