import pytest
from chatlas import ChatAuto
from chatlas._openai import OpenAIProvider

from .conftest import (
    assert_turns_existing,
    assert_turns_system,
)

@pytest.mark.parametrize("provider, args", [
    # ("bedrock:anthropic", {"model": "anthropic.claude-3-5-sonnet-20240620-v1:0", "aws_region": "us-east-1", "kwargs": {"max_retries": 2}}),
    ("openai", {"model": "gpt-4o", "kwargs": {"max_retries": 2}}), 
    ("anthropic", {"kwargs": {"max_retries": 2}}),
    ("google", {"kwargs": {"max_retries": 2}}),
    ("azure:openai", {"kwargs": {"max_retries": 2}}),
])
def test_auto_simple_request(provider, args):
    chat = ChatAuto(
        provider=provider,
        system_prompt="Be as terse as possible; no punctuation",
        **args,
    )
    assert chat.provider._client.max_retries == 2
    response = chat.chat("What is 1 + 1?")
    assert str(response) == "2"
    turn = chat.get_last_turn()
    assert turn is not None


@pytest.mark.parametrize("provider, args", [
    # ("bedrock:anthropic", {"model": "anthropic.claude-3-5-sonnet-20240620-v1:0", "aws_region": "us-east-1", "kwargs": {"max_retries": 2}}),
    ("openai", {"kwargs": {"max_retries": 2}}), 
    ("anthropic", {"kwargs": {"max_retries": 2}}),
    ("google", {"kwargs": {"max_retries": 2}}),
    ("azure:openai", {"kwargs": {"max_retries": 2}}),
])
@pytest.mark.asyncio
async def test_auto_simple_streaming_request(provider, args):
    chat = ChatAuto(
        provider=provider,
        system_prompt="Be as terse as possible; no punctuation",
        **args
    )
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
    monkeypatch.setenv("CHATLAS_CHAT_ARGS", '''{
    "model": "gpt-4o", 
    "system_prompt": "Be as terse as possible; no punctuation",
    "kwargs": {"max_retries": 2}
}''')

    chat = ChatAuto()

    assert chat.provider._client.max_retries == 2
    response = chat.chat("What is 1 + 1?")
    assert str(response) == "2"
    turn = chat.get_last_turn()
    assert turn is not None

def test_auto_settings_from_env_unknown_arg_fails(monkeypatch):
    monkeypatch.setenv("CHATLAS_CHAT_PROVIDER", "openai")
    monkeypatch.setenv("CHATLAS_CHAT_ARGS", '{"model": "gpt-4o", "aws_region": "us-east-1"}')

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


