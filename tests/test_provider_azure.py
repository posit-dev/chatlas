import os

import pytest

from chatlas import ChatAzureOpenAI

from ._test_providers import TestChatAzureOpenAI

do_test = os.getenv("TEST_AZURE", "true")
if do_test.lower() == "false":
    pytest.skip("Skipping Azure tests", allow_module_level=True)


def chat_func(system_prompt: str = "Be as terse as possible; no punctuation"):
    return TestChatAzureOpenAI(
        system_prompt=system_prompt,
        endpoint="https://carso-mhl2zrqb-eastus2.cognitiveservices.azure.com/",
        deployment_id="gpt-5-nano",
        api_version="2025-03-01-preview",
    )


@pytest.mark.vcr
def test_azure_simple_request():
    chat = chat_func()
    response = chat.chat("What is 1 + 1?")
    assert "2" == response.get_content()
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert turn.tokens[0] == 26
    assert chat.provider.name == "Azure/OpenAI"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_azure_simple_request_async():
    chat = chat_func()

    response = await chat.chat_async("What is 1 + 1?")
    assert "2" == await response.get_content()
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert turn.tokens[0] == 26


def test_connect_without_openai_key(monkeypatch):
    # Ensure OPENAI_API_KEY is not set
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # This should not raise an error
    chat = chat_func()
    assert chat is not None
