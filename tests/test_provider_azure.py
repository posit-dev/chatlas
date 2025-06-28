import os

import pytest

from chatlas import ChatAzureOpenAI

do_test = os.getenv("TEST_AZURE", "true")
if do_test.lower() == "false":
    pytest.skip("Skipping Azure tests", allow_module_level=True)


def test_azure_simple_request():
    chat = ChatAzureOpenAI(
        system_prompt="Be as terse as possible; no punctuation",
        endpoint="https://chatlas-testing.openai.azure.com",
        deployment_id="gpt-4o-mini",
        api_version="2024-08-01-preview",
    )

    response = chat.chat("What is 1 + 1?")
    assert "2" == response.get_content()
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens == (27, 2)
    assert chat.provider.name == "OpenAIAzure"


@pytest.mark.asyncio
async def test_azure_simple_request_async():
    chat = ChatAzureOpenAI(
        system_prompt="Be as terse as possible; no punctuation",
        endpoint="https://chatlas-testing.openai.azure.com",
        deployment_id="gpt-4o-mini",
        api_version="2024-08-01-preview",
    )

    response = await chat.chat_async("What is 1 + 1?")
    assert "2" == await response.get_content()
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens == (27, 2)
