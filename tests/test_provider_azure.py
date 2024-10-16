import pytest
from chatlas import ChatAzureOpenAI


# Azure Provider Tests
@pytest.mark.asyncio
async def test_azure_simple_request():
    chat = ChatAzureOpenAI(
        system_prompt="Be as terse as possible; no punctuation",
        endpoint="https://ai-hwickhamai260967855527.openai.azure.com",
        deployment_id="gpt-4o-mini",
    )

    resp = chat.chat("What is 1 + 1?")
    assert "2" in resp
    assert chat.last_turn().tokens == [27, 1]

    resp = await chat.chat_async("What is 1 + 1?")
    assert "2" in resp
    assert chat.last_turn().tokens == [44, 1]
