import pytest
from chatlas import ChatDatabricks

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_tools_async,
    assert_tools_simple,
    assert_turns_existing,
    assert_turns_system,
    make_vcr_config,
)


# Override VCR config to ignore host - Databricks host varies by environment
# but cassettes were recorded with a specific host
@pytest.fixture(scope="module")
def vcr_config():
    config = make_vcr_config()
    # Remove "host" from match_on since Databricks host varies by environment
    config["match_on"] = ["method", "scheme", "port", "path", "body"]
    return config


@pytest.mark.vcr
def test_databricks_simple_request():
    chat = ChatDatabricks(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert len(turn.tokens) == 3
    assert turn.tokens[0] == 26
    # Not testing turn.tokens[1] because it's not deterministic. Typically 1 or 2.
    assert turn.finish_reason == "stop"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_databricks_simple_streaming_request():
    chat = ChatDatabricks(
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "stop"


@pytest.mark.vcr
def test_databricks_respects_turns_interface():
    chat_fun = ChatDatabricks
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


@pytest.mark.vcr
def test_databricks_empty_response():
    chat = ChatDatabricks()
    chat.chat("Respond with only two blank lines")
    resp = chat.chat("What's 1+1? Just give me the number")
    assert "2" == str(resp).strip()


@pytest.mark.vcr
def test_databricks_tool_variations():
    chat_fun = ChatDatabricks
    assert_tools_simple(chat_fun)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_databricks_tool_variations_async():
    await assert_tools_async(ChatDatabricks)


@pytest.mark.vcr
def test_databricks_data_extraction():
    assert_data_extraction(ChatDatabricks)


@pytest.mark.vcr
def test_databricks_images():
    chat_fun = ChatDatabricks
    assert_images_inline(chat_fun)
    # Remote images don't seem to be supported yet
    # assert_images_remote(chat_fun)


# PDF doesn't seem to be supported yet
#
# def test_databricks_pdf():
#     chat_fun = ChatDatabricks
#     assert_pdf_local(chat_fun)


def test_connect_without_openai_key(monkeypatch):
    # Ensure OPENAI_API_KEY is not set
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # This should not raise an error
    chat = ChatDatabricks()
    assert chat is not None
