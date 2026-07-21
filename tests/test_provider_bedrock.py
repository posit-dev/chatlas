import pytest
from chatlas import ChatBedrockAnthropic

from ._vcr_helpers_aws import _filter_aws_response, _scrub_aws_request
from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote_error,
    assert_list_models,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_turns_existing,
    assert_turns_system,
    make_vcr_config,
)


@pytest.fixture(scope="module")
def vcr_config():
    """AWS-specific VCR configuration with credential scrubbing."""
    config = make_vcr_config()
    config["before_record_response"] = _filter_aws_response
    config["before_record_request"] = _scrub_aws_request
    return config


class TestBedrockCacheDefault:
    def test_default_enables_caching(self):
        chat = ChatBedrockAnthropic(
            aws_secret_key="fake",
            aws_access_key="fake",
            aws_region="us-east-1",
        )
        assert chat.provider._cache == "5m"
        assert chat.provider._cache_control() == {"type": "ephemeral", "ttl": "5m"}

    def test_none_disables_caching(self):
        chat = ChatBedrockAnthropic(
            cache="none",
            aws_secret_key="fake",
            aws_access_key="fake",
            aws_region="us-east-1",
        )
        assert chat.provider._cache == "none"
        assert chat.provider._cache_control() is None


# ---------------------------------------------------------------------------
# Live API tests (require Bedrock credentials)
# ---------------------------------------------------------------------------

_has_bedrock_credentials = True
try:
    _chat = ChatBedrockAnthropic()
    _chat.chat("What is 1 + 1?")
except Exception:
    _has_bedrock_credentials = False

requires_bedrock = pytest.mark.skipif(
    not _has_bedrock_credentials,
    reason="Bedrock credentials aren't configured",
)


@requires_bedrock
@pytest.mark.vcr
def test_anthropic_simple_request():
    chat = ChatBedrockAnthropic(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens == (26, 5, 0)


@requires_bedrock
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_simple_streaming_request():
    chat = ChatBedrockAnthropic(
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    foo = await chat.stream_async("What is 1 + 1?")
    async for x in foo:
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.finish_reason == "success"


@requires_bedrock
@pytest.mark.vcr
def test_anthropic_respects_turns_interface():
    chat_fun = ChatBedrockAnthropic
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


@requires_bedrock
@pytest.mark.vcr
def test_anthropic_tool_variations():
    chat_fun = ChatBedrockAnthropic
    assert_tools_simple(chat_fun)
    assert_tools_parallel(chat_fun)
    assert_tools_sequential(chat_fun, total_calls=6)


@requires_bedrock
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_anthropic_tool_variations_async():
    await assert_tools_async(ChatBedrockAnthropic)


@requires_bedrock
@pytest.mark.vcr
def test_data_extraction():
    assert_data_extraction(ChatBedrockAnthropic)


@requires_bedrock
@pytest.mark.vcr
def test_anthropic_images():
    chat_fun = ChatBedrockAnthropic
    assert_images_inline(chat_fun)
    assert_images_remote_error(chat_fun)


@requires_bedrock
@pytest.mark.vcr
def test_anthropic_models():
    assert_list_models(ChatBedrockAnthropic)


@requires_bedrock
@pytest.mark.vcr
def test_reasoning():
    from chatlas._content import ContentThinking

    chat = ChatBedrockAnthropic(reasoning=4000)
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    thinking = [c for c in turn.contents if isinstance(c, ContentThinking)]
    assert len(thinking) == 1
    assert len(thinking[0].thinking) > 0
