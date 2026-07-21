import warnings

import httpx
import pytest
from chatlas import ChatOpenAI, tool_code_execution, tool_web_search
from chatlas._provider_openai import (
    normalize_finish_reason as openai_normalize_finish_reason,
)
from openai.types.responses import ResponseOutputMessage, ResponseOutputText

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote,
    assert_list_models,
    assert_pdf_local,
    assert_tool_web_search,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_tools_simple_stream_content,
    assert_turns_existing,
    assert_turns_system,
)


def test_normalize_finish_reason_completed():
    assert openai_normalize_finish_reason("completed") == "success"


def test_normalize_finish_reason_incomplete_max_tokens():
    assert (
        openai_normalize_finish_reason("incomplete", "max_output_tokens")
        == "max_tokens"
    )


def test_normalize_finish_reason_incomplete_content_filter():
    assert (
        openai_normalize_finish_reason("incomplete", "content_filter")
        == "content_filter"
    )


def test_normalize_finish_reason_incomplete_unknown_reason():
    assert (
        openai_normalize_finish_reason("incomplete", "some_other_reason")
        == "some_other_reason"
    )


def test_normalize_finish_reason_incomplete_no_reason():
    assert openai_normalize_finish_reason("incomplete", None) == "incomplete"


def test_normalize_finish_reason_passes_through_unknown_status():
    assert openai_normalize_finish_reason("failed") == "failed"
    assert openai_normalize_finish_reason("cancelled") == "cancelled"


def test_normalize_finish_reason_handles_none():
    assert openai_normalize_finish_reason(None) is None


@pytest.mark.vcr
def test_openai_simple_request():
    chat = ChatOpenAI(
        system_prompt="Be as terse as possible; no punctuation",
    )
    chat.chat("What is 1 + 1?")
    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert len(turn.tokens) == 3
    assert turn.tokens[0] == 26
    # Not testing turn.tokens[1] because it's not deterministic. Typically 1 or 2.


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_simple_streaming_request():
    chat = ChatOpenAI(
        system_prompt="Be as terse as possible; no punctuation",
    )
    res = []
    async for x in await chat.stream_async("What is 1 + 1?"):
        res.append(x)
    assert "2" in "".join(res)
    turn = chat.get_last_turn()
    assert turn is not None


@pytest.mark.vcr
def test_openai_respects_turns_interface():
    chat_fun = ChatOpenAI
    assert_turns_system(chat_fun)
    assert_turns_existing(chat_fun)


@pytest.mark.vcr
def test_openai_tool_variations():
    chat_fun = ChatOpenAI
    assert_tools_simple(chat_fun)
    assert_tools_simple_stream_content(chat_fun)
    assert_tools_parallel(chat_fun)
    assert_tools_sequential(chat_fun, total_calls=6)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_tool_variations_async():
    await assert_tools_async(ChatOpenAI)


@pytest.mark.vcr
def test_data_extraction():
    assert_data_extraction(ChatOpenAI)


@pytest.mark.vcr
def test_openai_web_search():
    def chat_fun(**kwargs):
        return ChatOpenAI(model="gpt-4.1", **kwargs)

    assert_tool_web_search(
        chat_fun,
        tool_web_search(),
        hint="The CRAN archive page has this info.",
    )


@pytest.mark.vcr
def test_openai_images():
    chat_fun = ChatOpenAI
    assert_images_inline(chat_fun)
    assert_images_remote(chat_fun)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_logprobs():
    chat = ChatOpenAI()
    chat.set_model_params(log_probs=True)

    pieces = []
    async for x in await chat.stream_async("Hi"):
        pieces.append(x)

    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.completion is not None
    output = turn.completion.output[0]
    assert isinstance(output, ResponseOutputMessage)
    content = output.content[0]
    assert isinstance(content, ResponseOutputText)
    logprobs = content.logprobs
    assert logprobs is not None
    assert len(logprobs) == len(pieces)


@pytest.mark.vcr
def test_openai_pdf():
    assert_pdf_local(ChatOpenAI)


def test_openai_custom_http_client():
    ChatOpenAI(kwargs={"http_client": httpx.AsyncClient()})


@pytest.mark.vcr
def test_openai_list_models():
    assert_list_models(ChatOpenAI)


def test_openai_service_tier():
    chat = ChatOpenAI(service_tier="flex")
    assert chat.kwargs_chat.get("service_tier") == "flex"


@pytest.mark.vcr
def test_openai_service_tier_affects_pricing():
    from chatlas._tokens import get_token_cost

    chat = ChatOpenAI(service_tier="priority")
    chat.chat("What is 1+1?")

    turn = chat.get_last_turn()
    assert turn is not None
    assert turn.tokens is not None
    assert turn.cost is not None

    # Verify that cost was calculated using priority pricing
    tokens = turn.tokens
    priority_cost = get_token_cost("OpenAI", chat.provider.model, tokens, "priority")
    assert priority_cost is not None
    assert turn.cost == priority_cost

    # Verify priority pricing is more expensive than default
    default_cost = get_token_cost("OpenAI", chat.provider.model, tokens, "")
    assert default_cost is not None
    assert turn.cost > default_cost


def test_can_extract_custom_id_from_malformed_json():
    from chatlas._provider_openai_generic import (
        _extract_custom_id,
        _openai_json_fallback,
    )

    # Test _extract_custom_id
    assert _extract_custom_id('{"custom_id": "request-123", ') == "request-123"
    assert _extract_custom_id('{"custom_id":"request-456"}') == "request-456"
    assert _extract_custom_id('{"custom_id" : "request-789" }') == "request-789"
    assert _extract_custom_id("no custom id here") == ""
    assert _extract_custom_id("") == ""

    # Test _openai_json_fallback
    result = _openai_json_fallback('{"custom_id": "request-123", ')
    assert result == {
        "custom_id": "request-123",
        "response": {"status_code": 500},
    }


def test_openai_web_search_call_action_types():
    """Handle non-search web_search_call action types (open_page, find_in_page)."""
    from chatlas._content import ContentToolRequestSearch
    from chatlas._provider_openai import OpenAIProvider

    chat = ChatOpenAI()
    provider = chat.provider
    assert isinstance(provider, OpenAIProvider)

    def make_response(action: dict):
        """Create a minimal Response with a web_search_call output."""
        from openai.types.responses import Response

        return Response.model_validate(
            {
                "id": "resp_1",
                "created_at": 0,
                "model": "gpt-4.1",
                "object": "response",
                "output": [
                    {
                        "id": "ws_1",
                        "type": "web_search_call",
                        "status": "completed",
                        "action": action,
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }
        )

    # search action with query
    resp = make_response({"type": "search", "query": "test query"})
    turn = provider._response_as_turn(resp, has_data_model=False)
    assert isinstance(turn.contents[0], ContentToolRequestSearch)
    assert turn.contents[0].query == "test query"

    # open_page action with url
    resp = make_response({"type": "open_page", "url": "https://example.com"})
    turn = provider._response_as_turn(resp, has_data_model=False)
    assert isinstance(turn.contents[0], ContentToolRequestSearch)
    assert turn.contents[0].query == "https://example.com"

    # find_in_page action with pattern
    resp = make_response(
        {"type": "find_in_page", "pattern": "find this", "url": "https://example.com"}
    )
    turn = provider._response_as_turn(resp, has_data_model=False)
    assert isinstance(turn.contents[0], ContentToolRequestSearch)
    assert turn.contents[0].query == "find this"

    # search action without query but with queries
    resp = make_response({"type": "search", "query": "", "queries": ["first query"]})
    turn = provider._response_as_turn(resp, has_data_model=False)
    assert isinstance(turn.contents[0], ContentToolRequestSearch)
    assert turn.contents[0].query == "first query"

    # fallback to "web search" when no useful info
    resp = make_response({"type": "search", "query": ""})
    turn = provider._response_as_turn(resp, has_data_model=False)
    assert isinstance(turn.contents[0], ContentToolRequestSearch)
    assert turn.contents[0].query == "web search"


def test_openai_function_call_finish_reason_is_tool_use():
    """A completed response containing a function_call should normalize to tool_use."""
    from chatlas._provider_openai import OpenAIProvider
    from openai.types.responses import Response

    chat = ChatOpenAI()
    provider = chat.provider
    assert isinstance(provider, OpenAIProvider)

    resp = Response.model_validate(
        {
            "id": "resp_1",
            "created_at": 0,
            "model": "gpt-4.1",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_date",
                    "arguments": "{}",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
    )
    turn = provider._response_as_turn(resp, has_data_model=False)
    assert turn.finish_reason == "tool_use"


def test_openai_custom_base_url_warning():
    from chatlas._provider_openai import check_base_url

    with pytest.warns(UserWarning, match="ChatOpenAICompletions"):
        check_base_url("http://localhost:8000/v1")

    with pytest.warns(UserWarning, match="ChatOpenAICompletions"):
        check_base_url("https://my-proxy.example.com/v1")

    # Default URL should not warn
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_base_url("https://api.openai.com/v1")


def test_openai_code_execution_call_parses_request_and_response():
    """A code_interpreter_call output splits into request + response content."""
    from chatlas._content import (
        ContentToolRequestCodeExecution,
        ContentToolResponseCodeExecution,
    )
    from chatlas._provider_openai import OpenAIProvider

    chat = ChatOpenAI()
    provider = chat.provider
    assert isinstance(provider, OpenAIProvider)

    def make_response(outputs: list[dict]):
        from openai.types.responses import Response

        return Response.model_validate(
            {
                "id": "resp_1",
                "created_at": 0,
                "model": "gpt-4.1",
                "object": "response",
                "output": [
                    {
                        "id": "ci_1",
                        "type": "code_interpreter_call",
                        "status": "completed",
                        "code": "print(1 + 1)",
                        "container_id": "cntr_abc123",
                        "outputs": outputs,
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }
        )

    resp = make_response([{"type": "logs", "logs": "2"}])
    turn = provider._response_as_turn(resp, has_data_model=False)

    assert len(turn.contents) == 2
    request = turn.contents[0]
    assert isinstance(request, ContentToolRequestCodeExecution)
    assert request.code == "print(1 + 1)"

    response = turn.contents[1]
    assert isinstance(response, ContentToolResponseCodeExecution)
    assert response.output == "2"
    assert response.container_id == "cntr_abc123"


def test_openai_code_execution_response_does_not_duplicate_on_round_trip():
    """The response half shouldn't be resubmitted -- it's bundled into the request's extra."""
    from chatlas._content import ContentToolResponseCodeExecution
    from chatlas._provider_openai import as_input_param

    content = ContentToolResponseCodeExecution(
        output="2", extra={"type": "code_interpreter_call"}
    )
    assert as_input_param(content, "assistant") is None


def test_openai_code_execution_tool_uses_auto_container_by_default():
    from chatlas._provider_openai import OpenAIProvider

    chat = ChatOpenAI()
    chat.register_tool(tool_code_execution())
    provider = chat.provider
    assert isinstance(provider, OpenAIProvider)

    kwargs = provider._chat_perform_args(
        stream=False,
        turns=[],
        tools=chat._tools,  # type: ignore[reportPrivateUsage]
        data_model=None,
        kwargs=None,
    )
    tools = kwargs["tools"]
    assert tools[0]["type"] == "code_interpreter"
    assert tools[0]["container"] == {"type": "auto"}


def test_openai_code_execution_tool_reuses_container_from_turn_history():
    from chatlas._content import ContentToolResponseCodeExecution
    from chatlas._provider_openai import OpenAIProvider
    from chatlas._turn import AssistantTurn

    chat = ChatOpenAI()
    chat.register_tool(tool_code_execution())
    provider = chat.provider
    assert isinstance(provider, OpenAIProvider)

    turns = [
        AssistantTurn(
            [ContentToolResponseCodeExecution(output="1", container_id="cntr_xyz")]
        )
    ]
    kwargs = provider._chat_perform_args(
        stream=False,
        turns=turns,
        tools=chat._tools,  # type: ignore[reportPrivateUsage]
        data_model=None,
        kwargs=None,
    )
    tools = kwargs["tools"]
    assert tools[0]["container"] == "cntr_xyz"
