from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Sequence
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("opentelemetry.sdk")

from chatlas import ChatAnthropic, ChatOpenAI, _chat, _otel
from chatlas._content import (
    Content,
    ContentImageInline,
    ContentText,
    ContentThinkingDelta,
    ContentToolRequest,
    ContentToolResult,
    ToolInfo,
)
from chatlas._stream_controller import StreamController
from chatlas._tools import ToolBuiltIn
from chatlas._turn import AssistantTurn, Turn, user_turn
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind
from pydantic import BaseModel


@pytest.fixture(autouse=True)
def otel_setup():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    orig_tracer = _otel.tracer
    _otel.tracer = provider.get_tracer("co.posit.python-package.chatlas")

    yield exporter

    exporter.clear()
    _otel.tracer = orig_tracer


@pytest.fixture
def otel_metrics():
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    orig_meter = _otel.meter
    _otel.meter = provider.get_meter("co.posit.python-package.chatlas")
    # Instruments are cached per meter; don't leak them across tests.
    _otel._histograms.clear()
    _otel._counters.clear()

    yield reader

    _otel.meter = orig_meter
    _otel._histograms.clear()
    _otel._counters.clear()


def _metric_points(reader: InMemoryMetricReader) -> dict[str, list[Any]]:
    """Flatten recorded metrics into data point lists named by instrument."""
    data = reader.get_metrics_data()
    points: dict[str, list[Any]] = {}
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                points.setdefault(metric.name, []).extend(metric.data.data_points)
    return points


def _mock_completion(
    chat: ChatOpenAI,
    tokens: tuple[int, int, int] = (10, 5, 3),
    reasoning: int | None = 2,
):
    """Mock the (non-streaming) provider call to return a completion with
    known usage info. ``tokens`` is (input, output, cached_input)."""
    usage = SimpleNamespace(
        input_tokens=tokens[0] + tokens[2],
        output_tokens=tokens[1],
        input_tokens_details=SimpleNamespace(cached_tokens=tokens[2]),
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )
    completion = SimpleNamespace(
        model="gpt-4o-mini-2024-07-18",
        id="resp_123",
        usage=usage,
        service_tier=None,
    )

    def fake_value_turn(completion, has_data_model, turns=()):
        return AssistantTurn(
            "hi",
            tokens=tokens,
            finish_reason="success",
            completion=completion,
        )

    chat.provider.chat_perform = lambda *args, **kwargs: completion  # type: ignore[method-assign]
    chat.provider.value_turn = fake_value_turn  # type: ignore[method-assign]
    return completion


def _mock_streaming_provider(chat: ChatOpenAI, n_chunks: int = 3):
    def streaming_perform(*args, **kwargs):
        def gen():
            for _ in range(n_chunks):
                yield object()

        return gen()

    chat.provider.chat_perform = streaming_perform  # type: ignore[method-assign]
    chat.provider.stream_content = lambda chunk, completion, turns=(): [  # type: ignore[method-assign]
        ContentText(text="hi")
    ]
    chat.provider.stream_merge_chunks = (  # type: ignore[method-assign]
        lambda result, chunk: chunk
    )
    chat.provider.stream_turn = (  # type: ignore[method-assign]
        lambda completion, has_data_model, turns=(): AssistantTurn(
            "hi", finish_reason="success"
        )
    )


def _finished_span(exporter: InMemorySpanExporter, prefix: str):
    return next(
        s for s in exporter.get_finished_spans() if s.name.startswith(prefix)
    )


@pytest.mark.vcr
def test_span_hierarchy_with_tools(otel_setup: InMemorySpanExporter):
    chat = ChatOpenAI(
        model="gpt-4o-mini",
        system_prompt="Always use the get_date tool to answer questions about the date.",
    )

    def get_date() -> str:
        "Return the current date"
        return "2026-05-12"

    chat.register_tool(get_date)
    chat.chat("What is today's date?")

    spans = otel_setup.get_finished_spans()
    span_names = [s.name for s in spans]

    agent_spans = [s for s in spans if s.name == "invoke_agent"]
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    tool_spans = [s for s in spans if s.name.startswith("execute_tool ")]

    assert len(agent_spans) == 1, (
        f"Expected 1 invoke_agent span, got {len(agent_spans)}. Spans: {span_names}"
    )
    assert len(chat_spans) >= 2, (
        f"Expected >=2 chat spans, got {len(chat_spans)}. Spans: {span_names}"
    )
    assert len(tool_spans) >= 1, (
        f"Expected >=1 execute_tool span, got {len(tool_spans)}. Spans: {span_names}"
    )

    agent_span = agent_spans[0]
    from opentelemetry.trace import SpanContext

    assert agent_span.context is not None
    agent_ctx = cast(SpanContext, agent_span.context)

    for s in chat_spans:
        assert s.parent is not None, f"Chat span {s.name!r} has no parent"
        parent = cast(SpanContext, s.parent)
        assert parent.span_id == agent_ctx.span_id, (
            f"Chat span {s.name!r} parent span_id mismatch"
        )
    for s in tool_spans:
        assert s.parent is not None, f"Tool span {s.name!r} has no parent"
        parent = cast(SpanContext, s.parent)
        assert parent.span_id == agent_ctx.span_id, (
            f"Tool span {s.name!r} parent span_id mismatch"
        )


@pytest.mark.vcr
def test_token_usage_recorded(otel_setup: InMemorySpanExporter):
    chat = ChatOpenAI(model="gpt-4o-mini")
    chat.chat("Say hello.")

    spans = otel_setup.get_finished_spans()
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1

    # Check the last chat span (which has usage info from the final response)
    chat_span = chat_spans[-1]
    attrs = chat_span.attributes or {}
    input_tokens = attrs.get("gen_ai.usage.input_tokens")
    output_tokens = attrs.get("gen_ai.usage.output_tokens")
    assert isinstance(input_tokens, (int, float)) and input_tokens > 0, (
        f"Expected input_tokens > 0, got {input_tokens!r}"
    )
    assert isinstance(output_tokens, (int, float)) and output_tokens > 0, (
        f"Expected output_tokens > 0, got {output_tokens!r}"
    )


@pytest.mark.vcr
def test_content_capture_off_by_default(otel_setup: InMemorySpanExporter):
    chat = ChatOpenAI(model="gpt-4o-mini")
    chat.chat("Say hello.")

    spans = otel_setup.get_finished_spans()
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1

    for chat_span in chat_spans:
        attrs = chat_span.attributes or {}
        assert "gen_ai.system_instructions" not in attrs, (
            "gen_ai.system_instructions should not be recorded by default"
        )
        assert "gen_ai.input.messages" not in attrs, (
            "gen_ai.input.messages should not be recorded by default"
        )
        assert "gen_ai.output.messages" not in attrs, (
            "gen_ai.output.messages should not be recorded by default"
        )


@pytest.mark.vcr
def test_content_capture_enabled(
    otel_setup: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")

    chat = ChatOpenAI(model="gpt-4o-mini", system_prompt="Be terse.")
    chat.chat("Say hello.")

    spans = otel_setup.get_finished_spans()
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1

    # The first chat span should capture input content (system instructions + messages)
    first_chat_span = chat_spans[0]
    attrs = first_chat_span.attributes or {}

    assert "gen_ai.system_instructions" in attrs, (
        "gen_ai.system_instructions should be recorded when content capture is enabled"
    )
    assert "gen_ai.input.messages" in attrs, (
        "gen_ai.input.messages should be recorded when content capture is enabled"
    )

    # All chat spans should have output messages
    for chat_span in chat_spans:
        span_attrs = chat_span.attributes or {}
        assert "gen_ai.output.messages" in span_attrs, (
            f"gen_ai.output.messages should be recorded; span: {chat_span.name}"
        )

    # Verify structure of system instructions
    sys_instructions = json.loads(str(attrs["gen_ai.system_instructions"]))
    assert isinstance(sys_instructions, list)
    assert len(sys_instructions) >= 1
    assert sys_instructions[0]["type"] == "text"

    # Verify structure of input messages
    input_msgs = json.loads(str(attrs["gen_ai.input.messages"]))
    assert isinstance(input_msgs, list)
    assert len(input_msgs) >= 1
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["type"] == "text"

    # Verify structure of output messages
    output_msgs = json.loads(str(chat_spans[-1].attributes["gen_ai.output.messages"]))  # type: ignore[index]
    assert isinstance(output_msgs, list)
    assert len(output_msgs) >= 1
    assert output_msgs[0]["role"] == "assistant"


def test_missing_tool_error_recorded(otel_setup: InMemorySpanExporter):
    chat = ChatOpenAI(model="gpt-4o-mini")
    request = ContentToolRequest(id="missing-tool", name="missing_tool", arguments={})

    results = list(chat._invoke_tool(request))

    assert len(results) == 1
    assert isinstance(results[0].error, RuntimeError)

    spans = otel_setup.get_finished_spans()
    tool_spans = [s for s in spans if s.name == "execute_tool missing_tool"]
    assert len(tool_spans) == 1

    tool_span = tool_spans[0]
    assert tool_span.status.status_code.name == "ERROR"
    attrs = tool_span.attributes or {}
    assert attrs.get("error.type") == "RuntimeError"


def test_builtin_tool_error_recorded(otel_setup: InMemorySpanExporter):
    chat = ChatOpenAI(model="gpt-4o-mini")
    chat.register_tool(ToolBuiltIn(name="builtin_tool", definition={}))
    request = ContentToolRequest(id="builtin-tool", name="builtin_tool", arguments={})

    results = list(chat._invoke_tool(request))

    assert len(results) == 1
    assert isinstance(results[0].error, RuntimeError)

    spans = otel_setup.get_finished_spans()
    tool_spans = [s for s in spans if s.name == "execute_tool builtin_tool"]
    assert len(tool_spans) == 1

    tool_span = tool_spans[0]
    assert tool_span.status.status_code.name == "ERROR"
    attrs = tool_span.attributes or {}
    assert attrs.get("error.type") == "RuntimeError"


@pytest.mark.vcr
def test_tool_error_recorded(otel_setup: InMemorySpanExporter):
    chat = ChatOpenAI(
        model="gpt-4o-mini",
        system_prompt="Always use the fail_tool to answer. Don't retry if it errors.",
    )

    def fail_tool() -> str:
        "A tool that always fails"
        raise ValueError("intentional test error")

    chat.register_tool(fail_tool)
    chat.chat("Please call the fail_tool.")

    spans = otel_setup.get_finished_spans()
    tool_spans = [s for s in spans if s.name.startswith("execute_tool ")]
    assert len(tool_spans) >= 1

    error_spans = [s for s in tool_spans if s.status.status_code.name == "ERROR"]
    assert len(error_spans) >= 1, (
        f"Expected at least 1 ERROR tool span. Tool spans: {[(s.name, s.status.status_code) for s in tool_spans]}"
    )

    error_span = error_spans[0]
    attrs = error_span.attributes or {}
    assert attrs.get("error.type") == "ValueError", (
        f"Expected error.type == 'ValueError', got {attrs.get('error.type')!r}"
    )


@pytest.mark.vcr
def test_streaming_span_lifecycle(otel_setup: InMemorySpanExporter):
    chat = ChatOpenAI(model="gpt-4o-mini")
    result = chat.stream("Say hello.")
    # Consume the stream fully
    "".join(result)

    spans = otel_setup.get_finished_spans()
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) == 1, (
        f"Expected exactly 1 chat span for a simple stream, got {len(chat_spans)}"
    )
    # Verify the span is finished (it should be in finished spans already)
    chat_span = chat_spans[0]
    assert chat_span.end_time is not None, (
        "Chat span should be finished after stream is consumed"
    )
    # Successful requests are marked OK (matching ellmer), so dashboards can
    # distinguish them from spans still in flight.
    assert chat_span.status.status_code.name == "OK", (
        f"Successful chat span should be OK, got {chat_span.status.status_code.name}"
    )
    # Time to first token is recorded on the streamed chat span.
    attrs = chat_span.attributes or {}
    ttft = attrs.get("gen_ai.response.time_to_first_chunk")
    assert isinstance(ttft, float) and ttft > 0, (
        f"Expected gen_ai.response.time_to_first_chunk > 0, got {ttft!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "first_content",
    [ContentText(text="hi"), ContentThinkingDelta(thinking="hmm")],
    ids=["text", "thinking"],
)
async def test_ttft_recorded_at_first_non_empty_token(
    otel_setup: InMemorySpanExporter,
    otel_metrics: InMemoryMetricReader,
    monkeypatch: pytest.MonkeyPatch,
    async_mode: bool,
    first_content: ContentText | ContentThinkingDelta,
) -> None:
    chat = ChatOpenAI(model="gpt-4o-mini")
    now = 0.0
    clock = SimpleNamespace(monotonic=lambda: now)
    monkeypatch.setattr(_chat, "time", clock)
    monkeypatch.setattr(_otel, "time", clock)

    def chunks() -> Iterator[object]:
        nonlocal now
        for timestamp in (2.0, 5.0, 9.0):
            now = timestamp
            yield object()

    def streaming_perform(*args: object, **kwargs: object) -> Iterator[object]:
        nonlocal now
        now = 1.0
        return chunks()

    async def streaming_perform_async(
        *args: object, **kwargs: object
    ) -> AsyncIterator[object]:
        nonlocal now
        now = 1.0

        async def gen() -> AsyncIterator[object]:
            for chunk in chunks():
                yield chunk

        return gen()

    content_chunks: list[list[Content]] = [
        [ContentText.model_construct(text="")],
        [first_content],
        [ContentText(text="there")],
    ]
    stream_contents = iter(content_chunks)

    def stream_content(
        chunk: object, completion: object, turns: Sequence[Turn] = ()
    ) -> list[Content]:
        return next(stream_contents)

    def stream_merge_chunks(result: object, chunk: object) -> object:
        return chunk

    def stream_turn(
        completion: object, has_data_model: bool, turns: Sequence[Turn] = ()
    ) -> AssistantTurn[object]:
        return AssistantTurn("hi there")

    monkeypatch.setattr(chat.provider, "chat_perform", streaming_perform)
    monkeypatch.setattr(chat.provider, "chat_perform_async", streaming_perform_async)
    monkeypatch.setattr(chat.provider, "stream_content", stream_content)
    monkeypatch.setattr(chat.provider, "stream_merge_chunks", stream_merge_chunks)
    monkeypatch.setattr(chat.provider, "stream_turn", stream_turn)

    if async_mode:
        async for _ in await chat.stream_async("Say hello."):
            pass
    else:
        list(chat.stream("Say hello."))

    spans = otel_setup.get_finished_spans()
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) == 1
    attrs = chat_spans[0].attributes or {}
    assert attrs.get("gen_ai.response.time_to_first_chunk") == 5.0
    points = _metric_points(otel_metrics)["gen_ai.client.inference.time_to_first_chunk"]
    assert len(points) == 1
    assert points[0].count == 1
    assert points[0].sum == 2.0


def test_chat_error_recorded(otel_setup: InMemorySpanExporter):
    # A provider failure during a (non-streaming) chat call should mark the chat
    # span as errored with the GenAI `error.type` attribute, mirroring how tool
    # failures are recorded on the execute_tool span.
    chat = ChatOpenAI(model="gpt-4o-mini")

    def boom(*args, **kwargs):
        raise RuntimeError("provider blew up")

    chat.provider.chat_perform = boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="provider blew up"):
        chat.chat("Say hello.")

    spans = otel_setup.get_finished_spans()
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) == 1
    chat_span = chat_spans[0]

    assert chat_span.status.status_code.name == "ERROR"
    attrs = chat_span.attributes or {}
    assert attrs.get("error.type") == "RuntimeError", (
        f"Expected error.type == 'RuntimeError', got {attrs.get('error.type')!r}"
    )
    exc_events = [e for e in chat_span.events if e.name == "exception"]
    assert len(exc_events) == 1, (
        f"Expected exactly one exception event, got {len(exc_events)}"
    )


def test_chat_error_recorded_streaming(otel_setup: InMemorySpanExporter):
    # Streaming errors surface while iterating the response, which happens
    # *outside* the span-activation scope around `chat_perform`. The chat span
    # must still capture them.
    chat = ChatOpenAI(model="gpt-4o-mini")

    def boom_stream(*args, **kwargs):
        def gen():
            raise RuntimeError("stream blew up")
            yield  # pragma: no cover - marks this a generator

        return gen()

    chat.provider.chat_perform = boom_stream  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="stream blew up"):
        list(chat.stream("Say hello."))

    spans = otel_setup.get_finished_spans()
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) == 1
    chat_span = chat_spans[0]

    assert chat_span.status.status_code.name == "ERROR"
    attrs = chat_span.attributes or {}
    assert attrs.get("error.type") == "RuntimeError", (
        f"Expected error.type == 'RuntimeError', got {attrs.get('error.type')!r}"
    )
    exc_events = [e for e in chat_span.events if e.name == "exception"]
    assert len(exc_events) == 1, (
        f"Expected exactly one exception event, got {len(exc_events)}"
    )


def test_conversation_id_recorded_on_spans(otel_setup: InMemorySpanExporter):
    # When a conversation id is set on the Chat, both the invoke_agent and chat
    # spans should carry it as `gen_ai.conversation.id` (the provider call is
    # forced to fail so the test runs without network; the spans are still
    # created and finished with their start-time attributes).
    chat = ChatOpenAI(model="gpt-4o-mini")
    chat.conversation_id = "conv_123"

    def boom(*args, **kwargs):
        raise RuntimeError("provider blew up")

    chat.provider.chat_perform = boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="provider blew up"):
        chat.chat("Say hello.")

    spans = otel_setup.get_finished_spans()
    agent_spans = [s for s in spans if s.name == "invoke_agent"]
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(agent_spans) == 1
    assert len(chat_spans) == 1

    for span in (*agent_spans, *chat_spans):
        attrs = span.attributes or {}
        assert attrs.get("gen_ai.conversation.id") == "conv_123", (
            f"Expected gen_ai.conversation.id on span {span.name!r}"
        )


def test_conversation_id_absent_by_default(otel_setup: InMemorySpanExporter):
    # Per the GenAI semantic conventions, no fallback identifier should be
    # invented when the app hasn't provided one.
    chat = ChatOpenAI(model="gpt-4o-mini")

    def boom(*args, **kwargs):
        raise RuntimeError("provider blew up")

    chat.provider.chat_perform = boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="provider blew up"):
        chat.chat("Say hello.")

    for span in otel_setup.get_finished_spans():
        attrs = span.attributes or {}
        assert "gen_ai.conversation.id" not in attrs, (
            f"Unexpected gen_ai.conversation.id on span {span.name!r}"
        )


def test_conversation_id_property():
    chat = ChatOpenAI(model="gpt-4o-mini")
    assert chat.conversation_id is None

    chat.conversation_id = "conv_123"
    assert chat.conversation_id == "conv_123"

    chat.conversation_id = None
    assert chat.conversation_id is None

    with pytest.raises(TypeError, match="conversation_id"):
        chat.conversation_id = 123  # type: ignore[assignment]


def test_noop_without_provider():
    from opentelemetry import trace
    from opentelemetry.trace import NonRecordingSpan

    # The default module-level tracer (no SDK TracerProvider configured)
    # should produce non-recording spans that are effectively no-ops.
    default_tracer = trace.get_tracer("test-noop")
    span = default_tracer.start_span("test")
    assert isinstance(span, NonRecordingSpan)
    assert not span.is_recording()
    span.end()


def test_structured_tool_response_nests_not_double_encoded():
    # A non-string tool-result value should remain structured so it nests inside
    # the serialized message, rather than being embedded as a quoted JSON string.
    result = ContentToolResult(value={"temp": 70, "unit": "F"})
    part = _otel.as_otel_part(result)
    assert part["response"] == {"temp": 70, "unit": "F"}

    # Round-trip through the same serialization used for span attributes: the
    # response stays a nested object, not a double-encoded JSON string.
    serialized = json.loads(_otel.to_json([part]))
    assert serialized[0]["response"] == {"temp": 70, "unit": "F"}


def test_unserializable_tool_response_falls_back_to_str():
    class NotJson:
        def __repr__(self) -> str:
            return "<not-json>"

    result = ContentToolResult(value=NotJson())
    part = _otel.as_otel_part(result)
    assert part["response"] == "<not-json>"


def _parent_span_id(span: object) -> int | None:
    parent = getattr(span, "parent", None)
    return parent.span_id if parent is not None else None


@pytest.mark.vcr
def test_provider_http_span_nests_under_chat_span(otel_setup: InMemorySpanExporter):
    # A provider auto-instrumentor (openllmetry, opentelemetry-instrumentation-openai)
    # creates a span off the *current context* at the moment the SDK is called.
    # chatlas must activate its chat span around that call so the HTTP span nests
    # underneath it rather than starting a disconnected root trace.
    chat = ChatOpenAI(model="gpt-4o-mini")

    orig_perform = chat.provider.chat_perform

    def wrapped_perform(*args, **kwargs):
        with _otel.tracer.start_as_current_span("openai.request"):
            return orig_perform(*args, **kwargs)

    chat.provider.chat_perform = wrapped_perform  # type: ignore[method-assign]
    chat.chat("Say hello.")

    spans = otel_setup.get_finished_spans()
    chat_span = next(s for s in spans if s.name.startswith("chat "))
    http_span = next(s for s in spans if s.name == "openai.request")
    assert chat_span.context is not None
    assert _parent_span_id(http_span) == chat_span.context.span_id, (
        "provider HTTP span should nest under the chatlas chat span"
    )


def test_stream_iteration_provider_span_nests_under_chat_span(
    otel_setup: InMemorySpanExporter,
):
    chat = ChatOpenAI(model="gpt-4o-mini")

    def streaming_perform(*args, **kwargs):
        def gen():
            with _otel.tracer.start_as_current_span("openai.request.iter"):
                yield object()

        return gen()

    chat.provider.chat_perform = streaming_perform  # type: ignore[method-assign]
    chat.provider.stream_content = lambda chunk, completion, turns=(): [
        ContentText(text="hi")
    ]  # type: ignore[method-assign]
    chat.provider.stream_merge_chunks = (  # type: ignore[method-assign]
        lambda result, chunk: chunk
    )
    chat.provider.stream_turn = (  # type: ignore[method-assign]
        lambda completion, has_data_model, turns=(): AssistantTurn("hi")
    )

    list(chat.stream("Say hello."))

    spans = otel_setup.get_finished_spans()
    chat_span = next(s for s in spans if s.name.startswith("chat "))
    iter_span = next(s for s in spans if s.name == "openai.request.iter")
    assert chat_span.context is not None
    assert _parent_span_id(iter_span) == chat_span.context.span_id, (
        "provider span created while iterating the stream should nest under the chat span"
    )


@pytest.mark.asyncio
async def test_async_stream_iteration_provider_span_nests_under_chat_span(
    otel_setup: InMemorySpanExporter,
):
    chat = ChatOpenAI(model="gpt-4o-mini")

    async def streaming_perform_async(*args, **kwargs):
        async def gen():
            with _otel.tracer.start_as_current_span("openai.request.iter.async"):
                yield object()

        return gen()

    chat.provider.chat_perform_async = streaming_perform_async  # type: ignore[method-assign]
    chat.provider.stream_content = lambda chunk, completion, turns=(): [
        ContentText(text="hi")
    ]  # type: ignore[method-assign]
    chat.provider.stream_merge_chunks = (  # type: ignore[method-assign]
        lambda result, chunk: chunk
    )
    chat.provider.stream_turn = (  # type: ignore[method-assign]
        lambda completion, has_data_model, turns=(): AssistantTurn("hi")
    )

    chunks = []
    async for chunk in await chat.stream_async("Say hello."):
        chunks.append(chunk)

    spans = otel_setup.get_finished_spans()
    chat_span = next(s for s in spans if s.name.startswith("chat "))
    iter_span = next(s for s in spans if s.name == "openai.request.iter.async")
    assert chat_span.context is not None
    assert _parent_span_id(iter_span) == chat_span.context.span_id, (
        "provider span created while iterating the async stream should nest under the chat span"
    )
    # Time to first token is recorded on the async streamed chat span.
    attrs = chat_span.attributes or {}
    ttft = attrs.get("gen_ai.response.time_to_first_chunk")
    assert isinstance(ttft, float) and ttft > 0, (
        f"Expected gen_ai.response.time_to_first_chunk > 0, got {ttft!r}"
    )


@pytest.mark.vcr
def test_tool_internal_span_nests_under_tool_span(otel_setup: InMemorySpanExporter):
    chat = ChatOpenAI(
        model="gpt-4o-mini",
        system_prompt="Always use the get_date tool to answer questions about the date.",
    )

    def get_date() -> str:
        "Return the current date"
        # A tool that does instrumented work (DB query, HTTP call, sub-agent).
        with _otel.tracer.start_as_current_span("db_query"):
            pass
        return "2026-05-12"

    chat.register_tool(get_date)
    chat.chat("What is today's date?")

    spans = otel_setup.get_finished_spans()
    tool_span = next(s for s in spans if s.name.startswith("execute_tool "))
    db_span = next(s for s in spans if s.name == "db_query")
    assert tool_span.context is not None
    assert _parent_span_id(db_span) == tool_span.context.span_id, (
        "span created inside a tool should nest under the execute_tool span"
    )


def test_generator_tool_internal_span_nests_under_tool_span(
    otel_setup: InMemorySpanExporter,
):
    chat = ChatOpenAI()

    def get_date():
        "Return the current date."
        with _otel.tracer.start_as_current_span("db_query"):
            pass
        yield "2026-05-12"

    chat.register_tool(get_date)
    tool = chat._tools["get_date"]
    request = ContentToolRequest(
        id="test-id",
        name="get_date",
        arguments={},
        tool=ToolInfo.from_tool(tool),
    )

    results = list(chat._invoke_tool(request))

    assert len(results) == 1
    tool_span = next(
        s for s in otel_setup.get_finished_spans() if s.name == "execute_tool get_date"
    )
    db_span = next(s for s in otel_setup.get_finished_spans() if s.name == "db_query")
    assert tool_span.context is not None
    assert _parent_span_id(db_span) == tool_span.context.span_id, (
        "span created inside a generator tool should nest under the execute_tool span"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_tool_internal_span_nests_under_tool_span_async(
    otel_setup: InMemorySpanExporter,
):
    chat = ChatOpenAI(
        model="gpt-4o-mini",
        system_prompt="Always use the get_date tool to answer questions about the date.",
    )

    def get_date() -> str:
        "Return the current date"
        with _otel.tracer.start_as_current_span("db_query"):
            pass
        return "2026-05-12"

    chat.register_tool(get_date)
    await chat.chat_async("What is today's date?")

    spans = otel_setup.get_finished_spans()
    tool_span = next(s for s in spans if s.name.startswith("execute_tool "))
    db_span = next(s for s in spans if s.name == "db_query")
    assert tool_span.context is not None
    assert _parent_span_id(db_span) == tool_span.context.span_id, (
        "span created inside an async tool should nest under the execute_tool span"
    )


@pytest.mark.asyncio
async def test_async_generator_tool_internal_span_nests_under_tool_span(
    otel_setup: InMemorySpanExporter,
):
    chat = ChatOpenAI()

    async def get_date():
        "Return the current date."
        with _otel.tracer.start_as_current_span("db_query"):
            pass
        yield "2026-05-12"

    chat.register_tool(get_date)
    tool = chat._tools["get_date"]
    request = ContentToolRequest(
        id="test-id",
        name="get_date",
        arguments={},
        tool=ToolInfo.from_tool(tool),
    )

    results = []
    async for result in chat._invoke_tool_async(request):
        results.append(result)

    assert len(results) == 1
    tool_span = next(
        s for s in otel_setup.get_finished_spans() if s.name == "execute_tool get_date"
    )
    db_span = next(s for s in otel_setup.get_finished_spans() if s.name == "db_query")
    assert tool_span.context is not None
    assert _parent_span_id(db_span) == tool_span.context.span_id, (
        "span created inside an async generator tool should nest under the execute_tool span"
    )


# ---------------------------------------------------------------------------
# GenAI semantic convention attributes and metrics
# ---------------------------------------------------------------------------


def test_agent_and_chat_span_kinds(otel_setup: InMemorySpanExporter):
    chat = ChatOpenAI(model="gpt-4o-mini")
    _mock_completion(chat)
    chat.chat("Say hello.", stream=False)

    agent_span = _finished_span(otel_setup, "invoke_agent")
    chat_span = _finished_span(otel_setup, "chat ")
    # chatlas's agent loop runs in-process, so invoke_agent is an internal span
    assert agent_span.kind == SpanKind.INTERNAL
    assert chat_span.kind == SpanKind.CLIENT


def test_chat_span_status_ok_on_success(otel_setup: InMemorySpanExporter):
    chat = ChatOpenAI(model="gpt-4o-mini")
    _mock_completion(chat)
    chat.chat("Say hello.", stream=False)

    chat_span = _finished_span(otel_setup, "chat ")
    assert chat_span.status.status_code.name == "OK"


def test_chat_span_response_and_usage_attributes(otel_setup: InMemorySpanExporter):
    chat = ChatOpenAI(model="gpt-4o-mini")
    _mock_completion(chat, tokens=(10, 5, 3), reasoning=2)
    chat.chat("Say hello.", stream=False)

    chat_span = _finished_span(otel_setup, "chat ")
    attrs = chat_span.attributes or {}
    assert attrs.get("gen_ai.response.model") == "gpt-4o-mini-2024-07-18"
    assert attrs.get("gen_ai.response.id") == "resp_123"
    assert attrs.get("gen_ai.usage.input_tokens") == 13
    assert attrs.get("gen_ai.usage.output_tokens") == 5
    assert attrs.get("gen_ai.usage.cache_read.input_tokens") == 3
    assert attrs.get("gen_ai.usage.reasoning.output_tokens") == 2
    assert tuple(attrs.get("gen_ai.response.finish_reasons") or ()) == ("success",)
    # Not a streaming request, so the stream attribute is left unset
    assert "gen_ai.request.stream" not in attrs


def test_server_attributes_on_chat_span(otel_setup: InMemorySpanExporter):
    chat = ChatOpenAI(model="gpt-4o-mini")
    _mock_completion(chat)
    chat.chat("Say hello.", stream=False)

    chat_span = _finished_span(otel_setup, "chat ")
    attrs = chat_span.attributes or {}
    assert attrs.get("server.address") == "api.openai.com"
    assert attrs.get("server.port") == 443


def test_request_params_recorded_on_chat_and_agent_spans(
    otel_setup: InMemorySpanExporter,
):
    chat = ChatOpenAI(model="gpt-4o-mini")
    chat.set_model_params(max_tokens=50, temperature=0.5)
    # seed isn't supported by OpenAI's Responses API, so it's dropped (with a
    # warning) at set-time and must not show up on the spans.
    with pytest.warns(UserWarning, match="not supported"):
        chat.set_model_params(seed=42)
    _mock_completion(chat)
    chat.chat("Say hello.", stream=False)

    for prefix in ("invoke_agent", "chat "):
        span = _finished_span(otel_setup, prefix)
        attrs = span.attributes or {}
        assert attrs.get("gen_ai.request.max_tokens") == 50
        assert attrs.get("gen_ai.request.temperature") == 0.5
        assert "gen_ai.request.seed" not in attrs
        assert "gen_ai.request.top_p" not in attrs


def test_reasoning_level_recorded_from_kwargs(otel_setup: InMemorySpanExporter):
    chat = ChatOpenAI(model="gpt-5-mini", reasoning="low")
    _mock_completion(chat)
    chat.chat("Say hello.", stream=False)

    chat_span = _finished_span(otel_setup, "chat ")
    attrs = chat_span.attributes or {}
    assert attrs.get("gen_ai.request.reasoning.level") == "low"


def test_output_type_and_tool_definitions_with_data_model(
    otel_setup: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")

    class Person(BaseModel):
        name: str

    chat = ChatOpenAI(model="gpt-4o-mini")
    _mock_completion(chat)

    def get_date() -> str:
        "Return the current date"
        return "2026-05-12"

    chat.register_tool(get_date)

    turn = user_turn("Say hello.")
    for _ in chat._submit_turns(
        turn,
        echo="none",
        stream=False,
        data_model=Person,
        controller=StreamController(),
    ):
        pass

    chat_span = _finished_span(otel_setup, "chat ")
    attrs = chat_span.attributes or {}
    assert attrs.get("gen_ai.output.type") == "json"

    defs = json.loads(str(attrs["gen_ai.tool.definitions"]))
    assert isinstance(defs, list)
    # OpenAI implements structured output natively, so only user tools appear
    assert [d["name"] for d in defs] == ["get_date"]
    get_date_def = defs[0]
    assert get_date_def["type"] == "function"
    assert get_date_def["description"] == "Return the current date"
    assert get_date_def["parameters"]["type"] == "object"


def test_tool_definitions_omitted_without_content_capture(
    otel_setup: InMemorySpanExporter,
):
    chat = ChatOpenAI(model="gpt-4o-mini")
    _mock_completion(chat)

    def get_date() -> str:
        "Return the current date"
        return "2026-05-12"

    chat.register_tool(get_date)
    chat.chat("Say hello.", stream=False)

    chat_span = _finished_span(otel_setup, "chat ")
    attrs = chat_span.attributes or {}
    assert "gen_ai.tool.definitions" not in attrs


def test_anthropic_structured_output_tool_modes():
    class Person(BaseModel):
        name: str

    # claude-3-5 lacks native structured outputs, so a private tool is used
    chat = ChatAnthropic(model="claude-3-5-sonnet-latest")
    tool = chat.provider.structured_output_tool(Person)
    assert tool is not None
    assert tool.name == "_structured_tool_call"

    # Claude 4.5+ uses native structured outputs, so no tool is needed
    chat = ChatAnthropic(model="claude-sonnet-4-5")
    assert chat.provider.structured_output_tool(Person) is None

    # ...unless tool mode is forced
    chat = ChatAnthropic(model="claude-sonnet-4-5", structured_output_mode="tool")
    assert chat.provider.structured_output_tool(Person) is not None


def test_otel_provider_name_mapping():
    assert _otel.otel_provider_name(ChatOpenAI(model="gpt-4o-mini").provider) == "openai"
    assert (
        _otel.otel_provider_name(ChatAnthropic(model="claude-sonnet-4-5").provider)
        == "anthropic"
    )


def test_inference_metrics_recorded(
    otel_setup: InMemorySpanExporter, otel_metrics: InMemoryMetricReader
):
    chat = ChatOpenAI(model="gpt-4o-mini")
    _mock_completion(chat, tokens=(10, 5, 3), reasoning=2)
    chat.chat("Say hello.", stream=False)

    points = _metric_points(otel_metrics)

    duration = points["gen_ai.client.inference.duration"]
    assert len(duration) == 1
    assert duration[0].count == 1
    assert duration[0].sum >= 0
    attrs = duration[0].attributes
    assert "error.type" not in attrs
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
    assert attrs["gen_ai.response.model"] == "gpt-4o-mini-2024-07-18"
    assert attrs["server.address"] == "api.openai.com"

    in_hist = points["gen_ai.client.inference.operation.input_tokens"]
    assert len(in_hist) == 1
    assert in_hist[0].sum == 13
    out_hist = points["gen_ai.client.inference.operation.output_tokens"]
    assert len(out_hist) == 1
    assert out_hist[0].sum == 5

    def counter_value(name: str) -> int:
        pts = points[name]
        assert len(pts) == 1
        assert pts[0].attributes["gen_ai.token.modality"] == "unknown"
        assert pts[0].attributes["gen_ai.provider.name"] == "openai"
        return pts[0].value

    assert counter_value("gen_ai.client.inference.usage.input_tokens") == 13
    assert counter_value("gen_ai.client.inference.usage.output_tokens") == 5
    assert counter_value("gen_ai.client.inference.usage.cache_read.input_tokens") == 3
    assert counter_value("gen_ai.client.inference.usage.reasoning.output_tokens") == 2


def test_agent_metrics_and_span_attributes(
    otel_setup: InMemorySpanExporter, otel_metrics: InMemoryMetricReader
):
    chat = ChatOpenAI(model="gpt-4o-mini")

    def get_date() -> str:
        "Return the current date"
        return "2026-05-12"

    chat.register_tool(get_date)

    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
        output_tokens_details=SimpleNamespace(reasoning_tokens=None),
    )
    completion = SimpleNamespace(
        model="gpt-4o-mini-2024-07-18", id="resp_123", usage=usage, service_tier=None
    )
    turns = [
        AssistantTurn(
            [ContentToolRequest(id="call_1", name="get_date", arguments={})],
            tokens=(10, 5, 0),
            finish_reason="tool_use",
            completion=completion,
        ),
        AssistantTurn(
            "The date is 2026-05-12.",
            tokens=(20, 8, 0),
            finish_reason="success",
            completion=completion,
        ),
    ]
    chat.provider.chat_perform = lambda *args, **kwargs: completion  # type: ignore[method-assign]
    chat.provider.value_turn = lambda *args, **kwargs: turns.pop(0)  # type: ignore[method-assign]

    chat.chat("What is today's date?", stream=False)

    agent_span = _finished_span(otel_setup, "invoke_agent")
    attrs = agent_span.attributes or {}
    assert attrs.get("gen_ai.invoke_agent.inference_calls") == 2
    assert attrs.get("gen_ai.invoke_agent.tool_calls") == 1
    assert isinstance(attrs.get("gen_ai.invoke_agent.duration"), float)
    assert attrs.get("gen_ai.usage.input_tokens") == 30
    assert attrs.get("gen_ai.usage.output_tokens") == 13
    assert agent_span.status.status_code.name == "UNSET"

    points = _metric_points(otel_metrics)
    assert points["gen_ai.invoke_agent.inference_calls"][0].sum == 2
    assert points["gen_ai.invoke_agent.tool_calls"][0].sum == 1
    agent_duration = points["gen_ai.invoke_agent.duration"][0]
    assert "error.type" not in agent_duration.attributes
    assert agent_duration.attributes["gen_ai.operation.name"] == "invoke_agent"

    tool_duration = points["gen_ai.execute_tool.duration"][0]
    assert tool_duration.attributes["gen_ai.tool.name"] == "get_date"
    assert tool_duration.attributes["gen_ai.tool.type"] == "function"
    assert "error.type" not in tool_duration.attributes


def test_tool_error_duration_metric(otel_metrics: InMemoryMetricReader):
    chat = ChatOpenAI(model="gpt-4o-mini")
    request = ContentToolRequest(id="missing-tool", name="missing_tool", arguments={})

    results = list(chat._invoke_tool(request))
    assert isinstance(results[0].error, RuntimeError)

    points = _metric_points(otel_metrics)
    tool_duration = points["gen_ai.execute_tool.duration"][0]
    assert tool_duration.attributes["gen_ai.tool.name"] == "missing_tool"
    assert tool_duration.attributes["error.type"] == "RuntimeError"


def test_chat_error_duration_metric(
    otel_setup: InMemorySpanExporter, otel_metrics: InMemoryMetricReader
):
    chat = ChatOpenAI(model="gpt-4o-mini")

    def boom(*args, **kwargs):
        raise RuntimeError("provider blew up")

    chat.provider.chat_perform = boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="provider blew up"):
        chat.chat("Say hello.", stream=False)

    points = _metric_points(otel_metrics)
    duration = points["gen_ai.client.inference.duration"][0]
    assert duration.attributes["error.type"] == "RuntimeError"
    # No usage info is available for a failed request
    assert "gen_ai.client.inference.operation.input_tokens" not in points

    chat_span = _finished_span(otel_setup, "chat ")
    attrs = chat_span.attributes or {}
    assert tuple(attrs.get("gen_ai.response.finish_reasons") or ()) == ("error",)

    agent_duration = points["gen_ai.invoke_agent.duration"][0]
    assert agent_duration.attributes["error.type"] == "RuntimeError"


def test_streaming_ttft_and_chunk_metrics(
    otel_setup: InMemorySpanExporter, otel_metrics: InMemoryMetricReader
):
    chat = ChatOpenAI(model="gpt-4o-mini")
    _mock_streaming_provider(chat, n_chunks=3)
    list(chat.stream("Say hello."))

    chat_span = _finished_span(otel_setup, "chat ")
    attrs = chat_span.attributes or {}
    assert attrs.get("gen_ai.request.stream") is True
    ttft = attrs.get("gen_ai.response.time_to_first_chunk")
    assert isinstance(ttft, float) and ttft >= 0
    assert tuple(attrs.get("gen_ai.response.finish_reasons") or ()) == ("success",)
    assert chat_span.status.status_code.name == "OK"

    points = _metric_points(otel_metrics)
    assert len(points["gen_ai.client.inference.time_to_first_chunk"]) == 1
    # One point per chunk after the first
    chunk_points = points["gen_ai.client.inference.time_per_output_chunk"]
    assert len(chunk_points) == 1
    assert chunk_points[0].count == 2


def test_cancelled_stream_recorded_with_error_status(
    otel_setup: InMemorySpanExporter, otel_metrics: InMemoryMetricReader
):
    chat = ChatOpenAI(model="gpt-4o-mini")
    _mock_streaming_provider(chat, n_chunks=10)
    ctrl = StreamController()

    chunks = []
    for chunk in chat.stream("Say hello.", controller=ctrl):
        chunks.append(chunk)
        ctrl.cancel()
    assert len(chunks) == 1

    chat_span = _finished_span(otel_setup, "chat ")
    assert chat_span.status.status_code.name == "ERROR"
    attrs = chat_span.attributes or {}
    assert attrs.get("error.type") == "cancelled"
    assert tuple(attrs.get("gen_ai.response.finish_reasons") or ()) == ("cancelled",)

    agent_span = _finished_span(otel_setup, "invoke_agent")
    assert agent_span.status.status_code.name == "ERROR"
    assert (agent_span.attributes or {}).get("error.type") == "cancelled"

    points = _metric_points(otel_metrics)
    duration = points["gen_ai.client.inference.duration"][0]
    assert duration.attributes["error.type"] == "cancelled"
    agent_duration = points["gen_ai.invoke_agent.duration"][0]
    assert agent_duration.attributes["error.type"] == "cancelled"

    # The partial turn is still preserved in the chat history
    last = chat.get_last_turn(role="assistant")
    assert last is not None and last.is_partial


def test_abandoned_stream_recorded_as_interrupted(
    otel_setup: InMemorySpanExporter, otel_metrics: InMemoryMetricReader
):
    chat = ChatOpenAI(model="gpt-4o-mini")
    _mock_streaming_provider(chat, n_chunks=10)

    gen = chat.stream("Say hello.")
    next(gen)
    # Abandoning the stream unwinds it without a success or error record
    gen.close()

    chat_span = _finished_span(otel_setup, "chat ")
    assert chat_span.status.status_code.name == "ERROR"
    attrs = chat_span.attributes or {}
    assert attrs.get("error.type") == "interrupted"
    assert tuple(attrs.get("gen_ai.response.finish_reasons") or ()) == (
        "interrupted",
    )

    agent_span = _finished_span(otel_setup, "invoke_agent")
    assert (agent_span.attributes or {}).get("error.type") == "interrupted"

    points = _metric_points(otel_metrics)
    duration = points["gen_ai.client.inference.duration"][0]
    assert duration.attributes["error.type"] == "interrupted"
    agent_duration = points["gen_ai.invoke_agent.duration"][0]
    assert agent_duration.attributes["error.type"] == "interrupted"


def test_content_tool_result_value_becomes_otel_part():
    # A `Content` tool-result value (e.g. an image) converts to an otel part
    # rather than stringifying or dropping the message.
    img = ContentImageInline(image_content_type="image/png", data="aW1hZ2U=")
    part = _otel.as_otel_part(ContentToolResult(value=img))
    assert part["response"] == {"type": "generic", "class": "ContentImageInline"}

    # A list of `Content` values converts element-wise.
    result = ContentToolResult(
        value=[ContentText(text="a"), ContentText(text="b")]
    )
    part = _otel.as_otel_part(result)
    assert part["response"] == [
        {"type": "text", "content": "a"},
        {"type": "text", "content": "b"},
    ]
