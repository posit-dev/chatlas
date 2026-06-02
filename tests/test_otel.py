from __future__ import annotations

import json
from typing import cast

import pytest

pytest.importorskip("opentelemetry.sdk")

from chatlas import ChatOpenAI, _otel
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture(autouse=True)
def otel_setup():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    orig_tracer = _otel.tracer
    _otel.tracer = provider.get_tracer("com.posit.python-package.chatlas")

    yield exporter

    exporter.clear()
    _otel.tracer = orig_tracer


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
    # Flip the module-level flag for this test (env var is only read at import time).
    _otel.capture_content = True

    chat = ChatOpenAI(model="gpt-4o-mini", system_prompt="Be terse.")
    chat.chat("Say hello.")

    _otel.capture_content = False

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
