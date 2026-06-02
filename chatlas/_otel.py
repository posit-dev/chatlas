from __future__ import annotations

import os
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, Optional

import orjson
from opentelemetry import trace
from opentelemetry.trace import SpanKind, StatusCode

from ._content import ContentText, ContentToolRequest, ContentToolResult
from ._turn import SystemTurn, UserTurn

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from ._content import Content
    from ._provider import Provider
    from ._turn import AssistantTurn, Turn


tracer = trace.get_tracer("co.posit.python-package.chatlas")

capture_content: bool = os.environ.get(
    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", ""
).lower() in ("true", "1")


def start_agent_span(provider: Provider[Any, Any, Any, Any]) -> Span:
    return tracer.start_span(
        "invoke_agent",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.provider.name": provider.name.lower(),
            "gen_ai.request.model": provider.model,
        },
    )


def start_chat_span(
    provider: Provider[Any, Any, Any, Any],
    turns: list[Turn],
    parent: Optional[Span],
) -> Span:
    ctx = trace.set_span_in_context(parent) if parent is not None else None
    span = tracer.start_span(
        f"chat {provider.model}",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": provider.name.lower(),
            "gen_ai.request.model": provider.model,
        },
        context=ctx,
    )

    if capture_content and span.is_recording():
        # Separate the system turn (recorded as system_instructions) from the
        # rest of the conversation (recorded as input.messages), per the GenAI
        # semantic conventions.
        system_turn = turns[0] if turns and isinstance(turns[0], SystemTurn) else None
        messages = turns[1:] if system_turn is not None else turns
        record_input_content(span, messages, system_turn)

    return span


def start_tool_span(
    request: ContentToolRequest,
    parent: Optional[Span],
) -> Span:
    ctx = trace.set_span_in_context(parent) if parent is not None else None

    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": request.name,
        "gen_ai.tool.call.id": request.id,
    }
    if request.tool is not None and request.tool.description:
        attrs["gen_ai.tool.description"] = request.tool.description

    return tracer.start_span(
        f"execute_tool {request.name}",
        attributes=attrs,
        context=ctx,
    )


def record_chat_result(
    span: Span,
    turn: AssistantTurn[Any],
) -> None:
    if not span.is_recording():
        return

    if turn.tokens is not None:
        input_tokens, output_tokens, cached_input = turn.tokens
        total_input = input_tokens + cached_input
        if total_input > 0 or output_tokens > 0:
            span.set_attribute("gen_ai.usage.input_tokens", total_input)
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

    completion = turn.completion
    if completion is not None:
        response_model = getattr(completion, "model", None) or getattr(
            completion, "model_version", None
        )
        if response_model is not None:
            span.set_attribute("gen_ai.response.model", str(response_model))

        response_id = getattr(completion, "id", None) or getattr(
            completion, "response_id", None
        )
        if response_id is not None:
            span.set_attribute("gen_ai.response.id", str(response_id))

    if capture_content:
        try:
            msg = as_otel_message(turn)
            span.set_attribute("gen_ai.output.messages", to_json([msg]))
        except Exception:
            pass

    # Per the OTel spec, instrumentation libraries leave a successful span's
    # status UNSET rather than marking it OK; errors are recorded via
    # `record_error`.


def record_error(span: Span, error: Exception) -> None:
    if not span.is_recording():
        return

    span.record_exception(error)
    span.set_attribute("error.type", type(error).__name__)
    span.set_status(StatusCode.ERROR, str(error))


def end_span(span: Span) -> None:
    span.end()


def activate_span(span: Span) -> AbstractContextManager[Span]:
    """Make `span` the current span for the enclosing `with` block.

    Wrap the bounded provider call and tool invocation so third-party spans (a
    provider's HTTP instrumentor, or work done inside a tool) nest under ours via
    the OTel context. Not used for the agent span, which brackets the whole
    streaming loop -- staying active across a `yield` would leak the context into
    the consumer's scope.

    The span is not ended on exit (callers end it via `end_span`), and
    non-recording spans are skipped so disabled tracing never touches the context.

    Exception recording is disabled here: callers record failures explicitly via
    `record_error` so every span in the failing path gets a semconv `error.type`
    and exactly one exception event (letting `use_span` also record would double
    up for calls bounded by this context manager).
    """
    if not span.is_recording():
        return nullcontext(span)
    return trace.use_span(
        span,
        end_on_exit=False,
        record_exception=False,
        set_status_on_exception=False,
    )


def as_otel_message(turn: Turn) -> dict[str, Any]:
    is_tool_turn = (
        isinstance(turn, UserTurn)
        and len(turn.contents) > 0
        and all(isinstance(c, ContentToolResult) for c in turn.contents)
    )

    return {
        "role": "tool" if is_tool_turn else turn.role,
        "parts": [as_otel_part(c) for c in turn.contents],
    }


def as_otel_part(content: Content) -> dict[str, Any]:
    if isinstance(content, ContentText):
        return {"type": "text", "content": content.text}

    if isinstance(content, ContentToolRequest):
        return {
            "type": "tool_call",
            "id": content.id,
            "name": content.name,
            "arguments": content.arguments,
        }

    if isinstance(content, ContentToolResult):
        part: dict[str, Any] = {"type": "tool_call_response"}
        if content.request is not None:
            part["id"] = content.request.id
        if content.error is not None:
            part["response"] = str(content.error)
        elif isinstance(content.value, str):
            part["response"] = content.value
        else:
            part["response"] = json_safe(content.value)
        return part

    return {"type": "generic", "class": type(content).__name__}


def to_json(obj: Any) -> str:
    return orjson.dumps(obj).decode("utf-8")


def json_safe(value: Any) -> Any:
    """Return `value` unchanged if it is JSON-serializable, else its `str()`.

    Keeps structured tool-result values structured so they nest inside the
    enclosing `gen_ai.*.messages` JSON, rather than being embedded as a quoted
    (double-encoded) JSON string.
    """
    try:
        orjson.dumps(value)
    except Exception:
        return str(value)
    return value


def record_input_content(
    span: Span,
    turns: list[Turn],
    system_turn: Optional[SystemTurn],
) -> None:
    try:
        if system_turn is not None:
            parts = [as_otel_part(c) for c in system_turn.contents]
            span.set_attribute("gen_ai.system_instructions", to_json(parts))

        msgs = [as_otel_message(t) for t in turns]
        span.set_attribute("gen_ai.input.messages", to_json(msgs))
    except Exception:
        pass
