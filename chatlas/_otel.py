from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from ._content import Content, ContentToolRequest
    from ._provider import Provider
    from ._turn import AssistantTurn, SystemTurn, Turn


initialized: bool = False
is_tracing: bool = False
capture_content: bool = False


def get_tracer() -> Any:
    return tracer_ref


tracer_ref: Any = None


def cache_tracer() -> None:
    global initialized, is_tracing, capture_content, tracer_ref  # noqa: PLW0603

    try:
        from opentelemetry import trace
    except ImportError:
        initialized = True
        is_tracing = False
        return

    tracer_ref = trace.get_tracer("com.posit.python-package.chatlas")
    initialized = True
    is_tracing = not isinstance(tracer_ref, trace.NoOpTracer)
    capture_content = os.environ.get(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", ""
    ).lower() in ("true", "1")


def start_agent_span(provider: Provider[Any, Any, Any, Any]) -> Optional[Span]:
    if not initialized:
        cache_tracer()
    if not is_tracing:
        return None

    from opentelemetry.trace import SpanKind

    return get_tracer().start_span(
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
    system_turn: Optional[SystemTurn],
    parent: Optional[Span],
) -> Optional[Span]:
    if not is_tracing:
        return None

    from opentelemetry import trace
    from opentelemetry.trace import SpanKind

    ctx = trace.set_span_in_context(parent) if parent is not None else None
    span = get_tracer().start_span(
        f"chat {provider.model}",
        kind=SpanKind.CLIENT,
        attributes={
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": provider.name.lower(),
            "gen_ai.request.model": provider.model,
        },
        context=ctx,
    )

    if capture_content:
        record_input_content(span, turns, system_turn)

    return span


def start_tool_span(
    request: ContentToolRequest,
    parent: Optional[Span],
) -> Optional[Span]:
    if not is_tracing:
        return None

    from opentelemetry import trace

    ctx = trace.set_span_in_context(parent) if parent is not None else None

    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": request.name,
        "gen_ai.tool.call.id": request.id,
    }
    if request.tool is not None and request.tool.description:
        attrs["gen_ai.tool.description"] = request.tool.description

    return get_tracer().start_span(
        f"execute_tool {request.name}",
        attributes=attrs,
        context=ctx,
    )


def record_chat_result(
    span: Optional[Span],
    turn: AssistantTurn[Any],
) -> None:
    if span is None:
        return

    from opentelemetry.trace import StatusCode

    if turn.tokens is not None:
        input_tokens, output_tokens, cached_input = turn.tokens
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens + cached_input)
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

    span.set_status(StatusCode.OK)


def record_tool_error(span: Optional[Span], error: Exception) -> None:
    if span is None:
        return

    from opentelemetry.trace import StatusCode

    span.record_exception(error)
    span.set_attribute("error.type", type(error).__name__)
    span.set_status(StatusCode.ERROR, str(error))


def end_span(span: Optional[Span]) -> None:
    if span is not None:
        span.end()


def as_otel_message(turn: Turn) -> dict[str, Any]:
    from ._content import ContentToolResult
    from ._turn import UserTurn

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
    from ._content import ContentText, ContentToolRequest, ContentToolResult

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
            try:
                part["response"] = to_json(content.value)
            except Exception:
                part["response"] = str(content.value)
        return part

    return {"type": "generic", "class": type(content).__name__}


def to_json(obj: Any) -> str:
    import orjson

    return orjson.dumps(obj).decode("utf-8")


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
