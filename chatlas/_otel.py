from __future__ import annotations

import os
import time
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, Mapping, Optional
from urllib.parse import urlparse

import orjson
from opentelemetry import metrics, trace
from opentelemetry.trace import SpanKind, StatusCode

from ._content import Content, ContentText, ContentToolRequest, ContentToolResult
from ._tools import Tool
from ._turn import AssistantTurn, SystemTurn, UserTurn

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from ._provider import Provider, StandardModelParams
    from ._turn import Turn


tracer = trace.get_tracer("co.posit.python-package.chatlas")
meter = metrics.get_meter("co.posit.python-package.chatlas")

# Histograms from the GenAI semantic conventions for metrics. The
# `gen_ai.client.inference.*` instruments are defined by the client inference
# conventions; the agent and tool instruments by the general metrics page.
# See: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md
# See: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/
#
# The conventions also advise explicit bucket boundaries (e.g. 0.01 to 81.92 s
# for durations, 1 to 67,108,864 for token counts). They aren't set here, so
# the SDK's default buckets are used; users can configure views in their SDK
# or collector to apply the advised boundaries.
OTEL_HISTOGRAM_SPECS: dict[str, tuple[str, str]] = {
    "gen_ai.client.inference.duration": (
        "GenAI client inference operation duration",
        "s",
    ),
    "gen_ai.client.inference.operation.input_tokens": (
        "Number of input tokens used per inference operation",
        "{token}",
    ),
    "gen_ai.client.inference.operation.output_tokens": (
        "Number of output tokens used per inference operation",
        "{token}",
    ),
    "gen_ai.client.inference.time_to_first_chunk": (
        "Time to receive the first chunk of a streamed response",
        "s",
    ),
    "gen_ai.client.inference.time_per_output_chunk": (
        "Time per output chunk, for each chunk after the first",
        "s",
    ),
    "gen_ai.execute_tool.duration": (
        "The duration of a single tool execution",
        "s",
    ),
    "gen_ai.invoke_agent.duration": (
        "The end-to-end duration of a single agent invocation",
        "s",
    ),
    "gen_ai.invoke_agent.inference_calls": (
        "The number of inference calls made during an agent invocation",
        "{inference_call}",
    ),
    "gen_ai.invoke_agent.tool_calls": (
        "The number of tool calls made during an agent invocation",
        "{tool_call}",
    ),
}

# Monotonic token usage counters from the client inference conventions. Each
# counter is broken down by `gen_ai.token.modality`; providers only report
# totals, so chatlas reports the `unknown` modality.
# See: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-token-metrics.md
OTEL_COUNTER_SPECS: dict[str, tuple[str, str]] = {
    "gen_ai.client.inference.usage.input_tokens": (
        "Number of input tokens used, including cached tokens",
        "{token}",
    ),
    "gen_ai.client.inference.usage.output_tokens": (
        "Number of output tokens used, including reasoning tokens",
        "{token}",
    ),
    "gen_ai.client.inference.usage.cache_read.input_tokens": (
        "Number of input tokens served from a provider-managed cache",
        "{token}",
    ),
    "gen_ai.client.inference.usage.reasoning.output_tokens": (
        "Number of output tokens used for reasoning",
        "{token}",
    ),
}

# Instruments are created lazily so they bind to whatever MeterProvider is
# configured when they're first used (tests swap out `meter`).
_histograms: dict[str, metrics.Histogram] = {}
_counters: dict[str, metrics.Counter] = {}


def _get_histogram(name: str) -> metrics.Histogram:
    histogram = _histograms.get(name)
    if histogram is None:
        description, unit = OTEL_HISTOGRAM_SPECS[name]
        histogram = meter.create_histogram(name, description=description, unit=unit)
        _histograms[name] = histogram
    return histogram


def _get_counter(name: str) -> metrics.Counter:
    counter = _counters.get(name)
    if counter is None:
        description, unit = OTEL_COUNTER_SPECS[name]
        counter = meter.create_counter(name, description=description, unit=unit)
        _counters[name] = counter
    return counter


def record_histogram(name: str, value: float, attributes: Mapping[str, Any]) -> None:
    _get_histogram(name).record(value, attributes=_clean_attributes(attributes))


def record_counter(name: str, value: int, attributes: Mapping[str, Any]) -> None:
    _get_counter(name).add(value, attributes=_clean_attributes(attributes))


def _clean_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    # `None` isn't a valid attribute value; treat it as unset.
    return {k: v for k, v in attributes.items() if v is not None}


# `gen_ai.provider.name` well-known values, keyed by the provider's display
# name. Other providers fall back to the lowercased display name.
# See: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
OTEL_PROVIDER_NAMES: dict[str, str] = {
    "Anthropic": "anthropic",
    "AWS/Bedrock": "aws.bedrock",
    "Azure/OpenAI": "azure.ai.openai",
    "DeepSeek": "deepseek",
    "Google/Gemini": "gcp.gemini",
    "Google/Vertex": "gcp.vertex_ai",
    "Groq": "groq",
    "Mistral": "mistral_ai",
    "OpenAI": "openai",
    "Perplexity": "perplexity",
}


def otel_provider_name(provider: Provider[Any, Any, Any, Any]) -> str:
    name = provider.name
    return OTEL_PROVIDER_NAMES.get(name, name.lower())


def otel_server_attributes(provider: Provider[Any, Any, Any, Any]) -> dict[str, Any]:
    """`server.address` and `server.port` from the provider's base URL.

    The port falls back to the scheme default since semconv requires it when
    the address is set. Best-effort: providers that don't expose a base URL
    (directly or via their SDK client) report nothing.
    """
    base_url = getattr(provider, "base_url", None)
    if base_url is None:
        base_url = getattr(getattr(provider, "_client", None), "base_url", None)
    if base_url is None:
        return {}
    parsed = urlparse(str(base_url))
    if not parsed.hostname:
        return {}
    # urlparse strips the brackets around IPv6 literals, which semconv also
    # omits from `server.address`.
    attributes: dict[str, Any] = {"server.address": parsed.hostname}
    try:
        port = parsed.port
    except ValueError:
        # `.port` raises on a malformed port; treat that as absent.
        port = None
    if port is None:
        port = {"https": 443, "http": 80}.get(parsed.scheme)
    if port is not None:
        attributes["server.port"] = port
    return attributes


def otel_metric_attributes(
    provider: Provider[Any, Any, Any, Any],
    result: Any = None,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": otel_provider_name(provider),
        "gen_ai.request.model": provider.model,
        **otel_server_attributes(provider),
    }
    response_model = _response_model(result)
    if response_model is not None:
        attributes["gen_ai.response.model"] = str(response_model)
    return attributes


def _response_model(result: Any) -> Any:
    if result is None:
        return None
    if isinstance(result, Mapping):
        return result.get("model") or result.get("model_version")
    return getattr(result, "model", None) or getattr(result, "model_version", None)


def _response_id(result: Any) -> Any:
    if result is None:
        return None
    if isinstance(result, Mapping):
        return result.get("id") or result.get("response_id")
    return getattr(result, "id", None) or getattr(result, "response_id", None)


def capture_content() -> bool:
    return os.environ.get(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", ""
    ).lower() in ("true", "1")


def start_agent_span(
    provider: Provider[Any, Any, Any, Any],
    conversation_id: Optional[str] = None,
) -> Span:
    attributes: dict[str, Any] = {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.provider.name": otel_provider_name(provider),
        "gen_ai.request.model": provider.model,
    }
    if conversation_id is not None:
        attributes["gen_ai.conversation.id"] = conversation_id
    return tracer.start_span(
        "invoke_agent",
        # chatlas's agent loop runs in-process, so this is the semconv
        # `gen_ai.invoke_agent.internal` span.
        kind=SpanKind.INTERNAL,
        attributes=attributes,
    )


def start_chat_span(
    provider: Provider[Any, Any, Any, Any],
    turns: list[Turn],
    parent: Optional[Span],
    conversation_id: Optional[str] = None,
    *,
    stream: bool = False,
    data_model: Any = None,
    tools: Optional[Mapping[str, Any]] = None,
) -> Span:
    ctx = trace.set_span_in_context(parent) if parent is not None else None
    attributes: dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": otel_provider_name(provider),
        "gen_ai.request.model": provider.model,
        **otel_server_attributes(provider),
    }
    if conversation_id is not None:
        # Per the GenAI semantic conventions, only set when a conversation
        # identifier is readily available; never invent a fallback value.
        attributes["gen_ai.conversation.id"] = conversation_id
    if stream:
        # Only set when streaming; unset means non-streaming per semconv.
        attributes["gen_ai.request.stream"] = True
    if data_model is not None:
        attributes["gen_ai.output.type"] = "json"
    span = tracer.start_span(
        f"chat {provider.model}",
        kind=SpanKind.CLIENT,
        attributes=attributes,
        context=ctx,
    )

    if capture_content() and span.is_recording():
        definitions = otel_tool_definitions(provider, tools, data_model)
        if definitions:
            span.set_attribute("gen_ai.tool.definitions", to_json(definitions))

        # Separate the system turn (recorded as system_instructions) from the
        # rest of the conversation (recorded as input.messages), per the GenAI
        # semantic conventions.
        system_turn = turns[0] if turns and isinstance(turns[0], SystemTurn) else None
        messages = turns[1:] if system_turn is not None else turns
        record_input_content(span, messages, system_turn)

    return span


def otel_tool_definitions(
    provider: Provider[Any, Any, Any, Any],
    tools: Optional[Mapping[str, Any]],
    data_model: Any = None,
) -> list[dict[str, Any]]:
    # Built-in (provider-executed) tools have no argument schema, so only
    # `Tool`s are recorded. Providers that implement structured output via
    # tool calling add a private tool to the request; record it so the span
    # matches.
    definitions = [
        as_otel_tool_definition(tool)
        for tool in (tools or {}).values()
        if isinstance(tool, Tool)
    ]
    if data_model is not None:
        structured_tool = provider.structured_output_tool(data_model)
        if structured_tool is not None:
            definitions.append(as_otel_tool_definition(structured_tool))
    return definitions


# A GenAI semconv tool definition for a Tool.
def as_otel_tool_definition(tool: Tool) -> dict[str, Any]:
    function = tool.schema["function"]
    parameters = function.get("parameters")
    definition: dict[str, Any] = {
        "type": "function",
        "name": tool.name,
        "description": function.get("description"),
        # An empty schema isn't a useful JSON Schema object, so omit it.
        "parameters": parameters if isinstance(parameters, dict) and parameters else None,
    }
    return _clean_attributes(definition)


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


def record_chat_ttft_attr(span: Span, start: float) -> None:
    """Record the time to first token (in seconds) on the chat span as the
    `gen_ai.response.time_to_first_chunk` semconv attribute."""
    if not span.is_recording():
        return
    span.set_attribute(
        "gen_ai.response.time_to_first_chunk",
        time.monotonic() - start,
    )


# Sets the `gen_ai.request.*` attributes on a chat or invoke_agent span.
# Unlike ellmer (which learns of dropped params at request time), chatlas
# drops provider-unsupported standard params when they're set on the Chat, so
# everything in `params` was (or will be) sent.
def record_request_params(
    span: Optional[Span],
    params: StandardModelParams,
    kwargs: Optional[Mapping[str, Any]],
) -> None:
    if span is None or not span.is_recording():
        return
    for name, value in otel_request_attributes(params, kwargs).items():
        span.set_attribute(name, value)


# Map the standard model params onto the `gen_ai.request.*` span attributes.
def otel_request_attributes(
    params: StandardModelParams,
    kwargs: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    return _clean_attributes(
        {
            "gen_ai.request.temperature": params.get("temperature"),
            "gen_ai.request.top_p": params.get("top_p"),
            "gen_ai.request.top_k": params.get("top_k"),
            "gen_ai.request.frequency_penalty": params.get("frequency_penalty"),
            "gen_ai.request.presence_penalty": params.get("presence_penalty"),
            "gen_ai.request.seed": params.get("seed"),
            "gen_ai.request.max_tokens": params.get("max_tokens"),
            "gen_ai.request.stop_sequences": params.get("stop_sequences"),
            "gen_ai.request.reasoning.level": _reasoning_level(kwargs),
        }
    )


def _reasoning_level(kwargs: Optional[Mapping[str, Any]]) -> Optional[str]:
    # Reasoning effort isn't a standard chatlas param; each provider threads it
    # through its own kwargs shape. Best-effort across the known ones
    # (OpenAI's `reasoning={"effort": ...}`, Anthropic's
    # `output_config={"effort": ...}`, the completions API's `reasoning_effort`).
    if not kwargs:
        return None
    for key in ("reasoning", "output_config"):
        value = kwargs.get(key)
        if isinstance(value, Mapping):
            effort = value.get("effort")
            if isinstance(effort, str):
                return effort
    effort = kwargs.get("reasoning_effort")
    return effort if isinstance(effort, str) else None


# Records a completed model request: the operation duration, token usage
# histograms and counters, and the usage/response attributes on the chat
# span. A user-cancelled stream is still a completed request (tokens and
# response id may be known), but is reported with an error status and
# `error_type = "cancelled"` on the span and the duration metric, matching
# the `cancelled` finish reason set on the partial turn.
#
# A successful request's OK status is deliberately NOT set here: it is
# stamped later, via `record_chat_span_ok()`, once nothing more can fail.
def record_chat_span_status(
    span: Span,
    provider: Provider[Any, Any, Any, Any],
    result: Any,
    start: float,
    error_type: Optional[str] = None,
) -> None:
    attributes = otel_metric_attributes(provider, result)
    record_histogram(
        "gen_ai.client.inference.duration",
        time.monotonic() - start,
        {**attributes, "error.type": error_type},
    )

    tokens = _safe_value_tokens(provider, result)
    reasoning = _safe_value_reasoning_tokens(provider, result) or 0
    input_tokens = output_tokens = cached_input = 0
    if tokens is not None:
        # tokens are (input, output, cached_input)
        input_tokens = tokens[0] + tokens[2]
        output_tokens = tokens[1]
        cached_input = tokens[2]

    if input_tokens > 0 or output_tokens > 0:
        record_histogram(
            "gen_ai.client.inference.operation.input_tokens",
            input_tokens,
            attributes,
        )
        record_histogram(
            "gen_ai.client.inference.operation.output_tokens",
            output_tokens,
            attributes,
        )
        usage = {
            "gen_ai.client.inference.usage.input_tokens": input_tokens,
            "gen_ai.client.inference.usage.output_tokens": output_tokens,
            "gen_ai.client.inference.usage.cache_read.input_tokens": cached_input,
            "gen_ai.client.inference.usage.reasoning.output_tokens": reasoning,
        }
        usage_attributes = {**attributes, "gen_ai.token.modality": "unknown"}
        for name, value in usage.items():
            if value > 0:
                record_counter(name, value, usage_attributes)

    if not span.is_recording():
        return

    response_model = _response_model(result)
    if response_model is not None:
        span.set_attribute("gen_ai.response.model", str(response_model))
    response_id = _response_id(result)
    if response_id is not None:
        span.set_attribute("gen_ai.response.id", str(response_id))
    if input_tokens > 0 or output_tokens > 0:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
    if cached_input > 0:
        span.set_attribute("gen_ai.usage.cache_read.input_tokens", cached_input)
    if reasoning > 0:
        span.set_attribute("gen_ai.usage.reasoning.output_tokens", reasoning)

    if error_type is not None:
        span.set_status(StatusCode.ERROR)
        span.set_attribute("error.type", error_type)


# Marks a successfully completed request OK. Set last -- once the whole
# request path (turn assembly, structured-output parsing, response cleanup)
# has finished -- because the SDK treats Ok as final: it overrides any prior
# Error and ignores any later one (the spec's `Ok > Error > Unset` total
# order), so stamping it earlier could mask a late failure.
def record_chat_span_ok(span: Span) -> None:
    if not span.is_recording():
        return
    span.set_status(StatusCode.OK)


def _safe_value_tokens(
    provider: Provider[Any, Any, Any, Any], result: Any
) -> Optional[tuple[int, int, int]]:
    # Telemetry must never break the chat: a partial (cancelled) result may
    # not have usage info in the shape the provider expects.
    if result is None:
        return None
    try:
        return provider.value_tokens(result)
    except Exception:
        return None


def _safe_value_reasoning_tokens(
    provider: Provider[Any, Any, Any, Any], result: Any
) -> Optional[int]:
    if result is None:
        return None
    try:
        return provider.value_reasoning_tokens(result)
    except Exception:
        return None


# Sets the finish reasons and (when content capture is enabled) the output
# messages on the chat span, from the completed (or cancelled) turn.
def record_chat_span_output(span: Span, turn: Turn) -> None:
    if not span.is_recording():
        return
    if not isinstance(turn, AssistantTurn):
        return
    # A partial turn only reaches here when the stream was cancelled; it
    # reports a `cancelled` finish reason, matching `interrupted` from
    # `record_chat_span_interrupted()`.
    if turn.is_partial:
        span.set_attribute("gen_ai.response.finish_reasons", ["cancelled"])
    elif turn.finish_reason is not None:
        span.set_attribute("gen_ai.response.finish_reasons", [turn.finish_reason])
    if not capture_content():
        return
    try:
        msg = as_otel_message(turn)
        span.set_attribute("gen_ai.output.messages", to_json([msg]))
    except Exception:
        pass


# Records a failed model request: the operation duration metric tagged with
# `error.type`, plus the exception and error status on the chat span. `start`
# is `None` once the duration has already been recorded for this request.
def record_chat_span_error(
    span: Span,
    provider: Provider[Any, Any, Any, Any],
    error: Exception,
    start: Optional[float],
) -> None:
    if start is not None:
        record_chat_duration_error(provider, type(error).__name__, start)
    record_error(span, error)
    if span.is_recording():
        span.set_attribute("gen_ai.response.finish_reasons", ["error"])


def record_chat_duration_error(
    provider: Provider[Any, Any, Any, Any],
    error_type: str,
    start: float,
) -> None:
    record_histogram(
        "gen_ai.client.inference.duration",
        time.monotonic() - start,
        {**otel_metric_attributes(provider), "error.type": error_type},
    )


# Records a model request that was unwound before it was recorded as a
# success or an error, e.g. by a keyboard interrupt or an abandoned stream.
# Runs from a `finally`, mirroring how `TurnAccumulator.finalize_turn()`
# marks the partial turn as interrupted from its own `finally`.
def record_chat_span_interrupted(
    span: Span,
    provider: Provider[Any, Any, Any, Any],
    start: float,
) -> None:
    record_chat_duration_error(provider, "interrupted", start)
    if not span.is_recording():
        return
    span.set_status(StatusCode.ERROR)
    span.set_attribute("error.type", "interrupted")
    span.set_attribute("gen_ai.response.finish_reasons", ["interrupted"])


# Records the `gen_ai.client.inference.time_to_first_chunk` histogram. Per the
# client inference conventions, this is measured from request issuance to the
# first chunk received, whether or not that chunk contains model output.
# The span attribute is measured at the first non-empty token by
# `record_chat_ttft_attr()`.
# See: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/client-inference.md
def record_chat_ttft(
    provider: Provider[Any, Any, Any, Any],
    start: float,
) -> None:
    ttft = time.monotonic() - start
    record_histogram(
        "gen_ai.client.inference.time_to_first_chunk",
        ttft,
        otel_metric_attributes(provider),
    )


# Records the `gen_ai.client.inference.time_per_output_chunk` histogram. Per
# the client inference conventions, it is "recorded for each chunk received
# after the first one, measured as the time elapsed from the end of the
# previous chunk to the end of the current chunk". `previous_chunk_time` is
# the time the previous chunk was received.
def record_chat_chunk(
    provider: Provider[Any, Any, Any, Any],
    previous_chunk_time: float,
) -> None:
    record_histogram(
        "gen_ai.client.inference.time_per_output_chunk",
        time.monotonic() - previous_chunk_time,
        otel_metric_attributes(provider),
    )


# Per-invocation counts backing the invoke_agent span attributes and metrics.
class AgentOtelTally:
    def __init__(self) -> None:
        self.start = time.monotonic()
        self.inference_calls = 0
        self.tool_calls = 0
        # (input, output, cached_input), summed over assistant turns
        self.tokens = [0, 0, 0]
        # `error.type` for the invocation: an error class name or "cancelled".
        # An invocation that ends without completing or setting an error type
        # was interrupted.
        self.error_type: Optional[str] = None
        self.completed = False

    # Inference calls are counted when the request starts (so failed requests
    # are included); tool calls and tokens come from the completed turn.
    def note_request(self) -> None:
        self.inference_calls += 1

    def note_turn(self, turn: AssistantTurn[Any]) -> None:
        self.tool_calls += sum(
            1 for c in turn.contents if isinstance(c, ContentToolRequest)
        )
        if turn.tokens is not None:
            for i in range(3):
                self.tokens[i] += turn.tokens[i]


# Records the invoke_agent metrics and, mirroring the chat span, also sets
# them as attributes on the invoke_agent span.
def record_agent_otel(
    span: Span,
    provider: Provider[Any, Any, Any, Any],
    tally: AgentOtelTally,
) -> None:
    attributes = otel_metric_attributes(provider)
    attributes["gen_ai.operation.name"] = "invoke_agent"
    error_type = tally.error_type or (None if tally.completed else "interrupted")

    duration = time.monotonic() - tally.start
    # Only the duration metric defines `error.type` in the semantic conventions.
    record_histogram(
        "gen_ai.invoke_agent.duration",
        duration,
        {**attributes, "error.type": error_type},
    )
    record_histogram(
        "gen_ai.invoke_agent.inference_calls",
        tally.inference_calls,
        attributes,
    )
    record_histogram(
        "gen_ai.invoke_agent.tool_calls",
        tally.tool_calls,
        attributes,
    )

    if not span.is_recording():
        return
    span.set_attribute("gen_ai.invoke_agent.duration", duration)
    span.set_attribute("gen_ai.invoke_agent.inference_calls", tally.inference_calls)
    span.set_attribute("gen_ai.invoke_agent.tool_calls", tally.tool_calls)
    if error_type is not None:
        # The exception event is recorded once, on the chat or tool span where
        # it occurred; the agent span only carries the status and error type.
        span.set_status(StatusCode.ERROR)
        span.set_attribute("error.type", error_type)
    input_tokens = tally.tokens[0] + tally.tokens[2]
    output_tokens = tally.tokens[1]
    if input_tokens > 0 or output_tokens > 0:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)


def record_tool_otel_result(
    request: ContentToolRequest,
    start: float,
    result: Optional[ContentToolResult],
) -> None:
    error_type = None
    if result is not None and result.error is not None:
        error_type = _tool_error_type(result.error)
    record_tool_otel_duration(request, start, error_type)


def record_tool_otel_duration(
    request: ContentToolRequest,
    start: float,
    error_type: Optional[str] = None,
) -> None:
    record_histogram(
        "gen_ai.execute_tool.duration",
        time.monotonic() - start,
        {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": request.name,
            "gen_ai.tool.type": "function",
            "error.type": error_type,
        },
    )


def _tool_error_type(error: Any) -> str:
    return "error" if isinstance(error, str) else type(error).__name__


# The tool counterpart of `record_chat_span_interrupted()`: runs from a
# `finally` and records the call only if it was unwound before
# `record_tool_otel_result()` ran.
def record_tool_otel_interrupted(
    span: Span,
    request: ContentToolRequest,
    start: float,
) -> None:
    record_tool_otel_duration(request, start, "interrupted")
    if not span.is_recording():
        return
    span.set_status(StatusCode.ERROR)
    span.set_attribute("error.type", "interrupted")


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
        else:
            part["response"] = tool_otel_response(content.value)
        return part

    return {"type": "generic", "class": type(content).__name__}


def tool_otel_response(value: Any) -> Any:
    # `Content` values (e.g. an image) aren't JSON-serializable as-is, so
    # convert them to otel parts rather than dropping the message.
    if isinstance(value, Content):
        return as_otel_part(value)
    if isinstance(value, (list, tuple)) and all(
        isinstance(v, Content) for v in value
    ):
        return [as_otel_part(v) for v in value]
    return json_safe(value)


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
