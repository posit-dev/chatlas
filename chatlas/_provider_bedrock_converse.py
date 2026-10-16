from __future__ import annotations

import asyncio
import base64
import json
import os
import warnings
from collections.abc import Mapping
from functools import cache
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Generator,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Protocol,
    Sequence,
    cast,
    overload,
)
from urllib.parse import quote

import httpx
from pydantic import BaseModel

from ._content import (
    Content,
    ContentImageInline,
    ContentImageRemote,
    ContentJson,
    ContentPDF,
    ContentText,
    ContentThinking,
    ContentToolRequest,
    ContentToolResult,
)
from ._content_file import ensure_bytes
from ._provider import (
    ModelInfo,
    Provider,
    StandardModelParamNames,
    StandardModelParams,
)
from ._provider_anthropic import (
    AnthropicProvider,
    StructuredOutputMode,
    supports_structured_outputs,
)
from ._provider_bedrock import (
    CROSS_REGION_PREFIX,
    bedrock_base_url,
    bedrock_credentials,
)
from ._tokens import get_price_info
from ._tools import Tool, ToolBuiltIn
from ._turn import (
    AssistantTurn,
    FinishReason,
    SystemTurn,
    Turn,
    UserTurn,
    caller_stacklevel,
    check_finish_reason,
)
from ._typing_extensions import NotRequired, TypedDict
from ._utils import is_async_callable
from .types.bedrock import ChatClientArgs as ConverseClientArgs

try:
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.eventstream import EventStreamBuffer
    from botocore.loaders import Loader
    from botocore.model import ServiceModel
    from botocore.parsers import EventStreamJSONParser
except ImportError:
    raise ImportError(
        '`ChatBedrock(api="converse")` requires the `botocore` package. '
        "Install it with `pip install chatlas[bedrock]`."
    )

if TYPE_CHECKING:
    from botocore.credentials import Credentials, ReadOnlyCredentials
    from botocore.model import Shape
    from mypy_boto3_bedrock_runtime.literals import ImageFormatType
    from mypy_boto3_bedrock_runtime.type_defs import (
        CachePointBlockTypeDef,
        ContentBlockOutputTypeDef,
        ContentBlockUnionTypeDef,
        ConverseRequestTypeDef,
        ConverseResponseTypeDef,
        InferenceConfigurationTypeDef,
        MessageUnionTypeDef,
        OutputConfigTypeDef,
        SystemContentBlockTypeDef,
        TokenUsageTypeDef,
        ToolChoiceTypeDef,
        ToolConfigurationTypeDef,
        ToolTypeDef,
    )


def decode_eventstream(chunks: Iterator[bytes]) -> Iterator[dict]:
    """
    Decode bedrock-runtime's binary eventstream framing into Converse events.

    `converse-stream` does not use SSE -- it uses AWS's own binary framing, so
    the raw bytes are fed through botocore's buffer, which yields whole frames
    once enough bytes have arrived.
    """
    buffer = EventStreamBuffer()
    parser = EventStreamJSONParser()
    shape = converse_stream_shape()
    for chunk in chunks:
        buffer.add_data(chunk)
        yield from events_from_buffer(buffer, parser, shape)


async def decode_eventstream_async(chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
    buffer = EventStreamBuffer()
    parser = EventStreamJSONParser()
    shape = converse_stream_shape()
    async for chunk in chunks:
        buffer.add_data(chunk)
        for event in events_from_buffer(buffer, parser, shape):
            yield event


@cache
def converse_stream_shape() -> Shape:
    # Loading the service model is slow enough to be worth caching, and the
    # shape never changes within a process.
    model = ServiceModel(Loader().load_service_model("bedrock-runtime", "service-2"))
    return model.shape_for("ConverseStreamOutput")


def events_from_buffer(
    buffer: EventStreamBuffer, parser: EventStreamJSONParser, shape: Shape
) -> Iterator[dict]:
    # Parsing a buffered frame is synchronous either way, so both decoders
    # share this loop and only differ in how they feed the buffer.
    for event in buffer:
        parsed = parser.parse(event.to_response_dict(), shape)
        raise_for_converse_stream_event(parsed)
        yield parsed


def raise_for_converse_stream_event(event: dict) -> None:
    error = event.get("Error")
    if not isinstance(error, dict):
        return
    code = error.get("Code") or "unknown error"
    message = error.get("Message") or "No error message returned."
    raise ValueError(f"Bedrock Converse stream failed with {code}: {message}")


class CredentialsLike(Protocol):
    """Anything that can hand back frozen SigV4 credentials on demand.

    A structural (not nominal) type: `botocore.credentials.Credentials`
    satisfies this without any change, and so does `LazyCredentials` below,
    which `BedrockConverseProvider` passes instead -- deferring the botocore
    credential chain to the first signed request rather than resolving it
    when the auth hook is constructed.
    """

    def get_frozen_credentials(self) -> "ReadOnlyCredentials": ...


class BedrockSigV4Auth(httpx.Auth):
    """
    Signs bedrock-runtime requests with AWS SigV4.

    The Converse API is spoken over raw httpx (no vendor SDK), so signing
    hooks in at the httpx layer. Note the service name is "bedrock"
    (bedrock-runtime), unlike the mantle endpoint's "bedrock-mantle".
    """

    requires_request_body = True

    # Sign only headers that reach the wire unchanged. httpx/httpcore rewrite
    # `accept-encoding` and `connection` after auth runs -- signing them
    # yields a signature mismatch at AWS.
    signed_headers = frozenset({"host", "content-type"})

    def __init__(self, credentials: CredentialsLike, region: str):
        self._credentials = credentials
        self._region = region

    def sign(self, request: httpx.Request) -> httpx.Request:
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() in self.signed_headers
        }
        aws_request = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers=headers,
        )
        # Frozen per request so SSO/STS credential refresh is picked up.
        SigV4Auth(
            self._credentials.get_frozen_credentials(), "bedrock", self._region
        ).add_auth(aws_request)
        request.headers.update(dict(aws_request.headers))
        return request

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        yield self.sign(request)

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        if self.requires_request_body:
            await request.aread()
        # httpx's default async_auth_flow runs auth_flow on the event loop
        # thread; botocore may refresh SSO/STS credentials with a synchronous
        # network call there, stalling every other task on the loop.
        yield await asyncio.to_thread(self.sign, request)


class BedrockBearerAuth(httpx.Auth):
    """Adds an AWS Bedrock bearer token to direct Converse API requests."""

    def __init__(self, token: str):
        self._token = token

    def auth_flow(
        self, request: httpx.Request
    ) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self._token}"
        yield request


def as_converse_content(
    content: Content, *, document_index: int = 0
) -> ContentBlockUnionTypeDef:
    if isinstance(content, ContentText):
        return {"text": content.text}
    elif isinstance(content, ContentJson):
        return {"text": json.dumps(content.value, separators=(",", ":"))}
    elif isinstance(content, ContentImageInline):
        if content.image_content_type not in CONVERSE_IMAGE_FORMATS:
            raise ValueError(
                f"Unsupported image content type for Bedrock Converse: "
                f"{content.image_content_type}"
            )
        return {
            "image": {
                "format": CONVERSE_IMAGE_FORMATS[content.image_content_type],
                "source": {"bytes": base64.b64decode(content.data)},
            }
        }
    elif isinstance(content, ContentImageRemote):
        raise ValueError(
            "Remote images aren't supported by Bedrock's Converse API, which "
            "only accepts inline bytes or an S3 location. Consider downloading "
            "the image and using content_image_file() instead."
        )
    elif isinstance(content, ContentPDF):
        return {
            "document": {
                "format": "pdf",
                # DocumentBlock.name rejects many characters (e.g. those in
                # user filenames) and must be unique per request, so it's
                # derived from position rather than from `content.filename`.
                "name": f"document-{document_index}",
                "source": {"bytes": ensure_bytes(content, "document")},
            }
        }
    elif isinstance(content, ContentToolRequest):
        return {
            "toolUse": {
                "toolUseId": content.id,
                "name": content.name,
                # `arguments` is deliberately typed `object` upstream (it's
                # whatever a tool call decoded), so a cast is needed to hand
                # it to Converse's `Mapping[str, Any]` input.
                "input": cast(dict, content.arguments),
            }
        }
    elif isinstance(content, ContentToolResult):
        value = content.get_model_value()
        text = value if isinstance(value, str) else str(value)
        return {
            "toolResult": {
                "toolUseId": content.id,
                "content": [{"text": text}],
                "status": "error" if content.error else "success",
            }
        }
    elif isinstance(content, ContentThinking):
        extra = content.extra or {}
        redacted_content = extra.get("redactedContent")
        if redacted_content is not None:
            if not isinstance(redacted_content, bytes):
                raise ValueError(
                    "Bedrock Converse redacted reasoning content must be bytes."
                )
            return {"reasoningContent": {"redactedContent": redacted_content}}
        signature = extra.get("signature")
        if not signature:
            # Claude models reject unsigned reasoning, so replay it as text
            return {"text": str(content)}
        return {
            "reasoningContent": {
                "reasoningText": {
                    "text": content.thinking,
                    "signature": signature,
                }
            }
        }
    raise ValueError(f"Unknown content type: {type(content)}")


def as_converse_messages(turns: list[Turn]) -> list[MessageUnionTypeDef]:
    messages: list[MessageUnionTypeDef] = []
    index = 0
    for turn in turns:
        if isinstance(turn, SystemTurn):
            continue  # system prompt passed as separate arg
        if not isinstance(turn, (UserTurn, AssistantTurn)):
            raise ValueError(f"Unknown role {turn.role}")

        role = "user" if isinstance(turn, UserTurn) else "assistant"
        content: list[ContentBlockUnionTypeDef] = []
        for c in turn.contents:
            content.append(as_converse_content(c, document_index=index))
            index += 1

        # Converse requires non-empty content, and dropping the turn instead
        # would leave two consecutive user messages.
        if role == "assistant" and not content:
            content = [{"text": "[empty string]"}]

        messages.append({"role": role, "content": content})
    return messages


def as_converse_system(turns: list[Turn]) -> list[SystemContentBlockTypeDef]:
    return [{"text": turn.text} for turn in turns if isinstance(turn, SystemTurn)]


def as_converse_tools(tools: dict[str, Tool | ToolBuiltIn]) -> ToolConfigurationTypeDef:
    return {"tools": [converse_tool_spec(tool) for tool in tools.values()]}


def converse_cache_point_enabled(
    cache: Literal["auto", "5m", "1h", "none"], model: str
) -> bool:
    """Whether to append a Converse cache point to this request's system blocks."""
    if cache == "none":
        return False
    if cache in ("5m", "1h"):
        return True
    model_id = CROSS_REGION_PREFIX.sub("", model)
    return model_id.startswith(("anthropic.", "amazon.nova"))


def converse_cache_point(
    cache: Literal["auto", "5m", "1h", "none"],
) -> CachePointBlockTypeDef:
    """Build the cache point block for `cache`.

    `ttl` is omitted for `"5m"`/`"auto"` since `type: "default"` alone already
    means a 5-minute TTL -- only `"1h"` needs to be spelled out explicitly.
    """
    if cache == "1h":
        return {"type": "default", "ttl": "1h"}
    return {"type": "default"}


CONVERSE_IMAGE_FORMATS: dict[str, ImageFormatType] = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}


def converse_tool_spec(tool: Tool | ToolBuiltIn) -> ToolTypeDef:
    if isinstance(tool, ToolBuiltIn):
        raise ValueError(
            f"Built-in tool '{tool.name}' is not supported by "
            '`ChatBedrock(api="converse")`.'
        )
    fn = tool.schema["function"]
    return {
        "toolSpec": {
            "name": fn["name"],
            "description": fn.get("description") or "",
            "inputSchema": {"json": fn.get("parameters") or {"type": "object"}},
        }
    }


class BedrockConverseProvider(
    Provider[
        "ConverseResponseTypeDef",
        dict,
        "ConverseAccumulator",
        "ConverseSubmitArgs",
    ]
):
    """
    Talks to bedrock-runtime's Converse API directly over httpx.

    Unlike every other chatlas provider, there is no vendor chat SDK to
    subclass here: `BedrockSigV4Auth` (this module's own httpx auth hook, from
    Task 1) signs requests instead.
    """

    def __init__(
        self,
        *,
        model: str,
        aws_profile: Optional[str],
        aws_region: str,
        base_url: Optional[str],
        max_tokens: int = 4096,
        cache: Literal["auto", "5m", "1h", "none"] = "auto",
        structured_output_mode: StructuredOutputMode = "auto",
        name: str = "AWS/Bedrock",
        kwargs: Optional[ConverseClientArgs] = None,
    ):
        super().__init__(name=name, model=model)
        self._aws_profile = aws_profile
        self._aws_region = aws_region
        self._max_tokens = max_tokens
        self._cache: Literal["auto", "5m", "1h", "none"] = cache
        self._structured_output_mode = structured_output_mode
        # Models that have rejected `outputConfig` with a 400 (see
        # `_native_retry_args`); only populated in "auto" mode.
        self._native_unsupported_models: set[str] = set()

        resolved_base_url = base_url or bedrock_base_url("converse", aws_region)
        # `LazyCredentials` defers the botocore credential chain to the first
        # signed request. Resolving it here instead (e.g. via
        # `bedrock_credentials(aws_profile)`, which eagerly freezes credentials
        # to fail fast on a broken chain) would break offline construction --
        # this provider is built well before, or without ever, sending a
        # request.
        api_key = kwargs.get("api_key") if kwargs else None
        client_kwargs = cast(dict[str, Any], dict(kwargs or {}))
        client_kwargs.pop("api_key", None)
        token = api_key
        if token is None and aws_profile is None:
            token = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        auth: httpx.Auth
        if token:
            auth = BedrockBearerAuth(token)
        else:
            auth = BedrockSigV4Auth(LazyCredentials(aws_profile), aws_region)
        sync_client_kwargs, async_client_kwargs = split_httpx_client_kwargs(
            client_kwargs
        )
        self._client = httpx.Client(
            auth=auth, base_url=resolved_base_url, **sync_client_kwargs
        )
        self._async_client = httpx.AsyncClient(
            auth=auth, base_url=resolved_base_url, **async_client_kwargs
        )

    def close(self) -> None:
        self._client.close()

    async def close_async(self) -> None:
        self.close()
        await self._async_client.aclose()

    def list_models(self) -> list[ModelInfo]:
        # boto3 should come via the `bedrock` extra's `anthropic[bedrock]`,
        # same precondition as AnthropicBedrockProvider.list_models().
        import boto3

        bedrock = boto3.Session(
            profile_name=self._aws_profile, region_name=self._aws_region
        ).client("bedrock")
        resp = bedrock.list_foundation_models()

        res: list[ModelInfo] = []
        for m in resp["modelSummaries"]:
            pricing = get_price_info(self.name, m["modelId"]) or {}
            info: ModelInfo = {
                "id": m["modelId"],
                "name": m["modelName"],
                "provider": m["providerName"],
                "input": pricing.get("input"),
                "output": pricing.get("output"),
                "cached_input": pricing.get("cached_input"),
            }
            res.append(info)

        return res

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional[ConverseSubmitArgs] = None,
    ) -> ConverseResponseTypeDef: ...

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional[ConverseSubmitArgs] = None,
    ) -> Iterable[dict]: ...

    def chat_perform(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional[ConverseSubmitArgs] = None,
    ) -> Iterable[dict] | ConverseResponseTypeDef:
        args = self._chat_perform_args(stream, turns, tools, data_model, kwargs)
        if stream:
            return self._converse_stream_with_retry(
                args, stream, turns, tools, data_model, kwargs
            )
        try:
            return self._converse(args)
        except ValueError as error:
            retry = self._native_retry_args(
                error, args, stream, turns, tools, data_model, kwargs
            )
            if retry is None:
                raise
            return self._converse(retry)

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional[ConverseSubmitArgs] = None,
    ) -> ConverseResponseTypeDef: ...

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional[ConverseSubmitArgs] = None,
    ) -> AsyncIterable[dict]: ...

    async def chat_perform_async(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional[ConverseSubmitArgs] = None,
    ) -> AsyncIterable[dict] | ConverseResponseTypeDef:
        args = self._chat_perform_args(stream, turns, tools, data_model, kwargs)
        if stream:
            return self._converse_stream_async_with_retry(
                args, stream, turns, tools, data_model, kwargs
            )
        try:
            return await self._converse_async(args)
        except ValueError as error:
            retry = self._native_retry_args(
                error, args, stream, turns, tools, data_model, kwargs
            )
            if retry is None:
                raise
            return await self._converse_async(retry)

    def _native_retry_args(
        self,
        error: ValueError,
        args: ConverseRequestTypeDef,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]],
        kwargs: Optional[ConverseSubmitArgs],
    ) -> Optional[ConverseRequestTypeDef]:
        """Rebuild request args for tool-based extraction after a native-mode 400.

        Converse's `outputConfig` support is per-model in a way that can't be
        derived from the model ID (e.g. `claude-sonnet-5` rejects it while
        `claude-sonnet-4-6` accepts it), so "auto" mode is optimistic: on the
        specific rejection, remember the model as unsupported and retry with
        the forced tool call. Returns `None` (caller re-raises) when the
        failure wasn't a native-mode attempt -- including explicit "native"
        mode, where the user asked for `outputConfig` and gets its error.

        Retry eligibility comes from the failed request's own `outputConfig`,
        not the unsupported-model cache: a concurrent request may have
        populated the cache while this one was in flight, and it still
        deserves its retry.
        """
        if (
            data_model is None
            or self._structured_output_mode != "auto"
            or "outputConfig" not in args
            or not self._is_native_rejection(error)
        ):
            return None
        if self.model not in self._native_unsupported_models:
            self._native_unsupported_models.add(self.model)
            warnings.warn(
                f"Bedrock Converse model '{self.model}' rejected native "
                "structured outputs (`outputConfig`); falling back to "
                "tool-based extraction. This provider will use tool-based "
                "extraction for the model from now on.",
                stacklevel=caller_stacklevel(),
            )
        # `_use_native_structured_output()` is now False, so this rebuilds the
        # args with the forced `_structured_tool_call` instead of `outputConfig`.
        return self._chat_perform_args(stream, turns, tools, data_model, kwargs)

    @staticmethod
    def _is_native_rejection(error: ValueError) -> bool:
        # The specific 400 returned by models that lack `outputConfig` support
        # (e.g. "... status 400: ... output_config.format: Extra inputs are
        # not permitted"). Other errors mentioning `output_config` -- notably
        # schema-validation 400s, which imply the model *does* support native
        # output -- must propagate rather than disable native mode.
        message = str(error)
        return (
            "output_config" in message
            and "Extra inputs are not permitted" in message
        )

    def _converse_stream_with_retry(
        self,
        args: ConverseRequestTypeDef,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]],
        kwargs: Optional[ConverseSubmitArgs],
    ) -> Iterator[dict]:
        if data_model is not None and "outputConfig" not in args:
            # A tool-based structured request can't stream (tool input
            # arrives in its own event type, not as text), so serve it via
            # the non-streaming endpoint. `Chat` normally downgrades these
            # via `can_stream()`, but a concurrent request can mark the model
            # unsupported after that check and before args were built.
            yield from self._response_as_stream_events(self._converse(args))
            return
        events: Optional[Iterator[dict]] = None
        try:
            try:
                # `_converse_stream` is a generator, so the request (and any
                # error it fails with) doesn't fire until the first pull.
                events = self._converse_stream(args)
                first = next(events)
            except StopIteration:
                return
            except ValueError as error:
                retry = self._native_retry_args(
                    error, args, stream, turns, tools, data_model, kwargs
                )
                if retry is None:
                    raise
                # A tool-mode stream retry would yield zero chunks (the bug
                # #451 fixed), so retry on the non-streaming endpoint and
                # synthesize events that surface the extracted JSON as a
                # single text chunk.
                yield from self._response_as_stream_events(self._converse(retry))
                return
            yield first
            yield from events
        finally:
            # Closing this wrapper (e.g. on cancellation) must also close the
            # inner generator, which owns the httpx streaming response.
            close = getattr(events, "close", None)
            if close is not None:
                close()

    async def _converse_stream_async_with_retry(
        self,
        args: ConverseRequestTypeDef,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]],
        kwargs: Optional[ConverseSubmitArgs],
    ) -> AsyncIterator[dict]:
        if data_model is not None and "outputConfig" not in args:
            # See the sync variant: tool-based structured requests can't
            # stream, so serve them via the non-streaming endpoint.
            for event in self._response_as_stream_events(
                await self._converse_async(args)
            ):
                yield event
            return
        events: Optional[AsyncIterator[dict]] = None
        try:
            try:
                events = self._converse_stream_async(args)
                first = await events.__anext__()
            except StopAsyncIteration:
                return
            except ValueError as error:
                retry = self._native_retry_args(
                    error, args, stream, turns, tools, data_model, kwargs
                )
                if retry is None:
                    raise
                for event in self._response_as_stream_events(
                    await self._converse_async(retry)
                ):
                    yield event
                return
            yield first
            async for event in events:
                yield event
        finally:
            # `async for` doesn't forward `aclose()`, so closing this wrapper
            # (e.g. on cancellation) would leave the inner generator -- and
            # its httpx streaming response -- open until GC.
            aclose = getattr(events, "aclose", None)
            if aclose is not None:
                await aclose()

    @staticmethod
    def _response_as_stream_events(
        response: ConverseResponseTypeDef,
    ) -> Iterator[dict]:
        """Replay a non-streaming response as a minimal Converse event stream.

        Structured output is re-expressed as a text delta (rather than the
        tool-use block it arrived as) so `stream_content()` yields the
        extracted JSON as a single chunk and `_as_turn`'s native branch
        records it as `ContentJson`.
        """
        message = response["output"].get("message")
        yield {"messageStart": {"role": (message or {}).get("role", "assistant")}}
        if message is not None:
            blocks = message["content"]
            has_structured_tool = any(
                (cast(dict, block).get("toolUse") or {}).get("name")
                == "_structured_tool_call"
                for block in blocks
            )
            for index, block in enumerate(blocks):
                block = cast(dict, block)
                tool_use = block.get("toolUse") or {}
                if tool_use.get("name") == "_structured_tool_call":
                    text: Optional[str] = json.dumps(
                        tool_use.get("input", {}).get("data")
                    )
                elif has_structured_tool:
                    # Prose alongside the structured tool call would corrupt
                    # the extracted-JSON stream; drop it from the events.
                    continue
                else:
                    text = block.get("text")
                if text is not None:
                    yield {
                        "contentBlockDelta": {
                            "contentBlockIndex": index,
                            "delta": {"text": text},
                        }
                    }
                    yield {"contentBlockStop": {"contentBlockIndex": index}}
        yield {"messageStop": {"stopReason": response["stopReason"]}}
        yield {"metadata": {"usage": response["usage"]}}

    def can_stream(self, data_model: Optional[type[BaseModel]]) -> bool:
        # Native structured outputs (`outputConfig`) stream as ordinary text
        # deltas, so they honor the extracted-JSON contract. The tool-based
        # strategy streams tool input in its own event type (and
        # `{"data": ...}`-wrapped at that), so it can't -- `Chat` falls back
        # to a single non-streaming request there.
        if data_model is None:
            return True
        return self._use_native_structured_output()

    def _use_native_structured_output(self) -> bool:
        if self.model in self._native_unsupported_models:
            return False
        mode = self._structured_output_mode
        return mode == "native" or (
            mode == "auto" and supports_structured_outputs(self.model)
        )

    def stream_content(
        self,
        chunk: dict,
        completion: Optional[ConverseAccumulator],
        turns: Sequence[Turn] = (),
    ) -> Sequence[Content]:
        delta_event = chunk.get("contentBlockDelta")
        if delta_event is None:
            return []
        text = delta_event.get("delta", {}).get("text")
        if text is None:
            return []
        return [ContentText.model_construct(text=text)]

    def stream_merge_chunks(
        self,
        completion: Optional[ConverseAccumulator],
        chunk: dict,
    ) -> Optional[ConverseAccumulator]:
        merged: ConverseAccumulator = completion or {}
        if "messageStart" in chunk:
            merged["role"] = chunk["messageStart"]["role"]
        elif "contentBlockStart" in chunk:
            start_event = chunk["contentBlockStart"]
            tool_use = start_event.get("start", {}).get("toolUse")
            if tool_use is not None:
                blocks = merged.setdefault("blocks", {})
                blocks[start_event["contentBlockIndex"]] = {
                    "toolUseId": tool_use["toolUseId"],
                    "name": tool_use["name"],
                    "input": [],
                }
        elif "contentBlockDelta" in chunk:
            delta_event = chunk["contentBlockDelta"]
            blocks = merged.setdefault("blocks", {})
            index = delta_event["contentBlockIndex"]
            text = delta_event.get("delta", {}).get("text")
            if text is not None:
                block = blocks.setdefault(index, {})
                block["text"] = block.get("text", "") + text
            tool_use = delta_event.get("delta", {}).get("toolUse")
            if tool_use is not None:
                block = blocks.setdefault(index, {})
                block.setdefault("input", []).append(tool_use["input"])
            reasoning = delta_event.get("delta", {}).get("reasoningContent")
            if reasoning is not None:
                block = blocks.setdefault(index, {})
                if "text" in reasoning:
                    block["reasoningText"] = (
                        block.get("reasoningText", "") + reasoning["text"]
                    )
                elif "signature" in reasoning:
                    block["reasoningSignature"] = reasoning["signature"]
        elif "contentBlockStop" in chunk:
            blocks = merged.get("blocks", {})
            block = blocks.get(chunk["contentBlockStop"]["contentBlockIndex"])
            if block is not None and "toolUseId" in block:
                raw_arguments = "".join(block.get("input", []))
                tool_name = block.get("name", "<unknown>")
                try:
                    arguments = json.loads(raw_arguments) if raw_arguments else {}
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f'Invalid JSON streamed for tool "{tool_name}".'
                    ) from exc
                block["arguments"] = cast(dict[str, Any], arguments)
        elif "messageStop" in chunk:
            merged["stopReason"] = chunk["messageStop"]["stopReason"]
        elif "metadata" in chunk:
            usage = chunk["metadata"].get("usage")
            if usage is not None:
                merged["usage"] = usage
        return merged

    def stream_turn(
        self,
        completion: ConverseAccumulator,
        has_data_model: bool,
        turns: Sequence[Turn] = (),
    ) -> AssistantTurn[ConverseResponseTypeDef]:
        # Reassemble into a ConverseResponseTypeDef shape so this can share
        # `_as_turn` with `value_turn` -- matching the pattern
        # SnowflakeProvider.stream_turn() uses for the same completion/chunk
        # type mismatch.
        content_blocks: list[ContentBlockOutputTypeDef] = []
        for _, block in sorted(completion.get("blocks", {}).items()):
            if "text" in block:
                content_blocks.append(
                    cast("ContentBlockOutputTypeDef", {"text": block["text"]})
                )
            elif "toolUseId" in block and "name" in block:
                content_blocks.append(
                    cast(
                        "ContentBlockOutputTypeDef",
                        {
                            "toolUse": {
                                "toolUseId": block["toolUseId"],
                                "name": block["name"],
                                "input": block.get("arguments", {}),
                            }
                        },
                    )
                )
            elif "reasoningText" in block or "reasoningSignature" in block:
                content_blocks.append(
                    cast(
                        "ContentBlockOutputTypeDef",
                        {
                            "reasoningContent": {
                                "reasoningText": {
                                    "text": block.get("reasoningText", ""),
                                    "signature": block.get("reasoningSignature", ""),
                                }
                            }
                        },
                    )
                )
        usage = completion.get("usage") or {
            "inputTokens": 0,
            "outputTokens": 0,
            "totalTokens": 0,
        }
        response = cast(
            "ConverseResponseTypeDef",
            {
                "output": {
                    "message": {
                        "role": completion.get("role", "assistant"),
                        "content": content_blocks,
                    }
                },
                "stopReason": completion.get("stopReason", "end_turn"),
                "usage": usage,
            },
        )
        return self._as_turn(response, has_data_model)

    def value_turn(
        self,
        completion: ConverseResponseTypeDef,
        has_data_model: bool,
        turns: Sequence[Turn] = (),
    ) -> AssistantTurn[ConverseResponseTypeDef]:
        return self._as_turn(completion, has_data_model)

    def value_tokens(
        self,
        completion: ConverseResponseTypeDef,
    ) -> tuple[int, int, int] | None:
        return converse_tokens(completion["usage"])

    def token_count(
        self,
        turns: list[Turn],
        *,
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]],
    ) -> int:
        raise self._no_token_count_support()

    async def token_count_async(
        self,
        turns: list[Turn],
        *,
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]],
    ) -> int:
        raise self._no_token_count_support()

    @staticmethod
    def _no_token_count_support() -> NotImplementedError:
        return NotImplementedError(
            "Bedrock's Converse API has no standalone token-counting endpoint. "
            "Call .chat() and read the usage from the returned turn instead "
            "(e.g. chat.get_last_turn().tokens)."
        )

    def translate_model_params(self, params: StandardModelParams) -> ConverseSubmitArgs:
        inference_config: InferenceConfigurationTypeDef = {}
        if "max_tokens" in params:
            inference_config["maxTokens"] = params["max_tokens"]
        if "temperature" in params:
            inference_config["temperature"] = params["temperature"]
        if "top_p" in params:
            inference_config["topP"] = params["top_p"]
        if "stop_sequences" in params:
            inference_config["stopSequences"] = params["stop_sequences"]

        if not inference_config:
            return {}
        return {"inferenceConfig": inference_config}

    def supported_model_params(self) -> set[StandardModelParamNames]:
        return {"max_tokens", "temperature", "top_p", "stop_sequences"}

    def _chat_perform_args(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional[ConverseSubmitArgs] = None,
    ) -> ConverseRequestTypeDef:
        inference_config: InferenceConfigurationTypeDef = {
            "maxTokens": self._max_tokens
        }
        args: ConverseRequestTypeDef = {
            "modelId": self.model,
            "messages": as_converse_messages(turns),
            "inferenceConfig": inference_config,
        }

        use_native_output = data_model is not None and (
            self._use_native_structured_output()
        )
        data_model_tool: Optional[Tool] = None
        tool_specs = [converse_tool_spec(tool) for tool in tools.values()]
        tool_config: Optional[ToolConfigurationTypeDef] = None
        if data_model is not None and not use_native_output:
            data_model_tool = AnthropicProvider.create_data_model_tool(data_model)
            tool_specs.append(converse_tool_spec(data_model_tool))

        if tool_specs:
            tool_config = {"tools": tool_specs}
            if data_model_tool is not None:
                tool_config["toolChoice"] = cast(
                    "ToolChoiceTypeDef",
                    {"tool": {"name": data_model_tool.name}},
                )
            args["toolConfig"] = tool_config

        extra = dict(cast(dict, kwargs or {}))
        # Merge inferenceConfig rather than replacing it outright, so a
        # per-request override (e.g. from `set_model_params(temperature=...)`)
        # doesn't silently drop `maxTokens`.
        extra_inference_config = extra.pop("inferenceConfig", None)
        if extra_inference_config:
            inference_config.update(extra_inference_config)
        args.update(extra)

        if data_model is not None and not use_native_output:
            # A caller-supplied raw `outputConfig` can't be honored for a
            # tool-based structured request (and would break the fallback in
            # `_native_retry_args`, whose rebuild re-applies `kwargs`).
            extra.pop("outputConfig", None)
            args.pop("outputConfig", None)

        if data_model is not None and use_native_output:
            from anthropic import transform_schema

            # After `args.update(extra)`, so chatlas's `outputConfig` wins
            # over a user-supplied one (mirroring `ChatAnthropic`).
            args["outputConfig"] = cast(
                "OutputConfigTypeDef",
                {
                    "textFormat": {
                        "type": "json_schema",
                        "structure": {
                            "jsonSchema": {
                                # The schema is a JSON *string* on the wire.
                                "schema": json.dumps(transform_schema(data_model)),
                                "name": data_model.__name__,
                            }
                        },
                    }
                },
            )

        extra_system = extra.get("system")
        system = (
            list(cast("list[SystemContentBlockTypeDef]", extra_system))
            if extra_system is not None
            else as_converse_system(turns)
        )
        if (
            converse_cache_point_enabled(self._cache, self.model)
            and system
            and not any("cachePoint" in block for block in system)
        ):
            system.append({"cachePoint": converse_cache_point(self._cache)})
        if system:
            args["system"] = system

        messages = args["messages"]
        if (
            converse_cache_point_enabled(self._cache, self.model)
            and messages
            and not any("cachePoint" in block for block in messages[-1]["content"])
        ):
            # Copy the list and the last message rather than mutating them in
            # place -- `messages` may be the caller's own `kwargs["messages"]`.
            messages = list(messages)
            messages[-1] = cast(
                "MessageUnionTypeDef",
                {
                    **messages[-1],
                    "content": [
                        *messages[-1]["content"],
                        {"cachePoint": converse_cache_point(self._cache)},
                    ],
                },
            )
            args["messages"] = messages

        if data_model_tool is not None:
            assert tool_config is not None
            args["toolConfig"] = tool_config

        return args

    def _request_target(
        self,
        args: ConverseRequestTypeDef,
        endpoint: Literal["converse", "converse-stream"],
    ) -> tuple[str, dict]:
        # `modelId` is a URL parameter at the wire level, not a body field --
        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        model_id = args["modelId"]
        body = {k: v for k, v in args.items() if k != "modelId"}
        return (
            f"/model/{quote(model_id, safe=':')}/{endpoint}",
            converse_wire_value(body),
        )

    def _converse(self, args: ConverseRequestTypeDef) -> ConverseResponseTypeDef:
        url, body = self._request_target(args, "converse")
        response = self._client.post(url, json=body)
        raise_for_converse_status(response)
        return converse_response_from_json(response.json())

    async def _converse_async(
        self, args: ConverseRequestTypeDef
    ) -> ConverseResponseTypeDef:
        url, body = self._request_target(args, "converse")
        response = await self._async_client.post(url, json=body)
        raise_for_converse_status(response)
        return converse_response_from_json(response.json())

    def _converse_stream(self, args: ConverseRequestTypeDef) -> Iterator[dict]:
        url, body = self._request_target(args, "converse-stream")
        with self._client.stream("POST", url, json=body) as response:
            if not response.is_success:
                response.read()
                raise_for_converse_status(response)
            yield from decode_eventstream(response.iter_bytes())

    async def _converse_stream_async(
        self, args: ConverseRequestTypeDef
    ) -> AsyncIterator[dict]:
        url, body = self._request_target(args, "converse-stream")
        async with self._async_client.stream("POST", url, json=body) as response:
            if not response.is_success:
                await response.aread()
                raise_for_converse_status(response)
            async for event in decode_eventstream_async(response.aiter_bytes()):
                yield event

    def _as_turn(
        self,
        completion: ConverseResponseTypeDef,
        has_data_model: bool,
    ) -> AssistantTurn[ConverseResponseTypeDef]:
        finish_reason = converse_stop_reason(completion["stopReason"])
        tokens = self.value_tokens(completion)
        if has_data_model:
            check_finish_reason(finish_reason, "error")

        # `message` is NotRequired -- absent e.g. when a guardrail intervened
        # before the model produced any content.
        message = completion["output"].get("message")
        contents: list[Content] = []
        if message is not None:
            # Structured output arrives in one of two forms, depending on
            # `structured_output_mode`: a `_structured_tool_call` toolUse
            # block (tool mode), or as ordinary text (native `outputConfig`
            # mode) -- mirroring `AnthropicProvider._as_turn`.
            uses_tool_approach = has_data_model and any(
                (cast(dict, block).get("toolUse") or {}).get("name")
                == "_structured_tool_call"
                for block in message["content"]
            )
            for block in message["content"]:
                tool_use = cast(dict, block).get("toolUse")
                if (
                    uses_tool_approach
                    and tool_use
                    and tool_use["name"] == "_structured_tool_call"
                ):
                    tool_input = tool_use["input"]
                    if not isinstance(tool_input, dict):
                        raise ValueError(
                            "Expected data extraction tool to return a dictionary."
                        )
                    data = tool_input.get("data")
                    if not isinstance(data, dict):
                        raise ValueError(
                            "Expected data extraction tool to return a 'data' dictionary."
                        )
                    contents.append(ContentJson(value=data))
                elif (
                    has_data_model
                    and not uses_tool_approach
                    and (text := cast(dict, block).get("text")) is not None
                ):
                    contents.append(ContentJson(value=json.loads(text)))
                elif (content := content_from_converse_block(block)) is not None:
                    contents.append(content)
        return AssistantTurn(
            contents,
            finish_reason=finish_reason,
            tokens=tokens,
            completion=completion,
        )


def split_httpx_client_kwargs(
    client_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sync_client_kwargs = client_kwargs.copy()
    async_client_kwargs = client_kwargs.copy()

    transport = client_kwargs.get("transport")
    if transport is not None:
        if not isinstance(transport, httpx.BaseTransport):
            sync_client_kwargs.pop("transport", None)
        if not isinstance(transport, httpx.AsyncBaseTransport):
            async_client_kwargs.pop("transport", None)

    mounts = client_kwargs.get("mounts")
    if isinstance(mounts, Mapping):
        sync_client_kwargs["mounts"] = {
            pattern: transport
            for pattern, transport in mounts.items()
            if transport is None or isinstance(transport, httpx.BaseTransport)
        }
        async_client_kwargs["mounts"] = {
            pattern: transport
            for pattern, transport in mounts.items()
            if transport is None or isinstance(transport, httpx.AsyncBaseTransport)
        }

    event_hooks = client_kwargs.get("event_hooks")
    if isinstance(event_hooks, Mapping):
        sync_client_kwargs["event_hooks"] = {
            event: [hook for hook in hooks if not is_async_callable(hook)]
            for event, hooks in event_hooks.items()
        }
        async_client_kwargs["event_hooks"] = {
            event: [hook for hook in hooks if is_async_callable(hook)]
            for event, hooks in event_hooks.items()
        }

    return sync_client_kwargs, async_client_kwargs


# https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html#API_runtime_Converse_RequestSyntax
CONVERSE_FINISH_REASONS: dict[str, FinishReason] = {
    "end_turn": "success",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
    "model_context_window_exceeded": "context_window",
}


def converse_stop_reason(reason: str) -> str:
    # `Turn.finish_reason` accepts `FinishReason | str`, so an unrecognized
    # Converse stop reason passes through raw rather than raising.
    return CONVERSE_FINISH_REASONS.get(reason, reason)


def converse_tokens(usage: TokenUsageTypeDef) -> tuple[int, int, int]:
    return (
        usage["inputTokens"],
        usage["outputTokens"],
        usage.get("cacheReadInputTokens", 0),
    )


def converse_wire_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, dict):
        return {key: converse_wire_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [converse_wire_value(item) for item in value]
    return value


def converse_response_from_json(
    response: dict[str, Any],
) -> ConverseResponseTypeDef:
    output = response.get("output")
    message = output.get("message") if isinstance(output, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            reasoning = block.get("reasoningContent")
            if not isinstance(reasoning, dict):
                continue
            redacted = reasoning.get("redactedContent")
            if isinstance(redacted, str):
                reasoning["redactedContent"] = base64.b64decode(redacted)

    return cast("ConverseResponseTypeDef", response)


def content_from_converse_block(
    block: ContentBlockOutputTypeDef,
) -> Optional[Content]:
    # Flat, all-NotRequired TypedDict (a union-by-optional-fields, not a
    # discriminated union), so dispatch on which key is present -- the same
    # shape Task 2's `as_converse_content` hits on the outbound side.
    fields = cast(dict, block)
    if "text" in fields:
        return ContentText(text=fields["text"])
    if "toolUse" in fields:
        tool_use = fields["toolUse"]
        return ContentToolRequest(
            id=tool_use["toolUseId"],
            name=tool_use["name"],
            arguments=tool_use["input"],
        )
    if "reasoningContent" in fields:
        reasoning_content = fields["reasoningContent"]
        if "redactedContent" in reasoning_content:
            return ContentThinking(
                thinking="",
                extra={"redactedContent": reasoning_content["redactedContent"]},
            )
        reasoning_text = reasoning_content.get("reasoningText") or {}
        return ContentThinking(
            thinking=reasoning_text.get("text", ""),
            extra={"signature": reasoning_text.get("signature", "")},
        )
    # Other block types (images, documents, citations, guardrail content,
    # search results) aren't part of a model's own turn; skip rather than
    # raise, so a new Converse block type doesn't break existing responses.
    return None


def converse_error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text
    if isinstance(body, dict):
        message = body.get("message") or body.get("Message")
        if message:
            return str(message)
    return response.text


def raise_for_converse_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    raise ValueError(
        f"Bedrock Converse request failed with status {response.status_code}: "
        f"{converse_error_message(response)}"
    )


class LazyCredentials:
    """Resolves AWS credentials from `profile` on first use, not at construction.

    `bedrock_credentials()` eagerly freezes credentials so a broken chain
    (expired SSO, no config) fails fast -- exactly the wrong behavior at
    `BedrockConverseProvider.__init__` time, since a provider is commonly
    constructed well before (or without ever) sending a request. Wrapping the
    profile here instead defers that resolution to the first `sign()` call,
    then memoizes it -- `.get_frozen_credentials()` is still called on every
    request after that, so SSO/STS refresh keeps working.
    """

    def __init__(self, profile: Optional[str]):
        self._profile = profile
        self._credentials: Optional[Credentials] = None

    def get_frozen_credentials(self) -> ReadOnlyCredentials:
        if self._credentials is None:
            self._credentials = bedrock_credentials(self._profile)
        return self._credentials.get_frozen_credentials()


class ConverseAccumulator(TypedDict):
    """Streaming state merged across `converse-stream` events into one dict.

    Blocks are keyed by `contentBlockIndex`, which keeps interleaved text and
    tool-use deltas separate until the stream is finalized.
    """

    role: NotRequired[str]
    blocks: NotRequired[dict[int, ConverseStreamBlock]]
    stopReason: NotRequired[str]
    usage: NotRequired[TokenUsageTypeDef]


class ConverseStreamBlock(TypedDict, total=False):
    text: str
    toolUseId: str
    name: str
    input: list[str]
    arguments: dict[str, Any]
    reasoningText: str
    reasoningSignature: str


class ConverseSubmitArgs(TypedDict, total=False):
    """Provider-specific args for `ChatBedrock(api="converse")`.

    A hand-written, all-optional subset of `ConverseRequestTypeDef`'s fields.
    `ConverseRequestTypeDef` itself marks `modelId` as its only required key
    (it's a URL parameter at the wire level, not a body field chatlas exposes
    here), which makes it unusable as-is for a "some or none of these" kwargs
    bag -- unlike every other provider's `SubmitInputArgs`, a literal or
    partial dict assigned to it fails a TypedDict completeness check.

    `additionalModelRequestFields` passes model-specific inference parameters
    straight through to Converse, e.g. to enable thinking on a Claude model:

    ```python
    chat.chat(
        "...",
        kwargs={
            "additionalModelRequestFields": {
                "thinking": {"type": "enabled", "budget_tokens": 4000}
            }
        },
    )
    ```
    """

    messages: Sequence[MessageUnionTypeDef]
    system: Sequence[SystemContentBlockTypeDef]
    inferenceConfig: InferenceConfigurationTypeDef
    toolConfig: ToolConfigurationTypeDef
    additionalModelRequestFields: Mapping[str, Any]
