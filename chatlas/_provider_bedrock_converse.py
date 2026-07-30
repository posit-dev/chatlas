from __future__ import annotations

import asyncio
import base64
from functools import cache
from typing import (
    TYPE_CHECKING,
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterator,
    cast,
)

import httpx

from ._content import (
    Content,
    ContentImageInline,
    ContentImageRemote,
    ContentPDF,
    ContentText,
    ContentThinking,
    ContentToolRequest,
    ContentToolResult,
)
from ._tools import Tool, ToolBuiltIn
from ._turn import AssistantTurn, SystemTurn, Turn, UserTurn

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
    from botocore.credentials import Credentials
    from botocore.model import Shape
    from mypy_boto3_bedrock_runtime.literals import ImageFormatType
    from mypy_boto3_bedrock_runtime.type_defs import (
        ContentBlockUnionTypeDef,
        MessageUnionTypeDef,
        SystemContentBlockTypeDef,
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
        yield parser.parse(event.to_response_dict(), shape)


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

    def __init__(self, credentials: Credentials, region: str):
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


def as_converse_content(
    content: Content, *, document_index: int = 0
) -> ContentBlockUnionTypeDef:
    if isinstance(content, ContentText):
        return {"text": content.text}
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
                "source": {"bytes": content.data},
            }
        }
    elif isinstance(content, ContentToolRequest):
        return {
            "toolUse": {
                "toolUseId": content.id,
                "name": content.name,
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
        return {
            "reasoningContent": {
                "reasoningText": {
                    "text": content.thinking,
                    "signature": extra.get("signature", ""),
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
        messages.append({"role": role, "content": content})
    return messages


def as_converse_system(turns: list[Turn]) -> list[SystemContentBlockTypeDef]:
    return [{"text": turn.text} for turn in turns if isinstance(turn, SystemTurn)]


def as_converse_tools(tools: dict[str, Tool | ToolBuiltIn]) -> ToolConfigurationTypeDef:
    return {"tools": [converse_tool_spec(tool) for tool in tools.values()]}


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
