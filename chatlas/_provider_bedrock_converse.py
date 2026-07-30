from __future__ import annotations

import asyncio
from functools import cache
from typing import TYPE_CHECKING, AsyncGenerator, AsyncIterator, Generator, Iterator

import httpx

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
