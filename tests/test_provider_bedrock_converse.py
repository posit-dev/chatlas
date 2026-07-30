import binascii
import json
import struct

import httpx
import pytest
from botocore.credentials import Credentials


def eventstream_frame(payload: bytes, event_type: str) -> bytes:
    """Encode one AWS eventstream frame the way bedrock-runtime does."""
    headers = b""
    for name, value in (
        (":event-type", event_type),
        (":content-type", "application/json"),
        (":message-type", "event"),
    ):
        headers += bytes([len(name)]) + name.encode()
        headers += b"\x07" + struct.pack(">H", len(value)) + value.encode()

    total_len = 16 + len(headers) + len(payload)
    prelude = struct.pack(">II", total_len, len(headers))
    prelude += struct.pack(">I", binascii.crc32(prelude))
    frame = prelude + headers + payload
    return frame + struct.pack(">I", binascii.crc32(frame))


def make_stream() -> bytes:
    return b"".join(
        [
            eventstream_frame(
                json.dumps({"role": "assistant"}).encode(), "messageStart"
            ),
            eventstream_frame(
                json.dumps(
                    {"contentBlockIndex": 0, "delta": {"text": "Hello"}}
                ).encode(),
                "contentBlockDelta",
            ),
            eventstream_frame(
                json.dumps(
                    {"contentBlockIndex": 0, "delta": {"text": " world"}}
                ).encode(),
                "contentBlockDelta",
            ),
            eventstream_frame(
                json.dumps({"contentBlockIndex": 0}).encode(), "contentBlockStop"
            ),
            eventstream_frame(
                json.dumps({"stopReason": "end_turn"}).encode(), "messageStop"
            ),
            eventstream_frame(
                json.dumps(
                    {"usage": {"inputTokens": 10, "outputTokens": 3, "totalTokens": 13}}
                ).encode(),
                "metadata",
            ),
        ]
    )


class TestEventstreamDecoding:
    def test_decodes_frames_into_converse_events(self):
        from chatlas._provider_bedrock_converse import decode_eventstream

        events = list(decode_eventstream(iter([make_stream()])))
        kinds = [next(iter(e)) for e in events]
        assert kinds == [
            "messageStart",
            "contentBlockDelta",
            "contentBlockDelta",
            "contentBlockStop",
            "messageStop",
            "metadata",
        ]
        text = "".join(
            e["contentBlockDelta"]["delta"]["text"]
            for e in events
            if "contentBlockDelta" in e
        )
        assert text == "Hello world"
        assert events[-1]["metadata"]["usage"]["totalTokens"] == 13

    def test_handles_frames_split_across_chunk_boundaries(self):
        # httpx yields arbitrary byte boundaries, so a frame can straddle chunks.
        from chatlas._provider_bedrock_converse import decode_eventstream

        raw = make_stream()
        chunks = [raw[i : i + 7] for i in range(0, len(raw), 7)]
        events = list(decode_eventstream(iter(chunks)))
        assert len(events) == 6

    @pytest.mark.asyncio
    async def test_async_decoding_matches_sync(self):
        from chatlas._provider_bedrock_converse import decode_eventstream_async

        async def chunks():
            raw = make_stream()
            for i in range(0, len(raw), 13):
                yield raw[i : i + 13]

        events = [e async for e in decode_eventstream_async(chunks())]
        assert len(events) == 6
        assert events[1]["contentBlockDelta"]["delta"]["text"] == "Hello"


def make_request(**extra_headers: str) -> httpx.Request:
    headers = {
        "host": "bedrock-runtime.us-west-2.amazonaws.com",
        "content-type": "application/json",
        **extra_headers,
    }
    return httpx.Request(
        "POST",
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/foo/converse-stream",
        headers=headers,
        content=b'{"messages": []}',
    )


class TestBedrockSigV4Auth:
    def test_sign_adds_authorization_and_date_headers(self):
        from chatlas._provider_bedrock_converse import BedrockSigV4Auth

        auth = BedrockSigV4Auth(Credentials("AKIAEXAMPLE", "secret"), "us-west-2")
        signed = auth.sign(make_request())

        assert "AWS4-HMAC-SHA256" in signed.headers["authorization"]
        assert "x-amz-date" in signed.headers

    def test_credential_scope_uses_region_and_bedrock_service(self):
        # Pins the service name to "bedrock" (bedrock-runtime's SigV4 service),
        # not "bedrock-runtime" -- a mismatch here only fails at request time
        # against real AWS, never locally.
        from chatlas._provider_bedrock_converse import BedrockSigV4Auth

        auth = BedrockSigV4Auth(Credentials("AKIAEXAMPLE", "secret"), "us-west-2")
        signed = auth.sign(make_request())

        credential = (
            signed.headers["authorization"].split("Credential=")[1].split(",")[0]
        )
        assert credential.endswith("/us-west-2/bedrock/aws4_request")

    def test_only_configured_headers_are_signed(self):
        # The trap this pins: httpx/httpcore rewrite accept-encoding after
        # auth runs, so signing it yields a signature mismatch at AWS.
        from chatlas._provider_bedrock_converse import BedrockSigV4Auth

        auth = BedrockSigV4Auth(Credentials("AKIAEXAMPLE", "secret"), "us-west-2")
        signed = auth.sign(make_request(**{"accept-encoding": "gzip"}))

        signed_headers = (
            signed.headers["authorization"].split("SignedHeaders=")[1].split(",")[0]
        )
        assert "host" in signed_headers.split(";")
        assert "accept-encoding" not in signed_headers.split(";")

    @pytest.mark.asyncio
    async def test_async_auth_flow_yields_signed_request(self):
        from chatlas._provider_bedrock_converse import BedrockSigV4Auth

        auth = BedrockSigV4Auth(Credentials("AKIAEXAMPLE", "secret"), "us-west-2")
        signed = await auth.async_auth_flow(make_request()).__anext__()

        assert "AWS4-HMAC-SHA256" in signed.headers["authorization"]
