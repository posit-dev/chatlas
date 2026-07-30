import binascii
import json
import struct
from typing import cast

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


class TestContentSerialization:
    def test_text_becomes_a_text_block(self):
        from chatlas._content import ContentText
        from chatlas._provider_bedrock_converse import as_converse_content

        assert as_converse_content(ContentText(text="hi")) == {"text": "hi"}

    def test_inline_image_becomes_an_image_block_with_raw_bytes(self):
        import base64

        from chatlas._content import ContentImageInline
        from chatlas._provider_bedrock_converse import as_converse_content

        raw = b"\x89PNG\r\n\x1a\n"
        content = ContentImageInline(
            image_content_type="image/png",
            data=base64.b64encode(raw).decode(),
        )
        # Converse takes raw bytes, not base64 -- unlike Anthropic's API.
        assert as_converse_content(content) == {
            "image": {"format": "png", "source": {"bytes": raw}}
        }

    def test_tool_request_becomes_a_tool_use_block(self):
        from chatlas._content import ContentToolRequest
        from chatlas._provider_bedrock_converse import as_converse_content

        req = ContentToolRequest(
            id="t1", name="get_weather", arguments={"city": "Paris"}
        )
        assert as_converse_content(req) == {
            "toolUse": {
                "toolUseId": "t1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }
        }

    def test_tool_result_becomes_a_tool_result_block(self):
        from chatlas._content import ContentToolRequest, ContentToolResult
        from chatlas._provider_bedrock_converse import as_converse_content

        req = ContentToolRequest(id="t1", name="get_weather", arguments={})
        result = ContentToolResult(value="sunny", request=req)
        # boto3-stubs models each Converse content block as one flat,
        # all-NotRequired TypedDict (a union-by-optional-fields, not a
        # discriminated union), so pyright can't narrow "toolResult" as
        # present from the runtime dispatch alone.
        block = cast(dict, as_converse_content(result))
        assert block["toolResult"]["toolUseId"] == "t1"
        assert block["toolResult"]["status"] == "success"
        assert block["toolResult"]["content"] == [{"text": "sunny"}]

    def test_failed_tool_result_is_marked_as_error(self):
        from chatlas._content import ContentToolRequest, ContentToolResult
        from chatlas._provider_bedrock_converse import as_converse_content

        req = ContentToolRequest(id="t1", name="boom", arguments={})
        result = ContentToolResult(value=None, error=RuntimeError("nope"), request=req)
        block = cast(dict, as_converse_content(result))
        assert block["toolResult"]["status"] == "error"

    def test_system_prompt_is_split_out_of_messages(self):
        from chatlas._provider_bedrock_converse import (
            as_converse_messages,
            as_converse_system,
        )
        from chatlas._turn import SystemTurn, UserTurn

        turns = [SystemTurn("be terse"), UserTurn("hi")]
        assert as_converse_system(turns) == [{"text": "be terse"}]
        messages = as_converse_messages(turns)
        assert [m["role"] for m in messages] == ["user"]

    def test_tools_become_a_tool_config(self):
        from chatlas._provider_bedrock_converse import as_converse_tools
        from chatlas._tools import Tool

        def get_weather(city: str) -> str:
            "Get weather for a city"
            return "sunny"

        tool = Tool.from_func(get_weather)
        config = as_converse_tools({tool.name: tool})
        spec = cast(dict, config["tools"][0])["toolSpec"]
        assert spec["name"] == "get_weather"
        assert spec["description"] == "Get weather for a city"
        assert spec["inputSchema"]["json"]["type"] == "object"
        assert "city" in spec["inputSchema"]["json"]["properties"]
