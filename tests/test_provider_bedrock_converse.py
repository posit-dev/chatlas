import binascii
import json
import struct

import pytest


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
