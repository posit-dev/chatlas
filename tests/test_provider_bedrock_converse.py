import binascii
import json
import struct
from typing import TYPE_CHECKING, AsyncIterator, Literal, cast

import boto3
import httpx
import pytest
from botocore.credentials import Credentials
from botocore.exceptions import (
    CredentialRetrievalError,
    InvalidConfigError,
    NoCredentialsError,
    PartialCredentialsError,
    ProfileNotFound,
    RefreshWithMFAUnsupportedError,
    SSOTokenLoadError,
    TokenRetrievalError,
    UnauthorizedSSOTokenError,
)
from chatlas import ChatBedrock
from chatlas._content import ContentThinking
from chatlas._provider_bedrock_converse import (
    BedrockConverseProvider,
    ConverseSubmitArgs,
    as_converse_content,
    content_from_converse_block,
    decode_eventstream,
    decode_eventstream_async,
)
from chatlas._turn import UserTurn

from .conftest import (
    assert_data_extraction,
    assert_images_inline,
    assert_images_remote_error,
    assert_list_models,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_turns_existing,
    assert_turns_system,
)

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.type_defs import (
        ContentBlockOutputTypeDef,
        ConverseRequestTypeDef,
        ConverseResponseTypeDef,
    )


def eventstream_frame(
    payload: bytes,
    event_type: str,
    *,
    message_type: str = "event",
) -> bytes:
    """Encode one AWS eventstream frame the way bedrock-runtime does."""
    headers = b""
    type_header = ":event-type" if message_type == "event" else ":exception-type"
    for name, value in (
        (type_header, event_type),
        (":content-type", "application/json"),
        (":message-type", message_type),
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

    def test_sync_exception_frame_raises(self):
        frame = eventstream_frame(
            json.dumps(
                {
                    "__type": "ThrottlingException",
                    "message": "Too many requests",
                }
            ).encode(),
            "throttlingException",
            message_type="exception",
        )

        with pytest.raises(
            ValueError,
            match="throttlingException: Too many requests",
        ):
            list(decode_eventstream(iter([frame])))

    @pytest.mark.asyncio
    async def test_async_exception_frame_raises(self):
        frame = eventstream_frame(
            json.dumps(
                {
                    "__type": "ValidationException",
                    "message": "Invalid request",
                }
            ).encode(),
            "validationException",
            message_type="exception",
        )

        async def chunks() -> AsyncIterator[bytes]:
            yield frame

        with pytest.raises(
            ValueError,
            match="validationException: Invalid request",
        ):
            _ = [event async for event in decode_eventstream_async(chunks())]


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

    def test_unsupported_image_content_type_raises(self):
        from chatlas._content import ContentImageInline
        from chatlas._provider_bedrock_converse import as_converse_content

        # Bypass the Literal-typed field validation to simulate a MIME type
        # Converse can't map, e.g. one that slipped in via `model_construct`
        # or a future ImageContentTypes addition this dispatch doesn't know.
        content = ContentImageInline.model_construct(
            image_content_type="image/tiff", data=""
        )
        with pytest.raises(ValueError, match="image/tiff"):
            as_converse_content(content)

    def test_remote_image_raises(self):
        from chatlas._content import ContentImageRemote
        from chatlas._provider_bedrock_converse import as_converse_content

        content = ContentImageRemote(url="https://example.com/cat.png")
        with pytest.raises(ValueError, match="Remote images aren't supported"):
            as_converse_content(content)

    def test_json_becomes_a_text_block(self):
        from chatlas._content import ContentJson
        from chatlas._provider_bedrock_converse import as_converse_content

        content = ContentJson(value={"a": 1})
        assert as_converse_content(content) == {"text": '{"a":1}'}

    def test_pdf_becomes_a_document_block(self):
        from chatlas._content import ContentPDF
        from chatlas._provider_bedrock_converse import as_converse_content

        content = ContentPDF(data=b"%PDF-1.4 ...", filename="report.pdf")
        assert as_converse_content(content, document_index=3) == {
            "document": {
                "format": "pdf",
                "name": "document-3",
                "source": {"bytes": b"%PDF-1.4 ..."},
            }
        }

    def test_multiple_pdfs_get_request_unique_names(self):
        from chatlas._content import ContentPDF
        from chatlas._provider_bedrock_converse import as_converse_messages
        from chatlas._turn import AssistantTurn, UserTurn

        turns = [
            UserTurn([ContentPDF(data=b"doc-one", filename="one.pdf")]),
            AssistantTurn(["ok"]),
            UserTurn([ContentPDF(data=b"doc-two", filename="two.pdf")]),
        ]
        # Turn 2's non-PDF content also consumes a counter slot, so the two
        # documents land on "document-0" and "document-2" -- if this were
        # a per-turn counter instead, turn 3's PDF would collide on
        # "document-0" too.
        messages = as_converse_messages(turns)
        names = [
            cast(dict, block)["document"]["name"]
            for message in messages
            for block in cast(list, message["content"])
            if "document" in block
        ]
        # Names are unique across the whole request, not reset per turn --
        # a regression to per-turn numbering would produce "document-0" twice.
        assert names == ["document-0", "document-2"]

    def test_thinking_becomes_a_reasoning_content_block(self):
        from chatlas._content import ContentThinking
        from chatlas._provider_bedrock_converse import as_converse_content

        content = ContentThinking(
            thinking="let me think", extra={"signature": "sig-123"}
        )
        assert as_converse_content(content) == {
            "reasoningContent": {
                "reasoningText": {"text": "let me think", "signature": "sig-123"}
            }
        }

    def test_non_string_tool_result_value_is_stringified(self):
        from chatlas._content import ContentToolRequest, ContentToolResult
        from chatlas._provider_bedrock_converse import as_converse_content

        req = ContentToolRequest(id="t1", name="get_weather", arguments={})
        result = ContentToolResult(
            value={"temp": 72}, model_format="as_is", request=req
        )
        block = cast(dict, as_converse_content(result))
        assert block["toolResult"]["content"] == [{"text": "{'temp': 72}"}]

    def test_builtin_tool_raises(self):
        from chatlas._provider_bedrock_converse import as_converse_tools
        from chatlas._tools import ToolBuiltIn

        tool = ToolBuiltIn(name="web_search", definition={"type": "web_search"})
        with pytest.raises(ValueError, match="web_search"):
            as_converse_tools({tool.name: tool})

    def test_unknown_turn_role_raises(self):
        from chatlas._provider_bedrock_converse import as_converse_messages
        from chatlas._turn import Turn

        turn = Turn.model_construct(role="tool", contents=[])
        with pytest.raises(ValueError, match="Unknown role"):
            as_converse_messages([turn])


class TestRequestTransport:
    def binary_request(self) -> "ConverseRequestTypeDef":
        return cast(
            "ConverseRequestTypeDef",
            {
                "modelId": "vendor/model:version",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": {
                                    "format": "png",
                                    "source": {"bytes": b"image-bytes"},
                                }
                            },
                            {
                                "document": {
                                    "format": "pdf",
                                    "name": "document-0",
                                    "source": {"bytes": b"pdf-bytes"},
                                }
                            },
                        ],
                    }
                ],
            },
        )

    def test_sync_request_base64_encodes_binary_content(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=CONVERSE_RESPONSE)

        provider = BedrockConverseProvider(
            model="vendor/model:version",
            aws_profile=None,
            aws_region="us-east-1",
            base_url=None,
        )
        provider._client = httpx.Client(
            base_url="https://bedrock-runtime.example.com",
            transport=httpx.MockTransport(handler),
        )
        args = self.binary_request()

        assert provider._converse(args) == CONVERSE_RESPONSE
        assert requests[0].url.raw_path == b"/model/vendor%2Fmodel:version/converse"
        body = json.loads(requests[0].content)
        content = body["messages"][0]["content"]
        assert content[0]["image"]["source"]["bytes"] == "aW1hZ2UtYnl0ZXM="
        assert content[1]["document"]["source"]["bytes"] == "cGRmLWJ5dGVz"
        messages = args.get("messages")
        assert messages is not None
        image_block = messages[0]["content"][0]
        assert "image" in image_block
        image_source = image_block["image"]["source"]
        assert "bytes" in image_source
        assert image_source["bytes"] == b"image-bytes"

    @pytest.mark.asyncio
    async def test_async_request_base64_encodes_binary_content(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=CONVERSE_RESPONSE)

        provider = BedrockConverseProvider(
            model="vendor/model:version",
            aws_profile=None,
            aws_region="us-east-1",
            base_url=None,
        )
        provider._async_client = httpx.AsyncClient(
            base_url="https://bedrock-runtime.example.com",
            transport=httpx.MockTransport(handler),
        )
        args = self.binary_request()

        assert await provider._converse_async(args) == CONVERSE_RESPONSE
        body = json.loads(requests[0].content)
        content = body["messages"][0]["content"]
        assert content[0]["image"]["source"]["bytes"] == "aW1hZ2UtYnl0ZXM="
        assert content[1]["document"]["source"]["bytes"] == "cGRmLWJ5dGVz"
        messages = args.get("messages")
        assert messages is not None
        document_block = messages[0]["content"][1]
        assert "document" in document_block
        document_source = document_block["document"]["source"]
        assert "bytes" in document_source
        assert document_source["bytes"] == b"pdf-bytes"

    def test_non_streaming_http_error_includes_status_and_message(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={"message": "Capacity is unavailable"},
            )

        provider = BedrockConverseProvider(
            model="test-model",
            aws_profile=None,
            aws_region="us-east-1",
            base_url=None,
        )
        provider._client = httpx.Client(
            base_url="https://bedrock-runtime.example.com",
            transport=httpx.MockTransport(handler),
        )
        args = cast(
            "ConverseRequestTypeDef",
            {"modelId": "test-model", "messages": []},
        )

        with pytest.raises(
            ValueError,
            match="status 429: Capacity is unavailable",
        ):
            provider._converse(args)

    def test_response_decodes_redacted_reasoning_before_replay(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"text": "Thinking was redacted."},
                                {
                                    "reasoningContent": {
                                        "redactedContent": (
                                            "ZW5jcnlwdGVkLXJlYXNvbmluZw=="
                                        )
                                    }
                                },
                            ],
                        }
                    },
                    "stopReason": "end_turn",
                    "usage": {
                        "inputTokens": 1,
                        "outputTokens": 1,
                        "totalTokens": 2,
                    },
                },
            )

        provider = BedrockConverseProvider(
            model="test-model",
            aws_profile=None,
            aws_region="us-east-1",
            base_url=None,
        )
        provider._client = httpx.Client(
            base_url="https://bedrock-runtime.example.com",
            transport=httpx.MockTransport(handler),
        )
        args = cast(
            "ConverseRequestTypeDef",
            {"modelId": "test-model", "messages": []},
        )

        completion = provider._converse(args)
        turn = provider.value_turn(completion, has_data_model=False)
        thinking = next(
            content for content in turn.contents if isinstance(content, ContentThinking)
        )

        message = completion["output"].get("message")
        assert message is not None
        assert message["content"][0] == {"text": "Thinking was redacted."}
        assert thinking.extra == {"redactedContent": b"encrypted-reasoning"}
        assert as_converse_content(thinking) == {
            "reasoningContent": {"redactedContent": b"encrypted-reasoning"}
        }


class TestRequestArguments:
    def test_reusing_kwargs_preserves_inference_config(self):
        provider = BedrockConverseProvider(
            model="test-model",
            aws_profile=None,
            aws_region="us-east-1",
            base_url=None,
        )
        kwargs = {"inferenceConfig": {"temperature": 0.25}}

        first = provider._chat_perform_args(
            False,
            [UserTurn("hello")],
            {},
            kwargs=cast("ConverseSubmitArgs", kwargs),
        )
        second = provider._chat_perform_args(
            False,
            [UserTurn("hello again")],
            {},
            kwargs=cast("ConverseSubmitArgs", kwargs),
        )

        assert kwargs == {"inferenceConfig": {"temperature": 0.25}}
        first_inference_config = first.get("inferenceConfig")
        second_inference_config = second.get("inferenceConfig")
        assert first_inference_config is not None
        assert second_inference_config is not None
        assert first_inference_config == {
            "maxTokens": 4096,
            "temperature": 0.25,
        }
        assert second_inference_config == {
            "maxTokens": 4096,
            "temperature": 0.25,
        }


class TestProviderCapabilities:
    def provider(self) -> BedrockConverseProvider:
        return BedrockConverseProvider(
            model="test-model",
            aws_profile="test-profile",
            aws_region="us-east-1",
            base_url=None,
        )

    def test_token_count_is_unsupported(self):
        with pytest.raises(
            NotImplementedError,
            match="no standalone token-counting endpoint",
        ):
            self.provider().token_count([], tools={}, data_model=None)

    @pytest.mark.asyncio
    async def test_async_token_count_is_unsupported(self):
        with pytest.raises(
            NotImplementedError,
            match="no standalone token-counting endpoint",
        ):
            await self.provider().token_count_async([], tools={}, data_model=None)

    def test_list_models_uses_bedrock_control_plane_without_aws(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        calls: dict[str, object] = {}

        class FakeBedrock:
            def list_foundation_models(self) -> dict[str, list[dict[str, str]]]:
                return {
                    "modelSummaries": [
                        {
                            "modelId": "vendor.test-model",
                            "modelName": "Test Model",
                            "providerName": "Vendor",
                        }
                    ]
                }

        class FakeSession:
            def __init__(self, *, profile_name: str, region_name: str):
                calls["profile_name"] = profile_name
                calls["region_name"] = region_name

            def client(self, service_name: str) -> FakeBedrock:
                calls["service_name"] = service_name
                return FakeBedrock()

        monkeypatch.setattr(boto3, "Session", FakeSession)

        models = self.provider().list_models()

        assert calls == {
            "profile_name": "test-profile",
            "region_name": "us-east-1",
            "service_name": "bedrock",
        }
        assert models == [
            {
                "id": "vendor.test-model",
                "name": "Test Model",
                "provider": "Vendor",
                "input": None,
                "output": None,
                "cached_input": None,
            }
        ]


# `ConverseResponseTypeDef` marks every field required (`modelId` is the only
# NotRequired-free exception on the request side, but the response side has
# none at all), so a literal missing fields like `metrics`/`trace` needs a
# cast rather than a plain annotation -- the same `cast(dict, ...)` pattern
# already used above for the all-NotRequired *Output block TypedDicts.
CONVERSE_RESPONSE = cast(
    "ConverseResponseTypeDef",
    {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "2"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 19,
            "outputTokens": 5,
            "totalTokens": 24,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokens": 0,
        },
    },
)


class TestResponseParsing:
    def provider(self):
        from chatlas._provider_bedrock_converse import BedrockConverseProvider

        return BedrockConverseProvider(
            model="us.anthropic.claude-sonnet-4-6",
            aws_profile=None,
            aws_region="us-east-1",
            base_url=None,
        )

    def test_text_response_becomes_an_assistant_turn(self):
        turn = self.provider().value_turn(CONVERSE_RESPONSE, has_data_model=False)
        assert turn.text == "2"
        assert turn.finish_reason == "success"

    def test_usage_includes_cache_tokens(self):
        turn = self.provider().value_turn(CONVERSE_RESPONSE, has_data_model=False)
        assert turn.tokens is not None
        assert turn.tokens[0] == 19
        assert turn.tokens[1] == 5

    def test_tool_use_response_becomes_a_tool_request(self):
        from chatlas._content import ContentToolRequest

        response = cast(
            "ConverseResponseTypeDef",
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "t1",
                                    "name": "get_weather",
                                    "input": {"city": "Paris"},
                                }
                            }
                        ],
                    }
                },
                "stopReason": "tool_use",
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            },
        )
        turn = self.provider().value_turn(response, has_data_model=False)
        requests = [c for c in turn.contents if isinstance(c, ContentToolRequest)]
        assert len(requests) == 1
        assert requests[0].name == "get_weather"
        assert requests[0].arguments == {"city": "Paris"}

    def test_redacted_reasoning_round_trips_to_the_next_request(self):
        block = cast(
            "ContentBlockOutputTypeDef",
            {"reasoningContent": {"redactedContent": b"encrypted-reasoning"}},
        )

        content = content_from_converse_block(block)

        assert isinstance(content, ContentThinking)
        assert content.thinking == ""
        assert content.extra == {"redactedContent": b"encrypted-reasoning"}
        assert as_converse_content(content) == {
            "reasoningContent": {"redactedContent": b"encrypted-reasoning"}
        }

    @pytest.mark.parametrize(
        "reason,expected",
        [
            ("end_turn", "success"),
            ("tool_use", "tool_use"),
            ("max_tokens", "max_tokens"),
            ("stop_sequence", "stop_sequence"),
            ("content_filtered", "content_filter"),
            ("guardrail_intervened", "content_filter"),
            ("model_context_window_exceeded", "context_window"),
        ],
    )
    def test_stop_reasons_map_to_finish_reasons(self, reason, expected):
        from chatlas._provider_bedrock_converse import converse_stop_reason

        assert converse_stop_reason(reason) == expected


class TestStreamAccumulation:
    def provider(self):
        return BedrockConverseProvider(
            model="test-model",
            aws_profile=None,
            aws_region="us-east-1",
            base_url=None,
        )

    def merge(self, events: list[dict]):
        merged = None
        for event in events:
            merged = self.provider().stream_merge_chunks(merged, event)
        assert merged is not None
        return merged

    def test_text_deltas_are_emitted_in_order(self):
        from chatlas._content import ContentText

        provider = self.provider()
        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hel"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "lo"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
        ]

        merged = None
        emitted = []
        for event in events:
            merged = provider.stream_merge_chunks(merged, event)
            emitted.extend(provider.stream_content(event, merged))

        assert (
            "".join(
                content.text for content in emitted if isinstance(content, ContentText)
            )
            == "Hello"
        )

    def test_tool_use_input_fragments_are_reassembled(self):
        from chatlas._content import ContentToolRequest

        events = [
            {"messageStart": {"role": "assistant"}},
            {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "t1", "name": "get_weather"}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": '{"city":'}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": '"Paris"}'}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "tool_use"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
                }
            },
        ]

        turn = self.provider().stream_turn(self.merge(events), has_data_model=False)
        requests = [
            content
            for content in turn.contents
            if isinstance(content, ContentToolRequest)
        ]

        assert len(requests) == 1
        assert requests[0].name == "get_weather"
        assert requests[0].arguments == {"city": "Paris"}
        assert turn.finish_reason == "tool_use"

    def test_interleaved_block_indices_stay_separate(self):
        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "A"}}},
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"text": "B"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "C"}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "end_turn"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
                }
            },
        ]

        turn = self.provider().stream_turn(self.merge(events), has_data_model=False)

        assert turn.text == "ACB"

    def test_invalid_tool_arguments_name_the_tool(self):
        events = [
            {"messageStart": {"role": "assistant"}},
            {
                "contentBlockStart": {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "t1", "name": "get_weather"}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"toolUse": {"input": "not json"}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
        ]

        with pytest.raises(ValueError, match="get_weather"):
            self.merge(events)

    def test_reasoning_content_deltas_are_reassembled(self):
        events = [
            {"messageStart": {"role": "assistant"}},
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"text": "let me "}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"text": "think"}},
                }
            },
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"signature": "sig-123"}},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": {"text": "The answer is 4."},
                }
            },
            {"contentBlockStop": {"contentBlockIndex": 1}},
            {"messageStop": {"stopReason": "end_turn"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
                }
            },
        ]

        turn = self.provider().stream_turn(self.merge(events), has_data_model=False)
        thinking = [
            content for content in turn.contents if isinstance(content, ContentThinking)
        ]

        assert len(thinking) == 1
        assert thinking[0].thinking == "let me think"
        assert thinking[0].extra == {"signature": "sig-123"}
        assert turn.text == "The answer is 4."


class TestConverseDispatch:
    def test_converse_model_builds_a_converse_provider(self):
        from chatlas import ChatBedrock
        from chatlas._provider_bedrock_converse import BedrockConverseProvider

        chat = ChatBedrock(model="amazon.nova-pro-v1:0", aws_region="us-east-1")

        assert isinstance(chat.provider, BedrockConverseProvider)

    def test_default_model_is_ellmers_claude_sonnet(self):
        from chatlas import ChatBedrock
        from chatlas._provider_bedrock_converse import BedrockConverseProvider

        chat = ChatBedrock(aws_region="us-east-1")

        assert chat.provider.model == "us.anthropic.claude-sonnet-4-6"
        assert isinstance(chat.provider, BedrockConverseProvider)

    def test_claude_routes_to_converse_not_mantle(self):
        from chatlas import ChatBedrock
        from chatlas._provider_bedrock_converse import BedrockConverseProvider

        chat = ChatBedrock(model="anthropic.claude-sonnet-5", aws_region="us-east-1")

        assert isinstance(chat.provider, BedrockConverseProvider)

    def test_converse_base_url_is_the_runtime_endpoint(self):
        from chatlas import ChatBedrock
        from chatlas._provider_bedrock_converse import BedrockConverseProvider

        chat = ChatBedrock(model="amazon.nova-pro-v1:0", aws_region="us-west-2")
        provider = cast(BedrockConverseProvider, chat.provider)

        assert "bedrock-runtime.us-west-2.amazonaws.com" in str(
            provider._client.base_url
        )

    def test_explicit_converse_forwards_config_without_validating_credentials(
        self, monkeypatch
    ):
        from chatlas import ChatBedrock
        from chatlas._provider_bedrock_converse import BedrockConverseProvider

        def credentials_must_not_be_resolved(_profile):
            raise AssertionError("Converse construction must not resolve credentials")

        monkeypatch.setattr(
            "chatlas._provider_bedrock.bedrock_credentials",
            credentials_must_not_be_resolved,
        )

        chat = ChatBedrock(
            api="converse",
            model="amazon.nova-pro-v1:0",
            aws_region="us-east-1",
            base_url="https://bedrock.example",
            max_tokens=123,
            cache="none",
        )
        provider = cast(BedrockConverseProvider, chat.provider)

        assert provider._max_tokens == 123
        assert provider._cache == "none"
        assert str(provider._client.base_url).rstrip("/") == "https://bedrock.example"

    def test_api_key_kwarg_authenticates_converse_http_requests(self):
        from chatlas import ChatBedrock
        from chatlas._provider_bedrock_converse import BedrockConverseProvider

        headers: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            headers.append(request.headers["Authorization"])
            return httpx.Response(200, request=request)

        chat = ChatBedrock(
            api="converse",
            model="amazon.nova-pro-v1:0",
            aws_region="us-east-1",
            kwargs={
                "api_key": "explicit-bedrock-key",
                "transport": httpx.MockTransport(handler),
            },
        )
        provider = cast(BedrockConverseProvider, chat.provider)
        provider._client.get("/")

        assert headers == ["Bearer explicit-bedrock-key"]

    def test_bearer_token_env_var_authenticates_converse_http_requests(
        self, monkeypatch
    ):
        from chatlas import ChatBedrock
        from chatlas._provider_bedrock_converse import BedrockConverseProvider

        headers: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            headers.append(request.headers["Authorization"])
            return httpx.Response(200, request=request)

        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-bedrock-key")
        chat = ChatBedrock(
            api="converse",
            model="amazon.nova-pro-v1:0",
            aws_region="us-east-1",
            kwargs={"transport": httpx.MockTransport(handler)},
        )
        provider = cast(BedrockConverseProvider, chat.provider)
        provider._client.get("/")

        assert headers == ["Bearer env-bedrock-key"]

    def test_converse_uses_sigv4_without_a_bearer_token(self, monkeypatch):
        from chatlas import ChatBedrock
        from chatlas._provider_bedrock_converse import (
            BedrockConverseProvider,
            BedrockSigV4Auth,
        )

        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        chat = ChatBedrock(
            api="converse",
            model="amazon.nova-pro-v1:0",
            aws_region="us-east-1",
        )
        provider = cast(BedrockConverseProvider, chat.provider)

        assert isinstance(provider._client.auth, BedrockSigV4Auth)

    def test_explicit_api_key_outranks_profile_and_env_bearer_token(self, monkeypatch):
        headers: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            headers.append(request.headers["Authorization"])
            return httpx.Response(200, request=request)

        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-bedrock-key")
        provider = BedrockConverseProvider(
            model="amazon.nova-pro-v1:0",
            aws_profile="some-profile",
            aws_region="us-east-1",
            base_url=None,
            kwargs={
                "api_key": "explicit-bedrock-key",
                "transport": httpx.MockTransport(handler),
            },
        )
        provider._client.get("/")

        assert headers == ["Bearer explicit-bedrock-key"]

    def test_explicit_profile_uses_sigv4_instead_of_env_bearer_token(self, monkeypatch):
        from chatlas._provider_bedrock_converse import BedrockSigV4Auth

        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-bedrock-key")
        provider = BedrockConverseProvider(
            model="amazon.nova-pro-v1:0",
            aws_profile="some-profile",
            aws_region="us-east-1",
            base_url=None,
        )

        assert isinstance(provider._client.auth, BedrockSigV4Auth)

    def test_sync_event_hooks_are_only_used_by_sync_client(self):
        def hook(_request: httpx.Request) -> None:
            pass

        provider = BedrockConverseProvider(
            model="amazon.nova-pro-v1:0",
            aws_profile=None,
            aws_region="us-east-1",
            base_url=None,
            kwargs={"event_hooks": {"request": [hook]}},
        )

        assert provider._client.event_hooks["request"] == [hook]
        assert provider._async_client.event_hooks["request"] == []

    def test_async_event_hooks_are_only_used_by_async_client(self):
        async def hook(_request: httpx.Request) -> None:
            pass

        provider = BedrockConverseProvider(
            model="amazon.nova-pro-v1:0",
            aws_profile=None,
            aws_region="us-east-1",
            base_url=None,
            kwargs={"event_hooks": {"request": [hook]}},
        )

        assert provider._client.event_hooks["request"] == []
        assert provider._async_client.event_hooks["request"] == [hook]

    def test_sync_transport_is_only_used_by_sync_client(self):
        transport = httpx.HTTPTransport()
        provider = BedrockConverseProvider(
            model="amazon.nova-pro-v1:0",
            aws_profile=None,
            aws_region="us-east-1",
            base_url=None,
            kwargs={"transport": transport, "trust_env": False},
        )

        assert provider._client._transport is transport
        assert isinstance(provider._async_client._transport, httpx.AsyncHTTPTransport)

    def test_async_transport_is_only_used_by_async_client(self):
        transport = httpx.AsyncHTTPTransport()
        provider = BedrockConverseProvider(
            model="amazon.nova-pro-v1:0",
            aws_profile=None,
            aws_region="us-east-1",
            base_url=None,
            kwargs={"transport": transport, "trust_env": False},
        )

        assert isinstance(provider._client._transport, httpx.HTTPTransport)
        assert provider._async_client._transport is transport

    def test_dual_mode_transport_is_used_by_both_clients(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200))
        provider = BedrockConverseProvider(
            model="amazon.nova-pro-v1:0",
            aws_profile=None,
            aws_region="us-east-1",
            base_url=None,
            kwargs={"transport": transport, "trust_env": False},
        )

        assert provider._client._transport is transport
        assert provider._async_client._transport is transport

    def test_sync_mount_is_only_used_by_sync_client(self):
        transport = httpx.HTTPTransport()
        provider = BedrockConverseProvider(
            model="amazon.nova-pro-v1:0",
            aws_profile=None,
            aws_region="us-east-1",
            base_url=None,
            kwargs={
                "mounts": {"https://mounted.example": transport},
                "trust_env": False,
            },
        )
        url = httpx.URL("https://mounted.example")

        assert provider._client._transport_for_url(url) is transport
        assert isinstance(
            provider._async_client._transport_for_url(url), httpx.AsyncHTTPTransport
        )

    def test_async_mount_is_only_used_by_async_client(self):
        transport = httpx.AsyncHTTPTransport()
        provider = BedrockConverseProvider(
            model="amazon.nova-pro-v1:0",
            aws_profile=None,
            aws_region="us-east-1",
            base_url=None,
            kwargs={
                "mounts": {"https://mounted.example": transport},
                "trust_env": False,
            },
        )
        url = httpx.URL("https://mounted.example")

        assert isinstance(provider._client._transport_for_url(url), httpx.HTTPTransport)
        assert provider._async_client._transport_for_url(url) is transport


class TestStructuredOutputAndCaching:
    def provider(
        self,
        *,
        model: str = "us.anthropic.claude-sonnet-4-6",
        cache: Literal["auto", "5m", "1h", "none"] = "auto",
    ):
        return BedrockConverseProvider(
            model=model,
            aws_profile=None,
            aws_region="us-east-1",
            base_url=None,
            cache=cache,
        )

    def test_data_model_forces_a_tool_choice(self):
        from chatlas._tools import Tool
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        def get_weather(city: str) -> str:
            return city

        weather_tool = Tool.from_func(get_weather)
        args = self.provider()._chat_perform_args(
            stream=False,
            turns=[UserTurn("Alice is 30")],
            tools={weather_tool.name: weather_tool},
            data_model=Person,
        )

        tool_config = args.get("toolConfig")
        assert tool_config is not None
        tool_choice = tool_config.get("toolChoice")
        assert tool_choice is not None
        selected_tool = tool_choice.get("tool")
        assert selected_tool is not None
        tool_name = selected_tool["name"]
        tool_names = [
            tool["toolSpec"]["name"]
            for tool in tool_config["tools"]
            if "toolSpec" in tool
        ]

        assert tool_name in tool_names
        assert weather_tool.name in tool_names

    def test_data_model_tool_choice_overrides_request_tool_config(self):
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str

        args = self.provider()._chat_perform_args(
            stream=False,
            turns=[UserTurn("Alice")],
            tools={},
            data_model=Person,
            kwargs=cast("ConverseSubmitArgs", {"toolConfig": {"tools": []}}),
        )

        tool_config = args.get("toolConfig")
        assert tool_config is not None
        tool_choice = tool_config.get("toolChoice")
        assert tool_choice == {"tool": {"name": "_structured_tool_call"}}

    def test_data_model_tool_response_becomes_json_content(self):
        from chatlas._content import ContentJson

        response = cast(
            "ConverseResponseTypeDef",
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "structured-1",
                                    "name": "_structured_tool_call",
                                    "input": {"data": {"name": "Alice", "age": 30}},
                                }
                            }
                        ],
                    }
                },
                "stopReason": "tool_use",
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            },
        )

        turn = self.provider().value_turn(response, has_data_model=True)
        contents = [
            content for content in turn.contents if isinstance(content, ContentJson)
        ]

        assert len(contents) == 1
        assert contents[0].value == {"name": "Alice", "age": 30}

    def test_malformed_structured_tool_input_raises_value_error(self):
        response = cast(
            "ConverseResponseTypeDef",
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "structured-1",
                                    "name": "_structured_tool_call",
                                    "input": [],
                                }
                            }
                        ],
                    }
                },
                "stopReason": "tool_use",
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            },
        )

        with pytest.raises(ValueError, match="data extraction tool"):
            self.provider().value_turn(response, has_data_model=True)

    def test_incomplete_structured_response_raises_the_finish_reason(self):
        response = cast(
            "ConverseResponseTypeDef",
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "structured-1",
                                    "name": "_structured_tool_call",
                                    "input": {},
                                }
                            }
                        ],
                    }
                },
                "stopReason": "max_tokens",
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            },
        )

        with pytest.raises(ValueError, match="max_tokens"):
            self.provider().value_turn(response, has_data_model=True)

    def test_cache_auto_adds_a_cache_point_for_claude(self):
        from chatlas._turn import SystemTurn

        args = self.provider(cache="auto")._chat_perform_args(
            stream=False,
            turns=[SystemTurn("be terse"), UserTurn("hi")],
            tools={},
        )

        self.assert_cache_point(args, expected=True)

    def test_cache_auto_adds_a_cache_point_for_nova(self):
        from chatlas._turn import SystemTurn

        args = self.provider(
            model="us.amazon.nova-pro-v1:0", cache="auto"
        )._chat_perform_args(
            stream=False,
            turns=[SystemTurn("be terse"), UserTurn("hi")],
            tools={},
        )

        self.assert_cache_point(args, expected=True)

    def test_cache_none_adds_no_cache_point(self):
        from chatlas._turn import SystemTurn

        args = self.provider(cache="none")._chat_perform_args(
            stream=False,
            turns=[SystemTurn("be terse"), UserTurn("hi")],
            tools={},
        )

        self.assert_cache_point(args, expected=False)

    def test_cache_auto_skips_the_cache_point_without_system_content(self):
        args = self.provider(cache="auto")._chat_perform_args(
            stream=False,
            turns=[UserTurn("hi")],
            tools={},
        )

        assert "system" not in args

    def test_cache_auto_adds_a_cache_point_to_the_last_message(self):
        from chatlas._turn import SystemTurn

        args = self.provider(cache="auto")._chat_perform_args(
            stream=False,
            turns=[SystemTurn("be terse"), UserTurn("hi"), UserTurn("again")],
            tools={},
        )

        messages = args["messages"]
        assert messages[-1]["content"][-1] == {"cachePoint": {"type": "default"}}
        assert not any("cachePoint" in block for block in messages[0]["content"])

    def test_cache_none_adds_no_cache_point_to_messages(self):
        args = self.provider(cache="none")._chat_perform_args(
            stream=False,
            turns=[UserTurn("hi")],
            tools={},
        )

        content = args["messages"][-1]["content"]
        assert not any("cachePoint" in block for block in content)

    def test_cache_does_not_duplicate_a_message_cache_point(self):
        args = self.provider(cache="auto")._chat_perform_args(
            stream=False,
            turns=[UserTurn("hi")],
            tools={},
            kwargs=cast(
                "ConverseSubmitArgs",
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"text": "hi"},
                                {"cachePoint": {"type": "default"}},
                            ],
                        }
                    ]
                },
            ),
        )

        content = args["messages"][-1]["content"]
        assert content.count({"cachePoint": {"type": "default"}}) == 1

    def test_message_cache_point_does_not_mutate_caller_kwargs(self):
        original_content = [{"text": "hi"}]
        original_message = {"role": "user", "content": original_content}
        kwargs = cast("ConverseSubmitArgs", {"messages": [original_message]})

        self.provider(cache="auto")._chat_perform_args(
            stream=False,
            turns=[UserTurn("hi")],
            tools={},
            kwargs=kwargs,
        )

        assert original_message["content"] == original_content

    def test_cache_point_is_preserved_with_request_system_blocks(self):
        args = self.provider(cache="auto")._chat_perform_args(
            stream=False,
            turns=[UserTurn("hi")],
            tools={},
            kwargs=cast(
                "ConverseSubmitArgs",
                {"system": [{"text": "be terse"}]},
            ),
        )

        self.assert_cache_point(args, expected=True)

    def test_cache_does_not_duplicate_a_system_cache_point(self):
        args = self.provider(cache="auto")._chat_perform_args(
            stream=False,
            turns=[UserTurn("hi")],
            tools={},
            kwargs=cast(
                "ConverseSubmitArgs",
                {
                    "system": [
                        {"text": "be terse"},
                        {"cachePoint": {"type": "default"}},
                    ]
                },
            ),
        )

        system = args.get("system")
        assert system is not None
        assert system.count({"cachePoint": {"type": "default"}}) == 1

    def test_system_request_kwargs_override_turns_without_cache(self):
        from chatlas._turn import SystemTurn

        args = self.provider(cache="none")._chat_perform_args(
            stream=False,
            turns=[SystemTurn("from turn"), UserTurn("hi")],
            tools={},
            kwargs=cast(
                "ConverseSubmitArgs",
                {"system": [{"text": "from kwargs"}]},
            ),
        )

        system = args.get("system")
        assert system == [{"text": "from kwargs"}]

    def test_additional_model_request_fields_reach_the_request(self):
        fields = {"thinking": {"type": "enabled", "budget_tokens": 4000}}
        args = self.provider()._chat_perform_args(
            stream=False,
            turns=[UserTurn("hi")],
            tools={},
            kwargs=cast(
                "ConverseSubmitArgs",
                {"additionalModelRequestFields": fields},
            ),
        )

        assert args["additionalModelRequestFields"] == fields

    @pytest.mark.parametrize(
        "cache,expected_cache_point",
        [
            ("5m", {"type": "default"}),
            ("1h", {"type": "default", "ttl": "1h"}),
        ],
    )
    def test_explicit_cache_ttls_add_a_cache_point(self, cache, expected_cache_point):
        from chatlas._turn import SystemTurn

        args = self.provider(cache=cache)._chat_perform_args(
            stream=False,
            turns=[SystemTurn("be terse"), UserTurn("hi")],
            tools={},
        )

        system = args.get("system")
        assert system is not None
        assert {"cachePoint": expected_cache_point} in system

    def test_cache_auto_is_disabled_for_models_without_support(self):
        from chatlas._turn import SystemTurn

        args = self.provider(
            model="meta.llama3-70b-instruct-v1:0", cache="auto"
        )._chat_perform_args(
            stream=False,
            turns=[SystemTurn("be terse"), UserTurn("hi")],
            tools={},
        )

        self.assert_cache_point(args, expected=False)

    def assert_cache_point(self, args, *, expected: bool):
        system = args.get("system")
        assert system is not None
        cache_point = {"cachePoint": {"type": "default"}}
        assert (cache_point in system) is expected


def test_blocked_network_skips_live_converse_before_credential_probe(monkeypatch):
    class BlockedConfig:
        def getoption(self, option: str) -> bool:
            assert option == "--block-network"
            return True

    class BlockedRequest:
        config = BlockedConfig()

    def unexpected_credential_probe(*args, **kwargs):
        raise AssertionError("ChatBedrock() must not run when network is blocked")

    monkeypatch.setitem(globals(), "ChatBedrock", unexpected_credential_probe)

    with pytest.raises(pytest.skip.Exception, match="network access is blocked"):
        check_live_converse_available(cast(pytest.FixtureRequest, BlockedRequest()))


# ---------------------------------------------------------------------------
# Live API tests (require Bedrock credentials; VCR can't record SigV4 auth)
# ---------------------------------------------------------------------------


def check_live_converse_available(request: pytest.FixtureRequest) -> None:
    if request.config.getoption("--block-network"):
        pytest.skip("Live Converse tests skipped because network access is blocked")

    try:
        ChatBedrock().chat("What is 1 + 1?")
    except (
        CredentialRetrievalError,
        InvalidConfigError,
        NoCredentialsError,
        PartialCredentialsError,
        ProfileNotFound,
        RefreshWithMFAUnsupportedError,
        SSOTokenLoadError,
        TokenRetrievalError,
        UnauthorizedSSOTokenError,
    ) as error:
        pytest.skip(f"Bedrock credentials aren't configured: {error}")
    except ValueError as error:
        if not is_converse_credential_or_access_error(error):
            raise
        pytest.skip(f"Bedrock credentials aren't configured: {error}")


def is_converse_credential_or_access_error(error: ValueError) -> bool:
    message = str(error)
    return message.startswith(
        ("No AWS credentials found.", "No AWS region found.")
    ) or message.startswith(
        (
            "Bedrock Converse request failed with status 401:",
            "Bedrock Converse request failed with status 403:",
        )
    )


@pytest.fixture(scope="class", autouse=True)
def require_live_converse(request: pytest.FixtureRequest) -> None:
    if request.cls is TestLiveConverse:
        check_live_converse_available(request)


class TestLiveConverse:
    def test_simple_request(self):
        chat = ChatBedrock(system_prompt="Be as terse as possible; no punctuation")
        chat.chat("What is 1 + 1?")
        turn = chat.get_last_turn()
        assert turn is not None
        assert "2" in turn.text
        assert turn.finish_reason == "success"
        assert turn.tokens is not None
        assert turn.tokens[0] > 0

    @pytest.mark.asyncio
    async def test_simple_streaming_request(self):
        chat = ChatBedrock(system_prompt="Be as terse as possible; no punctuation")
        res = []
        async for chunk in await chat.stream_async("What is 1 + 1?"):
            res.append(chunk)
        assert "2" in "".join(res)
        turn = chat.get_last_turn()
        assert turn is not None
        assert turn.finish_reason == "success"

    def test_respects_turns_interface(self):
        assert_turns_system(ChatBedrock)
        assert_turns_existing(ChatBedrock)

    def test_tool_variations(self):
        assert_tools_simple(ChatBedrock)
        assert_tools_parallel(ChatBedrock)
        assert_tools_sequential(ChatBedrock, total_calls=6)

    @pytest.mark.asyncio
    async def test_tool_variations_async(self):
        await assert_tools_async(ChatBedrock)

    def test_data_extraction(self):
        assert_data_extraction(ChatBedrock)

    def test_images(self):
        assert_images_inline(ChatBedrock)
        assert_images_remote_error(ChatBedrock)

    def test_list_models(self):
        assert_list_models(ChatBedrock)

    def test_non_claude_models_work(self):
        # The whole point of Converse: models chatlas couldn't reach before.
        for model in [
            "amazon.nova-lite-v1:0",
            "mistral.mistral-large-3-675b-instruct",
        ]:
            chat = ChatBedrock(model=model)
            chat.chat("What is 1 + 1? Just the number.")
            turn = chat.get_last_turn()
            assert turn is not None, f"{model} returned no turn"
            assert "2" in turn.text, f"{model} gave {turn.text!r}"

    def test_prompt_caching_reports_cache_tokens(self):
        chat = ChatBedrock(cache="5m", system_prompt="You are terse. " * 400)
        chat.chat("Hi")
        chat.chat("Hi again")
        turn = chat.get_last_turn()
        assert turn is not None
        assert turn.tokens is not None
        assert turn.tokens[2] > 0
