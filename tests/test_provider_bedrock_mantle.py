import base64
import json
from datetime import date
from importlib import resources
from typing import Literal
from unittest.mock import patch

import httpx
import httpx2
import pytest
from chatlas import Chat, ChatBedrock, UserTurn
from chatlas._content import ContentDocument, ContentImageRemote, ContentPDF
from chatlas._provider_bedrock import (
    bedrock_api_for_model,
    bedrock_base_url,
    bedrock_models_base_url,
    bedrock_region,
)
from openai.types import Model

from .conftest import (
    assert_data_extraction,
    assert_list_models,
    assert_tools_async,
    assert_tools_parallel,
    assert_tools_sequential,
    assert_tools_simple,
    assert_turns_existing,
    assert_turns_system,
)

BEDROCK_APIS = json.loads(
    resources.files("chatlas").joinpath("data/bedrock_apis.json").read_text("utf-8")
)


class TestBedrockApiTable:
    def test_table_is_a_flat_str_to_api_mapping(self):
        assert isinstance(BEDROCK_APIS, dict)
        assert BEDROCK_APIS
        for model, api in BEDROCK_APIS.items():
            assert isinstance(model, str)
            assert api in ("messages", "responses")

    def test_table_covers_the_models_this_feature_unlocks(self):
        # These exist only on bedrock-mantle; Converse cannot serve them.
        assert BEDROCK_APIS["openai.gpt-5.6-sol"] == "responses"
        assert BEDROCK_APIS["xai.grok-4.3"] == "responses"
        assert BEDROCK_APIS["anthropic.claude-mythos-5"] == "messages"

    def test_table_excludes_models_converse_can_serve(self):
        # gpt-oss is on the runtime endpoint too, so it must fall through to
        # converse rather than being routed to mantle.
        assert "openai.gpt-oss-120b" not in BEDROCK_APIS

    def test_table_keys_carry_no_cross_region_prefix(self):
        for model in BEDROCK_APIS:
            assert not model.startswith(("us.", "eu.", "apac.", "global."))


class TestApiResolution:
    def test_mantle_only_models_resolve_to_their_api(self):
        assert bedrock_api_for_model("openai.gpt-5.6-sol") == "responses"
        assert bedrock_api_for_model("xai.grok-4.3") == "responses"
        assert bedrock_api_for_model("anthropic.claude-mythos-5") == "messages"

    def test_unknown_models_fall_back_to_converse(self):
        # Most models are Converse-only, so it's the safe fallback: an
        # unrecognised id (custom ARN, new inference profile) keeps working.
        assert bedrock_api_for_model("amazon.nova-pro-v1:0") == "converse"
        assert bedrock_api_for_model("some.model-nobody-has-heard-of") == "converse"
        assert bedrock_api_for_model(None) == "converse"
        assert bedrock_api_for_model("") == "converse"

    def test_cross_region_prefix_is_stripped_before_lookup(self):
        assert bedrock_api_for_model("us.openai.gpt-5.6-sol") == "responses"
        assert bedrock_api_for_model("eu.anthropic.claude-mythos-5") == "messages"
        assert bedrock_api_for_model("global.xai.grok-4.3") == "responses"


class TestBaseUrl:
    def test_responses_uses_the_openai_v1_path(self):
        # Mantle serves newer models from /openai/v1 and older open-weight ones
        # from /v1; the two are disjoint, not aliases.
        assert bedrock_base_url("responses", "us-east-1") == (
            "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
        )

    def test_messages_url_omits_v1_because_the_sdk_appends_it(self):
        assert bedrock_base_url("messages", "us-east-1") == (
            "https://bedrock-mantle.us-east-1.api.aws/anthropic"
        )

    def test_models_url_rewrites_the_openai_path(self):
        # Listings live at /v1/models regardless of which host serves them.
        assert bedrock_models_base_url(
            "https://bedrock-mantle.us-east-1.api.aws/openai/v1/"
        ) == ("https://bedrock-mantle.us-east-1.api.aws/v1")

    def test_models_url_leaves_other_paths_alone(self):
        assert bedrock_models_base_url("https://proxy.example/v1") == (
            "https://proxy.example/v1"
        )

    def test_converse_uses_the_runtime_endpoint(self):
        assert bedrock_base_url("converse", "us-west-2") == (
            "https://bedrock-runtime.us-west-2.amazonaws.com"
        )

    def test_generic_endpoint_url_applies_to_converse(self, monkeypatch):
        monkeypatch.delenv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", raising=False)
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://proxy.example")
        assert bedrock_base_url("converse", "us-west-2") == "https://proxy.example"

    def test_generic_endpoint_url_applies_to_mantle(self, monkeypatch):
        monkeypatch.delenv("AWS_ENDPOINT_URL_BEDROCK_MANTLE", raising=False)
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://proxy.example")
        assert bedrock_base_url("messages", "us-west-2") == (
            "https://proxy.example/anthropic"
        )

    def test_service_specific_var_outranks_generic(self, monkeypatch):
        monkeypatch.setenv("AWS_ENDPOINT_URL", "https://generic.example")
        monkeypatch.setenv(
            "AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "https://specific.example"
        )
        assert bedrock_base_url("converse", "us-west-2") == "https://specific.example"


class TestChatBedrockDispatch:
    def test_responses_model_builds_an_openai_backed_provider(self):
        from chatlas._provider_bedrock import BedrockResponsesProvider

        chat = ChatBedrock(model="openai.gpt-5.6-sol", aws_region="us-east-1")
        assert isinstance(chat.provider, BedrockResponsesProvider)
        assert chat.provider.name == "AWS/Bedrock"
        assert chat.provider.model == "openai.gpt-5.6-sol"

    def test_client_points_at_the_openai_v1_mantle_path(self):
        chat = ChatBedrock(model="xai.grok-4.3", aws_region="us-west-2")
        base_url = str(chat.provider._client.base_url)
        assert base_url.rstrip("/") == (
            "https://bedrock-mantle.us-west-2.api.aws/openai/v1"
        )

    def test_cross_region_prefix_is_stripped_from_the_responses_model(self):
        chat = ChatBedrock(model="us.openai.gpt-5.6-sol", aws_region="us-east-1")
        assert chat.provider.model == "openai.gpt-5.6-sol"

    def test_list_models_uses_the_v1_mantle_path(self):
        # Mantle serves model listings at /v1/models; /openai/v1/models 404s,
        # so list_models() must hit a different base URL than chat requests do.
        chat = ChatBedrock(model="xai.grok-4.3", aws_region="us-west-2")
        assert chat.provider._models_base_url == (
            "https://bedrock-mantle.us-west-2.api.aws/v1"
        )

    def test_list_models_follows_a_custom_base_url(self):
        # Pointing `base_url` at a proxy or private endpoint must not be
        # silently bypassed when listing models.
        chat = ChatBedrock(
            model="openai.gpt-5.6-sol",
            aws_region="us-east-1",
            base_url="https://mantle.internal.example/openai/v1",
        )
        assert chat.provider._models_base_url == "https://mantle.internal.example/v1"

    def test_list_models_leaves_a_v1_base_url_alone(self):
        chat = ChatBedrock(
            model="openai.gpt-oss-120b",
            api="responses",
            aws_region="us-east-1",
            base_url="https://bedrock-mantle.us-east-1.api.aws/v1",
        )
        assert chat.provider._models_base_url == (
            "https://bedrock-mantle.us-east-1.api.aws/v1"
        )

    def test_explicit_api_overrides_model_based_detection(self):
        from chatlas._provider_bedrock import BedrockResponsesProvider

        chat = ChatBedrock(
            model="some.unlisted-model", api="responses", aws_region="us-east-1"
        )
        assert isinstance(chat.provider, BedrockResponsesProvider)

    def test_explicit_base_url_is_respected(self):
        # The documented escape hatch for mantle's other OpenAI path, which
        # serves older open-weight models like gpt-oss.
        chat = ChatBedrock(
            model="openai.gpt-oss-120b",
            api="responses",
            base_url="https://bedrock-mantle.us-east-1.api.aws/v1",
            aws_region="us-east-1",
        )
        base_url = str(chat.provider._client.base_url)
        assert base_url.rstrip("/") == "https://bedrock-mantle.us-east-1.api.aws/v1"

    def test_invalid_api_is_rejected(self):
        with pytest.raises(ValueError, match="api"):
            ChatBedrock(
                model="openai.gpt-5.6-sol",
                api="completions",  # type: ignore[arg-type]
                aws_region="us-east-1",
            )

    def test_max_tokens_is_rejected_for_the_responses_api(self):
        with pytest.raises(ValueError, match="max_tokens"):
            ChatBedrock(
                model="openai.gpt-5.6-sol", max_tokens=100, aws_region="us-east-1"
            )
        # test_responses_model_builds_an_openai_backed_provider (above) already
        # covers that a bare ChatBedrock() -- i.e. max_tokens left at its
        # MISSING sentinel default -- doesn't trip this check.


class TestMessagesProvider:
    def test_messages_model_builds_an_anthropic_backed_provider(self):
        from chatlas._provider_bedrock import BedrockMessagesProvider

        chat = ChatBedrock(model="anthropic.claude-mythos-5", aws_region="us-east-1")
        assert isinstance(chat.provider, BedrockMessagesProvider)
        assert chat.provider.name == "AWS/Bedrock"

    def test_client_points_at_the_anthropic_mantle_path(self):
        chat = ChatBedrock(
            model="anthropic.claude-haiku-4-5", api="messages", aws_region="us-east-1"
        )
        base_url = str(chat.provider._client.base_url)
        assert base_url.rstrip("/") == (
            "https://bedrock-mantle.us-east-1.api.aws/anthropic"
        )

    def test_default_model_is_served_by_mantle(self):
        from chatlas._provider_bedrock import DEFAULT_MODEL

        chat = ChatBedrock(api="messages", aws_region="us-east-1")
        assert chat.provider.model == "anthropic.claude-sonnet-5"
        assert DEFAULT_MODEL == "us.anthropic.claude-sonnet-5"

    def test_cross_region_prefix_is_stripped_from_the_messages_model(self):
        chat = ChatBedrock(model="us.anthropic.claude-mythos-5", aws_region="us-east-1")
        assert chat.provider.model == "anthropic.claude-mythos-5"

    def test_cache_auto_becomes_a_5m_ttl(self):
        chat = ChatBedrock(
            model="anthropic.claude-haiku-4-5", api="messages", aws_region="us-east-1"
        )
        assert chat.provider._cache == "5m"
        assert chat.provider._cache_control() == {"type": "ephemeral", "ttl": "5m"}

    def test_cache_none_disables_caching(self):
        chat = ChatBedrock(
            model="anthropic.claude-haiku-4-5",
            api="messages",
            cache="none",
            aws_region="us-east-1",
        )
        assert chat.provider._cache == "none"
        assert chat.provider._cache_control() is None

    def test_cache_is_rejected_for_the_responses_api(self):
        with pytest.raises(ValueError, match="cache"):
            ChatBedrock(model="openai.gpt-5.6-sol", cache="5m", aws_region="us-east-1")

    def test_default_max_tokens_is_4096(self):
        chat = ChatBedrock(
            model="anthropic.claude-haiku-4-5", api="messages", aws_region="us-east-1"
        )
        assert chat.provider._max_tokens == 4096

    def test_explicit_max_tokens_is_honored(self):
        chat = ChatBedrock(
            model="anthropic.claude-haiku-4-5",
            api="messages",
            max_tokens=100,
            aws_region="us-east-1",
        )
        assert chat.provider._max_tokens == 100

    def test_list_models_uses_mantles_combined_v1_path(self):
        chat = ChatBedrock(
            model="anthropic.claude-haiku-4-5", api="messages", aws_region="us-east-1"
        )
        assert str(chat.provider._models_client.base_url).rstrip("/") == (
            "https://bedrock-mantle.us-east-1.api.aws/v1"
        )

    def test_list_models_rewrites_a_custom_anthropic_path(self):
        chat = ChatBedrock(
            model="anthropic.claude-haiku-4-5",
            api="messages",
            aws_region="us-east-1",
            base_url="https://mantle.internal.example/anthropic",
        )
        assert str(chat.provider._models_client.base_url).rstrip("/") == (
            "https://mantle.internal.example/v1"
        )

    def test_list_models_returns_the_combined_listing(self, monkeypatch):
        chat = ChatBedrock(
            model="anthropic.claude-haiku-4-5", api="messages", aws_region="us-east-1"
        )
        monkeypatch.setattr(
            chat.provider._models_client.models,
            "list",
            lambda: [
                Model(
                    id="anthropic.claude-mythos-5",
                    created=43_200,
                    object="model",
                    owned_by="AWS",
                )
            ],
        )

        assert chat.list_models() == [
            {
                "id": "anthropic.claude-mythos-5",
                "owned_by": "AWS",
                "input": None,
                "output": None,
                "cached_input": None,
                "created_at": date(1970, 1, 1),
            }
        ]


class TestMessagesStructuredOutputMode:
    """`structured_output_mode` on the mantle Messages API: native
    `output_config` vs forced tool call, with "auto" probing per model.

    Mantle's `output_config` support can't be derived from the model ID (and
    differs from both the Anthropic API and Converse), so "auto" mirrors the
    Converse provider's probe semantics: an optimistic first native attempt,
    then a remembered fallback to the forced tool call on the specific 400.
    """

    @staticmethod
    def provider(
        model: str = "anthropic.claude-haiku-4-5",
        mode: Literal["auto", "native", "tool"] = "auto",
    ):
        from chatlas._provider_bedrock import BedrockMessagesProvider

        return BedrockMessagesProvider(
            model=model,
            aws_profile=None,
            aws_region="us-east-1",
            base_url=None,
            structured_output_mode=mode,
        )

    @staticmethod
    def _person():
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        return Person

    @staticmethod
    def _fake_message():
        from anthropic.types import Message, TextBlock, Usage

        return Message.construct(
            id="msg_1",
            type="message",
            role="assistant",
            model="anthropic.claude-haiku-4-5",
            stop_reason="end_turn",
            stop_sequence=None,
            content=[TextBlock(type="text", text='{"name": "John", "age": 15}')],
            usage=Usage.construct(input_tokens=1, output_tokens=1),
        )

    @staticmethod
    def _bad_request(message: str):
        from anthropic import BadRequestError

        request = httpx2.Request(
            "POST", "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"
        )
        return BadRequestError(
            message,
            response=httpx2.Response(400, request=request),
            body=None,
        )

    @classmethod
    def _native_rejection(cls):
        # The specific 400 mantle returns for models without `output_config`
        # support (verified live: haiku-4-5, opus-4-8).
        return cls._bad_request(
            "Error code: 400 - {'type': 'error', 'error': "
            "{'type': 'invalid_request_error', 'message': "
            "'output_config.format: Extra inputs are not permitted'}}"
        )

    def test_chat_bedrock_threads_the_mode_to_the_messages_provider(self):
        chat = ChatBedrock(
            model="anthropic.claude-haiku-4-5",
            api="messages",
            structured_output_mode="tool",
            aws_region="us-east-1",
        )
        assert chat.provider._structured_output_mode == "tool"

    def test_non_auto_mode_is_still_rejected_for_the_responses_api(self):
        with pytest.raises(ValueError, match="structured_output_mode"):
            ChatBedrock(
                model="openai.gpt-5.6-sol",
                structured_output_mode="native",
                aws_region="us-east-1",
            )

    def test_can_stream_matrix(self):
        Person = self._person()

        # No data_model always streams.
        assert self.provider().can_stream(None) is True
        # "auto" on a regex-supported model probes first: it can't stream
        # until a native request has succeeded ...
        unverified = self.provider()
        assert unverified.can_stream(Person) is False
        # ... and streams natively once verified.
        unverified._native_verified_models.add(unverified.model)
        assert unverified.can_stream(Person) is True
        # Models the regex doesn't match never probe ...
        assert (
            self.provider(model="anthropic.claude-sonnet-4").can_stream(Person) is False
        )
        # ... nor do models that already rejected `output_config`.
        rejected = self.provider()
        rejected._native_unsupported_models.add(rejected.model)
        assert rejected.can_stream(Person) is False
        # "native" forces streaming even on unsupported models ...
        assert (
            self.provider(model="anthropic.claude-sonnet-4", mode="native").can_stream(
                Person
            )
            is True
        )
        # ... and "tool" never streams with a data_model.
        assert self.provider(mode="tool").can_stream(Person) is False

    def _perform_with_fake_create(self, provider, data_model, kwargs=None):
        """Call `chat_perform` with a mocked client; return the captured calls."""
        calls: list[dict] = []

        def fake_create(**kw):
            calls.append(kw)
            return self._fake_message()

        with patch.object(provider._client.messages, "create", side_effect=fake_create):
            provider.chat_perform(
                stream=False,
                turns=[],
                tools={},
                data_model=data_model,
                kwargs=kwargs,
            )
        return calls

    def test_auto_mode_probes_with_output_config(self):
        (request,) = self._perform_with_fake_create(self.provider(), self._person())
        output_config = request["output_config"]
        assert output_config["format"]["type"] == "json_schema"
        assert "tool_choice" not in request

    def test_tool_mode_forces_the_structured_tool_call(self):
        (request,) = self._perform_with_fake_create(
            self.provider(mode="tool"), self._person()
        )
        assert "output_config" not in request
        assert request["tool_choice"] == {
            "type": "tool",
            "name": "_structured_tool_call",
        }

    def test_auto_mode_on_unmatched_model_uses_the_tool_call(self):
        (request,) = self._perform_with_fake_create(
            self.provider(model="anthropic.claude-sonnet-4"), self._person()
        )
        assert "output_config" not in request
        assert request["tool_choice"]["name"] == "_structured_tool_call"

    def test_successful_native_request_marks_the_model_verified(self):
        provider = self.provider()
        self._perform_with_fake_create(provider, self._person())
        assert provider.model in provider._native_verified_models

    def test_requests_without_native_format_dont_mark_verified(self):
        provider = self.provider()
        # No data_model: no `output_config.format`, nothing proven.
        self._perform_with_fake_create(provider, None)
        assert provider.model not in provider._native_verified_models
        # An effort-only `output_config` (from `reasoning`) proves nothing
        # about `format` support either.
        self._perform_with_fake_create(
            provider,
            None,
            kwargs={
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            },
        )
        assert provider.model not in provider._native_verified_models

    def test_native_rejection_retries_with_the_forced_tool_call(self):
        provider = self.provider()
        calls: list[dict] = []

        def fake_create(**kw):
            calls.append(kw)
            if len(calls) == 1:
                raise self._native_rejection()
            return self._fake_message()

        with (
            patch.object(provider._client.messages, "create", side_effect=fake_create),
            pytest.warns(UserWarning, match="rejected native structured outputs"),
        ):
            provider.chat_perform(
                stream=False, turns=[], tools={}, data_model=self._person()
            )

        assert len(calls) == 2
        # The retry carries the forced tool call, not `output_config.format`.
        retry = calls[1]
        assert "format" not in retry.get("output_config", {})
        assert retry["tool_choice"] == {
            "type": "tool",
            "name": "_structured_tool_call",
        }
        # The model is remembered as unsupported.
        assert provider.model in provider._native_unsupported_models
        assert provider.model not in provider._native_verified_models

    def test_explicit_native_mode_propagates_the_rejection(self):
        from anthropic import BadRequestError

        provider = self.provider(mode="native")
        with (
            patch.object(
                provider._client.messages,
                "create",
                side_effect=self._native_rejection(),
            ),
            pytest.raises(BadRequestError),
        ):
            provider.chat_perform(
                stream=False, turns=[], tools={}, data_model=self._person()
            )
        assert provider.model not in provider._native_unsupported_models

    def test_other_output_config_errors_propagate(self):
        # A schema-validation 400 implies the model *does* support native
        # output: it must surface, not disable native mode.
        from anthropic import BadRequestError

        provider = self.provider()
        error = self._bad_request(
            "Error code: 400 - {'type': 'error', 'error': "
            "{'type': 'invalid_request_error', 'message': "
            "'output_config.format.schema: invalid JSON schema'}}"
        )
        with (
            patch.object(provider._client.messages, "create", side_effect=error),
            pytest.raises(BadRequestError),
        ):
            provider.chat_perform(
                stream=False, turns=[], tools={}, data_model=self._person()
            )
        assert provider.model not in provider._native_unsupported_models

    def test_in_flight_native_request_still_retries_after_marking(self):
        # A concurrent request can mark the model unsupported while another
        # native request is in flight; that request's own 400 must still
        # fall back (eligibility comes from its args, not the cache).
        from chatlas import UserTurn

        provider = self.provider()
        Person = self._person()
        turns = [UserTurn("John, age 15")]
        args = provider._chat_perform_args(
            stream=False, turns=turns, tools={}, data_model=Person
        )
        assert "format" in args["output_config"]

        provider._native_unsupported_models.add(provider.model)
        retry = provider._native_retry_args(
            self._native_rejection(), args, False, turns, {}, Person, None
        )
        assert retry is not None
        assert "format" not in retry.get("output_config", {})
        assert retry["tool_choice"] == {
            "type": "tool",
            "name": "_structured_tool_call",
        }

    @pytest.mark.asyncio
    async def test_native_rejection_retries_async(self):
        provider = self.provider()
        calls: list[dict] = []

        async def fake_create(**kw):
            calls.append(kw)
            if len(calls) == 1:
                raise self._native_rejection()
            return self._fake_message()

        with pytest.warns(UserWarning, match="rejected native structured outputs"):
            with patch.object(
                provider._async_client.messages, "create", side_effect=fake_create
            ):
                await provider.chat_perform_async(
                    stream=False, turns=[], tools={}, data_model=self._person()
                )

        assert len(calls) == 2
        assert "format" not in calls[1].get("output_config", {})
        assert calls[1]["tool_choice"]["name"] == "_structured_tool_call"

    def test_chat_structured_end_to_end(self):
        provider = self.provider()
        chat = Chat(provider=provider)
        with patch.object(
            provider._client.messages,
            "create",
            side_effect=lambda **kw: self._fake_message(),
        ):
            person = chat.chat_structured("John, age 15", data_model=self._person())
        assert person.name == "John"
        assert person.age == 15


class TestUrlContentFallsBackToBytes:
    """Mantle can't fetch URLs, so PDF/document content built with a URL must
    be sent as bytes instead (https://github.com/posit-dev/chatlas/issues/410).
    """

    def test_messages_pdf_url_downloads_bytes_instead(self, monkeypatch):
        from chatlas._provider_bedrock import BedrockMessagesProvider

        monkeypatch.setattr(
            "chatlas._content_file.download_bytes", lambda url: b"%PDF-1.4 fake"
        )
        c = ContentPDF(filename="a.pdf", url="https://example.com/a.pdf")

        block = BedrockMessagesProvider._as_content_block(c)

        assert block["source"] == {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(b"%PDF-1.4 fake").decode("utf-8"),
        }
        # The downloaded bytes are cached back onto the original content.
        assert c.data == b"%PDF-1.4 fake"

    def test_messages_pdf_with_data_ignores_url_without_downloading(self, monkeypatch):
        from chatlas._provider_bedrock import BedrockMessagesProvider

        def fail_download(url):
            raise AssertionError("shouldn't download when data is already present")

        monkeypatch.setattr("chatlas._content_file.download_bytes", fail_download)
        c = ContentPDF(
            data=b"%PDF-1.4", filename="a.pdf", url="https://example.com/a.pdf"
        )

        block = BedrockMessagesProvider._as_content_block(c)

        assert block["source"]["type"] == "base64"
        assert block["source"]["data"] == base64.b64encode(b"%PDF-1.4").decode("utf-8")

    def test_responses_pdf_and_document_urls_download_bytes_instead(self, monkeypatch):
        monkeypatch.setattr(
            "chatlas._content_file.download_bytes", lambda url: b"downloaded"
        )
        chat = ChatBedrock(model="openai.gpt-5.6-sol", aws_region="us-east-1")
        pdf = ContentPDF(filename="a.pdf", url="https://example.com/a.pdf")
        doc = ContentDocument(
            filename="notes.txt",
            mime_type="text/plain",
            url="https://example.com/notes.txt",
        )
        turn = UserTurn([pdf, doc])

        inputs = chat.provider._turns_as_inputs([turn])

        expected_data = (
            f"data:application/pdf;base64,"
            f"{base64.b64encode(b'downloaded').decode('utf-8')}"
        )
        for item in inputs:
            part = item["content"][0]
            assert part["type"] == "input_file"
            assert "file_url" not in part
            assert part["file_data"].startswith("data:")
        assert inputs[0]["content"][0]["file_data"] == expected_data
        # Downloaded bytes are cached back onto the original content objects.
        assert pdf.data == b"downloaded"
        assert doc.data == b"downloaded"

    def test_responses_pdf_with_data_ignores_url_without_downloading(self, monkeypatch):
        def fail_download(url):
            raise AssertionError("shouldn't download when data is already present")

        monkeypatch.setattr("chatlas._content_file.download_bytes", fail_download)
        chat = ChatBedrock(model="openai.gpt-5.6-sol", aws_region="us-east-1")
        pdf = ContentPDF(
            data=b"%PDF-1.4", filename="a.pdf", url="https://example.com/a.pdf"
        )
        turn = UserTurn([pdf])

        inputs = chat.provider._turns_as_inputs([turn])

        part = inputs[0]["content"][0]
        assert part["type"] == "input_file"
        assert "file_url" not in part
        assert part["file_data"] == (
            f"data:application/pdf;base64,{base64.b64encode(b'%PDF-1.4').decode('utf-8')}"
        )

    def test_messages_remote_image_raises(self):
        from chatlas._provider_bedrock import BedrockMessagesProvider

        with pytest.raises(ValueError, match="Remote images aren't supported"):
            BedrockMessagesProvider._as_content_block(
                ContentImageRemote(url="https://example.com/i.png")
            )

    def test_responses_remote_image_raises(self):
        chat = ChatBedrock(model="openai.gpt-5.6-sol", aws_region="us-east-1")
        turn = UserTurn([ContentImageRemote(url="https://example.com/i.png")])

        with pytest.raises(ValueError, match="Remote images aren't supported"):
            chat.provider._turns_as_inputs([turn])


class TestNativeSdkClients:
    def test_responses_default_clients_stabilize_the_connection_header(self):
        chat = ChatBedrock(model="openai.gpt-5.6-sol", aws_region="us-east-1")
        assert chat.provider._client._client.headers["connection"] == ""
        assert chat.provider._async_client._client.headers["connection"] == ""

    def test_responses_provider_uses_the_sdk_bedrock_client(self):
        from openai import AsyncBedrockOpenAI, BedrockOpenAI

        chat = ChatBedrock(model="openai.gpt-5.6-sol", aws_region="us-east-1")
        assert isinstance(chat.provider._client, BedrockOpenAI)
        assert isinstance(chat.provider._async_client, AsyncBedrockOpenAI)

    def test_responses_client_resolves_the_requested_region(self):
        chat = ChatBedrock(model="openai.gpt-5.6-sol", aws_region="us-west-2")
        assert chat.provider._client.aws_region == "us-west-2"

    def test_responses_provider_accepts_a_custom_http_client(self):
        # Auth lives at the SDK layer now, not the httpx transport, so a
        # user-supplied http_client no longer conflicts with signing.
        http_client = httpx.Client()
        chat = ChatBedrock(
            model="openai.gpt-5.6-sol",
            aws_region="us-east-1",
            kwargs={"http_client": http_client},
        )
        assert chat.provider._client._client is http_client

    def test_messages_provider_uses_the_sdk_mantle_client(self):
        from anthropic.lib.bedrock import (
            AnthropicBedrockMantle,
            AsyncAnthropicBedrockMantle,
        )

        chat = ChatBedrock(model="anthropic.claude-mythos-5", aws_region="us-east-1")
        assert isinstance(chat.provider._client, AnthropicBedrockMantle)
        assert isinstance(chat.provider._async_client, AsyncAnthropicBedrockMantle)

    def test_messages_client_uses_sigv4_when_no_bearer_token_is_set(self):
        chat = ChatBedrock(
            model="anthropic.claude-haiku-4-5", api="messages", aws_region="us-east-1"
        )
        assert chat.provider._client._use_sigv4 is True

    def test_messages_provider_accepts_a_custom_http_client(self):
        # The messages API is served by the anthropic SDK, which requires httpx2
        http_client = httpx2.Client()
        chat = ChatBedrock(
            model="anthropic.claude-haiku-4-5",
            api="messages",
            aws_region="us-east-1",
            kwargs={"http_client": http_client},
        )
        assert chat.provider._client._client is http_client


class TestCredentialResolutionFailsFast:
    def test_construction_raises_when_credentials_cannot_be_frozen(self, monkeypatch):
        class BrokenCredentials:
            def get_frozen_credentials(self):
                raise RuntimeError("token has expired and refresh failed")

        class FakeSession:
            def get_credentials(self):
                return BrokenCredentials()

        monkeypatch.setattr(
            "chatlas._provider_bedrock.botocore_session",
            lambda profile: FakeSession(),
        )

        with pytest.raises(RuntimeError, match="token has expired"):
            ChatBedrock(model="openai.gpt-5.6-sol", aws_region="us-east-1")


class TestBearerTokenAuth:
    def _no_credentials_session(self, monkeypatch):
        class NoCredentialsSession:
            def get_credentials(self):
                return None

        monkeypatch.setattr(
            "chatlas._provider_bedrock.botocore_session",
            lambda profile: NoCredentialsSession(),
        )

    def test_bearer_env_var_skips_sigv4_chain_validation(self, monkeypatch):
        self._no_credentials_session(monkeypatch)
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key")
        # aws_region passed explicitly so region resolution needs no botocore.
        chat = ChatBedrock(model="openai.gpt-5.6-sol", aws_region="us-east-1")
        assert chat.provider._client.api_key == "bedrock-api-key"

    def test_anthropic_bearer_env_var_also_skips_validation(self, monkeypatch):
        self._no_credentials_session(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_AWS_API_KEY", "anthropic-bedrock-key")
        chat = ChatBedrock(model="anthropic.claude-mythos-5", aws_region="us-east-1")
        assert chat.provider._client._use_sigv4 is False
        assert chat.provider._models_client.api_key == "anthropic-bedrock-key"

    def test_api_key_kwarg_skips_validation(self, monkeypatch):
        self._no_credentials_session(monkeypatch)
        chat = ChatBedrock(
            model="openai.gpt-5.6-sol",
            aws_region="us-east-1",
            kwargs={"api_key": "explicit-bedrock-key"},
        )
        assert chat.provider._client.api_key == "explicit-bedrock-key"

    def test_an_explicit_profile_outranks_a_bearer_token(self, monkeypatch):
        # Both SDKs treat an explicit `aws_profile` as a request for SigV4 and
        # ignore the bearer token env vars, so the chain still has to hold up.
        self._no_credentials_session(monkeypatch)
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bedrock-api-key")
        with pytest.raises(ValueError, match="No AWS credentials"):
            ChatBedrock(
                model="openai.gpt-5.6-sol",
                aws_region="us-east-1",
                aws_profile="some-profile",
            )

    def test_static_credential_kwargs_skip_validation(self, monkeypatch):
        # Credentials handed straight to the SDK never reach botocore's chain,
        # so validating the chain would fail for the wrong reason.
        self._no_credentials_session(monkeypatch)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        chat = ChatBedrock(
            model="anthropic.claude-mythos-5",
            aws_region="us-east-1",
            # The aws_* keys live on the SDK Bedrock clients, not on the plain
            # vendor constructor surface that ChatClientArgs describes.
            kwargs={  # type: ignore[arg-type]
                "aws_access_key": "AKIAEXAMPLE",
                "aws_secret_key": "secret",
            },
        )
        assert chat.provider._client._use_sigv4 is True

    def test_without_a_bearer_token_validation_still_fails_fast(self, monkeypatch):
        self._no_credentials_session(monkeypatch)
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("ANTHROPIC_AWS_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No AWS credentials"):
            ChatBedrock(model="openai.gpt-5.6-sol", aws_region="us-east-1")


# ---------------------------------------------------------------------------
# Live API tests (require Bedrock credentials; VCR can't record SigV4 auth)
# ---------------------------------------------------------------------------

_has_mantle_credentials = True
try:
    _chat = ChatBedrock(model="openai.gpt-5.6-sol")
    _chat.chat("What is 1 + 1?")
except Exception:
    _has_mantle_credentials = False

requires_mantle = pytest.mark.skipif(
    not _has_mantle_credentials,
    reason="Bedrock mantle credentials aren't configured",
)


def chat_responses(**kwargs) -> "Chat":
    return ChatBedrock(model="openai.gpt-5.6-sol", **kwargs)


def chat_messages(**kwargs) -> "Chat":
    return ChatBedrock(model="anthropic.claude-haiku-4-5", api="messages", **kwargs)


@requires_mantle
class TestLiveResponses:
    def test_simple_request(self):
        chat = chat_responses(system_prompt="Be as terse as possible; no punctuation")
        chat.chat("What is 1 + 1?")
        turn = chat.get_last_turn()
        assert turn is not None
        assert "2" in turn.text
        assert turn.finish_reason == "success"

    @pytest.mark.asyncio
    async def test_simple_streaming_request(self):
        chat = chat_responses(system_prompt="Be as terse as possible; no punctuation")
        res = []
        async for chunk in await chat.stream_async("What is 1 + 1?"):
            res.append(chunk)
        assert "2" in "".join(res)
        turn = chat.get_last_turn()
        assert turn is not None
        assert turn.finish_reason == "success"

    def test_respects_turns_interface(self):
        assert_turns_system(chat_responses)
        assert_turns_existing(chat_responses)

    def test_tool_variations(self):
        assert_tools_simple(chat_responses)
        assert_tools_parallel(chat_responses)
        assert_tools_sequential(chat_responses, total_calls=6)

    @pytest.mark.asyncio
    async def test_tool_variations_async(self):
        await assert_tools_async(chat_responses)

    def test_data_extraction(self):
        assert_data_extraction(chat_responses)

    def test_grok_also_works(self):
        chat = ChatBedrock(model="xai.grok-4.3")
        chat.chat("What is 1 + 1? Just the number.")
        turn = chat.get_last_turn()
        assert turn is not None
        assert "2" in turn.text

    def test_list_models(self):
        assert_list_models(chat_responses)


@requires_mantle
class TestLiveMessages:
    def test_simple_request(self):
        chat = chat_messages(system_prompt="Be as terse as possible; no punctuation")
        chat.chat("What is 1 + 1?")
        turn = chat.get_last_turn()
        assert turn is not None
        assert "2" in turn.text

    @pytest.mark.asyncio
    async def test_simple_streaming_request(self):
        chat = chat_messages(system_prompt="Be as terse as possible; no punctuation")
        res = []
        async for chunk in await chat.stream_async("What is 1 + 1?"):
            res.append(chunk)
        assert "2" in "".join(res)

    def test_tool_variations(self):
        assert_tools_simple(chat_messages)
        assert_tools_parallel(chat_messages)

    def test_data_extraction(self):
        assert_data_extraction(chat_messages)

    def test_data_extraction_native(self):
        # claude-opus-5 accepts `output_config` on mantle (verified
        # 2026-10-16); haiku-4-5 and opus-4-8 reject it.
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        chat = ChatBedrock(
            model="anthropic.claude-opus-5",
            api="messages",
            structured_output_mode="native",
        )
        person = chat.chat_structured("John is 15 years old.", data_model=Person)
        assert person.name == "John"
        assert person.age == 15

    def test_stream_data_extraction_native(self):
        # Native mode streams the JSON incrementally (no non-streaming
        # fallback).
        import warnings

        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        chat = ChatBedrock(
            model="anthropic.claude-opus-5",
            api="messages",
            structured_output_mode="native",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            chunks = list(chat.stream("John is 15 years old.", data_model=Person))
        assert len(chunks) > 1
        person = Person.model_validate_json("".join(chunks))
        assert person.name == "John"
        assert person.age == 15

    def test_data_extraction_auto_falls_back_to_the_tool_call(self):
        # haiku-4-5 rejects `output_config` on mantle, so "auto" warns,
        # remembers, and falls back to the forced tool call.
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        chat = chat_messages()
        with pytest.warns(UserWarning, match="rejected native structured outputs"):
            person = chat.chat_structured("John is 15 years old.", data_model=Person)
        assert person.name == "John"
        assert person.age == 15
        provider = chat.provider
        assert provider.model in provider._native_unsupported_models
        # The next request goes straight to the tool call: no second warning.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            person2 = chat.chat_structured("Jane is 16 years old.", data_model=Person)
        assert person2.name == "Jane"

    def test_data_extraction_tool_mode(self):
        from pydantic import BaseModel

        class Person(BaseModel):
            name: str
            age: int

        chat = chat_messages(structured_output_mode="tool")
        person = chat.chat_structured("John is 15 years old.", data_model=Person)
        assert person.name == "John"
        assert person.age == 15


@requires_mantle
def test_openai_v1_path_rejects_gpt_oss_with_a_clear_error():
    # Documents mantle's disjoint OpenAI paths: /openai/v1 and /v1 are not
    # aliases, so gpt-oss needs an explicit base_url.
    chat = ChatBedrock(model="openai.gpt-oss-120b", api="responses")
    with pytest.raises(Exception, match="does not support"):
        chat.chat("Hi")


@requires_mantle
def test_gpt_oss_works_via_the_v1_base_url_escape_hatch():
    region = bedrock_region(None, None)
    chat = ChatBedrock(
        model="openai.gpt-oss-120b",
        api="responses",
        base_url=f"https://bedrock-mantle.{region}.api.aws/v1",
    )
    chat.chat("What is 1 + 1? Just the number.")
    turn = chat.get_last_turn()
    assert turn is not None
    assert "2" in turn.text


class TestChatAutoRegistration:
    def test_bedrock_is_selectable_via_chat_auto(self):
        from chatlas import ChatAuto
        from chatlas._provider_bedrock import BedrockResponsesProvider

        chat = ChatAuto(
            provider_model="bedrock/openai.gpt-5.6-sol", aws_region="us-east-1"
        )
        assert isinstance(chat.provider, BedrockResponsesProvider)
