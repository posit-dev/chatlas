import json
from datetime import date
from importlib import resources

import httpx
import httpx2
import pytest
from chatlas import Chat, ChatBedrock
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
