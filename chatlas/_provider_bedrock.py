from __future__ import annotations

import json
import os
import re
import warnings
from importlib import resources
from typing import TYPE_CHECKING, Any, Literal, Optional, cast, overload

import httpx

from ._chat import Chat
from ._content_file import materialize_url_content
from ._logging import log_model_default
from ._provider import ModelInfo, no_file_management
from ._provider_anthropic import AnthropicProvider, StructuredOutputMode
from ._provider_openai import OpenAIProvider
from ._provider_openai_generic import openai_models_to_info
from ._turn import caller_stacklevel
from ._utils import MISSING, MISSING_TYPE, AnyTypeDict, split_http_client_kwargs

if TYPE_CHECKING:
    from anthropic import BadRequestError
    from botocore.credentials import Credentials
    from botocore.session import Session
    from pydantic import BaseModel

    from ._content import Content
    from ._provider_anthropic import ContentBlockParam
    from ._provider_openai import ResponseInputItemParam
    from ._tools import Tool, ToolBuiltIn
    from ._turn import Role, Turn
    from .types.anthropic import ChatClientArgs as AnthropicClientArgs
    from .types.anthropic import SubmitInputArgs as AnthropicSubmitInputArgs
    from .types.bedrock import ChatClientArgs as ConverseClientArgs
    from .types.openai import ChatClientArgs as OpenAIClientArgs

BedrockAPI = Literal["converse", "messages", "responses"]

DEFAULT_MODEL = "us.anthropic.claude-sonnet-5"

MANTLE_HOST = "https://bedrock-mantle.{region}.api.aws"

# Model ids can carry a cross-region inference prefix, which is stripped before
# looking them up in the table.
CROSS_REGION_PREFIX = re.compile(r"^(us|eu|apac|au|jp|ca|global)\.")


def ChatBedrock(
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    api: Optional[BedrockAPI] = None,
    aws_profile: Optional[str] = None,
    aws_region: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int | MISSING_TYPE = MISSING,
    cache: Literal["auto", "5m", "1h", "none"] = "auto",
    structured_output_mode: StructuredOutputMode = "auto",
    kwargs: Optional[
        "OpenAIClientArgs | AnthropicClientArgs | ConverseClientArgs"
    ] = None,
) -> Chat:
    """
    Chat with a model hosted on AWS Bedrock.

    Bedrock exposes three APIs across two endpoints, and `api` selects which
    one to use and which request format to send:

    * `"responses"` uses the OpenAI Responses API on the `bedrock-mantle`
      endpoint. This is the only way to reach the GPT-5 family, Grok, and Gemma
      on Bedrock.
    * `"messages"` uses the Anthropic Messages API on the `bedrock-mantle`
      endpoint. Only Claude models are available here, but it includes some
      (like Claude Mythos) that no other Bedrock API serves.
    * `"converse"` uses the Converse API on the `bedrock-runtime` endpoint.
      This is the default path for models available through Converse.

    By default the API is picked from `model`, falling back to `"converse"` for
    models that aren't recognised as mantle-only. Set `api` explicitly to
    override this -- which is also how you reach a mantle model that Converse
    can also serve, since those aren't auto-routed.

    Note that the two endpoints have separate token quotas, so moving a model
    from one to the other changes which quota it consumes.

    Prerequisites
    -------------
    ::: {.callout-note}
    ## AWS credentials

    Authentication uses botocore's standard credential chain, so environment
    variables, `~/.aws/config`, SSO, and instance roles all work. Pass
    `aws_profile` to select a named profile.

    For direct Converse requests, pass `api_key` via `kwargs` or set
    `AWS_BEARER_TOKEN_BEDROCK` to authenticate with a bearer token instead of
    SigV4. An explicit `api_key` takes priority; otherwise an explicit
    `aws_profile` uses SigV4 before the environment bearer token is considered.
    :::

    The direct Converse client uses raw HTTPX plus AWS libraries. The `bedrock`
    extra provides the AWS libraries and vendor SDKs for the mantle APIs
    (`openai` for `api="responses"` and `anthropic` for `api="messages"`). A
    chat uses one of these three API clients.

    Parameters
    ----------
    system_prompt
        A system prompt to set the behavior of the assistant.
    model
        The model to use for the chat. Defaults to
        `"us.anthropic.claude-sonnet-5"`.
    api
        Which Bedrock API to use. The default, `None`, picks the API from
        `model`.
    aws_profile
        The AWS profile to use. Defaults to botocore's default profile.
    aws_region
        The AWS region to use. Defaults to the region from your AWS config.
    base_url
        Override the endpoint URL. The default is the standard endpoint for
        the selected `api` and your region, honoring the official AWS SDKs'
        endpoint override environment variables (a service-specific variable
        outranks the generic `AWS_ENDPOINT_URL`):
        `AWS_ENDPOINT_URL_BEDROCK_RUNTIME` for `"converse"`, and
        `AWS_ENDPOINT_URL_BEDROCK_MANTLE` for `"messages"` and `"responses"`
        (which append their API-specific path to the override). For the
        mantle APIs, an override is also needed to reach mantle's other
        OpenAI-compatible path, `/v1`, which serves older open-weight models
        like `gpt-oss` and rejects the models `/openai/v1` serves.
    max_tokens
        Maximum number of tokens to generate, defaulting to 4096 when
        `api="converse"` or `api="messages"`. Passing this when `api="responses"` raises, since
        the Responses API has no constructor-level equivalent -- set a cap
        per-request instead via `chat.set_model_params(max_tokens=...)`.
    cache
        Prompt caching for `api="converse"` and `api="messages"`. The Responses
        API caches automatically, so this must be left at `"auto"` when
        `api="responses"`.
    structured_output_mode
        How to extract structured data when a `data_model` is given. `"auto"`
        (default) uses Bedrock's native structured outputs API
        (`outputConfig` for `api="converse"`, `output_config` for
        `api="messages"`) for models known to support it (currently Anthropic
        Claude models), and a forced tool call otherwise; if a model rejects
        the native API, chatlas warns, remembers, and falls back to the tool
        call for subsequent requests. `"native"` forces the native API;
        `"tool"` forces the tool call. Not supported when
        `api="responses"`. Native mode streams structured output
        incrementally; tool mode falls back to a single non-streaming
        request (with a warning) from `.stream()`/`.stream_async()`.
    kwargs
        Additional client arguments. For `api="converse"`, these are raw
        `httpx.Client` arguments (except `auth` and `base_url`, which
        `ChatBedrock` manages) and `api_key` is a Bedrock bearer token. For
        `api="responses"` and `api="messages"`, these are arguments for the
        respective OpenAI or Anthropic SDK client. On the Converse path,
        `api_key` is consumed to create an `Authorization: Bearer ...` header,
        not passed to `httpx.Client`.

    Returns
    -------
    Chat
        A Chat object.

    Examples
    --------
    ```python
    from chatlas import ChatBedrock

    # Frontier OpenAI models, which only exist on bedrock-mantle
    chat = ChatBedrock(model="openai.gpt-5.6-sol")
    chat.chat("What is 1 + 1? Just the number.")

    # Claude through the Anthropic Messages API on mantle
    chat = ChatBedrock(model="anthropic.claude-haiku-4-5", api="messages")
    ```
    """
    if model is None:
        model = log_model_default(DEFAULT_MODEL)

    api = api or bedrock_api_for_model(model)
    if api not in ("converse", "messages", "responses"):
        raise ValueError(
            f'Invalid `api` value "{api}". '
            'Must be one of "converse", "messages", or "responses".'
        )
    if structured_output_mode != "auto" and api == "responses":
        raise ValueError(
            '`structured_output_mode` is not supported when `api="responses"`.'
        )

    region = bedrock_region(aws_profile, aws_region)
    if api != "converse" and bedrock_uses_credential_chain(
        cast("Optional[OpenAIClientArgs | AnthropicClientArgs]", kwargs), aws_profile
    ):
        # Fail fast with botocore's own error (expired SSO, broken assume-role)
        # instead of a request-time failure buried by the SDK's retry machinery.
        bedrock_credentials(aws_profile)

    if api == "responses":
        if cache != "auto":
            raise ValueError(
                '`cache` is not supported when `api="responses"`. '
                "The Responses API caches prompts automatically."
            )
        if not isinstance(max_tokens, MISSING_TYPE):
            raise ValueError(
                '`max_tokens` is not supported when `api="responses"`. '
                "Set it per-request instead: chat.set_model_params(max_tokens=...)"
            )
        return Chat(
            provider=BedrockResponsesProvider(
                model=bedrock_strip_region_prefix(model),
                aws_profile=aws_profile,
                aws_region=region,
                base_url=base_url,
                # `api` selects which arm of the OpenAIClientArgs | AnthropicClientArgs
                # union applies; the type system can't express that through a
                # runtime branch.
                kwargs=cast("Optional[OpenAIClientArgs]", kwargs),
            ),
            system_prompt=system_prompt,
        )

    if api == "converse":
        # Imported here because BedrockConverseProvider uses this module's
        # endpoint and credential helpers.
        from ._provider_bedrock_converse import BedrockConverseProvider

        return Chat(
            provider=BedrockConverseProvider(
                model=model,
                aws_profile=aws_profile,
                aws_region=region,
                base_url=base_url,
                max_tokens=4096 if isinstance(max_tokens, MISSING_TYPE) else max_tokens,
                cache=cache,
                structured_output_mode=structured_output_mode,
                kwargs=cast("Optional[ConverseClientArgs]", kwargs),
            ),
            system_prompt=system_prompt,
        )

    return Chat(
        provider=BedrockMessagesProvider(
            model=bedrock_strip_region_prefix(model),
            aws_profile=aws_profile,
            aws_region=region,
            base_url=base_url,
            max_tokens=4096 if isinstance(max_tokens, MISSING_TYPE) else max_tokens,
            # The Messages API caches via cache_control blocks, which have no
            # "auto" equivalent; all Claude models support caching.
            cache="5m" if cache == "auto" else cache,
            structured_output_mode=structured_output_mode,
            kwargs=cast("Optional[AnthropicClientArgs]", kwargs),
        ),
        system_prompt=system_prompt,
    )


@no_file_management
class BedrockResponsesProvider(OpenAIProvider):
    """Reaches bedrock-mantle's OpenAI Responses API."""

    def __init__(
        self,
        *,
        model: str,
        aws_profile: Optional[str],
        aws_region: str,
        base_url: Optional[str],
        name: str = "AWS/Bedrock",
        kwargs: Optional["OpenAIClientArgs"] = None,
    ):
        try:
            from openai import AsyncBedrockOpenAI, BedrockOpenAI
        except ImportError:
            raise ImportError(
                "`ChatBedrock()` requires `openai` >= 2.50.0 for its native "
                "Bedrock client. Upgrade with `pip install -U 'chatlas[bedrock]'`."
            )

        resolved_base_url = base_url or bedrock_base_url("responses", aws_region)

        super().__init__(
            name=name,
            model=model,
            base_url=resolved_base_url,
            # The Bedrock clients constructed below replace the plain OpenAI
            # clients, but the OpenAI constructor still requires some value.
            api_key="not-used",
        )

        sync_kwargs, async_kwargs = bedrock_responses_client_kwargs(
            bedrock_client_kwargs(
                aws_profile=aws_profile,
                aws_region=aws_region,
                base_url=resolved_base_url,
                kwargs=kwargs,
            )
        )
        self._client = BedrockOpenAI(**sync_kwargs)
        self._async_client = AsyncBedrockOpenAI(**async_kwargs)
        self._models_base_url = bedrock_models_base_url(resolved_base_url)

    def list_models(self) -> list[ModelInfo]:
        models_client = self._client.with_options(base_url=self._models_base_url)
        return openai_models_to_info(models_client.models.list(), self.name)

    @staticmethod
    def _as_input_param(content: "Content", role: "Role") -> "ResponseInputItemParam":
        content = materialize_url_content(content)
        return OpenAIProvider._as_input_param(content, role)


@no_file_management
class BedrockMessagesProvider(AnthropicProvider):
    """Reaches bedrock-mantle's Anthropic Messages API."""

    def __init__(
        self,
        *,
        model: str,
        aws_profile: Optional[str],
        aws_region: str,
        base_url: Optional[str],
        max_tokens: int = 4096,
        cache: Literal["5m", "1h", "none"] = "5m",
        structured_output_mode: StructuredOutputMode = "auto",
        name: str = "AWS/Bedrock",
        kwargs: Optional["AnthropicClientArgs"] = None,
    ):
        try:
            from anthropic.lib.bedrock import (
                AnthropicBedrockMantle,
                AsyncAnthropicBedrockMantle,
            )
            from openai import BedrockOpenAI
        except ImportError:
            raise ImportError(
                '`ChatBedrock(api="messages")` requires the `anthropic` and `openai` '
                "packages. "
                "Install it with `pip install chatlas[bedrock]`."
            )

        super().__init__(
            name=name,
            model=model,
            max_tokens=max_tokens,
            cache=cache,
            structured_output_mode=structured_output_mode,
        )
        # Models that have rejected `output_config` with a 400, and models
        # that have accepted it (see `_native_retry_args`/`can_stream`);
        # only populated in "auto" mode.
        self._native_unsupported_models: set[str] = set()
        self._native_verified_models: set[str] = set()

        sync_kwargs, async_kwargs = split_http_client_kwargs(
            bedrock_client_kwargs(
                aws_profile=aws_profile,
                aws_region=aws_region,
                base_url=base_url or bedrock_base_url("messages", aws_region),
                kwargs=kwargs,
            )
        )
        self._client = AnthropicBedrockMantle(**sync_kwargs)
        # Signing here runs on the event loop: the mantle client's async
        # _prepare_request() calls botocore synchronously, where the openai SDK
        # offloads it to a thread. Left alone deliberately -- signing a request
        # with an already-resolved credential is sub-millisecond CPU work, and
        # wrapping it would mean overriding an SDK private method to fix what is
        # upstream's to fix.
        self._async_client = AsyncAnthropicBedrockMantle(**async_kwargs)
        self._models_client = BedrockOpenAI(
            **bedrock_models_client_kwargs(
                base_url=bedrock_models_base_url(str(self._client.base_url)),
                api_key=self._client.api_key,
                aws_access_key=self._client.aws_access_key,
                aws_profile=self._client.aws_profile,
                aws_region=self._client.aws_region,
                aws_secret_key=self._client.aws_secret_key,
                aws_session_token=self._client.aws_session_token,
                kwargs=sync_kwargs,
            )
        )

    def list_models(self) -> list[ModelInfo]:
        return openai_models_to_info(self._models_client.models.list(), self.name)

    @staticmethod
    def _as_content_block(content: "Content") -> "ContentBlockParam":
        return AnthropicProvider._as_content_block(materialize_url_content(content))

    # -- Native structured-output probing ------------------------------------
    #
    # Mantle's `output_config` support is per-model in a way that can't be
    # derived from the model ID, and differs from both the Anthropic API and
    # Converse (verified live 2026-10-16: `claude-opus-5` accepts it while
    # `claude-haiku-4-5` and `claude-opus-4-8` 400 with
    # "output_config.format: Extra inputs are not permitted"). So "auto" mode
    # is optimistic, exactly like the Converse provider: the first native
    # attempt doubles as the probe, the specific rejection is remembered, and
    # the request is retried with the forced tool call.

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[False],
        turns: list["Turn"],
        tools: dict[str, "Tool | ToolBuiltIn"],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["AnthropicSubmitInputArgs"] = None,
    ): ...

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[True],
        turns: list["Turn"],
        tools: dict[str, "Tool | ToolBuiltIn"],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["AnthropicSubmitInputArgs"] = None,
    ): ...

    def chat_perform(
        self,
        *,
        stream: bool,
        turns: list["Turn"],
        tools: dict[str, "Tool | ToolBuiltIn"],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["AnthropicSubmitInputArgs"] = None,
    ):
        from anthropic import BadRequestError

        args = self._chat_perform_args(stream, turns, tools, data_model, kwargs)
        if stream:
            # No retry here: `can_stream()` keeps unverified native attempts
            # out of the streaming paths, so a structured stream only carries
            # `output_config` once the model has accepted one (or in explicit
            # "native" mode, whose errors the user asked for).
            return self._client.messages.create(**args)  # type: ignore
        try:
            response = self._client.messages.create(**args)  # type: ignore
        except BadRequestError as error:
            retry = self._native_retry_args(
                error, args, stream, turns, tools, data_model, kwargs
            )
            if retry is None:
                raise
            return self._client.messages.create(**retry)  # type: ignore
        # A successful request that carried `output_config.format` proves the
        # model accepts it, so "auto" mode may stream natively from now on
        # (see `can_stream`).
        if self._carries_native_format(args) and self._structured_output_mode == "auto":
            self._native_verified_models.add(self.model)
        return response

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[False],
        turns: list["Turn"],
        tools: dict[str, "Tool | ToolBuiltIn"],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["AnthropicSubmitInputArgs"] = None,
    ): ...

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[True],
        turns: list["Turn"],
        tools: dict[str, "Tool | ToolBuiltIn"],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["AnthropicSubmitInputArgs"] = None,
    ): ...

    async def chat_perform_async(
        self,
        *,
        stream: bool,
        turns: list["Turn"],
        tools: dict[str, "Tool | ToolBuiltIn"],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["AnthropicSubmitInputArgs"] = None,
    ):
        from anthropic import BadRequestError

        args = self._chat_perform_args(stream, turns, tools, data_model, kwargs)
        if stream:
            return await self._async_client.messages.create(**args)  # type: ignore
        try:
            response = await self._async_client.messages.create(**args)  # type: ignore
        except BadRequestError as error:
            retry = self._native_retry_args(
                error, args, stream, turns, tools, data_model, kwargs
            )
            if retry is None:
                raise
            return await self._async_client.messages.create(**retry)  # type: ignore
        if self._carries_native_format(args) and self._structured_output_mode == "auto":
            self._native_verified_models.add(self.model)
        return response

    def _native_retry_args(
        self,
        error: "BadRequestError",
        args: "AnthropicSubmitInputArgs",
        stream: bool,
        turns: list["Turn"],
        tools: dict[str, "Tool | ToolBuiltIn"],
        data_model: Optional[type[BaseModel]],
        kwargs: Optional["AnthropicSubmitInputArgs"],
    ) -> Optional["AnthropicSubmitInputArgs"]:
        """Rebuild request args for tool-based extraction after a native 400.

        Since mantle's `output_config` support can't be derived from the
        model ID, "auto" mode is optimistic: on the specific rejection,
        remember the model as unsupported and retry with the forced tool
        call. Returns `None` (caller re-raises) when the failure wasn't a
        native-mode attempt -- including explicit "native" mode, where the
        user asked for `output_config` and gets its error. (Only called from
        the non-streaming paths: `can_stream()` keeps unverified native
        attempts out of the streaming paths, so a structured stream never
        carries `output_config` until the model has accepted one.)

        Retry eligibility comes from the failed request's own
        `output_config.format`, not the unsupported-model cache: a concurrent
        request may have populated the cache while this one was in flight,
        and it still deserves its retry.
        """
        if (
            data_model is None
            or self._structured_output_mode != "auto"
            or not self._carries_native_format(args)
            or not self._is_native_rejection(error)
        ):
            return None
        if self.model not in self._native_unsupported_models:
            self._native_unsupported_models.add(self.model)
            warnings.warn(
                f"Bedrock model '{self.model}' rejected native structured "
                "outputs (`output_config`); falling back to tool-based "
                "extraction. This provider will use tool-based extraction "
                "for the model from now on.",
                stacklevel=caller_stacklevel(),
            )
        # `_use_native_structured_output()` is now False, so this rebuilds
        # the args with the forced `_structured_tool_call` instead of
        # `output_config`.
        return self._chat_perform_args(stream, turns, tools, data_model, kwargs)

    @staticmethod
    def _carries_native_format(args: "AnthropicSubmitInputArgs") -> bool:
        # `output_config` also carries adaptive-thinking effort (from the
        # `reasoning` param), which proves nothing about `format` support.
        output_config = args.get("output_config")
        return isinstance(output_config, dict) and "format" in output_config

    @staticmethod
    def _is_native_rejection(error: "BadRequestError") -> bool:
        # The specific 400 returned by mantle models that lack `output_config`
        # support ("output_config.format: Extra inputs are not permitted").
        # Other errors mentioning `output_config` -- notably schema-validation
        # 400s, which imply the model *does* support native output -- must
        # propagate rather than disable native mode.
        message = str(error)
        return (
            "output_config" in message and "Extra inputs are not permitted" in message
        )

    def _use_native_structured_output(self) -> bool:
        if self.model in self._native_unsupported_models:
            return False
        return super()._use_native_structured_output()

    def can_stream(self, data_model: Optional[type[BaseModel]]) -> bool:
        # Native structured outputs (`output_config`) stream as ordinary text
        # deltas, so they honor the extracted-JSON contract. The tool-based
        # strategy streams tool input in its own event type (and
        # `{"data": ...}`-wrapped at that), so it can't -- `Chat` falls back
        # to a single non-streaming request there.
        if data_model is None:
            return True
        if self._structured_output_mode == "native":
            return True
        if not self._use_native_structured_output():
            return False
        # Mantle's `output_config` support can't be derived from the model
        # ID, so "auto" proves it first: until a native request has
        # succeeded, `Chat`'s non-streaming fallback doubles as the probe
        # (and records the verdict either way).
        return self.model in self._native_verified_models


def bedrock_client_kwargs(
    *,
    aws_profile: Optional[str],
    aws_region: str,
    base_url: str,
    kwargs: "Optional[OpenAIClientArgs | AnthropicClientArgs]",
) -> AnyTypeDict:
    """
    Constructor kwargs for either SDK Bedrock client, with `kwargs` last so a
    user-supplied value wins.

    The return type is deliberately key-less: `ChatClientArgs` describes the
    *plain* OpenAI/Anthropic constructor, while the Bedrock clients accept that
    surface plus the `aws_*` keys, and a TypedDict can't express the union of
    two vendors' signatures. The callers unpack into the real constructors,
    which is where the keys actually get checked.
    """
    return cast(
        AnyTypeDict,
        {
            "aws_profile": aws_profile,
            "aws_region": aws_region,
            "base_url": base_url,
            **(kwargs or {}),
        },
    )


def bedrock_responses_client_kwargs(
    kwargs: AnyTypeDict,
) -> tuple[AnyTypeDict, AnyTypeDict]:
    sync_kwargs, async_kwargs = split_http_client_kwargs(kwargs)
    sync_kwargs_dict = cast(dict[str, Any], sync_kwargs)
    async_kwargs_dict = cast(dict[str, Any], async_kwargs)
    # Mantle rewrites this hop-by-hop header after OpenAI's SDK signs it, so
    # stabilize the signed value until the SDK stops including it in SigV4.
    if "http_client" not in sync_kwargs_dict:
        sync_kwargs_dict["http_client"] = httpx.Client(headers={"Connection": ""})
    if "http_client" not in async_kwargs_dict:
        async_kwargs_dict["http_client"] = httpx.AsyncClient(headers={"Connection": ""})
    return cast(AnyTypeDict, sync_kwargs_dict), cast(AnyTypeDict, async_kwargs_dict)


def bedrock_models_client_kwargs(
    *,
    base_url: str,
    api_key: Optional[str],
    aws_access_key: Optional[str],
    aws_profile: Optional[str],
    aws_region: Optional[str],
    aws_secret_key: Optional[str],
    aws_session_token: Optional[str],
    kwargs: AnyTypeDict,
) -> AnyTypeDict:
    """Adapt resolved Anthropic Bedrock client settings for `BedrockOpenAI`."""
    kwargs_dict = cast(dict[str, Any], kwargs)
    result: dict[str, Any] = {
        "aws_profile": aws_profile,
        "aws_region": aws_region,
        "base_url": base_url,
    }
    for key in (
        "aws_credentials_provider",
        "bedrock_token_provider",
        "default_headers",
        "default_query",
        "http_client",
        "max_retries",
        "timeout",
        "_strict_response_validation",
    ):
        if key in kwargs_dict:
            result[key] = kwargs_dict[key]
    if api_key is not None:
        result["api_key"] = api_key
    if aws_access_key is not None:
        result["aws_access_key_id"] = aws_access_key
    if aws_secret_key is not None:
        result["aws_secret_access_key"] = aws_secret_key
    if aws_session_token is not None:
        result["aws_session_token"] = aws_session_token
    # The two Bedrock clients accept overlapping but non-identical keyword
    # arguments; the keyless TypedDict represents that dynamic bridge.
    return cast(AnyTypeDict, result)


def load_model_apis() -> dict[str, BedrockAPI]:
    raw = resources.files("chatlas").joinpath("data/bedrock_apis.json")
    return json.loads(raw.read_text(encoding="utf-8"))


MODEL_APIS = load_model_apis()


def bedrock_api_for_model(model: Optional[str]) -> BedrockAPI:
    """
    Which Bedrock API serves `model`.

    The table only lists models Converse *can't* serve, so anything unlisted
    falls through to Converse. That keeps a data refresh from silently moving a
    working call onto a different endpoint with different quotas -- it can only
    ever turn a failure into a success -- and lets unfamiliar ids (custom ARNs,
    new inference profiles) keep working.
    """
    if not model:
        return "converse"
    return MODEL_APIS.get(CROSS_REGION_PREFIX.sub("", model), "converse")


def bedrock_strip_region_prefix(model: str) -> str:
    """
    Strip a cross-region inference prefix (e.g. `"us."`) from `model`.

    Converse needs the prefix on the model id it's sent, since that's how it
    picks the inference profile. Mantle's Anthropic and OpenAI-compatible
    endpoints have no such concept and 404 if the prefix is included, so it
    must be stripped from the model id used in requests to `"messages"` and
    `"responses"`.
    """
    return CROSS_REGION_PREFIX.sub("", model)


def aws_endpoint_url(var: str, default: str) -> str:
    """An AWS service endpoint override env var, or `default` when unset."""
    url = os.environ.get(var, "")
    return url.rstrip("/") if url else default


def bedrock_base_url(api: BedrockAPI, region: str) -> str:
    # Match the official AWS SDKs' endpoint resolution precedence: the
    # service-specific override (AWS_ENDPOINT_URL_BEDROCK_RUNTIME for the
    # runtime (converse) service, AWS_ENDPOINT_URL_BEDROCK_MANTLE for mantle)
    # outranks the generic AWS_ENDPOINT_URL, which outranks the regional
    # endpoint computed from `region`.
    if api == "converse":
        return aws_endpoint_url(
            "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
            aws_endpoint_url(
                "AWS_ENDPOINT_URL",
                f"https://bedrock-runtime.{region}.amazonaws.com",
            ),
        )

    host = aws_endpoint_url(
        "AWS_ENDPOINT_URL_BEDROCK_MANTLE",
        aws_endpoint_url("AWS_ENDPOINT_URL", MANTLE_HOST.format(region=region)),
    )
    if api == "messages":
        # The Anthropic SDK appends "/v1/messages" itself, so the "/v1" is
        # deliberately absent here.
        return f"{host}/anthropic"

    # Mantle has two OpenAI-compatible paths: newer models are served from
    # /openai/v1 and older open-weight models (like gpt-oss) from /v1. They are
    # disjoint, not aliases -- each rejects the other's models -- so reaching a
    # /v1 model needs an explicit base_url.
    return f"{host}/openai/v1"


def bedrock_models_base_url(base_url: str) -> str:
    """
    Where to list models for a chat endpoint at `base_url`.

    Mantle serves the listing from `/v1/models`, whereas chat requests use
    `/openai/v1` or `/anthropic`. Rewriting the path (rather than rebuilding
    the URL from the region) keeps a proxy or private endpoint in play for
    `.list_models()` too.
    """
    return re.sub(r"/(?:openai/v1|anthropic)/?$", "/v1", base_url)


def bedrock_credentials(profile: Optional[str]) -> Credentials:
    credentials = botocore_session(profile).get_credentials()
    if credentials is None:
        raise ValueError(
            "No AWS credentials found. Configure them via `aws configure`, a "
            "named profile (`aws_profile=`), or the standard AWS environment "
            "variables."
        )
    # Eagerly resolve credentials so a refresh failure (expired SSO, broken
    # assume-role) raises botocore's own error here, at construction time,
    # instead of being wrapped into an opaque connection error by the vendor
    # SDK when signing runs mid-request.
    credentials.get_frozen_credentials()
    return credentials


# Client kwargs that steer credential resolution away from botocore's chain --
# the union of what the two SDK Bedrock clients accept, since either provider
# can receive them.
SDK_CREDENTIAL_KWARGS = frozenset(
    {
        "api_key",
        "skip_auth",
        "bedrock_token_provider",
        "aws_access_key",
        "aws_access_key_id",
        "aws_secret_key",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_credentials_provider",
    }
)


def bedrock_uses_credential_chain(
    kwargs: "Optional[OpenAIClientArgs | AnthropicClientArgs]",
    aws_profile: Optional[str],
) -> bool:
    """
    Whether the SDK clients will authenticate through botocore's credential
    chain -- the only case where validating that chain up front says anything.

    Both SDKs resolve auth in the same order: credentials passed to the client,
    then an explicit profile (SigV4 through that profile), then a bearer token
    from the environment, then the default chain. So an `api_key` or static
    keys mean the chain is bypassed entirely, while `aws_profile` means it is
    still used and worth validating -- even alongside a bearer token, which an
    explicit profile outranks. ANTHROPIC_AWS_API_KEY is honored only by the
    anthropic SDK, but treating it as a bearer token for both paths just trades
    a construction-time chain error for the SDK's own (clearer) one.
    """
    if kwargs and not SDK_CREDENTIAL_KWARGS.isdisjoint(kwargs):
        return False
    if aws_profile is not None:
        return True
    return not (
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or os.environ.get("ANTHROPIC_AWS_API_KEY")
    )


def bedrock_region(profile: Optional[str], region: Optional[str]) -> str:
    if region is not None:
        return region
    resolved = botocore_session(profile).get_config_variable("region")
    if resolved is None:
        raise ValueError(
            "No AWS region found. Pass `aws_region=`, set AWS_REGION, or set a "
            "region in your AWS config."
        )
    return resolved


def botocore_session(profile: Optional[str]) -> Session:
    try:
        from botocore.session import Session
    except ImportError:
        raise ImportError(
            "`ChatBedrock()` requires the `botocore` package. "
            "Install it with `pip install chatlas[bedrock]`."
        )
    return Session(profile=profile)
