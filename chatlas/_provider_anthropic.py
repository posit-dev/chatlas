from __future__ import annotations

import base64
import re
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Optional,
    Sequence,
    Union,
    cast,
    overload,
)

import orjson
from pydantic import BaseModel

from ._chat import Chat
from ._content import (
    Content,
    ContentImageInline,
    ContentImageRemote,
    ContentJson,
    ContentPDF,
    ContentText,
    ContentThinking,
    ContentToolRequest,
    ContentToolRequestFetch,
    ContentToolRequestSearch,
    ContentToolResponseFetch,
    ContentToolResponseSearch,
    ContentToolResult,
)
from ._logging import log_model_default
from ._provider import (
    BatchStatus,
    ModelInfo,
    Provider,
    StandardModelParamNames,
    StandardModelParams,
)
from ._tokens import get_price_info
from ._tools import Tool, ToolBuiltIn, basemodel_to_param_schema
from ._tools_builtin import ToolWebFetch, ToolWebSearch
from ._turn import AssistantTurn, SystemTurn, Turn, UserTurn, user_turn
from ._utils import split_http_client_kwargs

# Models that support the new structured outputs API (output_format parameter)
# https://platform.claude.com/docs/en/build-with-claude/structured-outputs
STRUCTURED_OUTPUT_MODELS = {
    "claude-sonnet-4-5",
    "claude-opus-4-1",
    "claude-opus-4-5",
    "claude-haiku-4-5",
}

STRUCTURED_OUTPUTS_BETA = "structured-outputs-2025-11-13"


def use_beta_structured_output(model: str, data_model: Optional[type[BaseModel]]) -> bool:
    """
    Check if we should use the beta structured outputs API.

    Returns True if data_model is provided and the model supports the new API.
    """
    if data_model is None:
        return False
    # Handle dated model versions like "claude-sonnet-4-5-20250514"
    # by checking if any supported model is a prefix
    for supported in STRUCTURED_OUTPUT_MODELS:
        if model.startswith(supported):
            return True
    return False


if TYPE_CHECKING:
    from anthropic.types import (
        Message,
        MessageParam,
        RawMessageStreamEvent,
        TextBlock,
        ThinkingBlock,
        ThinkingBlockParam,
        ToolUnionParam,
        ToolUseBlock,
    )
    from anthropic.types.cache_control_ephemeral_param import CacheControlEphemeralParam
    from anthropic.types.document_block_param import DocumentBlockParam
    from anthropic.types.image_block_param import ImageBlockParam
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request as BatchRequest
    from anthropic.types.model_param import ModelParam
    from anthropic.types.text_block_param import TextBlockParam
    from anthropic.types.thinking_config_enabled_param import ThinkingConfigEnabledParam
    from anthropic.types.tool_result_block_param import ToolResultBlockParam
    from anthropic.types.tool_use_block_param import ToolUseBlockParam

    from .types.anthropic import ChatBedrockClientArgs, ChatClientArgs, SubmitInputArgs

    ContentBlockParam = Union[
        TextBlockParam,
        ImageBlockParam,
        ToolUseBlockParam,
        ToolResultBlockParam,
        DocumentBlockParam,
        ThinkingBlockParam,
    ]
else:
    Message = object
    RawMessageStreamEvent = object


def ChatAnthropic(
    *,
    system_prompt: Optional[str] = None,
    model: "Optional[ModelParam]" = None,
    max_tokens: int = 4096,
    reasoning: Optional["int | ThinkingConfigEnabledParam"] = None,
    cache: Literal["5m", "1h", "none"] = "5m",
    api_key: Optional[str] = None,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", Message]:
    """
    Chat with an Anthropic Claude model.

    [Anthropic](https://www.anthropic.com) provides a number of chat based
    models under the [Claude](https://www.anthropic.com/claude) moniker.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## API key

    Note that a Claude Pro membership does not give you the ability to call
    models via the API. You will need to go to the [developer
    console](https://console.anthropic.com/account/keys) to sign up (and pay
    for) a developer account that will give you an API key that you can use with
    this package.
    :::

    ::: {.callout-note}
    ## Python requirements

    `ChatAnthropic` requires the `anthropic` package: `pip install "chatlas[anthropic]"`.
    :::

    Examples
    --------

    ```python
    import os
    from chatlas import ChatAnthropic

    chat = ChatAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    system_prompt
        A system prompt to set the behavior of the assistant.
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly
        choosing a model for all but the most casual use.
    max_tokens
        Maximum number of tokens to generate before stopping.
    reasoning
        Determines how many tokens Claude can be allocated to reasoning. Must be
        ≥1024 and less than `max_tokens`. Larger budgets can enable more
        thorough analysis for complex problems, improving response quality.  See
        [extended
        thinking](https://docs.claude.com/en/docs/build-with-claude/extended-thinking)
        for details.
    cache
        How long to cache inputs? Defaults to "5m" (five minutes).
        Set to "none" to disable caching or "1h" to cache for one hour.
        See the Caching section for details.
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `ANTHROPIC_API_KEY` environment
        variable.
    kwargs
        Additional arguments to pass to the `anthropic.Anthropic()` client
        constructor.

    Returns
    -------
    Chat
        A Chat object.

    Note
    ----
    Pasting an API key into a chat constructor (e.g., `ChatAnthropic(api_key="...")`)
    is the simplest way to get started, and is fine for interactive use, but is
    problematic for code that may be shared with others.

    Instead, consider using environment variables or a configuration file to manage
    your credentials. One popular way to manage credentials is to use a `.env` file
    to store your credentials, and then use the `python-dotenv` package to load them
    into your environment.

    ```shell
    pip install python-dotenv
    ```

    ```shell
    # .env
    ANTHROPIC_API_KEY=...
    ```

    ```python
    from chatlas import ChatAnthropic
    from dotenv import load_dotenv

    load_dotenv()
    chat = ChatAnthropic()
    chat.console()
    ```

    Another, more general, solution is to load your environment variables into the shell
    before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

    ```shell
    export ANTHROPIC_API_KEY=...
    ```

    Caching
    -------

    Caching with Claude is a bit more complicated than other providers but we
    believe that on average it will save you both money and time, so we have
    enabled it by default. With other providers, like OpenAI and Google,
    you only pay for cache reads, which cost 10% of the normal price. With
    Claude, you also pay for cache writes, which cost 125% of the normal price
    for 5 minute caching and 200% of the normal price for 1 hour caching.

    How does this affect the total cost of a conversation? Imagine the first
    turn sends 1000 input tokens and receives 200 output tokens. The second
    turn must first send both the input and output from the previous turn
    (1200 tokens). It then sends a further 1000 tokens and receives 200 tokens
    back.

    To compare the prices of these two approaches we can ignore the cost of
    output tokens, because they are the same for both. How much will the input
    tokens cost? If we don't use caching, we send 1000 tokens in the first turn
    and 2200 (1000 + 200 + 1000) tokens in the second turn for a total of 3200
    tokens. If we use caching, we'll send (the equivalent of) 1000 * 1.25 = 1250
    tokens in the first turn. In the second turn, 1000 of the input tokens will
    be cached so the total cost is 1000 * 0.1 + (200 + 1000) * 1.25 = 1600
    tokens. That makes a total of 2850 tokens, i.e. 11% fewer tokens,
    decreasing the overall cost.

    Obviously, the details will vary from conversation to conversation, but
    if you have a large system prompt that you re-use many times you should
    expect to see larger savings. You can see exactly how many input and
    cache input tokens each turn uses, along with the total cost,
    with `chat.get_tokens()`. If you don't see savings for your use case, you can
    suppress caching with `cache="none"`.

    Note: Claude will only cache longer prompts, with caching requiring at least
    1024-4096 tokens, depending on the model. So don't be surprised if you
    don't see any differences with caching if you have a short prompt.

    See all the details at
    <https://docs.claude.com/en/docs/build-with-claude/prompt-caching>.
    """

    if model is None:
        model = log_model_default("claude-sonnet-4-5")

    kwargs_chat: "SubmitInputArgs" = {}
    if reasoning is not None:
        if isinstance(reasoning, int):
            reasoning = {"type": "enabled", "budget_tokens": reasoning}
        kwargs_chat = {"thinking": reasoning}

    return Chat(
        provider=AnthropicProvider(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            cache=cache,
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
        kwargs_chat=kwargs_chat,
    )


class AnthropicProvider(
    Provider[Message, RawMessageStreamEvent, Message, "SubmitInputArgs"]
):
    def __init__(
        self,
        *,
        max_tokens: int = 4096,
        model: str,
        api_key: Optional[str] = None,
        name: str = "Anthropic",
        cache: Literal["5m", "1h", "none"] = "5m",
        kwargs: Optional["ChatClientArgs"] = None,
    ):
        super().__init__(name=name, model=model)
        try:
            from anthropic import Anthropic, AsyncAnthropic
        except ImportError:
            raise ImportError(
                "`ChatAnthropic()` requires the `anthropic` package. "
                "You can install it with 'pip install anthropic'."
            )
        self._max_tokens = max_tokens
        self._cache: Literal["5m", "1h", "none"] = cache

        kwargs_full: "ChatClientArgs" = {
            "api_key": api_key,
            **(kwargs or {}),
        }

        sync_kwargs, async_kwargs = split_http_client_kwargs(kwargs_full)

        # TODO: worth bringing in sync types?
        self._client = Anthropic(**sync_kwargs)  # type: ignore
        self._async_client = AsyncAnthropic(**async_kwargs)

    def list_models(self):
        models = self._client.models.list()

        res: list[ModelInfo] = []
        for m in models:
            pricing = get_price_info(self.name, m.id) or {}
            info: ModelInfo = {
                "id": m.id,
                "name": m.display_name,
                "created_at": m.created_at.date(),
                "input": pricing.get("input"),
                "output": pricing.get("output"),
                "cached_input": pricing.get("cached_input"),
            }
            res.append(info)

        # Sort list by created_by field (more recent first)
        res.sort(
            key=lambda x: x.get("created_at", 0),
            reverse=True,
        )

        return res

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ): ...

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ): ...

    def chat_perform(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ):
        api_kwargs = self._chat_perform_args(stream, turns, tools, data_model, kwargs)
        if use_beta_structured_output(self.model, data_model):
            # Beta API has slightly different type signatures but is runtime compatible
            return self._client.beta.messages.create(
                betas=[STRUCTURED_OUTPUTS_BETA],
                **api_kwargs,  # type: ignore[arg-type]
            )
        return self._client.messages.create(**api_kwargs)  # type: ignore

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ): ...

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ): ...

    async def chat_perform_async(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ):
        api_kwargs = self._chat_perform_args(stream, turns, tools, data_model, kwargs)
        if use_beta_structured_output(self.model, data_model):
            # Beta API has slightly different type signatures but is runtime compatible
            return await self._async_client.beta.messages.create(
                betas=[STRUCTURED_OUTPUTS_BETA],
                **api_kwargs,  # type: ignore[arg-type]
            )
        return await self._async_client.messages.create(**api_kwargs)  # type: ignore

    def _chat_perform_args(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ) -> "SubmitInputArgs":
        """Build kwargs for the Anthropic messages API."""
        tool_schemas = [self._anthropic_tool_schema(tool) for tool in tools.values()]

        use_beta = use_beta_structured_output(self.model, data_model)
        data_model_tool: Tool | None = None

        if data_model is not None and not use_beta:
            # Fall back to the old tool-based approach for older models
            data_model_tool = self.create_data_model_tool(data_model)
            tool_schemas.append(self._anthropic_tool_schema(data_model_tool))

            if stream:
                stream = False
                warnings.warn(
                    "Anthropic does not support structured data extraction in "
                    "streaming mode for older models. Consider using a newer model "
                    f"like {', '.join(sorted(STRUCTURED_OUTPUT_MODELS))}.",
                    stacklevel=4,
                )

        kwargs_full: "SubmitInputArgs" = {
            "stream": stream,
            "messages": self._as_message_params(turns),
            "model": self.model,
            "max_tokens": self._max_tokens,
            "tools": tool_schemas,
            **(kwargs or {}),
        }

        if use_beta and data_model is not None:
            # Use the new output_format parameter for structured outputs
            from anthropic import transform_schema

            kwargs_full["output_format"] = {  # type: ignore[typeddict-unknown-key]
                "type": "json_schema",
                "schema": transform_schema(data_model),
            }

        if data_model_tool:
            # Old approach: force tool use
            kwargs_full["tool_choice"] = {
                "type": "tool",
                "name": data_model_tool.name,
            }

        if "system" not in kwargs_full:
            if len(turns) > 0 and isinstance(turns[0], SystemTurn):
                sys_param: "TextBlockParam" = {
                    "type": "text",
                    "text": turns[0].text,
                }
                if self._cache_control():
                    sys_param["cache_control"] = self._cache_control()
                kwargs_full["system"] = [sys_param]

        return kwargs_full

    def create_data_model_tool(self, data_model: type[BaseModel]) -> Tool:
        """Create a fake tool for structured data extraction (old approach)."""

        def _structured_tool_call(**kwargs: Any):
            """Extract structured data"""
            pass

        data_model_tool = Tool.from_func(_structured_tool_call)

        data_model_schema = basemodel_to_param_schema(data_model)

        # Extract $defs from the nested schema and place at top level
        # JSON Schema $ref pointers like "#/$defs/..." need $defs at the root
        defs = data_model_schema.pop("$defs", None)

        params: dict[str, Any] = {
            "type": "object",
            "properties": {
                "data": data_model_schema,
            },
        }
        if defs:
            params["$defs"] = defs

        data_model_tool.schema["function"]["parameters"] = params

        return data_model_tool

    def stream_text(self, chunk) -> Optional[str]:
        if chunk.type == "content_block_delta":
            if chunk.delta.type == "text_delta":
                return chunk.delta.text
            if chunk.delta.type == "thinking_delta":
                return chunk.delta.thinking
        return None

    def stream_merge_chunks(self, completion, chunk):
        if chunk.type == "message_start":
            return chunk.message
        completion = cast("Message", completion)

        if chunk.type == "content_block_start":
            completion.content.append(chunk.content_block)
        elif chunk.type == "content_block_delta":
            this_content = completion.content[chunk.index]
            if chunk.delta.type == "text_delta":
                this_content = cast("TextBlock", this_content)
                this_content.text += chunk.delta.text
            elif chunk.delta.type == "input_json_delta":
                this_content = cast("ToolUseBlock", this_content)
                json_delta = chunk.delta.partial_json
                # For some reason Anthropic recently changed .input's type from
                # object to dict, but it doesn't seem to contain anything
                # useful. Maybe there is a better way to get at this streaming info,
                # but for now, we'll just accumulate the JSON string delta.
                if not isinstance(this_content.input, str):
                    this_content.input = ""  # type: ignore
                this_content.input += json_delta  # type: ignore
            elif chunk.delta.type == "thinking_delta":
                this_content = cast("ThinkingBlock", this_content)
                this_content.thinking += chunk.delta.thinking
            elif chunk.delta.type == "signature_delta":
                this_content = cast("ThinkingBlock", this_content)
                this_content.signature += chunk.delta.signature
            elif chunk.delta.type == "citations_delta":
                # https://docs.claude.com/en/docs/build-with-claude/citations#streaming-support
                # Accumulate citations on the content block
                if hasattr(this_content, "citations"):
                    this_content.citations.append(chunk.delta.citation)  # type: ignore
        elif chunk.type == "content_block_stop":
            this_content = completion.content[chunk.index]
            if this_content.type == "tool_use" and isinstance(this_content.input, str):
                try:
                    this_content.input = orjson.loads(this_content.input or "{}")
                except orjson.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON input: {e}")
        elif chunk.type == "message_delta":
            completion.stop_reason = chunk.delta.stop_reason
            completion.stop_sequence = chunk.delta.stop_sequence
            completion.usage.output_tokens = chunk.usage.output_tokens

        return completion

    def stream_turn(self, completion, has_data_model):
        return self._as_turn(completion, has_data_model)

    def value_turn(self, completion, has_data_model):
        return self._as_turn(completion, has_data_model)

    def value_tokens(self, completion):
        usage = completion.usage
        input_tokens = completion.usage.input_tokens

        # Account for cache writes by adjusting input tokens
        # Cache writes cost 125% for 5m and 200% for 1h
        # https://docs.claude.com/en/docs/build-with-claude/prompt-caching
        cache_input = usage.cache_creation_input_tokens or 0
        cache_mult = 2.0 if self._cache == "1h" else 1.25

        return (
            input_tokens + int(cache_input * cache_mult),
            completion.usage.output_tokens,
            usage.cache_read_input_tokens if usage.cache_read_input_tokens else 0,
        )

    def token_count(
        self,
        *args: Content | str,
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]],
    ) -> int:
        kwargs = self._token_count_args(
            *args,
            tools=tools,
            data_model=data_model,
        )
        res = self._client.messages.count_tokens(**kwargs)
        return res.input_tokens

    async def token_count_async(
        self,
        *args: Content | str,
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]],
    ) -> int:
        kwargs = self._token_count_args(
            *args,
            tools=tools,
            data_model=data_model,
        )
        res = await self._async_client.messages.count_tokens(**kwargs)
        return res.input_tokens

    def _token_count_args(
        self,
        *args: Content | str,
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]],
    ) -> dict[str, Any]:
        turn = user_turn(*args)

        api_kwargs = self._chat_perform_args(
            stream=False,
            turns=[turn],
            tools=tools,
            data_model=data_model,
        )

        args_to_keep = [
            "messages",
            "model",
            "system",
            "tools",
            "tool_choice",
        ]

        return {arg: api_kwargs[arg] for arg in args_to_keep if arg in api_kwargs}

    def translate_model_params(self, params: StandardModelParams) -> "SubmitInputArgs":
        res: "SubmitInputArgs" = {}
        if "temperature" in params:
            res["temperature"] = params["temperature"]

        if "top_p" in params:
            res["top_p"] = params["top_p"]

        if "top_k" in params:
            res["top_k"] = params["top_k"]

        if "max_tokens" in params:
            res["max_tokens"] = params["max_tokens"]

        if "stop_sequences" in params:
            res["stop_sequences"] = params["stop_sequences"]

        return res

    def supported_model_params(self) -> set[StandardModelParamNames]:
        return {
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "stop_sequences",
        }

    def _as_message_params(self, turns: Sequence[Turn]) -> list["MessageParam"]:
        messages: list["MessageParam"] = []
        for i, turn in enumerate(turns):
            if isinstance(turn, SystemTurn):
                continue  # system prompt passed as separate arg
            if not isinstance(turn, (UserTurn, AssistantTurn)):
                raise ValueError(f"Unknown role {turn.role}")

            content = [self._as_content_block(c) for c in turn.contents]

            # Drop empty assistant turns to avoid an API error
            # (all messages must have non-empty content)
            if turn.role == "assistant" and len(content) == 0:
                continue

            # Add cache control to the last content block in the last turn
            # https://docs.claude.com/en/docs/build-with-claude/prompt-caching#how-automatic-prefix-checking-works
            is_last_turn = i == len(turns) - 1
            if self._cache_control() and is_last_turn and len(content) > 0:
                # Note: ThinkingBlockParam (i.e., type: "thinking") doesn't support cache_control
                if content[-1].get("type") != "thinking":
                    content[-1]["cache_control"] = self._cache_control()  # type: ignore

            role = "user" if isinstance(turn, UserTurn) else "assistant"
            messages.append({"role": role, "content": content})
        return messages

    @staticmethod
    def _as_content_block(content: Content) -> "ContentBlockParam":
        if isinstance(content, ContentText):
            return {"text": content.text, "type": "text"}
        elif isinstance(content, ContentJson):
            text = orjson.dumps(content.value).decode("utf-8")
            return {"text": text, "type": "text"}
        elif isinstance(content, ContentPDF):
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.b64encode(content.data).decode("utf-8"),
                },
            }
        elif isinstance(content, ContentImageInline):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": content.image_content_type,
                    "data": content.data or "",
                },
            }
        elif isinstance(content, ContentImageRemote):
            return {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": content.url,
                },
            }
        elif isinstance(content, ContentToolRequest):
            return {
                "type": "tool_use",
                "id": content.id,
                "name": content.name,
                "input": cast(dict, content.arguments),
            }
        elif isinstance(content, ContentToolResult):
            res: ToolResultBlockParam = {
                "type": "tool_result",
                "tool_use_id": content.id,
                "is_error": content.error is not None,
                # Anthropic supports non-text contents like ImageBlockParam
                "content": content.get_model_value(),  # type: ignore
            }

            return res
        elif isinstance(content, ContentThinking):
            extra = content.extra or {}
            return {
                "type": "thinking",
                "thinking": content.thinking,
                "signature": extra.get("signature", ""),
            }
        elif isinstance(
            content,
            (
                ContentToolRequestSearch,
                ContentToolResponseSearch,
                ContentToolRequestFetch,
                ContentToolResponseFetch,
            ),
        ):
            # extra contains the full original content block param
            return cast("ContentBlockParam", content.extra)

        raise ValueError(f"Unknown content type: {type(content)}")

    @staticmethod
    def _anthropic_tool_schema(tool: "Tool | ToolBuiltIn") -> "ToolUnionParam":
        if isinstance(tool, ToolWebSearch):
            return tool.get_definition("anthropic")
        if isinstance(tool, ToolWebFetch):
            # N.B. seems the return type here (BetaWebFetchTool20250910Param) is
            # not a member of ToolUnionParam since it's still in beta?
            return tool.get_definition("anthropic")  # type: ignore
        if isinstance(tool, ToolBuiltIn):
            return tool.definition  # type: ignore

        fn = tool.schema["function"]
        name = fn["name"]

        res: "ToolUnionParam" = {
            "name": name,
            "input_schema": {
                "type": "object",
            },
        }

        if "description" in fn:
            res["description"] = fn["description"]

        if "parameters" in fn:
            res["input_schema"] = {
                "type": "object",
                **fn["parameters"],
            }

        return res

    def _as_turn(self, completion: Message, has_data_model=False) -> AssistantTurn:
        contents = []

        # Detect which structured output approach was used:
        # - Old approach: has a _structured_tool_call tool_use block
        # - New approach: has_data_model=True but no _structured_tool_call (JSON in text)
        uses_old_tool_approach = has_data_model and any(
            c.type == "tool_use" and c.name == "_structured_tool_call"
            for c in completion.content
        )
        uses_new_output_format = has_data_model and not uses_old_tool_approach

        for content in completion.content:
            if content.type == "text":
                if uses_new_output_format:
                    # New API: JSON response is in text content
                    contents.append(ContentJson(value=orjson.loads(content.text)))
                else:
                    contents.append(ContentText(text=content.text))
            elif content.type == "tool_use":
                if uses_old_tool_approach and content.name == "_structured_tool_call":
                    # Old API: extract from tool input
                    if not isinstance(content.input, dict):
                        raise ValueError(
                            "Expected data extraction tool to return a dictionary."
                        )
                    if "data" not in content.input:
                        raise ValueError(
                            "Expected data extraction tool to return a 'data' field."
                        )
                    else:
                        d = cast(dict, content.input["data"])
                        contents.append(ContentJson(value=d))
                else:
                    contents.append(
                        ContentToolRequest(
                            id=content.id,
                            name=content.name,
                            arguments=content.input,
                        )
                    )
            elif content.type == "thinking":
                contents.append(
                    ContentThinking(
                        thinking=content.thinking,
                        extra={"signature": content.signature},
                    )
                )
            elif content.type == "server_tool_use":
                # Unfortunately, content.model_dump() includes fields like "url"
                # that aren't acceptable as API input, so we manually construct
                # the extra dict
                if isinstance(content.input, str):
                    input_data = orjson.loads(content.input)
                else:
                    input_data = content.input

                extra = {
                    "type": content.type,
                    "id": content.id,
                    "name": content.name,
                    "input": input_data,
                }
                # https://docs.claude.com/en/docs/agents-and-tools/tool-use/web-search-tool#response
                if content.name == "web_search":
                    contents.append(
                        ContentToolRequestSearch(
                            query=str(input_data.get("query", "")),
                            extra=extra,
                        )
                    )
                # https://docs.claude.com/en/docs/agents-and-tools/tool-use/web-fetch-tool#response
                elif content.name == "web_fetch":
                    # N.B. type checker thinks this is unreachable due to
                    # ToolUnionParam not including BetaWebFetchTool20250910Param
                    # yet
                    contents.append(
                        ContentToolRequestFetch(
                            url=str(input_data.get("url", "")),
                            extra=extra,
                        )
                    )
                else:
                    raise ValueError(f"Unknown server tool: {content.name}")
            elif content.type == "web_search_tool_result":
                # https://docs.claude.com/en/docs/agents-and-tools/tool-use/web-search-tool#response
                urls: list[str] = []
                if isinstance(content.content, list):
                    urls = [x.url for x in content.content]
                contents.append(
                    ContentToolResponseSearch(
                        urls=urls,
                        extra=content.model_dump(),
                    )
                )
            elif content.type == "web_fetch_tool_result":
                # N.B. type checker thinks this is unreachable due to
                # ToolUnionParam not including BetaWebFetchTool20250910Param
                # yet. Also, at run-time, the SDK is currently giving non-sense
                # of type(content) == TextBlock, but it doesn't even fit that
                # shape?!? Anyway, content.content has a dict with the content
                # we want.
                content_fetch = cast("dict", getattr(content, "content", {}))
                if not content_fetch:
                    raise ValueError(
                        "web_fetch_tool_result content is empty. Please report this issue."
                    )
                extra = {
                    "type": "web_fetch_tool_result",
                    "tool_use_id": content.tool_use_id,  # type: ignore
                    "content": content_fetch,
                }
                contents.append(
                    ContentToolResponseFetch(
                        url=content_fetch.get("url", "failed"),
                        extra=extra,
                    )
                )

        return AssistantTurn(
            contents,
            finish_reason=completion.stop_reason,
            completion=completion,
        )

    def has_batch_support(self) -> bool:
        return True

    def batch_submit(
        self,
        conversations: list[list[Turn]],
        data_model: Optional[type[BaseModel]] = None,
    ):
        from anthropic import NotGiven

        requests: list["BatchRequest"] = []

        for i, turns in enumerate(conversations):
            api_kwargs = self._chat_perform_args(
                stream=False,
                turns=turns,
                tools={},
                data_model=data_model,
            )

            params: "MessageCreateParamsNonStreaming" = {
                "messages": api_kwargs.get("messages", {}),
                "model": self.model,
                "max_tokens": api_kwargs.get("max_tokens", 4096),
            }

            # If data_model, tools/tool_choice should be present (old API)
            # or output_format (new API)
            tools = api_kwargs.get("tools")
            tool_choice = api_kwargs.get("tool_choice")
            output_format = api_kwargs.get("output_format")
            if tools and not isinstance(tools, NotGiven):
                params["tools"] = tools
            if tool_choice and not isinstance(tool_choice, NotGiven):
                params["tool_choice"] = tool_choice
            if output_format and not isinstance(output_format, NotGiven):
                params["output_format"] = output_format  # type: ignore[typeddict-unknown-key]

            requests.append({"custom_id": f"request-{i}", "params": params})

        batch = self._client.messages.batches.create(requests=requests)
        return batch.model_dump()

    def batch_poll(self, batch):
        from anthropic.types.messages import MessageBatch

        batch = MessageBatch.model_validate(batch)
        b = self._client.messages.batches.retrieve(batch.id)
        return b.model_dump()

    def batch_status(self, batch) -> "BatchStatus":
        from anthropic.types.messages import MessageBatch

        batch = MessageBatch.model_validate(batch)
        status = batch.processing_status
        counts = batch.request_counts

        return BatchStatus(
            working=status != "ended",
            n_processing=counts.processing,
            n_succeeded=counts.succeeded,
            n_failed=counts.errored + counts.canceled + counts.expired,
        )

    # https://docs.anthropic.com/en/api/retrieving-message-batch-results
    def batch_retrieve(self, batch):
        from anthropic.types.messages import MessageBatch

        batch = MessageBatch.model_validate(batch)
        if batch.results_url is None:
            raise ValueError("Batch has no results URL")

        results: list[dict[str, Any]] = []
        for res in self._client.messages.batches.results(batch.id):
            results.append(res.model_dump())

        # Sort by custom_id to maintain order
        def extract_id(x: str):
            match = re.search(r"-(\d+)$", x)
            return int(match.group(1)) if match else 0

        results.sort(key=lambda x: extract_id(x.get("custom_id", "")))

        return results

    def batch_result_turn(self, result, has_data_model: bool = False):
        from anthropic.types.messages.message_batch_individual_response import (
            MessageBatchIndividualResponse,
        )

        result = MessageBatchIndividualResponse.model_validate(result)
        if result.result.type != "succeeded":
            # TODO: offer advice on what to do?
            warnings.warn(f"Batch request didn't succeed: {result.result}")
            return None

        message = result.result.message
        return self._as_turn(message, has_data_model)

    def _cache_control(self) -> "Optional[CacheControlEphemeralParam]":
        if self._cache == "none":
            return None
        return {
            "type": "ephemeral",
            "ttl": self._cache,
        }


def ChatBedrockAnthropic(
    *,
    model: Optional[str] = None,
    max_tokens: int = 4096,
    cache: Literal["5m", "1h", "none"] = "none",
    aws_secret_key: Optional[str] = None,
    aws_access_key: Optional[str] = None,
    aws_region: Optional[str] = None,
    aws_profile: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    base_url: Optional[str] = None,
    system_prompt: Optional[str] = None,
    kwargs: Optional["ChatBedrockClientArgs"] = None,
) -> Chat["SubmitInputArgs", Message]:
    """
    Chat with an AWS bedrock model.

    [AWS Bedrock](https://aws.amazon.com/bedrock/) provides a number of chat
    based models, including those Anthropic's
    [Claude](https://aws.amazon.com/bedrock/claude/).

    Prerequisites
    -------------

    ::: {.callout-note}
    ## AWS credentials

    Consider using the approach outlined in this guide to manage your AWS credentials:
    <https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html>
    :::

    ::: {.callout-note}
    ## Python requirements

    `ChatBedrockAnthropic`, requires the `anthropic` package with the `bedrock` extras:
    `pip install "chatlas[bedrock-anthropic]"`
    :::

    Examples
    --------

    ```python
    from chatlas import ChatBedrockAnthropic

    chat = ChatBedrockAnthropic(
        aws_profile="...",
        aws_region="us-east",
        aws_secret_key="...",
        aws_access_key="...",
        aws_session_token="...",
    )
    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    model
        The model to use for the chat.
    max_tokens
        Maximum number of tokens to generate before stopping.
    cache
        How long to cache inputs? Defaults to "5m" (five minutes).
        Set to "none" to disable caching or "1h" to cache for one hour.
        See the Caching section of `ChatAnthropic` for details.
    aws_secret_key
        The AWS secret key to use for authentication.
    aws_access_key
        The AWS access key to use for authentication.
    aws_region
        The AWS region to use. Defaults to the AWS_REGION environment variable.
        If that is not set, defaults to `'us-east-1'`.
    aws_profile
        The AWS profile to use.
    aws_session_token
        The AWS session token to use.
    base_url
        The base URL to use. Defaults to the ANTHROPIC_BEDROCK_BASE_URL
        environment variable. If that is not set, defaults to
        `f"https://bedrock-runtime.{aws_region}.amazonaws.com"`.
    system_prompt
        A system prompt to set the behavior of the assistant.
    kwargs
        Additional arguments to pass to the `anthropic.AnthropicBedrock()`
        client constructor.

    Troubleshooting
    ---------------

    If you encounter 400 or 403 errors when trying to use the model, keep the
    following in mind:

    ::: {.callout-note}
    #### Incorrect model name

    If the model name is completely incorrect, you'll see an error like
    `Error code: 400 - {'message': 'The provided model identifier is invalid.'}`

    Make sure the model name is correct and active in the specified region.
    :::

    ::: {.callout-note}
    #### Models are region specific

    If you encounter errors similar to `Error code: 403 - {'message': "You don't
    have access to the model with the specified model ID."}`, make sure your
    model is active in the relevant `aws_region`.

    Keep in mind, if `aws_region` is not specified, and AWS_REGION is not set,
    the region defaults to us-east-1, which may not match to your AWS config's
    default region.
    :::

    ::: {.callout-note}
    #### Cross region inference ID

    In some cases, even if you have the right model and the right region, you
    may still encounter an error like  `Error code: 400 - {'message':
    'Invocation of model ID anthropic.claude-3-5-sonnet-20240620-v1:0 with
    on-demand throughput isn't supported. Retry your request with the ID or ARN
    of an inference profile that contains this model.'}`

    In this case, you'll need to look up the 'cross region inference ID' for
    your model. This might required opening your `aws-console` and navigating to
    the 'Anthropic Bedrock' service page. From there, go to the 'cross region
    inference' tab and copy the relevant ID.

    For example, if the desired model ID is
    `anthropic.claude-3-5-sonnet-20240620-v1:0`, the cross region ID might look
    something like `us.anthropic.claude-3-5-sonnet-20240620-v1:0`.
    :::


    Returns
    -------
    Chat
        A Chat object.
    """

    if model is None:
        model = log_model_default("us.anthropic.claude-sonnet-4-5-20250929-v1:0")

    return Chat(
        provider=AnthropicBedrockProvider(
            model=model,
            max_tokens=max_tokens,
            cache=cache,
            aws_secret_key=aws_secret_key,
            aws_access_key=aws_access_key,
            aws_region=aws_region,
            aws_profile=aws_profile,
            aws_session_token=aws_session_token,
            base_url=base_url,
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
    )


class AnthropicBedrockProvider(AnthropicProvider):
    def __init__(
        self,
        *,
        model: str,
        aws_secret_key: str | None,
        aws_access_key: str | None,
        aws_region: str | None,
        aws_profile: str | None,
        aws_session_token: str | None,
        max_tokens: int = 4096,
        cache: Literal["5m", "1h", "none"] = "none",
        base_url: str | None,
        name: str = "AWS/Bedrock",
        kwargs: Optional["ChatBedrockClientArgs"] = None,
    ):
        super().__init__(
            name=name,
            model=model,
            max_tokens=max_tokens,
            cache=cache,
        )

        try:
            from anthropic import AnthropicBedrock, AsyncAnthropicBedrock
        except ImportError:
            raise ImportError(
                "`ChatBedrockAnthropic()` requires the `anthropic` package. "
                "Install it with `pip install anthropic[bedrock]`."
            )

        kwargs_full: "ChatBedrockClientArgs" = {
            "aws_secret_key": aws_secret_key,
            "aws_access_key": aws_access_key,
            "aws_region": aws_region,
            "aws_profile": aws_profile,
            "aws_session_token": aws_session_token,
            "base_url": base_url,
            **(kwargs or {}),
        }

        self._client = AnthropicBedrock(**kwargs_full)  # type: ignore
        self._async_client = AsyncAnthropicBedrock(**kwargs_full)  # type: ignore

    def list_models(self):
        # boto3 should come via anthropic's bedrock extras
        import boto3

        bedrock = boto3.client("bedrock")
        resp = bedrock.list_foundation_models()
        models = resp["modelSummaries"]

        res: list[ModelInfo] = []
        for m in models:
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
