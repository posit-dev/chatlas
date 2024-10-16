import json
from typing import TYPE_CHECKING, Literal, Optional, Union, cast, overload

from ._chat import Chat
from ._content import (
    Content,
    ContentImageInline,
    ContentImageRemote,
    ContentText,
    ContentToolRequest,
    ContentToolResult,
)
from ._provider import Provider, ToolDef
from ._tokens import tokens_log
from ._turn import Turn, normalize_turns
from ._utils import inform_model_default

if TYPE_CHECKING:
    from anthropic.types import (
        Message,
        MessageParam,
        RawMessageStreamEvent,
        TextBlock,
        ToolParam,
        ToolUseBlock,
    )
    from anthropic.types.image_block_param import ImageBlockParam
    from anthropic.types.model_param import ModelParam
    from anthropic.types.text_block_param import TextBlockParam
    from anthropic.types.tool_result_block_param import ToolResultBlockParam
    from anthropic.types.tool_use_block_param import ToolUseBlockParam

    from .types._anthropic_client import ProviderClientArgs
    from .types._anthropic_create import CreateCompletionArgs

    ContentBlockParam = Union[
        TextBlockParam,
        ImageBlockParam,
        ToolUseBlockParam,
        ToolResultBlockParam,
    ]
else:
    Message = object
    RawMessageStreamEvent = object


def ChatAnthropic(
    *,
    system_prompt: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
    model: "Optional[ModelParam]" = None,
    api_key: Optional[str] = None,
    max_tokens: int = 4096,
    kwargs: Optional["ProviderClientArgs"] = None,
) -> Chat["CreateCompletionArgs"]:
    """
    Chat with an Anthropic Claude model.

    Anthropic (https://www.anthropic.com) provides a number of chat based
    models under the Claude (https://www.anthropic.com/claude) moniker.

    Note that a Claude Prop membership does not give you the ability to call
    models via the API. You will need to go to the developer console
    (https://console.anthropic.com/account/keys) to sign up (and pay for)
    a developer account that will give you an API key that you can use with
    this package.

    Parameters
    ----------
    system_prompt
        A system prompt to set the behavior of the assistant.
    turns
        A list of turns to start the chat with (i.e., continuing a previous
        conversation). If not provided, the conversation begins from scratch.
        Do not provide non-None values for both `turns` and `system_prompt`.
        Each message in the list should be a dictionary with at least `role`
        (usually `system`, `user`, or `assistant`, but `tool` is also possible).
        Normally there is also a `content` field, which is a string.
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly choosing
        a model for all but the most casual use.
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `ANTHROPIC_API_KEY` environment variable.
    max_tokens
        Maximum number of tokens to generate before stopping.
    kwargs
        Additional arguments to pass to the `anthropic.Anthropic()` client constructor.

    Returns
    -------
    Chat
        A Chat object.
    """

    if model is None:
        model = inform_model_default("claude-3-5-sonnet-latest")

    return Chat(
        provider=ClaudeProvider(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            kwargs=kwargs,
        ),
        turns=normalize_turns(
            turns or [],
            system_prompt,
        ),
    )


class ClaudeProvider(Provider[Message, RawMessageStreamEvent, Message]):
    def __init__(
        self,
        *,
        max_tokens: int,
        model: str,
        api_key: str | None,
        kwargs: Optional["ProviderClientArgs"] = None,
    ):
        try:
            from anthropic import Anthropic, AsyncAnthropic
        except ImportError:
            raise ImportError(
                "`ChatAnthropic()` requires the `anthropic` package. "
                "You can install it with 'pip install anthropic'."
            )

        self._model = model
        self._max_tokens = max_tokens

        kwargs_full: "ProviderClientArgs" = {
            "api_key": api_key,
            **(kwargs or {}),
        }

        # TODO: worth bringing in sync types?
        self._client = Anthropic(**kwargs_full)  # type: ignore
        self._async_client = AsyncAnthropic(**kwargs_full)

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, ToolDef],
        kwargs: Optional["CreateCompletionArgs"] = None,
    ): ...

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, ToolDef],
        kwargs: Optional["CreateCompletionArgs"] = None,
    ): ...

    def chat_perform(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, ToolDef],
        kwargs: Optional["CreateCompletionArgs"] = None,
    ):
        kwargs = self._chat_perform_args(stream, turns, tools, kwargs)
        return self._client.messages.create(**kwargs)  # type: ignore

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, ToolDef],
        kwargs: Optional["CreateCompletionArgs"] = None,
    ): ...

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, ToolDef],
        kwargs: Optional["CreateCompletionArgs"] = None,
    ): ...

    async def chat_perform_async(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, ToolDef],
        kwargs: Optional["CreateCompletionArgs"] = None,
    ):
        kwargs = self._chat_perform_args(stream, turns, tools, kwargs)
        return await self._async_client.messages.create(**kwargs)  # type: ignore

    def _chat_perform_args(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, ToolDef],
        kwargs: Optional["CreateCompletionArgs"],
    ) -> "CreateCompletionArgs":
        tool_schemas = [self._get_tool_schema(tool) for tool in tools.values()]

        kwargs_full: "CreateCompletionArgs" = {
            "stream": stream,
            "messages": self._as_message_params(turns),
            "model": self._model,
            "max_tokens": self._max_tokens,
            "tools": tool_schemas,
            **(kwargs or {}),
        }

        if "system" not in kwargs_full:
            if len(turns) > 0 and turns[0].role == "system":
                kwargs_full["system"] = turns[0].text

        return kwargs_full

    def stream_text(self, chunk) -> Optional[str]:
        if chunk.type == "content_block_delta" and chunk.delta.type == "text_delta":
            return chunk.delta.text
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
                if not isinstance(this_content.input, str):
                    this_content.input = ""
                this_content.input += chunk.delta.partial_json
        elif chunk.type == "content_block_stop":
            this_content = completion.content[chunk.index]
            if this_content.type == "tool_use" and isinstance(this_content.input, str):
                try:
                    this_content.input = json.loads(this_content.input or "{}")
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON input: {e}")
        elif chunk.type == "message_delta":
            completion.stop_reason = chunk.delta.stop_reason
            completion.stop_sequence = chunk.delta.stop_sequence
            completion.usage.output_tokens = chunk.usage.output_tokens

        return completion

    def stream_turn(self, completion) -> Turn:
        return self._as_turn(completion)

    def value_turn(self, completion) -> Turn:
        return self._as_turn(completion)

    def _as_message_params(self, turns: list[Turn]) -> list["MessageParam"]:
        messages: list["MessageParam"] = []
        for turn in turns:
            if turn.role == "system":
                continue  # system prompt passed as separate arg
            if turn.role not in ["user", "assistant"]:
                raise ValueError(f"Unknown role {turn.role}")

            content = [self._as_content_block(c) for c in turn.contents]
            role = "user" if turn.role == "user" else "assistant"
            messages.append({"role": role, "content": content})
        return messages

    @staticmethod
    def _as_content_block(content: Content) -> "ContentBlockParam":
        if isinstance(content, ContentText):
            return {"text": content.text, "type": "text"}
        elif isinstance(content, ContentImageInline):
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": content.content_type,
                    "data": content.data or "",
                },
            }
        elif isinstance(content, ContentImageRemote):
            raise NotImplementedError(
                "Remote images aren't supported by Anthropic (Claude). "
                "Consider downloading the image and using content_image_file() instead."
            )
        elif isinstance(content, ContentToolRequest):
            return {
                "type": "tool_use",
                "id": content.id,
                "name": content.name,
                "input": content.arguments,
            }
        elif isinstance(content, ContentToolResult):
            content_ = (
                str(content.value) if content.value is not None else content.error
            )
            return {
                "type": "tool_result",
                "tool_use_id": content.id,
                "content": content_ or "",
                "is_error": content.error is not None,
            }
        raise ValueError(f"Unknown content type: {type(content)}")

    @staticmethod
    def _get_tool_schema(tool: ToolDef) -> "ToolParam":
        fn = tool.schema["function"]
        name = fn["name"]
        return {
            "name": name,
            "description": fn["description"],
            "input_schema": {
                "type": "object",
                "properties": fn["parameters"]["properties"],
            },
        }

    @staticmethod
    def _as_turn(completion: Message) -> Turn:
        contents = []
        for content in completion.content:
            if content.type == "text":
                contents.append(ContentText(content.text))
            elif content.type == "tool_use":
                # For some reason, the type is a general object?
                if not isinstance(content.input, dict):
                    raise ValueError(
                        f"Expected a dictionary of input arguments, got {type(content.input)}."
                    )
                contents.append(
                    ContentToolRequest(
                        content.id,
                        name=content.name,
                        arguments=content.input,
                    )
                )

        tokens = completion.usage.input_tokens, completion.usage.output_tokens

        tokens_log("Anthropic", tokens)

        return Turn("assistant", contents, tokens=tokens)
