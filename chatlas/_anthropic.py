from __future__ import annotations

import base64
import warnings
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast, overload

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
    ContentToolRequest,
    ContentToolResult,
    ContentToolResultImage,
    ContentToolResultResource,
)
from ._logging import log_model_default
from ._provider import Provider, StandardModelParamNames, StandardModelParams
from ._tokens import tokens_log
from ._tools import Tool, basemodel_to_param_schema
from ._turn import Turn, user_turn
from ._utils import split_http_client_kwargs

if TYPE_CHECKING:
    from anthropic.types import (
        Message,
        MessageParam,
        RawMessageStreamEvent,
        TextBlock,
        ToolParam,
        ToolUseBlock,
    )
    from anthropic.types.document_block_param import DocumentBlockParam
    from anthropic.types.image_block_param import ImageBlockParam
    from anthropic.types.model_param import ModelParam
    from anthropic.types.text_block_param import TextBlockParam
    from anthropic.types.tool_result_block_param import ToolResultBlockParam
    from anthropic.types.tool_use_block_param import ToolUseBlockParam
    from openai.types.chat import ChatCompletionToolParam

    from .types.anthropic import ChatBedrockClientArgs, ChatClientArgs, SubmitInputArgs

    ContentBlockParam = Union[
        TextBlockParam,
        ImageBlockParam,
        ToolUseBlockParam,
        ToolResultBlockParam,
        DocumentBlockParam,
    ]
else:
    Message = object
    RawMessageStreamEvent = object


def ChatAnthropic(
    *,
    system_prompt: Optional[str] = None,
    model: "Optional[ModelParam]" = None,
    api_key: Optional[str] = None,
    max_tokens: int = 4096,
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
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `ANTHROPIC_API_KEY` environment
        variable.
    max_tokens
        Maximum number of tokens to generate before stopping.
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
    """

    if model is None:
        model = log_model_default("claude-3-7-sonnet-latest")

    return Chat(
        provider=AnthropicProvider(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
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

        kwargs_full: "ChatClientArgs" = {
            "api_key": api_key,
            **(kwargs or {}),
        }

        sync_kwargs, async_kwargs = split_http_client_kwargs(kwargs_full)

        # TODO: worth bringing in sync types?
        self._client = Anthropic(**sync_kwargs)  # type: ignore
        self._async_client = AsyncAnthropic(**async_kwargs)

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ): ...

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ): ...

    def chat_perform(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ):
        kwargs = self._chat_perform_args(stream, turns, tools, data_model, kwargs)
        return self._client.messages.create(**kwargs)  # type: ignore

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ): ...

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ): ...

    async def chat_perform_async(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ):
        kwargs = self._chat_perform_args(stream, turns, tools, data_model, kwargs)
        return await self._async_client.messages.create(**kwargs)  # type: ignore

    def _chat_perform_args(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ) -> "SubmitInputArgs":
        tool_schemas = [
            self._anthropic_tool_schema(tool.schema) for tool in tools.values()
        ]

        # If data extraction is requested, add a "mock" tool with parameters inferred from the data model
        data_model_tool: Tool | None = None
        if data_model is not None:

            def _structured_tool_call(**kwargs: Any):
                """Extract structured data"""
                pass

            data_model_tool = Tool.from_func(_structured_tool_call)

            data_model_tool.schema["function"]["parameters"] = {
                "type": "object",
                "properties": {
                    "data": basemodel_to_param_schema(data_model),
                },
            }

            tool_schemas.append(self._anthropic_tool_schema(data_model_tool.schema))

            if stream:
                stream = False
                warnings.warn(
                    "Anthropic does not support structured data extraction in streaming mode.",
                    stacklevel=2,
                )

        kwargs_full: "SubmitInputArgs" = {
            "stream": stream,
            "messages": self._as_message_params(turns),
            "model": self.model,
            "max_tokens": self._max_tokens,
            "tools": tool_schemas,
            **(kwargs or {}),
        }

        if data_model_tool:
            kwargs_full["tool_choice"] = {
                "type": "tool",
                "name": data_model_tool.name,
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
                    this_content.input = orjson.loads(this_content.input or "{}")
                except orjson.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON input: {e}")
        elif chunk.type == "message_delta":
            completion.stop_reason = chunk.delta.stop_reason
            completion.stop_sequence = chunk.delta.stop_sequence
            completion.usage.output_tokens = chunk.usage.output_tokens

        return completion

    def stream_turn(self, completion, has_data_model) -> Turn:
        return self._as_turn(completion, has_data_model)

    def value_turn(self, completion, has_data_model) -> Turn:
        return self._as_turn(completion, has_data_model)

    def token_count(
        self,
        *args: Content | str,
        tools: dict[str, Tool],
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
        tools: dict[str, Tool],
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
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
    ) -> dict[str, Any]:
        turn = user_turn(*args)

        kwargs = self._chat_perform_args(
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

        return {arg: kwargs[arg] for arg in args_to_keep if arg in kwargs}

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
        elif isinstance(content, ContentJson):
            return {"text": "<structured data/>", "type": "text"}
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
                "input": content.arguments,
            }
        elif isinstance(content, ContentToolResult):
            res: ToolResultBlockParam = {
                "type": "tool_result",
                "tool_use_id": content.id,
                "is_error": content.error is not None,
            }

            if isinstance(content, ContentToolResultImage):
                res["content"] = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": content.mime_type,
                            "data": content.value,
                        },
                    }
                ]
            elif isinstance(content, ContentToolResultResource):
                raise NotImplementedError(
                    "ContentToolResultResource is not currently supported by Anthropic."
                )
            else:
                # Anthropic supports non-text contents like ImageBlockParam
                res["content"] = content.get_model_value()  # type: ignore

            return res

        raise ValueError(f"Unknown content type: {type(content)}")

    @staticmethod
    def _anthropic_tool_schema(schema: "ChatCompletionToolParam") -> "ToolParam":
        fn = schema["function"]
        name = fn["name"]

        res: "ToolParam" = {
            "name": name,
            "input_schema": {
                "type": "object",
            },
        }

        if "description" in fn:
            res["description"] = fn["description"]

        if "parameters" in fn:
            res["input_schema"]["properties"] = fn["parameters"]["properties"]

        return res

    def _as_turn(self, completion: Message, has_data_model=False) -> Turn:
        contents = []
        for content in completion.content:
            if content.type == "text":
                contents.append(ContentText(text=content.text))
            elif content.type == "tool_use":
                if has_data_model and content.name == "_structured_tool_call":
                    if not isinstance(content.input, dict):
                        raise ValueError(
                            "Expected data extraction tool to return a dictionary."
                        )
                    if "data" not in content.input:
                        raise ValueError(
                            "Expected data extraction tool to return a 'data' field."
                        )
                    contents.append(ContentJson(value=content.input["data"]))
                else:
                    contents.append(
                        ContentToolRequest(
                            id=content.id,
                            name=content.name,
                            arguments=content.input,
                        )
                    )

        tokens = completion.usage.input_tokens, completion.usage.output_tokens

        tokens_log(self, tokens)

        return Turn(
            "assistant",
            contents,
            tokens=tokens,
            finish_reason=completion.stop_reason,
            completion=completion,
        )


def ChatBedrockAnthropic(
    *,
    model: Optional[str] = None,
    max_tokens: int = 4096,
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
        # Default model from https://github.com/anthropics/anthropic-sdk-python?tab=readme-ov-file#aws-bedrock
        model = log_model_default("anthropic.claude-3-5-sonnet-20241022-v2:0")

    return Chat(
        provider=AnthropicBedrockProvider(
            model=model,
            max_tokens=max_tokens,
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
        base_url: str | None,
        name: str = "AnthropicBedrock",
        kwargs: Optional["ChatBedrockClientArgs"] = None,
    ):
        super().__init__(name=name, model=model, max_tokens=max_tokens)

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
