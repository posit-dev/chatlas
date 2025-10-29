from __future__ import annotations

import base64
import warnings
from typing import TYPE_CHECKING, Literal, Optional, cast

import orjson
from openai.types.responses import Response, ResponseStreamEvent
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
    ContentToolResult,
)
from ._logging import log_model_default
from ._provider import StandardModelParamNames, StandardModelParams
from ._provider_openai_completions import load_tool_request_args
from ._provider_openai_generic import BatchResult, OpenAIAbstractProvider
from ._tools import Tool, basemodel_to_param_schema
from ._turn import Turn

if TYPE_CHECKING:
    from openai.types.responses import (
        ResponseInputContentParam,
        ResponseInputItemParam,
        ResponseReasoningItemParam,
    )
    from openai.types.responses.easy_input_message_param import EasyInputMessageParam
    from openai.types.responses.tool_param import ToolParam
    from openai.types.shared_params.responses_model import ResponsesModel

    from .types.openai import ChatClientArgs
    from .types.openai import ResponsesSubmitInputArgs as SubmitInputArgs

Role = Literal["user", "assistant", "system"]


def ChatOpenAI(
    *,
    system_prompt: Optional[str] = None,
    model: "Optional[ResponsesModel | str]" = None,
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", Response]:
    """
    Chat with an OpenAI model using the responses API.

    [OpenAI](https://openai.com/) provides a number of chat-based models,
    mostly under the [ChatGPT](https://chat.openai.com/) brand.

    Prerequisites
    --------------

    ::: {.callout-note}
    ## API key

    Note that a ChatGPT Plus membership does not give you the ability to call
    models via the API. You will need to go to the [developer
    platform](https://platform.openai.com) to sign up (and pay for) a developer
    account that will give you an API key that you can use with this package.
    :::

    Examples
    --------
    ```python
    import os
    from chatlas import ChatOpenAI

    chat = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
        this directly, but instead set the `OPENAI_API_KEY` environment
        variable.
    base_url
        The base URL to the endpoint; the default uses OpenAI.
    kwargs
        Additional arguments to pass to the `openai.OpenAI()` client
        constructor.

    Returns
    -------
    Chat
        A chat object that retains the state of the conversation.

    Note
    ----
    The responses API does not support the `seed` parameter. If you need
    reproducible output, use [](`~chatlas.ChatOpenAI`) instead.

    Pasting an API key into a chat constructor (e.g., `ChatOpenAI(api_key="...")`)
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
    OPENAI_API_KEY=...
    ```

    ```python
    from chatlas import ChatOpenAI
    from dotenv import load_dotenv

    load_dotenv()
    chat = ChatOpenAI()
    chat.console()
    ```

    Another, more general, solution is to load your environment variables into the shell
    before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

    ```shell
    export OPENAI_API_KEY=...
    ```
    """
    if model is None:
        model = log_model_default("gpt-4.1")

    return Chat(
        provider=OpenAIProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            # Note: We don't pass seed to the parent since the responses API doesn't support it
            seed=None,
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
    )


class OpenAIProvider(
    OpenAIAbstractProvider[
        Response,
        ResponseStreamEvent,
        Response,
        "SubmitInputArgs",
    ]
):
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
        return self._client.responses.create(**kwargs)  # type: ignore

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
        return await self._async_client.responses.create(**kwargs)  # type: ignore

    def _chat_perform_args(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ) -> "SubmitInputArgs":
        kwargs_full: "SubmitInputArgs" = {
            "stream": stream,
            "input": self._turns_as_inputs(turns),
            "model": self.model,
            "store": False,
            **(kwargs or {}),
        }

        tool_schemas = [tool.schema for tool in tools.values()]
        if tool_schemas:
            # Convert completion tool format to responses format
            responses_tools: list["ToolParam"] = []
            for schema in tool_schemas:
                func = schema["function"]
                responses_tools.append(
                    {
                        "type": "function",
                        "name": func["name"],
                        "description": func.get("description", None),
                        "parameters": func.get("parameters", None),
                        "strict": func.get("strict", True),
                    }
                )
            if responses_tools:
                kwargs_full["tools"] = responses_tools

        # Add structured data extraction if present
        if data_model is not None:
            params = basemodel_to_param_schema(data_model)
            params = cast(dict, params)
            params["additionalProperties"] = False
            kwargs_full["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "structured_data",
                    "schema": params,
                    "strict": True,
                }
            }

        # Request reasoning content for reasoning models
        include = []
        if self._is_reasoning_model():
            include.append("reasoning.encrypted_content")

        if "log_probs" in kwargs_full:
            include.append("message.output_text.logprobs")

        if include:
            kwargs_full["include"] = include

        return kwargs_full

    def stream_text(self, chunk):
        if chunk.type == "response.output_text.delta":
            return chunk.delta
        return None

    def stream_merge_chunks(self, completion, chunk):
        if chunk.type == "response.completed":
            return chunk.response
        # Since this value won't actually be used, we can lie about the type
        return cast(Response, None)

    def stream_turn(self, completion, has_data_model):
        return self._response_as_turn(completion, has_data_model)

    def value_turn(self, completion, has_data_model):
        return self._response_as_turn(completion, has_data_model)

    def value_tokens(self, completion):
        usage = completion.usage
        if usage is None:
            return None
        cached_tokens = usage.input_tokens_details.cached_tokens
        return (
            usage.input_tokens - cached_tokens,
            usage.output_tokens,
            cached_tokens,
        )

    def batch_result_turn(
        self,
        result,
        has_data_model: bool = False,
    ) -> Turn | None:
        response = BatchResult.model_validate(result).response
        if response.status_code != 200:
            # TODO: offer advice on what to do?
            warnings.warn(f"Batch request failed: {response.body}")
            return None

        completion = Response.construct(**response.body)
        return self._response_as_turn(completion, has_data_model)

    @staticmethod
    def _response_as_turn(completion: Response, has_data_model: bool) -> Turn:
        contents: list[Content] = []
        for output in completion.output:
            if output.type == "message":
                for x in output.content:
                    # TODO: handle refusals?
                    if x.type != "output_text":
                        continue
                    if has_data_model:
                        data = orjson.loads(x.text)
                        contents.append(ContentJson(value=data))
                    else:
                        contents.append(ContentText(text=x.text))

            elif output.type == "function_call":
                args = load_tool_request_args(output.arguments, output.name)
                contents.append(
                    ContentToolRequest(
                        id=output.id or "_missing_id_",
                        name=output.name,
                        arguments=args,
                    )
                )

            elif output.type == "reasoning":
                if output.content:
                    thinking = "".join(x.text for x in output.content)
                    contents.append(
                        ContentThinking(
                            thinking=thinking,
                            extra=output.model_dump(),
                        )
                    )
            else:
                raise ValueError(f"Unknown output type: {output.type}")

        return Turn(
            "assistant",
            contents,
            completion=completion,
        )

    def _is_reasoning_model(self) -> bool:
        # https://platform.openai.com/docs/models/compare
        return self.model.startswith("o") or self.model.startswith("gpt-5")

    @staticmethod
    def _turns_as_inputs(turns: list[Turn]) -> "list[ResponseInputItemParam]":
        res: "list[ResponseInputItemParam]" = []
        for turn in turns:
            res.extend([as_input_param(x, turn.role) for x in turn.contents])
        return res

    def translate_model_params(self, params: StandardModelParams) -> "SubmitInputArgs":
        res: "SubmitInputArgs" = {}
        if "temperature" in params:
            res["temperature"] = params["temperature"]

        if "top_p" in params:
            res["top_p"] = params["top_p"]

        if "max_tokens" in params:
            res["max_output_tokens"] = params["max_tokens"]

        if "log_probs" in params:
            # This isn't a formal submit argument, but we use it internally to
            # determine whether to include `message.output_text.logprobs`
            res["log_probs"] = params["log_probs"]  # type: ignore

        if "top_k" in params:
            res["top_logprobs"] = params["top_k"]

        return res

    def supported_model_params(self) -> set[StandardModelParamNames]:
        return {
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "log_probs",
        }

    @staticmethod
    def _batch_endpoint():
        return "/v1/responses"


def as_input_param(content: Content, role: Role) -> "ResponseInputItemParam":
    if isinstance(content, ContentText):
        if role == "assistant":
            # OpenAI's type for this value (ResponseOutputMessageParam) currently has a bunch
            # of fields marked as Required that probably shouldn't be?
            # When that gets fixed, this can be updated to be simpler (i.e., as_message() call)
            return {
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content.text,
                        "annotations": [],
                    }
                ],
                "status": "completed",
                "type": "message",
                "id": "msg_missing_id",  # Not sure if it matters if we have a fake id here?
            }
        else:
            return as_message({"type": "input_text", "text": content.text}, role)
    elif isinstance(content, ContentJson):
        return as_message(
            {
                "type": "input_text",
                "text": orjson.dumps(content.value).decode("utf-8"),
            },
            role,
        )
    elif isinstance(content, ContentImageRemote):
        return as_message(
            {
                "type": "input_image",
                "image_url": content.url,
                "detail": content.detail,
            },
            role,
        )
    elif isinstance(content, ContentImageInline):
        return as_message(
            {
                "type": "input_image",
                "image_url": f"data:{content.image_content_type};base64,{content.data}",
                "detail": "auto",
            },
            role,
        )
    elif isinstance(content, ContentPDF):
        return as_message(
            {
                "type": "input_file",
                # TODO: populate filename if we have it
                "filename": "",
                "file_data": f"data:application/pdf;base64,{base64.b64encode(content.data).decode('utf-8')}",
            },
            role,
        )
    elif isinstance(content, ContentThinking):
        return cast("ResponseReasoningItemParam", content.extra)
    elif isinstance(content, ContentToolResult):
        return {
            "type": "function_call_output",
            "call_id": content.id,
            "output": cast(str, content.get_model_value()),
        }
    elif isinstance(content, ContentToolRequest):
        return {
            "type": "function_call",
            "call_id": content.id,
            "name": content.name,
            "arguments": orjson.dumps(content.arguments).decode("utf-8"),
        }
    else:
        raise ValueError(f"Unsupported content type: {type(content)}")


def as_message(x: "ResponseInputContentParam", role: Role) -> "EasyInputMessageParam":
    return {"role": role, "content": [x]}
