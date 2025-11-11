from __future__ import annotations

import base64
import warnings
from typing import TYPE_CHECKING, Any, Optional, cast

import orjson
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
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
from ._merge import merge_dicts
from ._provider import StandardModelParamNames, StandardModelParams
from ._provider_openai_generic import BatchResult, OpenAIAbstractProvider
from ._tools import Tool, basemodel_to_param_schema
from ._turn import Turn
from ._utils import MISSING, MISSING_TYPE, is_testing

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam
    from openai.types.chat.chat_completion_assistant_message_param import (
        ContentArrayOfContentPart,
    )
    from openai.types.chat.chat_completion_content_part_param import (
        ChatCompletionContentPartParam,
    )
    from openai.types.chat_model import ChatModel

    from .types.openai import ChatClientArgs, SubmitInputArgs


# The dictionary form of ChatCompletion (TODO: stronger typing)?
ChatCompletionDict = dict[str, Any]


def ChatOpenAICompletions(
    *,
    base_url: str = "https://api.openai.com/v1",
    system_prompt: Optional[str] = None,
    model: "Optional[ChatModel | str]" = None,
    api_key: Optional[str] = None,
    seed: int | None | MISSING_TYPE = MISSING,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", ChatCompletion]:
    """
    Chat with an OpenAI-compatible model (via the Completions API).

    This function exists mainly for historical reasons; new code should
    prefer `ChatOpenAI()`, which uses the newer Responses API.

    This function may also be useful for using an "OpenAI-compatible model"
    hosted by another provider (e.g., vLLM, Ollama, etc.) that supports the
    OpenAI Completions API.
    """
    if isinstance(seed, MISSING_TYPE):
        seed = 1014 if is_testing() else None

    if model is None:
        model = log_model_default("gpt-4.1")

    return Chat(
        provider=OpenAICompletionsProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            seed=seed,
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
    )


class OpenAICompletionsProvider(
    OpenAIAbstractProvider[
        ChatCompletion,
        ChatCompletionChunk,
        ChatCompletionDict,
        "SubmitInputArgs",
    ]
):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        name: str = "OpenAI",
        seed: int | None = None,
        kwargs: Optional["ChatClientArgs"] = None,
    ):
        super().__init__(
            api_key=api_key,
            model=model,
            base_url=base_url,
            name=name,
            kwargs=kwargs,
        )
        self._seed = seed

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
        return self._client.chat.completions.create(**kwargs)  # type: ignore

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
        return await self._async_client.chat.completions.create(**kwargs)  # type: ignore

    def _chat_perform_args(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ) -> "SubmitInputArgs":
        tool_schemas = [tool.schema for tool in tools.values()]

        kwargs_full: "SubmitInputArgs" = {
            "stream": stream,
            "messages": self._turns_as_inputs(turns),
            "model": self.model,
            **(kwargs or {}),
        }

        if self._seed is not None:
            kwargs_full["seed"] = self._seed

        if tool_schemas:
            kwargs_full["tools"] = tool_schemas

        if data_model is not None:
            params = basemodel_to_param_schema(data_model)
            params = cast(dict, params)
            params["additionalProperties"] = False
            kwargs_full["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_data",
                    "description": params.get("description", ""),
                    "schema": params,
                    "strict": True,
                },
            }
            # Apparently OpenAI gets confused if you include
            # both response_format and tools
            if "tools" in kwargs_full:
                del kwargs_full["tools"]

        if stream and "stream_options" not in kwargs_full:
            kwargs_full["stream_options"] = {"include_usage": True}

        return kwargs_full

    def stream_text(self, chunk):
        if not chunk.choices:
            return None
        return chunk.choices[0].delta.content

    def stream_merge_chunks(self, completion, chunk):
        chunkd = chunk.model_dump()
        if completion is None:
            return chunkd
        return merge_dicts(completion, chunkd)

    def stream_turn(self, completion, has_data_model):
        delta = completion["choices"][0].pop("delta")
        completion["choices"][0]["message"] = delta
        completion = ChatCompletion.construct(**completion)
        return self._response_as_turn(completion, has_data_model)

    def value_turn(self, completion, has_data_model):
        return self._response_as_turn(completion, has_data_model)

    def value_tokens(self, completion):
        usage = completion.usage
        if usage is None:
            # For some reason ChatGroq() includes tokens under completion.x_groq
            # Groq does not support caching, so we set cached_tokens to 0
            if hasattr(completion, "x_groq"):
                usage = completion.x_groq["usage"]  # type: ignore
                return usage["prompt_tokens"], usage["completion_tokens"], 0
            else:
                return None

        if usage.prompt_tokens_details is not None:
            cached_tokens = (
                usage.prompt_tokens_details.cached_tokens
                if usage.prompt_tokens_details.cached_tokens
                else 0
            )
        else:
            cached_tokens = 0

        return (
            usage.prompt_tokens - cached_tokens,
            usage.completion_tokens,
            cached_tokens,
        )

    @staticmethod
    def _turns_as_inputs(turns: list[Turn]) -> list["ChatCompletionMessageParam"]:
        res: list["ChatCompletionMessageParam"] = []
        for turn in turns:
            if turn.role == "system":
                res.append(
                    ChatCompletionSystemMessageParam(content=turn.text, role="system")
                )
            elif turn.role == "assistant":
                content_parts: list["ContentArrayOfContentPart"] = []
                tool_calls: list["ChatCompletionMessageToolCallParam"] = []
                for x in turn.contents:
                    if isinstance(x, ContentText):
                        content_parts.append({"type": "text", "text": x.text})
                    elif isinstance(x, ContentJson):
                        text = orjson.dumps(x.value).decode("utf-8")
                        content_parts.append({"type": "text", "text": text})
                    elif isinstance(x, ContentToolRequest):
                        tool_calls.append(
                            {
                                "id": x.id,
                                "function": {
                                    "name": x.name,
                                    "arguments": orjson.dumps(x.arguments).decode(
                                        "utf-8"
                                    ),
                                },
                                "type": "function",
                            }
                        )
                    else:
                        raise ValueError(
                            f"Don't know how to handle content type {type(x)} for role='assistant'."
                        )

                # Some OpenAI-compatible models (e.g., Groq) don't work nicely with empty content
                args = {
                    "role": "assistant",
                    "content": content_parts,
                    "tool_calls": tool_calls,
                }
                if not content_parts:
                    del args["content"]
                if not tool_calls:
                    del args["tool_calls"]

                res.append(ChatCompletionAssistantMessageParam(**args))

            elif turn.role == "user":
                contents: list["ChatCompletionContentPartParam"] = []
                tool_results: list["ChatCompletionToolMessageParam"] = []
                for x in turn.contents:
                    if isinstance(x, ContentText):
                        contents.append({"type": "text", "text": x.text})
                    elif isinstance(x, ContentJson):
                        text = orjson.dumps(x.value).decode("utf-8")
                        contents.append({"type": "text", "text": text})
                    elif isinstance(x, ContentPDF):
                        contents.append(
                            {
                                "type": "file",
                                "file": {
                                    "filename": x.filename,
                                    "file_data": (
                                        "data:application/pdf;base64,"
                                        f"{base64.b64encode(x.data).decode('utf-8')}"
                                    ),
                                },
                            }
                        )
                    elif isinstance(x, ContentImageRemote):
                        contents.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": x.url,
                                    "detail": x.detail,
                                },
                            }
                        )
                    elif isinstance(x, ContentImageInline):
                        contents.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{x.image_content_type};base64,{x.data}"
                                },
                            }
                        )
                    elif isinstance(x, ContentToolResult):
                        if isinstance(
                            x, (ContentToolResultImage, ContentToolResultResource)
                        ):
                            raise NotImplementedError(
                                "OpenAI does not support tool results with images or resources."
                            )
                        tool_results.append(
                            ChatCompletionToolMessageParam(
                                # Currently, OpenAI only allows for text content in tool results
                                content=cast(str, x.get_model_value()),
                                tool_call_id=x.id,
                                role="tool",
                            )
                        )
                    else:
                        raise ValueError(
                            f"Don't know how to handle content type {type(x)} for role='user'."
                        )

                if contents:
                    res.append(
                        ChatCompletionUserMessageParam(content=contents, role="user")
                    )
                res.extend(tool_results)

            else:
                raise ValueError(f"Unknown role: {turn.role}")

        return res

    @staticmethod
    def _response_as_turn(
        completion: "ChatCompletion", has_data_model: bool
    ) -> Turn[ChatCompletion]:
        message = completion.choices[0].message

        contents: list[Content] = []
        if message.content is not None:
            if has_data_model:
                data = message.content
                # Some providers (e.g., Cloudflare) may already provide a dict
                if not isinstance(data, dict):
                    data = orjson.loads(data)
                contents = [ContentJson(value=data)]
            else:
                contents = [ContentText(text=message.content)]

        tool_calls = message.tool_calls

        if tool_calls is not None:
            for call in tool_calls:
                if call.type != "function":
                    continue
                func = call.function
                if func is None:
                    continue
                args = load_tool_request_args(func.arguments, func.name)
                contents.append(
                    ContentToolRequest(
                        id=call.id,
                        name=func.name,
                        arguments=args,
                    )
                )

        return Turn(
            "assistant",
            contents,
            finish_reason=completion.choices[0].finish_reason,
            completion=completion,
        )

    def translate_model_params(self, params: StandardModelParams) -> "SubmitInputArgs":
        res: "SubmitInputArgs" = {}
        if "temperature" in params:
            res["temperature"] = params["temperature"]

        if "top_p" in params:
            res["top_p"] = params["top_p"]

        if "frequency_penalty" in params:
            res["frequency_penalty"] = params["frequency_penalty"]

        if "presence_penalty" in params:
            res["presence_penalty"] = params["presence_penalty"]

        if "seed" in params:
            res["seed"] = params["seed"]

        if "max_tokens" in params:
            res["max_tokens"] = params["max_tokens"]

        if "log_probs" in params:
            res["logprobs"] = params["log_probs"]

        if "stop_sequences" in params:
            res["stop"] = params["stop_sequences"]

        return res

    def supported_model_params(self) -> set[StandardModelParamNames]:
        return {
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "max_tokens",
            "log_probs",
            "stop_sequences",
        }

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

        completion = ChatCompletion.construct(**response.body)
        return self._response_as_turn(completion, has_data_model)

    @staticmethod
    def _batch_endpoint():
        return "/v1/chat/completions"


def load_tool_request_args(args: str, name: str) -> dict[str, Any]:
    """Load tool request arguments from a JSON string."""
    res = {}
    try:
        res = orjson.loads(args) if args else {}
    except orjson.JSONDecodeError:
        raise ValueError(
            f"The model's completion included a tool request ({name}) "
            "with invalid JSON for input arguments: '{args}'"
            "This can happen if the model hallucinates parameters not defined by "
            "your function schema. Try revising your tool description and system "
            "prompt to be more specific about the expected input arguments to this function."
        )
    return res
