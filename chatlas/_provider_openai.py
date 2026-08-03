from __future__ import annotations

import base64
import mimetypes
import warnings
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, Optional, cast
from urllib.parse import urlparse

import orjson
from openai.types.responses import Response, ResponseStreamEvent
from openai.types.responses.response_function_web_search import (
    ActionFind,
    ActionOpenPage,
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_output_text import AnnotationURLCitation
from pydantic import BaseModel

from ._chat import Chat
from ._content import (
    PROVIDER_ANNOTATION_TYPES,
    Content,
    ContentCitation,
    ContentDocument,
    ContentImageInline,
    ContentImageRemote,
    ContentJson,
    ContentPDF,
    ContentText,
    ContentThinking,
    ContentThinkingDelta,
    ContentToolRequest,
    ContentToolRequestFetch,
    ContentToolRequestSearch,
    ContentToolResult,
    ContentUploaded,
    ProviderAnnotation,
    WebSource,
    check_image_content_type_supported,
)
from ._content_file import ensure_bytes
from ._files import FileMetadata, maybe_write, open_binary
from ._logging import log_model_default
from ._provider import StandardModelParamNames, StandardModelParams
from ._provider_openai_completions import load_tool_request_args
from ._provider_openai_generic import BatchResult, OpenAIAbstractProvider
from ._tools import Tool, ToolBuiltIn, basemodel_to_param_schema
from ._tools_builtin import ToolWebSearch
from ._turn import AssistantTurn, FinishReason, Turn, check_finish_reason

if TYPE_CHECKING:
    import os
    from typing import IO

    from openai.types.file_object import FileObject
    from openai.types.responses import (
        ResponseInputContentParam,
        ResponseInputFileParam,
        ResponseInputItemParam,
        ResponseReasoningItemParam,
    )
    from openai.types.responses.easy_input_message_param import EasyInputMessageParam
    from openai.types.responses.tool_param import ToolParam
    from openai.types.shared.reasoning_effort import ReasoningEffort
    from openai.types.shared_params.reasoning import Reasoning
    from openai.types.shared_params.responses_model import ResponsesModel

    from ._turn import Role
    from .types.openai import ChatClientArgs
    from .types.openai import ResponsesSubmitInputArgs as SubmitInputArgs


def ChatOpenAI(
    *,
    system_prompt: Optional[str] = None,
    model: "Optional[ResponsesModel | str]" = None,
    base_url: str = "https://api.openai.com/v1",
    reasoning: "Optional[ReasoningEffort | Reasoning]" = None,
    service_tier: Optional[
        Literal["auto", "default", "flex", "scale", "priority"]
    ] = None,
    api_key: Optional[str] = None,
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
    base_url
        The base URL to the endpoint; the default uses OpenAI. Since
        `ChatOpenAI()` uses the Responses API, most third-party backends
        won't work here. For OpenAI-compatible backends (e.g., vLLM,
        Ollama, LiteLLM), use [](`~chatlas.ChatOpenAICompletions`) instead.
    reasoning
        The reasoning effort to use (for reasoning-capable models like the o and
        gpt-5 series).
    service_tier
        Request a specific service tier. Options:
        - `"auto"` (default): uses the service tier configured in Project settings.
        - `"default"`: standard pricing and performance.
        - `"flex"`: slower and cheaper.
        - `"scale"`: batch-like pricing for high-volume use.
        - `"priority"`: faster and more expensive.
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `OPENAI_API_KEY` environment
        variable.
    kwargs
        Additional arguments to pass to the `openai.OpenAI()` client
        constructor.

    Returns
    -------
    Chat
        A chat object that retains the state of the conversation.

    Note
    ----
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

    Note
    ----
    The responses API does not support the `seed` parameter. If you need
    reproducible output, use [](`~chatlas.ChatOpenAICompletions`) instead.
    """
    check_base_url(base_url)

    if model is None:
        model = log_model_default("gpt-5.6-terra")

    kwargs_chat: "SubmitInputArgs" = {}

    if reasoning is not None:
        if not is_reasoning_model(model):
            warnings.warn(f"Model {model} is not reasoning-capable", UserWarning)
        if isinstance(reasoning, str):
            reasoning = {"effort": reasoning, "summary": "auto"}
        kwargs_chat["reasoning"] = reasoning

    if service_tier is not None:
        kwargs_chat["service_tier"] = service_tier

    return Chat(
        provider=OpenAIProvider(
            api_key=api_key,
            model=model,
            base_url=base_url,
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
        kwargs_chat=kwargs_chat,
    )


# https://platform.openai.com/docs/api-reference/responses/get
_OPENAI_INCOMPLETE_REASON_MAP: dict[str, FinishReason] = {
    "max_output_tokens": "max_tokens",
    "content_filter": "content_filter",
}


def normalize_finish_reason(
    status: str | None, incomplete_reason: str | None = None
) -> str | None:
    if status is None:
        return None
    if status == "completed":
        return "success"
    if status != "incomplete":
        return status
    reason = incomplete_reason or status
    return _OPENAI_INCOMPLETE_REASON_MAP.get(reason, reason)


class OpenAIProvider(
    OpenAIAbstractProvider[
        Response,
        ResponseStreamEvent,
        Response,
        "SubmitInputArgs",
    ]
):
    supported_builtin_tools = (ToolWebSearch,)

    def chat_perform(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
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
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ):
        kwargs = self._chat_perform_args(stream, turns, tools, data_model, kwargs)
        return await self._async_client.responses.create(**kwargs)  # type: ignore

    def _chat_perform_args(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
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

        tool_params: list["ToolParam"] = []
        for tool in tools.values():
            if isinstance(tool, ToolBuiltIn):
                self.check_builtin_tool_support(tool)

            if isinstance(tool, ToolWebSearch):
                tool_params.append(tool.get_definition("openai"))
            elif isinstance(tool, ToolBuiltIn):
                tool_params.append(cast("ToolParam", tool.definition))
            else:
                schema = tool.schema
                func = schema["function"]
                tool_params.append(
                    {
                        "type": "function",
                        "name": func["name"],
                        "description": func.get("description", None),
                        "parameters": func.get("parameters", None),
                        "strict": func.get("strict", True),
                    }
                )

        if tool_params:
            kwargs_full["tools"] = tool_params

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
        if "reasoning" in kwargs_full or is_reasoning_model(self.model):
            include.append("reasoning.encrypted_content")

        if "log_probs" in kwargs_full:
            include.append("message.output_text.logprobs")
            # Remove from kwargs since it's not a formal argument
            kwargs_full.pop("log_probs")

        if include:
            kwargs_full["include"] = include

        return kwargs_full

    def token_count(
        self,
        turns: list[Turn],
        *,
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]],
    ) -> int:
        kwargs = self._token_count_args(turns, tools=tools, data_model=data_model)
        res = self._client.responses.input_tokens.count(**kwargs)
        return res.input_tokens

    async def token_count_async(
        self,
        turns: list[Turn],
        *,
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]],
    ) -> int:
        kwargs = self._token_count_args(turns, tools=tools, data_model=data_model)
        res = await self._async_client.responses.input_tokens.count(**kwargs)
        return res.input_tokens

    def _token_count_args(
        self,
        turns: list[Turn],
        *,
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]],
    ) -> dict[str, Any]:
        kwargs = self._chat_perform_args(
            stream=False,
            turns=turns,
            tools=tools,
            data_model=data_model,
        )
        # `input_tokens` accepts a subset of `responses.create` params; drop the
        # rest (e.g. stream/store/include) which the endpoint rejects.
        args_to_keep = ["input", "model", "tools", "text", "tool_choice", "reasoning"]
        return {arg: kwargs[arg] for arg in args_to_keep if arg in kwargs}

    def stream_content(self, chunk, completion) -> list[Content]:
        if chunk.type == "response.output_text.delta":
            # https://platform.openai.com/docs/api-reference/responses-streaming/response/output_text/delta
            return [ContentText.model_construct(text=chunk.delta)]
        if chunk.type == "response.output_text.annotation.added":
            # https://platform.openai.com/docs/api-reference/responses-streaming/response/output_text/annotation_added
            # annotation is a plain dict at runtime (SDK types it as `object`)
            ann: dict = chunk.annotation  # type: ignore[assignment]
            if ann.get("type") == "url_citation":
                return [
                    ContentCitation(
                        source=WebSource(url=ann["url"], title=ann.get("title")),
                        # No grounded_span here: OpenAI streams the annotation
                        # with start_index/end_index into text that hasn't fully
                        # arrived yet, so the span is resolved on the final turn.
                        extra=ann,
                    )
                ]
            return []
        if chunk.type == "response.output_item.done":
            item = chunk.item
            if isinstance(item, ResponseFunctionWebSearch):
                return [openai_web_search_request(item)]
            return []
        if chunk.type == "response.reasoning_summary_text.delta":
            # https://platform.openai.com/docs/api-reference/responses-streaming/response/reasoning_summary_text/delta
            return [ContentThinkingDelta(thinking=chunk.delta)]
        return []

    def stream_merge_chunks(self, completion, chunk):
        if chunk.type == "response.completed" or chunk.type == "response.incomplete":
            return chunk.response
        elif chunk.type == "response.failed":
            error = chunk.response.error
            if error is None:
                msg = "Request failed with an unknown error."
            else:
                msg = f"Request failed ({error.code}): {error.message}"
            raise RuntimeError(msg)
        elif chunk.type == "error":
            raise RuntimeError(f"Request errored: {chunk.message}")

        return completion

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

    def value_cost(
        self,
        completion,
        tokens: tuple[int, int, int] | None = None,
    ) -> float | None:
        """
        Compute the cost for a completion, using service_tier if available.
        """
        from ._tokens import get_token_cost

        if tokens is None:
            tokens = self.value_tokens(completion)
        if tokens is None:
            return None

        service_tier = ""
        if completion is not None:
            service_tier = completion.service_tier or ""

        return get_token_cost(self.name, self.model, tokens, service_tier)

    def batch_result_turn(self, result, has_data_model: bool = False):
        response = BatchResult.model_validate(result).response
        if response.status_code != 200:
            # TODO: offer advice on what to do?
            warnings.warn(f"Batch request failed: {response.body}")
            return None

        completion = Response.construct(**response.body)
        return self._response_as_turn(completion, has_data_model)

    @staticmethod
    def _response_as_turn(completion: Response, has_data_model: bool) -> AssistantTurn:
        incomplete_reason = None
        if completion.incomplete_details is not None:
            incomplete_reason = completion.incomplete_details.reason

        finish_reason = normalize_finish_reason(completion.status, incomplete_reason)
        if has_data_model:
            # Must precede the JSON parse below; see check_finish_reason().
            check_finish_reason(finish_reason, "error")

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
                        for a in x.annotations or []:
                            if not isinstance(a, AnnotationURLCitation):
                                continue
                            grounded = x.text[a.start_index : a.end_index] or None
                            contents.append(
                                ContentCitation(
                                    source=WebSource(url=a.url, title=a.title),
                                    grounded_span=grounded,
                                    extra=a.model_dump(),
                                )
                            )

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
                contents.append(
                    ContentThinking(
                        thinking="".join(x.text for x in output.summary),
                        extra=output.model_dump(),
                    )
                )

            elif output.type == "image_generation_call":
                result = output.result
                if result:
                    mime_type = "image/png"
                    if "image/jpeg" in result:
                        mime_type = "image/jpeg"
                    elif "image/webp" in result:
                        mime_type = "image/webp"
                    elif "image/gif" in result:
                        mime_type = "image/gif"

                    contents.append(
                        ContentImageInline(
                            data=result,
                            image_content_type=mime_type,
                        )
                    )

            elif output.type == "web_search_call":
                contents.append(openai_web_search_request(output))

            else:
                raise ValueError(f"Unknown output type: {output.type}")

        if finish_reason == "success" and any(
            isinstance(x, ContentToolRequest) for x in contents
        ):
            finish_reason = "tool_use"

        return AssistantTurn(
            contents,
            finish_reason=finish_reason,
            completion=completion,
        )

    def _turns_as_inputs(self, turns: list[Turn]) -> "list[ResponseInputItemParam]":
        res: "list[ResponseInputItemParam]" = []
        for turn in turns:
            for x in turn.contents:
                if isinstance(x, PROVIDER_ANNOTATION_TYPES) and not openai_replayable(
                    x
                ):
                    continue
                res.append(as_input_param(x, turn.role))
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

    def file_upload(
        self,
        file: "str | os.PathLike[str] | IO[bytes]",
        *,
        mime_type: Optional[str] = None,
    ) -> ContentUploaded:
        with open_binary(file) as f:
            obj = self._client.files.create(file=f, purpose="user_data")
        return openai_uploaded(obj, mime_type)

    async def file_upload_async(
        self,
        file: "str | os.PathLike[str] | IO[bytes]",
        *,
        mime_type: Optional[str] = None,
    ) -> ContentUploaded:
        with open_binary(file) as f:
            obj = await self._async_client.files.create(file=f, purpose="user_data")
        return openai_uploaded(obj, mime_type)

    def file_list(self) -> list[FileMetadata]:
        return [openai_meta(o) for o in self._client.files.list()]

    async def file_list_async(self) -> list[FileMetadata]:
        page = await self._async_client.files.list()
        return [openai_meta(o) async for o in page]

    def file_get(self, id: str) -> FileMetadata:  # noqa: A002
        return openai_meta(self._client.files.retrieve(id))

    async def file_get_async(self, id: str) -> FileMetadata:  # noqa: A002
        return openai_meta(await self._async_client.files.retrieve(id))

    def file_download(
        self,
        id: str,  # noqa: A002
        path: "str | os.PathLike[str] | None" = None,
    ) -> bytes:
        data = self._client.files.content(id).read()
        return maybe_write(data, path)

    async def file_download_async(
        self,
        id: str,  # noqa: A002
        path: "str | os.PathLike[str] | None" = None,
    ) -> bytes:
        resp = await self._async_client.files.content(id)
        return maybe_write(resp.read(), path)

    def file_delete(self, id: str) -> None:  # noqa: A002
        self._client.files.delete(id)

    async def file_delete_async(self, id: str) -> None:  # noqa: A002
        await self._async_client.files.delete(id)


def openai_web_search_request(
    item: "ResponseFunctionWebSearch",
) -> ContentToolRequestSearch | ContentToolRequestFetch:
    """Map a `web_search_call` item onto request content.

    https://platform.openai.com/docs/guides/tools-web-search#output-and-citations

    The action is a closed union of three verbs, and they aren't all searches:
    `open_page` fetches a URL, and `find_in_page` matches a pattern within an
    already-open page.
    """
    action = item.action
    extra = item.model_dump()
    if isinstance(action, ActionOpenPage):
        return ContentToolRequestFetch(url=action.url or "", extra=extra)
    if isinstance(action, ActionFind):
        return ContentToolRequestSearch(query=action.pattern, extra=extra)
    queries = action.queries or []
    return ContentToolRequestSearch(
        query=action.query or (queries[0] if queries else "web search"),
        extra=extra,
    )


def openai_replayable(content: ProviderAnnotation) -> bool:
    """Whether `content.extra` holds a Responses API item we can resend.

    Only the `web_search_call` item round-trips; the rest of what a web search
    produces (results, citations) is reported client-side with no item to send
    back. Content from another provider carries that provider's payload, which
    the Responses API would reject.
    """
    extra = content.extra
    return (
        isinstance(content, (ContentToolRequestSearch, ContentToolRequestFetch))
        and isinstance(extra, dict)
        and extra.get("type") == "web_search_call"
    )


def as_input_param(content: Content, role: Role) -> "ResponseInputItemParam":
    if isinstance(content, ContentText):
        if role == "assistant":
            # Assistant messages use output_text, but the SDK incorrectly requires an id.
            return cast(
                "ResponseInputItemParam",
                {
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
                },
            )
        else:
            return as_message({"type": "input_text", "text": content.text}, role)
    elif isinstance(content, ContentJson):
        text = orjson.dumps(content.value).decode("utf-8")
        return as_input_param(ContentText(text=text), role)
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
        check_image_content_type_supported("OpenAI", content.image_content_type)
        return as_message(
            {
                "type": "input_image",
                "image_url": f"data:{content.image_content_type};base64,{content.data}",
                "detail": "auto",
            },
            role,
        )
    elif isinstance(content, ContentPDF):
        return as_message(as_input_file_param(content, "application/pdf"), role)
    elif isinstance(content, ContentDocument):
        return as_message(as_input_file_param(content, content.mime_type), role)
    elif isinstance(content, ContentThinking):
        # Filter out 'status' which is output-only and not accepted as input
        extra = content.extra or {}
        return cast(
            "ResponseReasoningItemParam",
            {k: v for k, v in extra.items() if k != "status"},
        )
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
    elif isinstance(content, (ContentToolRequestSearch, ContentToolRequestFetch)):
        # The raw `web_search_call` item, replayed verbatim (see openai_replayable)
        return cast("ResponseInputItemParam", content.extra)
    elif isinstance(content, ContentUploaded):
        if content.provider != "openai":
            raise ValueError(
                f"This file was uploaded to provider '{content.provider}', but "
                "is being used with OpenAI. Re-upload it with an OpenAI chat."
            )
        part: "ResponseInputContentParam"
        if content.mime_type.startswith("image/"):
            part = {"type": "input_image", "file_id": content.id, "detail": "auto"}
        else:
            part = {"type": "input_file", "file_id": content.id}
        return as_message(part, role)
    else:
        raise ValueError(f"Unsupported content type: {type(content)}")


def as_message(x: "ResponseInputContentParam", role: Role) -> "EasyInputMessageParam":
    return {"role": role, "content": [x]}


def as_input_file_param(
    content: "ContentPDF | ContentDocument", mime_type: str
) -> "ResponseInputFileParam":
    """Build an `input_file` param, preferring a URL over re-sending bytes.

    The Responses API accepts `file_url` for any file type (not just PDFs),
    so a `ContentPDF`/`ContentDocument` with a URL never needs to download it.
    `filename` is deliberately omitted on that path: the API treats it as
    mutually exclusive with `file_url` and rejects requests carrying both.
    """
    if content.url is not None:
        return {
            "type": "input_file",
            "file_url": content.url,
        }
    data = ensure_bytes(content, "file")
    return {
        "type": "input_file",
        "filename": content.filename,
        "file_data": f"data:{mime_type};base64,{base64.b64encode(data).decode('utf-8')}",
    }


def check_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.hostname != "api.openai.com":
        warnings.warn(
            "ChatOpenAI() uses OpenAI's Responses API, which is not supported by most "
            "third-party backends. Use ChatOpenAICompletions() instead for "
            "OpenAI-compatible backends (e.g., vLLM, Ollama, LiteLLM, etc.).",
            UserWarning,
            stacklevel=3,
        )


def is_reasoning_model(model: str) -> bool:
    # https://platform.openai.com/docs/models/compare
    return model.startswith("o") or model.startswith("gpt-5")


def openai_uploaded(obj: "FileObject", mime_type: Optional[str]) -> ContentUploaded:
    guessed = (
        mime_type
        or mimetypes.guess_type(obj.filename or "")[0]
        or "application/octet-stream"
    )
    return ContentUploaded(
        id=obj.id,
        mime_type=guessed,
        provider="openai",
        extra={"filename": obj.filename, "bytes": obj.bytes},
    )


def openai_meta(obj: "FileObject") -> FileMetadata:
    return FileMetadata(
        id=obj.id,
        filename=obj.filename,
        mime_type=None,
        size_bytes=obj.bytes,
        created_at=datetime.fromtimestamp(obj.created_at, tz=timezone.utc),
        expires_at=None,
        provider="openai",
        extra=obj,
    )
