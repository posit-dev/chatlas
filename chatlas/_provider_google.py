from __future__ import annotations

import base64
import re
import warnings
from typing import TYPE_CHECKING, Any, Literal, Optional, cast, overload

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
    ContentThinkingDelta,
    ContentToolRequest,
    ContentToolResult,
)
from ._logging import log_model_default
from ._merge import merge_dicts
from ._provider import (
    BatchStatus,
    ModelInfo,
    Provider,
    StandardModelParamNames,
    StandardModelParams,
)
from ._tokens import get_price_info
from ._tools import Tool, ToolBuiltIn
from ._tools_builtin import ToolWebFetch, ToolWebSearch
from ._turn import AssistantTurn, SystemTurn, Turn, UserTurn, user_turn

if TYPE_CHECKING:
    from google.genai.types import Content as GoogleContent
    from google.genai.types import (
        GenerateContentConfigDict,
        GenerateContentResponse,
        GenerateContentResponseDict,
        Part,
        PartDict,
        ThinkingConfigDict,
    )

    from .types.google import ChatClientArgs, SubmitInputArgs
else:
    GenerateContentResponse = object


ReasoningEffort = Literal["minimal", "low", "medium", "high"]


def ChatGoogle(
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    reasoning: Optional["int | ReasoningEffort | ThinkingConfigDict"] = None,
    api_key: Optional[str] = None,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", GenerateContentResponse]:
    """
    Chat with a Google Gemini model.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## Authentication

    The simplest way to authenticate is with an API key. Sign up for an account
    and [get an API key](https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=python),
    then pass it via `api_key` or set the `GOOGLE_API_KEY` environment variable.

    For OAuth, service accounts, or Application Default Credentials, pass a
    `google.auth.credentials.Credentials` object via `kwargs`:

    ```python
    import google.auth

    credentials, _ = google.auth.default()
    chat = ChatGoogle(kwargs={"credentials": credentials})
    ```
    :::

    ::: {.callout-note}
    ## Python requirements

    `ChatGoogle` requires the `google-genai` package: `pip install "chatlas[google]"`.
    :::

    Examples
    --------

    ```python
    import os
    from chatlas import ChatGoogle

    chat = ChatGoogle(api_key=os.getenv("GOOGLE_API_KEY"))
    chat.chat("What is the capital of France?")
    ```

    Parameters
    ----------
    system_prompt
        A system prompt to set the behavior of the assistant.
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly choosing
        a model for all but the most casual use.
    reasoning
        If provided, enables reasoning (a.k.a. "thoughts") in the model's
        responses. This can be an integer number of tokens to use for reasoning
        (a thinking budget), a string thinking level (`"minimal"`, `"low"`,
        `"medium"`, or `"high"`), or a full `ThinkingConfigDict` to customize
        the reasoning behavior.
    api_key
        The API key to use for authentication. You generally should not supply
        this directly, but instead set the `GOOGLE_API_KEY` environment variable.
    kwargs
        Additional arguments to pass to the `genai.Client` constructor.

    Returns
    -------
    Chat
        A Chat object.

    Note
    ----
    Pasting an API key into a chat constructor (e.g., `ChatGoogle(api_key="...")`)
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
    GOOGLE_API_KEY=...
    ```

    ```python
    from chatlas import ChatGoogle
    from dotenv import load_dotenv

    load_dotenv()
    chat = ChatGoogle()
    chat.console()
    ```

    Another, more general, solution is to load your environment variables into the shell
    before starting Python (maybe in a `.bashrc`, `.zshrc`, etc. file):

    ```shell
    export GOOGLE_API_KEY=...
    ```
    """

    if model is None:
        model = log_model_default("gemini-3.5-flash")

    kwargs_chat: "SubmitInputArgs" = {}
    if reasoning is not None:
        thinking_config: "ThinkingConfigDict"
        if isinstance(reasoning, int):
            thinking_config = {
                "thinking_budget": reasoning,
                "include_thoughts": True,
            }
        elif isinstance(reasoning, str):
            from google.genai.types import ThinkingLevel

            thinking_config = {
                "thinking_level": ThinkingLevel(reasoning.upper()),
                "include_thoughts": True,
            }
        else:
            thinking_config = reasoning
        kwargs_chat["config"] = {"thinking_config": thinking_config}

    return Chat(
        provider=GoogleProvider(
            model=model,
            api_key=api_key,
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
        kwargs_chat=kwargs_chat,
    )


class GoogleProvider(
    Provider[
        GenerateContentResponse,
        GenerateContentResponse,
        "GenerateContentResponseDict",
        "SubmitInputArgs",
    ]
):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        name: str = "Google/Gemini",
        kwargs: Optional["ChatClientArgs"],
    ):
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                f"The {self.__class__.__name__} class requires the `google-genai` package. "
                "Install it with `pip install google-genai`."
            )
        super().__init__(name=name, model=model)

        kwargs_full: "ChatClientArgs" = {
            "api_key": api_key,
            **(kwargs or {}),
        }

        self._client = genai.Client(**kwargs_full)

    def list_models(self):
        models = self._client.models.list()

        res: list[ModelInfo] = []
        for m in models:
            name = m.name or "[unknown]"
            pricing = get_price_info(self.name, name) or {}
            info: ModelInfo = {
                "id": name,
                "name": m.display_name or "[unknown]",
                "input": pricing.get("input"),
                "output": pricing.get("output"),
                "cached_input": pricing.get("cached_input"),
            }
            res.append(info)

        # Sort list by created_by field (more recent first)
        res.sort(
            key=lambda x: x.get("created", 0),
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
        kwargs = self._chat_perform_args(turns, tools, data_model, kwargs)
        if stream:
            return self._client.models.generate_content_stream(**kwargs)
        else:
            return self._client.models.generate_content(**kwargs)

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
        kwargs = self._chat_perform_args(turns, tools, data_model, kwargs)
        if stream:
            return await self._client.aio.models.generate_content_stream(**kwargs)
        else:
            return await self._client.aio.models.generate_content(**kwargs)

    def _chat_perform_args(
        self,
        turns: list[Turn],
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ) -> "SubmitInputArgs":
        from google.genai.types import (
            FunctionDeclaration,
            GenerateContentConfig,
            Schema,
            ToolConfig,
            ToolListUnion,
        )
        from google.genai.types import Tool as GoogleTool

        kwargs_full: "SubmitInputArgs" = {
            "model": self.model,
            "contents": cast("GoogleContent", self._google_contents(turns)),
            **(kwargs or {}),
        }

        config = kwargs_full.get("config")
        if config is None:
            config = GenerateContentConfig()
        if isinstance(config, dict):
            config = GenerateContentConfig.model_construct(**config)

        if config.system_instruction is None:
            if len(turns) > 0 and isinstance(turns[0], SystemTurn):
                config.system_instruction = turns[0].text

        if data_model:
            config.response_schema = data_model
            config.response_mime_type = "application/json"

        if tools:
            google_tools: ToolListUnion = []
            has_builtin_tool = False
            has_custom_tool = False
            for tool in tools.values():
                if isinstance(tool, ToolBuiltIn):
                    has_builtin_tool = True
                else:
                    has_custom_tool = True

                if isinstance(tool, ToolWebSearch):
                    gtool = GoogleTool(google_search=tool.get_definition("google"))
                    google_tools.append(gtool)
                elif isinstance(tool, ToolWebFetch):
                    gtool = GoogleTool(url_context=tool.get_definition("google"))
                    google_tools.append(gtool)
                elif isinstance(tool, ToolBuiltIn):
                    gtool = GoogleTool.model_validate(tool.definition)
                    google_tools.append(gtool)
                else:
                    func = tool.schema["function"]
                    params = func.get("parameters")
                    gtool = GoogleTool(
                        function_declarations=[
                            FunctionDeclaration(
                                name=func["name"],
                                description=func.get("description"),
                                parameters=Schema.model_validate(
                                    _strip_additional_properties(params)
                                )
                                if params
                                else None,
                            )
                        ]
                    )
                    google_tools.append(gtool)

            if google_tools:
                config.tools = google_tools

            # Mixing built-in and custom tools requires an explicit opt-in on
            # Gemini 3+ models. `include_server_side_tool_invocations` is only
            # valid for the Gemini Developer API, not Vertex AI, which raises
            # a client-side error if it's set at all.
            if (
                has_builtin_tool
                and has_custom_tool
                and self.name == "Google/Gemini"
                and google_supports_mixed_tools(self.model)
            ):
                if config.tool_config is None:
                    config.tool_config = ToolConfig(
                        include_server_side_tool_invocations=True
                    )
                else:
                    config.tool_config.include_server_side_tool_invocations = True

        kwargs_full["config"] = config

        return kwargs_full

    def stream_content(self, chunk) -> Optional[Content]:
        candidates = getattr(chunk, "candidates", None)
        if not candidates:
            return None
        content = candidates[0].content
        if content is None:
            return None
        parts = content.parts
        if not parts:
            return None
        part = parts[0]
        text = getattr(part, "text", None)
        if text is None:
            return None
        if getattr(part, "thought", None):
            return ContentThinkingDelta(thinking=text)
        return ContentText.model_construct(text=text)

    def stream_merge_chunks(self, completion, chunk):
        chunkd = chunk.model_dump()
        if completion is None:
            return cast("GenerateContentResponseDict", chunkd)
        return cast(
            "GenerateContentResponseDict",
            merge_dicts(completion, chunkd),  # type: ignore
        )

    def stream_turn(self, completion, has_data_model):
        return self._as_turn(
            completion,
            has_data_model,
        )

    def value_turn(self, completion, has_data_model):
        completion = cast("GenerateContentResponseDict", completion.model_dump())
        return self._as_turn(completion, has_data_model)

    def value_tokens(self, completion):
        if isinstance(completion, dict):
            # Currently value_turn() attached a dict completion
            from google.genai.types import GenerateContentResponseUsageMetadata

            usage = GenerateContentResponseUsageMetadata.model_validate(
                completion.get("usage_metadata", {})
            )
        else:
            usage = completion.usage_metadata

        if usage is None:
            return None
        cached = usage.cached_content_token_count or 0
        return (
            (usage.prompt_token_count or 0) - cached,
            (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
            usage.cached_content_token_count or 0,
        )

    def token_count(
        self,
        *args: Content | str,
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]],
    ):
        kwargs = self._token_count_args(
            *args,
            tools=tools,
            data_model=data_model,
        )

        res = self._client.models.count_tokens(**kwargs)
        return res.total_tokens or 0

    async def token_count_async(
        self,
        *args: Content | str,
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]],
    ):
        kwargs = self._token_count_args(
            *args,
            tools=tools,
            data_model=data_model,
        )

        res = await self._client.aio.models.count_tokens(**kwargs)
        return res.total_tokens or 0

    def _token_count_args(
        self,
        *args: Content | str,
        tools: dict[str, Tool | ToolBuiltIn],
        data_model: Optional[type[BaseModel]],
    ) -> dict[str, Any]:
        turn = user_turn(*args)

        kwargs = self._chat_perform_args(
            turns=[turn],
            tools=tools,
            data_model=data_model,
        )

        args_to_keep = ["model", "contents", "tools"]

        return {arg: kwargs[arg] for arg in args_to_keep if arg in kwargs}

    def _google_contents(self, turns: list[Turn]) -> list["GoogleContent"]:
        from google.genai.types import Content as GoogleContent

        contents: list["GoogleContent"] = []
        for turn in turns:
            if isinstance(turn, SystemTurn):
                continue  # System messages are handled separately
            elif isinstance(turn, UserTurn):
                parts = [self._as_part_type(c) for c in turn.contents]
                contents.append(GoogleContent(role=turn.role, parts=parts))
            elif isinstance(turn, AssistantTurn):
                parts = [self._as_part_type(c) for c in turn.contents]
                contents.append(GoogleContent(role="model", parts=parts))
            else:
                raise ValueError(f"Unknown role {turn.role}")
        return contents

    def _as_part_type(self, content: Content) -> "Part":
        from google.genai.types import FunctionCall, FunctionResponse, Part

        if isinstance(content, ContentText):
            return Part.from_text(text=content.text)
        elif isinstance(content, ContentJson):
            text = orjson.dumps(content.value).decode("utf-8")
            return Part.from_text(text=text)
        elif isinstance(content, ContentPDF):
            from google.genai.types import Blob

            return Part(
                inline_data=Blob(
                    data=content.data,
                    mime_type="application/pdf",
                    # Not supported?
                    # display_name=content.filename,
                )
            )
        elif isinstance(content, ContentImageInline) and content.data:
            return Part.from_bytes(
                data=base64.b64decode(content.data),
                mime_type=content.image_content_type,
            )
        elif isinstance(content, ContentImageRemote):
            raise NotImplementedError(
                "Remote images aren't supported by Google (Gemini). "
                "Consider downloading the image and using content_image_file() instead."
            )
        elif isinstance(content, ContentToolRequest):
            return Part(
                function_call=FunctionCall(
                    id=content.id if content.name != content.id else None,
                    name=content.name,
                    # Goes in a dict, so should come out as a dict
                    args=cast(dict[str, Any], content.arguments),
                ),
                thought_signature=content.extra.get("thought_signature"),  # type: ignore
            )
        elif isinstance(content, ContentToolResult):
            if content.error:
                resp = {"error": content.error}
            else:
                resp = {"result": content.get_model_value()}
            return Part(
                # TODO: seems function response parts might need role='tool'???
                # https://github.com/googleapis/python-genai/blame/c8cfef85c/README.md#L344
                function_response=FunctionResponse(
                    id=content.id if content.name != content.id else None,
                    name=content.name,
                    response=resp,
                )
            )
        raise ValueError(f"Unknown content type: {type(content)}")

    def _as_turn(
        self,
        message: "GenerateContentResponseDict",
        has_data_model: bool,
    ) -> AssistantTurn:
        from google.genai.types import FinishReason

        candidates = message.get("candidates")
        if not candidates:
            return AssistantTurn("")

        parts: list["PartDict"] = []
        finish_reason = None
        for candidate in candidates:
            content = candidate.get("content")
            if content:
                parts.extend(content.get("parts") or {})
            finish = candidate.get("finish_reason")
            if finish:
                finish_reason = finish

        contents: list[Content] = []
        for part in parts:
            text = part.get("text")
            if text:
                if has_data_model:
                    contents.append(ContentJson(value=orjson.loads(text)))
                elif part.get("thought"):
                    contents.append(ContentThinking(thinking=text))
                else:
                    contents.append(ContentText(text=text))
            function_call = part.get("function_call")
            if function_call:
                # Seems name is required but id is optional?
                name = function_call.get("name")
                if name:
                    extra: dict[str, object] = {}
                    thought_signature = part.get("thought_signature")
                    if thought_signature is not None:
                        extra["thought_signature"] = thought_signature
                    contents.append(
                        ContentToolRequest(
                            id=function_call.get("id") or name,
                            name=name,
                            arguments=function_call.get("args"),
                            extra=extra,
                        )
                    )
            function_response = part.get("function_response")
            if function_response:
                # Seems name is required but id is optional?
                name = function_response.get("name")
                if name:
                    contents.append(
                        ContentToolResult(
                            value=function_response.get("response"),
                            request=ContentToolRequest(
                                id=function_response.get("id") or name,
                                name=name,
                                # TODO: how to get the arguments?
                                arguments={},
                            ),
                        )
                    )
            inline_data = part.get("inline_data")
            if inline_data:
                mime_type = inline_data.get("mime_type")
                data = inline_data.get("data")
                if mime_type and data:
                    contents.append(
                        ContentImageInline(
                            data=data.decode("utf-8"),
                            image_content_type=mime_type,  # type: ignore
                        )
                    )

        if isinstance(finish_reason, FinishReason):
            finish_reason = finish_reason.name

        return AssistantTurn(
            contents,
            finish_reason=finish_reason,
            completion=message,
        )

    def translate_model_params(self, params: StandardModelParams) -> "SubmitInputArgs":
        config: "GenerateContentConfigDict" = {}
        if "temperature" in params:
            config["temperature"] = params["temperature"]

        if "top_p" in params:
            config["top_p"] = params["top_p"]

        if "top_k" in params:
            config["top_k"] = params["top_k"]

        if "frequency_penalty" in params:
            config["frequency_penalty"] = params["frequency_penalty"]

        if "presence_penalty" in params:
            config["presence_penalty"] = params["presence_penalty"]

        if "seed" in params:
            config["seed"] = params["seed"]

        if "max_tokens" in params:
            config["max_output_tokens"] = params["max_tokens"]

        if "log_probs" in params:
            config["logprobs"] = params["log_probs"]

        if "stop_sequences" in params:
            config["stop_sequences"] = params["stop_sequences"]

        res: "SubmitInputArgs" = {"config": config}

        return res

    def supported_model_params(self) -> set[StandardModelParamNames]:
        return {
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "max_tokens",
            "log_probs",
            "stop_sequences",
        }

    def has_batch_support(self) -> bool:
        # Only the Gemini Developer API has a batch API today; Vertex AI's
        # batch API takes GCS bucket URIs instead of inline requests, which
        # is a different shape entirely.
        return self.name == "Google/Gemini"

    def batch_submit(
        self,
        conversations: list[list[Turn]],
        data_model: Optional[type[BaseModel]] = None,
    ):
        from google.genai import types
        from google.genai.types import GenerateContentConfig

        requests: list["types.InlinedRequest"] = []
        for turns in conversations:
            kwargs = self._chat_perform_args(turns, {}, data_model)
            contents = cast(types.ContentListUnion, kwargs.get("contents"))
            config = cast(Optional[GenerateContentConfig], kwargs.get("config"))
            requests.append(types.InlinedRequest(contents=contents, config=config))

        batch = self._client.batches.create(model=self.model, src=requests)
        # mode="json" is required, not cosmetic: reasoning-capable Gemini
        # models (e.g. gemini-3.6-flash) attach an opaque thought_signature
        # byte blob to every response part, not just tool-call parts, and
        # those bytes are not valid UTF-8. Plain model_dump() leaves them as
        # raw bytes, which crashes BatchState.model_dump_json() once this
        # dict is persisted to the batch state file. mode="json" base64-
        # encodes bytes fields, and model_validate() decodes them back
        # losslessly (verified against a real completed batch response).
        return batch.model_dump(mode="json")

    def batch_poll(self, batch):
        from google.genai import types

        batch = types.BatchJob.model_validate(batch)
        if batch.name is None:
            raise ValueError("Batch has no name")
        b = self._client.batches.get(name=batch.name)
        return b.model_dump(mode="json")

    def batch_status(self, batch) -> "BatchStatus":
        from google.genai import types

        batch = types.BatchJob.model_validate(batch)
        terminal_states = {
            types.JobState.JOB_STATE_SUCCEEDED,
            types.JobState.JOB_STATE_FAILED,
            types.JobState.JOB_STATE_CANCELLED,
            types.JobState.JOB_STATE_EXPIRED,
            types.JobState.JOB_STATE_PARTIALLY_SUCCEEDED,
        }

        stats = batch.completion_stats
        n_succeeded = (stats.successful_count or 0) if stats else 0
        n_failed = (stats.failed_count or 0) if stats else 0
        n_processing = (stats.incomplete_count or 0) if stats else 0

        return BatchStatus(
            working=batch.state not in terminal_states,
            n_processing=n_processing,
            n_succeeded=n_succeeded,
            n_failed=n_failed,
        )

    def batch_retrieve(self, batch):
        from google.genai import types

        batch = types.BatchJob.model_validate(batch)
        if batch.dest is None or batch.dest.inlined_responses is None:
            raise ValueError("Batch has no results")

        # No custom-id/index-based reordering needed here (unlike the
        # file-based batch_retrieve() in _provider_openai_generic.py and
        # _provider_anthropic.py): the SDK's inlined_responses are documented
        # to preserve input order.
        return [r.model_dump(mode="json") for r in batch.dest.inlined_responses]

    def batch_result_turn(self, result, has_data_model: bool = False):
        from google.genai import types

        result = types.InlinedResponse.model_validate(result)
        if result.error is not None:
            warnings.warn(f"Batch request failed: {result.error}")
            return None
        if result.response is None:
            warnings.warn("Batch request returned no response")
            return None

        completion = cast("GenerateContentResponseDict", result.response.model_dump())
        return self._as_turn(completion, has_data_model)


def ChatVertex(
    *,
    model: Optional[str] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    kwargs: Optional["ChatClientArgs"] = None,
) -> Chat["SubmitInputArgs", GenerateContentResponse]:
    """
    Chat with a Google Vertex AI model.

    Prerequisites
    -------------

    ::: {.callout-note}
    ## Python requirements

    `ChatGoogle` requires the `google-genai` package: `pip install "chatlas[vertex]"`.
    :::

    ::: {.callout-note}
    ## Credentials

    To use Google's models (i.e., Vertex AI), you'll need to sign up for an account
    with [Vertex AI](https://cloud.google.com/vertex-ai), then specify the appropriate
    model, project, and location.

    Vertex AI typically authenticates via Application Default Credentials (ADC).
    You can also pass a `google.auth.credentials.Credentials` object via `kwargs`:

    ```python
    import google.auth

    credentials, project = google.auth.default()
    chat = ChatVertex(
        project=project,
        location="us-central1",
        kwargs={"credentials": credentials},
    )
    ```
    :::

    Parameters
    ----------
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly choosing
        a model for all but the most casual use.
    project
        The Google Cloud project ID (e.g., "your-project-id"). If not provided, the
        GOOGLE_CLOUD_PROJECT environment variable will be used.
    location
        The Google Cloud location (e.g., "us-central1"). If not provided, the
        GOOGLE_CLOUD_LOCATION environment variable will be used.
    system_prompt
        A system prompt to set the behavior of the assistant.

    Returns
    -------
    Chat
        A Chat object.

    Examples
    --------

    ```python
    import os
    from chatlas import ChatVertex

    chat = ChatVertex(
        project="your-project-id",
        location="us-central1",
    )
    chat.chat("What is the capital of France?")
    ```
    """

    if kwargs is None:
        kwargs = {}

    kwargs["vertexai"] = True
    kwargs["project"] = project
    kwargs["location"] = location

    if model is None:
        model = log_model_default("gemini-3.5-flash")

    return Chat(
        provider=GoogleProvider(
            model=model,
            api_key=api_key,
            name="Google/Vertex",
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
    )


def google_supports_mixed_tools(model: str) -> bool:
    """
    Whether `model` supports combining custom (function-calling) tools with
    built-in (server-side) tools in the same request. Only Gemini 3+ models
    support this; older models reject the combination outright.
    """
    # `list_models()` surfaces IDs prefixed with "models/" (e.g.
    # "models/gemini-3.5-flash"), so strip that before matching.
    model = model.removeprefix("models/")
    return bool(re.match(r"^gemini-([3-9]|[0-9]{2,})", model))


def _strip_additional_properties(params: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively remove additionalProperties from JSON schema.

    Google's API doesn't accept additionalProperties in tool schemas,
    so we strip it before passing to Schema.model_validate().
    """
    result = {k: v for k, v in params.items() if k != "additionalProperties"}

    if "properties" in result and isinstance(result["properties"], dict):
        result["properties"] = {
            k: _strip_additional_properties(v) if isinstance(v, dict) else v
            for k, v in result["properties"].items()
        }

    if "items" in result and isinstance(result["items"], dict):
        result["items"] = _strip_additional_properties(result["items"])

    return result
