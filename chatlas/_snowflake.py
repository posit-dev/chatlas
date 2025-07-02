from typing import (
    TYPE_CHECKING,
    Generator,
    Literal,
    Optional,
    TypedDict,
    Union,
    overload,
)

import orjson
from pydantic import BaseModel

from ._chat import Chat
from ._content import (
    Content,
    ContentJson,
    ContentText,
    ContentToolRequest,
    ContentToolResult,
)
from ._logging import log_model_default
from ._provider import Provider, StandardModelParamNames, StandardModelParams
from ._tokens import tokens_log
from ._tools import Tool, basemodel_to_param_schema
from ._turn import Turn
from ._utils import drop_none

if TYPE_CHECKING:
    import snowflake.core.cortex.inference_service._generated.models as models
    from snowflake.core.rest import Event, SSEClient

    Completion = models.NonStreamingCompleteResponse
    CompletionChunk = models.StreamingCompleteResponseDataEvent

    # Manually constructed TypedDict equivalent of models.CompleteRequest
    class CompleteRequest(TypedDict, total=False):
        """
        CompleteRequest parameters for Snowflake Cortex LLMs.

        See `snowflake.core.cortex.inference_service.CompleteRequest` for more details.
        """

        temperature: Union[float, int]
        """Temperature controls the amount of randomness used in response generation. A higher temperature corresponds to more randomness."""

        top_p: Union[float, int]
        """Threshold probability for nucleus sampling. A higher top-p value increases the diversity of tokens that the model considers, while a lower value results in more predictable output."""

        max_tokens: int
        """The maximum number of output tokens to produce. The default value is model-dependent."""

        guardrails: models.GuardrailsConfig
        """Controls whether guardrails are enabled."""

        tool_choice: models.ToolChoice
        """Determines how tools are selected."""


def ChatSnowflake(
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    connection_name: Optional[str] = None,
    account: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    private_key_file: Optional[str] = None,
    private_key_file_pwd: Optional[str] = None,
    kwargs: Optional[dict[str, "str | int"]] = None,
) -> Chat["CompleteRequest", "Completion"]:
    """
    Chat with a Snowflake Cortex LLM

    https://docs.snowflake.com/en/user-guide/snowflake-cortex/llm-functions

    Prerequisites
    -------------

    ::: {.callout-note}
    ## Python requirements

    `ChatSnowflake`, requires the `snowflake-ml-python` package:
    `pip install "chatlas[snowflake]"`.
    :::

    ::: {.callout-note}
    ## Snowflake credentials

    Snowflake provides a handful of ways to authenticate, but it's recommended
    to use [key-pair
    auth](https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-connect#label-python-connection-toml)
    to generate a `private_key_file`. It's also recommended to place your
    credentials in a [`connections.toml`
    file](https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session#connect-by-using-the-connections-toml-file).

    This way, once your credentials are in the `connections.toml` file, you can
    simply call `ChatSnowflake(connection_name="my_connection")` to
    authenticate. If you don't want to use a `connections.toml` file, you can
    specify the connection parameters directly (with `account`, `user`,
    `password`, etc.).
    :::


    Parameters
    ----------
    system_prompt
        A system prompt to set the behavior of the assistant.
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly
        choosing a model for all but the most casual use.
    connection_name
        The name of the connection (i.e., section) within the connections.toml file.
        This is useful if you want to keep your credentials in a connections.toml file
        rather than specifying them directly in the arguments.
        https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session#connect-by-using-the-connections-toml-file
    account
        Your Snowflake account identifier. Required if `connection_name` is not provided.
        https://docs.snowflake.com/en/user-guide/admin-account-identifier
    user
        Your Snowflake user name. Required if `connection_name` is not provided.
    password
        Your Snowflake password. Required if doing password authentication and
        `connection_name` is not provided.
    private_key_file
        The path to your private key file. Required if you are using key pair authentication.
        https://docs.snowflake.com/en/user-guide/key-pair-auth
    private_key_file_pwd
        The password for your private key file. Required if you are using key pair authentication.
        https://docs.snowflake.com/en/user-guide/key-pair-auth
    kwargs
        Additional keyword arguments passed along to the Snowflake connection builder. These can
        include any parameters supported by the `snowflake-ml-python` package.
        https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session#connect-by-specifying-connection-parameters
    """

    if model is None:
        model = log_model_default("claude-3-7-sonnet")

    return Chat(
        provider=SnowflakeProvider(
            model=model,
            connection_name=connection_name,
            account=account,
            user=user,
            password=password,
            private_key_file=private_key_file,
            private_key_file_pwd=private_key_file_pwd,
            kwargs=kwargs,
        ),
        system_prompt=system_prompt,
    )


class SnowflakeProvider(
    Provider["Completion", "CompletionChunk", "CompletionChunk", "CompleteRequest"]
):
    def __init__(
        self,
        *,
        model: str,
        connection_name: Optional[str],
        account: Optional[str],
        user: Optional[str],
        password: Optional[str],
        private_key_file: Optional[str],
        private_key_file_pwd: Optional[str],
        name: str = "Snowflake",
        kwargs: Optional[dict[str, "str | int"]],
    ):
        try:
            from snowflake.core import Root
            from snowflake.snowpark import Session
        except ImportError:
            raise ImportError(
                "`ChatSnowflake()` requires the `snowflake-ml-python` package. "
                "Please install it via `pip install snowflake-ml-python`."
            )
        super().__init__(name=name, model=model)

        configs: dict[str, str | int] = drop_none(
            {
                "connection_name": connection_name,
                "account": account,
                "user": user,
                "password": password,
                "private_key_file": private_key_file,
                "private_key_file_pwd": private_key_file_pwd,
                **(kwargs or {}),
            }
        )

        session = Session.builder.configs(configs).create()
        self._cortex_service = Root(session).cortex_inference_service

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["CompleteRequest"] = None,
    ): ...

    @overload
    def chat_perform(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["CompleteRequest"] = None,
    ): ...

    def chat_perform(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["CompleteRequest"] = None,
    ):
        req = self._complete_request(stream, turns, tools, data_model, kwargs)
        client = self._cortex_service.complete(req)

        try:
            events = client.events()
        except Exception as e:
            data = parse_request_object(client)
            if data is None:
                raise e
            return data

        if stream:
            return generate_event_data(events)

        for evt in events:
            if evt.data:
                return parse_event_data(evt.data, stream=False)

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[False],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["CompleteRequest"] = None,
    ): ...

    @overload
    async def chat_perform_async(
        self,
        *,
        stream: Literal[True],
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["CompleteRequest"] = None,
    ): ...

    async def chat_perform_async(
        self,
        *,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["CompleteRequest"] = None,
    ):
        req = self._complete_request(stream, turns, tools, data_model, kwargs)
        res = self._cortex_service.complete_async(req)
        # TODO: is there a way to get the SSEClient result without blocking?
        client = res.result()

        try:
            events = client.events()
        except Exception as e:
            data = parse_request_object(client)
            if data is None:
                raise e
            return data

        if stream:
            return generate_event_data_async(events)

        for evt in events:
            if evt.data:
                return parse_event_data(evt.data, stream=False)

    def _complete_request(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["CompleteRequest"] = None,
    ):
        from snowflake.core.cortex.inference_service import CompleteRequest

        req = CompleteRequest(
            model=self.model,
            messages=self._as_request_messages(turns),
            stream=stream,
        )

        if tools:
            req.tools = req.tools or []
            snow_tools = [self._as_snowflake_tool(tool) for tool in tools.values()]
            req.tools.extend(snow_tools)

        if data_model is not None:
            import snowflake.core.cortex.inference_service._generated.models as models

            params = basemodel_to_param_schema(data_model)
            req.response_format = models.CompleteRequestResponseFormat(
                type="json",
                schema={
                    "type": "object",
                    "properties": params["properties"],
                    "required": params["required"],
                },
            )

        if kwargs:
            for k, v in kwargs.items():
                if hasattr(req, k):
                    setattr(req, k, v)
                else:
                    raise ValueError(
                        f"Unknown parameter {k} for Snowflake CompleteRequest. "
                        "Please check the Snowflake documentation for valid parameters."
                    )

        return req

    def stream_text(self, chunk):
        if not chunk.choices:
            return None
        delta = chunk.choices[0].delta
        if delta is None or "content" not in delta:
            return None
        return delta["content"]

    # Snowflake sort-of follows OpenAI/Anthropic streaming formats except they
    # don't have the critical "index" field in the delta that the merge logic
    # depends on (i.e., OpenAI), or official start/stop events (i.e.,
    # Anthropic). So we have to do some janky merging here.
    #
    # This was done in a panic to get working asap, so don't judge :) I wouldn't
    # be surprised if Snowflake realizes how bad this streaming format is and
    # changes it in the future (thus probably breaking this code :( ).
    def stream_merge_chunks(self, completion, chunk):
        if completion is None:
            return chunk

        if completion.choices is None or chunk.choices is None:
            raise ValueError(
                "Unexpected None for completion.choices. Please report this issue."
            )

        if completion.choices[0].delta is None or chunk.choices[0].delta is None:
            raise ValueError(
                "Unexpected None for completion.choices[0].delta. Please report this issue."
            )

        delta = completion.choices[0].delta
        new_delta = chunk.choices[0].delta
        if "content_list" not in delta or "content_list" not in new_delta:
            raise ValueError(
                "Expected content_list to be in completion.choices[0].delta. Please report this issue."
            )

        content_list = delta["content_list"]
        new_content_list = new_delta["content_list"]
        if not isinstance(content_list, list) or not isinstance(new_content_list, list):
            raise ValueError(
                f"Expected content_list to be a list, got {type(new_content_list)}"
            )

        if new_delta["type"] == "tool_use":
            # Presence of "tool_use_id" indicates a new tool request; otherwise, we're
            # expecting input parameters
            if "tool_use_id" in new_delta:
                del new_delta["text"]  # why is this here :eye-roll:?
                content_list.append(new_delta)
            elif "input" in new_delta:
                # find most recent content with type: "tool_use" and append to that
                for i in range(len(content_list) - 1, -1, -1):
                    if "tool_use_id" in content_list[i]:
                        content_list[i]["input"] = content_list[i].get("input", "")
                        content_list[i]["input"] += new_delta["input"]
                        break
            else:
                raise ValueError(
                    f"Unexpected tool_use delta: {new_delta}. Please report this issue."
                )
        elif new_delta["type"] == "text":
            text = new_delta["text"]
            # find most recent content with type: "text" and append to that
            for i in range(len(content_list) - 1, -1, -1):
                if content_list[i].get("type") == "text":
                    content_list[i]["text"] += text
                    break
            else:
                # if we don't find it, just append to the end
                # this shouldn't happen, but just in case
                content_list.append({"type": "text", "text": text})
        else:
            raise ValueError(
                f"Unexpected streaming delta type: {new_delta['type']}. Please report this issue."
            )

        completion.choices[0].delta["content_list"] = content_list

        return completion

    def stream_turn(self, completion, has_data_model) -> Turn:
        import snowflake.core.cortex.inference_service._generated.models as models

        completion_dict = completion.model_dump()
        delta = completion_dict["choices"][0].pop("delta")
        completion_dict["choices"][0]["message"] = delta
        completion = models.NonStreamingCompleteResponse.model_construct(
            **completion_dict
        )
        return self._as_turn(completion, has_data_model)

    def value_turn(self, completion, has_data_model) -> Turn:
        return self._as_turn(completion, has_data_model)

    def token_count(
        self,
        *args: "Content | str",
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
    ) -> int:
        raise NotImplementedError(
            "Snowflake does not currently support token counting."
        )

    async def token_count_async(
        self,
        *args: "Content | str",
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
    ) -> int:
        raise NotImplementedError(
            "Snowflake does not currently support token counting."
        )

    def _as_request_messages(self, turns: list[Turn]):
        from snowflake.core.cortex.inference_service import CompleteRequestMessagesInner

        res: list[CompleteRequestMessagesInner] = []
        for turn in turns:
            req = CompleteRequestMessagesInner(
                role=turn.role,
                content=turn.text,
            )
            for x in turn.contents:
                if isinstance(x, ContentToolRequest):
                    req.content_list = req.content_list or []
                    req.content_list.append(
                        {
                            "type": "tool_use",
                            "tool_use": {
                                "tool_use_id": x.id,
                                "name": x.name,
                                "input": x.arguments,
                            },
                        }
                    )
                elif isinstance(x, ContentToolResult):
                    # Snowflake does like empty content
                    req.content = req.content or "[tool_result]"
                    req.content_list = req.content_list or []
                    req.content_list.append(
                        {
                            "type": "tool_results",
                            "tool_results": {
                                "tool_use_id": x.id,
                                "name": x.name,
                                "content": [
                                    {"type": "text", "text": x.get_model_value()}
                                ],
                            },
                        }
                    )
                elif isinstance(x, ContentJson):
                    req.content = req.content or "<structured data/>"

            res.append(req)
        return res

    def _as_turn(self, completion: "Completion", has_data_model: bool) -> Turn:
        import snowflake.core.cortex.inference_service._generated.models as models

        if not completion.choices:
            return Turn("assistant", [])

        choice = completion.choices[0]
        if isinstance(choice, dict):
            choice = models.NonStreamingCompleteResponseChoicesInner.from_dict(choice)

        message = choice.message
        if message is None:
            return Turn("assistant", [])

        contents: list[Content] = []
        content_list = message.content_list or []
        for content in content_list:
            if "text" in content:
                if has_data_model:
                    data = orjson.loads(content["text"])
                    contents.append(ContentJson(value=data))
                else:
                    contents.append(ContentText(text=content["text"]))
            elif "tool_use_id" in content:
                params = content.get("input", "{}")
                try:
                    params = orjson.loads(params)
                except orjson.JSONDecodeError:
                    raise ValueError(
                        f"Failed to parse tool_use input: {params}. Please report this issue."
                    )
                contents.append(
                    ContentToolRequest(
                        name=content["name"],
                        id=content["tool_use_id"],
                        arguments=params,
                    )
                )

        usage = completion.usage
        if usage is None:
            tokens = (0, 0)
        else:
            tokens = (usage.prompt_tokens or 0, usage.completion_tokens or 0)

        tokens_log(self, tokens)

        return Turn(
            "assistant",
            contents,
            tokens=tokens,
            # TODO: no finish_reason in Snowflake?
            # finish_reason=completion.choices[0].finish_reason,
            completion=completion,
        )

    # N.B. this is currently the best documentation I can find for how tool calling works
    # https://quickstarts.snowflake.com/guide/getting-started-with-tool-use-on-cortex-and-anthropic-claude/index.html#5
    def _as_snowflake_tool(self, tool: Tool):
        import snowflake.core.cortex.inference_service._generated.models as models

        func = tool.schema["function"]
        params = func.get("parameters", {})

        props = params.get("properties", {})
        if not isinstance(props, dict):
            raise ValueError(
                f"Tool function parameters must be a dictionary, got {type(props)}"
            )

        required = params.get("required", [])
        if not isinstance(required, list):
            raise ValueError(
                f"Tool function required parameters must be a list, got {type(required)}"
            )

        input_schema = models.ToolToolSpecInputSchema(
            type="object",
            properties=props or None,
            required=required or None,
        )

        spec = models.ToolToolSpec(
            type="generic",
            name=func["name"],
            description=func.get("description", ""),
            input_schema=input_schema,
        )

        return models.Tool(tool_spec=spec)

    def translate_model_params(self, params: StandardModelParams) -> "CompleteRequest":
        res: "CompleteRequest" = {}
        if "temperature" in params:
            res["temperature"] = params["temperature"]

        if "top_p" in params:
            res["top_p"] = params["top_p"]

        if "max_tokens" in params:
            res["max_tokens"] = params["max_tokens"]

        return res

    def supported_model_params(self) -> set[StandardModelParamNames]:
        return {
            "temperature",
            "top_p",
            "max_tokens",
        }


# Yield parsed event data from the Snowflake SSEClient
# (this is only needed for the streaming case).
def generate_event_data(events: Generator["Event", None, None]):
    for x in events:
        if x.data:
            yield parse_event_data(x.data, stream=True)


# Same thing for the async case.
async def generate_event_data_async(events: Generator["Event", None, None]):
    for x in events:
        if x.data:
            yield parse_event_data(x.data, stream=True)


@overload
def parse_event_data(
    data: str, stream: Literal[True]
) -> "models.StreamingCompleteResponseDataEvent": ...


@overload
def parse_event_data(
    data: str, stream: Literal[False]
) -> "models.NonStreamingCompleteResponse": ...


def parse_event_data(
    data: str, stream: bool
) -> "models.NonStreamingCompleteResponse | models.StreamingCompleteResponseDataEvent":
    "Parse the (JSON) event data from Snowflake using the relevant pydantic model."
    import snowflake.core.cortex.inference_service._generated.models as models

    try:
        if stream:
            return models.StreamingCompleteResponseDataEvent.from_json(data)
        else:
            return models.NonStreamingCompleteResponse.from_json(data)
    except Exception:
        raise ValueError(
            f"Failed to parse Snowflake event data: {data}. "
            "Please report this error here: https://github.com/posit-dev/chatlas/issues/new"
        )


# At the time writing, .events() flat out errors in the stream=False case since
# the Content-Type is set to application/json;charset=utf-8, and SSEClient
# doesn't know how to handle that.
# https://github.com/snowflakedb/snowflake-ml-python/blob/6910e96/snowflake/cortex/_sse_client.py#L69
#
# So, do some janky stuff here to get the data out of the response.
#
# If and when snowflake fixes this, we can remove the try/except block.
def parse_request_object(
    client: "SSEClient",
) -> "Optional[models.NonStreamingCompleteResponse]":
    try:
        import urllib3

        if isinstance(client._event_source, urllib3.response.HTTPResponse):
            return parse_event_data(
                client._event_source.data.decode("utf-8"),
                stream=False,
            )
    except Exception:
        pass

    return None
