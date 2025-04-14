import asyncio
import json
from typing import TYPE_CHECKING, Iterable, Literal, Optional, TypedDict, cast, overload

from pydantic import BaseModel

from ._chat import Chat
from ._content import Content, ContentJson, ContentText
from ._logging import log_model_default
from ._provider import Provider
from ._tools import Tool, basemodel_to_param_schema
from ._turn import Turn, normalize_turns
from ._utils import drop_none, wrap_async_iterable

if TYPE_CHECKING:
    from snowflake.snowpark import Column

    # Types inferred from the return type of the `snowflake.cortex.complete` function
    Completion = str | Column
    CompletionChunk = str

    from .types.snowflake import SubmitInputArgs


# The main prompt input type for Snowflake
# This was copy-pasted from `snowflake.cortex._complete.ConversationMessage`
class ConversationMessage(TypedDict):
    role: str
    content: str


def ChatSnowflake(
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
    connection_name: Optional[str] = None,
    account: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    private_key_file: Optional[str] = None,
    private_key_file_pwd: Optional[str] = None,
    kwargs: Optional[dict[str, "str | int"]] = None,
) -> Chat["SubmitInputArgs", "Completion"]:
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
    turns
        A list of turns to start the chat with (i.e., continuing a previous
        conversation). If not provided, the conversation begins from scratch. Do
        not provide non-None values for both `turns` and `system_prompt`. Each
        message in the list should be a dictionary with at least `role` (usually
        `system`, `user`, or `assistant`, but `tool` is also possible). Normally
        there is also a `content` field, which is a string.
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
        model = log_model_default("llama3.1-70b")

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
        turns=normalize_turns(
            turns or [],
            system_prompt,
        ),
    )


class SnowflakeProvider(Provider["Completion", "CompletionChunk", "CompletionChunk"]):
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
        kwargs: Optional[dict[str, "str | int"]],
    ):
        try:
            from snowflake.snowpark import Session
        except ImportError:
            raise ImportError(
                "`ChatSnowflake()` requires the `snowflake-ml-python` package. "
                "Please install it via `pip install snowflake-ml-python`."
            )

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

        self._model = model
        self._session = Session.builder.configs(configs).create()

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
        from snowflake.cortex import complete

        kwargs = self._chat_perform_args(stream, turns, tools, data_model, kwargs)
        return complete(**kwargs)

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
        from snowflake.cortex import complete

        kwargs = self._chat_perform_args(stream, turns, tools, data_model, kwargs)

        # Prevent the main thread from being blocked (Snowflake doesn't have native async support)
        res = await asyncio.to_thread(complete, **kwargs)

        # When streaming, res is an iterable of strings, but Chat() wants an async iterable
        if stream:
            res = wrap_async_iterable(cast(Iterable[str], res))

        return res

    def _chat_perform_args(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ):
        kwargs_full: "SubmitInputArgs" = {
            "stream": stream,
            "prompt": self._as_prompt_input(turns),
            "model": self._model,
            "session": self._session,
            **(kwargs or {}),
        }

        # TODO: get tools working
        if tools:
            raise ValueError("Snowflake does not currently support tools.")

        if data_model is not None:
            params = basemodel_to_param_schema(data_model)
            opts = kwargs_full.get("options") or {}
            opts["response_format"] = {
                "type": "json",
                "schema": {
                    "type": "object",
                    "properties": params["properties"],
                    "required": params["required"],
                },
            }
            kwargs_full["options"] = opts

        return kwargs_full

    def stream_text(self, chunk):
        return chunk

    def stream_merge_chunks(self, completion, chunk):
        if completion is None:
            return chunk
        return completion + chunk

    def stream_turn(self, completion, has_data_model) -> Turn:
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

    def _as_prompt_input(self, turns: list[Turn]) -> list["ConversationMessage"]:
        res: list["ConversationMessage"] = []
        for turn in turns:
            res.append(
                {
                    "role": turn.role,
                    "content": str(turn),
                }
            )
        return res

    def _as_turn(self, completion, has_data_model) -> Turn:
        completion = cast(str, completion)

        if has_data_model:
            data = json.loads(completion)
            contents = [ContentJson(value=data)]
        else:
            contents = [ContentText(text=completion)]

        return Turn("assistant", contents)
