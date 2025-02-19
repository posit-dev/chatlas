from typing import TYPE_CHECKING, Literal, Optional, TypedDict, overload

from pydantic import BaseModel

from ._chat import Chat
from ._content import Content
from ._logging import log_model_default
from ._provider import Provider
from ._tools import Tool
from ._turn import Turn, normalize_turns
from ._utils import drop_none

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
    role: Optional[str] = None,
    warehouse: Optional[str] = None,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    authenticator: Optional[str] = None,
) -> Chat["SubmitInputArgs", "Completion"]:
    """
    Chat with a Snowflake Cortex LLM

    Prerequisites
    -------------

    ::: {.callout-note}
    ## Snowflake credentials

    Snowflake provides at least a few ways to authenticate. You can use a
    `connections.toml` file (and specify the `connection_name` argument), specify the
    connection parameters directly (with `account`, `user`, `password`, etc.),
    or use single sign-on (SSO) through a web browser.

    For more information, see the Snowflake documentation:
    https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session
    :::

    ::: {.callout-note}
    ## Python requirements

    `ChatSnowflake`, requires the `snowflake-ml-python` package
    (e.g., `pip install snowflake-ml-python`).
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
    account
        Your Snowflake account identifier. Required if `connection_name` is not provided.
        https://docs.snowflake.com/en/user-guide/admin-account-identifier
    user
        Your Snowflake user name. Required if `connection_name` is not provided.
    password
        Your Snowflake password. Required if `connection_name` is not provided and
        you are not using single sign-on (SSO).
    role
        Your Snowflake role.
    warehouse
        Your Snowflake warehouse.
    database
        Your Snowflake database.
    schema
        Your Snowflake schema.
    authenticator
        The authenticator to use. Only required if you are using single sign-on (SSO).
        The only supported value in this case is "externalbrowser".
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
            role=role,
            warehouse=warehouse,
            database=database,
            schema=schema,
            authenticator=authenticator,
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
        connection_name: str | None,
        account: str | None,
        user: str | None,
        password: str | None,
        role: str | None,
        warehouse: str | None,
        database: str | None,
        schema: str | None,
        authenticator: str | None,
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
                "role": role,
                "warehouse": warehouse,
                "database": database,
                "schema": schema,
                "authenticator": authenticator,
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
        raise NotImplementedError(
            "Snowflake does not currently support async completions."
        )

    def _chat_perform_args(
        self,
        stream: bool,
        turns: list[Turn],
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]] = None,
        kwargs: Optional["SubmitInputArgs"] = None,
    ):
        # Cortex doesn't seem to support tools
        if tools:
            raise ValueError("Snowflake does not currently support tools.")

        # TODO: implement data_model?
        # https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api#structured-output-example
        # https://github.com/snowflakedb/snowflake-ml-python/pull/141

        kwargs_full: "SubmitInputArgs" = {
            "stream": stream,
            "prompt": self._as_prompt_input(turns),
            "model": self._model,
            **(kwargs or {}),
        }

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
        *args: Content | str,
        tools: dict[str, Tool],
        data_model: Optional[type[BaseModel]],
    ) -> int:
        raise NotImplementedError(
            "Snowflake does not currently support token counting."
        )

    async def token_count_async(
        self,
        *args: Content | str,
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
                    "content": turn.text,
                }
            )
        return res

    def _as_turn(self, completion, has_data_model) -> Turn:
        return Turn("assistant", completion)
