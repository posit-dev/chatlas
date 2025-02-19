from typing import TYPE_CHECKING, Any, Literal, Optional, TypedDict, cast, overload

from pydantic import BaseModel

from ._chat import Chat
from ._content import Content
from ._logging import log_model_default
from ._provider import Provider
from ._tools import Tool
from ._turn import Turn, normalize_turns
from ._typing_extensions import NotRequired

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


class ConnectionToml(TypedDict):
    """
    Connect by using the connections.toml file

    As described in the Snowpark documentation:
    https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session#connect-by-using-the-connections-toml-file
    """

    connection_name: str
    "The name of the connection (i.e., section) within the connections.toml file"


class Connection(TypedDict):
    """
    Connect by specifying connection parameters

    As described in the Snowpark documentation:
    https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session#connect-by-specifying-connection-parameters
    """

    account: str
    """
    Your snowflake account identifier

    As described in the Snowpark documentation:
    https://docs.snowflake.com/en/user-guide/admin-account-identifier
    """

    user: str
    "Your snowflake user name"

    password: str
    "Your snowflake password"

    role: NotRequired[str]
    "Your snowflake role"

    warehouse: NotRequired[str]
    "Your snowflake warehouse"

    database: NotRequired[str]
    "Your snowflake database"

    schema: NotRequired[str]
    "Your snowflake schema"


class ConnectionSSO(TypedDict):
    """
    Use single sign-on (SSO) through a web browser

    As described in the Snowpark documentation:
    https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-session#using-single-sign-on-sso-through-a-web-browser
    """

    account: str
    """
    Your snowflake account identifier

    As described in the Snowpark documentation:
    https://docs.snowflake.com/en/user-guide/admin-account-identifier
    """

    user: str
    "Your snowflake user name"

    role: str
    "Your snowflake role"

    database: str
    "Your snowflake database"

    schema: str
    "Your snowflake schema"

    warehouse: str
    "Your snowflake warehouse"

    authenticator: Literal["externalbrowser"]


def ChatSnowflake(
    *,
    connection_config: ConnectionToml | Connection | ConnectionSSO,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    turns: Optional[list[Turn]] = None,
) -> Chat["SubmitInputArgs", Completion]:
    """
    Chat with a model hosted on Snowflake

    Parameters
    ----------
    connection_config
        The connection configuration to use. This can be either a `ConnectionToml`,
        `Connection`, or `ConnectionSSO` dictionary.
    system_prompt
        A system prompt to set the behavior of the assistant.
    model
        The model to use for the chat. The default, None, will pick a reasonable
        default, and warn you about it. We strongly recommend explicitly
        choosing a model for all but the most casual use.
    """

    if model is None:
        model = log_model_default("llama3.1-70b")

    return Chat(
        provider=SnowflakeProvider(
            connection_config=connection_config,
            model=model,
        ),
        turns=normalize_turns(
            turns or [],
            system_prompt,
        ),
    )


class SnowflakeProvider(Provider[Completion, CompletionChunk, CompletionChunk]):
    def __init__(
        self,
        *,
        model: str,
        connection_config: ConnectionToml | Connection | ConnectionSSO,
    ):
        try:
            from snowflake.snowpark import Session
        except ImportError:
            raise ImportError(
                "`ChatSnowflake()` requires the `snowflake-ml-python` package. "
                "Please install it via `pip install snowflake-ml-python`."
            )

        config = cast(dict[str, Any], connection_config)

        self._model = model
        self._session = Session.builder.configs(config).create()

    def __del__(self):
        self._session.close()

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
