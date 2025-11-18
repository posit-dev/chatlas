from __future__ import annotations

import copy
import inspect
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Generic,
    Iterator,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    cast,
    overload,
)

from pydantic import BaseModel

from ._callbacks import CallbackManager
from ._content import (
    Content,
    ContentJson,
    ContentText,
    ContentToolRequest,
    ContentToolResult,
    ToolInfo,
)
from ._display import (
    EchoDisplayOptions,
    IPyMarkdownDisplay,
    LiveMarkdownDisplay,
    MarkdownDisplay,
    MockMarkdownDisplay,
)
from ._logging import log_tool_error
from ._mcp_manager import MCPSessionManager
from ._provider import ModelInfo, Provider, StandardModelParams, SubmitInputArgsT
from ._tokens import compute_cost, get_token_pricing, tokens_log
from ._tools import Tool, ToolRejectError
from ._turn import AssistantTurn, SystemTurn, Turn, UserTurn, user_turn
from ._typing_extensions import TypedDict, TypeGuard
from ._utils import MISSING, MISSING_TYPE, html_escape, wrap_async

if TYPE_CHECKING:
    from inspect_ai.model import ChatMessage as InspectChatMessage
    from inspect_ai.solver import TaskState as InspectTaskState

    from ._content import ToolAnnotations


class TokensDict(TypedDict):
    """
    A TypedDict representing the token counts for a turn in the chat.
    This is used to represent the token counts for each turn in the chat.
        `role` represents the role of the turn (i.e., "user" or "assistant").
        `tokens` represents the new tokens used in the turn.
        `tokens_total` represents the total tokens used in the turn.
        Ex. A new user input of 2 tokens is sent, plus 10 tokens of context from prior turns (input and output).
         This would have a `tokens_total` of 12.
    """

    role: Literal["user", "assistant"]
    tokens: int
    tokens_total: int
    tokens_cached: int


CompletionT = TypeVar("CompletionT")

EchoOptions = Literal["output", "all", "none", "text"]

T = TypeVar("T")
BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


def is_present(value: T | None | MISSING_TYPE) -> TypeGuard[T]:
    return value is not None and not isinstance(value, MISSING_TYPE)


class Chat(Generic[SubmitInputArgsT, CompletionT]):
    """
    A chat object that can be used to interact with a language model.

    A `Chat` is an sequence of sequence of user and assistant
    [](`~chatlas.Turn`)s sent to a specific [](`~chatlas.Provider`). A `Chat`
    takes care of managing the state associated with the chat; i.e. it records
    the messages that you send to the server, and the messages that you receive
    back. If you register a tool (i.e. an function that the assistant can call
    on your behalf), it also takes care of the tool loop.

    You should generally not create this object yourself, but instead call
    [](`~chatlas.ChatOpenAI`) or friends instead.
    """

    def __init__(
        self,
        provider: Provider,
        system_prompt: Optional[str] = None,
        kwargs_chat: Optional[SubmitInputArgsT] = None,
    ):
        """
        Create a new chat object.

        Parameters
        ----------
        provider
            A [](`~chatlas.Provider`) object.
        system_prompt
            A system prompt to set the behavior of the assistant.
        kwargs_chat
            Additional arguments to pass to the provider when submitting input.
            These arguments persist across all chat interactions and will be
            merged with any kwargs passed to individual methods like `chat()` or
            `stream()`. They also take precedence over any parameters set via
            `set_model_params()`.
        """
        self.provider = provider
        self._turns: list[Turn] = []
        self.system_prompt = system_prompt
        self.kwargs_chat: SubmitInputArgsT = kwargs_chat or {}

        self._tools: dict[str, Tool] = {}
        self._on_tool_request_callbacks = CallbackManager()
        self._on_tool_result_callbacks = CallbackManager()
        self._current_display: Optional[MarkdownDisplay] = None
        self._echo_options: EchoDisplayOptions = {
            "rich_markdown": {},
            "rich_console": {},
            "css_styles": {},
        }
        self._mcp_manager = MCPSessionManager()

        # Chat input parameters from `set_model_params()`
        self._standard_model_params: StandardModelParams = {}

    def list_models(self) -> list[ModelInfo]:
        """
        List all models available for the provider.

        This method returns detailed information about all models supported by the provider,
        including model IDs, pricing information, creation dates, and other metadata. This is
        useful for discovering available models and their characteristics without needing to
        consult provider documentation.

        Examples
        --------
        Get all available models:

        ```python
        from chatlas import ChatOpenAI

        chat = ChatOpenAI()
        models = chat.list_models()
        print(f"Found {len(models)} models")
        print(f"First model: {models[0]['id']}")
        ```

        View models in a table format:

        ```python
        import pandas as pd
        from chatlas import ChatAnthropic

        chat = ChatAnthropic()
        df = pd.DataFrame(chat.list_models())
        print(df[["id", "input", "output"]].head())  # Show pricing info
        ```

        Find models by criteria:

        ```python
        from chatlas import ChatGoogle

        chat = ChatGoogle()
        models = chat.list_models()

        # Find cheapest input model
        cheapest = min(models, key=lambda m: m.get("input", float("inf")))
        print(f"Cheapest model: {cheapest['id']}")
        ```

        Returns
        -------
        list[ModelInfo]
            A list of ModelInfo dictionaries containing model information. Each dictionary
            contains:

            - `id` (str): The model identifier to use with the Chat constructor
            - `name` (str, optional): Human-readable model name
            - `input` (float, optional): Cost per input token in USD per million tokens
            - `output` (float, optional): Cost per output token in USD per million tokens
            - `cached_input` (float, optional): Cost per cached input token in USD per million tokens
            - `created_at` (date, optional): Date the model was created
            - `owned_by` (str, optional): Organization that owns the model
            - `provider` (str, optional): Model provider name
            - `size` (int, optional): Model size in bytes
            - `url` (str, optional): URL with more information about the model

            The list is typically sorted by creation date (most recent first).

        Note
        ----
        Not all providers support this method. Some providers may raise NotImplementedError
        with information about where to find model listings online.
        """
        return self.provider.list_models()

    def get_turns(
        self,
        *,
        include_system_prompt: bool = False,
        tool_result_role: Literal["assistant", "user"] = "user",
    ) -> list[Turn]:
        """
        Get all the turns (i.e., message contents) in the chat.

        Parameters
        ----------
        include_system_prompt
            Whether to include the system prompt in the turns.
        tool_result_role
            The role to assign to turns containing tool results. By default,
            tool results are assigned a role of "user" since they represent
            information provided to the assistant. If set to "assistant" tool
            result content (plus the surrounding assistant turn contents) is
            collected into a single assistant turn. This is convenient for
            display purposes and more generally if you want the tool calling
            loop to be contained in a single turn.
        """

        if not self._turns:
            return self._turns

        if not include_system_prompt and isinstance(self._turns[0], SystemTurn):
            turns = self._turns[1:]
        else:
            turns = self._turns

        if tool_result_role == "user":
            return turns

        if tool_result_role != "assistant":
            raise ValueError(
                f"Expected `tool_result_role` to be one of 'user' or 'assistant', not '{tool_result_role}'"
            )

        # If a turn is purely a tool result, change its role
        turns2: list[Turn] = []
        for turn in turns:
            turn2 = turn
            if all(isinstance(c, ContentToolResult) for c in turn.contents):
                if tool_result_role == "assistant":
                    turn2 = AssistantTurn(contents=turn.contents)
            turns2.append(turn2)

        # If two consecutive turns have the same role (i.e., assistant), collapse them into one
        final_turns: list[Turn] = []
        for x in turns2:
            if not final_turns:
                final_turns.append(x)
                continue
            if x.role != final_turns[-1].role:
                final_turns.append(x)
            else:
                final_turns[-1].contents.extend(x.contents)

        return final_turns

    @overload
    def get_last_turn(self) -> AssistantTurn[CompletionT] | None: ...

    @overload
    def get_last_turn(
        self, *, role: Literal["assistant"]
    ) -> AssistantTurn[CompletionT] | None: ...

    @overload
    def get_last_turn(self, *, role: Literal["user"]) -> UserTurn | None: ...

    @overload
    def get_last_turn(self, *, role: Literal["system"]) -> SystemTurn | None: ...

    def get_last_turn(
        self,
        *,
        role: Literal["assistant", "user", "system"] = "assistant",
    ) -> Turn | None:
        """
        Get the last turn in the chat with a specific role.

        Parameters
        ----------
        role
            The role of the turn to return.
        """
        for turn in reversed(self._turns):
            if turn.role == role:
                return turn
        return None

    def set_turns(self, turns: Sequence[Turn]):
        """
        Set the turns of the chat.

        Replaces the current chat history state (i.e., turns) with the provided turns.
        This can be useful for:
            * Clearing (or trimming) the chat history (i.e., `.set_turns([])`).
            * Restoring context from a previous chat.

        Parameters
        ----------
        turns
            The turns to set. Turns with the role "system" are not allowed.
        """
        if any(isinstance(x, SystemTurn) for x in turns):
            idx = next(i for i, x in enumerate(turns) if isinstance(x, SystemTurn))
            raise ValueError(
                f"Turn {idx} has a role 'system', which is not allowed. "
                "The system prompt must be set separately using the `.system_prompt` property. "
                "Consider removing this turn and setting the `.system_prompt` separately "
                "if you want to change the system prompt."
            )

        turns_list = list(turns)
        # Preserve the system prompt if it exists
        if self._turns and isinstance(self._turns[0], SystemTurn):
            turns_list.insert(0, self._turns[0])
        self._turns = turns_list

    def add_turn(self, turn: Turn):
        """
        Add a turn to the chat.

        Parameters
        ----------
        turn
            The turn to add. Turns with the role "system" are not allowed.
        """
        if isinstance(turn, SystemTurn):
            raise ValueError(
                "Turns with the role 'system' are not allowed. "
                "The system prompt must be set separately using the `.system_prompt` property."
            )
        self._turns.append(turn)

    @property
    def system_prompt(self) -> str | None:
        """
        A property to get (or set) the system prompt for the chat.

        Returns
        -------
        str | None
            The system prompt (if any).
        """
        if self._turns and isinstance(self._turns[0], SystemTurn):
            return self._turns[0].text
        return None

    @system_prompt.setter
    def system_prompt(self, value: str | None):
        if self._turns and isinstance(self._turns[0], SystemTurn):
            self._turns.pop(0)
        if value is not None:
            self._turns.insert(0, SystemTurn(value))

    def get_tokens(self) -> list[TokensDict]:
        """
        Get the tokens for each turn in the chat.

        Returns
        -------
        list[TokensDict]
             A list of dictionaries with the token counts for each (non-system) turn

        Raises
        ------
        ValueError
            If the chat's turns (i.e., `.get_turns()`) are not in an expected
            format. This may happen if the chat history is manually set (i.e.,
            `.set_turns()`). In this case, you can inspect the "raw" token
            values via the `.get_turns()` method (each turn has a `.tokens`
            attribute).
        """

        turns = self.get_turns(include_system_prompt=False)

        if len(turns) == 0:
            return []

        err_info = (
            "This can happen if the chat history is manually set (i.e., `.set_turns()`). "
            "Consider getting the 'raw' token values via the `.get_turns()` method "
            "(each turn has a `.tokens` attribute)."
        )

        # Sanity checks for the assumptions made to figure out user token counts
        if len(turns) == 1:
            raise ValueError(
                "Expected at least two turns in the chat history. " + err_info
            )

        if len(turns) % 2 != 0:
            raise ValueError(
                "Expected an even number of turns in the chat history. " + err_info
            )

        if turns[0].role != "user":
            raise ValueError(
                "Expected the 1st non-system turn to have role='user'. " + err_info
            )

        if not isinstance(turns[1], AssistantTurn):
            raise ValueError(
                "Expected the 2nd turn non-system to have role='assistant'. " + err_info
            )

        if turns[1].tokens is None:
            raise ValueError(
                "Expected the 1st assistant turn to contain token counts. " + err_info
            )

        res: list[TokensDict] = [
            # Implied token count for the 1st user input
            {
                "role": "user",
                "tokens": turns[1].tokens[0],
                # Number of tokens currently cached (reduces input token usage)
                "tokens_cached": turns[1].tokens[2],
                "tokens_total": turns[1].tokens[0],
            },
            # The token count for the 1st assistant response
            {
                "role": "assistant",
                "tokens": turns[1].tokens[1],
                "tokens_cached": 0,
                "tokens_total": turns[1].tokens[1],
            },
        ]

        for i in range(1, len(turns) - 1, 2):
            ti = turns[i]
            tj = turns[i + 2]
            if not isinstance(ti, AssistantTurn) or not isinstance(tj, AssistantTurn):
                raise ValueError(
                    "Expected even turns to have role='assistant'." + err_info
                )
            if ti.tokens is None or tj.tokens is None:
                raise ValueError(
                    "Expected role='assistant' turns to contain token counts."
                    + err_info
                )
            res.extend(
                [
                    {
                        "role": "user",
                        # Implied new token count for the user input (input tokens - context - cached reads)
                        # Cached reads are only subtracted for particular providers
                        "tokens": tj.tokens[0] - sum(ti.tokens),
                        # Number of tokens currently cached (reduces input token usage depending on provider's API)
                        "tokens_cached": tj.tokens[2],
                        # Total tokens = Total User Tokens for the Turn = Distinct new tokens + context sent
                        "tokens_total": tj.tokens[0],
                    },
                    {
                        "role": "assistant",
                        # The token count for the assistant response
                        "tokens": tj.tokens[1],
                        # Total tokens = Total Assistant tokens used in the turn
                        "tokens_cached": 0,
                        "tokens_total": tj.tokens[1],
                    },
                ]
            )

        return res

    def get_cost(
        self,
        options: Literal["all", "last"] = "all",
        token_price: Optional[tuple[float, float, float]] = None,
    ) -> float:
        """
        Estimate the cost of the chat.

        Note
        ----
        This is a rough estimate, treat it as such. Providers may change their
        pricing frequently and without notice.

        Parameters
        ----------
        options
            One of the following (default is "all"):
              - `"all"`: Return the total cost of all turns in the chat.
              - `"last"`: Return the cost of the last turn in the chat.
        token_price
            An optional tuple in the format of (input_token_cost,
            output_token_cost, cached_token_cost) for bringing your own cost information.
                 - `"input_token_cost"`: The cost per user token in USD per
                   million tokens.
                 - `"output_token_cost"`: The cost per assistant token in USD
                   per million tokens.
                - `"cached_token_cost"`: The cost per cached token read in USD
                   per million tokens.

        Returns
        -------
        float
            The cost of the chat, in USD.
        """

        # Look up token cost for user and input tokens based on the provider and model
        turns_tokens = self.get_tokens()
        if token_price:
            input_token_price = token_price[0] / 1e6
            output_token_price = token_price[1] / 1e6
            cached_token_price = token_price[2] / 1e6
        else:
            price_token = get_token_pricing(self.provider.name, self.provider.model)
            if not price_token:
                raise KeyError(
                    f"We could not locate pricing information for model '{self.provider.model}'"
                    f" from provider '{self.provider.name}'. "
                    "If you know the pricing for this model, specify it in `token_price`."
                )

            input_token_price = price_token["input"] / 1e6
            output_token_price = price_token.get("output", 0) / 1e6
            cached_token_price = price_token.get("cached_input", 0) / 1e6

        if len(turns_tokens) == 0:
            return 0.0

        if options not in ("all", "last"):
            raise ValueError(
                f"Expected `options` to be one of 'all' or 'last', not '{options}'"
            )

        if options == "all":
            asst_tokens = sum(
                u["tokens_total"] for u in turns_tokens if u["role"] == "assistant"
            )
            user_tokens = sum(
                u["tokens_total"] for u in turns_tokens if u["role"] == "user"
            )
            # We add the cached tokens here because for relevant providers they have already been subtracted
            # from the user tokens. This assumes the provider uses (reads) the cache each time.
            cached_token_reads = sum(
                u["tokens_cached"] for u in turns_tokens if u["role"] == "user"
            )

            cost = (
                (asst_tokens * output_token_price)
                + (user_tokens * input_token_price)
                + (cached_token_reads * cached_token_price)
            )
            return cost

        last_turn = turns_tokens[-1]
        if last_turn["role"] == "assistant":
            return last_turn["tokens"] * output_token_price
        if last_turn["role"] == "user":
            return (last_turn["tokens_total"] * input_token_price) + (
                last_turn["tokens_cached"] * cached_token_price
            )
        raise ValueError(
            f"Expected last turn to have a role of 'user' or `'assistant'`, not '{last_turn['role']}'"
        )

    def token_count(
        self,
        *args: Content | str,
        data_model: Optional[type[BaseModel]] = None,
    ) -> int:
        """
        Get an estimated token count for the given input.

        Estimate the token size of input content. This can help determine whether input(s)
        and/or conversation history (i.e., `.get_turns()`) should be reduced in size before
        sending it to the model.

        Parameters
        ----------
        args
            The input to get a token count for.
        data_model
            If the input is meant for data extraction (i.e., `.chat_structured()`), then
            this should be the Pydantic model that describes the structure of the data to
            extract.

        Returns
        -------
        int
            The token count for the input.

        Note
        ----
        Remember that the token count is an estimate. Also, models based on
        `ChatOpenAI()` currently does not take tools into account when
        estimating token counts.

        Examples
        --------
        ```python
        from chatlas import ChatAnthropic

        chat = ChatAnthropic()
        # Estimate the token count before sending the input
        print(chat.token_count("What is 2 + 2?"))

        # Once input is sent, you can get the actual input and output
        # token counts from the chat object
        chat.chat("What is 2 + 2?", echo="none")
        print(chat.token_usage())
        ```
        """

        return self.provider.token_count(
            *args,
            tools=self._tools,
            data_model=data_model,
        )

    async def token_count_async(
        self,
        *args: Content | str,
        data_model: Optional[type[BaseModel]] = None,
    ) -> int:
        """
        Get an estimated token count for the given input asynchronously.

        Estimate the token size of input content. This can help determine whether input(s)
        and/or conversation history (i.e., `.get_turns()`) should be reduced in size before
        sending it to the model.

        Parameters
        ----------
        args
            The input to get a token count for.
        data_model
            If this input is meant for data extraction (i.e., `.chat_structured_async()`),
            then this should be the Pydantic model that describes the structure of the data
            to extract.

        Returns
        -------
        int
            The token count for the input.
        """

        return await self.provider.token_count_async(
            *args,
            tools=self._tools,
            data_model=data_model,
        )

    def app(
        self,
        *,
        stream: bool = True,
        port: int = 0,
        host: str = "127.0.0.1",
        launch_browser: bool = True,
        bookmark_store: Literal["url", "server", "disable"] = "url",
        bg_thread: Optional[bool] = None,
        echo: Optional[EchoOptions] = None,
        content: Literal["text", "all"] = "all",
        kwargs: Optional[SubmitInputArgsT] = None,
    ):
        """
        Enter a web-based chat app to interact with the LLM.

        Parameters
        ----------
        stream
            Whether to stream the response (i.e., have the response appear in chunks).
        port
            The port to run the app on (the default is 0, which will choose a random port).
        host
            The host to run the app on (the default is "127.0.0.1").
        launch_browser
            Whether to launch a browser window.
        bookmark_store
            One of the following (default is "url"):
              - `"url"`: Store bookmarks in the URL (default).
              - `"server"`: Store bookmarks on the server (requires a server-side
                storage backend).
              - `"disable"`: Disable bookmarking.
        bg_thread
            Whether to run the app in a background thread. If `None`, the app will
            run in a background thread if the current environment is a notebook.
        echo
            One of the following (defaults to `"none"` when `stream=True` and `"text"` when
            `stream=False`):
              - `"text"`: Echo just the text content of the response.
              - `"output"`: Echo text and tool call content.
              - `"all"`: Echo both the assistant and user turn.
              - `"none"`: Do not echo any content.
        content
            Whether to display text content or all content (i.e., tool calls).
        kwargs
            Additional keyword arguments to pass to the method used for requesting
            the response.
        """

        try:
            from shiny import App, run_app, ui
        except ImportError:
            raise ImportError(
                "The `shiny` package is required for the `app()` method. "
                "Install it with `pip install shiny`."
            )

        try:
            from shinychat import (
                Chat,
                chat_ui,
                message_content,  # pyright: ignore[reportAttributeAccessIssue]
            )
        except ImportError:
            raise ImportError(
                "The `shinychat` package is required for the `app()` method. "
                "Install it with `pip install shinychat`."
            )

        messages = [
            message_content(x) for x in self.get_turns(tool_result_role="assistant")
        ]

        def app_ui(x):
            return ui.page_fillable(
                chat_ui("chat", messages=messages),
                fillable_mobile=True,
            )

        def server(input):  # noqa: A002
            chat = Chat("chat")

            chat.enable_bookmarking(self)

            @chat.on_user_submit
            async def _(user_input: str):
                if stream:
                    await chat.append_message_stream(
                        await self.stream_async(
                            user_input,
                            kwargs=kwargs,
                            echo=echo or "none",
                            content=content,
                        )
                    )
                else:
                    await chat.append_message(
                        str(
                            self.chat(
                                user_input,
                                kwargs=kwargs,
                                stream=False,
                                echo=echo or "text",
                            )
                        )
                    )

        app = App(app_ui, server, bookmark_store=bookmark_store)

        def _run_app():
            run_app(app, launch_browser=launch_browser, port=port, host=host)

        # Use bg_thread by default in Jupyter and Positron
        if bg_thread is None:
            from rich.console import Console

            console = Console()
            bg_thread = console.is_jupyter or (os.getenv("POSITRON") == "1")

        if bg_thread:
            thread = Thread(target=_run_app, daemon=True)
            thread.start()
        else:
            _run_app()

        return None

    def console(
        self,
        *,
        echo: EchoOptions = "output",
        stream: bool = True,
        kwargs: Optional[SubmitInputArgsT] = None,
    ):
        """
        Enter a chat console to interact with the LLM.

        To quit, input 'exit' or press Ctrl+C.

        Parameters
        ----------
        echo
            One of the following (default is "output"):
              - `"text"`: Echo just the text content of the response.
              - `"output"`: Echo text and tool call content.
              - `"all"`: Echo both the assistant and user turn.
              - `"none"`: Do not echo any content.
        stream
            Whether to stream the response (i.e., have the response appear in chunks).
        kwargs
            Additional keyword arguments to pass to the method used for requesting
            the response

        Returns
        -------
        None
        """

        print("\nEntering chat console. To quit, input 'exit' or press Ctrl+C.\n")

        while True:
            user_input = input("?> ")
            if user_input.strip().lower() in ("exit", "exit()"):
                break
            print("")
            self.chat(user_input, echo=echo, stream=stream, kwargs=kwargs)
            print("")

    def to_solver(
        self,
        *,
        include_system_prompt: bool = False,
        include_turns: bool = False,
    ):
        """
        Create an InspectAI solver from this chat.

        Translates this Chat instance into an InspectAI solver function that can
        be used with InspectAI's evaluation framework. This solver will capture
        (and translate) important state from the chat, including the model,
        system prompt, previous turns, registered tools, model parameters, etc.

        Parameters
        ----------
        include_system_prompt
            Whether to include the system prompt in the solver's starting
            messages.
        include_turns
            Whether to include the chat's existing turns in the solver's
            starting messages.

        Note
        ----
        Both `include_system_prompt` and `include_turns` default to `False` since
        `.export_eval()` captures this information already. Therefore,
        including them here would lead to duplication of context in the
        evaluation. However, in some cases you may want to include them, for
        example if you are manually constructing an evaluation dataset that
        does not include this information. Or, if you want to always have the
        same starting context regardless of the evaluation dataset.

        Returns
        -------
        An [InspectAI solver](https://inspect.ai-safety-institute.org.uk/solvers.html)
        function that can be used with InspectAI's evaluation framework.

        Examples
        --------
        First, put this code in a python script, perhaps named `eval_chat.py`

        ```{.python filename="eval_chat.py"}
        from chatlas import ChatOpenAI
        from inspect_ai import Task, task
        from inspect_ai.dataset import csv_dataset
        from inspect_ai.scorer import model_graded_qa

        chat = ChatOpenAI(system_prompt="You are a helpful assistant.")

        @task
        def my_eval(grader_model: str = "openai/gpt-4o"):
            return Task(
                dataset=csv_dataset("my_eval_dataset.csv"),
                solver=chat.to_solver(),
                scorer=model_graded_qa(model=grader_model)
            )
        ```

        Then run the evaluation with InspectAI's CLI:

        ```bash
        inspect eval eval_chat.py -T --grader-model openai/gpt-4o
        ```

        Note
        ----
        Learn more about this method and InspectAI's evaluation framework
        in the [Chatlas documentation](https://posit-dev.github.io/chatlas/misc/evals.html).
        """

        from ._inspect import (
            inspect_content_as_chatlas,
            inspect_messages_as_turns,
            try_import_inspect,
        )

        (imodel, isolver, _) = try_import_inspect()

        model = self.provider.model

        @isolver.solver(f"chatlas_{self.provider.name}_{model}")
        def _solver():
            async def solve(state: InspectTaskState, generate):
                start_time = time.perf_counter()

                # Fork the chat to avoid mutating the original
                chat_instance = copy.deepcopy(self)

                # Ignore initial turns (by default)
                if not include_turns:
                    chat_instance.set_turns([])

                # Map initial chatlas turns to Inspect message state
                initial_turns = chat_instance.get_turns(
                    include_system_prompt=include_system_prompt
                )
                initial_messages: list["InspectChatMessage"] = []
                for turn in initial_turns:
                    initial_messages.extend(turn.to_inspect_messages(model))

                # N.B. at this point, state.messages can include non-trivial
                # input (e.g., System messages, etc) from the eval dataset
                state.messages = initial_messages + state.messages

                # Separate out system/user/other messages in order to
                # map them appropriately to chatlas state
                system_messages: list["InspectChatMessage"] = []
                other_messages: list["InspectChatMessage"] = []
                user_prompt: "InspectChatMessage | None" = None
                for x in reversed(state.messages):
                    if x.role == "user" and user_prompt is None:
                        user_prompt = x
                    elif x.role == "system":
                        system_messages.append(x)
                    else:
                        other_messages.append(x)

                other_messages.reverse()

                # Sanity check
                if user_prompt is None:
                    raise ValueError("No user prompt found in InspectAI state messages")

                # Set the system prompt on the chat instance
                if len(system_messages) == 1:
                    chat_instance.system_prompt = str(system_messages[0])
                elif len(system_messages) > 1:
                    raise ValueError(
                        "Multiple system prompts detected in `.to_solver()`, but chatlas only "
                        "supports a single system prompt. This usually indicates that the system "
                        "prompt is mistakenly included in both the eval dataset (via `.export_eval()`) "
                        "and on the chat instance. Consider dropping the system prompt from "
                        "the chat instance by setting `.to_solver(include_system_prompt=False)`."
                    )

                # Map Inspect message state back to chatlas state
                initial_turns = inspect_messages_as_turns(other_messages)
                chat_instance.set_turns(initial_turns)

                # Map Inspect dataset input to chatlas input content
                input_content = user_prompt.content
                if isinstance(input_content, str):
                    input_content = [input_content]
                input_content = [inspect_content_as_chatlas(x) for x in input_content]

                # Generate the response (this can generate multiple turns!)
                await chat_instance.chat_async(*input_content, echo="none")

                # Map change in chatlas Turn state back to Inspect message.state
                # (Note: we skip the user prompt turn since it's already included)
                turns = chat_instance.get_turns(include_system_prompt=False)
                usage = imodel.ModelUsage()
                for i in range(len(initial_turns) + 1, len(turns)):
                    turn = turns[i]
                    state.messages.extend(turn.to_inspect_messages(model))
                    if isinstance(turn, AssistantTurn) and turn.tokens:
                        usage += imodel.ModelUsage(
                            input_tokens=turn.tokens[0],
                            output_tokens=turn.tokens[1],
                            total_tokens=turn.tokens[0] + turn.tokens[1],
                            input_tokens_cache_read=turn.tokens[2],
                        )

                last_message = state.messages[-1]
                if not isinstance(last_message, imodel.ChatMessageAssistant):
                    raise ValueError(
                        "Expected the last message in InspectAI state to be an assistant message"
                    )

                state.output = imodel.ModelOutput(
                    model=model,
                    choices=[imodel.ChatCompletionChoice(message=last_message)],
                    completion=turns[-1].text,
                    usage=usage,
                    time=time.perf_counter() - start_time,
                )
                return state

            return solve

        return _solver()

    def chat(
        self,
        *args: Content | str,
        echo: EchoOptions = "output",
        stream: bool = True,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> ChatResponse:
        """
        Generate a response from the chat.

        Parameters
        ----------
        args
            The user input(s) to generate a response from.
        echo
            One of the following (default is "output"):
              - `"text"`: Echo just the text content of the response.
              - `"output"`: Echo text and tool call content.
              - `"all"`: Echo both the assistant and user turn.
              - `"none"`: Do not echo any content.
        stream
            Whether to stream the response (i.e., have the response appear in
            chunks).
        kwargs
            Additional keyword arguments to pass to the method used for
            requesting the response.

        Returns
        -------
        ChatResponse
            A (consumed) response from the chat. Apply `str()` to this object to
            get the text content of the response.
        """
        turn = user_turn(*args, prior_turns=self.get_turns())

        display = self._markdown_display(echo=echo)

        response = ChatResponse(
            self._chat_impl(
                turn,
                echo=echo,
                content="text",
                stream=stream,
                kwargs=kwargs,
            )
        )

        with display:
            for _ in response:
                pass

        return response

    async def chat_async(
        self,
        *args: Content | str,
        echo: EchoOptions = "output",
        stream: bool = True,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> ChatResponseAsync:
        """
        Generate a response from the chat asynchronously.

        Parameters
        ----------
        args
            The user input(s) to generate a response from.
        echo
            One of the following (default is "output"):
              - `"text"`: Echo just the text content of the response.
              - `"output"`: Echo text and tool call content.
              - `"all"`: Echo both the assistant and user turn.
              - `"none"`: Do not echo any content.
        stream
            Whether to stream the response (i.e., have the response appear in
            chunks).
        kwargs
            Additional keyword arguments to pass to the method used for
            requesting the response.

        Returns
        -------
        ChatResponseAsync
            A (consumed) response from the chat. Apply `str()` to this object to
            get the text content of the response.
        """
        turn = user_turn(*args, prior_turns=self.get_turns())

        display = self._markdown_display(echo=echo)

        response = ChatResponseAsync(
            self._chat_impl_async(
                turn,
                echo=echo,
                content="text",
                stream=stream,
                kwargs=kwargs,
            ),
        )

        with display:
            async for _ in response:
                pass

        return response

    @overload
    def stream(
        self,
        *args: Content | str,
        content: Literal["text"] = "text",
        echo: EchoOptions = "none",
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> Generator[str, None, None]: ...

    @overload
    def stream(
        self,
        *args: Content | str,
        content: Literal["all"],
        echo: EchoOptions = "none",
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> Generator[str | ContentToolRequest | ContentToolResult, None, None]: ...

    def stream(
        self,
        *args: Content | str,
        content: Literal["text", "all"] = "text",
        echo: EchoOptions = "none",
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> Generator[str | ContentToolRequest | ContentToolResult, None, None]:
        """
        Generate a response from the chat in a streaming fashion.

        Parameters
        ----------
        args
            The user input(s) to generate a response from.
        content
            Whether to yield just text content or include rich content objects
            (e.g., tool calls) when relevant.
        echo
            One of the following (default is "none"):
              - `"text"`: Echo just the text content of the response.
              - `"output"`: Echo text and tool call content.
              - `"all"`: Echo both the assistant and user turn.
              - `"none"`: Do not echo any content.
        kwargs
            Additional keyword arguments to pass to the method used for requesting
            the response.

        Returns
        -------
        ChatResponse
            An (unconsumed) response from the chat. Iterate over this object to
            consume the response.
        """
        turn = user_turn(*args, prior_turns=self.get_turns())

        display = self._markdown_display(echo=echo)

        generator = self._chat_impl(
            turn,
            stream=True,
            echo=echo,
            content=content,
            kwargs=kwargs,
        )

        def wrapper() -> Generator[
            str | ContentToolRequest | ContentToolResult, None, None
        ]:
            with display:
                for chunk in generator:
                    yield chunk

        return wrapper()

    @overload
    async def stream_async(
        self,
        *args: Content | str,
        content: Literal["text"] = "text",
        echo: EchoOptions = "none",
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> AsyncGenerator[str, None]: ...

    @overload
    async def stream_async(
        self,
        *args: Content | str,
        content: Literal["all"],
        echo: EchoOptions = "none",
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> AsyncGenerator[str | ContentToolRequest | ContentToolResult, None]: ...

    async def stream_async(
        self,
        *args: Content | str,
        content: Literal["text", "all"] = "text",
        echo: EchoOptions = "none",
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> AsyncGenerator[str | ContentToolRequest | ContentToolResult, None]:
        """
        Generate a response from the chat in a streaming fashion asynchronously.

        Parameters
        ----------
        args
            The user input(s) to generate a response from.
        content
            Whether to yield just text content or include rich content objects
            (e.g., tool calls) when relevant.
        echo
            One of the following (default is "none"):
              - `"text"`: Echo just the text content of the response.
              - `"output"`: Echo text and tool call content.
              - `"all"`: Echo both the assistant and user turn.
              - `"none"`: Do not echo any content.
        kwargs
            Additional keyword arguments to pass to the method used for requesting
            the response.

        Returns
        -------
        ChatResponseAsync
            An (unconsumed) response from the chat. Iterate over this object to
            consume the response.
        """
        turn = user_turn(*args, prior_turns=self.get_turns())

        display = self._markdown_display(echo=echo)

        async def wrapper() -> AsyncGenerator[
            str | ContentToolRequest | ContentToolResult, None
        ]:
            with display:
                async for chunk in self._chat_impl_async(
                    turn,
                    stream=True,
                    echo=echo,
                    content=content,
                    kwargs=kwargs,
                ):
                    yield chunk

        return wrapper()

    def chat_structured(
        self,
        *args: Content | str,
        data_model: type[BaseModelT],
        echo: EchoOptions = "none",
        stream: bool = False,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> BaseModelT:
        """
        Extract structured data.

        Parameters
        ----------
        args
            The input to send to the chatbot. This is typically the text you
            want to extract data from, but it can be omitted if the data is
            obvious from the existing conversation.
        data_model
            A Pydantic model describing the structure of the data to extract.
        echo
            One of the following (default is "none"):
              - `"text"`: Echo just the text content of the response.
              - `"output"`: Echo text and tool call content.
              - `"all"`: Echo both the assistant and user turn.
              - `"none"`: Do not echo any content.
        stream
            Whether to stream the response (i.e., have the response appear in chunks).
        kwargs
            Additional keyword arguments to pass to the method used for requesting
            the response.

        Returns
        -------
        BaseModelT
            An instance of the provided `data_model` containing the extracted data.
        """
        dat = self._submit_and_extract_data(
            *args,
            data_model=data_model,
            echo=echo,
            stream=stream,
            kwargs=kwargs,
        )
        return data_model.model_validate(dat)

    def extract_data(
        self,
        *args: Content | str,
        data_model: type[BaseModel],
        echo: EchoOptions = "none",
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Deprecated: use `.chat_structured()` instead.
        """
        warnings.warn(
            "The `extract_data()` method is deprecated and will be removed in a future release. "
            "Use the `chat_structured()` method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._submit_and_extract_data(
            *args,
            data_model=data_model,
            echo=echo,
            stream=stream,
        )

    def _submit_and_extract_data(
        self,
        *args: Content | str,
        data_model: type[BaseModel],
        echo: EchoOptions = "none",
        stream: bool = False,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> dict[str, Any]:
        display = self._markdown_display(echo=echo)

        response = ChatResponse(
            self._submit_turns(
                user_turn(*args, prior_turns=self.get_turns()),
                data_model=data_model,
                echo=echo,
                stream=stream,
                kwargs=kwargs,
            )
        )

        with display:
            for _ in response:
                pass

        turn = self.get_last_turn()
        assert turn is not None

        return Chat._extract_turn_json(turn)

    async def chat_structured_async(
        self,
        *args: Content | str,
        data_model: type[BaseModelT],
        echo: EchoOptions = "none",
        stream: bool = False,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> BaseModelT:
        """
        Extract structured data from the given input asynchronously.

        Parameters
        ----------
        args
            The input to send to the chatbot. This is typically the text you
            want to extract data from, but it can be omitted if the data is
            obvious from the existing conversation.
        data_model
            A Pydantic model describing the structure of the data to extract.
        echo
            One of the following (default is "none"):
              - `"text"`: Echo just the text content of the response.
              - `"output"`: Echo text and tool call content.
              - `"all"`: Echo both the assistant and user turn.
              - `"none"`: Do not echo any content.
        stream
            Whether to stream the response (i.e., have the response appear in chunks).
            Defaults to `True` if `echo` is not "none".
        kwargs
            Additional keyword arguments to pass to the method used for requesting
            the response.

        Returns
        -------
        BaseModelT
            An instance of the provided `data_model` containing the extracted data.
        """
        dat = await self._submit_and_extract_data_async(
            *args,
            data_model=data_model,
            echo=echo,
            stream=stream,
            kwargs=kwargs,
        )
        return data_model.model_validate(dat)

    async def extract_data_async(
        self,
        *args: Content | str,
        data_model: type[BaseModel],
        echo: EchoOptions = "none",
        stream: bool = False,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> dict[str, Any]:
        """
        Deprecated: use `.chat_structured_async()` instead.
        """
        warnings.warn(
            "The `extract_data_async()` method is deprecated and will be removed in a future release. "
            "Use the `chat_structured_async()` method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return await self._submit_and_extract_data_async(
            *args,
            data_model=data_model,
            echo=echo,
            stream=stream,
            kwargs=kwargs,
        )

    async def _submit_and_extract_data_async(
        self,
        *args: Content | str,
        data_model: type[BaseModel],
        echo: EchoOptions = "none",
        stream: bool = False,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> dict[str, Any]:
        display = self._markdown_display(echo=echo)

        response = ChatResponseAsync(
            self._submit_turns_async(
                user_turn(*args, prior_turns=self.get_turns()),
                data_model=data_model,
                echo=echo,
                stream=stream,
                kwargs=kwargs,
            )
        )

        with display:
            async for _ in response:
                pass

        turn = self.get_last_turn()
        assert turn is not None

        return Chat._extract_turn_json(turn)

    @staticmethod
    def _extract_turn_json(turn: AssistantTurn) -> dict[str, Any]:
        res: list[ContentJson] = []
        for x in turn.contents:
            if isinstance(x, ContentJson):
                res.append(x)

        if len(res) != 1:
            raise ValueError(
                f"Data extraction failed: {len(res)} data results received."
            )

        json = res[0]
        return json.value

    def set_model_params(
        self,
        *,
        temperature: float | None | MISSING_TYPE = MISSING,
        top_p: float | None | MISSING_TYPE = MISSING,
        top_k: int | None | MISSING_TYPE = MISSING,
        frequency_penalty: float | None | MISSING_TYPE = MISSING,
        presence_penalty: float | None | MISSING_TYPE = MISSING,
        seed: int | None | MISSING_TYPE = MISSING,
        max_tokens: int | None | MISSING_TYPE = MISSING,
        log_probs: bool | None | MISSING_TYPE = MISSING,
        stop_sequences: list[str] | None | MISSING_TYPE = MISSING,
    ):
        """
        Set common model parameters for the chat.

        A unified interface for setting common model parameters
        across different providers. This method is useful for setting
        parameters that are commonly supported by most providers, such as
        temperature, top_p, etc.

        By default, if the parameter is not set (i.e., set to `MISSING`),
        the provider's default value is used. If you want to reset a
        parameter to its default value, set it to `None`.

        Parameters
        ----------
        temperature
            Temperature of the sampling distribution.
        top_p
            The cumulative probability for token selection.
        top_k
            The number of highest probability vocabulary tokens to keep.
        frequency_penalty
            Frequency penalty for generated tokens.
        presence_penalty
            Presence penalty for generated tokens.
        seed
            Seed for random number generator.
        max_tokens
            Maximum number of tokens to generate.
        log_probs
            Include the log probabilities in the output?
        stop_sequences
            A character vector of tokens to stop generation on.
        """

        params: StandardModelParams = {}

        # Collect specified parameters
        if is_present(temperature):
            params["temperature"] = temperature
        if is_present(top_p):
            params["top_p"] = top_p
        if is_present(top_k):
            params["top_k"] = top_k
        if is_present(frequency_penalty):
            params["frequency_penalty"] = frequency_penalty
        if is_present(presence_penalty):
            params["presence_penalty"] = presence_penalty
        if is_present(seed):
            params["seed"] = seed
        if is_present(max_tokens):
            params["max_tokens"] = max_tokens
        if is_present(log_probs):
            params["log_probs"] = log_probs
        if is_present(stop_sequences):
            params["stop_sequences"] = stop_sequences

        # Warn about un-supported parameters
        supported = self.provider.supported_model_params()
        unsupported = set(params.keys()) - set(supported)
        if unsupported:
            warnings.warn(
                f"The following parameters are not supported by the provider: {unsupported}. "
                "Please check the provider's documentation for supported parameters.",
                UserWarning,
            )
            # Drop the unsupported parameters
            for key in unsupported:
                del params[key]

        # Drop parameters that are set to None
        discard = []
        if temperature is None:
            discard.append("temperature")
        if top_p is None:
            discard.append("top_p")
        if top_k is None:
            discard.append("top_k")
        if frequency_penalty is None:
            discard.append("frequency_penalty")
        if presence_penalty is None:
            discard.append("presence_penalty")
        if seed is None:
            discard.append("seed")
        if max_tokens is None:
            discard.append("max_tokens")
        if log_probs is None:
            discard.append("log_probs")
        if stop_sequences is None:
            discard.append("stop_sequences")

        for key in discard:
            if key in self._standard_model_params:
                del self._standard_model_params[key]

        # Update the standard model parameters
        self._standard_model_params.update(params)

    async def register_mcp_tools_http_stream_async(
        self,
        *,
        url: str,
        include_tools: Sequence[str] = (),
        exclude_tools: Sequence[str] = (),
        name: Optional[str] = None,
        namespace: Optional[str] = None,
        transport_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Register tools from an MCP server using streamable HTTP transport.

        Connects to an MCP server (that communicates over a streamable HTTP
        transport) and registers the available tools. This is useful for
        utilizing tools provided by an MCP server running on a remote server (or
        locally) over HTTP.

        Pre-requisites
        --------------

        ::: {.callout-note}
        Requires the `mcp` package to be installed. Install it with:

        ```bash
        pip install mcp
        ```
        :::

        Parameters
        ----------
        url
            URL endpoint where the Streamable HTTP server is mounted (e.g.,
            `http://localhost:8000/mcp`)
        name
            A unique name for the MCP server session. If not provided, the name
            is derived from the MCP server information. This name is primarily
            useful for cleanup purposes (i.e., to close a particular MCP
            session).
        include_tools
            List of tool names to include. By default, all available tools are
            included.
        exclude_tools
            List of tool names to exclude. This parameter and `include_tools`
            are mutually exclusive.
        namespace
            A namespace to prepend to tool names (i.e., `namespace.tool_name`)
            from this MCP server. This is primarily useful to avoid name
            collisions with other tools already registered with the chat. This
            namespace applies when tools are advertised to the LLM, so try
            to use a meaningful name that describes the server and/or the tools
            it provides. For example, if you have a server that provides tools
            for mathematical operations, you might use `math` as the namespace.
        transport_kwargs
            Additional keyword arguments for the transport layer (i.e.,
            `mcp.client.streamable_http.streamablehttp_client`).

        Returns
        -------
        None

        See Also
        --------
        * `.cleanup_mcp_tools_async()` : Cleanup registered MCP tools.
        * `.register_mcp_tools_stdio_async()` : Register tools from an MCP server using stdio transport.

        Note
        ----
        Unlike the `.register_mcp_tools_stdio_async()` method, this method does
        not launch an MCP server. Instead, it assumes an HTTP server is already
        running at the specified URL. This is useful for connecting to an
        existing MCP server that is already running and serving tools.

        Examples
        --------

        Assuming you have a Python script `my_mcp_server.py` that implements an
        MCP server like so:

        ```python
        from mcp.server.fastmcp import FastMCP

        app = FastMCP("my_server")

        @app.tool(description="Add two numbers.")
        def add(x: int, y: int) -> int:
            return x + y

        app.run(transport="streamable-http")
        ```

        You can launch this server like so:

        ```bash
        python my_mcp_server.py
        ```

        Then, you can register this server with the chat as follows:

        ```python
        await chat.register_mcp_tools_http_stream_async(
            url="http://localhost:8080/mcp"
        )
        ```
        """
        if isinstance(exclude_tools, str):
            exclude_tools = [exclude_tools]
        if isinstance(include_tools, str):
            include_tools = [include_tools]

        session_info = await self._mcp_manager.register_http_stream_tools(
            name=name,
            url=url,
            include_tools=include_tools,
            exclude_tools=exclude_tools,
            namespace=namespace,
            transport_kwargs=transport_kwargs or {},
        )

        overlapping_tools = set(self._tools.keys()) & set(session_info.tools)
        if overlapping_tools:
            await self._mcp_manager.close_sessions([session_info.name])
            raise ValueError(
                f"The following tools are already registered: {overlapping_tools}. "
                "Consider providing a namespace when registering this MCP server "
                "to avoid name collisions."
            )

        self._tools.update(session_info.tools)

    async def register_mcp_tools_stdio_async(
        self,
        *,
        command: str,
        args: list[str],
        name: Optional[str] = None,
        include_tools: Sequence[str] = (),
        exclude_tools: Sequence[str] = (),
        namespace: Optional[str] = None,
        transport_kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Register tools from a MCP server using stdio (standard input/output) transport.

        Useful for launching an MCP server and registering its tools with the chat -- all
        from the same Python process.

        In more detail, this method:

        1. Executes the given `command` with the provided `args`.
            * This should start an MCP server that communicates via stdio.
        2. Establishes a client connection to the MCP server using the `mcp` package.
        3. Registers the available tools from the MCP server with the chat.
        4. Returns a cleanup callback to close the MCP session and remove the tools.

        Pre-requisites
        --------------

        ::: {.callout-note}
        Requires the `mcp` package to be installed. Install it with:

        ```bash
        pip install mcp
        ```
        :::

        Parameters
        ----------
        command
            System command to execute to start the MCP server (e.g., `python`).
        args
            Arguments to pass to the system command (e.g., `["-m",
            "my_mcp_server"]`).
        name
            A unique name for the MCP server session. If not provided, the name
            is derived from the MCP server information. This name is primarily
            useful for cleanup purposes (i.e., to close a particular MCP
            session).
        include_tools
            List of tool names to include. By default, all available tools are
            included.
        exclude_tools
            List of tool names to exclude. This parameter and `include_tools`
            are mutually exclusive.
        namespace
            A namespace to prepend to tool names (i.e., `namespace.tool_name`)
            from this MCP server. This is primarily useful to avoid name
            collisions with other tools already registered with the chat. This
            namespace applies when tools are advertised to the LLM, so try
            to use a meaningful name that describes the server and/or the tools
            it provides. For example, if you have a server that provides tools
            for mathematical operations, you might use `math` as the namespace.
        transport_kwargs
            Additional keyword arguments for the stdio transport layer (i.e.,
            `mcp.client.stdio.stdio_client`).

        Returns
        -------
        None

        See Also
        --------
        * `.cleanup_mcp_tools_async()` : Cleanup registered MCP tools.
        * `.register_mcp_tools_http_stream_async()` : Register tools from an MCP server using streamable HTTP transport.

        Examples
        --------

        Assuming you have a Python script `my_mcp_server.py` that implements an
        MCP server like so

        ```python
        from mcp.server.fastmcp import FastMCP

        app = FastMCP("my_server")

        @app.tool(description="Add two numbers.")
        def add(y: int, z: int) -> int:
            return y - z

        app.run(transport="stdio")
        ```

        You can register this server with the chat as follows:

        ```python
        from chatlas import ChatOpenAI

        chat = ChatOpenAI()

        await chat.register_mcp_tools_stdio_async(
            command="python",
            args=["-m", "my_mcp_server"],
        )
        ```
        """
        if isinstance(exclude_tools, str):
            exclude_tools = [exclude_tools]
        if isinstance(include_tools, str):
            include_tools = [include_tools]

        session_info = await self._mcp_manager.register_stdio_tools(
            command=command,
            args=args,
            name=name,
            include_tools=include_tools,
            exclude_tools=exclude_tools,
            namespace=namespace,
            transport_kwargs=transport_kwargs or {},
        )

        overlapping_tools = set(self._tools.keys()) & set(session_info.tools)
        if overlapping_tools:
            await self._mcp_manager.close_sessions([session_info.name])
            raise ValueError(
                f"The following tools are already registered: {overlapping_tools}. "
                "Consider providing a namespace when registering this MCP server "
                "to avoid name collisions."
            )

        self._tools.update(session_info.tools)

    async def cleanup_mcp_tools(self, names: Optional[Sequence[str]] = None):
        """
        Close MCP server connections (and their corresponding tools).

        This method closes the MCP client sessions and removes the tools registered
        from the MCP servers. If a specific `name` is provided, it will only clean
        up the tools and session associated with that name. If no name is provided,
        it will clean up all registered MCP tools and sessions.

        Parameters
        ----------
        names
            If provided, only clean up the tools and session associated
            with these names. If not provided, clean up all registered MCP tools and sessions.

        Returns
        -------
        None
        """
        closed_sessions = await self._mcp_manager.close_sessions(names)

        # Remove relevant MCP tools from the main tools registry
        for session in closed_sessions:
            for tool_name in session.tools:
                if tool_name in self._tools:
                    del self._tools[tool_name]

    def register_tool(
        self,
        func: Callable[..., Any] | Callable[..., Awaitable[Any]] | Tool,
        *,
        force: bool = False,
        name: Optional[str] = None,
        model: Optional[type[BaseModel]] = None,
        annotations: "Optional[ToolAnnotations]" = None,
    ):
        """
        Register a tool (function) with the chat.

        The function will always be invoked in the current Python process.

        Examples
        --------

        If your tool has straightforward input parameters, you can just
        register the function directly (type hints and a docstring explaning
        both what the function does and what the parameters are for is strongly
        recommended):

        ```python
        from chatlas import ChatOpenAI


        def add(a: int, b: int) -> int:
            '''
            Add two numbers together.

            Parameters
            ----------
            a : int
                The first number to add.
            b : int
                The second number to add.
            '''
            return a + b


        chat = ChatOpenAI()
        chat.register_tool(add)
        chat.chat("What is 2 + 2?")
        ```

        If your tool has more complex input parameters, you can provide a Pydantic
        model that corresponds to the input parameters for the function, This way, you
        can have fields that hold other model(s) (for more complex input parameters),
        and also more directly document the input parameters:

        ```python
        from chatlas import ChatOpenAI
        from pydantic import BaseModel, Field


        class AddParams(BaseModel):
            '''Add two numbers together.'''

            a: int = Field(description="The first number to add.")

            b: int = Field(description="The second number to add.")


        def add(a: int, b: int) -> int:
            return a + b


        chat = ChatOpenAI()
        chat.register_tool(add, model=AddParams)
        chat.chat("What is 2 + 2?")
        ```

        Parameters
        ----------
        func
            The function to be invoked when the tool is called.
        force
            If `True`, overwrite any existing tool with the same name. If `False`
            (the default), raise an error if a tool with the same name already exists.
        name
            The name of the tool. If not provided, the name will be inferred from the
            `func`'s name (or the `model`'s name, if provided).
        model
            A Pydantic model that describes the input parameters for the function.
            If not provided, the model will be inferred from the function's type hints.
            The primary reason why you might want to provide a model in
            Note that the name and docstring of the model takes precedence over the
            name and docstring of the function.
        annotations
            Additional properties that describe the tool and its behavior.

        Raises
        ------
        ValueError
            If a tool with the same name already exists and `force` is `False`.
        """
        if isinstance(func, Tool):
            name = name or func.name
            annotations = annotations or func.annotations
            if model is not None:
                func = Tool.from_func(
                    func.func, name=name, model=model, annotations=annotations
                )
            func = func.func

        tool = Tool.from_func(func, name=name, model=model, annotations=annotations)
        if tool.name in self._tools and not force:
            raise ValueError(
                f"Tool with name '{tool.name}' is already registered. "
                "Set `force=True` to overwrite it."
            )
        self._tools[tool.name] = tool

    def get_tools(self) -> list[Tool]:
        """
        Get the list of registered tools.

        Returns
        -------
        list[Tool]
            A list of `Tool` instances that are currently registered with the chat.
        """
        return list(self._tools.values())

    def set_tools(
        self, tools: list[Callable[..., Any] | Callable[..., Awaitable[Any]] | Tool]
    ):
        """
        Set the tools for the chat.

        This replaces any previously registered tools with the provided list of
        tools. This is for advanced usage -- typically, you would use
        `.register_tool()` to register individual tools as needed.

        Parameters
        ----------
        tools
            A list of `Tool` instances to set as the chat's tools.
        """
        self._tools = {}
        for tool in tools:
            if isinstance(tool, Tool):
                self._tools[tool.name] = tool
            else:
                self.register_tool(tool)

    def on_tool_request(self, callback: Callable[[ContentToolRequest], None]):
        """
        Register a callback for a tool request event.

        A tool request event occurs when the assistant requests a tool to be
        called on its behalf. Before invoking the tool, `on_tool_request`
        handlers are called with the relevant `ContentToolRequest` object. This
        is useful if you want to handle tool requests in a custom way, such as
        requiring logging them or requiring user approval before invoking the
        tool

        Parameters
        ----------
        callback
            A function to be called when a tool request event occurs.
            This function must have a single argument, which will be the
            tool request (i.e., a `ContentToolRequest` object).

        Returns
        -------
        A callable that can be used to remove the callback later.
        """
        return self._on_tool_request_callbacks.add(callback)

    def on_tool_result(self, callback: Callable[[ContentToolResult], None]):
        """
        Register a callback for a tool result event.

        A tool result event occurs when a tool has been invoked and the
        result is ready to be provided to the assistant. After the tool
        has been invoked, `on_tool_result` handlers are called with the
        relevant `ContentToolResult` object. This is useful if you want to
        handle tool results in a custom way such as logging them.

        Parameters
        ----------
        callback
            A function to be called when a tool result event occurs.
            This function must have a single argument, which will be the
            tool result (i.e., a `ContentToolResult` object).

        Returns
        -------
        A callable that can be used to remove the callback later.
        """
        return self._on_tool_result_callbacks.add(callback)

    @property
    def current_display(self) -> Optional[MarkdownDisplay]:
        """
        Get the currently active markdown display, if any.

        The display represents the place where `.chat(echo)` content is
        being displayed. In a notebook/Quarto, this is a wrapper around
        `IPython.display`. Otherwise, it is a wrapper around a
        `rich.live.Live()` console.

        This is primarily useful if you want to add custom content to the
        display while the chat is running, but currently blocked by something
        like a tool call.

        Example
        -------
        ```python
        import requests
        from chatlas import ChatOpenAI

        chat = ChatOpenAI()


        def get_current_weather(latitude: float, longitude: float):
            "Get the current weather given a latitude and longitude."

            lat_lng = f"latitude={latitude}&longitude={longitude}"
            url = (
                "https://api.open-meteo.com/v1/forecast?"
                f"{lat_lng}&current=temperature_2m,wind_speed_10m"
                "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
            )
            response = requests.get(url)
            json = response.json()
            if chat.current_display:
                chat.current_display.echo("My custom tool display!!!")
            return json["current"]


        chat.register_tool(get_current_weather)

        chat.chat("What's the current temperature in Duluth, MN?", echo="text")
        ```


        Returns
        -------
        Optional[MarkdownDisplay]
            The currently active markdown display, if any.
        """
        return self._current_display

    def _echo_content(self, x: str):
        if self._current_display:
            self._current_display.echo(x)

    def export(
        self,
        filename: str | Path,
        *,
        turns: Optional[Sequence[Turn]] = None,
        title: Optional[str] = None,
        content: Literal["text", "all"] = "text",
        include_system_prompt: bool = True,
        overwrite: bool = False,
    ):
        """
        Export the chat history to a file.

        Parameters
        ----------
        filename
            The filename to export the chat to. Currently this must
            be a `.md` or `.html` file.
        turns
            The `.get_turns()` to export. If not provided, the chat's current turns
            will be used.
        title
            A title to place at the top of the exported file.
        overwrite
            Whether to overwrite the file if it already exists.
        content
            Whether to include text content, all content (i.e., tool calls), or no
            content.
        include_system_prompt
            Whether to include the system prompt in a <details> tag.

        Returns
        -------
        Path
            The path to the exported file.
        """
        if not turns:
            turns = self.get_turns(include_system_prompt=False)
        if not turns:
            raise ValueError("No turns to export.")

        if isinstance(filename, str):
            filename = Path(filename)

        filename = filename.resolve()
        if filename.exists() and not overwrite:
            raise ValueError(
                f"File {filename} already exists. Set `overwrite=True` to overwrite."
            )

        if filename.suffix not in {".md", ".html"}:
            raise ValueError("The filename must have a `.md` or `.html` extension.")

        # When exporting to HTML, we lean on shiny's chat component for rendering markdown and styling
        is_html = filename.suffix == ".html"

        # Get contents from each turn
        content_arr: list[str] = []
        for turn in turns:
            turn_content = "\n\n".join(
                [
                    str(x).strip()
                    for x in turn.contents
                    if content == "all" or isinstance(x, ContentText)
                ]
            )
            if is_html:
                msg_type = "user" if isinstance(turn, UserTurn) else "chat"
                content_attr = html_escape(turn_content)
                turn_content = f"<shiny-{msg_type}-message content='{content_attr}'></shiny-{msg_type}-message>"
            else:
                turn_content = f"## {turn.role.capitalize()}\n\n{turn_content}"
            content_arr.append(turn_content)
        contents = "\n\n".join(content_arr)

        # Shiny chat message components requires container elements
        if is_html:
            contents = f"<shiny-chat-messages>\n{contents}\n</shiny-chat-messages>"
            contents = f"<shiny-chat-container>{contents}</shiny-chat-container>"

        # Add title to the top
        if title:
            if is_html:
                contents = f"<h1>{title}</h1>\n\n{contents}"
            else:
                contents = f"# {title}\n\n{contents}"

        # Add system prompt to the bottom
        if include_system_prompt and self.system_prompt:
            contents += f"\n<br><br>\n<details><summary>System prompt</summary>\n\n{self.system_prompt}\n\n</details>"

        # Wrap in HTML template if exporting to HTML
        if is_html:
            contents = self._html_template(contents)

        with open(filename, "w") as f:
            f.write(contents)

        return filename

    @staticmethod
    def _html_template(contents: str) -> str:
        version = "1.2.1"
        shiny_www = (
            f"https://cdn.jsdelivr.net/gh/posit-dev/py-shiny@{version}/shiny/www/"
        )

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
          <script src="{shiny_www}/py-shiny/chat/chat.js"></script>
          <link rel="stylesheet" href="{shiny_www}/py-shiny/chat/chat.css">
          <link rel="stylesheet" href="{shiny_www}/shared/bootstrap/bootstrap.min.css">
        </head>
        <body>
          <div style="max-width:700px; margin:0 auto; padding-top:20px;">
            {contents}
          </div>
        </body>
        </html>
        """

    def export_eval(
        self,
        filename: str | Path,
        *,
        target: Optional[str] = None,
        include_system_prompt: bool = True,
        turns: Optional[list[Turn]] = None,
        overwrite: Literal["append", True, False] = "append",
        **kwargs: Any,
    ):
        """
        Creates an Inspect AI eval dataset sample from the current chat.

        Creates an Inspect AI eval
        [Sample](https://inspect.aisi.org.uk/reference/inspect_ai.dataset.html#sample)
        from the current chat and appends it to a JSONL file. In Inspect, a eval
        dataset is a collection of Samples, where each Sample represents a
        single `input` (i.e., user prompt) and the expected `target` (i.e., the
        target answer and/or grading guidance for it). Note that each `input` of
        a particular sample can contain a series of messages (from both the user
        and assistant).

        Note
        ----
        Each call to this method appends a single Sample as a new line in the
        specified JSONL file. If the file does not exist, it will be created.

        Parameters
        ----------
        filename
            The filename to export the chat to. Currently this must
            be a `.jsonl` file.
        target
            The target output for the eval sample. By default, this is
            taken to be the content of the last assistant turn.
        include_system_prompt
            Whether to include the system prompt (if any) as the
            first turn in the eval sample.
        turns
            The input turns for the eval sample. By default, this is
            taken to be all turns except the last (assistant) turn.
            Note that system prompts are not allowed here, but controlled
            separately via the `include_system_prompt` parameter.
        overwrite
            Behavior when the file already exists:
              - `"append"` (default): Append to the existing file.
              - `True`: Overwrite the existing file.
              - `False`: Raise an error if the file already exists.
        kwargs
            Additional keyword arguments to pass to the `Sample()` constructor.
            This is primarily useful for setting an ID or metadata on the sample.

        Examples
        --------

        Step 1: export the chat to an eval JSONL file

        ```python
        from chatlas import ChatOpenAI

        chat = ChatOpenAI(system_prompt="You are a helpful assistant.")
        chat.chat("Hello, how are you?")

        chat.export_eval("my_eval_1.jsonl")
        ```

        Step 2: load the eval JSONL file into an Inspect AI eval task

        ```python
        from chatlas import ChatOpenAI
        from inspect_ai import Task, task
        from inspect_ai.dataset import json_dataset
        from inspect_ai.scorer import model_graded_qa

        # No need to load in system prompt -- it's included in the eval JSONL file by default
        chat = ChatOpenAI()


        @task
        def my_eval():
            return Task(
                dataset=json_dataset("my_eval.jsonl"),
                solver=chat.to_solver(),
                scorer=model_graded_qa(model="openai/gpt-4o-mini"),
            )
        ```
        """

        if isinstance(filename, str):
            filename = Path(filename)

        filename = filename.resolve()
        if filename.exists() and overwrite is False:
            raise ValueError(
                f"File {filename} already exists. Set `overwrite=True` to overwrite or `overwrite='append'` to append."
            )

        if filename.suffix not in {".jsonl"}:
            raise ValueError("The filename must have a `.jsonl` extension.")

        if turns is None:
            turns = cast(list[Turn], self.get_turns(include_system_prompt=False))

        if any(isinstance(x, SystemTurn) for x in turns):
            raise ValueError("System prompts are not allowed in eval input turns.")

        if not any(isinstance(x, UserTurn) for x in turns):
            raise ValueError("At least one user turn is required in eval input turns.")

        if include_system_prompt:
            system_turn = self.get_last_turn(role="system")
            if system_turn:
                turns = [system_turn] + turns

        input_turns, target_turn = turns[:-1], turns[-1]
        if target_turn.role != "assistant":
            raise ValueError("The last turn must be an assistant turn.")

        if target is None:
            target = str(target_turn)

        input_messages = []
        for x in input_turns:
            input_messages.extend(x.to_inspect_messages(self.provider.model))

        from inspect_ai.dataset import Sample

        sample = Sample(input=input_messages, target=target, **kwargs)
        sample_json = sample.model_dump_json(exclude_none=True)

        mode = "a" if overwrite == "append" and filename.exists() else "w"
        with open(filename, mode) as f:
            f.write(sample_json + "\n")

        return filename

    @overload
    def _chat_impl(
        self,
        user_turn: UserTurn,
        echo: EchoOptions,
        content: Literal["text"],
        stream: bool,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> Generator[str, None, None]: ...

    @overload
    def _chat_impl(
        self,
        user_turn: UserTurn,
        echo: EchoOptions,
        content: Literal["all"],
        stream: bool,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> Generator[str | ContentToolRequest | ContentToolResult, None, None]: ...

    def _chat_impl(
        self,
        user_turn: UserTurn,
        echo: EchoOptions,
        content: Literal["text", "all"],
        stream: bool,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> Generator[str | ContentToolRequest | ContentToolResult, None, None]:
        user_turn_result: UserTurn | None = user_turn
        while user_turn_result is not None:
            for chunk in self._submit_turns(
                user_turn_result,
                echo=echo,
                stream=stream,
                kwargs=kwargs,
            ):
                yield chunk

            turn = self.get_last_turn(role="assistant")
            assert turn is not None
            user_turn_result = None

            all_results: list[ContentToolResult] = []
            for x in turn.contents:
                if isinstance(x, ContentToolRequest):
                    tool = self._tools.get(x.name)
                    if tool is not None:
                        x.tool = ToolInfo.from_tool(tool)
                    if echo == "output":
                        self._echo_content(f"\n\n{x}\n\n")
                    if content == "all":
                        yield x
                    results = self._invoke_tool(x)
                    for res in results:
                        if echo == "output":
                            self._echo_content(f"\n\n{res}\n\n")
                        if content == "all":
                            yield res
                        all_results.append(res)

            if all_results:
                user_turn_result = UserTurn(all_results)

    @overload
    def _chat_impl_async(
        self,
        user_turn: UserTurn,
        echo: EchoOptions,
        content: Literal["text"],
        stream: bool,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> AsyncGenerator[str, None]: ...

    @overload
    def _chat_impl_async(
        self,
        user_turn: UserTurn,
        echo: EchoOptions,
        content: Literal["all"],
        stream: bool,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> AsyncGenerator[str | ContentToolRequest | ContentToolResult, None]: ...

    async def _chat_impl_async(
        self,
        user_turn: UserTurn,
        echo: EchoOptions,
        content: Literal["text", "all"],
        stream: bool,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> AsyncGenerator[str | ContentToolRequest | ContentToolResult, None]:
        user_turn_result: UserTurn | None = user_turn
        while user_turn_result is not None:
            async for chunk in self._submit_turns_async(
                user_turn_result,
                echo=echo,
                stream=stream,
                kwargs=kwargs,
            ):
                yield chunk

            turn = self.get_last_turn(role="assistant")
            assert turn is not None
            user_turn_result = None

            all_results: list[ContentToolResult] = []
            for x in turn.contents:
                if isinstance(x, ContentToolRequest):
                    tool = self._tools.get(x.name)
                    if tool is not None:
                        x.tool = ToolInfo.from_tool(tool)
                    if echo == "output":
                        self._echo_content(f"\n\n{x}\n\n")
                    if content == "all":
                        yield x
                    results = self._invoke_tool_async(x)
                    async for res in results:
                        if echo == "output":
                            self._echo_content(f"\n\n{res}\n\n")
                        if content == "all":
                            yield res
                        else:
                            yield "\n\n"
                        all_results.append(res)

            if all_results:
                user_turn_result = UserTurn(all_results)

    def _submit_turns(
        self,
        user_turn: UserTurn,
        echo: EchoOptions,
        stream: bool,
        data_model: type[BaseModel] | None = None,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> Generator[str, None, None]:
        if any(x._is_async for x in self._tools.values()):
            raise ValueError("Cannot use async tools in a synchronous chat")

        def emit(text: str | Content):
            self._echo_content(str(text))

        emit("<br>\n\n")

        if echo == "all":
            emit_user_contents(user_turn, emit)

        # Start collecting additional keyword args (from model parameters)
        all_kwargs = self._collect_all_kwargs(kwargs)

        if stream:
            response = self.provider.chat_perform(
                stream=True,
                turns=[*self._turns, user_turn],
                tools=self._tools,
                data_model=data_model,
                kwargs=all_kwargs,
            )

            result = None
            for chunk in response:
                text = self.provider.stream_text(chunk)
                if text:
                    emit(text)
                    yield text
                result = self.provider.stream_merge_chunks(result, chunk)

            turn = self.provider.stream_turn(
                result,
                has_data_model=data_model is not None,
            )

            if echo == "all":
                emit_other_contents(turn, emit)

        else:
            response = self.provider.chat_perform(
                stream=False,
                turns=[*self._turns, user_turn],
                tools=self._tools,
                data_model=data_model,
                kwargs=all_kwargs,
            )

            turn = self.provider.value_turn(
                response, has_data_model=data_model is not None
            )
            if turn.text:
                emit(turn.text)
                yield turn.text

            if echo == "all":
                emit_other_contents(turn, emit)

        if not isinstance(turn, AssistantTurn):
            raise TypeError(
                f"Expected turn to be AssistantTurn, got {type(turn).__name__}"
            )
        if turn.tokens is None and turn.completion:
            turn.tokens = self.provider.value_tokens(turn.completion)
        if turn.tokens is not None:
            tokens_log(self.provider, turn.tokens)
        self._turns.extend([user_turn, turn])

    async def _submit_turns_async(
        self,
        user_turn: UserTurn,
        echo: EchoOptions,
        stream: bool,
        data_model: type[BaseModel] | None = None,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> AsyncGenerator[str, None]:
        def emit(text: str | Content):
            self._echo_content(str(text))

        emit("<br>\n\n")

        if echo == "all":
            emit_user_contents(user_turn, emit)

        # Start collecting additional keyword args (from model parameters)
        all_kwargs = self._collect_all_kwargs(kwargs)

        if stream:
            response = await self.provider.chat_perform_async(
                stream=True,
                turns=[*self._turns, user_turn],
                tools=self._tools,
                data_model=data_model,
                kwargs=all_kwargs,
            )

            result = None
            async for chunk in response:
                text = self.provider.stream_text(chunk)
                if text:
                    emit(text)
                    yield text
                result = self.provider.stream_merge_chunks(result, chunk)

            turn = self.provider.stream_turn(
                result,
                has_data_model=data_model is not None,
            )

            if echo == "all":
                emit_other_contents(turn, emit)

        else:
            response = await self.provider.chat_perform_async(
                stream=False,
                turns=[*self._turns, user_turn],
                tools=self._tools,
                data_model=data_model,
                kwargs=all_kwargs,
            )

            turn = self.provider.value_turn(
                response, has_data_model=data_model is not None
            )
            if turn.text:
                emit(turn.text)
                yield turn.text

            if echo == "all":
                emit_other_contents(turn, emit)

        if not isinstance(turn, AssistantTurn):
            raise TypeError(
                f"Expected turn to be AssistantTurn, got {type(turn).__name__}"
            )
        if turn.tokens is None and turn.completion:
            turn.tokens = self.provider.value_tokens(turn.completion)
        if turn.tokens is not None:
            tokens_log(self.provider, turn.tokens)
        self._turns.extend([user_turn, turn])

    def _collect_all_kwargs(
        self,
        kwargs: Optional[SubmitInputArgsT],
    ) -> SubmitInputArgsT:
        # Start collecting additional keyword args (from model parameters)
        all_kwargs = self.provider.translate_model_params(
            params=self._standard_model_params,
        )

        # Add any additional kwargs provided by the user
        if self.kwargs_chat:
            all_kwargs.update(self.kwargs_chat)

        if kwargs:
            all_kwargs.update(kwargs)

        return all_kwargs

    def _invoke_tool(self, request: ContentToolRequest):
        tool = self._tools.get(request.name)
        func = tool.func if tool is not None else None

        if func is None:
            yield self._handle_tool_error_result(
                request,
                error=RuntimeError("Unknown tool."),
            )
            return

        # First, invoke the request callbacks. If a ToolRejectError is raised,
        # treat it like a tool failure (i.e., gracefully handle it).
        result: ContentToolResult | None = None
        try:
            self._on_tool_request_callbacks.invoke(request)
        except ToolRejectError as e:
            yield self._handle_tool_error_result(request, e)
            return

        try:
            if isinstance(request.arguments, dict):
                res = func(**request.arguments)
            else:
                res = func(request.arguments)

            # Normalize res as a generator of results.
            if not inspect.isgenerator(res):

                def _as_generator(res):
                    yield res

                res = _as_generator(res)

            for x in res:
                if isinstance(x, ContentToolResult):
                    result = x
                else:
                    result = ContentToolResult(value=x)

                result.request = request

                self._on_tool_result_callbacks.invoke(result)
                yield result

        except Exception as e:
            yield self._handle_tool_error_result(request, e)

    async def _invoke_tool_async(self, request: ContentToolRequest):
        tool = self._tools.get(request.name)

        if tool is None:
            yield self._handle_tool_error_result(
                request,
                error=RuntimeError("Unknown tool."),
            )
            return

        if tool._is_async:
            func = tool.func
        else:
            func = wrap_async(tool.func)

        # First, invoke the request callbacks. If a ToolRejectError is raised,
        # treat it like a tool failure (i.e., gracefully handle it).
        result: ContentToolResult | None = None
        try:
            await self._on_tool_request_callbacks.invoke_async(request)
        except ToolRejectError as e:
            yield self._handle_tool_error_result(request, e)
            return

        # Invoke the tool (if it hasn't been rejected).
        try:
            if isinstance(request.arguments, dict):
                res = await func(**request.arguments)
            else:
                res = await func(request.arguments)

            # Normalize res into a generator of results.
            if not inspect.isasyncgen(res):

                async def _as_async_generator(res):
                    yield res

                res = _as_async_generator(res)

            async for x in res:
                if isinstance(x, ContentToolResult):
                    result = x
                else:
                    result = ContentToolResult(value=x)

                result.request = request
                await self._on_tool_result_callbacks.invoke_async(result)
                yield result

        except Exception as e:
            yield self._handle_tool_error_result(request, e)

    def _handle_tool_error_result(self, request: ContentToolRequest, error: Exception):
        warnings.warn(
            f"Calling tool '{request.name}' led to an error: {error}",
            ToolFailureWarning,
            stacklevel=2,
        )
        traceback.print_exc()
        log_tool_error(request.name, str(request.arguments), error)
        result = ContentToolResult(value=None, error=error, request=request)
        self._on_tool_result_callbacks.invoke(result)
        return result

    def _markdown_display(self, echo: EchoOptions) -> ChatMarkdownDisplay:
        """
        Get a markdown display object based on the echo option.

        The idea here is to use rich for consoles and IPython.display.Markdown
        for notebooks, since the latter is much more responsive to different
        screen sizes.
        """
        if echo == "none":
            return ChatMarkdownDisplay(MockMarkdownDisplay(), self)

        # rich does a lot to detect a notebook environment, but it doesn't
        # detect Quarto, or a Positron notebook
        from rich.console import Console

        is_web = Console().is_jupyter or is_quarto() or is_positron_notebook()

        opts = self._echo_options

        if is_web:
            display = IPyMarkdownDisplay(opts)
        else:
            display = LiveMarkdownDisplay(opts)

        return ChatMarkdownDisplay(display, self)

    def set_echo_options(
        self,
        rich_markdown: Optional[dict[str, Any]] = None,
        rich_console: Optional[dict[str, Any]] = None,
        css_styles: Optional[dict[str, str]] = None,
    ):
        """
        Set echo styling options for the chat.

        Parameters
        ----------
        rich_markdown
            A dictionary of options to pass to `rich.markdown.Markdown()`.
            This is only relevant when outputting to the console.
        rich_console
            A dictionary of options to pass to `rich.console.Console()`.
            This is only relevant when outputting to the console.
        css_styles
            A dictionary of CSS styles to apply to `IPython.display.Markdown()`.
            This is only relevant when outputing to the browser.
        """
        self._echo_options: EchoDisplayOptions = {
            "rich_markdown": rich_markdown or {},
            "rich_console": rich_console or {},
            "css_styles": css_styles or {},
        }

    def __str__(self):
        turns = self.get_turns(include_system_prompt=False)
        res = ""
        for turn in turns:
            icon = "👤" if isinstance(turn, UserTurn) else "🤖"
            res += f"## {icon} {turn.role.capitalize()} turn:\n\n{str(turn)}\n\n"
        return res

    def __repr__(self):
        turns = self.get_turns(include_system_prompt=True)
        tokens = self.get_tokens()
        tokens_asst = sum(u["tokens_total"] for u in tokens if u["role"] == "assistant")
        tokens_user = sum(u["tokens_total"] for u in tokens if u["role"] == "user")
        tokens_cached = sum(u["tokens_cached"] for u in tokens if u["role"] == "user")

        res = (
            f"<Chat {self.provider.name}/{self.provider.model} turns={len(turns)}"
            f" tokens={tokens_user + tokens_cached}/{tokens_asst}"
        )

        # Add cost info only if we can compute it
        cost = compute_cost(
            self.provider.name,
            self.provider.model,
            tokens_user,
            tokens_asst,
            tokens_cached,
        )
        if cost is not None:
            res += f" ${round(cost, ndigits=2)}"

        res += ">"
        for turn in turns:
            res += "\n" + turn.__repr__(indent=2)
        return res + "\n"

    def __deepcopy__(self, memo):
        result = self.__class__.__new__(self.__class__)

        # Avoid recursive references
        memo[id(self)] = result

        # Copy all attributes except the problematic provider attribute
        for key, value in self.__dict__.items():
            if key != "provider":
                setattr(result, key, copy.deepcopy(value, memo))
            else:
                setattr(result, key, value)

        return result


class ChatResponse:
    """
    Chat response object.

    An object that, when displayed, will simulatenously consume (if not
    already consumed) and display the response in a streaming fashion.

    This is useful for interactive use: if the object is displayed, it can
    be viewed as it is being generated. And, if the object is not displayed,
    it can act like an iterator that can be consumed by something else.

    Attributes
    ----------
    content
        The content of the chat response.

    Properties
    ----------
    consumed
        Whether the response has been consumed. If the response has been fully
        consumed, then it can no longer be iterated over, but the content can
        still be retrieved (via the `content` attribute).
    """

    def __init__(self, generator: Generator[str, None]):
        self._generator = generator
        self.content: str = ""

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        chunk = next(self._generator)
        self.content += chunk  # Keep track of accumulated content
        return chunk

    def get_content(self) -> str:
        """
        Get the chat response content as a string.
        """
        for _ in self:
            pass
        return self.content

    @property
    def consumed(self) -> bool:
        return inspect.getgeneratorstate(self._generator) == inspect.GEN_CLOSED

    def __str__(self) -> str:
        return self.get_content()


class ChatResponseAsync:
    """
    Chat response (async) object.

    An object that, when displayed, will simulatenously consume (if not
    already consumed) and display the response in a streaming fashion.

    This is useful for interactive use: if the object is displayed, it can
    be viewed as it is being generated. And, if the object is not displayed,
    it can act like an iterator that can be consumed by something else.

    Attributes
    ----------
    content
        The content of the chat response.

    Properties
    ----------
    consumed
        Whether the response has been consumed. If the response has been fully
        consumed, then it can no longer be iterated over, but the content can
        still be retrieved (via the `content` attribute).
    """

    def __init__(self, generator: AsyncGenerator[str, None]):
        self._generator = generator
        self.content: str = ""

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        chunk = await self._generator.__anext__()
        self.content += chunk  # Keep track of accumulated content
        return chunk

    async def get_content(self) -> str:
        "Get the chat response content as a string."
        async for _ in self:
            pass
        return self.content

    @property
    def consumed(self) -> bool:
        if sys.version_info < (3, 12):
            raise NotImplementedError(
                "Checking for consumed state is only supported in Python 3.12+"
            )
        return inspect.getasyncgenstate(self._generator) == inspect.AGEN_CLOSED


# ----------------------------------------------------------------------------
# Helpers for emitting content
# ----------------------------------------------------------------------------


def emit_user_contents(
    x: Turn,
    emit: Callable[[Content | str], None],
):
    if x.role != "user":
        raise ValueError("Expected a user turn")
    emit(f"## 👤 User turn:\n\n{str(x)}\n\n")
    emit_other_contents(x, emit)
    emit("\n\n## 🤖 Assistant turn:\n\n")


def emit_other_contents(
    x: Turn,
    emit: Callable[[Content | str], None],
):
    # Gather other content to emit in _reverse_ order
    to_emit: list[str] = []

    if isinstance(x, AssistantTurn) and x.finish_reason:
        to_emit.append(f"\n\n<< 🤖 finish reason: {x.finish_reason} \\>\\>\n\n")

    has_text = False
    has_other = False
    for content in reversed(x.contents):
        if isinstance(content, ContentText):
            has_text = True
        else:
            has_other = True
            to_emit.append(str(content))

    if has_text and has_other:
        if isinstance(x, UserTurn):
            to_emit.append("<< 👤 other content >>")
        else:
            to_emit.append("<< 🤖 other content >>")

    to_emit.reverse()

    emit("\n\n".join(to_emit))


# Helper/wrapper class to let Chat know about the currently active display
class ChatMarkdownDisplay:
    def __init__(self, display: MarkdownDisplay, chat: Chat):
        self._display = display
        self._chat = chat

    def __enter__(self):
        self._chat._current_display = self._display
        return self._display.__enter__()

    def __exit__(self, *args, **kwargs):
        result = self._display.__exit__(*args, **kwargs)
        self._chat._current_display = None
        return result

    def append(self, content):
        return self._display.echo(content)


class ToolFailureWarning(RuntimeWarning):
    pass


# By default warnings are shown once; we want to always show them.
warnings.simplefilter("always", ToolFailureWarning)


def is_quarto():
    return os.getenv("QUARTO_PYTHON", None) is not None


def is_positron_notebook():
    try:
        mode = get_ipython().session_mode  # noqa: F821 # type: ignore
        return mode == "notebook"
    except Exception:
        return False
