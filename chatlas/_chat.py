from __future__ import annotations

import os
from contextlib import contextmanager
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
)

from pydantic import BaseModel

from ._content import (
    Content,
    ContentJson,
    ContentText,
    ContentToolRequest,
    ContentToolResult,
)
from ._provider import Provider
from ._tools import Tool
from ._turn import Turn, user_turn
from ._typing_extensions import TypedDict

if TYPE_CHECKING:
    import rich.live


class AnyTypeDict(TypedDict, total=False):
    pass


SubmitInputArgsT = TypeVar("SubmitInputArgsT", bound=AnyTypeDict)
"""
A TypedDict representing the arguments that can be passed to the `.chat()`
method of a [](`~chatlas.Chat`) instance.
"""


class Chat(Generic[SubmitInputArgsT]):
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
        turns: Optional[Sequence[Turn]] = None,
    ):
        """
        Create a new chat object.

        Parameters
        ----------
        provider
            A [](`~chatlas.Provider`) object.
        turns
            A list of [](`~chatlas.Turn`) objects to initialize the chat with.
        """
        self.provider = provider
        self._turns: list[Turn] = list(turns or [])
        self.tools: dict[str, Tool] = {}

    def get_turns(
        self,
        *,
        include_system_prompt: bool = False,
    ) -> list[Turn]:
        """
        Get all the turns (i.e., message contents) in the chat.

        Parameters
        ----------
        include_system_prompt
            Whether to include the system prompt in the turns.
        """

        if not self._turns:
            return self._turns

        if not include_system_prompt and self._turns[0].role == "system":
            return self._turns[1:]
        return self._turns

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

        This method is primarily useful for clearing or setting the turns of the
        chat (i.e., limiting the context window).

        Parameters
        ----------
        turns
            The turns to set. Turns with the role "system" are not allowed.
        """
        if any(x.role == "system" for x in turns):
            idx = next(i for i, x in enumerate(turns) if x.role == "system")
            raise ValueError(
                f"Turn {idx} has a role 'system', which is not allowed. "
                "The system prompt must be set separately using the `.system_prompt` property. "
                "Consider removing this turn and setting the `.system_prompt` separately "
                "if you want to change the system prompt."
            )
        self._turns = list(turns)

    @property
    def system_prompt(self) -> str | None:
        """
        A property to get (or set) the system prompt for the chat.

        Returns
        -------
        str | None
            The system prompt (if any).
        """
        if self._turns and self._turns[0].role == "system":
            return self._turns[0].text
        return None

    @system_prompt.setter
    def system_prompt(self, value: str | None):
        if self._turns and self._turns[0].role == "system":
            self._turns.pop(0)
        if value is not None:
            self._turns.insert(0, Turn("system", value))

    def tokens(self) -> list[tuple[int, int]]:
        """
        Get the tokens for each turn in the chat.

        Returns
        -------
        list[tuple[int, int]]
            A list of tuples, where each tuple contains the start and end token
            indices for a turn.
        """
        return [turn.tokens for turn in self._turns]

    def app(
        self,
        *,
        stream: bool = True,
        launch_browser: bool = True,
        port: int = 0,
        kwargs: Optional[SubmitInputArgsT] = None,
    ):
        """
        Enter a chat browser to interact with the LLM.

        Parameters
        ----------
        stream
            Whether to stream the response (i.e., have the response appear in chunks).
        launch_browser
            Whether to launch a browser window.
        port
            The port to run the app on (the default is 0, which will choose a random port).
        kwargs
            Additional keyword arguments to pass to the method used for requesting
            the response.
        """

        try:
            from shiny import App, run_app, ui
        except ImportError:
            raise ImportError(
                "The `shiny` package is required for the `browser` method. "
                "Install it with `pip install shiny`."
            )

        app_ui = ui.page_fillable(
            ui.chat_ui("chat"),
            fillable_mobile=True,
        )

        def server(input):  # noqa: A002
            chat = ui.Chat(
                "chat",
                messages=[
                    {"role": turn.role, "content": turn.text}
                    for turn in self.get_turns()
                ],
            )

            @chat.on_user_submit
            async def _():
                user_input = chat.user_input()
                if user_input is None:
                    return
                response = self.chat(user_input, kwargs=kwargs, stream=stream)
                await chat.append_message_stream(response)

        run_app(
            App(app_ui, server),
            launch_browser=launch_browser,
            port=port,
        )

    def console(
        self,
        *,
        stream: bool = True,
        kwargs: Optional[SubmitInputArgsT] = None,
    ):
        """
        Enter a chat console to interact with the LLM.

        Press Ctrl+C to quit.

        Parameters
        ----------
        stream
            Whether to stream the response (i.e., have the response appear in chunks).
        kwargs
            Additional keyword arguments to pass to the method used for requesting
            the response

        Returns
        -------
        None
        """

        print("\nEntering chat console. Press Ctrl+C to quit.\n")

        while True:
            user_input = input("?> ")
            if user_input.strip().lower() in ("exit", "exit()"):
                break
            print("")
            self.chat(user_input, stream=stream, kwargs=kwargs)
            print("")

    def chat(
        self,
        *args: Content | str,
        echo: Literal["text", "all", "none"] = "text",
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
            Whether to echo text content, all content (i.e., tool calls), or no
            content.
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
        turn = user_turn(*args)

        response = ChatResponse(
            self._chat_impl(
                turn,
                echo=echo,
                stream=stream,
                kwargs=kwargs,
            )
        )

        for _ in response:
            pass

        return response

    async def chat_async(
        self,
        *args: Content | str,
        echo: Literal["text", "all", "none"] = "text",
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
            Whether to echo text content, all content (i.e., tool calls, images,
            etc), or no content.
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
        turn = user_turn(*args)

        response = ChatResponseAsync(
            self._chat_impl_async(
                turn,
                echo=echo,
                stream=stream,
                kwargs=kwargs,
            ),
        )

        async for _ in response:
            pass

        return response

    def stream(
        self,
        *args: Content | str,
        echo: Literal["text", "all", "none"] = "none",
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> ChatResponse:
        """
        Generate a response from the chat in a streaming fashion.

        Parameters
        ----------
        args
            The user input(s) to generate a response from.
        echo
            Whether to echo text content, all content (i.e., tool calls), or no
            content.
        kwargs
            Additional keyword arguments to pass to the method used for requesting
            the response.

        Returns
        -------
        ChatResponse
            An (unconsumed) response from the chat. Iterate over this object to
            consume the response.
        """
        turn = user_turn(*args)
        return ChatResponse(
            self._chat_impl(
                turn,
                stream=True,
                echo=echo,
                kwargs=kwargs,
            ),
        )

    async def stream_async(
        self,
        *args: Content | str,
        echo: Literal["text", "all", "none"] = "none",
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> ChatResponseAsync:
        """
        Generate a response from the chat in a streaming fashion asynchronously.

        Parameters
        ----------
        args
            The user input(s) to generate a response from.
        echo
            Whether to echo text content, all content (i.e., tool calls), or no
            content.
        kwargs
            Additional keyword arguments to pass to the method used for requesting
            the response.

        Returns
        -------
        ChatResponseAsync
            An (unconsumed) response from the chat. Iterate over this object to
            consume the response.
        """
        turn = user_turn(*args)
        return ChatResponseAsync(
            self._chat_impl_async(
                turn,
                stream=True,
                echo=echo,
                kwargs=kwargs,
            ),
        )

    def extract_data(
        self,
        *args: Content | str,
        data_model: type[BaseModel],
        echo: Literal["text", "all", "none"] = "none",
    ) -> dict[str, Any]:
        """
        Extract structured data from the given input.

        Parameters
        ----------
        args
            The input to extract data from.
        data_model
            A Pydantic model describing the structure of the data to extract.
        echo
            Whether to echo text content, all content (i.e., tool calls), or no content.

        Returns
        -------
        dict[str, Any]
            The extracted data.
        """

        with JupyterFriendlyLive() as live:
            response = ChatResponse(
                self._submit_turns(
                    user_turn(*args),
                    data_model=data_model,
                    echo=echo,
                    live=live,
                    stream=echo != "none",
                )
            )

        for _ in response:
            pass

        turn = self.get_last_turn()
        assert turn is not None

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

    async def extract_data_async(
        self,
        *args: Content | str,
        data_model: type[BaseModel],
        echo: Literal["text", "all", "none"] = "none",
    ) -> dict[str, Any]:
        """
        Extract structured data from the given input asynchronously.

        Parameters
        ----------
        args
            The input to extract data from.
        data_model
            A Pydantic model describing the structure of the data to extract.
        echo
            Whether to echo text content, all content (i.e., tool calls), or no content

        Returns
        -------
        dict[str, Any]
            The extracted data.
        """

        with JupyterFriendlyLive() as live:
            response = ChatResponseAsync(
                self._submit_turns_async(
                    user_turn(*args),
                    data_model=data_model,
                    echo=echo,
                    live=live,
                    stream=echo != "none",
                )
            )

        async for _ in response:
            pass

        turn = self.get_last_turn()
        assert turn is not None

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

    def register_tool(
        self,
        func: Callable[..., Any] | Callable[..., Awaitable[Any]],
        *,
        model: Optional[type[BaseModel]] = None,
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
        from chatlas import ChatOpenAI, Tool


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
        from chatlas import ChatOpenAI, Tool
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
        model
            A Pydantic model that describes the input parameters for the function.
            If not provided, the model will be inferred from the function's type hints.
            The primary reason why you might want to provide a model in
            Note that the name and docstring of the model takes precedence over the
            name and docstring of the function.
        """
        tool = Tool(func, model=model)
        self.tools[tool.name] = tool

    def _chat_impl(
        self,
        user_turn: Turn,
        echo: Literal["text", "all", "none"],
        stream: bool,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> Generator[str, None, None]:
        user_turn_result: Turn | None = user_turn

        with JupyterFriendlyLive() as live:
            while user_turn_result is not None:
                for chunk in self._submit_turns(
                    user_turn_result,
                    echo=echo,
                    live=live,
                    stream=stream,
                    kwargs=kwargs,
                ):
                    yield chunk
                user_turn_result = self._invoke_tools()

    async def _chat_impl_async(
        self,
        user_turn: Turn,
        echo: Literal["text", "all", "none"],
        stream: bool,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> AsyncGenerator[str, None]:
        user_turn_result: Turn | None = user_turn

        with JupyterFriendlyLive() as live:
            while user_turn_result is not None:
                async for chunk in self._submit_turns_async(
                    user_turn_result,
                    echo=echo,
                    live=live,
                    stream=stream,
                    kwargs=kwargs,
                ):
                    yield chunk
                user_turn_result = await self._invoke_tools_async()

    def _submit_turns(
        self,
        user_turn: Turn,
        echo: Literal["text", "all", "none"],
        live: "rich.live.Live",
        stream: bool,
        data_model: type[BaseModel] | None = None,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> Generator[str, None, None]:
        if any(x._is_async for x in self.tools.values()):
            raise ValueError("Cannot use async tools in a synchronous chat")

        emit = emitter(echo, live)

        if echo == "all":
            emit_user_contents(user_turn, emit)

        if stream:
            response = self.provider.chat_perform(
                stream=True,
                turns=[*self._turns, user_turn],
                tools=self.tools,
                data_model=data_model,
                kwargs=kwargs,
            )

            result = None
            for chunk in response:
                text = self.provider.stream_text(chunk)
                if text:
                    emit(text)
                    yield text
                result = self.provider.stream_merge_chunks(result, chunk)

            turn = self.provider.stream_turn(
                result, has_data_model=data_model is not None
            )

            if echo == "all":
                emit_other_contents(turn, emit)

        else:
            response = self.provider.chat_perform(
                stream=False,
                turns=[*self._turns, user_turn],
                tools=self.tools,
                data_model=data_model,
                kwargs=kwargs,
            )

            turn = self.provider.value_turn(
                response, has_data_model=data_model is not None
            )
            if turn.text:
                emit(turn.text)
                yield turn.text

            if echo == "all":
                emit_other_contents(turn, emit)

        self._turns.extend([user_turn, turn])

    async def _submit_turns_async(
        self,
        user_turn: Turn,
        echo: Literal["text", "all", "none"],
        live: "rich.live.Live",
        stream: bool,
        data_model: type[BaseModel] | None = None,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> AsyncGenerator[str, None]:
        emit = emitter(echo, live)

        if echo == "all":
            emit_user_contents(user_turn, emit)

        if stream:
            response = await self.provider.chat_perform_async(
                stream=True,
                turns=[*self._turns, user_turn],
                tools=self.tools,
                data_model=data_model,
                kwargs=kwargs,
            )

            result = None
            async for chunk in response:
                text = self.provider.stream_text(chunk)
                if text:
                    emit(text)
                    yield text
                result = self.provider.stream_merge_chunks(result, chunk)

            turn = self.provider.stream_turn(
                result, has_data_model=data_model is not None
            )

            if echo == "all":
                emit_other_contents(turn, emit)

        else:
            response = await self.provider.chat_perform_async(
                stream=False,
                turns=[*self._turns, user_turn],
                tools=self.tools,
                data_model=data_model,
                kwargs=kwargs,
            )

            turn = self.provider.value_turn(
                response, has_data_model=data_model is not None
            )
            if turn.text:
                emit(turn.text)
                yield turn.text

            if echo == "all":
                emit_other_contents(turn, emit)

        self._turns.extend([user_turn, turn])

    def _invoke_tools(self) -> Turn | None:
        turn = self.get_last_turn()
        if turn is None:
            return None

        results: list[ContentToolResult] = []
        for x in turn.contents:
            if isinstance(x, ContentToolRequest):
                tool_def = self.tools.get(x.name, None)
                func = tool_def.func if tool_def is not None else None
                results.append(self._invoke_tool(func, x.arguments, x.id))

        if not results:
            return None

        return Turn("user", results)

    async def _invoke_tools_async(self) -> Turn | None:
        turn = self.get_last_turn()
        if turn is None:
            return None

        results: list[ContentToolResult] = []
        for x in turn.contents:
            if isinstance(x, ContentToolRequest):
                tool_def = self.tools.get(x.name, None)
                func = tool_def.func if tool_def is not None else None
                results.append(await self._invoke_tool_async(func, x.arguments, x.id))

        if not results:
            return None

        return Turn("user", results)

    @staticmethod
    def _invoke_tool(
        func: Callable[..., Any] | None,
        arguments: object,
        id_: str,
    ) -> ContentToolResult:
        if func is None:
            return ContentToolResult(id_, None, "Unknown tool")

        try:
            if isinstance(arguments, dict):
                result = func(**arguments)
            else:
                result = func(arguments)

            return ContentToolResult(id_, result, None)
        except Exception as e:
            return ContentToolResult(id_, None, str(e))

    @staticmethod
    async def _invoke_tool_async(
        func: Callable[..., Awaitable[Any]] | None,
        arguments: object,
        id_: str,
    ) -> ContentToolResult:
        if func is None:
            return ContentToolResult(id_, None, "Unknown tool")

        try:
            if isinstance(arguments, dict):
                result = await func(**arguments)
            else:
                result = await func(arguments)

            return ContentToolResult(id_, result, None)
        except Exception as e:
            return ContentToolResult(id_, None, str(e))

    def __str__(self):
        turns = self.get_turns(include_system_prompt=True)
        tokens = sum(sum(turn.tokens) for turn in turns)
        output = f"<Chat turns={len(turns)} tokens={tokens}>\n"
        for turn in turns:
            output += f"--- {turn.role} ---\n"
            for content in turn.contents:
                output += f"{content}\n"
        return output

    def __repr__(self):
        return str(self)


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
        return self._generator.gi_frame is None

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
        return self._generator.ag_frame is None


# ----------------------------------------------------------------------------
# Jupyter-friendly rich live context managers
# ----------------------------------------------------------------------------


@contextmanager
def JupyterFriendlyLive():
    """
    A special `rich.live.Live` context manager with special handling for Jupyter.
    """
    import rich.live

    with JupyterFriendlyConsole() as console:
        with rich.live.Live(console=console, auto_refresh=False) as live:
            yield live


@contextmanager
def JupyterFriendlyConsole():
    import rich.console
    import rich.jupyter

    # Force jupyter mode if running in Quarto (so that side-effects are captured)
    is_quarto = os.getenv("QUARTO_PYTHON", None) is not None
    console = rich.console.Console(
        force_jupyter=True if is_quarto else None,
    )

    # Prevent rich from inserting line breaks in a Jupyter context
    # (and, instead, rely on the browser to wrap text)
    console.soft_wrap = console.is_jupyter

    html_format = rich.jupyter.JUPYTER_HTML_FORMAT

    # Remove the `white-space:pre;` CSS style since the LLM's response is
    # (usually) already pre-formatted and essentially assumes a browser context
    rich.jupyter.JUPYTER_HTML_FORMAT = html_format.replace(
        "white-space:pre;", "text-wrap-mode:wrap;word-break:break-word;"
    )
    yield console

    rich.jupyter.JUPYTER_HTML_FORMAT = html_format


# ----------------------------------------------------------------------------
# Helpers for emitting content
# ----------------------------------------------------------------------------


def emitter(
    echo: Literal["text", "all", "none"],
    live: "rich.live.Live",
) -> Callable[[Content | str], None]:
    if echo == "none":
        return lambda _: None

    from rich.markdown import Markdown

    def emit(x: Content | str):
        x = str(x)
        current = live.get_renderable()
        if isinstance(current, Markdown):
            x = current.markup + x
        live.update(Markdown(x), refresh=True)

    return emit


def emit_user_contents(
    x: Turn,
    emit: Callable[[Content | str], None],
):
    if x.role != "user":
        raise ValueError("Expected a user turn")
    emit(f"\\>\\> **`User` turn:**\n\n{x.text}\n\n")
    emit_other_contents(x, emit)
    emit("\n\n<< **`Assistant` turn:**\n\n")


def emit_other_contents(
    x: Turn,
    emit: Callable[[Content | str], None],
):
    if all(isinstance(x, ContentText) for x in x.contents):
        return

    if x.role == "user":
        emit("\n\n\\>\\>\\> **Other `user` content:**\n\n")
    else:
        emit("\n\n<<< **Other `assistant` content:**\n\n")

    for content in x.contents:
        if not isinstance(content, ContentText):
            # For some really odd reason ContentToolResult won't display without
            # a non-whitespace character before it. The &nbsp; is a hack to work
            # around that, but it's also decent for readability.
            emit(f"&nbsp;{str(content)}\n\n")
