from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from threading import Thread
from typing import (
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
from uuid import uuid4

from pydantic import BaseModel
from rich.live import Live

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
from ._utils import html_escape


class AnyTypeDict(TypedDict, total=False):
    pass


SubmitInputArgsT = TypeVar("SubmitInputArgsT", bound=AnyTypeDict)
"""
A TypedDict representing the arguments that can be passed to the `.chat()`
method of a [](`~chatlas.Chat`) instance.
"""

CompletionT = TypeVar("CompletionT")


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
        self._echo_options: EchoOptions = {
            "rich_markdown": {},
            "rich_console": {},
            "css_styles": {},
        }

    def turns(
        self,
        *,
        include_system_prompt: bool = False,
    ) -> list[Turn[CompletionT]]:
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

    def last_turn(
        self,
        *,
        role: Literal["assistant", "user", "system"] = "assistant",
    ) -> Turn[CompletionT] | None:
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
        Get the system prompt for the chat.
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

    def tokens(self) -> list[tuple[int, int] | None]:
        """
        Get the tokens for each turn in the chat.

        Returns
        -------
        list[tuple[int, int] | None]
            A list of tuples, where each tuple contains the start and end token
            indices for a turn.
        """
        return [turn.tokens for turn in self._turns]

    def app(
        self,
        *,
        stream: bool = True,
        port: int = 0,
        launch_browser: bool = True,
        bg_thread: Optional[bool] = None,
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
        launch_browser
            Whether to launch a browser window.
        bg_thread
            Whether to run the app in a background thread. If `None`, the app will
            run in a background thread if the current environment is a notebook.
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
                    {"role": turn.role, "content": turn.text} for turn in self.turns()
                ],
            )

            @chat.on_user_submit
            async def _():
                user_input = chat.user_input()
                if user_input is None:
                    return
                if stream:
                    await chat.append_message_stream(
                        self.stream(user_input, kwargs=kwargs)
                    )
                else:
                    await chat.append_message(str(self.chat(user_input, kwargs=kwargs)))

        app = App(app_ui, server)

        def _run_app():
            run_app(app, launch_browser=launch_browser, port=port)

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
        echo: Literal["text", "all", "none"] = "text",
        stream: bool = True,
        kwargs: Optional[SubmitInputArgsT] = None,
    ):
        """
        Enter a chat console to interact with the LLM.

        To quit, input 'exit' or press Ctrl+C.

        Parameters
        ----------
        echo
            Whether to echo text content, all content (i.e., tool calls), or no
            content.
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
        stream: bool = False,
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
        stream
            Whether to stream the response (i.e., have the response appear in chunks).

        Returns
        -------
        dict[str, Any]
            The extracted data.
        """

        with self._display_context() as display:
            response = ChatResponse(
                self._submit_turns(
                    user_turn(*args),
                    data_model=data_model,
                    echo=echo,
                    display=display,
                    stream=stream,
                )
            )

        for _ in response:
            pass

        turn = self.last_turn()
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
        stream: bool = False,
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
        stream
            Whether to stream the response (i.e., have the response appear in chunks).
            Defaults to `True` if `echo` is not "none".

        Returns
        -------
        dict[str, Any]
            The extracted data.
        """

        with self._display_context() as display:
            response = ChatResponseAsync(
                self._submit_turns_async(
                    user_turn(*args),
                    data_model=data_model,
                    echo=echo,
                    display=display,
                    stream=stream,
                )
            )

        async for _ in response:
            pass

        turn = self.last_turn()
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

    def export(
        self,
        filename: str | Path,
        *,
        turns: Optional[Sequence[Turn]] = None,
        title: Optional[str] = None,
        include: Literal["text", "all"] = "text",
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
            The `.turns()` to export. If not provided, the chat's current turns
            will be used.
        title
            A title to place at the top of the exported file.
        overwrite
            Whether to overwrite the file if it already exists.
        include
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
            turns = self.turns(include_system_prompt=False)
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
        contents = ""
        for turn in turns:
            turn_content = "\n\n".join(
                [
                    str(content)
                    for content in turn.contents
                    if include == "all" or isinstance(content, ContentText)
                ]
            )
            if is_html:
                msg_type = "user" if turn.role == "user" else "chat"
                content_attr = html_escape(turn_content)
                turn_content = f"<shiny-{msg_type}-message content='{content_attr}'></shiny-{msg_type}-message>"
            else:
                turn_content = f"## {turn.role.capitalize()}\n\n{turn_content}"
            contents += f"{turn_content}\n\n"

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

    def _chat_impl(
        self,
        user_turn: Turn,
        echo: Literal["text", "all", "none"],
        stream: bool,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> Generator[str, None, None]:
        user_turn_result: Turn | None = user_turn

        with self._display_context() as display:
            while user_turn_result is not None:
                for chunk in self._submit_turns(
                    user_turn_result,
                    echo=echo,
                    display=display,
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

        with self._display_context() as display:
            while user_turn_result is not None:
                async for chunk in self._submit_turns_async(
                    user_turn_result,
                    echo=echo,
                    display=display,
                    stream=stream,
                    kwargs=kwargs,
                ):
                    yield chunk
                user_turn_result = await self._invoke_tools_async()

    def _submit_turns(
        self,
        user_turn: Turn,
        echo: Literal["text", "all", "none"],
        display: LiveMarkdownDisplay | IPyMarkdownDisplay,
        stream: bool,
        data_model: type[BaseModel] | None = None,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> Generator[str, None, None]:
        if any(x._is_async for x in self.tools.values()):
            raise ValueError("Cannot use async tools in a synchronous chat")

        emit = emitter(echo, display)

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
                result,
                has_data_model=data_model is not None,
                stream=response,
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
        display: LiveMarkdownDisplay | IPyMarkdownDisplay,
        stream: bool,
        data_model: type[BaseModel] | None = None,
        kwargs: Optional[SubmitInputArgsT] = None,
    ) -> AsyncGenerator[str, None]:
        emit = emitter(echo, display)

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

            turn = await self.provider.stream_turn_async(
                result,
                has_data_model=data_model is not None,
                stream=response,
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
        turn = self.last_turn()
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
        turn = self.last_turn()
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

    @contextmanager
    def _display_context(
        self,
    ) -> Generator[LiveMarkdownDisplay | IPyMarkdownDisplay, None, None]:
        opts = self._echo_options
        display = LiveMarkdownDisplay(opts)

        # rich seems to be pretty good at detecting a (Jupyter) notebook
        # context, so utilize that, but use IPython.display.Markdown instead if
        # we're in a notebook (or Quarto) since that's a much more responsive
        # way to display markdown
        is_web = (
            display.live.console.is_jupyter
            or os.getenv("QUARTO_PYTHON", None) is not None
        )

        if is_web:
            with IPyMarkdownDisplay(opts) as d:
                yield d
        else:
            with display:
                yield display

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
        self._echo_options: EchoOptions = {
            "rich_markdown": rich_markdown or {},
            "rich_console": rich_console or {},
            "css_styles": css_styles or {},
        }

    def __str__(self):
        turns = self.turns(include_system_prompt=False)
        res = ""
        for turn in turns:
            icon = "👤" if turn.role == "user" else "🤖"
            res += f"## {icon} {turn.role.capitalize()} turn:\n\n{str(turn)}\n\n"
        return res

    def __repr__(self):
        turns = self.turns(include_system_prompt=True)
        tokens = sum(sum(turn.tokens) for turn in turns if turn.tokens)
        res = f"<Chat turns={len(turns)} tokens={tokens}>"
        for turn in turns:
            res += "\n" + turn.__repr__(indent=2)
        return res + "\n"


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
# Helpers for emitting content
# ----------------------------------------------------------------------------


def emitter(
    echo: Literal["text", "all", "none"],
    display: LiveMarkdownDisplay | IPyMarkdownDisplay,
) -> Callable[[Content | str], None]:
    if echo == "none":
        return lambda _: None
    else:
        return lambda x: display.update(str(x))


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

    if x.finish_reason:
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
        if x.role == "user":
            to_emit.append("<< 👤 other content >>")
        else:
            to_emit.append("<< 🤖 other content >>")

    to_emit.reverse()

    emit("\n\n".join(to_emit))


class LiveMarkdownDisplay:
    def __init__(self, echo_options: EchoOptions):
        from rich.console import Console

        self.content: str = ""
        self.live = Live(
            auto_refresh=False,
            vertical_overflow="visible",
            console=Console(
                **echo_options["rich_console"],
            ),
        )
        self._markdown_options = echo_options["rich_markdown"]

    def update(self, content: str):
        from rich.markdown import Markdown

        self.content += content
        self.live.update(
            Markdown(
                self.content,
                **self._markdown_options,
            ),
            refresh=True,
        )

    def __enter__(self):
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.content = ""
        return self.live.__exit__(exc_type, exc_value, traceback)


class IPyMarkdownDisplay:
    def __init__(self, echo_options: EchoOptions):
        self.content: str = ""
        self._css_styles = echo_options["css_styles"]

    def update(self, content: str):
        from IPython.display import Markdown, update_display

        self.content += content
        update_display(
            Markdown(self.content),
            display_id=self._ipy_display_id,
        )

    def _init_display(self) -> str:
        try:
            from IPython.display import HTML, Markdown, display
        except ImportError:
            raise ImportError(
                "The IPython package is required for displaying content in a Jupyter notebook. "
                "Install it with `pip install ipython`."
            )

        if self._css_styles:
            id_ = uuid4().hex
            css = "".join(f"{k}: {v}; " for k, v in self._css_styles.items())
            display(HTML(f"<style>#{id_} + .chatlas-markdown {{ {css} }}</style>"))
            display(HTML(f"<div id='{id_}' class='chatlas-markdown'>"))
        else:
            # Unfortunately, there doesn't seem to be a proper way to wrap
            # Markdown() in a div?
            display(HTML("<div class='chatlas-markdown'>"))

        handle = display(Markdown(""), display_id=True)
        if handle is None:
            raise ValueError("Failed to create display handle")
        return handle.display_id

    def __enter__(self):
        self._ipy_display_id = self._init_display()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._ipy_display_id = None


class EchoOptions(TypedDict):
    rich_markdown: dict[str, Any]
    rich_console: dict[str, Any]
    css_styles: dict[str, str]
