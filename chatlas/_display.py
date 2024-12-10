from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from rich.live import Live
from rich.logging import RichHandler

from ._logging import logger
from ._typing_extensions import TypedDict


class MarkdownDisplay(ABC):
    @abstractmethod
    def update(self, content: str):
        pass

    @abstractmethod
    def __enter__(self) -> "MarkdownDisplay":
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass


class MockMarkdownDisplay(MarkdownDisplay):
    def update(self, content: str):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class LiveMarkdownDisplay(MarkdownDisplay):
    """
    Stream chunks of markdown into a rich-based live updating console.
    """

    def __init__(self, echo_options: "EchoOptions"):
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
        # When logging is enabled, the RichHandler should be the second handler.
        # That handler needs to know about the live console so it can add logs to it.
        if len(logger.handlers) > 1 and isinstance(logger.handlers[1], RichHandler):
            logger.handlers[1].console = self.live.console
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.content = ""
        return self.live.__exit__(exc_type, exc_value, traceback)


class IPyMarkdownDisplay(MarkdownDisplay):
    """
    Stream chunks of markdown into an IPython notebook.
    """

    def __init__(self, echo_options: "EchoOptions"):
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
