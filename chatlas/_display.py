import logging
from abc import ABC, abstractmethod
from typing import Any
from uuid import uuid4

from rich.live import Live
from rich.logging import RichHandler

from ._live_render import LiveRender
from ._logging import logger
from ._typing_extensions import TypedDict


class MarkdownDisplay(ABC):
    """Base class for displaying markdown content in different environments."""

    @abstractmethod
    def echo(self, content: str):
        """
        Display the provided markdown string. This will append the content
        to the current display.
        """
        pass

    @abstractmethod
    def __enter__(self) -> "MarkdownDisplay":
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        pass


class MockMarkdownDisplay(MarkdownDisplay):
    def echo(self, content: str):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class LiveMarkdownDisplay(MarkdownDisplay):
    """
    Stream chunks of markdown into a rich-based live updating console.
    """

    def __init__(self, echo_options: "EchoDisplayOptions"):
        from rich.console import Console

        self.content: str = ""
        live = Live(
            auto_refresh=False,
            console=Console(
                **echo_options["rich_console"],
            ),
        )

        # Monkeypatch LiveRender() with our own version that add "crop_above"
        # https://github.com/Textualize/rich/blob/43d3b047/rich/live.py#L87-L89
        live.vertical_overflow = "crop_above"
        live._live_render = LiveRender(  # pyright: ignore[reportAttributeAccessIssue]
            live.get_renderable(), vertical_overflow="crop_above"
        )

        self.live = live

        self._markdown_options = echo_options["rich_markdown"]

    def echo(self, content: str):
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
        # Live() isn't smart enough to know to automatically display logs when
        # when they get handled while it Live() is active.
        # However, if the logging handler is a RichHandler, it can be told
        # about the live console so it can add logs to the top of the Live console.
        handlers = [*logging.getLogger().handlers, *logger.handlers]
        for h in handlers:
            if isinstance(h, RichHandler):
                h.console = self.live.console

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.content = ""
        return self.live.__exit__(exc_type, exc_value, traceback)


class IPyMarkdownDisplay(MarkdownDisplay):
    """
    Stream chunks of markdown into an IPython notebook.
    """

    def __init__(self, echo_options: "EchoDisplayOptions"):
        self.content: str = ""
        self._css_styles = echo_options["css_styles"]

    def echo(self, content: str):
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


class EchoDisplayOptions(TypedDict):
    rich_markdown: dict[str, Any]
    rich_console: dict[str, Any]
    css_styles: dict[str, str]
