import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union
from uuid import uuid4

from rich._loop import loop_last
from rich.live import Live
from rich.logging import RichHandler

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions

from ._content import (
    TOOL_CSS,
    Content,
    ContentText,
    ContentThinking,
    ContentThinkingDelta,
    ContentToolResult,
)
from ._live_render import LiveRender
from ._logging import logger
from ._typing_extensions import TypedDict


class MarkdownDisplay(ABC):
    """
    Base class for displaying markdown content in different environments.

    Accumulates what it's told to display as a list of segments rather than one
    string, so subclasses can render each content type on its own terms —
    reasoning as a collapsible/dimmed aside, a large tool result inside a bounded
    container, everything else as markdown.
    """

    def __init__(self, echo_options: "EchoDisplayOptions"):
        self._echo_options = echo_options
        self._segments: list["DisplaySegment"] = []

    def echo(self, content: Union[str, Content]):
        """
        Display the provided content. This will append the content to the
        current display.
        """
        self._append(content)
        self._render()

    @abstractmethod
    def _render(self) -> None:
        """Repaint the accumulated segments."""

    def _append(self, content: Union[str, Content]) -> None:
        if isinstance(content, ContentThinking):
            # A complete block, so there's nothing left to stream into it.
            self._segments.append(
                ThinkingSegment(thinking=content.thinking, is_open=False)
            )
            return

        if isinstance(content, ContentThinkingDelta):
            self._append_thinking_delta(content)
            return

        # Anything else means reasoning is over, even if the provider never sent
        # a closing delta.
        self._close_thinking()

        if isinstance(content, ContentText):
            content = content.text

        if isinstance(content, str):
            last = self._segments[-1] if self._segments else None
            if isinstance(last, TextSegment):
                last.text += content
            else:
                self._segments.append(TextSegment(text=content))
        else:
            self._segments.append(ContentSegment(content=content))

    def _append_thinking_delta(self, content: ContentThinkingDelta) -> None:
        last = self._segments[-1] if self._segments else None
        segment = last if isinstance(last, ThinkingSegment) and last.is_open else None

        if segment is None:
            # A stray closing delta has nothing to close.
            if content.phase == "end":
                return
            segment = ThinkingSegment()
            self._segments.append(segment)

        segment.thinking += content.thinking
        if content.phase == "end":
            segment.is_open = False

    def _close_thinking(self) -> None:
        last = self._segments[-1] if self._segments else None
        if isinstance(last, ThinkingSegment):
            last.is_open = False

    def __enter__(self) -> "MarkdownDisplay":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._segments = []


class MockMarkdownDisplay(MarkdownDisplay):
    def echo(self, content: Union[str, Content]):
        pass

    def _render(self) -> None:
        pass


class LiveMarkdownDisplay(MarkdownDisplay):
    """
    Stream chunks of markdown into a rich-based live updating console.
    """

    def __init__(self, echo_options: "EchoDisplayOptions"):
        from rich.console import Console

        super().__init__(echo_options)

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

    def _render(self) -> None:
        from rich.console import Group
        from rich.text import Text

        # rich puts no space between the members of a Group, but the accumulated
        # markdown this replaced had blank lines between blocks. Restore them so
        # a panel or tool result doesn't abut the text around it.
        spaced: list[Any] = []
        for renderable in self._renderables():
            if spaced:
                spaced.append(Text(""))
            spaced.append(renderable)

        self.live.update(Group(*spaced), refresh=True)

    def _renderables(self) -> list[Any]:
        out: list[Any] = []
        for segment in self._segments:
            if isinstance(segment, TextSegment):
                out.append(self._markdown(segment.text))
            elif isinstance(segment, ThinkingSegment):
                out.append(
                    ThinkingPanel(
                        self._markdown(segment.thinking),
                        max_lines=self._echo_options["thinking_max_lines"],
                    )
                )
            else:
                out.append(self._markdown(self._content_markdown(segment.content)))
        return out

    def _content_markdown(self, content: Content) -> str:
        if isinstance(content, ContentToolResult):
            return content.to_display_markdown(
                max_lines=self._echo_options["tool_result_max_lines"]
            )
        return str(content)

    def _markdown(self, text: str) -> Any:
        from rich.markdown import Markdown

        return Markdown(text, **self._markdown_options)

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
        super().__exit__(exc_type, exc_value, traceback)
        return self.live.__exit__(exc_type, exc_value, traceback)


class IPyMarkdownDisplay(MarkdownDisplay):
    """
    Stream chunks of markdown into an IPython notebook.
    """

    def _render(self) -> None:
        from IPython.display import Markdown, update_display

        update_display(
            Markdown(self._markdown()),
            display_id=self._ipy_display_id,
        )

    def _markdown(self) -> str:
        parts: list[str] = []
        for segment in self._segments:
            if isinstance(segment, TextSegment):
                parts.append(segment.text)
            elif isinstance(segment, ThinkingSegment):
                parts.append(thinking_html(segment.thinking, is_open=segment.is_open))
            else:
                parts.append(self._content_markdown(segment.content))
        return "".join(parts)

    def _content_markdown(self, content: Content) -> str:
        if isinstance(content, ContentToolResult):
            # Already collapsed, and TOOL_CSS bounds its height once expanded.
            return f"\n\n{content.to_html()}\n\n"
        return f"\n\n{content}\n\n"

    def _init_display(self) -> str:
        try:
            from IPython.display import HTML, Markdown, display
        except ImportError:
            raise ImportError(
                "The IPython package is required for displaying content in a Jupyter notebook. "
                "Install it with `pip install ipython`."
            )

        max_height = self._echo_options["tool_result_max_height"]
        wrapper_style = f"--chatlas-tool-result-max-height: {max_height}"

        if self._css_styles:
            id_ = uuid4().hex
            css = "".join(f"{k}: {v}; " for k, v in self._css_styles.items())
            display(
                HTML(
                    f"<style>{TOOL_CSS}\n#{id_} + .chatlas-markdown {{ {css} }}</style>"
                )
            )
            display(
                HTML(
                    f"<div id='{id_}' class='chatlas-markdown' style='{wrapper_style}'>"
                )
            )
        else:
            display(HTML(f"<style>{TOOL_CSS}</style>"))
            # Unfortunately, there doesn't seem to be a proper way to wrap
            # Markdown() in a div?
            display(HTML(f"<div class='chatlas-markdown' style='{wrapper_style}'>"))

        handle = display(Markdown(""), display_id=True)
        if handle is None:
            raise ValueError("Failed to create display handle")
        return handle.display_id

    @property
    def _css_styles(self) -> dict[str, str]:
        return self._echo_options["css_styles"]

    def __enter__(self):
        self._ipy_display_id = self._init_display()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        self._ipy_display_id = None


class EchoDisplayOptions(TypedDict):
    rich_markdown: dict[str, Any]
    rich_console: dict[str, Any]
    css_styles: dict[str, str]
    tool_result_max_lines: Optional[int]
    tool_result_max_height: Optional[str]
    thinking_max_lines: Optional[int]


DEFAULT_TOOL_RESULT_MAX_LINES = 20
DEFAULT_TOOL_RESULT_MAX_HEIGHT = "400px"
# Lower than the tool-result cap on purpose: reasoning is an aside, so it should
# take up less room than the answer it precedes.
DEFAULT_THINKING_MAX_LINES = 10


def thinking_html(thinking: str, is_open: bool) -> str:
    """
    Wrap reasoning in a `<details>` block.

    The blank lines matter: they end the opening HTML block so the body is still
    markdown-rendered (CommonMark html block type 6), which is the same thing
    `Chat.export()` relies on for its system prompt block.
    """
    open_attr = " open" if is_open else ""
    return (
        f"\n\n<details{open_attr}><summary>Thinking</summary>"
        f"\n\n{thinking}\n\n</details>\n\n"
    )


class ThinkingPanel:
    """
    A panel of reasoning, capped at `max_lines` by dropping the *oldest* lines.

    Two things make this different from truncating a tool result:

    - It counts *wrapped* lines, not source lines. Reasoning is prose, so it
      wraps well past its own newline count (a real Anthropic sample: 35
      newlines, 44 rendered lines at width 70) and some providers send no
      newlines at all. So the cap has to render first and crop the result,
      which also can't break the markdown the way slicing the source could
      (e.g. halving a code fence).
    - It keeps the tail rather than the head. The newest reasoning is what's
      streaming in; cropping the other way would pin the panel to text the
      reader has already finished, and look like a hang.
    """

    def __init__(self, body: Any, max_lines: Optional[int]):
        self.body = body
        self.max_lines = max_lines

    def __rich_console__(self, console: "Console", options: "ConsoleOptions"):
        from rich.panel import Panel
        from rich.segment import Segment, Segments

        title = "Thinking"
        body = self.body

        if self.max_lines is not None:
            # Clamped to >=1 because `lines[-0:]` is the whole list, so a cap of 0
            # would keep everything while the title claimed it all got dropped.
            keep = max(1, self.max_lines)
            # Panel spends two columns on borders and two on padding.
            inner = options.update_width(max(options.max_width - 4, 1))
            lines = console.render_lines(self.body, inner, pad=False)
            dropped = len(lines) - keep
            if dropped > 0:
                kept = lines[-keep:]
                # The crop often lands on a paragraph break, which would spend
                # the first row of a hard-won budget on nothing.
                while kept and not any(seg.text.strip() for seg in kept[0]):
                    kept.pop(0)
                    dropped += 1
                segments: list[Segment] = []
                for last, line in loop_last(kept):
                    segments.extend(line)
                    if not last:
                        segments.append(Segment.line())
                body = Segments(segments)
                plural = "" if dropped == 1 else "s"
                title = f"Thinking (… {dropped} earlier line{plural})"

        yield Panel(body, title=title, title_align="left", border_style="dim")


@dataclass
class TextSegment:
    text: str = ""


@dataclass
class ThinkingSegment:
    thinking: str = ""
    is_open: bool = True


@dataclass
class ContentSegment:
    content: Content


DisplaySegment = Union[TextSegment, ThinkingSegment, ContentSegment]
