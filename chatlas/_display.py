import base64
import binascii
import io
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from html import escape
from typing import TYPE_CHECKING, Any, Optional, Union
from urllib.parse import urlparse
from uuid import uuid4

from rich.live import Live
from rich.logging import RichHandler

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions

from ._content import (
    PROVIDER_ANNOTATION_TYPES,
    TOOL_CSS,
    Content,
    ContentCitation,
    ContentImageInline,
    ContentImageRemote,
    ContentText,
    ContentThinking,
    ContentThinkingDelta,
    ContentToolRequestFetch,
    ContentToolRequestSearch,
    ContentToolResponseFetch,
    ContentToolResponseSearch,
    ContentToolResult,
    WebSource,
)
from ._live_render import LiveRender
from ._logging import logger
from ._typing_extensions import TypedDict
from ._utils import format_bytes


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
        self._last_web: Optional["WebActivitySegment"] = None

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

        if isinstance(content, PROVIDER_ANNOTATION_TYPES):
            self._append_web_activity(content)
            return

        self._close_web_activity()

        if isinstance(content, ContentText):
            content = content.text

        if isinstance(content, str):
            last = self._segments[-1] if self._segments else None
            if isinstance(last, TextSegment):
                last.text += content
            else:
                self._segments.append(TextSegment(text=content))
        elif isinstance(content, ContentImageInline):
            self._segments.append(
                ImageSegment(
                    content=content,
                    thumbnail=ImageThumbnail(
                        mime_type=content.image_content_type,
                        data=content.data,
                        max_lines=self._echo_options["image_max_lines"],
                    ),
                )
            )
        elif isinstance(content, ContentToolResult):
            images = tool_result_images(content)
            if images:
                shown = content.model_copy(
                    update={"value": replace_images(content.value)}
                )
                self._segments.append(ContentSegment(content=shown))
                for image in images:
                    self._append(image)
                return
            self._segments.append(ContentSegment(content=content))
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

    def _append_web_activity(self, content: Content) -> None:
        if isinstance(content, ContentCitation):
            # Citations arrive *after* the text they ground, so they attach to the
            # most recent episode -- open or closed -- without reopening it.
            # Opening a fresh segment here would split one search into two panels.
            segment = self._last_web
            if segment is None:
                segment = self._open_web_activity()
            segment.add_citation(content)
            return

        last = self._last_web
        segment = (
            last if last is not None and last.is_open else self._open_web_activity()
        )

        if isinstance(content, ContentToolRequestSearch):
            segment.add_query(content.query)
        elif isinstance(content, ContentToolResponseSearch):
            segment.add_sources(content.sources)
        elif isinstance(content, ContentToolRequestFetch):
            segment.add_fetch(content.url, None)
        elif isinstance(content, ContentToolResponseFetch):
            segment.add_fetch(content.url, content.status)

    def _open_web_activity(self) -> "WebActivitySegment":
        segment = WebActivitySegment()
        self._segments.append(segment)
        self._last_web = segment
        return segment

    def _close_web_activity(self) -> None:
        if self._last_web is not None:
            self._last_web.is_open = False

    def __enter__(self) -> "MarkdownDisplay":
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # A provider that appends grounding metadata (queries, sources,
        # citations) after the full answer text -- Google does this -- leaves
        # nothing to trigger `_close_web_activity` via `_append`. Close and
        # repaint here so the last frame isn't stuck showing "Searching…".
        if exc_type is None and self._last_web is not None and self._last_web.is_open:
            self._close_web_activity()
            self._render()
        self._segments = []
        self._last_web = None


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
            elif isinstance(segment, WebActivitySegment):
                out.append(
                    WebActivityPanel(
                        segment,
                        max_sources=self._echo_options["web_activity_max_sources"],
                    )
                )
            elif isinstance(segment, ImageSegment):
                out.append(segment.thumbnail)
            else:
                content = segment.content
                if isinstance(content, ContentToolResult):
                    out.append(
                        ToolResultBlock(
                            self._markdown(content.to_display_markdown()),
                            max_lines=self._echo_options["tool_result_max_lines"],
                        )
                    )
                else:
                    out.append(self._markdown(str(content)))
        return out

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
            elif isinstance(segment, WebActivitySegment):
                parts.append(web_activity_html(segment))
            elif isinstance(segment, ImageSegment):
                parts.append(self._content_markdown(segment.content))
            else:
                parts.append(self._content_markdown(segment.content))
        return "".join(parts)

    def _content_markdown(self, content: Content) -> str:
        if isinstance(content, ContentToolResult):
            # Already collapsed, and TOOL_CSS bounds its height once expanded.
            return f"\n\n{content.to_html()}\n\n"
        if isinstance(content, ContentImageRemote):
            return f"\n\n{remote_image_html(content.url)}\n\n"
        return f"\n\n{content}\n\n"

    def _init_display(self) -> str:
        try:
            from IPython.display import HTML, Markdown, display
        except ImportError:
            raise ImportError(
                "The IPython package is required for displaying content in a Jupyter notebook. "
                "Install it with `pip install ipython`."
            )

        # `none` (the CSS keyword) rather than omitting the property, which
        # would fall through to the 400px fallback baked into TOOL_CSS.
        max_height = self._echo_options["tool_result_max_height"] or "none"
        wrapper_style = f"--chatlas-tool-result-max-height: {max_height}"

        if self._css_styles:
            id_ = uuid4().hex
            css = "".join(f"{k}: {v}; " for k, v in self._css_styles.items())
            # A compound selector (no combinator): id and class are on the same
            # element, the wrapper div displayed just below.
            display(
                HTML(
                    f"<style>{TOOL_CSS}\n{WEB_CSS}\n"
                    f"#{id_}.chatlas-markdown {{ {css} }}</style>"
                )
            )
            display(
                HTML(
                    f"<div id='{id_}' class='chatlas-markdown' style='{wrapper_style}'>"
                )
            )
        else:
            display(HTML(f"<style>{TOOL_CSS}\n{WEB_CSS}</style>"))
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
    web_activity_max_sources: Optional[int]
    image_max_lines: Optional[int]


DEFAULT_TOOL_RESULT_MAX_LINES = 20
DEFAULT_TOOL_RESULT_MAX_HEIGHT = "400px"
DEFAULT_IMAGE_MAX_LINES = 16
# Lower than the tool-result cap on purpose: reasoning is an aside, so it should
# take up less room than the answer it precedes.
DEFAULT_THINKING_MAX_LINES = 10
# A source list is a pointer, not the content -- four is enough to see where the
# answer came from. A notebook shows all of them.
DEFAULT_WEB_ACTIVITY_MAX_SOURCES = 4


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


def web_activity_safe_url(url: Optional[str]) -> Optional[str]:
    """
    Only `http(s)` URLs become an `href`.

    Source URLs are model-influenced and land in HTML, so anything else (e.g.
    `javascript:`) renders as plain text instead.
    """
    if not url:
        return None
    return url if urlparse(url).scheme in ("http", "https") else None


def remote_image_html(url: str) -> str:
    """A remote image linked to its source, or plain text for an unsafe URL."""
    safe = web_activity_safe_url(url)
    if safe is None:
        return escape(url)
    escaped = escape(safe, quote=True)
    return (
        f'<a href="{escaped}" target="_blank" rel="noopener noreferrer">'
        f'<img src="{escaped}" alt="" /></a>'
    )


def web_activity_html(segment: "WebActivitySegment") -> str:
    """
    Wrap web activity in a collapsed `<details>` block.

    Collapsed in *every* state, unlike `thinking_html`: reasoning is something you
    watch happen, while provenance is something you consult afterwards. This also
    matches shinychat's `WebActivity`, which is collapsed by default.

    The blank lines matter for the same reason they do in `thinking_html`: they end
    the opening HTML block so the body stays markdown-renderable.
    """

    def link(label: str, url: Optional[str]) -> str:
        safe = web_activity_safe_url(url)
        if safe is None:
            return escape(label)
        return (
            f'<a href="{escape(safe, quote=True)}" target="_blank" '
            f'rel="noopener noreferrer">{escape(label)}</a>'
        )

    parts: list[str] = []

    for query in segment.queries:
        parts.append(
            f"<div class='chatlas-web-q'><span class='chatlas-web-ico'>🔍</span>"
            f"<em>{escape(query)}</em></div>"
        )

    for url, status in segment.fetches:
        mark = "✗" if status == "error" else "✓" if status == "success" else "…"
        cls = "err" if status == "error" else "ok" if status == "success" else "pending"
        fetch_label = web_safe_fetch_label(url)
        parts.append(
            f"<div class='chatlas-web-q'><span class='chatlas-web-ico'>🌐</span>"
            f"<span class='chatlas-web-lbl'>Read</span> "
            f"{link(fetch_label, url)}"
            f" <span class='chatlas-web-st {cls}'>{mark}</span></div>"
        )

    if segment.sources:
        items: list[str] = []
        for row in segment.sources:
            cited = " cited" if row.cited else ""
            quote = ""
            if row.cited and row.quote:
                quote = f"<blockquote>{escape(row.quote)}</blockquote>"
            domain = web_domain(web_activity_safe_url(row.url))
            label = web_safe_source_label(row)
            items.append(
                f"<li class='chatlas-web-src{cited}'>"
                f"{link(label, row.url)}"
                f"<span class='chatlas-web-dom'>{escape(domain)}</span>"
                f"{quote}</li>"
            )
        parts.append(f"<ol class='chatlas-web-srcs'>{''.join(items)}</ol>")

    detail = segment.detail()
    summary = escape(segment.header())
    if detail:
        summary += f" <span class='chatlas-web-detail'>({escape(detail)})</span>"

    return (
        f"\n\n<div class='chatlas-web-activity'><details>"
        f"<summary>{summary}</summary>"
        f"<div class='chatlas-web-body'>{''.join(parts)}</div>"
        f"</details></div>\n\n"
    )


WEB_CSS = """
.chatlas-web-activity details{border:1px solid rgba(128,128,128,.35);border-radius:6px;
  padding:.25rem .6rem;margin:.5rem 0;font-size:.92em}
.chatlas-web-activity summary{cursor:pointer;font-weight:600;opacity:.85}
.chatlas-web-detail{font-weight:400;opacity:.6}
.chatlas-web-body{max-height:var(--chatlas-tool-result-max-height,400px);
  overflow-y:auto;margin:.35rem 0 .3rem}
.chatlas-web-q{margin:.15rem 0}
.chatlas-web-ico{opacity:.55;margin-right:.35rem}
.chatlas-web-lbl{font-size:.85em;text-transform:uppercase;letter-spacing:.05em;
  opacity:.55}
.chatlas-web-st.ok{color:#1a7f37}
.chatlas-web-st.err{color:#b42318}
.chatlas-web-st.pending{opacity:.55}
ol.chatlas-web-srcs{margin:.3rem 0 .1rem 1.3rem;padding:0}
li.chatlas-web-src{margin:.15rem 0}
li.chatlas-web-src.cited > a{font-weight:600}
li.chatlas-web-src.cited::marker{content:"* "}
.chatlas-web-dom{opacity:.5;font-size:.85em;margin-left:.4rem}
.chatlas-web-activity blockquote{margin:.2rem 0;padding-left:.5rem;
  border-left:2px solid rgba(128,128,128,.4);font-size:.9em;opacity:.75}
"""


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
                for i, line in enumerate(kept):
                    if i > 0:
                        segments.append(Segment.line())
                    segments.extend(line)
                body = Segments(segments)
                plural = "" if dropped == 1 else "s"
                title = f"Thinking (… {dropped} earlier line{plural})"

        yield Panel(body, title=title, title_align="left", border_style="dim")


class ToolResultBlock:
    """A tool result capped by rendered terminal lines."""

    def __init__(self, body: Any, max_lines: Optional[int]):
        self.body = body
        self.max_lines = max_lines

    def __rich_console__(self, console: "Console", options: "ConsoleOptions"):
        from rich.segment import Segment, Segments
        from rich.text import Text

        if self.max_lines is None:
            yield self.body
            return

        keep = max(1, self.max_lines)
        rendered = console.render(self.body, options)
        lines = Segment.split_and_crop_lines(
            rendered, options.max_width, pad=False, include_new_lines=False
        )
        kept: list[list[Segment]] = []
        dropped = 0
        for line in lines:
            if len(kept) < keep:
                kept.append(line)
            else:
                dropped += 1

        segments: list[Segment] = []
        for index, line in enumerate(kept):
            if index:
                segments.append(Segment.line())
            segments.extend(line)

        if dropped <= 0:
            yield Segments(segments)
            return

        plural = "" if dropped == 1 else "s"
        segments.append(Segment.line())
        yield Segments(segments)
        yield Text(
            f"… {dropped} more line{plural}",
            style="dim italic",
        )


class ImageThumbnail:
    """A terminal thumbnail rendered as pairs of RGB pixels in half-block cells."""

    def __init__(self, mime_type: str, data: str, max_lines: Optional[int]):
        self.mime_type = mime_type
        self.data = data
        self.max_lines = max_lines
        self._image: Any | None = None
        self._decode_failed = False
        self._segments_by_width: dict[int, Any] = {}

    def __rich_console__(self, console: "Console", options: "ConsoleOptions"):
        from rich.text import Text

        if console.no_color or console.color_system not in ("truecolor", "256"):
            yield Text(image_label(self.mime_type, self.data), style="dim")
            return

        image = self._decode()
        if image is None:
            yield Text(image_label(self.mime_type, self.data), style="dim")
            return

        yield self._segments(image, options.max_width)
        yield Text(
            image_label(self.mime_type, self.data, image.size),
            style="dim",
        )

    def _decode(self) -> Any | None:
        if self._decode_failed:
            return None
        if self._image is not None:
            return self._image

        try:
            from PIL import Image

            data = "".join(self.data.split())
            decoded = base64.b64decode(data + "=" * (-len(data) % 4), validate=True)
            with Image.open(io.BytesIO(decoded)) as image:
                self._image = image.convert("RGBA")
        except (ImportError, OSError, ValueError, binascii.Error):
            self._decode_failed = True
            return None

        return self._image

    def _segments(self, image: Any, max_width: int) -> Any:
        from rich.color import Color
        from rich.segment import Segment, Segments
        from rich.style import Style

        width = max(max_width, 1)
        cached = self._segments_by_width.get(width)
        if cached is not None:
            return cached

        max_height = image.height
        if self.max_lines is not None:
            max_height = max(max(self.max_lines, 1) * 2, 1)

        scale = min(width / image.width, max_height / image.height, 1)
        size = (
            max(round(image.width * scale), 1),
            max(round(image.height * scale), 1),
        )
        thumbnail = image.resize(size)
        pixels = thumbnail.load()

        segments: list[Any] = []
        for y in range(0, thumbnail.height, 2):
            for x in range(thumbnail.width):
                top = pixels[x, y]
                bottom = pixels[x, min(y + 1, thumbnail.height - 1)]
                if top[3] == 0 and bottom[3] == 0:
                    segments.append(Segment(" "))
                    continue
                segments.append(
                    Segment(
                        "▀",
                        Style(
                            color=Color.from_rgb(*composite_rgba(top)),
                            bgcolor=Color.from_rgb(*composite_rgba(bottom)),
                        ),
                    )
                )
            segments.append(Segment.line())

        rendered = Segments(segments)
        self._segments_by_width[width] = rendered
        return rendered


class WebActivityPanel:
    """
    A panel of web activity, hard-capped at `max_sources` sources.

    Opposite direction from `ThinkingPanel`: reasoning streams toward a conclusion
    so its tail is what matters, but a source list is a reference whose leading
    entries are the ones the model leaned on -- except a cited row, which is
    evidence the model actually used and so claims a slot ahead of uncited rows
    regardless of its position. See `capped_sources` for the exact selection
    policy. Query and fetch rows don't count against the cap -- they're few, and
    the cap exists to bound the part that runs long.
    """

    def __init__(self, segment: "WebActivitySegment", max_sources: Optional[int]):
        self.segment = segment
        self.max_sources = max_sources

    def __rich_console__(self, console: "Console", options: "ConsoleOptions"):
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        segment = self.segment
        rows: list[Any] = []

        for query in segment.queries:
            line = Text("🔍 ", style="dim")
            line.append(query, style="italic")
            rows.append(line)

        for url, status in segment.fetches:
            mark = "✗" if status == "error" else "✓" if status == "success" else "…"
            safe_url = web_activity_safe_url(url)
            line = Text("🌐 Read ", style="dim")
            line.append(
                web_safe_fetch_label(url), style=f"link {safe_url}" if safe_url else ""
            )
            line.append(f" {mark}", style="dim")
            rows.append(line)

        shown, dropped = capped_sources(segment.sources, self.max_sources)

        if shown:
            # Title left, domain right. The domain is what makes Gemini's opaque
            # grounding-redirect URLs legible.
            table = Table.grid(padding=(0, 1), expand=True)
            table.add_column(width=1, no_wrap=True)
            table.add_column(ratio=1, overflow="ellipsis", no_wrap=True)
            table.add_column(justify="right", no_wrap=True)
            for row in shown:
                safe_url = web_activity_safe_url(row.url)
                label = web_safe_source_label(row)
                # A styled `link` becomes an OSC-8 hyperlink on a real terminal, so
                # the URL stays one click away without spending width on it. Only a
                # safe URL gets one -- an unsafe scheme (e.g. `javascript:`) renders
                # as plain text instead, same as the notebook.
                table.add_row(
                    Text(
                        "*" if row.cited else " ", style="yellow" if row.cited else ""
                    ),
                    Text(label, style=f"link {safe_url}" if safe_url else ""),
                    Text(web_domain(safe_url), style="dim"),
                )
            rows.append(table)

        if dropped:
            rows.append(Text(f"  … {dropped} more", style="dim italic"))

        detail = segment.detail()
        title = segment.header() + (f"  ({detail})" if detail else "")

        yield Panel(
            Group(*rows) if rows else Text(""),
            title=title,
            title_align="left",
            border_style="dim",
        )


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


@dataclass
class ImageSegment:
    content: ContentImageInline
    thumbnail: ImageThumbnail


@dataclass
class WebActivityRow:
    """
    One source row.

    Display-local rather than a `WebSource` because `cited` and `quote` are
    presentational state that must not end up serialized onto a turn.
    """

    url: Optional[str] = None
    title: Optional[str] = None
    cited: bool = False
    quote: Optional[str] = None


@dataclass
class WebActivitySegment:
    """
    One web-activity episode: every query, fetch, result, and citation from one
    stretch of provider-executed web work.

    Grouped rather than one segment per content object, so a citation can mark the
    source row it points at instead of becoming a row of its own.
    """

    queries: list[str] = field(default_factory=list)
    fetches: list[tuple[str, Optional[str]]] = field(default_factory=list)
    sources: list[WebActivityRow] = field(default_factory=list)
    is_open: bool = True

    def add_query(self, query: str) -> None:
        self.queries.append(query)

    def add_fetch(self, url: str, status: Optional[str]) -> None:
        # A fetch emits a request and then a result for the same URL. That's one
        # row; prefer whichever carries the status.
        for i, (existing, existing_status) in enumerate(self.fetches):
            if existing == url:
                if status and not existing_status:
                    self.fetches[i] = (url, status)
                return
        self.fetches.append((url, status))

    def add_sources(self, sources: Sequence[WebSource]) -> None:
        for source in sources:
            if not any(row.url == source.url for row in self.sources):
                self.sources.append(WebActivityRow(url=source.url, title=source.title))

    def add_citation(self, citation: ContentCitation) -> None:
        """Mark the source this citation points at, rather than adding a row."""
        source = citation.source
        url = source.url if isinstance(source, WebSource) else None
        title = (source.title if isinstance(source, WebSource) else None) or (
            citation.grounded_span
        )

        for row in self.sources:
            # Match on URL when there is one. A source-less citation (Google can
            # ground a span with no resolvable URL) matches on its label instead,
            # so repeats don't accumulate blank rows.
            if (url and row.url == url) or (
                not url and not row.url and row.title == title
            ):
                row.cited = True
                row.quote = row.quote or citation.cited_quote
                return

        self.sources.append(
            WebActivityRow(url=url, title=title, cited=True, quote=citation.cited_quote)
        )

    @property
    def is_fetch_only(self) -> bool:
        return bool(self.fetches) and not self.queries

    def header(self) -> str:
        if self.is_open:
            return "Reading the web…" if self.is_fetch_only else "Searching the web…"
        return "Read the web" if self.is_fetch_only else "Searched the web"

    def detail(self) -> str:
        bits: list[str] = []
        if self.sources:
            n = len(self.sources)
            bits.append(f"{n} result{'' if n == 1 else 's'}")
        n_cited = sum(1 for row in self.sources if row.cited)
        if n_cited:
            # The `*` matches the marker on each cited row, so the header doubles
            # as the legend that explains it.
            bits.append(f"* {n_cited} cited")
        return " · ".join(bits)


def web_domain(url: Optional[str]) -> str:
    "Best-effort hostname, matching shinychat's `domain_from_url`."
    if not url:
        return ""
    return urlparse(url).hostname or url


def web_safe_source_label(row: WebActivityRow) -> str:
    """
    What to show for a source. Gemini's URLs are opaque, so the title comes first.

    Falls back to the domain, then a neutral placeholder -- never all the way
    to the raw URL. An unsafe scheme (e.g. `javascript:`) has no `href`, so its
    raw text must not surface as the visible label either, which is why the
    domain here comes from the *safe* URL rather than the raw one.
    """
    return row.title or web_domain(web_activity_safe_url(row.url)) or "(source)"


def web_safe_fetch_label(url: str) -> str:
    "Display text for a fetched URL; an unsafe scheme falls back to a placeholder."
    return web_display_url(url) if web_activity_safe_url(url) else "(url)"


def capped_sources(
    sources: list[WebActivityRow], max_sources: Optional[int]
) -> tuple[list[WebActivityRow], int]:
    """
    The rows to show under a source cap, and how many were dropped.

    A hard bound: `shown` never has more than `max_sources` rows. Within that
    bound, a cited row -- evidence the model actually used -- wins a slot ahead
    of an uncited row regardless of position, earliest-first among cited rows;
    uncited rows then fill whatever budget remains, earliest-first. If cited
    rows alone exceed `max_sources`, only the earliest `max_sources` of them are
    shown and every uncited row is dropped. The result keeps the original
    relative order among whatever it keeps.
    """
    if max_sources is None or len(sources) <= max_sources:
        return sources, 0

    cited = [row for row in sources if row.cited]
    non_cited = [row for row in sources if not row.cited]

    cited_shown = cited[:max_sources]
    budget = max(max_sources - len(cited_shown), 0)
    non_cited_shown = non_cited[:budget]

    keep = {id(row) for row in cited_shown} | {id(row) for row in non_cited_shown}
    shown = [row for row in sources if id(row) in keep]
    return shown, len(sources) - len(shown)


def web_display_url(url: str) -> str:
    "A fetched URL without its scheme, which is noise in a narrow panel."
    return re.sub(r"^https?://(www\.)?", "", url or "")


def image_label(
    mime_type: str, data: str, size: Optional[tuple[int, int]] = None
) -> str:
    """
    A one-line stand-in for an image: `🖼 image/png · 800×600 · 219 KB`.

    Used both as the caption under a thumbnail and as the whole rendering when a
    thumbnail isn't possible. `size` is omitted when the image wasn't decoded.
    """
    bits = [mime_type]
    if size is not None:
        bits.append(f"{size[0]}×{size[1]}")
    bits.append(format_bytes(base64_nbytes(data)))
    return "🖼 " + " · ".join(bits)


def tool_result_images(
    content: ContentToolResult,
) -> list[ContentImageInline | ContentImageRemote]:
    """Images carried by a tool result's value."""
    return inline_images(content.value)


def inline_images(value: object) -> list[ContentImageInline | ContentImageRemote]:
    if isinstance(value, (ContentImageInline, ContentImageRemote)):
        return [value]
    if isinstance(value, (list, tuple)):
        images: list[ContentImageInline | ContentImageRemote] = []
        for item in value:
            images.extend(inline_images(item))
        return images
    if isinstance(value, Mapping):
        images = []
        for item in value.values():
            images.extend(inline_images(item))
        return images
    return []


def replace_images(value: object) -> object:
    """Swap inline images for labels in the display-only copy of a result."""
    if isinstance(value, ContentImageInline):
        return image_label(value.image_content_type, value.data)
    if isinstance(value, ContentImageRemote):
        return f"image: {value.url}"
    if isinstance(value, list):
        return [replace_images(item) for item in value]
    if isinstance(value, tuple):
        return tuple(replace_images(item) for item in value)
    if isinstance(value, Mapping):
        return {key: replace_images(item) for key, item in value.items()}
    return value


def base64_nbytes(data: str) -> int:
    """
    Decoded length of base64 `data`, computed from the string.

    An echoed image can be megabytes; decoding it just to report its size would
    double the work the thumbnail already does.
    """
    s = "".join(char for char in data if char not in " \t\n\r\v\f")
    groups, remainder = divmod(len(s), 4)
    nbytes = groups * 3 + (0, 0, 1, 2)[remainder]
    return max(0, nbytes - s.count("="))


def composite_rgba(pixel: tuple[int, int, int, int]) -> tuple[int, int, int]:
    red, green, blue, alpha = pixel
    return (
        composite_channel(red, alpha),
        composite_channel(green, alpha),
        composite_channel(blue, alpha),
    )


def composite_channel(channel: int, alpha: int) -> int:
    return (channel * alpha + 128 * (255 - alpha) + 127) // 255


DisplaySegment = Union[
    TextSegment,
    ThinkingSegment,
    WebActivitySegment,
    ContentSegment,
    ImageSegment,
]
