"""
Tests for `echo=` output, as rendered by the rich-based live console display.

`LiveMarkdownDisplay` writes to a `rich.console.Console` built from
`Chat.set_echo_options(rich_console=)`, so pointing that at a `StringIO` lets us
assert on what a user would actually see. On a non-terminal console, `rich.Live`
skips incremental refreshes and prints only the final frame, which keeps these
assertions deterministic and free of ANSI escapes.
"""

import base64
import logging
import re
import sys
from collections.abc import Sequence
from io import BytesIO, StringIO
from typing import Any, Callable, Literal, Optional, overload

import pytest
from chatlas import Chat
from chatlas._content import (
    Content,
    ContentCitation,
    ContentImageInline,
    ContentImageRemote,
    ContentText,
    ContentThinking,
    ContentThinkingDelta,
    ContentToolRequest,
    ContentToolRequestFetch,
    ContentToolRequestSearch,
    ContentToolResponseFetch,
    ContentToolResponseSearch,
    ContentToolResult,
    WebSource,
)
from chatlas._display import (
    DEFAULT_IMAGE_MAX_LINES,
    ImageThumbnail,
    IPyMarkdownDisplay,
    LiveMarkdownDisplay,
    ToolResultBlock,
    WebActivityRow,
    WebActivitySegment,
    base64_nbytes,
    capped_sources,
    image_label,
    remote_image_html,
    replace_images,
    tool_result_images,
    web_domain,
)
from chatlas._live_render import LiveRender
from chatlas._logging import _rich_handler
from chatlas._provider import AnyTypeDict, Provider
from chatlas._turn import AssistantTurn
from chatlas._utils import MISSING, MISSING_TYPE, format_bytes
from PIL import Image
from rich.console import Console
from rich.text import Text


def test_echo_none_writes_nothing():
    chat = make_chat([[text("Hello there")]])
    output = capture_echo(chat)
    chat.chat("hi", echo="none")
    assert output() == ""


def test_echo_text_renders_assistant_text():
    chat = make_chat([[text("Hello **there**")]])
    output = capture_echo(chat)
    chat.chat("hi", echo="text")
    assert output() == "Hello there"


def test_echo_text_omits_user_turn():
    chat = make_chat([[text("Hello there")]])
    output = capture_echo(chat)
    chat.chat("what is 2+2?", echo="text")
    assert "what is 2+2?" not in output()
    assert "User turn" not in output()


def test_echo_all_includes_user_turn_and_headers():
    chat = make_chat([[text("Hello there")]])
    output = capture_echo(chat)
    chat.chat("what is 2+2?", echo="all")
    res = output()
    assert "👤 User turn:" in res
    assert "what is 2+2?" in res
    assert "🤖 Assistant turn:" in res
    assert "Hello there" in res


def test_echo_renders_markdown(snapshot):
    chunks = [
        text("# Heading\n\n"),
        text("Some **bold** and `code`, then a list:\n\n"),
        text("- one\n- two\n\n"),
        text("```python\nx = 1\n```\n"),
    ]
    chat = make_chat([chunks])
    output = capture_echo(chat)
    chat.chat("hi", echo="output")
    assert snapshot == output()


def test_echo_output_includes_tool_request_and_result():
    chat = make_chat([[tool_request()], [text("It is 30°F.")]])
    chat.register_tool(get_temperature)
    output = capture_echo(chat)
    chat.chat("temp in Duluth?", echo="output")
    res = output()
    assert "🔧 tool request" in res
    assert 'get_temperature(city="Duluth")' in res
    assert "✅ tool result" in res
    assert "30" in res
    assert "It is 30°F." in res


def test_echo_text_omits_tool_content():
    chat = make_chat([[tool_request()], [text("It is 30°F.")]])
    chat.register_tool(get_temperature)
    output = capture_echo(chat)
    chat.chat("temp in Duluth?", echo="text")
    res = output()
    assert "tool request" not in res
    assert "tool result" not in res
    assert "It is 30°F." in res


def test_echo_non_streaming():
    chat = make_chat([[text("Hello **there**")]])
    output = capture_echo(chat)
    chat.chat("hi", echo="output", stream=False)
    assert output() == "Hello there"


def test_echo_stream_generator_renders_incrementally():
    chat = make_chat([[text("Hello "), text("there")]])
    output = capture_echo(chat)
    for _ in chat.stream("hi", echo="output"):
        pass
    assert output() == "Hello there"


@pytest.mark.asyncio
async def test_echo_all_async_matches_sync():
    chunks = [text("Hello **there**")]
    sync_chat = make_chat([list(chunks)])
    sync_output = capture_echo(sync_chat)
    sync_chat.chat("what is 2+2?", echo="all")

    async_chat = make_chat([list(chunks)])
    async_output = capture_echo(async_chat)
    await async_chat.chat_async("what is 2+2?", echo="all")

    assert async_output() == sync_output()
    assert "👤 User turn:" in async_output()


@pytest.mark.asyncio
async def test_echo_output_async_includes_tool_content():
    chat = make_chat([[tool_request()], [text("It is 30°F.")]])
    chat.register_tool(get_temperature)
    output = capture_echo(chat)
    await chat.chat_async("temp in Duluth?", echo="output")
    res = output()
    assert "🔧 tool request" in res
    assert "✅ tool result" in res
    assert "It is 30°F." in res


@pytest.mark.asyncio
async def test_echo_none_async_writes_nothing():
    chat = make_chat([[text("Hello there")]])
    output = capture_echo(chat)
    await chat.chat_async("hi", echo="none")
    assert output() == ""


def test_echo_renders_thinking(snapshot):
    chat = make_chat([thinking_chunks()])
    output = capture_echo(chat)
    chat.chat("what is 2+2?", echo="all")
    res = output()
    assert "2+2 is 4." in res
    assert "Thinking" in res
    assert snapshot == res


def test_echo_renders_thinking_in_a_panel():
    """Reasoning is boxed, so it reads as an aside rather than as the answer."""
    chat = make_chat([thinking_chunks()])
    output = capture_echo(chat)
    chat.chat("what is 2+2?", echo="output")
    res = output()
    # Panel corners, with the title on the top border.
    assert "╭─ Thinking" in res
    assert "╰" in res
    # The answer sits outside the panel.
    assert "The answer is 4." in res.rsplit("╰", 1)[-1]


def test_echo_text_omits_thinking():
    chat = make_chat([thinking_chunks()])
    output = capture_echo(chat)
    chat.chat("what is 2+2?", echo="text")
    res = output()
    assert "2+2 is 4." not in res
    assert "Thinking" not in res
    assert res == "The answer is 4."


def test_echo_all_renders_thinking_exactly_once():
    """
    The completed turn also carries a `ContentThinking`, so reasoning is
    reachable twice: once streamed, once via `emit_other_contents`.
    """
    chat = make_chat([thinking_chunks()])
    output = capture_echo(chat)
    chat.chat("what is 2+2?", echo="all")
    res = output()
    assert res.count("2+2 is 4.") == 1
    assert res.count("Thinking") == 1
    # ...and the turn really does hold it, i.e. the assertion above isn't vacuous.
    turn = chat.get_last_turn()
    assert turn is not None
    assert any(isinstance(c, ContentThinking) for c in turn.contents)


@pytest.mark.parametrize("echo", ["output", "all"])
def test_echo_non_streaming_renders_thinking(echo: Literal["output", "all"]):
    chat = make_chat([thinking_chunks()])
    output = capture_echo(chat)
    chat.chat("what is 2+2?", echo=echo, stream=False)
    res = output()
    assert res.count("2+2 is 4.") == 1
    assert "Thinking" in res


def test_echo_text_non_streaming_omits_thinking():
    chat = make_chat([thinking_chunks()])
    output = capture_echo(chat)
    chat.chat("what is 2+2?", echo="text", stream=False)
    assert output() == "The answer is 4."


def test_echo_thinking_then_tool_request():
    """A tool request has to close the thinking block, not land inside it."""
    chunks = [
        ContentThinkingDelta(thinking="I need the temperature."),
        tool_request(),
    ]
    chat = make_chat([chunks, [text("It is 30°F.")]])
    chat.register_tool(get_temperature)
    output = capture_echo(chat)
    chat.chat("temp in Duluth?", echo="output")
    res = output()
    assert "I need the temperature." in res
    # The request renders after the panel closes.
    assert "🔧 tool request" in res.rsplit("╰", 1)[-1]


def test_echo_caps_long_reasoning_keeping_the_NEWEST_lines():
    """
    Reasoning is cropped from the top, unlike a tool result. The newest text is
    what's still streaming, so cropping the other way would pin the panel to
    lines the reader already finished and look like a hang.
    """
    chat = make_chat([long_reasoning_chunks()])
    output = capture_echo(chat, width=70, thinking_max_lines=6)
    chat.chat("q", echo="output")
    res = output()

    assert "step 39" in res  # newest reasoning kept
    assert "step 00" not in res  # oldest dropped
    assert "earlier line" in res
    assert "The answer is 4." in res


def test_reasoning_cap_counts_WRAPPED_lines():
    """
    Reasoning arrives as one long paragraph with no newlines, so a
    `splitlines()`-style cap (what tool results use) would do nothing at all
    here. The panel body must be bounded by *rendered* line count.
    """
    chunks = long_reasoning_chunks()
    assert "\n" not in "".join(
        c.thinking for c in chunks if isinstance(c, ContentThinkingDelta)
    ), "fixture must be newline-free for this test to mean anything"

    chat = make_chat([chunks])
    output = capture_echo(chat, width=70, thinking_max_lines=6)
    chat.chat("q", echo="output")
    res = output()

    # 6 body lines + 2 border lines is the whole panel.
    panel = [ln for ln in res.splitlines() if ln.startswith(("╭", "│", "╰"))]
    assert len(panel) == 8, panel


def test_reasoning_cap_reports_how_many_lines_were_dropped():
    chat = make_chat([long_reasoning_chunks()])
    output = capture_echo(chat, width=70, thinking_max_lines=6)
    chat.chat("q", echo="output")

    match = re.search(r"… (\d+) earlier lines?", output())
    assert match is not None
    dropped = int(match.group(1))

    # Re-render uncapped to learn the true height, then check the count adds up.
    uncapped_chat = make_chat([long_reasoning_chunks()])
    uncapped = capture_echo(uncapped_chat, width=70, thinking_max_lines=None)
    uncapped_chat.chat("q", echo="output")
    total_body = len([ln for ln in uncapped().splitlines() if ln.startswith("│")])
    assert dropped == total_body - 6


def test_reasoning_cap_can_be_disabled_with_None():
    chat = make_chat([long_reasoning_chunks()])
    output = capture_echo(chat, width=70, thinking_max_lines=None)
    chat.chat("q", echo="output")
    res = output()
    assert "step 00" in res
    assert "step 39" in res
    assert "earlier line" not in res


def test_short_reasoning_is_left_alone():
    chat = make_chat([thinking_chunks()])
    output = capture_echo(chat, width=70, thinking_max_lines=6)
    chat.chat("what is 2+2?", echo="output")
    res = output()
    assert "2+2 is 4." in res
    assert "earlier line" not in res


def test_reasoning_cap_defaults_to_ten_lines():
    chat = make_chat([long_reasoning_chunks()])
    output = capture_echo(chat, width=70)
    chat.chat("q", echo="output")
    body = [ln for ln in output().splitlines() if ln.startswith("│")]
    assert len(body) == 10, body


def test_reasoning_cap_does_not_open_with_a_blank_row():
    """
    Real reasoning has paragraph breaks, so the crop often lands on one. Spending
    the first row of the budget on a blank line is a waste, and it reads as a
    rendering wart.
    """
    paragraphs = "\n\n".join(
        f"Paragraph {i}: reasoning about possibility {i} at some length here."
        for i in range(20)
    )
    chunks: list[Content] = [
        ContentThinkingDelta(thinking=paragraphs[i : i + 60])
        for i in range(0, len(paragraphs), 60)
    ]
    chunks.append(text("Done."))

    chat = make_chat([chunks])
    output = capture_echo(chat, width=70, thinking_max_lines=8)
    chat.chat("q", echo="output")

    body = [ln for ln in output().splitlines() if ln.startswith("│")]
    assert body, "expected a panel"
    assert body[0].strip("│ ") != "", f"panel opens with a blank row: {body[0]!r}"
    assert len(body) <= 8


def test_reasoning_cap_counts_stripped_blanks_as_dropped():
    """Whatever the panel drops -- cropped or blank -- the title has to own it."""
    paragraphs = "\n\n".join(f"Paragraph {i} of reasoning." for i in range(20))
    chunks: list[Content] = [
        ContentThinkingDelta(thinking=paragraphs[i : i + 60])
        for i in range(0, len(paragraphs), 60)
    ]
    chunks.append(text("Done."))

    def body_lines(cap: "int | None") -> tuple[int, str]:
        chat = make_chat([list(chunks)])
        out = capture_echo(chat, width=70, thinking_max_lines=cap)
        chat.chat("q", echo="output")
        res = out()
        return len([ln for ln in res.splitlines() if ln.startswith("│")]), res

    kept, capped = body_lines(8)
    total, _ = body_lines(None)

    match = re.search(r"… (\d+) earlier lines?", capped)
    assert match is not None
    assert int(match.group(1)) == total - kept


def test_reasoning_cap_of_zero_does_not_silently_keep_everything():
    """`lines[-0:]` is the whole list, so a cap of 0 has to be clamped."""
    chat = make_chat([long_reasoning_chunks()])
    output = capture_echo(chat, width=70, thinking_max_lines=0)
    chat.chat("q", echo="output")
    res = output()
    assert "step 00" not in res
    body = [ln for ln in res.splitlines() if ln.startswith("│")]
    assert len(body) == 1, body


def test_tool_result_truncation_can_be_disabled_with_None():
    chat = make_chat([[tool_request(name="big_tool", arguments={})], [text("Done.")]])
    chat.register_tool(big_tool)
    output = capture_echo(chat, width=80, tool_result_max_lines=None)
    chat.chat("go", echo="output")
    res = output()
    assert "'row-29'" in res
    assert "more line" not in res


def test_echo_truncates_a_long_tool_result():
    chat = make_chat([[tool_request(name="big_tool", arguments={})], [text("Done.")]])
    chat.register_tool(big_tool)
    output = capture_echo(chat, width=80, tool_result_max_lines=5)
    chat.chat("go", echo="output")
    res = output()

    assert "'row-0'" in res
    assert "'row-29'" not in res
    assert re.search(r"… \d+ more lines", res)

    # The full value is untouched on the turn -- only the display is bounded.
    turn = chat.get_last_turn(role="user")
    assert turn is not None
    result = turn.contents[0]
    assert isinstance(result, ContentToolResult)
    assert isinstance(result.value, list)
    assert len(result.value) == 30


def test_echo_all_truncates_a_long_tool_result_exactly_once():
    """
    `echo="all"` reaches a tool result twice: the user turn it's attached to is
    echoed as a whole, and `emit_other_contents` echoes its non-text contents.
    Both have to go through the display, or the bounded copy sits next to an
    unbounded one.
    """
    chat = make_chat([[tool_request(name="big_tool", arguments={})], [text("Done.")]])
    chat.register_tool(big_tool)
    output = capture_echo(chat, width=80, tool_result_max_lines=5)
    chat.chat("go", echo="all")
    res = output()

    assert res.count("✅ tool result") == 1
    assert res.count("more lines") == 1
    assert "'row-29'" not in res


def test_echo_all_still_shows_the_user_prompt():
    """`emit_user_contents` emits `turn.text` rather than `str(turn)` now."""
    chat = make_chat([[text("Hi")]])
    output = capture_echo(chat)
    chat.chat("what is 2+2?", echo="all")
    res = output()
    assert "👤 User turn:" in res
    assert "what is 2+2?" in res


def test_echo_does_not_truncate_a_short_tool_result():
    chat = make_chat([[tool_request()], [text("It is 30°F.")]])
    chat.register_tool(get_temperature)
    output = capture_echo(chat, tool_result_max_lines=5)
    chat.chat("temp in Duluth?", echo="output")
    res = output()
    assert "30" in res
    assert "more line" not in res


def test_tool_result_max_lines_defaults_to_twenty():
    chat = make_chat([[tool_request(name="big_tool", arguments={})], [text("Done.")]])
    chat.register_tool(big_tool)
    output = capture_echo(chat, width=80)
    chat.chat("go", echo="output")
    assert re.search(r"… \d+ more lines", output())


def test_tool_result_block_caps_wrapped_terminal_lines():
    rendered = render_tool_result_block(Text("word " * 20), max_lines=2, width=20)

    assert len(rendered.splitlines()) == 3
    assert rendered.splitlines()[-1] == "… 3 more lines"


def test_tool_result_block_keeps_the_head():
    rendered = render_tool_result_block(
        Text("first\nsecond\nthird"), max_lines=2, width=20
    )

    assert "first" in rendered
    assert "second" in rendered
    assert "third" not in rendered


@pytest.mark.parametrize(
    ("body", "expected_footer"),
    [
        ("first\nsecond", "… 1 more line"),
        ("first\nsecond\nthird", "… 2 more lines"),
    ],
)
def test_tool_result_block_footer_counts_dropped_lines(
    body: str, expected_footer: str
):
    rendered = render_tool_result_block(Text(body), max_lines=1, width=20)

    assert rendered.splitlines()[-1] == expected_footer


def test_tool_result_block_wraps_a_narrow_footer():
    rendered = render_tool_result_block(Text("first\nsecond"), max_lines=1, width=10)

    assert rendered.endswith("… 1 more\nline")


def test_tool_result_block_without_a_cap_renders_every_line():
    rendered = render_tool_result_block(Text("first\nsecond\nthird"), max_lines=None)

    assert rendered == "first\nsecond\nthird"


def test_tool_result_block_leaves_short_content_alone():
    rendered = render_tool_result_block(Text("first\nsecond"), max_lines=5)

    assert rendered == "first\nsecond"


def test_tool_result_block_does_not_materialize_all_rendered_lines(
    monkeypatch: pytest.MonkeyPatch,
):
    console = Console(width=20)
    block = ToolResultBlock(Text("first\nsecond"), max_lines=1)

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("render_lines() materializes all rendered lines")

    monkeypatch.setattr(console, "render_lines", fail)

    assert len(list(block.__rich_console__(console, console.options))) == 2


@pytest.mark.parametrize("max_lines", [0, -1])
def test_tool_result_block_nonpositive_cap_keeps_one_line(max_lines: int):
    rendered = render_tool_result_block(
        Text("first\nsecond\nthird"), max_lines=max_lines, width=20
    )

    assert rendered == "first\n… 2 more lines"


def test_echo_caps_a_single_line_tool_result_by_terminal_lines():
    def large_tool_result() -> str:
        return "x" * 4000

    chat = make_chat(
        [[tool_request(name="large_tool_result", arguments={})], [text("Done.")]]
    )
    chat.register_tool(large_tool_result)
    output = capture_echo(chat, width=80, tool_result_max_lines=5)
    chat.chat("go", echo="output")
    rendered = output()

    assert "x" * 50 in rendered
    assert "x" * 4000 not in rendered
    assert re.search(r"… \d+ more lines", rendered)


def test_echo_of_an_image_tool_result_is_bounded():
    image = inline_png((8, 6))

    def screenshot_tool() -> ContentImageInline:
        return image

    chat = make_chat(
        [[tool_request(name="screenshot_tool", arguments={})], [text("Done.")]]
    )
    chat.register_tool(screenshot_tool)
    output = capture_echo(chat, width=80, tool_result_max_lines=20, image_max_lines=16)
    chat.chat("take a screenshot", echo="output")
    rendered = output()

    assert image.data[:40] not in rendered
    assert "🖼 image/png" in rendered
    assert len(rendered.splitlines()) <= 20 + 16 + 4


def test_image_tool_result_shows_the_result_block_and_the_image():
    image = inline_png((8, 6))

    def screenshot_tool() -> ContentImageInline:
        return image

    chat = make_chat(
        [[tool_request(name="screenshot_tool", arguments={})], [text("Done.")]]
    )
    chat.register_tool(screenshot_tool)
    output = capture_echo(chat, width=80)
    chat.chat("go", echo="output")
    rendered = output()

    assert "✅ tool result" in rendered
    assert "🖼 image/png" in rendered
    assert "tool-content" not in rendered


def test_ipy_image_tool_result_renders_a_real_image(monkeypatch):
    image = inline_png((8, 6))

    def screenshot_tool() -> ContentImageInline:
        return image

    updates = capture_ipy(monkeypatch)
    chat = make_chat(
        [[tool_request(name="screenshot_tool", arguments={})], [text("Done.")]]
    )
    chat.register_tool(screenshot_tool)
    chat.chat("go", echo="output")
    final = updates[-1]

    assert "![](data:image/png;base64," in final
    assert "chatlas-tool-result" in final
    tool_result, _, _ = final.partition("![](data:image/png;base64,")
    assert image.data not in tool_result


def test_ipy_remote_image_tool_result_renders_a_real_image(monkeypatch):
    image = ContentImageRemote(url="https://example.com/image.png")

    def remote_image_tool() -> ContentImageRemote:
        return image

    updates = capture_ipy(monkeypatch)
    chat = make_chat(
        [[tool_request(name="remote_image_tool", arguments={})], [text("Done.")]]
    )
    chat.register_tool(remote_image_tool)
    chat.chat("go", echo="output")
    final = updates[-1]

    assert '<a href="https://example.com/image.png"' in final
    assert '<img src="https://example.com/image.png"' in final
    assert "chatlas-tool-result" in final


def test_ipy_image_in_mapping_tool_result_renders_outside_result(monkeypatch):
    image = inline_png((8, 6))

    def screenshot_tool() -> dict[str, object]:
        return {"image": image}

    updates = capture_ipy(monkeypatch)
    chat = make_chat(
        [[tool_request(name="screenshot_tool", arguments={})], [text("Done.")]]
    )
    chat.register_tool(screenshot_tool)
    chat.chat("go", echo="output")
    final = updates[-1]

    assert "![](data:image/png;base64," in final
    tool_result, _, _ = final.partition("![](data:image/png;base64,")
    assert image.data not in tool_result


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("javascript:alert(1)", "javascript:alert(1)"),
        (
            'https://example.com/image.png?caption="example"',
            "caption=&quot;example&quot;",
        ),
    ],
)
def test_remote_image_html_handles_unsafe_and_escaped_urls(url: str, expected: str):
    html = remote_image_html(url)

    assert expected in html
    if url.startswith("javascript:"):
        assert "<a " not in html
        assert "<img " not in html


def test_tool_result_html_has_one_disclosure():
    result = ContentToolResult(value="done", request=tool_request())

    assert result.to_html().count("<details") == 1


@pytest.mark.parametrize("nested", [False, True])
def test_image_inside_a_list_tool_result_is_split_out(nested: bool):
    image = inline_png((8, 6))
    value: list[object] = ["before", image]
    if nested:
        value = ["before", ["nested", image]]

    def screenshot_tool() -> list[object]:
        return value

    chat = make_chat(
        [[tool_request(name="screenshot_tool", arguments={})], [text("Done.")]]
    )
    chat.register_tool(screenshot_tool)
    output = capture_echo(chat, width=80)
    chat.chat("go", echo="output")
    rendered = output()

    assert image.data[:40] not in rendered
    assert "before" in rendered
    if nested:
        assert "nested" in rendered
    assert "🖼 image/png" in rendered


def test_tool_result_images_finds_images_at_any_supported_depth():
    image = inline_png((8, 6))
    request = tool_request()

    assert tool_result_images(ContentToolResult(value=image, request=request)) == [image]
    assert tool_result_images(
        ContentToolResult(value=["a", image], request=request)
    ) == [image]
    assert tool_result_images(
        ContentToolResult(value=[["a", image]], request=request)
    ) == [image]
    assert tool_result_images(
        ContentToolResult(value={"image": image}, request=request)
    ) == [image]
    assert tool_result_images(ContentToolResult(value="plain", request=request)) == []


def test_tool_result_str_is_never_truncated():
    """
    `str()` feeds `export()` and `Chat.__str__`, which are not display-bounded.
    """
    result = ContentToolResult(
        value="\n".join(f"line {i}" for i in range(50)),
        request=tool_request(),
    )
    assert "line 49" in str(result)
    assert "more line" not in str(result)


def test_current_display_echo_from_tool():
    def noisy_tool() -> str:
        "A tool that writes to the active display."
        assert chat.current_display is not None
        chat.current_display.echo("\n\nMy custom tool display!!!\n\n")
        return "done"

    chat = make_chat([[tool_request(name="noisy_tool", arguments={})], [text("Bye")]])
    chat.register_tool(noisy_tool)
    output = capture_echo(chat)
    chat.chat("go", echo="output")
    assert "My custom tool display!!!" in output()


def test_current_display_is_none_outside_of_echo():
    chat = make_chat([[text("Hello")]])
    assert chat.current_display is None
    chat.chat("hi", echo="output")
    assert chat.current_display is None


def test_set_echo_options_passes_rich_markdown_options():
    chat = make_chat([[text("See [the docs](https://example.com)")]])
    output = capture_echo(chat, rich_markdown={"hyperlinks": False})
    chat.chat("hi", echo="output")
    assert "https://example.com" in output()


def test_set_echo_options_replaces_all_options():
    """Each call replaces the full set: unspecified options revert to defaults."""
    chat = make_chat([[text("Hello")]])
    chat.set_echo_options(thinking_max_lines=None)
    chat.set_echo_options(rich_markdown={"hyperlinks": False})
    assert chat._echo_options["thinking_max_lines"] == 10


def test_logs_are_routed_into_the_live_console():
    """
    `LiveMarkdownDisplay.__enter__` repoints any `RichHandler` at the live
    console so log records aren't swallowed while the display is active.
    """
    # Only handlers on the root logger and on chatlas' own logger get
    # repointed, so attach to the latter rather than a child of it.
    log = logging.getLogger("chatlas")
    original_level = log.level
    log.setLevel(logging.INFO)
    handler = _rich_handler()
    log.addHandler(handler)

    chat = make_chat([[tool_request(name="logging_tool", arguments={})], [text("Hi")]])
    chat.register_tool(logging_tool(log))
    output = capture_echo(chat, width=100)
    try:
        chat.chat("go", echo="output")
    finally:
        log.removeHandler(handler)
        log.setLevel(original_level)

    assert "a log inside the live console" in output()


def test_notebook_environments_use_the_ipy_display(monkeypatch):
    chat = make_chat([[text("Hello")]])
    assert isinstance(chat._markdown_display("output")._display, LiveMarkdownDisplay)

    monkeypatch.setenv("QUARTO_PYTHON", "/usr/bin/python3")
    assert isinstance(chat._markdown_display("output")._display, IPyMarkdownDisplay)


def test_ipy_display_receives_markdown(monkeypatch):
    updates = capture_ipy(monkeypatch)
    chat = make_chat([[text("Hello "), text("there")]])
    chat.chat("hi", echo="output")
    assert updates[-1].endswith("Hello there")


def test_ipy_thinking_is_open_while_streaming_then_collapses(monkeypatch):
    updates = capture_ipy(monkeypatch)

    chat = make_chat([thinking_chunks()])
    chat.chat("what is 2+2?", echo="output")

    # While reasoning streams, the block is expanded so you can watch it arrive.
    mid = next(u for u in updates if "Let me think." in u)
    assert "<details open><summary>Thinking</summary>" in mid

    # Once it's done it folds away, so it costs almost no vertical space.
    assert "<details><summary>Thinking</summary>" in updates[-1]
    assert "<details open>" not in updates[-1]
    assert "2+2 is 4." in updates[-1]
    assert updates[-1].count("<summary>Thinking</summary>") == 1


def test_ipy_thinking_body_stays_markdown(monkeypatch):
    """
    Blank lines around the body end the opening HTML block, so a notebook still
    markdown-renders the reasoning rather than dumping it as raw text.
    """
    updates = capture_ipy(monkeypatch)
    chat = make_chat([[ContentThinkingDelta(thinking="- a\n- b"), text("Done.")]])
    chat.chat("hi", echo="output")
    assert "<summary>Thinking</summary>\n\n- a\n- b\n\n</details>" in updates[-1]


def test_ipy_text_only_response_has_no_thinking_block(monkeypatch):
    updates = capture_ipy(monkeypatch)
    chat = make_chat([[text("Hello there")]])
    chat.chat("hi", echo="output")
    assert "<details" not in updates[-1]
    assert updates[-1].endswith("Hello there")


def test_ipy_tool_result_renders_collapsed_html(monkeypatch):
    updates = capture_ipy(monkeypatch)

    chat = make_chat([[tool_request()], [text("It is 30°F.")]])
    chat.register_tool(get_temperature)
    chat.chat("temp in Duluth?", echo="output")

    res = updates[-1]
    # The shinychat markup, reused so the two can't drift.
    assert '<div class="chatlas-tool-result">' in res
    assert "Result from tool call: <code>get_temperature</code>" in res
    # Collapsed: the outer <details> has no `open` attribute.
    assert "<details><summary>Result from tool call" in res
    assert "It is 30°F." in res


def test_ipy_display_injects_tool_css_and_max_height(monkeypatch):
    html = capture_ipy_html(monkeypatch)
    chat = make_chat([[text("Hello")]])
    chat.chat("hi", echo="output")

    styles = [h for h in html if h.startswith("<style>")]
    assert len(styles) == 1
    assert ".chatlas-tool-result" in styles[0]
    assert "--chatlas-tool-result-max-height, 400px" in styles[0]

    wrapper = next(h for h in html if "chatlas-markdown" in h)
    assert "--chatlas-tool-result-max-height: 400px" in wrapper


def test_ipy_display_honors_max_height_option(monkeypatch):
    html = capture_ipy_html(monkeypatch)
    chat = make_chat([[text("Hello")]])
    chat.set_echo_options(tool_result_max_height="12rem")
    chat.chat("hi", echo="output")

    wrapper = next(h for h in html if "chatlas-markdown" in h)
    assert "--chatlas-tool-result-max-height: 12rem" in wrapper


def test_ipy_display_max_height_None_means_unbounded(monkeypatch):
    html = capture_ipy_html(monkeypatch)
    chat = make_chat([[text("Hello")]])
    chat.set_echo_options(tool_result_max_height=None)
    chat.chat("hi", echo="output")

    wrapper = next(h for h in html if "chatlas-markdown" in h)
    # `none` is CSS for "no max-height"; it must not fall through to the
    # 400px fallback in TOOL_CSS (or worse, a stringified Python None).
    assert "--chatlas-tool-result-max-height: none" in wrapper
    assert "None" not in wrapper


def test_ipy_display_keeps_css_styles_option(monkeypatch):
    html = capture_ipy_html(monkeypatch)
    chat = make_chat([[text("Hello")]])
    chat.set_echo_options(css_styles={"max-height": "300px"})
    chat.chat("hi", echo="output")

    styles = [h for h in html if h.startswith("<style>")]
    assert len(styles) == 1
    assert "max-height: 300px;" in styles[0]
    assert ".chatlas-tool-result" in styles[0]

    # The styles must target the wrapper div itself: its id and the
    # `chatlas-markdown` class are on the same element, so a sibling
    # selector (`#id + .chatlas-markdown`) would never match anything.
    wrapper = next(h for h in html if h.startswith("<div"))
    match = re.search(r"id='([^']+)'", wrapper)
    assert match
    assert f"#{match.group(1)}.chatlas-markdown {{" in styles[0]


def test_live_terminal_output_contains_final_frame():
    """
    On a real terminal, `Live` repaints on every chunk. Assert the escape
    sequences show up and that the last frame still holds the whole response
    (i.e. the `crop_above` LiveRender patch didn't eat it).
    """
    chat = make_chat([[text("# Title\n\n"), text("body text")]])
    output = capture_echo(chat, force_terminal=True, normalize=False)
    chat.chat("hi", echo="output")
    res = output()
    assert "\x1b[2K" in res  # erase-line, i.e. a repaint happened
    assert res.count("Title") > 1  # repainted at least once
    assert "body text" in res.rsplit("Title", 1)[-1]


def test_echo_all_shows_finish_reason_and_other_content_markers():
    chat = make_chat(
        [[text("Here you go"), ContentImageRemote(url="https://example.com/a.png")]],
        finish_reason="stop",
    )
    output = capture_echo(chat, width=100)
    chat.chat("hi", echo="all")
    res = output()
    assert "🤖 other content" in res
    assert "finish reason: stop" in res


@pytest.mark.parametrize(
    ("color_system", "color_escape"),
    [("truecolor", "\x1b[38;2;"), ("256", "\x1b[38;5;")],
)
def test_console_renders_inline_images_as_colored_thumbnails(
    color_system: Literal["truecolor", "256"], color_escape: str
):
    image = inline_png((8, 6))
    chat = make_chat([[text("Done.")]])
    output = capture_echo(chat, color_system=color_system)

    chat.chat("look at this", image, echo="all")

    rendered = output()
    assert image.data not in rendered
    assert "▀" in rendered
    assert color_escape in rendered
    assert image_label("image/png", image.data, (8, 6)) in rendered


def test_generated_image_is_echoed_at_the_default_echo():
    image = inline_png((8, 6))
    chat = make_chat([[text("Here it is."), image]])
    output = capture_echo(chat, width=60)

    chat.chat("draw a die", echo="output")

    rendered = output()
    assert "Here it is." in rendered
    assert "🖼 image/png" in rendered


def test_generated_image_is_echoed_exactly_once_at_echo_all():
    image = inline_png((8, 6))
    chat = make_chat([[text("Here it is."), image]])
    output = capture_echo(chat, width=60)

    chat.chat("draw a die", echo="all")

    assert output().count("🖼 image/png") == 1


def test_generated_image_is_not_echoed_at_echo_text():
    image = inline_png((8, 6))
    chat = make_chat([[text("Here it is."), image]])
    output = capture_echo(chat, width=60)

    chat.chat("draw a die", echo="text")

    rendered = output()
    assert "Here it is." in rendered
    assert "🖼" not in rendered


def test_user_supplied_image_still_shows_at_echo_all():
    image = inline_png((8, 6))
    chat = make_chat([[text("A die.")]])
    output = capture_echo(chat, width=60)

    chat.chat("what is this?", image, echo="all")

    assert "🖼 image/png" in output()


@pytest.mark.asyncio
async def test_generated_image_is_echoed_in_async_chats():
    image = inline_png((8, 6))
    chat = make_chat([[text("Here it is."), image]])
    output = capture_echo(chat, width=60)

    await chat.chat_async("draw a die", echo="output")

    assert "🖼 image/png" in output()


def test_generated_image_is_echoed_without_streaming():
    image = inline_png((8, 6))
    chat = make_chat([[text("Here it is."), image]])
    output = capture_echo(chat, width=60)

    chat.chat("draw a die", echo="output", stream=False)

    assert "🖼 image/png" in output()


def test_image_thumbnail_leaves_fully_transparent_pixel_pairs_unstyled():
    image = Image.new("RGBA", (1, 2))
    image.putdata([(255, 0, 0, 0), (0, 0, 255, 0)])

    pixel = thumbnail_pixel_segment(image)

    assert pixel.text == " "
    assert pixel.style is None


def test_image_thumbnail_composites_partially_transparent_pixels():
    image = Image.new("RGBA", (1, 2))
    image.putdata([(255, 0, 0, 128), (0, 255, 0, 64)])

    pixel = thumbnail_pixel_segment(image)

    assert pixel.text == "▀"
    assert pixel.style is not None
    assert pixel.style.color is not None
    assert pixel.style.bgcolor is not None
    assert pixel.style.color.triplet == (192, 64, 64)
    assert pixel.style.bgcolor.triplet == (96, 160, 96)


def test_console_caps_inline_image_thumbnail_rows():
    image = inline_png((60, 120))
    chat = make_chat([[text("Done.")]])
    output = capture_echo(
        chat,
        width=20,
        color_system="truecolor",
        image_max_lines=3,
    )

    chat.chat("look at this", image, echo="all")

    rows = [line for line in output().splitlines() if "▀" in line]
    assert len(rows) == 3


def test_console_inline_image_without_color_falls_back_to_a_label():
    image = inline_png((8, 6))
    chat = make_chat([[text("Done.")]])
    output = capture_echo(chat, color_system=None)

    chat.chat("look at this", image, echo="all")

    rendered = output()
    assert image.data not in rendered
    assert "▀" not in rendered
    assert image_label("image/png", image.data) in rendered


def test_console_inline_image_without_pillow_falls_back_to_a_label(monkeypatch):
    image = inline_png((8, 6))
    monkeypatch.setitem(sys.modules, "PIL", None)
    chat = make_chat([[text("Done.")]])
    output = capture_echo(chat, color_system="truecolor")

    chat.chat("look at this", image, echo="all")

    rendered = output()
    assert image.data not in rendered
    assert "▀" not in rendered
    assert image_label("image/png", image.data) in rendered


def test_console_corrupt_inline_image_falls_back_to_a_label():
    image = ContentImageInline(image_content_type="image/png", data="not image data")
    chat = make_chat([[text("Done.")]])
    output = capture_echo(chat, color_system="truecolor")

    chat.chat("look at this", image, echo="all")

    rendered = output()
    assert image.data not in rendered
    assert "▀" not in rendered
    assert image_label("image/png", image.data) in rendered


def test_console_keeps_remote_images_as_markdown():
    image = ContentImageRemote(url="https://example.com/a.png")
    chat = make_chat([[text("Done.")]])
    output = capture_echo(chat)

    chat.chat("look at this", image, echo="all")

    assert "🌆 a.png" in output()


def test_image_max_lines_defaults_to_sixteen():
    chat = make_chat([[text("Done.")]])

    assert DEFAULT_IMAGE_MAX_LINES == 16
    assert chat._echo_options["image_max_lines"] == DEFAULT_IMAGE_MAX_LINES


def test_notebook_keeps_inline_images_as_markdown(monkeypatch):
    updates = capture_ipy(monkeypatch)
    image = inline_png((8, 6))
    chat = make_chat([[text("Done.")]])

    chat.chat("look at this", image, echo="all")

    assert f"![](data:image/png;base64,{image.data})" in updates[-1]


def test_display_is_restored_when_the_provider_raises():
    """
    `chat()` runs the display as a context manager, so a mid-stream failure must
    still stop `Live` -- otherwise the user's cursor stays hidden and their
    terminal is left wedged.
    """
    chat = make_raising_chat([[text("partial "), text("answer")]])
    output = capture_echo(chat, force_terminal=True, normalize=False)

    with pytest.raises(RuntimeError, match="blew up mid-stream"):
        chat.chat("hi", echo="output")

    assert chat.current_display is None
    assert SHOW_CURSOR in output()


def test_display_is_restored_when_a_stream_is_abandoned():
    """
    `stream()` keeps the display open inside a generator, so it only unwinds
    when that generator is closed. Bailing out of the loop early is the common
    way users hit this.
    """
    chat = make_chat([[text("one "), text("two "), text("three")]])
    output = capture_echo(chat, force_terminal=True, normalize=False)

    stream = chat.stream("hi", echo="output")
    next(stream)
    stream.close()  # what garbage collection does after a `break`

    assert chat.current_display is None
    assert SHOW_CURSOR in output()


def test_consecutive_turns_do_not_replay_the_previous_one():
    """
    The display accumulates segments, so a second `chat()` must start from an
    empty list rather than repainting the first response above the second.
    """
    chat = make_chat([[text("first answer")], [text("second answer")]])

    first = capture_echo(chat)
    chat.chat("one", echo="output")
    assert first() == "first answer"

    second = capture_echo(chat)
    chat.chat("two", echo="output")
    assert second() == "second answer"


def test_consecutive_turns_do_not_replay_thinking(monkeypatch):
    """Same invariant for the notebook display, whose thinking segment is mutable."""
    updates = capture_ipy(monkeypatch)
    chat = make_chat([thinking_chunks(), [text("second answer")]])

    chat.chat("one", echo="output")
    chat.chat("two", echo="output")

    assert "2+2 is 4." not in updates[-1]
    assert "<details" not in updates[-1]
    assert updates[-1].endswith("second answer")


@pytest.mark.parametrize(
    "overflow, kept, dropped",
    [
        ("crop_above", "line-10", "line-01"),
        ("crop", "line-01", "line-10"),
    ],
)
def test_live_render_vertical_overflow(overflow: str, kept: str, dropped: str):
    """
    Unit-tests the vendored `LiveRender` patch, which exists solely to add
    `crop_above` (upstream rich only crops the tail). This is copied from rich
    internals, so it's the piece most likely to break on a rich upgrade.

    Driving it through `Live` doesn't work here: a `StringIO` console's height
    never reaches `options.size`, so the overflow branch is unreachable from the
    `Chat` API.
    """
    console = Console(file=StringIO(), width=30, height=5)
    body = Text("\n".join(f"line-{i:02d}" for i in range(1, 11)))

    lines = console.render_lines(
        LiveRender(body, vertical_overflow=overflow),  # pyright: ignore[reportArgumentType]
        console.options,
    )
    rendered = "".join(seg.text for line in lines for seg in line)

    assert kept in rendered
    assert dropped not in rendered


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHOW_CURSOR = "\x1b[?25h"


def capture_echo(
    chat: Chat,
    *,
    width: int = 60,
    force_terminal: bool = False,
    normalize: bool = True,
    rich_markdown: Optional[dict[str, object]] = None,
    # MISSING, not None: `None` means "don't bound this", so it can't double as
    # "the test didn't ask for anything".
    tool_result_max_lines: "int | None | MISSING_TYPE" = MISSING,
    thinking_max_lines: "int | None | MISSING_TYPE" = MISSING,
    web_activity_max_sources: "int | None | MISSING_TYPE" = MISSING,
    image_max_lines: "int | None | MISSING_TYPE" = MISSING,
    color_system: Literal["256", "truecolor"] | None = None,
    height: Optional[int] = None,
) -> Callable[[], str]:
    """
    Point `chat`'s echo console at a buffer. Returns a getter for its contents.

    By default the text is normalized: rich pads every line out to the console
    width, which would otherwise make assertions and snapshots whitespace-noisy.

    `height` bounds the console viewport, which is what triggers `Live`'s
    vertical-overflow handling.
    """
    buf = StringIO()
    console: dict[str, Any] = {
        "file": buf,
        "width": width,
        "force_terminal": force_terminal,
        "color_system": color_system,
    }
    if color_system is not None:
        console["no_color"] = False
    if height is not None:
        console["height"] = height
    chat.set_echo_options(
        rich_console=console,
        rich_markdown=rich_markdown,
        tool_result_max_lines=tool_result_max_lines,
        thinking_max_lines=thinking_max_lines,
        web_activity_max_sources=web_activity_max_sources,
        image_max_lines=image_max_lines,
    )

    def get() -> str:
        res = buf.getvalue()
        if not normalize:
            return res
        return "\n".join(line.rstrip() for line in res.splitlines()).strip("\n")

    return get


def render_tool_result_block(
    body: Text, max_lines: Optional[int], width: int = 60
) -> str:
    buffer = StringIO()
    console = Console(file=buffer, width=width)
    console.print(ToolResultBlock(body, max_lines=max_lines))
    return "\n".join(line.rstrip() for line in buffer.getvalue().splitlines()).strip(
        "\n"
    )


def capture_ipy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """
    Route `chat` at the notebook display and collect every markdown update.

    Returns the (growing) list, so assertions can look at intermediate frames as
    well as the final one.
    """
    import IPython.display

    updates: list[str] = []
    monkeypatch.setenv("QUARTO_PYTHON", "/usr/bin/python3")
    monkeypatch.setattr(IPython.display, "display", lambda *a, **kw: FakeHandle())
    monkeypatch.setattr(
        IPython.display,
        "update_display",
        lambda md, display_id: updates.append(md.data),
    )
    return updates


def capture_ipy_html(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Collect the raw HTML the notebook display emits when setting itself up."""
    import IPython.display

    html: list[str] = []

    def fake_display(*args: Any, **kwargs: Any) -> FakeHandle:
        for arg in args:
            if isinstance(arg, IPython.display.HTML):
                html.append(str(arg.data))
        return FakeHandle()

    monkeypatch.setenv("QUARTO_PYTHON", "/usr/bin/python3")
    monkeypatch.setattr(IPython.display, "display", fake_display)
    monkeypatch.setattr(IPython.display, "update_display", lambda md, display_id: None)
    return html


def long_reasoning_chunks() -> list[Content]:
    """
    Reasoning long enough to need capping, streamed in provider-sized chunks.

    Deliberately newline-free: that's how reasoning actually arrives, and it's
    what makes source-line counting useless.
    """
    reasoning = "".join(
        f"step {i:02d}: considering possibility {i} in some detail. " for i in range(40)
    )
    chunks: list[Content] = [
        ContentThinkingDelta(thinking=reasoning[i : i + 80])
        for i in range(0, len(reasoning), 80)
    ]
    chunks.append(text("The answer is **4**."))
    return chunks


def thinking_chunks() -> list[Content]:
    return [
        ContentThinkingDelta(thinking="Let me think. "),
        ContentThinkingDelta(thinking="2+2 is 4."),
        text("The answer is **4**."),
    ]


def text(value: str) -> ContentText:
    return ContentText.model_construct(text=value)


def inline_png(size: tuple[int, int]) -> ContentImageInline:
    image = Image.new("RGB", size, color=(255, 64, 32))
    return inline_image(image)


def thumbnail_pixel_segment(image: Image.Image):
    content = inline_image(image)
    thumbnail = ImageThumbnail(content.image_content_type, content.data, max_lines=None)
    decoded = thumbnail._decode()
    assert decoded is not None
    return next(
        segment
        for segment in thumbnail._segments(decoded, max_width=1).segments
        if segment.text != "\n"
    )


def inline_image(image: Image.Image) -> ContentImageInline:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return ContentImageInline(
        image_content_type="image/png",
        data=base64.b64encode(buffer.getvalue()).decode("ascii"),
    )


def tool_request(
    name: str = "get_temperature",
    arguments: Optional[dict[str, object]] = None,
) -> ContentToolRequest:
    if arguments is None:
        arguments = {"city": "Duluth"}
    return ContentToolRequest(id="call_1", name=name, arguments=arguments)


def get_temperature(city: str) -> int:
    "Get the current temperature in a city."
    return 30


def big_tool() -> list[dict[str, object]]:
    "A tool whose result is far too tall to show in full."
    return [{"i": i, "value": f"row-{i}"} for i in range(30)]


def logging_tool(log: logging.Logger) -> Callable[[], str]:
    def logging_tool() -> str:
        "A tool that logs."
        log.info("a log inside the live console")
        return "done"

    return logging_tool


class FakeHandle:
    display_id = "abc123"


class FakeChunk:
    def __init__(self, content: Optional[Content]):
        self.content = content


class EchoCompletion:
    def __init__(self, contents: list[Content]):
        self.contents = contents


class EchoProvider(Provider[EchoCompletion, FakeChunk, dict[str, Any], AnyTypeDict]):
    """
    Yields one canned sequence of content chunks per `chat_perform()` call, so a
    tool loop can be driven by handing it several sequences.
    """

    def __init__(
        self,
        responses: Sequence[Sequence[Content]],
        *,
        finish_reason: Optional[str] = None,
    ):
        super().__init__(name="echo", model="echo-model")
        self._responses = list(responses)
        self._current: list[Content] = []
        self._finish_reason = finish_reason

    def list_models(self) -> list[Any]:
        return []

    def _next_response(self) -> list[Content]:
        self._current = list(self._responses.pop(0))
        return self._current

    @overload
    def chat_perform(
        self, *, stream: Literal[False], turns, tools, data_model, kwargs
    ): ...

    @overload
    def chat_perform(
        self, *, stream: Literal[True], turns, tools, data_model, kwargs
    ): ...

    def chat_perform(self, *, stream, turns, tools, data_model, kwargs):
        contents = self._next_response()
        if stream:
            return iter([FakeChunk(c) for c in contents])
        return EchoCompletion(contents)

    @overload
    async def chat_perform_async(
        self, *, stream: Literal[False], turns, tools, data_model, kwargs
    ): ...

    @overload
    async def chat_perform_async(
        self, *, stream: Literal[True], turns, tools, data_model, kwargs
    ): ...

    async def chat_perform_async(self, *, stream, turns, tools, data_model, kwargs):
        contents = self._next_response()

        async def gen():
            for c in contents:
                yield FakeChunk(c)

        if stream:
            return gen()
        return EchoCompletion(contents)

    def stream_content(self, chunk: FakeChunk, completion) -> list[Content]:
        return [chunk.content] if chunk.content is not None else []

    def stream_merge_chunks(self, completion, chunk):
        return completion or {}

    def stream_turn(self, completion, has_data_model):
        return self._turn_from_current()

    def value_turn(self, completion, has_data_model):
        return self._turn_from_current()

    def _turn_from_current(self) -> AssistantTurn:
        """
        Build the completed turn the way a real provider does.

        Text chunks merge into one `ContentText`, thinking deltas merge into one
        `ContentThinking` (Anthropic/OpenAI both do this), tool requests pass
        through. Reasoning has to be present here or the display's
        no-double-echo guard wouldn't be exercised.
        """
        contents: list[Content] = []
        merged = ""
        thinking = ""
        for c in self._current:
            if isinstance(c, ContentText):
                merged += c.text
            elif isinstance(c, ContentThinkingDelta):
                thinking += c.thinking
            else:
                contents.append(c)
        if merged:
            contents.insert(0, text(merged))
        if thinking:
            contents.insert(0, ContentThinking(thinking=thinking))
        return AssistantTurn(
            contents=contents,
            tokens=None,
            completion=None,
            finish_reason=self._finish_reason,
        )

    def value_tokens(self, completion):
        return None

    def value_cost(self, completion, tokens=None):
        return None

    def token_count(self, *args, **kwargs):
        return 0

    async def token_count_async(self, *args, **kwargs):
        return 0

    def translate_model_params(self, *args, **kwargs) -> AnyTypeDict:
        return AnyTypeDict()

    def supported_model_params(self):
        return set()


class RaisingProvider(EchoProvider):
    """Fails partway through the stream, after some content has been echoed."""

    _calls = 0

    def stream_content(self, chunk: FakeChunk, completion) -> list[Content]:
        self._calls += 1
        if self._calls > 1:
            raise RuntimeError("provider blew up mid-stream")
        return super().stream_content(chunk, completion)


def make_chat(
    responses: Sequence[Sequence[Content]],
    *,
    finish_reason: Optional[str] = None,
) -> Chat:
    return Chat(provider=EchoProvider(responses, finish_reason=finish_reason))


def make_raising_chat(responses: Sequence[Sequence[Content]]) -> Chat:
    return Chat(provider=RaisingProvider(responses))


# ---------------------------------------------------------------------------
# WebActivitySegment (the grouping model)
# ---------------------------------------------------------------------------


def test_web_segment_groups_queries_and_sources():
    seg = WebActivitySegment()
    seg.add_query("ggplot2 release date")
    seg.add_sources(
        [
            WebSource(url="https://a.com/x", title="Alpha"),
            WebSource(url="https://b.com/y", title="Beta"),
        ]
    )

    assert seg.queries == ["ggplot2 release date"]
    assert [r.title for r in seg.sources] == ["Alpha", "Beta"]
    assert seg.header() == "Searching the web…"

    seg.is_open = False
    assert seg.header() == "Searched the web"
    assert seg.detail() == "2 results"


def test_web_segment_dedupes_the_fetch_request_and_result():
    """A fetch emits a request and a result for the same URL; that's one row."""
    seg = WebActivitySegment()
    seg.add_fetch("https://example.com/p", None)
    seg.add_fetch("https://example.com/p", "success")

    assert seg.fetches == [("https://example.com/p", "success")]
    assert seg.is_fetch_only is True
    assert seg.header() == "Reading the web…"

    seg.is_open = False
    assert seg.header() == "Read the web"


def test_web_segment_citation_marks_an_existing_source():
    """Google cites a URL that is already a result -- mark it, don't duplicate it."""
    seg = WebActivitySegment()
    seg.add_sources([WebSource(url="https://a.com/x", title="Alpha")])
    seg.add_citation(
        ContentCitation(
            source=WebSource(url="https://a.com/x", title="Alpha"),
            cited_quote="the quoted passage",
        )
    )

    assert len(seg.sources) == 1
    assert seg.sources[0].cited is True
    assert seg.sources[0].quote == "the quoted passage"
    seg.is_open = False
    assert seg.detail() == "1 result · * 1 cited"


def test_web_segment_citation_becomes_a_row_when_it_matches_nothing():
    """OpenAI returns citations and zero results, so the citation IS the row."""
    seg = WebActivitySegment()
    seg.add_query("ggplot2 CRAN archive")
    seg.add_citation(
        ContentCitation(source=WebSource(url="https://c.com/z", title="Gamma"))
    )

    assert [(r.title, r.cited) for r in seg.sources] == [("Gamma", True)]


def test_web_segment_sourceless_citations_do_not_pile_up():
    """Google can ground a span with no resolvable URL; don't add a blank row twice."""
    seg = WebActivitySegment()
    seg.add_citation(ContentCitation(grounded_span="released on 2014-05-21"))
    seg.add_citation(ContentCitation(grounded_span="released on 2014-05-21"))

    assert len(seg.sources) == 1
    assert seg.sources[0].title == "released on 2014-05-21"


def test_web_segment_add_sources_dedupes_by_url():
    seg = WebActivitySegment()
    seg.add_sources([WebSource(url="https://a.com/x", title="Alpha")])
    seg.add_sources([WebSource(url="https://a.com/x", title="Alpha again")])

    assert len(seg.sources) == 1


def test_web_domain_extracts_hostname_and_handles_falsy_input():
    assert web_domain("https://a.com/x") == "a.com"
    # Gemini's grounding-redirect URLs are opaque, but their hostname is still
    # legible -- that's what makes the domain badge worth showing at all.
    assert web_domain("https://vertexaisearch.cloud.google.com/x/y") == (
        "vertexaisearch.cloud.google.com"
    )
    assert web_domain(None) == ""


def test_capped_sources_returns_everything_when_uncapped():
    rows = [WebActivityRow(url=f"https://s{i}.com") for i in range(6)]
    shown, dropped = capped_sources(rows, max_sources=None)
    assert shown == rows
    assert dropped == 0


def test_capped_sources_returns_everything_under_the_cap():
    rows = [WebActivityRow(url=f"https://s{i}.com") for i in range(3)]
    shown, dropped = capped_sources(rows, max_sources=4)
    assert shown == rows
    assert dropped == 0


def test_capped_sources_prioritizes_a_cited_row_outside_the_naive_top_n():
    """A citation on row 5 of 10 must still show, not get sliced off by a naive top-4."""
    rows = [WebActivityRow(url=f"https://s{i}.com") for i in range(10)]
    rows[5].cited = True

    shown, dropped = capped_sources(rows, max_sources=4)

    assert shown == [rows[0], rows[1], rows[2], rows[5]]
    assert dropped == 6


def test_capped_sources_enforces_a_hard_bound_when_cited_rows_exceed_the_cap():
    """
    Six cited rows against a cap of 4 must not blow past the bound -- a bounded
    panel is the entire reason the console treatment exists, so the cap can't
    grow just because citations are plentiful.
    """
    rows = [WebActivityRow(url=f"https://s{i}.com", cited=(i < 6)) for i in range(10)]

    shown, dropped = capped_sources(rows, max_sources=4)

    assert shown == rows[:4]
    assert all(row.cited for row in shown)
    assert dropped == 6


# ---------------------------------------------------------------------------
# Web activity rendering
# ---------------------------------------------------------------------------


def search_chunks(n_sources: int = 6) -> list[Content]:
    """A search episode in provider order: request, results, text, citation, text."""
    sources = [
        WebSource(url=f"https://s{i}.com/page", title=f"Source {i}")
        for i in range(1, n_sources + 1)
    ]
    return [
        ContentToolRequestSearch(query="ggplot2 release date"),
        ContentToolResponseSearch(sources=sources),
        text("ggplot2 1.0.0 was released on 2014-05-21."),
        ContentCitation(
            source=sources[0], grounded_span="released on 2014-05-21",
            cited_quote="ggplot2 1.0.0 (2014-05-21)",
        ),
        text(" That's the CRAN date."),
    ]


def test_console_renders_a_web_activity_panel():
    chat = make_chat([search_chunks()])
    output = capture_echo(chat, width=78)
    chat.chat("when?", echo="output")
    out = output()

    assert "Searched the web" in out
    assert "6 results" in out
    assert "1 cited" in out
    assert "ggplot2 release date" in out
    # Panel border
    assert "╭" in out


def test_console_caps_sources_and_reports_the_remainder():
    chat = make_chat([search_chunks(n_sources=10)])
    output = capture_echo(chat, width=78)
    chat.chat("when?", echo="output")
    out = output()

    assert "Source 1" in out
    assert "Source 4" in out
    assert "Source 5" not in out
    assert "… 6 more" in out


def test_console_source_cap_is_tunable_and_disablable():
    chat = make_chat([search_chunks(n_sources=10)])
    output = capture_echo(chat, width=78, web_activity_max_sources=None)
    chat.chat("when?", echo="output")
    out = output()

    assert "Source 10" in out
    assert "more" not in out


def test_console_shows_title_and_domain():
    chat = make_chat([search_chunks(n_sources=1)])
    output = capture_echo(chat, width=78)
    chat.chat("when?", echo="output")
    out = output()

    assert "Source 1" in out
    assert "s1.com" in out


def test_console_marks_the_cited_source():
    chat = make_chat([search_chunks(n_sources=2)])
    output = capture_echo(chat, width=78)
    chat.chat("when?", echo="output")
    out = output()

    # The marker on the row, and the title legend that explains it.
    assert "* Source 1" in out
    assert "· * 1 cited" in out


def test_citation_after_text_joins_the_same_panel():
    """
    Providers send citations *after* the text they ground. Closing the episode on
    text and opening a new one for the citation would render two panels.
    """
    chat = make_chat([search_chunks()])
    output = capture_echo(chat, width=78)
    chat.chat("when?", echo="output")
    out = output()

    assert out.count("Searched the web") == 1


def test_fetch_renders_one_row_with_its_status():
    chat = make_chat(
        [
            [
                ContentToolRequestFetch(url="https://example.com/page"),
                ContentToolResponseFetch(
                    url="https://example.com/page", status="success"
                ),
                text("The page says hello."),
            ]
        ]
    )
    output = capture_echo(chat, width=78)
    chat.chat("what?", echo="output")
    out = output()

    assert "Read the web" in out
    assert out.count("example.com/page") == 1
    assert "✓" in out


def test_console_refuses_a_non_http_scheme_source_with_no_title():
    """
    Mirrors `test_notebook_refuses_a_non_http_scheme_with_no_title`: the console
    had no analog of the notebook's URL-scheme guard. An unsafe scheme must not
    become a live OSC-8 hyperlink, and -- since there's no title to fall back to
    -- it must not leak into the visible label text either.
    """
    chat = make_chat(
        [
            [
                ContentToolResponseSearch(
                    sources=[WebSource(url="javascript:alert(1)", title=None)]
                ),
                text("done"),
            ]
        ]
    )
    output = capture_echo(chat, width=78, force_terminal=True, normalize=False)
    chat.chat("go", echo="output")
    out = output()

    # Covers both failure modes at once: this substring shows up either as the
    # OSC-8 link target (`\x1b]8;id=...;javascript:alert(1)\x1b\\`) or as the
    # plain-text label, so its absence rules out both.
    assert "javascript:alert(1)" not in out
    assert "(source)" in out


def test_console_refuses_a_non_http_scheme_fetch():
    """Same guard, for a fetched URL rather than a search source."""
    chat = make_chat(
        [
            [
                ContentToolRequestFetch(url="javascript:alert(1)"),
                ContentToolResponseFetch(url="javascript:alert(1)", status="success"),
                text("done"),
            ]
        ]
    )
    output = capture_echo(chat, width=78, force_terminal=True, normalize=False)
    chat.chat("go", echo="output")
    out = output()

    assert "javascript:alert(1)" not in out
    assert "(url)" in out


def test_notebook_renders_a_collapsed_details_block(monkeypatch):
    updates = capture_ipy(monkeypatch)
    chat = make_chat([search_chunks()])
    chat.chat("when?", echo="output")
    final = updates[-1]

    assert "chatlas-web-activity" in final
    assert "<details>" in final
    # Collapsed in every state, unlike thinking -- provenance is consulted, not watched.
    assert "<details open>" not in final
    assert "Searched the web" in final


def test_notebook_keeps_every_source(monkeypatch):
    """The cap is console-only: a notebook can scroll."""
    updates = capture_ipy(monkeypatch)
    chat = make_chat([search_chunks(n_sources=10)])
    chat.chat("when?", echo="output")
    final = updates[-1]

    assert "Source 10" in final
    assert 'href="https://s10.com/page"' in final
    assert "… 6 more" not in final


def test_notebook_shows_the_cited_quote(monkeypatch):
    updates = capture_ipy(monkeypatch)
    chat = make_chat([search_chunks()])
    chat.chat("when?", echo="output")

    assert "ggplot2 1.0.0 (2014-05-21)" in updates[-1]


def test_notebook_injects_web_css(monkeypatch):
    html = capture_ipy_html(monkeypatch)
    chat = make_chat([search_chunks()])
    chat.chat("when?", echo="output")

    assert any("chatlas-web-activity" in h for h in html)


def test_notebook_escapes_model_supplied_text(monkeypatch):
    updates = capture_ipy(monkeypatch)
    chat = make_chat(
        [
            [
                ContentToolRequestSearch(query="<script>alert(1)</script>"),
                text("done"),
            ]
        ]
    )
    chat.chat("go", echo="output")

    assert "<script>alert(1)</script>" not in updates[-1]
    assert "&lt;script&gt;" in updates[-1]


def test_notebook_refuses_a_non_http_scheme(monkeypatch):
    updates = capture_ipy(monkeypatch)
    chat = make_chat(
        [
            [
                ContentToolResponseSearch(
                    sources=[WebSource(url="javascript:alert(1)", title="Bad")]
                ),
                text("done"),
            ]
        ]
    )
    chat.chat("go", echo="output")

    assert "javascript:alert(1)" not in updates[-1]
    assert "Bad" in updates[-1]


def test_notebook_refuses_a_non_http_scheme_with_no_title(monkeypatch):
    """
    Without a title, the label falls back toward the domain (or a placeholder)
    -- it must not let that fallback leak an unsafe-scheme URL the same way an
    absent `href` already refuses one.
    """
    updates = capture_ipy(monkeypatch)
    chat = make_chat(
        [
            [
                ContentToolResponseSearch(
                    sources=[WebSource(url="javascript:alert(1)", title=None)]
                ),
                text("done"),
            ]
        ]
    )
    chat.chat("go", echo="output")

    assert "javascript:alert(1)" not in updates[-1]
    assert "(source)" in updates[-1]


def test_notebook_fetch_marker_reflects_pending_success_and_error(monkeypatch):
    """
    A fetch emits a request (status is `None`) before its result arrives. Mid-
    stream that must render the pending marker, not the success checkmark --
    otherwise a turn that ends before the result arrives leaves a wrong ✓
    baked into the final cell.
    """
    updates = capture_ipy(monkeypatch)
    chat = make_chat(
        [
            [
                ContentToolRequestFetch(url="https://example.com/pending"),
                ContentToolRequestFetch(url="https://example.com/ok"),
                ContentToolResponseFetch(url="https://example.com/ok", status="success"),
                ContentToolRequestFetch(url="https://example.com/bad"),
                ContentToolResponseFetch(url="https://example.com/bad", status="error"),
                text("done"),
            ]
        ]
    )
    chat.chat("go", echo="output")
    final = updates[-1]

    assert "<span class='chatlas-web-st pending'>…</span>" in final
    assert "<span class='chatlas-web-st ok'>✓</span>" in final
    assert "<span class='chatlas-web-st err'>✗</span>" in final


def test_web_activity_is_visible_in_the_default_echo_mode():
    """
    The whole point of #256: with a built-in tool registered, `echo="output"`
    used to give no sign the model searched at all.
    """
    chat = make_chat([search_chunks()])
    output = capture_echo(chat, width=78)
    chat.chat("when?", echo="output")

    assert "Searched the web" in output()


def test_web_activity_precedes_the_answer_text():
    """It's emitted as it streams, so it lands above the answer, not after it."""
    chat = make_chat([search_chunks()])
    output = capture_echo(chat, width=78)
    chat.chat("when?", echo="output")
    out = output()

    assert out.index("Searched the web") < out.index("ggplot2 1.0.0 was released")


def test_echo_text_suppresses_web_activity():
    """`echo="text"` means just the answer, the same way it holds back reasoning."""
    chat = make_chat([search_chunks()])
    output = capture_echo(chat, width=78)
    chat.chat("when?", echo="text")
    out = output()

    assert "Searched the web" not in out
    assert "ggplot2 release date" not in out
    assert "ggplot2 1.0.0 was released" in out


def test_echo_none_writes_no_web_activity():
    chat = make_chat([search_chunks()])
    output = capture_echo(chat, width=78)
    chat.chat("when?", echo="none")

    assert output() == ""


def test_echo_all_shows_web_activity_exactly_once():
    """
    Regression against the duplication #362 had to solve for thinking: the panel
    is built while streaming, so `emit_other_contents` must not repeat it.
    """
    chat = make_chat([search_chunks()])
    output = capture_echo(chat, width=78)
    chat.chat("when?", echo="all")
    out = output()

    assert out.count("Searched the web") == 1
    assert out.count("ggplot2 release date") == 1
    # With web activity handled as a panel, it's no longer "other content".
    assert "other content" not in out


def test_two_searches_separated_by_text_render_two_panels():
    chat = make_chat(
        [
            [
                ContentToolRequestSearch(query="first query"),
                text("Some interim thinking out loud."),
                ContentToolRequestSearch(query="second query"),
                text("Final answer."),
            ]
        ]
    )
    output = capture_echo(chat, width=78)
    chat.chat("go", echo="output")

    assert output().count("Searched the web") == 2


def test_web_activity_streams_to_the_notebook_too(monkeypatch):
    updates = capture_ipy(monkeypatch)
    chat = make_chat([search_chunks()])
    chat.chat("when?", echo="output")

    assert any("Searching the web" in u for u in updates), (
        "the live header should appear while the episode is still open"
    )
    assert "Searched the web" in updates[-1]


def test_web_activity_renders_without_streaming():
    chat = make_chat([search_chunks()])
    output = capture_echo(chat, width=78)
    chat.chat("when?", echo="output", stream=False)
    out = output()

    assert "Searched the web" in out
    assert "ggplot2 release date" in out
    assert "6 results" in out


def test_non_streamed_web_activity_precedes_the_text():
    chat = make_chat([search_chunks()])
    output = capture_echo(chat, width=78)
    chat.chat("when?", echo="output", stream=False)
    out = output()

    assert out.index("Searched the web") < out.index("ggplot2 1.0.0 was released")


def test_non_streamed_web_activity_is_not_duplicated_by_echo_all():
    chat = make_chat([search_chunks()])
    output = capture_echo(chat, width=78)
    chat.chat("when?", echo="all", stream=False)

    assert output().count("Searched the web") == 1


@pytest.mark.asyncio
async def test_web_activity_renders_without_streaming_async():
    chat = make_chat([search_chunks()])
    output = capture_echo(chat, width=78)
    await chat.chat_async("when?", echo="output", stream=False)

    assert "Searched the web" in output()


def test_format_bytes_scales_units():
    assert format_bytes(0) == "0 B"
    assert format_bytes(512) == "512 B"
    assert format_bytes(1024) == "1.0 KB"
    assert format_bytes(5498) == "5.4 KB"
    assert format_bytes(224566) == "219 KB"
    assert format_bytes(3 * 1024 * 1024) == "3.0 MB"
    assert format_bytes(10_235) == "10 KB"
    assert format_bytes(1024 * 1024 - 1) == "1.0 MB"


def test_base64_nbytes_matches_the_decoded_length():
    """Derived from the string length so a large image is never decoded twice."""
    raw = b"\x89PNG\r\n\x1a\n" + b"x" * 3001
    for n in range(3000, 3010):
        data = base64.b64encode(raw[:n]).decode()
        assert base64_nbytes(data) == n


def test_base64_nbytes_ignores_mime_whitespace_and_unpadded_data():
    canonical = base64.b64encode(b"x" * 100).decode("ascii")
    mime_wrapped = "\r\n".join(
        canonical[index : index + 76] for index in range(0, len(canonical), 76)
    )

    assert base64_nbytes(f" \t{mime_wrapped}\n") == 100
    assert base64_nbytes("eA") == 1
    assert base64_nbytes("eHg") == 2


@pytest.mark.parametrize("data", ["=", "==", "==="])
def test_base64_nbytes_never_returns_a_negative_size(data: str):
    assert base64_nbytes(data) == 0


def test_replace_images_preserves_tuples():
    image = inline_png((8, 6))

    replaced = replace_images(("before", image))

    assert isinstance(replaced, tuple)
    assert replaced[0] == "before"
    assert replaced[1].startswith("🖼 image/png")


def test_image_label_includes_dimensions_only_when_known():
    data = base64.b64encode(b"x" * 224566).decode()
    assert image_label("image/png", data) == "🖼 image/png · 219 KB"
    assert (
        image_label("image/png", data, (800, 600))
        == "🖼 image/png · 800×600 · 219 KB"
    )
