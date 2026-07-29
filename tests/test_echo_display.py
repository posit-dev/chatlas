"""
Tests for `echo=` output, as rendered by the rich-based live console display.

`LiveMarkdownDisplay` writes to a `rich.console.Console` built from
`Chat.set_echo_options(rich_console=)`, so pointing that at a `StringIO` lets us
assert on what a user would actually see. On a non-terminal console, `rich.Live`
skips incremental refreshes and prints only the final frame, which keeps these
assertions deterministic and free of ANSI escapes.
"""

import logging
import re
from collections.abc import Sequence
from io import StringIO
from typing import Any, Callable, Literal, Optional, overload

import pytest
from chatlas import Chat
from chatlas._content import (
    Content,
    ContentImageRemote,
    ContentText,
    ContentThinking,
    ContentThinkingDelta,
    ContentToolRequest,
    ContentToolResult,
)
from chatlas._display import IPyMarkdownDisplay, LiveMarkdownDisplay
from chatlas._live_render import LiveRender
from chatlas._logging import _rich_handler
from chatlas._provider import AnyTypeDict, Provider
from chatlas._turn import AssistantTurn
from chatlas._utils import MISSING, MISSING_TYPE
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
    assert "'row-4'" in res
    # 30 rows over 5 kept lines, and pformat puts one row per line.
    assert "'row-29'" not in res
    assert "… 25 more lines" in res

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
    assert res.count("… 25 more lines") == 1
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
    assert "… 10 more lines" in output()


def test_truncated_tool_result_reports_a_single_dropped_line():
    result = ContentToolResult(
        value="\n".join(f"line {i}" for i in range(6)),
        request=tool_request(),
    )
    assert "… 1 more line" in result.to_display_markdown(max_lines=5)
    assert "more lines" not in result.to_display_markdown(max_lines=5)


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


def test_ipy_display_keeps_css_styles_option(monkeypatch):
    html = capture_ipy_html(monkeypatch)
    chat = make_chat([[text("Hello")]])
    chat.set_echo_options(css_styles={"max-height": "300px"})
    chat.chat("hi", echo="output")

    styles = [h for h in html if h.startswith("<style>")]
    assert len(styles) == 1
    assert "max-height: 300px;" in styles[0]
    assert ".chatlas-tool-result" in styles[0]


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
    }
    if height is not None:
        console["height"] = height
    chat.set_echo_options(
        rich_console=console,
        rich_markdown=rich_markdown,
        tool_result_max_lines=tool_result_max_lines,
        thinking_max_lines=thinking_max_lines,
    )

    def get() -> str:
        res = buf.getvalue()
        if not normalize:
            return res
        return "\n".join(line.rstrip() for line in res.splitlines()).strip("\n")

    return get


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
