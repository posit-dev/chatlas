"""
Web-activity display against real recorded provider responses.

The five shapes differ in ways a fake provider won't reproduce: Google cites a
URL that is already one of its results, OpenAI returns citations and *no*
`ContentToolResponseSearch` at all, and both fetch providers emit a request and a
result carrying the same URL.

Cassettes are matched on path rather than body, since these replay recordings made
by other test modules with their own prompts.
"""

from io import StringIO

import pytest
from chatlas import (
    Chat,
    ChatAnthropic,
    ChatGoogle,
    ChatOpenAI,
    tool_web_fetch,
    tool_web_search,
)
from chatlas._chat import EchoOptions

from .conftest import VCR_MATCH_ON_WITHOUT_BODY, make_vcr_config

SEARCH_PROMPT = "When was ggplot2 1.0.0 released to CRAN? Answer in YYYY-MM-DD format."
FETCH_PROMPT = (
    "What's the first movie listed on "
    "https://rvest.tidyverse.org/articles/starwars.html?"
)


@pytest.fixture(scope="module")
def vcr_config():
    return make_vcr_config(match_on=VCR_MATCH_ON_WITHOUT_BODY)


def anthropic_search() -> Chat:
    chat = ChatAnthropic(model="claude-haiku-4-5-20251001")
    chat.register_tool(tool_web_search())
    return chat


def anthropic_fetch() -> Chat:
    chat = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        kwargs={"default_headers": {"anthropic-beta": "web-fetch-2025-09-10"}},
    )
    chat.register_tool(tool_web_fetch())
    return chat


def google_search() -> Chat:
    chat = ChatGoogle(model="gemini-2.5-flash")
    chat.register_tool(tool_web_search())
    return chat


def google_fetch() -> Chat:
    chat = ChatGoogle(model="gemini-2.5-flash")
    chat.register_tool(tool_web_fetch())
    return chat


def openai_search() -> Chat:
    chat = ChatOpenAI(model="gpt-4.1")
    chat.register_tool(tool_web_search())
    return chat


def echo_to_buffer(chat: Chat, prompt: str, *, echo: EchoOptions = "output") -> str:
    buf = StringIO()
    chat.set_echo_options(rich_console={"file": buf, "width": 78})
    chat.chat(prompt, echo=echo)
    return "\n".join(line.rstrip() for line in buf.getvalue().splitlines())


@pytest.mark.vcr
def test_anthropic_search_panel():
    out = echo_to_buffer(anthropic_search(), SEARCH_PROMPT)
    assert "Searched the web" in out
    assert "ggplot2 1.0.0 CRAN release date" in out
    assert "cran.r-project.org" in out
    # 10 recorded sources, capped at 4
    assert "… 6 more" in out
    assert "· * 1 cited" in out, "the recorded turn has a citation"


@pytest.mark.vcr
def test_anthropic_fetch_panel():
    """This turn rendered as blank space before this work."""
    out = echo_to_buffer(anthropic_fetch(), FETCH_PROMPT)
    assert "Read the web" in out
    assert out.count("rvest.tidyverse.org/articles/starwars.html") == 1, (
        "the fetch request and result carry one URL, so they're one row"
    )
    assert "✓" in out


@pytest.mark.vcr
def test_google_search_panel_uses_titles_not_redirect_urls():
    """Gemini's source URLs are opaque grounding-redirect blobs."""
    out = echo_to_buffer(google_search(), SEARCH_PROMPT)
    assert "Searched the web" in out
    assert "ggplot2 1.0.0 CRAN release date" in out
    assert "ggplot2 release history" in out, "Gemini issued two queries"
    assert "rpkg.net" in out, "the title, which is legible"
    assert "grounding-api-redirect" not in out, "the URL, which is not"


@pytest.mark.vcr
def test_google_search_citation_does_not_duplicate_its_source():
    out = echo_to_buffer(google_search(), SEARCH_PROMPT)
    assert out.count("rpkg.net") == 1
    assert "1 result" in out
    assert "1 cited" in out


@pytest.mark.vcr
def test_google_fetch_panel():
    out = echo_to_buffer(google_fetch(), FETCH_PROMPT)
    assert "Read the web" in out
    assert "rvest.tidyverse.org" in out


@pytest.mark.vcr
def test_openai_citations_become_the_source_list():
    """OpenAI returns citations and zero ContentToolResponseSearch."""
    out = echo_to_buffer(openai_search(), SEARCH_PROMPT)
    assert "Searched the web" in out
    assert "stat.ethz.ch" in out
    assert "2 cited" in out


@pytest.mark.vcr
def test_export_keeps_citations_after_rendering(tmp_path):
    """
    Regression for the link-reference bug in `export()`.

    `export(content="all")` joins `str(x)` for every content and both output
    formats are markdown-rendered, so a bracketed label vanished from exported
    transcripts as well as from the console.
    """
    from markdown_it import MarkdownIt

    chat = anthropic_search()
    chat.chat(SEARCH_PROMPT, echo="none")

    path = tmp_path / "transcript.md"
    chat.export(path, content="all")
    rendered = MarkdownIt("commonmark").render(path.read_text())

    assert "citation" in rendered.lower()
    assert "web search request" in rendered.lower()
