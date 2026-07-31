import pytest
from chatlas import ChatOpenAI
from chatlas._content import (
    ContentCitation,
    ContentDocument,
    ContentPDF,
    ContentText,
    ContentToolRequestFetch,
    ContentToolRequestSearch,
    ContentToolResponseFetch,
    ContentToolResponseSearch,
    ContentUploaded,
    WebSource,
    create_content,
    create_source,
)


def test_invalid_inputs_give_useful_errors():
    chat = ChatOpenAI()
    with pytest.raises(TypeError):
        chat.chat(question="Are unicorns real?")  # type: ignore
    with pytest.raises(ValueError):
        chat.chat(True)  # type: ignore


def test_web_source_fields():
    s = WebSource(url="https://python.org")
    assert s.type == "web"
    assert s.url == "https://python.org"
    assert s.title is None
    assert "python.org" in str(s)


def test_create_source_dispatches_web():
    s = create_source({"type": "web", "url": "https://a.com", "title": "A"})
    assert isinstance(s, WebSource)
    assert s.url == "https://a.com"


def test_create_source_unknown_type_raises():
    with pytest.raises(ValueError):
        create_source({"type": "nope"})


def test_search_results_use_web_sources():
    r = ContentToolResponseSearch(
        sources=[WebSource(url="https://python.org", title="Python")]
    )
    assert r.sources[0].title == "Python"
    assert "python.org" in str(r)


def test_search_results_roundtrip_rebuilds_web_source():
    r = ContentToolResponseSearch(sources=[WebSource(url="https://a.com", title="A")])
    restored = create_content(r.model_dump())
    assert isinstance(restored, ContentToolResponseSearch)
    assert isinstance(restored.sources[0], WebSource)
    assert restored.sources[0].url == "https://a.com"


def test_fetch_results_status():
    r = ContentToolResponseFetch(url="https://python.org", status="success")
    assert r.status == "success"
    assert ContentToolResponseFetch(url="https://x.com").status is None


def test_content_citation_nests_source():
    c = ContentCitation(source=WebSource(url="https://python.org", title="Python"))
    assert c.source is not None and c.source.url == "https://python.org"
    assert c.source.title == "Python"
    assert c.content_type == "citation"
    assert "https://python.org" in str(c)


def test_content_citation_cited_quote():
    c = ContentCitation(
        source=WebSource(url="https://a.com"),
        grounded_span="the sky is blue",
        cited_quote="The sky is blue on a clear day.",
    )
    assert c.cited_quote == "The sky is blue on a clear day."


def test_content_citation_roundtrip_rebuilds_web_source():
    c = ContentCitation(source=WebSource(url="https://a.com", title="A"))
    restored = create_content(c.model_dump())
    assert isinstance(restored, ContentCitation)
    assert isinstance(restored.source, WebSource)
    assert restored.source.url == "https://a.com"
    assert restored.source.title == "A"


def test_content_citation_link_less_source_none():
    c = ContentCitation(grounded_span="ggplot2 1.0.0 was released on 2014-05-21.")
    assert c.source is None
    assert c.grounded_span == "ggplot2 1.0.0 was released on 2014-05-21."
    assert c.extra is None
    restored = create_content(c.model_dump())
    assert isinstance(restored, ContentCitation)
    assert restored.source is None
    assert restored.grounded_span == c.grounded_span


def test_source_and_web_source_exported_from_types():
    from chatlas.types import Source, WebSource  # noqa: F401


def test_content_citation_exported_from_types():
    from chatlas.types import ContentCitation  # noqa: F401


def test_contenttext_add_concatenates():
    merged = ContentText(text="foo") + ContentText(text="bar")
    assert merged.text == "foobar"


def test_content_uploaded_roundtrip():
    c = ContentUploaded(id="file_123", mime_type="application/pdf", provider="openai")
    assert c.content_type == "uploaded"
    assert str(c) == "`[uploaded file_123 · application/pdf]`"

    dumped = c.model_dump()
    restored = create_content(dumped)
    assert isinstance(restored, ContentUploaded)
    assert restored.id == "file_123"
    assert restored.provider == "openai"


def test_content_document_roundtrip():
    c = ContentDocument(data=b"hello", filename="a.txt", mime_type="text/plain")
    assert c.content_type == "document"

    dumped = c.model_dump(mode="json")
    restored = create_content(dumped)
    assert isinstance(restored, ContentDocument)
    assert restored.data == b"hello"
    assert restored.filename == "a.txt"
    assert restored.mime_type == "text/plain"


@pytest.mark.parametrize(
    "content, expected",
    [
        (
            ContentPDF(data=b"x" * (219 * 1024), filename="report.pdf"),
            "`[PDF report.pdf · 219 KB]`",
        ),
        (
            ContentPDF(
                filename="r.pdf",
                url="https://example.com/r.pdf",
            ),
            "`[PDF r.pdf · https://example.com/r.pdf]`",
        ),
        (
            ContentDocument(data=b"hello", filename="data.csv", mime_type="text/csv"),
            "`[document data.csv · text/csv]`",
        ),
        (
            ContentUploaded(
                id="file-abc123",
                mime_type="image/png",
                provider="openai",
            ),
            "`[uploaded file-abc123 · image/png]`",
        ),
    ],
)
def test_file_content_str_survives_markdown_rendering(content, expected):
    assert str(content) == expected
    rendered = expected[1:-1]
    assert rendered in render_console_markdown(str(content))
    assert rendered in render_notebook_markdown(str(content))


@pytest.mark.parametrize(
    "content, expected",
    [
        (
            ContentPDF(data=b"x", filename="x`\n# heading"),
            "``[PDF x`\\n# heading · 1 B]``",
        ),
        (
            ContentDocument(
                data=b"x",
                filename="x`\n# heading",
                mime_type="text/csv",
            ),
            "``[document x`\\n# heading · text/csv]``",
        ),
        (
            ContentUploaded(
                id="file-`abc\n# heading",
                mime_type="image/png",
                provider="openai",
            ),
            "``[uploaded file-`abc\\n# heading · image/png]``",
        ),
    ],
)
def test_file_content_with_backticks_and_newlines_stays_code(
    content, expected
):
    assert str(content) == expected

    console_out = render_console_markdown(str(content))
    assert "heading" in console_out

    notebook_out = render_notebook_markdown(str(content))
    assert "<h1>" not in notebook_out
    assert "<code>" in notebook_out
    assert r"\n# heading" in notebook_out


def render_console_markdown(md: str) -> str:
    """Render markdown the way the console echo display does."""
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console(width=80)
    with console.capture() as cap:
        console.print(Markdown(md))
    return cap.get().strip()


def render_notebook_markdown(md: str) -> str:
    """Render markdown the way a notebook front-end does (CommonMark)."""
    from markdown_it import MarkdownIt

    # The notebook display wraps each content in blank lines, which is what makes
    # a bare `[label]: url` a block-level link reference definition.
    return MarkdownIt("commonmark").render(f"\n\n{md}\n\n").strip()


WEB_CONTENT_CASES = [
    (
        ContentToolRequestSearch(query="ggplot2 release date"),
        "web search request",
        "ggplot2 release date",
    ),
    (
        ContentToolResponseSearch(
            sources=[WebSource(url="https://example.com/a", title="Alpha")]
        ),
        "web search results",
        "https://example.com/a",
    ),
    (
        ContentToolRequestFetch(url="https://example.com/page"),
        "web fetch request",
        "https://example.com/page",
    ),
    (
        ContentToolResponseFetch(url="https://example.com/page", status="success"),
        "web fetch result",
        "https://example.com/page",
    ),
    (
        ContentCitation(source=WebSource(url="https://example.com/cite")),
        "citation",
        "https://example.com/cite",
    ),
]


@pytest.mark.parametrize("content,label,detail", WEB_CONTENT_CASES)
def test_web_content_survives_markdown_rendering(content, label, detail):
    """
    Regression test for the link-reference-definition bug.

    `[label]: <url>` is valid CommonMark link-reference syntax, so a renderer
    consumes the line and emits nothing. Every built-in content type has to
    survive both renderers chatlas uses.
    """
    md = str(content)

    console_out = render_console_markdown(md)
    assert label in console_out, f"{type(content).__name__} vanished in the console"
    assert detail in console_out

    notebook_out = render_notebook_markdown(md)
    assert label in notebook_out, f"{type(content).__name__} vanished in a notebook"
    assert detail in notebook_out


@pytest.mark.parametrize("content,label,detail", WEB_CONTENT_CASES)
def test_web_content_str_has_no_link_reference_prefix(content, label, detail):
    """The `[label]:` form is the bug; assert it can't come back."""
    assert not str(content).startswith("[")
