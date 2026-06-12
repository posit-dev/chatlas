import pytest
from chatlas import ChatOpenAI
from chatlas._content import (
    Citation,
    ContentText,
    ContentToolResponseFetch,
    ContentToolResponseSearch,
    Source,
    create_content,
)


def test_invalid_inputs_give_useful_errors():
    chat = ChatOpenAI()

    with pytest.raises(TypeError):
        chat.chat(question="Are unicorns real?")  # type: ignore

    with pytest.raises(ValueError):
        chat.chat(True)  # type: ignore


def test_citation_defaults():
    c = Citation(url="https://python.org")
    assert c.url == "https://python.org"
    assert c.title is None
    assert c.cited_text is None


def test_citation_has_no_offsets():
    c = Citation(url="https://a.com", title="A", cited_text="span")
    assert c.cited_text == "span"
    assert not hasattr(c, "start_index")
    assert not hasattr(c, "end_index")


def test_source_defaults():
    s = Source(url="https://python.org")
    assert s.url == "https://python.org"
    assert s.title is None
    assert s.domain is None


def test_contenttext_citations_default_empty():
    t = ContentText(text="hello")
    assert t.citations == []


def test_contenttext_with_citations():
    t = ContentText(
        text="Python 3.14 is the latest release.",
        citations=[Citation(url="https://docs.python.org", title="docs", cited_text="Python 3.14")],
    )
    assert len(t.citations) == 1
    assert t.citations[0].cited_text == "Python 3.14"


def test_search_results_use_sources():
    r = ContentToolResponseSearch(
        sources=[Source(url="https://python.org", title="Python", domain="python.org")]
    )
    assert r.sources[0].domain == "python.org"
    assert "python.org" in str(r)
    assert not hasattr(r, "urls")


def test_fetch_results_status():
    r = ContentToolResponseFetch(url="https://python.org", status="success")
    assert r.status == "success"
    assert ContentToolResponseFetch(url="https://x.com").status is None


def test_search_results_roundtrip():
    r = ContentToolResponseSearch(sources=[Source(url="https://a.com", title="A")])
    restored = create_content(r.model_dump())
    assert isinstance(restored, ContentToolResponseSearch)
    assert restored.sources[0].url == "https://a.com"


def test_citation_source_exported_from_types():
    from chatlas.types import Citation, Source  # noqa: F401
    import chatlas
    # Not re-exported at top level (per design decision)
    assert not hasattr(chatlas, "Citation")


def test_contenttext_add_merges_citations():
    a = ContentText(text="foo", citations=[Citation(url="https://a.com")])
    b = ContentText(text="bar", citations=[Citation(url="https://b.com")])
    merged = a + b
    assert merged.text == "foobar"
    assert [c.url for c in merged.citations] == ["https://a.com", "https://b.com"]


def test_content_citation_wraps_citation():
    from chatlas._content import Citation, ContentCitation
    c = ContentCitation(citation=Citation(url="https://a.com", title="A", cited_text="snippet"))
    assert c.citation.url == "https://a.com"
    assert c.content_type == "citation"
    assert "https://a.com" in str(c)


def test_content_citation_roundtrip():
    from chatlas._content import Citation, ContentCitation, create_content
    c = ContentCitation(citation=Citation(url="https://a.com", cited_text="foo"))
    restored = create_content(c.model_dump())
    assert isinstance(restored, ContentCitation)
    assert restored.citation.cited_text == "foo"


def test_content_citation_exported_from_types():
    from chatlas.types import ContentCitation  # noqa: F401
