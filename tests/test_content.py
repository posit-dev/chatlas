import pytest
from chatlas import ChatOpenAI
from chatlas._content import (
    ContentCitation,
    ContentText,
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
    assert str(c) == "<uploaded file id=file_123 mime_type=application/pdf>"

    dumped = c.model_dump()
    restored = create_content(dumped)
    assert isinstance(restored, ContentUploaded)
    assert restored.id == "file_123"
    assert restored.provider == "openai"
