import pytest
from chatlas import ChatOpenAI
from chatlas._content import (
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


def test_source_defaults():
    s = Source(url="https://python.org")
    assert s.url == "https://python.org"
    assert s.title is None
    assert s.domain is None


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


def test_source_exported_from_types():
    from chatlas.types import Source  # noqa: F401
    import chatlas
    assert not hasattr(chatlas, "Citation")


def test_content_citation_fields():
    from chatlas._content import ContentCitation
    c = ContentCitation(url="https://python.org", title="Python")
    assert c.url == "https://python.org"
    assert c.title == "Python"
    assert c.content_type == "citation"
    assert "https://python.org" in str(c)


def test_content_citation_title_optional():
    from chatlas._content import ContentCitation
    c = ContentCitation(url="https://a.com")
    assert c.title is None


def test_content_citation_roundtrip():
    from chatlas._content import ContentCitation, create_content
    c = ContentCitation(url="https://a.com", title="A")
    restored = create_content(c.model_dump())
    assert isinstance(restored, ContentCitation)
    assert restored.url == "https://a.com"
    assert restored.title == "A"


def test_content_citation_exported_from_types():
    from chatlas.types import ContentCitation  # noqa: F401


def test_citation_class_not_exported():
    import chatlas.types
    assert not hasattr(chatlas.types, "Citation")


def test_contenttext_has_no_citations_field():
    from chatlas._content import ContentText
    t = ContentText(text="hello")
    assert not hasattr(t, "citations")


def test_contenttext_add_no_citations():
    from chatlas._content import ContentText
    a = ContentText(text="foo")
    b = ContentText(text="bar")
    merged = a + b
    assert merged.text == "foobar"
