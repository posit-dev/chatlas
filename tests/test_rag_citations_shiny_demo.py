import runpy
from pathlib import Path


def test_rag_citations_shiny_demo_imports_without_openai_api_key(
    monkeypatch,
):
    root = Path(__file__).parents[1]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    module = runpy.run_path(str(root / "scripts" / "rag_citations_shiny_demo.py"))

    assert module["app"] is not None
