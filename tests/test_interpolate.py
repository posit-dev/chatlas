import tempfile
from pathlib import Path

from chatlas import interpolate, interpolate_file


def test_interpolate():
    x = 1  # noqa

    assert interpolate("{{ x }}") == "1"
    assert interpolate("{{ x }}", variables={"x": 2}) == "2"


def test_interpolate_file(tmp_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "prompt.txt"
        path.write_text("{{ x }}")

        x = 1  # noqa
        assert interpolate_file(path) == "1"


def test_interpolate_file_reads_as_utf8(tmp_path, monkeypatch):
    # `interpolate_file()` must read the prompt file as UTF-8 regardless of
    # the platform's default locale encoding, otherwise non-ASCII prompt
    # content is silently corrupted (mojibake) instead of raising.
    text = "Wie heißt die Hauptstadt? 中文测试 café {{ name }}"
    path = tmp_path / "prompt.txt"
    path.write_text(text, encoding="utf-8")

    real_open = open

    def open_using_locale_default(file, mode="r", *args, encoding=None, **kwargs):
        # Simulate a non-UTF-8 preferred locale encoding by falling back to
        # latin-1 whenever the caller doesn't pass an explicit encoding.
        if "b" not in mode and encoding is None:
            encoding = "latin-1"
        return real_open(file, mode, *args, encoding=encoding, **kwargs)

    monkeypatch.setattr(
        "chatlas._interpolate.open", open_using_locale_default, raising=False
    )

    result = interpolate_file(path, variables={"name": "x"})
    assert "Wie heißt die Hauptstadt? 中文测试 café" in result
