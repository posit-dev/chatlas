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
