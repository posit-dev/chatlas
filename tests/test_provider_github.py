import pytest
from chatlas import ChatGithub


def test_github_is_defunct():
    with pytest.raises(RuntimeError, match="GitHub Models was retired"):
        ChatGithub()
