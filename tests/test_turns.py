import pytest

from chatlas._turn import Turn, normalize_turns
from chatlas.types import ContentJson, ContentText


def test_system_prompt_applied_correctly():
    sys_prompt = "foo"
    sys_msg = Turn("system", sys_prompt)
    user_msg = Turn("user", "bar")

    assert normalize_turns([]) == []
    assert normalize_turns([user_msg]) == [user_msg]
    assert normalize_turns([sys_msg]) == [sys_msg]

    assert normalize_turns([], sys_prompt)[0] == sys_msg
    assert normalize_turns([user_msg], sys_prompt) == [sys_msg, user_msg]
    assert normalize_turns([sys_msg, user_msg], sys_prompt) == [sys_msg, user_msg]


def test_normalize_turns_errors():
    sys_msg = Turn("system", "foo")
    user_msg = Turn("user", "bar")

    with pytest.raises(ValueError, match="conflicting system prompts"):
        normalize_turns([sys_msg, user_msg], "foo2")


def test_can_extract_text_easily():
    turn = Turn(
        "assistant",
        [
            ContentText(text="ABC"),
            ContentJson(value=dict(a="1")),
            ContentText(text="DEF"),
        ],
    )
    assert turn.text == "ABCDEF"
