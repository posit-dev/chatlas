import pytest

from chatlas import ChatAnthropic
from chatlas._turn import Turn
from chatlas.types import ContentJson, ContentText


def test_system_prompt_applied_correctly():
    sys_prompt = "foo"
    user_msg = Turn("user", "bar")

    # Test chat with no system prompt
    chat1 = ChatAnthropic()
    assert chat1.system_prompt is None
    assert chat1.get_turns() == []

    # Test chat with system prompt
    chat2 = ChatAnthropic(system_prompt=sys_prompt)
    assert chat2.system_prompt == sys_prompt
    assert chat2.get_turns() == []

    # Test adding turns to chat with system prompt
    chat2.add_turn(user_msg)
    assert chat2.get_turns() == [user_msg]
    assert chat2.get_turns(include_system_prompt=True) == [
        Turn("system", sys_prompt),
        user_msg,
    ]

    chat2.set_turns([user_msg])
    assert len(chat2.get_turns()) == 1
    assert len(chat2.get_turns(include_system_prompt=True)) == 2

    chat2.set_turns([])
    assert chat2.get_turns() == []
    assert chat2.get_turns(include_system_prompt=True) == [Turn("system", sys_prompt)]


def test_add_turn_system_role_error():
    sys_msg = Turn("system", "foo")
    chat = ChatAnthropic()

    with pytest.raises(
        ValueError, match="Turns with the role 'system' are not allowed"
    ):
        chat.add_turn(sys_msg)


def test_set_turns_functionality():
    user_msg1 = Turn("user", "hello")
    assistant_msg = Turn("assistant", "hi there")
    user_msg2 = Turn("user", "how are you?")

    chat = ChatAnthropic()

    # Test setting turns
    turns = [user_msg1, assistant_msg, user_msg2]
    chat.set_turns(turns)
    assert chat.get_turns() == turns

    # Test that system turns in set_turns raise error
    sys_msg = Turn("system", "foo")
    with pytest.raises(
        ValueError, match="Turn 0 has a role 'system', which is not allowed"
    ):
        chat.set_turns([sys_msg, user_msg1])


def test_system_prompt_property():
    chat = ChatAnthropic()

    # Test setting system prompt after creation
    chat.system_prompt = "be helpful"
    assert chat.system_prompt == "be helpful"
    assert len(chat.get_turns(include_system_prompt=True)) == 1

    # Test clearing system prompt
    chat.system_prompt = None
    assert chat.system_prompt is None
    assert len(chat.get_turns(include_system_prompt=True)) == 0


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
