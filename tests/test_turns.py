import pytest

from chatlas import ChatAnthropic
from chatlas._content import ToolInfo
from chatlas._turn import Turn
from chatlas.types import ContentJson, ContentText, ContentToolRequest, ContentToolResult


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


def test_get_turns_tool_result_role_default():
    """Test that tool results have default role of 'user'"""
    chat = ChatAnthropic()
    
    # Create mock tool request and result
    tool_request = ContentToolRequest(
        id="test-123",
        name="add",
        arguments={"x": 1, "y": 2},
    )
    tool_result = ContentToolResult(
        id="test-123", 
        name="add",
        arguments={"x": 1, "y": 2},
        value=3,
        request=tool_request
    )
    
    # Add turns with mixed content
    user_turn = Turn("user", "What is 1 + 2?")
    assistant_turn = Turn("assistant", [ContentText(text="I'll calculate that."), tool_request])
    tool_result_turn = Turn("user", [tool_result])
    final_assistant_turn = Turn("assistant", "The answer is 3.")
    
    chat.set_turns([user_turn, assistant_turn, tool_result_turn, final_assistant_turn])
    
    # Default behavior should keep tool result turn as "user"
    turns = chat.get_turns()
    assert len(turns) == 4
    assert turns[2].role == "user"  # Tool result turn remains as user
    assert all(isinstance(c, ContentToolResult) for c in turns[2].contents)


def test_get_turns_tool_result_role_assistant():
    """Test tool_result_role='assistant' changes tool result turn roles"""
    chat = ChatAnthropic()
    
    # Create mock tool request and result
    tool_request = ContentToolRequest(
        id="test-123",
        name="add", 
        arguments={"x": 1, "y": 2},
    )
    tool_result = ContentToolResult(
        id="test-123",
        name="add", 
        arguments={"x": 1, "y": 2},
        value=3,
        request=tool_request
    )
    
    # Add turns with mixed content
    user_turn = Turn("user", "What is 1 + 2?")
    assistant_turn = Turn("assistant", [ContentText(text="I'll calculate that."), tool_request])
    tool_result_turn = Turn("user", [tool_result])
    final_assistant_turn = Turn("assistant", "The answer is 3.")
    
    chat.set_turns([user_turn, assistant_turn, tool_result_turn, final_assistant_turn])
    
    # With tool_result_role="assistant", tool result turns should change to assistant
    turns = chat.get_turns(tool_result_role="assistant")
    assert len(turns) == 2  # Should be collapsed due to consecutive assistant turns
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"  # All assistant content collapsed
    
    # The second turn should now contain content from all assistant turns and tool result
    assert len(turns[1].contents) == 4  # Text + tool_request + tool_result + Text
    assert isinstance(turns[1].contents[0], ContentText)
    assert isinstance(turns[1].contents[1], ContentToolRequest) 
    assert isinstance(turns[1].contents[2], ContentToolResult)
    assert isinstance(turns[1].contents[3], ContentText)


def test_get_turns_tool_result_role_collapse_consecutive():
    """Test that consecutive assistant turns are collapsed when tool_result_role='assistant'"""
    chat = ChatAnthropic()
    
    # Create multiple tool results
    tool_result1 = ContentToolResult(
        id="test-1",
        name="add",
        arguments={"x": 1, "y": 2}, 
        value=3,
    )
    tool_result2 = ContentToolResult(
        id="test-2", 
        name="multiply",
        arguments={"x": 3, "y": 4},
        value=12,
    )
    
    # Create turns with multiple consecutive assistant turns via tool results
    user_turn = Turn("user", "Do some math")
    assistant_turn1 = Turn("assistant", "Let me calculate.")
    tool_result_turn1 = Turn("user", [tool_result1])
    tool_result_turn2 = Turn("user", [tool_result2]) 
    assistant_turn2 = Turn("assistant", "Done with calculations.")
    
    chat.set_turns([user_turn, assistant_turn1, tool_result_turn1, tool_result_turn2, assistant_turn2])
    
    # Default behavior - no collapsing
    turns_default = chat.get_turns()
    assert len(turns_default) == 5
    
    # With tool_result_role="assistant" - should collapse consecutive assistant turns
    turns_assistant = chat.get_turns(tool_result_role="assistant")
    assert len(turns_assistant) == 2
    assert turns_assistant[0].role == "user" 
    assert turns_assistant[1].role == "assistant"
    
    # Second turn should have content from all assistant turns and tool results
    assert len(turns_assistant[1].contents) == 4  # Text + tool_result1 + tool_result2 + Text
    assert isinstance(turns_assistant[1].contents[0], ContentText)
    assert isinstance(turns_assistant[1].contents[1], ContentToolResult) 
    assert isinstance(turns_assistant[1].contents[2], ContentToolResult)
    assert isinstance(turns_assistant[1].contents[3], ContentText)


def test_get_turns_tool_result_role_mixed_content():
    """Test tool_result_role with turns containing mixed content"""
    chat = ChatAnthropic()
    
    tool_result = ContentToolResult(
        id="test-123",
        name="weather",
        arguments={"city": "Boston"},
        value={"temp": 75, "condition": "sunny"},
    )
    
    # Turn with mixed content (text + tool result)
    user_turn = Turn("user", "Check the weather")
    assistant_turn = Turn("assistant", "I'll check that.")
    mixed_turn = Turn("user", [ContentText(text="Here's additional info:"), tool_result])
    final_turn = Turn("assistant", "The weather is nice!")
    
    chat.set_turns([user_turn, assistant_turn, mixed_turn, final_turn])
    
    # Default behavior - turn with mixed content stays as "user"
    turns_default = chat.get_turns()
    assert len(turns_default) == 4
    assert turns_default[2].role == "user"
    
    # With tool_result_role="assistant" - turn with mixed content should NOT change role
    # because it's not purely tool results
    turns_assistant = chat.get_turns(tool_result_role="assistant")
    assert len(turns_assistant) == 4
    assert turns_assistant[2].role == "user"  # Should remain user due to mixed content


def test_get_turns_tool_result_role_purely_tool_results():
    """Test that only turns with purely tool results change role"""
    chat = ChatAnthropic()
    
    tool_result1 = ContentToolResult(
        id="test-1",
        name="add",
        arguments={"x": 1, "y": 2},
        value=3,
    )
    tool_result2 = ContentToolResult(
        id="test-2", 
        name="multiply",
        arguments={"x": 3, "y": 4},
        value=12,
    )
    
    # Create different types of turns
    user_turn = Turn("user", "Do calculations")
    assistant_turn = Turn("assistant", "I'll help.")
    pure_tool_turn = Turn("user", [tool_result1, tool_result2])  # Pure tool results
    mixed_tool_turn = Turn("user", [ContentText(text="Results:"), tool_result1])  # Mixed content
    final_turn = Turn("assistant", "All done.")
    
    chat.set_turns([user_turn, assistant_turn, pure_tool_turn, mixed_tool_turn, final_turn])
    
    # With tool_result_role="assistant"
    turns = chat.get_turns(tool_result_role="assistant")
    
    # Pure tool turn should change role and be collapsed with assistant turns
    # Mixed tool turn should remain as user
    expected_roles = ["user", "assistant", "user", "assistant"] 
    actual_roles = [turn.role for turn in turns]
    assert actual_roles == expected_roles


def test_get_turns_tool_result_role_invalid_value():
    """Test that invalid tool_result_role values raise ValueError"""
    chat = ChatAnthropic()
    
    # Add some content so validation is triggered
    user_turn = Turn("user", "Hello")
    chat.set_turns([user_turn])
    
    with pytest.raises(ValueError, match="Expected `tool_result_role` to be one of 'user' or 'assistant', not 'invalid'"):
        chat.get_turns(tool_result_role="invalid")


def test_get_turns_tool_result_role_with_system_prompt():
    """Test tool_result_role works correctly with system prompt inclusion"""
    chat = ChatAnthropic(system_prompt="You are a helpful assistant.")
    
    tool_result = ContentToolResult(
        id="test-123",
        name="calculate",
        arguments={"expr": "2+2"},
        value=4,
    )
    
    user_turn = Turn("user", "Calculate 2+2")
    assistant_turn = Turn("assistant", "I'll calculate that.")
    tool_result_turn = Turn("user", [tool_result])
    final_turn = Turn("assistant", "The answer is 4.")
    
    chat.set_turns([user_turn, assistant_turn, tool_result_turn, final_turn])
    
    # Test with system prompt included
    turns_with_system = chat.get_turns(include_system_prompt=True, tool_result_role="assistant")
    assert len(turns_with_system) == 3  # system + user + collapsed assistant turns
    assert turns_with_system[0].role == "system"
    assert turns_with_system[1].role == "user"
    assert turns_with_system[2].role == "assistant"
    
    # Test without system prompt
    turns_without_system = chat.get_turns(include_system_prompt=False, tool_result_role="assistant") 
    assert len(turns_without_system) == 2  # user + collapsed assistant turns
    assert turns_without_system[0].role == "user"
    assert turns_without_system[1].role == "assistant"


def test_get_turns_tool_result_role_empty_chat():
    """Test tool_result_role with empty chat"""
    chat = ChatAnthropic()
    
    # Empty chat should return empty list regardless of tool_result_role
    assert chat.get_turns(tool_result_role="user") == []
    assert chat.get_turns(tool_result_role="assistant") == []


def test_get_turns_tool_result_role_no_tool_results():
    """Test tool_result_role with chat containing no tool results"""
    chat = ChatAnthropic()
    
    user_turn = Turn("user", "Hello")
    assistant_turn = Turn("assistant", "Hi there!")
    
    chat.set_turns([user_turn, assistant_turn])
    
    # Should behave identically regardless of tool_result_role when no tool results present
    turns_user = chat.get_turns(tool_result_role="user")
    turns_assistant = chat.get_turns(tool_result_role="assistant")
    
    assert len(turns_user) == 2
    assert len(turns_assistant) == 2
    assert turns_user[0].role == turns_assistant[0].role == "user"
    assert turns_user[1].role == turns_assistant[1].role == "assistant"
