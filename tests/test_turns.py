import warnings
from typing import Literal

import pytest

from chatlas import ChatAnthropic
from chatlas._chat import finalize_assistant_turn
from chatlas._content import ToolInfo
from chatlas._turn import (
    AssistantTurn,
    SystemTurn,
    Turn,
    UserTurn,
    check_finish_reason,
    normalize_turn_for_provider,
)
from chatlas.types import (
    ContentImageInline,
    ContentJson,
    ContentPDF,
    ContentText,
    ContentToolRequest,
    ContentToolResult,
)


def test_system_prompt_applied_correctly():
    sys_prompt = "foo"
    user_msg = UserTurn("bar")

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
        SystemTurn(sys_prompt),
        user_msg,
    ]

    chat2.set_turns([user_msg])
    assert len(chat2.get_turns()) == 1
    assert len(chat2.get_turns(include_system_prompt=True)) == 2

    chat2.set_turns([])
    assert chat2.get_turns() == []
    assert chat2.get_turns(include_system_prompt=True) == [SystemTurn(sys_prompt)]


def test_add_turn_system_role_error():
    sys_msg = SystemTurn("foo")
    chat = ChatAnthropic()

    with pytest.raises(
        ValueError, match="Turns with the role 'system' are not allowed"
    ):
        chat.add_turn(sys_msg)


def test_set_turns_functionality():
    user_msg1 = UserTurn("hello")
    assistant_msg = AssistantTurn("hi there")
    user_msg2 = UserTurn("how are you?")

    chat = ChatAnthropic()

    # Test setting turns
    turns = [user_msg1, assistant_msg, user_msg2]
    chat.set_turns(turns)
    assert chat.get_turns() == turns

    # Test that system turns in set_turns raise error
    sys_msg = SystemTurn("foo")
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
    turn = AssistantTurn(
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
    user_turn = UserTurn("What is 1 + 2?")
    assistant_turn = AssistantTurn([ContentText(text="I'll calculate that."), tool_request])
    tool_result_turn = UserTurn([tool_result])
    final_assistant_turn = AssistantTurn("The answer is 3.")
    
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
    user_turn = UserTurn("What is 1 + 2?")
    assistant_turn = AssistantTurn([ContentText(text="I'll calculate that."), tool_request])
    tool_result_turn = UserTurn([tool_result])
    final_assistant_turn = AssistantTurn("The answer is 3.")
    
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


def test_get_turns_rich_tool_result_role_assistant():
    request = ContentToolRequest(id="plot-1", name="plot", arguments={})
    image = ContentImageInline(
        data="aGVsbG8=",
        image_content_type="image/png",
    )
    result = ContentToolResult(
        value=[ContentText(text="Chart displayed."), image],
        model_format="as_is",
        request=request,
    )
    result_turn = UserTurn([result])

    assert result_turn.contents == [result]

    chat = ChatAnthropic()
    chat.set_turns(
        [
            UserTurn("plot the data"),
            AssistantTurn([request]),
            result_turn,
            AssistantTurn("The chart is ready."),
        ]
    )

    turns = chat.get_turns(tool_result_role="assistant")
    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert chat.get_turns()[1].contents == [request]


def test_rich_tool_result_json_round_trip_preserves_content_types():
    request = ContentToolRequest(id="plot-1", name="plot", arguments={})
    image = ContentImageInline(
        data="aGVsbG8=",
        image_content_type="image/png",
    )
    pdf = ContentPDF(data=b"pdf", filename="report.pdf")
    result = ContentToolResult(
        value=[ContentText(text="Chart displayed."), image, pdf],
        model_format="as_is",
        request=request,
    )

    restored = Turn.model_validate_json(UserTurn([result]).model_dump_json())
    restored_result = restored.contents[0]
    assert isinstance(restored_result, ContentToolResult)
    assert isinstance(restored_result.value[0], ContentText)
    assert isinstance(restored_result.value[1], ContentImageInline)
    assert isinstance(restored_result.value[2], ContentPDF)

    provider_turn = normalize_turn_for_provider(restored)
    assert any(isinstance(x, ContentImageInline) for x in provider_turn.contents)
    assert any(isinstance(x, ContentPDF) for x in provider_turn.contents)


def test_tool_result_content_type_mapping_remains_mapping():
    value = {"content_type": "text", "text": "ordinary tool data"}
    result = ContentToolResult(value=value, model_format="as_is")

    assert isinstance(result.value, dict)
    assert result.value == value

    restored = Turn.model_validate_json(UserTurn([result]).model_dump_json())
    restored_result = restored.contents[0]
    assert isinstance(restored_result, ContentToolResult)
    assert isinstance(restored_result.value, dict)
    assert restored_result.value == value


def test_tool_result_content_sentinel_mapping_remains_mapping():
    value = {"__chatlas_content__": "ordinary tool data"}
    result = ContentToolResult(value=value, model_format="as_is")

    assert result.value == value

    restored = Turn.model_validate_json(UserTurn([result]).model_dump_json())
    restored_result = restored.contents[0]
    assert isinstance(restored_result, ContentToolResult)
    assert restored_result.value == value


def test_rich_tool_result_json_round_trip_preserves_nested_mapping_content():
    image = ContentImageInline(
        data="aGVsbG8=",
        image_content_type="image/png",
    )
    result = ContentToolResult(value={"plot": image}, model_format="as_is")

    restored = Turn.model_validate_json(UserTurn([result]).model_dump_json())
    restored_result = restored.contents[0]
    assert isinstance(restored_result, ContentToolResult)
    assert isinstance(restored_result.value["plot"], ContentImageInline)


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
    user_turn = UserTurn("Do some math")
    assistant_turn1 = AssistantTurn("Let me calculate.")
    tool_result_turn1 = UserTurn([tool_result1])
    tool_result_turn2 = UserTurn([tool_result2]) 
    assistant_turn2 = AssistantTurn("Done with calculations.")
    
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
    user_turn = UserTurn("Check the weather")
    assistant_turn = AssistantTurn("I'll check that.")
    mixed_turn = UserTurn([ContentText(text="Here's additional info:"), tool_result])
    final_turn = AssistantTurn("The weather is nice!")
    
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
    user_turn = UserTurn("Do calculations")
    assistant_turn = AssistantTurn("I'll help.")
    pure_tool_turn = UserTurn([tool_result1, tool_result2])  # Pure tool results
    mixed_tool_turn = UserTurn([ContentText(text="Results:"), tool_result1])  # Mixed content
    final_turn = AssistantTurn("All done.")
    
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
    user_turn = UserTurn("Hello")
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
    
    user_turn = UserTurn("Calculate 2+2")
    assistant_turn = AssistantTurn("I'll calculate that.")
    tool_result_turn = UserTurn([tool_result])
    final_turn = AssistantTurn("The answer is 4.")
    
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


def test_assistant_turn_partial_reason():
    from chatlas._turn import AssistantTurn

    # Default: not partial
    turn = AssistantTurn("hello")
    assert turn.partial_reason is None
    assert turn.is_partial is False

    # Partial turn
    partial = AssistantTurn("partial", partial_reason="interrupted")
    assert partial.partial_reason == "interrupted"
    assert partial.is_partial is True

    # Custom reason
    cancelled = AssistantTurn("cancelled", partial_reason="cancelled")
    assert cancelled.partial_reason == "cancelled"
    assert cancelled.is_partial is True


def test_get_turns_tool_result_role_empty_chat():
    """Test tool_result_role with empty chat"""
    chat = ChatAnthropic()
    
    # Empty chat should return empty list regardless of tool_result_role
    assert chat.get_turns(tool_result_role="user") == []
    assert chat.get_turns(tool_result_role="assistant") == []


def test_get_turns_tool_result_role_no_tool_results():
    """Test tool_result_role with chat containing no tool results"""
    chat = ChatAnthropic()
    
    user_turn = UserTurn("Hello")
    assistant_turn = AssistantTurn("Hi there!")
    
    chat.set_turns([user_turn, assistant_turn])
    
    # Should behave identically regardless of tool_result_role when no tool results present
    turns_user = chat.get_turns(tool_result_role="user")
    turns_assistant = chat.get_turns(tool_result_role="assistant")
    
    assert len(turns_user) == 2
    assert len(turns_assistant) == 2
    assert turns_user[0].role == turns_assistant[0].role == "user"
    assert turns_user[1].role == turns_assistant[1].role == "assistant"


@pytest.mark.parametrize(
    "reason, match",
    [
        ("max_tokens", "max_tokens"),
        ("context_window", "context window"),
        ("content_filter", "content moderation"),
    ],
)
def test_check_finish_reason_errors_on_incomplete_response(reason: str, match: str):
    with pytest.raises(ValueError, match=match):
        check_finish_reason(reason, "error")


@pytest.mark.parametrize(
    "reason, match",
    [
        ("max_tokens", "max_tokens"),
        ("context_window", "context window"),
        ("content_filter", "content moderation"),
    ],
)
def test_check_finish_reason_warns_on_incomplete_response(reason: str, match: str):
    with pytest.warns(UserWarning, match=match):
        check_finish_reason(reason, "warn")


def test_check_finish_reason_error_suggests_increasing_max_tokens():
    # The whole point of gh-315: point the user at the knob they need to turn.
    with pytest.raises(ValueError, match="Increase"):
        check_finish_reason("max_tokens", "error")


@pytest.mark.parametrize(
    "reason", [None, "success", "tool_use", "stop_sequence", "some_new_reason"]
)
@pytest.mark.parametrize("signal", ["error", "warn"])
def test_check_finish_reason_ignores_complete_responses(
    reason: str | None, signal: Literal["error", "warn"]
):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_finish_reason(reason, signal) is None


def test_check_finish_reason_warning_points_at_caller():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        check_finish_reason("max_tokens", "warn")

    assert caught[0].filename == __file__


def test_check_finish_reason_warning_points_past_intervening_chatlas_frames():
    # Same warning, one frame deeper inside chatlas. The attributed frame must
    # still be this test, so the stacklevel can't be a fixed constant.
    turn = AssistantTurn("partial", finish_reason="max_tokens")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finalize_assistant_turn(ChatAnthropic().provider, turn)

    assert caught[0].filename == __file__
