from chatlas import ChatOpenAI
from chatlas._turn import AssistantTurn, UserTurn
from chatlas.types import ContentText, ContentToolRequest, ContentToolResult


def get_date() -> str:
    """Get today's date."""
    return "2001-02-09"


class TestDanglingToolRequests:
    def test_dangling_tool_requests_inserted_into_user_message(self):
        chat = ChatOpenAI()

        # Mock _chat_impl to capture the turn that gets submitted
        submitted_turn = None

        def mock_chat_impl(turn, **kwargs):
            nonlocal submitted_turn
            submitted_turn = turn
            yield "Assistant response"

        chat._chat_impl = mock_chat_impl  # pyright: ignore[reportAttributeAccessIssue]

        # Simulate a broken chat history with dangling tool request
        tool_request = ContentToolRequest(id="1", name="get_date", arguments={})
        chat.set_turns(
            [
                UserTurn("What year is it today?"),
                AssistantTurn([tool_request]),
            ]
        )

        # Call chat with new input
        chat.chat("try again")

        # Verify the submitted turn has both the tool result and the new text
        assert submitted_turn is not None
        assert len(submitted_turn.contents) == 2

        # First content should be the tool result
        assert isinstance(submitted_turn.contents[0], ContentToolResult)
        assert submitted_turn.contents[0].request == tool_request
        assert submitted_turn.contents[0].error is not None
        assert "Chat ended before the tool could be invoked" in str(
            submitted_turn.contents[0].error
        )

        # Second content should be the new text
        assert isinstance(submitted_turn.contents[1], ContentText)
        assert submitted_turn.contents[1].text == "try again"

    def test_can_resume_chat_after_dangling_tool_requests(self):
        chat = ChatOpenAI(system_prompt="Be terse and use tool results over your internal knowledge.")
        chat.register_tool(get_date)

        # Simulate a broken chat history with dangling tool request
        tool_request = ContentToolRequest(id="1", name="get_date", arguments={})
        chat.set_turns(
            [
                UserTurn("What year is it today?"),
                AssistantTurn([tool_request]),
            ]
        )

        # Resume chat and verify we get an answer, not an error
        response = str(chat.chat("try again"))
        # Should mention 2001 since our mocked get_date returns "2001-02-09"
        assert "2001" in response or "date" in response.lower()

    def test_multiple_dangling_tool_requests(self):
        chat = ChatOpenAI()

        submitted_turn = None

        def mock_chat_impl(turn, **kwargs):
            nonlocal submitted_turn
            submitted_turn = turn
            yield "Response"

        chat._chat_impl = mock_chat_impl  # pyright: ignore[reportAttributeAccessIssue]

        # Simulate multiple dangling tool requests
        tool_request_1 = ContentToolRequest(id="1", name="get_date", arguments={})
        tool_request_2 = ContentToolRequest(id="2", name="get_time", arguments={})
        chat.set_turns(
            [
                UserTurn("What's the date and time?"),
                AssistantTurn([tool_request_1, tool_request_2]),
            ]
        )

        chat.chat("try again")

        assert submitted_turn is not None
        assert len(submitted_turn.contents) == 3

        assert isinstance(submitted_turn.contents[0], ContentToolResult)
        assert submitted_turn.contents[0].request == tool_request_1
        assert isinstance(submitted_turn.contents[1], ContentToolResult)
        assert submitted_turn.contents[1].request == tool_request_2

        assert isinstance(submitted_turn.contents[2], ContentText)
        assert submitted_turn.contents[2].text == "try again"
