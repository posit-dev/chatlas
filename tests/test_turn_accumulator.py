from chatlas._content import ContentText, ContentThinking, ContentToolRequest
from chatlas._turn import AssistantTurn, UserTurn
from chatlas._turn_accumulator import TurnAccumulator
from chatlas._stream_controller import StreamController


def test_begin_turn_inserts_partial():
    turns: list = []
    controller = StreamController()
    acc = TurnAccumulator(turns, controller)
    user = UserTurn("hello")
    acc.begin_turn(user)
    assert len(turns) == 2
    assert turns[0] is user
    assert isinstance(turns[1], AssistantTurn)
    assert turns[1].is_partial


def test_update_turn_appends_content():
    turns: list = []
    controller = StreamController()
    acc = TurnAccumulator(turns, controller)
    acc.begin_turn(UserTurn("hello"))
    content = ContentText.model_construct(text="hi")
    acc.update_turn(content)
    assert len(turns[1].contents) == 1
    assert turns[1].contents[0].text == "hi"


def test_complete_turn_replaces_partial():
    turns: list = []
    controller = StreamController()
    acc = TurnAccumulator(turns, controller)
    acc.begin_turn(UserTurn("hello"))
    full_turn = AssistantTurn("response", tokens=(10, 5, 0))
    acc.complete_turn(full_turn)
    assert turns[1] is full_turn
    assert not turns[1].is_partial


def test_complete_turn_skipped_when_cancelled():
    turns: list = []
    controller = StreamController()
    acc = TurnAccumulator(turns, controller)
    acc.begin_turn(UserTurn("hello"))
    controller.cancel()
    full_turn = AssistantTurn("response", tokens=(10, 5, 0))
    acc.complete_turn(full_turn)
    assert turns[1].is_partial


def test_finalize_turn_merges_text_and_sets_reason():
    turns: list = []
    controller = StreamController()
    acc = TurnAccumulator(turns, controller)
    acc.begin_turn(UserTurn("hello"))
    acc.update_turn(ContentText.model_construct(text="a"))
    acc.update_turn(ContentText.model_construct(text="b"))
    acc.finalize_turn()
    assert turns[1].is_partial
    assert turns[1].partial_reason == "interrupted"
    assert len(turns[1].contents) == 1
    assert turns[1].contents[0].text == "ab"


def test_finalize_turn_uses_controller_reason():
    turns: list = []
    controller = StreamController()
    acc = TurnAccumulator(turns, controller)
    acc.begin_turn(UserTurn("hello"))
    acc.update_turn(ContentText.model_construct(text="partial"))
    controller.cancel(reason="user stopped")
    acc.finalize_turn()
    assert turns[1].partial_reason == "user stopped"


def test_finalize_turn_noops_after_complete():
    turns: list = []
    controller = StreamController()
    acc = TurnAccumulator(turns, controller)
    acc.begin_turn(UserTurn("hello"))
    full_turn = AssistantTurn("response", tokens=(10, 5, 0))
    acc.complete_turn(full_turn)
    acc.finalize_turn()
    assert turns[1] is full_turn
    assert not turns[1].is_partial
