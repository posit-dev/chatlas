from chatlas._content import (
    ContentCitation,
    ContentText,
    ContentToolRequestSearch,
    ContentToolResponseSearch,
    Source,
)
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


def test_update_turn_merges_adjacent_content():
    turns: list = []
    controller = StreamController()
    acc = TurnAccumulator(turns, controller)
    acc.begin_turn(UserTurn("hello"))
    acc._update_turn(ContentText.model_construct(text="a"))
    acc._update_turn(ContentText.model_construct(text="b"))
    assert len(turns[1].contents) == 1
    assert turns[1].contents[0].text == "ab"


def test_update_turn_appends_non_mergeable_adjacent_content():
    # Regression: consecutive same-typed content without __add__ (e.g. two
    # web-search requests, or multiple citations) must append, not crash.
    turns: list = []
    acc = TurnAccumulator(turns, StreamController())
    acc.begin_turn(UserTurn("hello"))
    acc._update_turn(ContentToolRequestSearch(query="a"))
    acc._update_turn(ContentToolRequestSearch(query="b"))
    acc._update_turn(ContentCitation(url="https://a.com"))
    acc._update_turn(ContentCitation(url="https://b.com"))
    contents = turns[1].contents
    assert len(contents) == 4
    assert [type(c).__name__ for c in contents] == [
        "ContentToolRequestSearch",
        "ContentToolRequestSearch",
        "ContentCitation",
        "ContentCitation",
    ]


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


def test_finalize_turn_stamps_reason():
    turns: list = []
    controller = StreamController()
    acc = TurnAccumulator(turns, controller)
    acc.begin_turn(UserTurn("hello"))
    acc._update_turn(ContentText.model_construct(text="a"))
    acc._update_turn(ContentText.model_construct(text="b"))
    acc.finalize_turn()
    assert turns[1].is_partial
    assert turns[1].partial_reason == "interrupted"
    # Content was already merged inline by _update_turn
    assert len(turns[1].contents) == 1
    assert turns[1].contents[0].text == "ab"


def test_finalize_turn_uses_controller_reason():
    turns: list = []
    controller = StreamController()
    acc = TurnAccumulator(turns, controller)
    acc.begin_turn(UserTurn("hello"))
    acc._update_turn(ContentText.model_construct(text="partial"))
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


def _acc() -> TurnAccumulator:
    turns: list = []
    acc = TurnAccumulator(turns, StreamController())
    acc.begin_turn(UserTurn("hi"))
    return acc


def test_process_content_yields_citation_in_all_mode():
    acc = _acc()
    cit = ContentCitation(url="https://a.com")
    out = list(acc.process_content(cit, None, "all", lambda x: None))
    assert out == [cit]


def test_process_content_yields_search_results_in_all_mode():
    acc = _acc()
    res = ContentToolResponseSearch(sources=[Source(url="https://a.com")])
    out = list(acc.process_content(res, None, "all", lambda x: None))
    assert out == [res]


def test_process_content_text_mode_does_not_yield_citation():
    acc = _acc()
    cit = ContentCitation(url="https://a.com")
    out = list(acc.process_content(cit, None, "text", lambda x: None))
    assert out == []
