from chatlas import ChatAnthropic, ChatGoogle, ChatOpenAI, Turn
from chatlas._openai import OpenAIAzureProvider, OpenAIProvider
from chatlas._tokens import token_usage, tokens_log, tokens_reset


def test_tokens_method():
    chat = ChatOpenAI()
    assert chat.tokens(values="discrete") == []

    chat = ChatOpenAI(
        turns=[
            Turn(role="user", contents="Hi"),
            Turn(role="assistant", contents="Hello", tokens=(2, 10)),
        ]
    )

    assert chat.tokens(values="discrete") == [2, 10]

    chat = ChatOpenAI(
        turns=[
            Turn(role="user", contents="Hi"),
            Turn(role="assistant", contents="Hello", tokens=(2, 10)),
            Turn(role="user", contents="Hi"),
            Turn(role="assistant", contents="Hello", tokens=(14, 10)),
        ]
    )

    assert chat.tokens(values="discrete") == [2, 10, 2, 10]
    assert chat.tokens(values="cumulative") == [None, (2, 10), None, (14, 10)]


def test_token_count_method():
    chat = ChatOpenAI(model="gpt-4o-mini")
    assert chat.token_count("What is 1 + 1?") == 31

    chat = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    assert chat.token_count("What is 1 + 1?") == 16

    chat = ChatGoogle(model="gemini-1.5-flash")
    assert chat.token_count("What is 1 + 1?") == 9


def test_usage_is_none():
    tokens_reset()
    assert token_usage() is None


def test_can_retrieve_and_log_tokens():
    tokens_reset()

    provider = OpenAIProvider(model="foo")

    tokens_log(provider, (10, 50))
    tokens_log(provider, (0, 10))
    usage = token_usage()
    assert usage is not None
    assert len(usage) == 1
    assert usage[0]["name"] == "OpenAI"
    assert usage[0]["input"] == 10
    assert usage[0]["output"] == 60

    provider2 = OpenAIAzureProvider(endpoint="foo", api_version="bar")

    tokens_log(provider2, (5, 25))
    usage = token_usage()
    assert usage is not None
    assert len(usage) == 2
    assert usage[1]["name"] == "OpenAIAzure"
    assert usage[1]["input"] == 5
    assert usage[1]["output"] == 25

    tokens_reset()
