import pytest
from chatlas import ChatOpenAI


def test_invalid_inputs_give_useful_errors():
    chat = ChatOpenAI()

    with pytest.raises(TypeError):
        chat.chat(question="Are unicorns real?")  # type: ignore

    with pytest.raises(ValueError):
        chat.chat(True)  # type: ignore
