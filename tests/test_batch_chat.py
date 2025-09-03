import tempfile

import pytest
from pydantic import BaseModel

from chatlas import ChatAnthropic, ChatGoogle, ChatOpenAI
from chatlas._batch_chat import (
    BatchJob,
    batch_chat,
    batch_chat_completed,
    batch_chat_text,
)
from chatlas._provider import BatchStatus


class CountryCapital(BaseModel):
    name: str


def test_can_retrieve_batch(test_batch_dir):
    chat = ChatOpenAI(model="gpt-4o-mini")
    prompts = ["What's the capital of France?", "What's the capital of Germany?"]

    chats = batch_chat(
        chat,
        prompts,
        test_batch_dir / "country-capitals.json",
    )
    assert len(chats) == 2
    assert chats[0] is not None
    assert chats[1] is not None

    out = batch_chat_text(
        chat,
        prompts,
        test_batch_dir / "country-capitals.json",
    )
    assert len(out) == 2
    assert out[0] is not None
    assert out[1] is not None
    assert "Paris" in out[0]
    assert "Berlin" in out[1]

    # TODO: incorporate structured output test
    # capitals = batch_chat_structured(
    #    chat,
    #    prompts,
    #    test_batch_dir / "country-capitals-structured.json",
    #    CountryCapital,
    # )
    # assert len(capitals) == 2
    # assert capitals[0] is not None
    # assert capitals[1] is not None
    # assert capitals[0].name == "Paris"
    # assert capitals[1].name == "Berlin"


def test_informative_errors(test_batch_dir):
    with pytest.raises(ValueError, match="not supported by this provider"):
        batch_chat(
            ChatGoogle(),
            [],
            "foo.json",
        )

    with pytest.raises(ValueError, match="provider doesn't match stored value"):
        batch_chat(
            ChatAnthropic(),
            [],
            test_batch_dir / "country-capitals.json",
        )

    with pytest.raises(ValueError, match="model doesn't match stored value"):
        batch_chat(
            ChatOpenAI(model="gpt-5"),
            [],
            test_batch_dir / "country-capitals.json",
        )

    with pytest.raises(ValueError, match="prompts doesn't match stored value"):
        batch_chat(
            ChatOpenAI(model="gpt-4o-mini"),
            ["foo"],
            test_batch_dir / "country-capitals.json",
        )

    with pytest.raises(ValueError, match="user_turns doesn't match stored value"):
        batch_chat(
            ChatOpenAI(model="gpt-4o-mini", system_prompt="foo"),
            [
                "What's the capital of France?",
                "What's the capital of Germany?",
            ],
            test_batch_dir / "country-capitals.json",
        )


def ChatOpenAIMockBatchSubmit(*, working: bool = False, **kwargs):
    chat = ChatOpenAI(**kwargs)

    def batch_submit_mock(*args, **kwargs):
        return {"id": "123"}

    def batch_poll_mock(*args, **kwargs):
        return {"id": "123", "results": True}

    def batch_status_mock(*args, **kwargs):
        return BatchStatus(
            working=working,
            n_processing=0,
            n_failed=0,
            n_succeeded=1,
        )

    def batch_retrieve_mock(*args, **kwargs):
        return [{"x": 1, "y": 2}]

    chat.provider.batch_submit = batch_submit_mock
    chat.provider.batch_poll = batch_poll_mock
    chat.provider.batch_status = batch_status_mock
    chat.provider.batch_retrieve = batch_retrieve_mock

    return chat


def test_steps_in_logical_order():
    with tempfile.NamedTemporaryFile() as temp_file:
        chat = ChatOpenAIMockBatchSubmit()
        prompts = ["What's your name?"]
        job = BatchJob(chat, prompts, temp_file.name)

        def completed():
            return batch_chat_completed(chat, prompts, temp_file.name)

        assert job.stage == "submitting"
        assert not completed()

        job.step()
        assert job.stage == "waiting"
        job._load_state()
        assert job.stage == "waiting"
        assert job.batch == {"id": "123"}
        assert completed()

        job.step()
        assert job.stage == "retrieving"
        job._load_state()
        assert job.stage == "retrieving"
        assert job.batch == {"id": "123", "results": True}
        assert completed()

        job.step()
        assert job.stage == "done"
        job._load_state()
        assert job.stage == "done"
        assert job.results == [{"x": 1, "y": 2}]
        assert completed()


def test_run_all_steps_at_once():
    with tempfile.NamedTemporaryFile() as temp_file:
        chat = ChatOpenAIMockBatchSubmit()
        prompts = ["What's your name?"]
        job = BatchJob(chat, prompts, temp_file.name)

        job = job.step_until_done()
        assert job is not None
        assert job.stage == "done"
        assert job.results == [{"x": 1, "y": 2}]


def test_can_avoid_blocking():
    with tempfile.NamedTemporaryFile() as temp_file:
        chat = ChatOpenAIMockBatchSubmit(working=True)
        job = BatchJob(
            chat,
            ["What's your name?"],
            temp_file.name,
            wait=False,
        )

        assert job.step_until_done() is None
