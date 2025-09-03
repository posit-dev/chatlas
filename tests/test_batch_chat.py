import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from chatlas import ChatOpenAI
from chatlas._batch_chat import (
    BatchJob,
    batch_chat,
    batch_chat_completed,
    batch_chat_structured,
    batch_chat_text,
)
from pydantic import BaseModel


class StateCapital(BaseModel):
    name: str


def test_batch_job_initialization():
    """Test BatchJob initialization."""
    chat = ChatOpenAI()
    prompts = ["What's the capital of France?", "What's the capital of Germany?"]

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "batch.json"

        job = BatchJob(chat, prompts, path, wait=False)
        assert job.stage == "submitting"
        assert len(job.user_turns) == 2
        assert job.should_wait is False


def test_batch_job_state_persistence():
    """Test that batch job state is saved and loaded correctly."""
    chat = ChatOpenAI()
    prompts = ["What's the capital of France?"]

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "batch.json"

        # Create initial job and save state
        job1 = BatchJob(chat, prompts, path, wait=False)
        job1.stage = "waiting"
        job1.batch = {"id": "batch_123", "status": "in_progress"}
        job1._save_state()

        # Create new job from same path - should load state
        job2 = BatchJob(chat, prompts, path, wait=False)
        assert job2.stage == "waiting"
        assert job2.batch["id"] == "batch_123"


def test_batch_job_hash_validation():
    """Test that hash validation prevents mismatched reuse."""
    chat = ChatOpenAI()
    prompts = ["What's the capital of France?"]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        # Create and save job
        job1 = BatchJob(chat, prompts, path, wait=False)
        job1._save_state()

        # Try to load with different prompts - should fail
        different_prompts = ["What's the capital of Italy?"]
        with pytest.raises(ValueError, match="doesn't match stored value"):
            BatchJob(chat, different_prompts, path, wait=False)

    finally:
        Path(path).unlink(missing_ok=True)


@patch("chatlas._provider_openai.OpenAIProvider.batch_submit")
@patch("chatlas._provider_openai.OpenAIProvider.batch_poll")
@patch("chatlas._provider_openai.OpenAIProvider.batch_status")
@patch("chatlas._provider_openai.OpenAIProvider.batch_retrieve")
@patch("chatlas._provider_openai.OpenAIProvider.batch_result_turn")
def test_batch_chat_mocked_openai(
    mock_result_turn, mock_retrieve, mock_status, mock_poll, mock_submit
):
    mock_submit.return_value = {"id": "batch_123", "status": "validating"}
    mock_poll.return_value = {"id": "batch_123", "status": "completed"}
    mock_status.return_value = {
        "working": False,
        "n_processing": 0,
        "n_succeeded": 2,
        "n_failed": 0,
    }
    mock_retrieve.return_value = [
        {"custom_id": "request-0", "response": {"body": {"id": "msg1"}}},
        {"custom_id": "request-1", "response": {"body": {"id": "msg2"}}},
    ]
    mock_result_turn.side_effect = lambda result, has_data_model: Mock(
        text=f"Response for {result['custom_id']}"
    )

    chat = ChatOpenAI(model="gpt-4o-mini", api_key="test-key")
    prompts = ["What's the capital of France?", "What's the capital of Germany?"]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        chats = batch_chat(chat, prompts, path)

        assert chats is not None
        assert len(chats) == 2
        assert all(c is not None for c in chats)

        # Verify mocks were called
        mock_submit.assert_called_once()
        mock_poll.assert_called_once()
        mock_status.assert_called_once()
        mock_retrieve.assert_called_once()
        assert mock_result_turn.call_count == 2

    finally:
        Path(path).unlink(missing_ok=True)


@patch("chatlas._provider_openai.OpenAIProvider.batch_submit")
@patch("chatlas._provider_openai.OpenAIProvider.batch_poll")
@patch("chatlas._provider_openai.OpenAIProvider.batch_status")
@patch("chatlas._provider_openai.OpenAIProvider.batch_retrieve")
@patch("chatlas._provider_openai.OpenAIProvider.batch_result_turn")
def test_batch_chat_text_mocked(
    mock_result_turn, mock_retrieve, mock_status, mock_poll, mock_submit
):
    """Test batch_chat_text with mocked provider."""
    # Setup mocks
    mock_submit.return_value = {"id": "batch_123"}
    mock_poll.return_value = {"id": "batch_123", "status": "completed"}
    mock_status.return_value = {
        "working": False,
        "n_processing": 0,
        "n_succeeded": 2,
        "n_failed": 0,
    }
    mock_retrieve.return_value = [
        {"custom_id": "request-0", "response": {"body": {}}},
        {"custom_id": "request-1", "response": {"body": {}}},
    ]

    # Mock turns with text attribute and role
    def create_mock_turn(text):
        turn = Mock()
        turn.text = text
        turn.role = "assistant"
        return turn

    turn1 = create_mock_turn("Paris")
    turn2 = create_mock_turn("Berlin")
    mock_result_turn.side_effect = [turn1, turn2]

    chat = ChatOpenAI(model="gpt-4o-mini", api_key="test-key")
    prompts = ["What's the capital of France?", "What's the capital of Germany?"]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        texts = batch_chat_text(chat, prompts, path)

        assert texts is not None
        assert texts == ["Paris", "Berlin"]

    finally:
        Path(path).unlink(missing_ok=True)


def test_batch_chat_wait_false():
    """Test batch_chat with wait=False when batch is not complete."""
    chat = ChatOpenAI(model="gpt-4o-mini", api_key="test-key")

    with (
        patch.object(chat.provider, "batch_submit") as mock_submit,
        patch.object(chat.provider, "batch_poll") as mock_poll,
        patch.object(chat.provider, "batch_status") as mock_status,
    ):
        mock_submit.return_value = {"id": "batch_123"}
        mock_poll.return_value = {"id": "batch_123", "status": "in_progress"}
        mock_status.return_value = {
            "working": True,
            "n_processing": 2,
            "n_succeeded": 0,
            "n_failed": 0,
        }

        prompts = ["What's the capital of France?", "What's the capital of Germany?"]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            result = batch_chat(chat, prompts, path, wait=False)
            assert result is None  # Should return None when not complete and wait=False

        finally:
            Path(path).unlink(missing_ok=True)


def test_batch_chat_completed():
    """Test batch_chat_completed function."""
    chat = ChatOpenAI(model="gpt-4o-mini", api_key="test-key")
    prompts = ["What's the capital of France?"]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        # Test with submitting stage
        job = BatchJob(chat, prompts, path, wait=False)
        job._save_state()
        assert batch_chat_completed(chat, prompts, path) is False

        # Test with done stage
        job.stage = "done"
        job._save_state()
        assert batch_chat_completed(chat, prompts, path) is True

    finally:
        Path(path).unlink(missing_ok=True)


def test_batch_chat_unsupported_provider():
    """Test that unsupported providers raise appropriate errors."""
    from chatlas._provider import Provider

    # Create a mock provider that doesn't support batch
    class MockUnsupportedProvider(Provider):
        def __init__(self):
            super().__init__(name="Mock", model="mock-model")

        def has_batch_support(self) -> bool:
            return False

        # Stub out required abstract methods
        def chat_perform(self, *args, **kwargs):
            pass

        def chat_perform_async(self, *args, **kwargs):
            pass

        def stream_text(self, *args, **kwargs):
            pass

        def stream_text_async(self, *args, **kwargs):
            pass

        def stream_turn(self, *args, **kwargs):
            pass

        def value_turn(self, *args, **kwargs):
            pass

        def stream_merge_chunks(self, *args, **kwargs):
            pass

        def token_count(self, *args, **kwargs):
            return 0

        def token_count_async(self, *args, **kwargs):
            return 0

        def translate_model_params(self, *args, **kwargs):
            return {}

        def supported_model_params(self, *args, **kwargs):
            return set()

    chat = Mock()
    chat.provider = MockUnsupportedProvider()

    prompts = ["What's the capital of France?"]

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "batch.json"

        with pytest.raises(ValueError, match="not supported by this provider"):
            BatchJob(chat, prompts, path)


def test_batch_chat_structured_data():
    """Test batch_chat_structured with structured data model."""
    chat = ChatOpenAI(model="gpt-4o-mini", api_key="test-key")

    with (
        patch.object(chat.provider, "batch_submit") as mock_submit,
        patch.object(chat.provider, "batch_poll") as mock_poll,
        patch.object(chat.provider, "batch_status") as mock_status,
        patch.object(chat.provider, "batch_retrieve") as mock_retrieve,
    ):
        mock_submit.return_value = {"id": "batch_123"}
        mock_poll.return_value = {"id": "batch_123", "status": "completed"}
        mock_status.return_value = {
            "working": False,
            "n_processing": 0,
            "n_succeeded": 1,
            "n_failed": 0,
        }
        mock_retrieve.return_value = [
            {"custom_id": "request-0", "response": {"name": "Paris"}}
        ]

        prompts = ["What's the capital of France?"]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            results = batch_chat_structured(chat, prompts, path, StateCapital)

            assert results is not None
            assert len(results) == 1
            assert results[0].name == "Paris"

        finally:
            Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])
