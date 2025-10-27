import datetime
import sys
from unittest.mock import patch

import pytest

from chatlas import Chat, ChatAnthropic, Turn

pytest.importorskip("inspect_ai")

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from inspect_ai.scorer import model_graded_qa

MODEL = "claude-haiku-4-5-20251001"
SCORER_MODEL = f"anthropic/{MODEL}"
SYSTEM_DEFAULT = "You are a helpful assistant that provides concise answers."


def chat_func(system_prompt: str | None = None) -> Chat:
    return ChatAnthropic(
        model=MODEL,
        system_prompt=system_prompt,
    )


def create_task(
    chat: Chat,
    dataset: list[Sample],
    include_system_prompt: bool = True,
    include_turns: bool = True,
) -> Task:
    return Task(
        dataset=dataset,
        solver=chat.to_solver(
            include_system_prompt=include_system_prompt,
            include_turns=include_turns,
        ),
        scorer=model_graded_qa(model=SCORER_MODEL),
    )


def test_inspect_dependency():
    chat = chat_func()
    with patch.dict(sys.modules, {"inspect_ai": None, "inspect_ai.model": None}):
        with pytest.raises(ImportError, match="pip install inspect-ai"):
            chat.to_solver()


# pytest.importorskip("non-existent-package", reason="Skipping to test ImportError")


class TestExportEval:
    def test_export_eval_basic(self, tmp_path):
        chat = chat_func(system_prompt=SYSTEM_DEFAULT)
        chat.set_turns(
            [
                Turn("user", "What is 2+2?"),
                Turn("assistant", "2 + 2 = 4"),
            ]
        )

        output_file = tmp_path / "test_eval.jsonl"
        result = chat.export_eval(output_file)

        assert result == output_file
        assert output_file.exists()

        dataset = json_dataset(str(output_file))
        samples = list(dataset)

        assert len(samples) == 1
        sample = samples[0]
        assert len(sample.input) == 2
        assert isinstance(sample.input[0], ChatMessageSystem)
        assert isinstance(sample.input[1], ChatMessageUser)
        assert str(sample.target) == "2 + 2 = 4"

    def test_export_eval_custom_target(self, tmp_path):
        chat = chat_func()
        chat.set_turns(
            [
                Turn("user", "Tell me a joke"),
                Turn(
                    "assistant",
                    "Why did the chicken cross the road? To get to the other side!",
                ),
            ]
        )

        output_file = tmp_path / "test_eval.jsonl"
        custom_target = "Response should be funny and appropriate"
        chat.export_eval(output_file, target=custom_target)

        dataset = json_dataset(str(output_file))
        samples = list(dataset)

        assert len(samples) == 1
        assert samples[0].target == custom_target

    def test_export_eval_append_mode(self, tmp_path):
        chat = chat_func()
        output_file = tmp_path / "test_eval.jsonl"

        chat.set_turns(
            [
                Turn("user", "What is 2+2?"),
                Turn("assistant", "2 + 2 = 4"),
            ]
        )
        chat.export_eval(output_file)

        chat.set_turns(
            [
                Turn("user", "What is 3+3?"),
                Turn("assistant", "3 + 3 = 6"),
            ]
        )
        chat.export_eval(output_file)

        dataset = json_dataset(str(output_file))
        samples = list(dataset)

        assert len(samples) == 2
        input_texts = [str(s.input) for s in samples]
        assert any("2+2" in text for text in input_texts)
        assert any("3+3" in text for text in input_texts)

    def test_export_eval_overwrite_true(self, tmp_path):
        chat = chat_func()
        output_file = tmp_path / "test_eval.jsonl"

        chat.set_turns(
            [
                Turn("user", "First question"),
                Turn("assistant", "First answer"),
            ]
        )
        chat.export_eval(output_file)

        chat.set_turns(
            [
                Turn("user", "Second question"),
                Turn("assistant", "Second answer"),
            ]
        )
        chat.export_eval(output_file, overwrite=True)

        dataset = json_dataset(str(output_file))
        samples = list(dataset)

        assert len(samples) == 1
        assert "Second question" in str(samples[0].input)

    def test_export_eval_overwrite_false_error(self, tmp_path):
        chat = chat_func()
        output_file = tmp_path / "test_eval.jsonl"

        chat.set_turns(
            [
                Turn("user", "First question"),
                Turn("assistant", "First answer"),
            ]
        )
        chat.export_eval(output_file)

        chat.set_turns(
            [
                Turn("user", "Second question"),
                Turn("assistant", "Second answer"),
            ]
        )
        with pytest.raises(ValueError, match="already exists"):
            chat.export_eval(output_file, overwrite=False)

    def test_export_eval_wrong_extension(self, tmp_path):
        chat = chat_func()
        chat.set_turns(
            [
                Turn("user", "What is 2+2?"),
                Turn("assistant", "2 + 2 = 4"),
            ]
        )

        output_file = tmp_path / "test_eval.csv"
        with pytest.raises(ValueError, match="must have a `.jsonl` extension"):
            chat.export_eval(output_file)

    def test_export_eval_no_assistant_turn(self, tmp_path):
        chat = chat_func()
        chat.set_turns([Turn("user", "Hello")])

        output_file = tmp_path / "test_eval.jsonl"
        with pytest.raises(ValueError, match="must be an assistant turn"):
            chat.export_eval(output_file)

    def test_export_eval_without_system_prompt(self, tmp_path):
        chat = chat_func()
        chat.set_turns(
            [
                Turn("user", "What is 2+2?"),
                Turn("assistant", "2 + 2 = 4"),
            ]
        )

        output_file = tmp_path / "test_eval.jsonl"
        chat.export_eval(output_file, include_system_prompt=False)

        dataset = json_dataset(str(output_file))
        samples = list(dataset)

        assert len(samples) == 1
        sample = samples[0]
        assert len(sample.input) == 1
        assert isinstance(sample.input[0], ChatMessageUser)

    def test_export_eval_custom_turns(self, tmp_path):
        chat = chat_func()
        chat.set_turns(
            [
                Turn("user", "First question"),
                Turn("assistant", "First answer"),
            ]
        )

        output_file = tmp_path / "test_eval.jsonl"
        chat.export_eval(
            output_file,
            turns=[
                Turn("user", "Second question"),
                Turn("assistant", "Second answer"),
            ],
        )

        dataset = json_dataset(str(output_file))
        samples = list(dataset)

        assert "Second question" in str(samples[0].input)


class TestInspectIntegration:
    def test_basic_eval(self):
        chat = chat_func(system_prompt=SYSTEM_DEFAULT)

        task = create_task(
            chat,
            dataset=[Sample(input="What is 2+2?", target="4")],
        )

        results = inspect_eval(task)[0].results

        assert results is not None
        accuracy = results.scores[0].metrics["accuracy"].value
        assert accuracy == 1, f"Expected accuracy of 1, but got {accuracy}"

    def test_system_prompt_override(self):
        chat = chat_func(system_prompt="You are Chuck Norris.")

        task = create_task(
            chat,
            dataset=[
                Sample(
                    input="Tell me a short story.",
                    target="The answer can be any story, but should be in the style of Chuck Norris.",
                )
            ],
        )

        results = inspect_eval(task)[0].results

        assert results is not None
        accuracy = results.scores[0].metrics["accuracy"].value
        assert accuracy == 1, f"Expected accuracy of 1, but got {accuracy}"

    def test_existing_turns(self):
        chat = chat_func(system_prompt=SYSTEM_DEFAULT)

        chat.set_turns(
            [
                Turn("user", "My name is Gregg."),
                Turn("assistant", "Hello Gregg! How can I assist you today?"),
            ]
        )

        task = create_task(
            chat,
            dataset=[
                Sample(
                    input="What is my name?",
                    target="The answer should include 'Gregg'",
                )
            ],
        )

        results = inspect_eval(task)[0].results

        assert results is not None
        accuracy = results.scores[0].metrics["accuracy"].value
        assert accuracy == 1, f"Expected accuracy of 1, but got {accuracy}"

    def test_tool_calling(self):
        chat = chat_func(system_prompt=SYSTEM_DEFAULT)

        def get_current_date():
            """Get the current date in YYYY-MM-DD format."""
            return datetime.datetime.now().strftime("%Y-%m-%d")

        chat.register_tool(get_current_date)

        task = create_task(
            chat,
            dataset=[
                Sample(
                    input="What is today's date?",
                    target="A valid date should be provided and be some time on or after Oct 23rd 2025.",
                )
            ],
        )

        results = inspect_eval(task)[0].results

        assert results is not None
        accuracy = results.scores[0].metrics["accuracy"].value
        assert accuracy == 1, f"Expected accuracy of 1, but got {accuracy}"
        assert accuracy == 1, f"Expected accuracy of 1, but got {accuracy}"
        assert accuracy == 1, f"Expected accuracy of 1, but got {accuracy}"
        assert accuracy == 1, f"Expected accuracy of 1, but got {accuracy}"
        assert accuracy == 1, f"Expected accuracy of 1, but got {accuracy}"
        assert accuracy == 1, f"Expected accuracy of 1, but got {accuracy}"
