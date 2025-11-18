import datetime
import sys
from unittest.mock import patch

import pytest

from chatlas import AssistantTurn, Chat, ChatAnthropic, ContentToolRequest, Turn, UserTurn
from chatlas._content import (
    ContentImageInline,
    ContentImageRemote,
    ContentJson,
    ContentPDF,
)
from chatlas._inspect import (
    chatlas_content_as_inspect,
    inspect_content_as_chatlas,
    inspect_messages_as_turns,
    turn_as_inspect_messages,
)

pytest.importorskip("inspect_ai")

import inspect_ai.model as i_model
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample, json_dataset
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
                UserTurn("What is 2+2?"),
                AssistantTurn("2 + 2 = 4"),
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
        assert isinstance(sample.input[0], i_model.ChatMessageSystem)
        assert isinstance(sample.input[1], i_model.ChatMessageUser)
        assert str(sample.target) == "2 + 2 = 4"

    def test_export_eval_custom_target(self, tmp_path):
        chat = chat_func()
        chat.set_turns(
            [
                UserTurn("Tell me a joke"),
                AssistantTurn(
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
                UserTurn("What is 2+2?"),
                AssistantTurn("2 + 2 = 4"),
            ]
        )
        chat.export_eval(output_file)

        chat.set_turns(
            [
                UserTurn("What is 3+3?"),
                AssistantTurn("3 + 3 = 6"),
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
                UserTurn("First question"),
                AssistantTurn("First answer"),
            ]
        )
        chat.export_eval(output_file)

        chat.set_turns(
            [
                UserTurn("Second question"),
                AssistantTurn("Second answer"),
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
                UserTurn("First question"),
                AssistantTurn("First answer"),
            ]
        )
        chat.export_eval(output_file)

        chat.set_turns(
            [
                UserTurn("Second question"),
                AssistantTurn("Second answer"),
            ]
        )
        with pytest.raises(ValueError, match="already exists"):
            chat.export_eval(output_file, overwrite=False)

    def test_export_eval_wrong_extension(self, tmp_path):
        chat = chat_func()
        chat.set_turns(
            [
                UserTurn("What is 2+2?"),
                AssistantTurn("2 + 2 = 4"),
            ]
        )

        output_file = tmp_path / "test_eval.csv"
        with pytest.raises(ValueError, match="must have a `.jsonl` extension"):
            chat.export_eval(output_file)

    def test_export_eval_no_assistant_turn(self, tmp_path):
        chat = chat_func()
        chat.set_turns([UserTurn("Hello")])

        output_file = tmp_path / "test_eval.jsonl"
        with pytest.raises(ValueError, match="must be an assistant turn"):
            chat.export_eval(output_file)

    def test_export_eval_without_system_prompt(self, tmp_path):
        chat = chat_func()
        chat.set_turns(
            [
                UserTurn("What is 2+2?"),
                AssistantTurn("2 + 2 = 4"),
            ]
        )

        output_file = tmp_path / "test_eval.jsonl"
        chat.export_eval(output_file, include_system_prompt=False)

        dataset = json_dataset(str(output_file))
        samples = list(dataset)

        assert len(samples) == 1
        sample = samples[0]
        assert len(sample.input) == 1
        assert isinstance(sample.input[0], i_model.ChatMessageUser)

    def test_export_eval_custom_turns(self, tmp_path):
        chat = chat_func()
        chat.set_turns(
            [
                UserTurn("First question"),
                AssistantTurn("First answer"),
            ]
        )

        output_file = tmp_path / "test_eval.jsonl"
        chat.export_eval(
            output_file,
            turns=[
                UserTurn("Second question"),
                AssistantTurn("Second answer"),
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
                UserTurn("My name is Gregg."),
                AssistantTurn("Hello Gregg! How can I assist you today?"),
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

    def test_multiple_samples_state_management(self):
        """Test that solver has independent state across multiple samples."""

        chat = chat_func(
            system_prompt="""
        The user is building a shopping list.
        Your task is to simply report all known ingredients on every response.
        Be very terse, no punctuation.
        """
        )

        task = create_task(
            chat,
            dataset=[
                Sample(
                    input="Add apples",
                    target="The response should contain only apples, no other fruit",
                ),
                Sample(
                    input="Add bananas",
                    target="The response should contain bananas, no other fruit",
                ),
            ],
        )

        results = inspect_eval(task)[0].results

        assert results is not None
        accuracy = results.scores[0].metrics["accuracy"].value
        assert accuracy == 1, f"Expected accuracy of 1, but got {accuracy}"

    def test_multiple_tools_multiple_samples(self):
        def get_weather():
            """Get current weather data for various cities."""
            return {
                "New York": {
                    "temp": 72,
                    "condition": "Sunny",
                    "humidity": 65,
                },
                "London": {
                    "temp": 58,
                    "condition": "Rainy",
                    "humidity": 85,
                },
                "Tokyo": {
                    "temp": 68,
                    "condition": "Cloudy",
                    "humidity": 70,
                },
                "Sydney": {
                    "temp": 82,
                    "condition": "Sunny",
                    "humidity": 60,
                },
                "Paris": {
                    "temp": 61,
                    "condition": "Partly Cloudy",
                    "humidity": 72,
                },
            }

        def calculate_average(temperatures: list[float]) -> float:
            """Calculate the average temperature."""

            if not temperatures:
                raise ValueError("No temperatures provided")
            return sum(temperatures) / len(temperatures)

        def compare_values(
            value1: float,
            value2: float,
            label1: str,
            label2: str,
        ) -> str:
            """Compare two numeric values."""

            if value1 > value2:
                return f"{label1} ({value1}) is greater than {label2} ({value2})"
            if value2 > value1:
                return f"{label2} ({value2}) is greater than {label1} ({value1})"
            return f"{label1} and {label2} are equal ({value1})"

        chat = chat_func(
            system_prompt=(
                "You are a helpful assistant with access to weather tools. "
                "Use the tools systematically to answer questions accurately. "
                "Make multiple tool calls as needed."
            ),
        )
        chat.register_tool(get_weather)
        chat.register_tool(calculate_average)
        chat.register_tool(compare_values)

        task = create_task(
            chat,
            dataset=[
                Sample(
                    input=(
                        "Using the available tools, get the weather data, "
                        "calculate the average temperature across all cities, "
                        "and compare the average temperature to London's "
                        "temperature. Tell me if the average is higher or "
                        "lower than London's temperature."
                    ),
                    target=(
                        "Average across cities is about 68.2°F. "
                        "London's temperature is 58°F. "
                        "Average (68.2°F) is about 10°F higher than London's "
                        "value (58°F)."
                    ),
                ),
                Sample(
                    input=(
                        "Get the weather information and determine which city "
                        "has the highest humidity. Then compare that city's "
                        "temperature to Tokyo's temperature."
                    ),
                    target=(
                        "London has the highest humidity at 85%. "
                        "London's temperature is 58°F, while Tokyo's temperature "
                        "is 68°F. Tokyo is warmer than London by 10 degrees."
                    ),
                ),
            ],
        )

        # Just want to make sure this runs without error
        results = inspect_eval(task)[0].results
        assert results is not None


class TestContentTranslation:
    """Test translation between chatlas and Inspect content types."""

    def test_round_trip_text_translation(self):
        """Test that text content survives round-trip translation."""

        original_turn = UserTurn("Hello, how are you?")
        inspect_msgs = turn_as_inspect_messages(original_turn, "user")
        recovered_turns = inspect_messages_as_turns(inspect_msgs)

        assert len(recovered_turns) == 1
        assert recovered_turns[0].text == original_turn.text
        assert recovered_turns[0].role == original_turn.role

    def test_round_trip_assistant_translation(self):
        """Test that assistant turns survive round-trip translation."""

        original_turn = AssistantTurn("I'm doing great, thank you!")
        inspect_msgs = turn_as_inspect_messages(original_turn, "assistant")
        recovered_turns = inspect_messages_as_turns(inspect_msgs)  # type: ignore

        assert len(recovered_turns) == 1
        assert recovered_turns[0].text == original_turn.text
        assert recovered_turns[0].role == original_turn.role

    def test_image_remote_translation(self):
        original = ContentImageRemote(
            url="https://example.com/image.png",
            detail="high",
        )
        inspect_content = chatlas_content_as_inspect(original)
        recovered = inspect_content_as_chatlas(inspect_content)

        assert isinstance(recovered, ContentImageRemote)
        assert recovered.url == original.url
        assert recovered.detail == original.detail

    def test_image_inline_translation(self):
        original = ContentImageInline(
            image_content_type="image/png",
            data=(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAA"
                "AAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            ),
        )
        inspect_content = chatlas_content_as_inspect(original)
        recovered = inspect_content_as_chatlas(inspect_content)

        assert isinstance(recovered, ContentImageInline)
        assert recovered.image_content_type == original.image_content_type
        assert recovered.data == original.data

    def test_pdf_translation(self):
        original = ContentPDF(data=b"fake pdf content", filename="document.pdf")
        inspect_content = chatlas_content_as_inspect(original)
        recovered = inspect_content_as_chatlas(inspect_content)

        assert isinstance(recovered, ContentPDF)
        assert recovered.data == original.data
        assert recovered.filename == original.filename

    def test_json_content_translation(self):
        original = ContentJson(value={"key": "value", "number": 42})
        inspect_content = chatlas_content_as_inspect(original)
        recovered = inspect_content_as_chatlas(inspect_content)

        assert isinstance(recovered, ContentJson)
        assert recovered.value == original.value

    def test_tool_request_translation(self):
        tool_request = ContentToolRequest(
            id="tool-123",
            name="get_weather",
            arguments={"city": "New York"},
        )
        original_turn = AssistantTurn([tool_request])

        inspect_msgs = turn_as_inspect_messages(original_turn, "assistant")
        recovered_turns = inspect_messages_as_turns(inspect_msgs)  # type: ignore

        assert len(recovered_turns) == 1
        assert len(recovered_turns[0].contents) == 1
        recovered_tool = recovered_turns[0].contents[0]
        assert isinstance(recovered_tool, ContentToolRequest)
        assert recovered_tool.name == tool_request.name
        assert recovered_tool.arguments == tool_request.arguments


class TestExportEvalWithComplexContent:
    """Test exporting eval datasets with complex content types."""

    def test_export_eval_with_images(self, tmp_path):
        chat = chat_func()
        chat.set_turns(
            [
                UserTurn(
                    [
                        "What's in this image?",
                        ContentImageRemote(url="https://example.com/image.png"),
                    ],
                ),
                AssistantTurn("I see a red car."),
            ]
        )

        output_file = tmp_path / "test_eval.jsonl"
        chat.export_eval(output_file)

        dataset = json_dataset(str(output_file))
        samples = list(dataset)

        assert len(samples) == 1
        assert len(samples[0].input) == 1
        user = samples[0].input[0]
        assert isinstance(user, i_model.ChatMessageUser)
        assert len(user.content) == 2
        assert isinstance(user.content[0], i_model.ContentText)
        assert isinstance(user.content[1], i_model.ContentImage)

    def test_export_eval_with_pdf(self, tmp_path):
        chat = chat_func()
        chat.set_turns(
            [
                UserTurn(
                    [
                        "Summarize this PDF:",
                        ContentPDF(data=b"fake pdf content", filename="document.pdf"),
                    ],
                ),
                AssistantTurn("This is a summary."),
            ]
        )

        output_file = tmp_path / "test_eval.jsonl"
        chat.export_eval(output_file)

        dataset = json_dataset(str(output_file))
        samples = list(dataset)

        assert len(samples) == 1
        user = samples[0].input[0]
        assert isinstance(user, i_model.ChatMessageUser)
        user = samples[0].input[0]
        assert isinstance(user, i_model.ChatMessageUser)
        assert len(user.content) == 2
        assert isinstance(user.content[0], i_model.ContentText)
        assert isinstance(user.content[1], i_model.ContentDocument)

    def test_export_eval_with_json_content(self, tmp_path):
        chat = chat_func()
        chat.set_turns(
            [
                UserTurn("What is the capital of France?"),
                AssistantTurn([ContentJson(value={"capital": "Paris"})]),
                UserTurn("Thank you!"),
                AssistantTurn("You're welcome!"),
            ]
        )

        output_file = tmp_path / "test_eval.jsonl"
        chat.export_eval(output_file)

        dataset = json_dataset(str(output_file))
        samples = list(dataset)

        assert len(samples) == 1
        sample = samples[0]
        assert len(sample.input) == 3
        assistant_msg = sample.input[1]
        assert isinstance(assistant_msg, i_model.ChatMessageAssistant)
        assert len(assistant_msg.content) == 1
        data = assistant_msg.content[0]
        assert isinstance(data, i_model.ContentData)
        assert data.data == {"capital": "Paris"}


class TestExportEvalEdgeCases:
    def test_export_eval_empty_turns(self, tmp_path):
        chat = chat_func()

        output_file = tmp_path / "test_eval.jsonl"
        with pytest.raises(ValueError, match="At least one user turn is required"):
            chat.export_eval(output_file)

    def test_export_eval_no_user_turn(self, tmp_path):
        chat = chat_func()
        output_file = tmp_path / "test_eval.jsonl"

        with pytest.raises(ValueError, match="At least one user turn is required"):
            chat.export_eval(
                output_file,
                turns=[AssistantTurn("Hello!")],
            )
