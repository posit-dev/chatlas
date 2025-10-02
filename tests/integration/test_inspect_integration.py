import datetime
from chatlas import ChatOpenAI, Turn
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import model_graded_qa

MODEL_NAME = "gpt-5-nano-2025-08-07"


def test_geography_evaluation_with_turns():
    chat = ChatOpenAI(
        model=MODEL_NAME,
        system_prompt="You are a helpful assistant that provides concise answers.",
    )

    chat.set_turns(
        [
            Turn("user", "What is the capital of California?"),
            Turn("assistant", "The capital of California is Sacramento."),
        ]
    )

    task = Task(
        dataset=[
            Sample(
                input="What major river was the city named after",
                target="Sacramento River",
                id="geography_1",
            ),
            Sample(
                input="What is the name of this city's NBA team?",
                target="The Sacramento Kings",
                id="geography_2",
            ),
        ],
        solver=chat.to_solver(),
        scorer=model_graded_qa(
            model=get_model(f"openai/{MODEL_NAME}"),
        ),
        model=get_model(f"openai/{MODEL_NAME}"),
    )

    results = inspect_eval(task)

    accuracy = results[0].results.scores[0].metrics["accuracy"].value
    assert accuracy >= 0.5, f"Expected accuracy of at least 0.5, but got {accuracy}"


def test_simple_evaluation():
    MODEL_NAME = "gpt-5-nano-2025-08-07"

    chat = ChatOpenAI(
        model=MODEL_NAME,
        system_prompt="You are a helpful assistant that provides concise answers.",
    )

    task = Task(
        dataset=[
            Sample(
                input="What is 2+2?",
                target="4",
                id="simple_math_1",
            ),
            Sample(
                input="What is the capital of France?",
                target="Paris",
                id="simple_geo_1",
            ),
        ],
        solver=chat.to_solver(),
        scorer=model_graded_qa(
            model=get_model(f"openai/{MODEL_NAME}"),
        ),
        model=get_model(f"openai/{MODEL_NAME}"),
    )

    results = inspect_eval(task)

    accuracy = results[0].results.scores[0].metrics["accuracy"].value
    assert accuracy == 1, f"Expected accuracy of 1, but got {accuracy}"


def test_evaluation_using_tool_call():
    chat = ChatOpenAI(
        model=MODEL_NAME,
        system_prompt="You are an assistant that can use tools and answer queries.",
    )

    def get_python_date():
        """A simple tool that returns the current time."""

        return datetime.datetime.now().strftime("%Y-%m-%d")

    chat.register_tool(
        get_python_date,
        name="get_date",
    )

    task = Task(
        dataset=[
            Sample(
                input="What is the current date?",
                target=(
                    "The current date should be in the format: YYYY-MM-DD and must be greater than Sept 30, 2025"
                ),
                id="date_query",
            ),
        ],
        solver=chat.to_solver(),
        scorer=model_graded_qa(
            model=get_model(f"openai/{MODEL_NAME}"),
            partial_credit=True,
        ),
        model=get_model(f"openai/{MODEL_NAME}"),
    )

    results = inspect_eval(task)

    accuracy = results[0].results.scores[0].metrics["accuracy"].value
    assert accuracy == 1, f"Expected accuracy of 1, but got {accuracy}"
