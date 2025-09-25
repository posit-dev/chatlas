from chatlas import ChatOpenAI

from inspect_ai import eval as inspect_eval, Task
from inspect_ai.dataset import Sample
from inspect_ai.model import get_model
from inspect_ai.scorer import model_graded_qa

MODEL_NAME = "gpt-5-nano-2025-08-07"

chat = ChatOpenAI(
    model=MODEL_NAME,
    system_prompt="You are a helpful assistant. Only respond in JSON format.",
)

task = Task(
    dataset=[
        Sample(
            input="Provide a JSON response with the tip amount for a $50 meal with a 20% tip.",
            target="tip amount should be 10 in json format",
            id="test_1",
        ),
        Sample(
            input="Provide the answer for difference between Shiny for R and Shiny for Python",
            target=(
                "Shiny for R is the original framework for building interactive web applications using R, "
                "while Shiny for Python is a newer implementation that allows developers to create similar "
                "applications using Python. Both frameworks share similar concepts and functionalities, but "
                "they are designed for different programming languages."
            ),
            id="test_2",
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

# run `inspect view` in terminal to see results
